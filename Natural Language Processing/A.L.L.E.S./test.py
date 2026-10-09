import os
import tempfile
import pickle
import random
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from module import (
    Vocabulary,
    GradientReversal,
    StyleDiscriminator,
    MemoryBank,
    Encoder,
    Attention,
    Decoder,
    Seq2Seq,
)
from main import (
    DiskIndexedALLESDataset,
    collate_fn,
    scan_metadata_limits,
    PersistentMemoryBank,
    CurriculumScheduler,
    EWC,
    calculate_bleu,
)

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_vocab():
    vocab = Vocabulary("test", min_freq=1)
    vocab.word2index = {
        "<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3,
        "hello": 4, "world": 5, "foo": 6, "bar": 7, "baz": 8
    }
    vocab.index2word = {v: k for k, v in vocab.word2index.items()}
    vocab.n_words = len(vocab.word2index)
    vocab.pad_idx = 0
    vocab.sos_idx = 1
    vocab.eos_idx = 2
    vocab.unk_idx = 3
    return vocab


@pytest.fixture
def temp_data_dir():
    tmp = tempfile.mkdtemp()
    src_path = os.path.join(tmp, "src.txt")
    trg_path = os.path.join(tmp, "trg.txt")
    lang_path = os.path.join(tmp, "lang.txt")
    style_path = os.path.join(tmp, "style.txt")

    lines = [
        ("hello world", "hola mundo", "0", "0"),
        ("foo bar", "foo bar", "1", "1"),
        ("baz", "baz", "0", "2"),
        ("hello foo bar", "hello foo bar", "1", "0"),
        ("world baz", "world baz", "0", "1"),
    ]

    with open(src_path, "w", encoding="utf-8") as fs, \
         open(trg_path, "w", encoding="utf-8") as ft, \
         open(lang_path, "w") as fl, \
         open(style_path, "w") as fst:
        for s, t, l, st in lines:
            fs.write(s + "\n")
            ft.write(t + "\n")
            fl.write(l + "\n")
            fst.write(st + "\n")

    yield {
        "src": src_path,
        "trg": trg_path,
        "lang": lang_path,
        "style": style_path,
        "dir": tmp,
    }
    shutil.rmtree(tmp)

class TestVocabulary:
    def test_init_defaults(self):
        v = Vocabulary("test")
        assert v.n_words == 4
        assert v.pad_idx == 0
        assert v.sos_idx == 1
        assert v.eos_idx == 2
        assert v.unk_idx == 3
        assert "<pad>" in v.word2index
        assert v.word2index["<unk>"] == 3

    def test_build_vocab_from_file(self, temp_data_dir):
        v = Vocabulary("test", min_freq=1)
        v.build_vocab_from_file(temp_data_dir["src"])
        assert "hello" in v.word2index
        assert "world" in v.word2index
        assert v.n_words > 4
        v2 = Vocabulary("test", min_freq=100)
        v2.build_vocab_from_file(temp_data_dir["src"])
        assert v2.n_words == 4  # only special tokens

    def test_save_load_roundtrip(self, small_vocab, tmp_path):
        path = tmp_path / "vocab.pkl"
        small_vocab.save(str(path))
        loaded = Vocabulary.load(str(path))
        assert loaded.n_words == small_vocab.n_words
        assert loaded.word2index == small_vocab.word2index
        assert loaded.index2word == small_vocab.index2word

    def test_len(self, small_vocab):
        assert len(small_vocab) == small_vocab.n_words

class TestGradientReversal:
    def test_forward_identity(self, device):
        x = torch.randn(4, 8, requires_grad=True, device=device)
        y = GradientReversal.apply(x, 1.0)
        assert torch.allclose(x, y)
        assert y.requires_grad

    def test_backward_reverses_gradient(self, device):
        x = torch.randn(3, 5, requires_grad=True, device=device)
        alpha = 0.7
        y = GradientReversal.apply(x, alpha)
        loss = y.sum()
        loss.backward()
        expected = -alpha * torch.ones_like(x)
        assert torch.allclose(x.grad, expected, atol=1e-6)

    def test_alpha_zero(self, device):
        x = torch.randn(2, 4, requires_grad=True, device=device)
        y = GradientReversal.apply(x, 0.0)
        y.sum().backward()
        assert torch.allclose(x.grad, torch.zeros_like(x))

class TestStyleDiscriminator:
    def test_forward_shape(self, device):
        disc = StyleDiscriminator(enc_hid_dim=32, num_styles=5).to(device)
        h = torch.randn(8, 64, device=device)  # enc_hid_dim * 2
        out = disc(h, alpha=1.0)
        assert out.shape == (8, 5)

    def test_gradient_reversal_effect(self, device):
        disc = StyleDiscriminator(32, 3).to(device)
        h = torch.randn(4, 64, requires_grad=True, device=device)
        out = disc(h, alpha=1.0)
        loss = out.sum()
        loss.backward()
        assert h.grad is not None
        assert h.grad.abs().sum() > 0

    def test_dropout_train_eval(self, device):
        disc = StyleDiscriminator(16, 2, dropout=0.9).to(device)
        h = torch.randn(10, 32, device=device)
        disc.train()
        out_train = disc(h)
        disc.eval()
        out_eval = disc(h)
        out_eval2 = disc(h)
        assert torch.allclose(out_eval, out_eval2)

class TestMemoryBank:
    def test_empty_memory_returns_zeros(self, device):
        mb = MemoryBank(hidden_dim=16)
        state = torch.randn(4, 16, device=device)
        ctx = mb(state, None, None)
        assert torch.allclose(ctx, torch.zeros_like(state))

    def test_zero_history_lens(self, device):
        mb = MemoryBank(16)
        state = torch.randn(2, 16, device=device)
        mem = torch.randn(2, 5, 16, device=device)
        lens = torch.zeros(2, dtype=torch.long, device=device)
        ctx = mb(state, mem, lens)
        assert torch.allclose(ctx, torch.zeros_like(state), atol=1e-6)

    def test_attention_shape_and_mask(self, device):
        mb = MemoryBank(8)
        state = torch.randn(3, 8, device=device)
        mem = torch.randn(3, 4, 8, device=device)
        lens = torch.tensor([2, 0, 4], device=device)
        ctx = mb(state, mem, lens)
        assert ctx.shape == (3, 8)
        assert torch.allclose(ctx[1], torch.zeros(8, device=device), atol=1e-5)

    def test_nan_protection(self, device):
        mb = MemoryBank(4)
        state = torch.randn(1, 4, device=device)
        mem = torch.randn(1, 3, 4, device=device)
        lens = torch.tensor([0], device=device)
        ctx = mb(state, mem, lens)
        assert not torch.isnan(ctx).any()
        
class TestEncoder:
    def test_forward_shapes(self, device, small_vocab):
        enc = Encoder(
            input_dim=len(small_vocab),
            emb_dim=32,
            enc_hid_dim=64,
            dec_hid_dim=48,
            dropout=0.1
        ).to(device)
        src = torch.tensor([[1, 4, 5, 2, 0], [1, 6, 2, 0, 0]], device=device)
        lengths = torch.tensor([4, 3])
        outputs, h, c, raw = enc(src, lengths)
        assert outputs.shape[0] == 2
        assert outputs.shape[2] == 128
        assert h.shape == (2, 48)
        assert c.shape == (2, 48)
        assert raw.shape == (2, 128)

    def test_packing_respects_lengths(self, device, small_vocab):
        enc = Encoder(len(small_vocab), 16, 32, 32, 0.0).to(device)
        src = torch.tensor([[1, 4, 5, 2], [1, 6, 0, 0]], device=device)
        lengths = torch.tensor([4, 2])
        outputs, _, _, _ = enc(src, lengths)
        assert outputs.shape[1] == 4

class TestAttention:
    def test_forward_and_mask(self, device):
        attn = Attention(enc_hid_dim=16, dec_hid_dim=32).to(device)
        hidden = torch.randn(4, 32, device=device)
        enc_out = torch.randn(4, 7, 32, device=device)
        mask = torch.tensor([
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0],
        ], device=device)
        weights = attn(hidden, enc_out, mask)
        assert weights.shape == (4, 7)
        assert torch.allclose(weights[0, 4:], torch.zeros(3, device=device), atol=1e-6)
        assert torch.allclose(weights[1, 2:], torch.zeros(5, device=device), atol=1e-6)
        assert torch.allclose(weights.sum(dim=1), torch.ones(4, device=device), atol=1e-5)

class TestDecoder:
    def test_single_step_shapes(self, device, small_vocab):
        dec = Decoder(
            output_dim=len(small_vocab),
            emb_dim=32,
            enc_hid_dim=16,
            dec_hid_dim=48,
            num_langs=3,
            num_styles=4,
            dropout=0.1
        ).to(device)
        batch = 5
        input_tok = torch.ones(batch, dtype=torch.long, device=device)
        hidden = torch.randn(batch, 48, device=device)
        cell = torch.randn(batch, 48, device=device)
        enc_out = torch.randn(batch, 6, 32, device=device)
        mask = torch.ones(batch, 6, device=device)
        lang = torch.randint(0, 3, (batch,), device=device)
        style = torch.randint(0, 4, (batch,), device=device)
        mem_ctx = torch.randn(batch, 48, device=device)

        pred, new_h, new_c = dec(
            input_tok, hidden, cell, enc_out, mask, lang, style, mem_ctx
        )
        assert pred.shape == (batch, len(small_vocab))
        assert new_h.shape == (batch, 48)
        assert new_c.shape == (batch, 48)

class TestSeq2Seq:
    @pytest.fixture
    def model(self, device, small_vocab):
        enc = Encoder(len(small_vocab), 24, 32, 40, 0.1)
        dec = Decoder(len(small_vocab), 24, 32, 40, num_langs=2, num_styles=3, dropout=0.1)
        m = Seq2Seq(enc, dec, src_pad_idx=0, sos_idx=1, eos_idx=2).to(device)
        return m

    def test_forward_teacher_forcing(self, model, device, small_vocab):
        batch, src_len, trg_len = 3, 5, 6
        src = torch.tensor([
            [1, 4, 5, 6, 2],
            [1, 4, 5, 2, 0],
            [1, 6, 2, 0, 0],
        ], device=device)
        lengths = torch.tensor([5, 4, 3], device=device)
        trg = torch.randint(1, len(small_vocab), (batch, trg_len), device=device)
        lang = torch.randint(0, 2, (batch,), device=device)
        style = torch.randint(0, 3, (batch,), device=device)

        outputs, raw_h, final_h = model(
            src, lengths, trg, lang, style,
            memory_tensor=None, history_lens=None,
            teacher_forcing_ratio=1.0
        )
        assert outputs.shape == (batch, trg_len, len(small_vocab))
        assert raw_h.shape[0] == batch
        assert final_h.shape == (batch, 40)

    def test_greedy_decode(self, model, device, small_vocab):
        batch = 2
        src = torch.tensor([[1, 4, 5, 2], [1, 6, 2, 0]], device=device)
        lengths = torch.tensor([4, 3])
        lang = torch.zeros(batch, dtype=torch.long, device=device)
        style = torch.ones(batch, dtype=torch.long, device=device)

        decoded, hidden = model.greedy_decode(
            src, lengths, lang, style, max_len=8
        )
        assert decoded.shape == (batch, 8)
        assert hidden.shape == (batch, 40)

    def test_create_mask(self, model, device):
        src = torch.tensor([[1, 2, 3, 0], [1, 2, 0, 0]], device=device)
        mask = model.create_mask(src)
        expected = torch.tensor([[True, True, True, False],
                                 [True, True, False, False]], device=device)
        assert torch.equal(mask, expected)

class TestDiskIndexedALLESDataset:
    def test_len_and_getitem(self, temp_data_dir, small_vocab):
        ds = DiskIndexedALLESDataset(
            temp_data_dir["src"],
            temp_data_dir["trg"],
            temp_data_dir["lang"],
            temp_data_dir["style"],
            small_vocab,
            max_len=20,
            is_pretrain=False
        )
        assert len(ds) == 5
        src, trg, lang, style, idx = ds[0]
        assert isinstance(src, list)
        assert src[0] == small_vocab.sos_idx
        assert src[-1] == small_vocab.eos_idx
        assert lang in (0, 1)
        assert style in (0, 1, 2)
        assert idx == 0
        ds.close()

    def test_pretrain_denoising(self, temp_data_dir, small_vocab):
        random.seed(42)
        ds = DiskIndexedALLESDataset(
            temp_data_dir["src"],
            temp_data_dir["trg"],
            temp_data_dir["lang"],
            temp_data_dir["style"],
            small_vocab,
            is_pretrain=True
        )
        src, trg, _, _, _ = ds[0]
        assert len(src) >= 2
        assert src[0] == small_vocab.sos_idx
        assert src[-1] == small_vocab.eos_idx
        ds.close()

    def test_offsets_consistency(self, temp_data_dir, small_vocab):
        ds = DiskIndexedALLESDataset(
            temp_data_dir["src"],
            temp_data_dir["trg"],
            temp_data_dir["lang"],
            temp_data_dir["style"],
            small_vocab
        )
        assert len(ds.offsets["src"]) == len(ds.offsets["trg"])
        assert len(ds.offsets["src"]) == len(ds.offsets["lang"])
        ds.close()

    def test_unknown_token_handling(self, temp_data_dir, small_vocab):
        with open(temp_data_dir["src"], "a") as f:
            f.write("unknown_word_xyz\n")
        with open(temp_data_dir["trg"], "a") as f:
            f.write("unknown_word_xyz\n")
        with open(temp_data_dir["lang"], "a") as f:
            f.write("0\n")
        with open(temp_data_dir["style"], "a") as f:
            f.write("0\n")

        ds = DiskIndexedALLESDataset(
            temp_data_dir["src"],
            temp_data_dir["trg"],
            temp_data_dir["lang"],
            temp_data_dir["style"],
            small_vocab
        )
        src, _, _, _, _ = ds[5]
        assert small_vocab.unk_idx in src
        ds.close()

class TestCollateFn:
    def test_padding_and_sorting(self, small_vocab):
        batch = [
            ([1, 4, 5, 2], [1, 4, 2], 0, 1, 0),
            ([1, 6, 2], [1, 6, 7, 2], 1, 0, 1),
            ([1, 4, 5, 6, 7, 2], [1, 2], 0, 2, 2),
        ]
        src, lengths, trg, lang, style, indices = collate_fn(batch)
        assert lengths.tolist() == [6, 4, 3]
        assert src.shape[0] == 3
        assert src.shape[1] == 6
        assert trg.shape[1] == 4
        assert (src[0] == 0).sum() == 0
        assert lang.shape == (3,)
        assert indices.tolist() == [2, 0, 1]

class TestScanMetadataLimits:
    def test_correct_counts(self, temp_data_dir):
        n_lang, n_style = scan_metadata_limits(
            temp_data_dir["lang"], temp_data_dir["style"]
        )
        assert n_lang == 2   # 0 and 1 → +1
        assert n_style == 3  # 0,1,2 → +1

class TestPersistentMemoryBank:
    def test_init_and_get(self, device):
        bank = PersistentMemoryBank(size=10, hidden_size=16, max_history=3)
        assert bank.bank.shape == (10, 3, 16)
        assert bank.history_lens.shape == (10,)
        idx = torch.tensor([0, 5, 9])
        ctx, lens = bank.get_context(idx, device)
        assert ctx.shape == (3, 3, 16)
        assert lens.device == device

    def test_update_fills_and_rolls(self, device):
        bank = PersistentMemoryBank(size=4, hidden_size=8, max_history=2)
        idx = torch.tensor([1, 2])
        new1 = torch.randn(2, 8)
        bank.update_context(idx, new1)
        assert bank.history_lens[1].item() == 1
        assert bank.history_lens[2].item() == 1

        new2 = torch.randn(2, 8)
        bank.update_context(idx, new2)
        assert bank.history_lens[1].item() == 2

        new3 = torch.randn(2, 8)
        bank.update_context(idx, new3)
        assert bank.history_lens[1].item() == 2
        assert torch.allclose(bank.bank[1, -1], new3[0].cpu())

class TestCurriculumScheduler:
    def test_progression(self, temp_data_dir, small_vocab):
        ds = DiskIndexedALLESDataset(
            temp_data_dir["src"],
            temp_data_dir["trg"],
            temp_data_dir["lang"],
            temp_data_dir["style"],
            small_vocab,
            max_len=10
        )
        sched = CurriculumScheduler(initial_max_len=15, progression_step=5)
        assert sched.current_max_len == 15

        sched.update_stage(0, ds)
        assert ds.max_len == 15
        assert ds.noise_ratio == pytest.approx(0.25)

        sched.update_stage(2, ds)
        assert ds.max_len == min(120, 15 + 2 * 5)
        assert ds.noise_ratio == pytest.approx(max(0.05, 0.25 - 2 * 0.02))
        ds.close()

class TestCalculateBleu:
    def test_perfect_match(self):
        pred = [torch.tensor([1, 4, 5, 2])]
        trg = [torch.tensor([1, 4, 5, 2])]
        score = calculate_bleu(pred, trg, pad_idx=0)
        assert score == pytest.approx(1.0)

    def test_partial_match(self):
        pred = [torch.tensor([1, 4, 5, 2])]
        trg = [torch.tensor([1, 4, 6, 2])]
        score = calculate_bleu(pred, trg)
        assert 0.0 < score < 1.0

    def test_empty_or_all_pad(self):
        pred = [torch.tensor([0, 0, 0])]
        trg = [torch.tensor([1, 2, 3])]
        score = calculate_bleu(pred, trg)
        assert score == 0.0

class TestEWC:
    def test_fisher_computation_smoke(self, device, small_vocab, temp_data_dir):
        enc = Encoder(len(small_vocab), 16, 24, 24, 0.0)
        dec = Decoder(len(small_vocab), 16, 24, 24, 2, 3, 0.0)
        model = Seq2Seq(enc, dec, 0, 1, 2).to(device)

        ds = DiskIndexedALLESDataset(
            temp_data_dir["src"],
            temp_data_dir["trg"],
            temp_data_dir["lang"],
            temp_data_dir["style"],
            small_vocab,
            max_len=10
        )
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        ewc = EWC(model, loader, criterion)
        assert len(ewc.fisher) > 0
        pen = ewc.penalty(model)
        assert pen.item() == pytest.approx(0.0, abs=1e-5)
        ds.close()

def test_one_training_step_smoke(device, small_vocab, temp_data_dir):
    ds = DiskIndexedALLESDataset(
        temp_data_dir["src"],
        temp_data_dir["trg"],
        temp_data_dir["lang"],
        temp_data_dir["style"],
        small_vocab,
        max_len=12
    )
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn, shuffle=False)

    enc = Encoder(len(small_vocab), 20, 32, 32, 0.1)
    dec = Decoder(len(small_vocab), 20, 32, 32, 2, 3, 0.1)
    model = Seq2Seq(enc, dec, 0, 1, 2).to(device)
    disc = StyleDiscriminator(32, 3).to(device)
    mem_bank = PersistentMemoryBank(len(ds), 32, max_history=3)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    src, lengths, trg, lang, style, gidx = next(iter(loader))
    src, lengths, trg, lang, style = (
        src.to(device), lengths.to(device), trg.to(device),
        lang.to(device), style.to(device)
    )
    mem, hlens = mem_bank.get_context(gidx, device)

    opt.zero_grad()
    outputs, raw_h, new_h = model(src, lengths, trg, lang, style, mem, hlens)
    loss = criterion(
        outputs[:, 1:].reshape(-1, outputs.size(-1)),
        trg[:, 1:].reshape(-1)
    )
    style_pred = disc(raw_h, alpha=0.5)
    adv = nn.CrossEntropyLoss()(style_pred, style)
    total = loss + 0.1 * adv
    total.backward()
    opt.step()

    mem_bank.update_context(gidx, new_h)
    assert total.item() > 0
    ds.close()

def test_vocab_unknown_token(small_vocab):
    idx = small_vocab.word2index.get("nonexistent", small_vocab.unk_idx)
    assert idx == small_vocab.unk_idx


def test_dataset_close_idempotent(temp_data_dir, small_vocab):
    ds = DiskIndexedALLESDataset(
        temp_data_dir["src"],
        temp_data_dir["trg"],
        temp_data_dir["lang"],
        temp_data_dir["style"],
        small_vocab
    )
    _ = ds[0]
    ds.close()
    ds.close()


def test_memory_bank_device_move(device):
    bank = PersistentMemoryBank(5, 10)
    idx = torch.tensor([0, 1, 2])
    ctx, lens = bank.get_context(idx, device)
    assert ctx.device == device
    assert lens.device == device