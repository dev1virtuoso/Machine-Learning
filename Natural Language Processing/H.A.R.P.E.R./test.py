import math
import random
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rhetoric_analyzer import RhetoricBiLSTM, RhetoricAnalyzer
from seq2seq import EncoderRNN, LuongAttention, ConditionalDecoderRNN, HarperSeq2Seq
from hierarchical_seq2seq import (
    UtteranceEncoder,
    ContextRNN,
    HierarchicalLuongAttention,
    HierarchicalStyleDecoder,
    HarperHRED,
)
from evaluator import AdversarialDiscriminator, EvaluationMetrics
from rlhf_policy_gradient import PolicyGradientRLHF
from prosody_tacotron import (
    PreNet,
    Highway,
    CBHG,
    LocationSensitiveAttention,
    PostNet,
    GriffinLimVocoder,
    ProsodyAcousticDecoder,
)
from main import StyleClassifier, ProductionDataFactory

import main
main.F = F

def _patch_attention_mask_compat():
    """Make old .byte() masks work with modern torch.masked_fill_"""
    original_luong = LuongAttention.forward
    original_hier = HierarchicalLuongAttention.forward

    def safe_luong(self, hidden, encoder_outputs, mask=None):
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.bool()
        return original_luong(self, hidden, encoder_outputs, mask)

    def safe_hier(self, decoder_hidden, context_outputs, mask=None):
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.bool()
        return original_hier(self, decoder_hidden, context_outputs, mask)

    LuongAttention.forward = safe_luong
    HierarchicalLuongAttention.forward = safe_hier

_patch_attention_mask_compat()

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_vocab_size():
    return 50


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def seq_len():
    return 12


@pytest.fixture
def hidden_size():
    return 32


@pytest.fixture
def embedding_dim():
    return 24


@pytest.fixture
def tone_dim():
    return 8


@pytest.fixture
def num_tones():
    return 3


@pytest.fixture
def rhetoric_dim():
    return 16

def make_sorted_lengths(batch_size, seq_len, min_len=3):
    lengths = np.random.randint(min_len, seq_len + 1, size=batch_size)
    lengths = np.sort(lengths)[::-1].copy()
    return torch.LongTensor(lengths)

class TestRhetoricBiLSTM:
    def test_forward_shape(self, device, small_vocab_size):
        model = RhetoricBiLSTM(small_vocab_size, 16, 20, num_classes=5).to(device)
        x = torch.randint(0, small_vocab_size, (3, 8), device=device)
        lengths = torch.tensor([8, 6, 4], device=device)
        logits, hidden = model(x, lengths)
        assert logits.shape == (3, 5)
        assert hidden.shape == (3, 40)

    def test_empty_sequence_handling(self, device, small_vocab_size):
        model = RhetoricBiLSTM(small_vocab_size, 8, 12, 3).to(device)
        x = torch.zeros(2, 1, dtype=torch.long, device=device)
        lengths = torch.tensor([1, 1], device=device)
        logits, _ = model(x, lengths)
        assert logits.shape == (2, 3)


class TestRhetoricAnalyzer:
    @pytest.fixture
    def analyzer(self):
        with patch("rhetoric_analyzer.spacy.load") as mock_load:
            mock_nlp = MagicMock()
            mock_load.return_value = mock_nlp
            analyzer = RhetoricAnalyzer(spacy_model="en_core_web_sm")
            analyzer.nlp = mock_nlp
            yield analyzer

    def test_init_without_ml_model(self, analyzer):
        assert analyzer.ml_model is None
        assert analyzer.rhetoric_dim == 256

    def test_detect_anaphora_positive(self, analyzer):
        def make_sent(words):
            tokens = []
            for w in words:
                t = MagicMock()
                t.is_punct = False
                t.is_alpha = True
                t.lower_ = w
                tokens.append(t)
            sent = MagicMock()
            sent.__iter__ = lambda self: iter(tokens)
            sent.__len__ = lambda self: len(tokens)
            return sent

        sent1 = make_sent(["the", "cat", "sat"])
        sent2 = make_sent(["the", "dog", "ran"])
        doc = MagicMock()
        doc.sents = [sent1, sent2]
        assert analyzer._detect_anaphora(doc) is True

    def test_detect_anaphora_negative(self, analyzer):
        sent1 = MagicMock()
        sent1.__iter__ = lambda self: iter([
            MagicMock(is_punct=False, is_alpha=True, lower_="hello"),
        ])
        doc = MagicMock()
        doc.sents = [sent1]
        assert analyzer._detect_anaphora(doc) is False

    def test_detect_chiasmus_positive(self, analyzer):
        tokens = [
            MagicMock(pos_="NOUN", dep_="nsubj", lemma_="man", is_punct=False),
            MagicMock(pos_="VERB", dep_="ROOT", lemma_="see", is_punct=False),
            MagicMock(pos_="VERB", dep_="ROOT", lemma_="see", is_punct=False),
            MagicMock(pos_="NOUN", dep_="dobj", lemma_="man", is_punct=False),
        ]
        doc = MagicMock()
        doc.__iter__ = lambda self: iter(tokens)
        assert analyzer._detect_chiasmus(doc) is True

    def test_detect_rhetorical_question_positive(self, analyzer):
        tokens = [
            MagicMock(lower_="why", pos_="ADV", dep_="advmod", text="Why"),
            MagicMock(lower_="not", pos_="PART", dep_="neg", text="not"),
            MagicMock(lower_="try", pos_="VERB", dep_="ROOT", text="try"),
            MagicMock(lower_="?", pos_="PUNCT", dep_="punct", text="?"),
        ]
        doc = MagicMock()
        doc.__len__ = lambda self: 4
        doc.__getitem__ = lambda self, i: tokens[i]
        doc.__iter__ = lambda self: iter(tokens)
        assert analyzer._detect_rhetorical_question(doc) is True

    def test_detect_metaphor_positive(self, analyzer):
        like_token = MagicMock(lower_="like", dep_="prep", pos_="ADP")
        like_token.head = MagicMock(pos_="VERB")
        like_token.children = [MagicMock()]
        doc = MagicMock()
        doc.__iter__ = lambda self: iter([like_token])
        assert analyzer._detect_metaphor(doc) is True

    def test_analyze_text_empty(self, analyzer):
        empty_doc = MagicMock()
        empty_doc.__len__ = lambda self: 0
        empty_doc.sents = []
        analyzer.nlp.return_value = empty_doc
        features, emb = analyzer.analyze_text("")
        assert features == {"anaphora": 0, "chiasmus": 0, "rhetorical_question": 0, "metaphor": 0}
        assert emb.shape == (1, 256)

    def test_analyze_text_with_features(self, analyzer):
        with patch.object(analyzer, "_detect_anaphora", return_value=True), \
             patch.object(analyzer, "_detect_chiasmus", return_value=True), \
             patch.object(analyzer, "_detect_rhetorical_question", return_value=True), \
             patch.object(analyzer, "_detect_metaphor", return_value=True):
            doc = MagicMock()
            doc.__len__ = lambda self: 5
            analyzer.nlp.return_value = doc
            features, emb = analyzer.analyze_text("dummy text")
            assert features["anaphora"] == 1
            assert features["chiasmus"] == 1
            assert features["rhetorical_question"] == 1
            assert features["metaphor"] == 1
            assert emb.shape[1] == 256

class TestEncoderRNN:
    def test_forward_shapes(self, device, small_vocab_size, hidden_size, embedding_dim, batch_size, seq_len):
        enc = EncoderRNN(small_vocab_size, hidden_size, embedding_dim, n_layers=2).to(device)
        input_seqs = torch.randint(1, small_vocab_size, (batch_size, seq_len), device=device)
        lengths = make_sorted_lengths(batch_size, seq_len).to(device)
        for i, l in enumerate(lengths):
            if l < seq_len:
                input_seqs[i, l:] = 0

        outputs, hidden = enc(input_seqs, lengths)
        max_len_in_batch = int(lengths.max().item())
        assert outputs.shape == (batch_size, max_len_in_batch, hidden_size)
        assert hidden.shape[0] == 4
        assert hidden.shape[1] == batch_size
        assert hidden.shape[2] == hidden_size


class TestLuongAttention:
    def test_forward_and_mask(self, device, hidden_size, batch_size, seq_len):
        attn = LuongAttention(hidden_size).to(device)
        hidden = torch.randn(1, batch_size, hidden_size, device=device)
        encoder_outputs = torch.randn(batch_size, seq_len, hidden_size, device=device)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        mask[:, -3:] = True

        weights = attn(hidden, encoder_outputs, mask)
        assert weights.shape == (batch_size, 1, seq_len)
        assert torch.all(weights.squeeze(1)[:, -3:] < 1e-5)


class TestConditionalDecoderRNN:
    def test_single_step(self, device, small_vocab_size, hidden_size, embedding_dim,
                         tone_dim, num_tones, rhetoric_dim, batch_size, seq_len):
        dec = ConditionalDecoderRNN(
            small_vocab_size, hidden_size, embedding_dim,
            tone_dim, num_tones, rhetoric_dim, n_layers=2
        ).to(device)
        input_step = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        last_hidden = torch.randn(2, batch_size, hidden_size, device=device)
        encoder_outputs = torch.randn(batch_size, seq_len, hidden_size, device=device)
        target_tone = torch.randint(0, num_tones, (batch_size,), device=device)
        target_rhetoric = torch.randn(batch_size, rhetoric_dim, device=device)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        output, hidden, attn = dec(
            input_step, last_hidden, encoder_outputs,
            target_tone, target_rhetoric, mask
        )
        assert output.shape == (batch_size, small_vocab_size)
        assert hidden.shape == (2, batch_size, hidden_size)
        assert attn.shape == (batch_size, 1, seq_len)


class TestHarperSeq2Seq:
    def test_forward_teacher_forcing(self, device, small_vocab_size, hidden_size,
                                 embedding_dim, tone_dim, num_tones, rhetoric_dim,
                                 batch_size, seq_len):
        enc = EncoderRNN(small_vocab_size, hidden_size, embedding_dim, n_layers=2)
        dec = ConditionalDecoderRNN(
            small_vocab_size, hidden_size, embedding_dim,
            tone_dim, num_tones, rhetoric_dim, n_layers=2
        )
        model = HarperSeq2Seq(enc, dec).to(device)

        input_seqs = torch.randint(1, small_vocab_size, (batch_size, seq_len), device=device)
        lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        target_seqs = torch.randint(1, small_vocab_size, (batch_size, seq_len), device=device)
        target_tones = torch.randint(0, num_tones, (batch_size,), device=device)
        target_rhetoric = torch.randn(batch_size, rhetoric_dim, device=device)

        outputs = model(
            Variable(input_seqs), Variable(lengths),
            Variable(target_seqs), Variable(target_tones),
            Variable(target_rhetoric), global_step=0, max_steps=10000
        )
        assert outputs.shape == (batch_size, seq_len, small_vocab_size)

class TestUtteranceEncoder:
    def test_forward(self, device, small_vocab_size, embedding_dim, hidden_size, batch_size, seq_len):
        enc = UtteranceEncoder(small_vocab_size, embedding_dim, hidden_size, n_layers=1).to(device)
        seqs = torch.randint(1, small_vocab_size, (batch_size, seq_len), device=device)
        lengths = make_sorted_lengths(batch_size, seq_len).to(device)
        for i, l in enumerate(lengths):
            if l < seq_len:
                seqs[i, l:] = 0
        out = enc(seqs, lengths)
        assert out.shape == (batch_size, hidden_size * 2)


class TestContextRNN:
    def test_forward(self, device, hidden_size, batch_size):
        ctx = ContextRNN(hidden_size * 2, hidden_size, n_layers=1).to(device)
        utt_vecs = torch.randn(batch_size, 3, hidden_size * 2, device=device)
        outputs, hidden = ctx(utt_vecs)
        assert outputs.shape == (batch_size, 3, hidden_size)
        assert hidden.shape == (1, batch_size, hidden_size)


class TestHierarchicalLuongAttention:
    def test_forward(self, device, hidden_size, batch_size):
        attn = HierarchicalLuongAttention(hidden_size).to(device)
        decoder_hidden = torch.randn(1, batch_size, hidden_size, device=device)
        context_outputs = torch.randn(batch_size, 4, hidden_size, device=device)
        weights = attn(decoder_hidden, context_outputs)
        assert weights.shape == (batch_size, 1, 4)
        assert torch.allclose(weights.sum(dim=2), torch.ones(batch_size, 1, device=device), atol=1e-5)


class TestHierarchicalStyleDecoder:
    def test_single_step(self, device, small_vocab_size, embedding_dim, hidden_size,
                         tone_dim, num_tones, rhetoric_dim, batch_size):
        dec = HierarchicalStyleDecoder(
            small_vocab_size, embedding_dim, hidden_size,
            tone_dim, num_tones, rhetoric_dim, n_layers=1
        ).to(device)
        input_step = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        last_hidden = torch.randn(1, batch_size, hidden_size, device=device)
        context_outputs = torch.randn(batch_size, 3, hidden_size, device=device)
        target_tone = torch.randint(0, num_tones, (batch_size,), device=device)
        target_rhetoric = torch.randn(batch_size, rhetoric_dim, device=device)

        output, hidden, attn = dec(
            input_step, last_hidden, context_outputs,
            target_tone, target_rhetoric
        )
        assert output.shape == (batch_size, small_vocab_size)
        assert hidden.shape == (1, batch_size, hidden_size)


class TestHarperHRED:
    def test_forward(self, device, small_vocab_size, embedding_dim, hidden_size,
                     tone_dim, num_tones, rhetoric_dim, batch_size, seq_len):
        model = HarperHRED(
            small_vocab_size, embedding_dim, hidden_size,
            tone_dim, num_tones, rhetoric_dim, n_layers=1
        ).to(device)

        num_turns = 3
        history_seqs = torch.randint(1, small_vocab_size, (batch_size, num_turns, seq_len), device=device)
        history_lengths = torch.randint(3, seq_len + 1, (batch_size, num_turns), device=device)
        history_lengths[0, -1] = 0

        target_seqs = torch.randint(1, small_vocab_size, (batch_size, seq_len), device=device)
        target_tones = torch.randint(0, num_tones, (batch_size,), device=device)
        target_rhetorics = torch.randn(batch_size, rhetoric_dim, device=device)

        outputs = model(
            Variable(history_seqs), Variable(history_lengths),
            Variable(target_seqs), Variable(target_tones),
            Variable(target_rhetorics), global_step=100, max_steps=10000
        )
        assert outputs.shape == (batch_size, seq_len, small_vocab_size)

class TestAdversarialDiscriminator:
    def test_forward_from_ids(self, device, small_vocab_size, embedding_dim, batch_size, seq_len):
        disc = AdversarialDiscriminator(small_vocab_size, embedding_dim).to(device)
        seqs = torch.randint(0, small_vocab_size, (batch_size, seq_len), device=device)
        scores = disc(seqs, is_embedding=False)
        assert scores.shape == (batch_size, 1)

    def test_forward_from_embeddings(self, device, embedding_dim, batch_size, seq_len):
        disc = AdversarialDiscriminator(50, embedding_dim).to(device)
        emb = torch.randn(batch_size, seq_len, embedding_dim, device=device)
        scores = disc(emb, is_embedding=True)
        assert scores.shape == (batch_size, 1)

    def test_get_embeddings(self, device, small_vocab_size, embedding_dim, batch_size, seq_len):
        disc = AdversarialDiscriminator(small_vocab_size, embedding_dim).to(device)
        seqs = torch.randint(0, small_vocab_size, (batch_size, seq_len), device=device)
        emb = disc.get_embeddings(seqs)
        assert emb.shape == (batch_size, seq_len, embedding_dim)


class TestEvaluationMetrics:
    def test_gradient_penalty(self, device, embedding_dim, batch_size, seq_len):
        disc = AdversarialDiscriminator(50, embedding_dim).to(device)
        real = torch.randn(batch_size, seq_len, embedding_dim, device=device, requires_grad=True)
        fake = torch.randn(batch_size, seq_len, embedding_dim, device=device, requires_grad=True)
        gp = EvaluationMetrics.compute_gradient_penalty(disc, real, fake)
        assert gp.ndimension() == 0
        assert gp.item() >= 0

    def test_calculate_perplexity(self, device, small_vocab_size, batch_size, seq_len):
        log_probs = F.log_softmax(torch.randn(batch_size, seq_len, small_vocab_size, device=device), dim=-1)
        targets = torch.randint(1, small_vocab_size, (batch_size, seq_len), device=device)
        targets[:, -2:] = 0
        ppl = EvaluationMetrics.calculate_perplexity(log_probs, targets, pad_token_id=0)
        assert isinstance(ppl, float)
        assert ppl > 0

    def test_calculate_perplexity_all_pad(self, device, small_vocab_size, batch_size, seq_len):
        log_probs = F.log_softmax(torch.randn(batch_size, seq_len, small_vocab_size, device=device), dim=-1)
        targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        ppl = EvaluationMetrics.calculate_perplexity(log_probs, targets, pad_token_id=0)
        assert ppl == 0.0

    def test_dtw_alignment(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.1, 2.2, 3.3])
        path_x, path_y = EvaluationMetrics._dtw_alignment(x, y)
        assert len(path_x) == len(path_y)
        assert path_x[0] == 0 and path_y[0] == 0
        assert path_x[-1] == len(x) - 1
        assert path_y[-1] == len(y) - 1

    def test_prosody_correlation(self, device, batch_size):
        pred = Variable(torch.randn(batch_size, 20, device=device))
        target = Variable(torch.randn(batch_size, 20, device=device))
        mask = Variable(torch.ones(batch_size, 20, device=device))
        corr = EvaluationMetrics.calculate_prosody_correlation(pred, target, mask)
        assert isinstance(corr, (float, np.floating))
        assert -1.0 <= float(corr) <= 1.0

    def test_prosody_correlation_short_sequences(self, device):
        pred = Variable(torch.randn(2, 1, device=device))
        target = Variable(torch.randn(2, 1, device=device))
        mask = Variable(torch.ones(2, 1, device=device))
        corr = EvaluationMetrics.calculate_prosody_correlation(pred, target, mask)
        assert corr == 0.0

class TestPolicyGradientRLHF:
    @pytest.fixture
    def rlhf_setup(self, device, small_vocab_size, hidden_size, embedding_dim,
                   tone_dim, num_tones, rhetoric_dim, batch_size, seq_len):
        enc = EncoderRNN(small_vocab_size, hidden_size, embedding_dim, n_layers=2)
        dec = ConditionalDecoderRNN(
            small_vocab_size, hidden_size, embedding_dim,
            tone_dim, num_tones, rhetoric_dim, n_layers=2
        )
        generator = HarperSeq2Seq(enc, dec).to(device)
        discriminator = AdversarialDiscriminator(small_vocab_size, embedding_dim).to(device)
        style_clf = StyleClassifier(small_vocab_size, embedding_dim, hidden_size, num_tones).to(device)

        opt = torch.optim.Adam(generator.parameters(), lr=1e-3)
        disc_opt = torch.optim.Adam(discriminator.parameters(), lr=5e-4)

        agent = PolicyGradientRLHF(
            generator_model=generator,
            discriminator=discriminator,
            style_classifier=style_clf,
            optimizer=opt,
            disc_optimizer=disc_opt,
            pad_token=0,
            sos_token=1
        )
        return agent, generator, discriminator, style_clf

    def test_sample_sequence(self, rlhf_setup, device, batch_size, seq_len,
                         num_tones, rhetoric_dim, small_vocab_size):
        agent, _, _, _ = rlhf_setup
        input_seqs = Variable(torch.randint(1, small_vocab_size, (batch_size, seq_len), device=device))
        lengths = Variable(torch.full((batch_size,), seq_len, dtype=torch.long, device=device))
        tones = Variable(torch.randint(0, num_tones, (batch_size,), device=device))
        rhetoric = Variable(torch.randn(batch_size, rhetoric_dim, device=device))

        seqs, log_probs, entropies = agent.sample_sequence(
            input_seqs, lengths, tones, rhetoric, max_len=8
        )
        assert seqs.shape == (batch_size, 8)
        assert log_probs.shape == (batch_size, 8)
        assert entropies.shape == (batch_size, 8)

    def test_compute_reward(self, rlhf_setup, device, batch_size, seq_len,
                            num_tones, small_vocab_size):
        agent, _, _, _ = rlhf_setup
        gen_seqs = Variable(torch.randint(1, small_vocab_size, (batch_size, seq_len), device=device))
        tones = Variable(torch.randint(0, num_tones, (batch_size,), device=device))
        rewards = agent.compute_reward(gen_seqs, tones)
        assert rewards.shape == (batch_size,)
        assert not rewards.requires_grad

    def test_train_step_smoke(self, rlhf_setup, device, batch_size, seq_len,
                          num_tones, rhetoric_dim, small_vocab_size):
        agent, _, _, _ = rlhf_setup
        input_seqs = Variable(torch.randint(1, small_vocab_size, (batch_size, seq_len), device=device))
        lengths = Variable(torch.full((batch_size,), seq_len, dtype=torch.long, device=device))
        tones = Variable(torch.randint(0, num_tones, (batch_size,), device=device))
        rhetoric = Variable(torch.randn(batch_size, rhetoric_dim, device=device))

        pg_loss, entropy, reward_mean = agent.train_step(
            input_seqs, lengths, tones, rhetoric
        )
        assert isinstance(pg_loss, float)
        assert isinstance(entropy, float)
        assert isinstance(reward_mean, float)

    def test_train_discriminator_step(self, rlhf_setup, device, batch_size, seq_len,
                                  num_tones, rhetoric_dim, small_vocab_size):
        agent, _, _, _ = rlhf_setup
        real_seqs = Variable(torch.randint(1, small_vocab_size, (batch_size, seq_len), device=device))
        input_seqs = Variable(torch.randint(1, small_vocab_size, (batch_size, seq_len), device=device))
        lengths = Variable(torch.full((batch_size,), seq_len, dtype=torch.long, device=device))
        tones = Variable(torch.randint(0, num_tones, (batch_size,), device=device))
        rhetoric = Variable(torch.randn(batch_size, rhetoric_dim, device=device))

        original_sample = agent.sample_sequence

        def limited_sample(*args, **kwargs):
            kwargs['max_len'] = seq_len
            return original_sample(*args, **kwargs)

        agent.sample_sequence = limited_sample
        try:
            d_loss = agent.train_discriminator_step(
                real_seqs, input_seqs, lengths, tones, rhetoric
            )
        finally:
            agent.sample_sequence = original_sample

        assert isinstance(d_loss, float)

class TestPreNet:
    def test_forward(self, device):
        prenet = PreNet(80, sizes=[64, 32]).to(device)
        x = torch.randn(4, 10, 80, device=device)
        out = prenet(x)
        assert out.shape == (4, 10, 32)


class TestHighway:
    def test_forward(self, device):
        hw = Highway(64).to(device)
        x = torch.randn(3, 5, 64, device=device)
        out = hw(x)
        assert out.shape == x.shape


class TestCBHG:
    def test_forward(self, device):
        cbhg = CBHG(in_dim=40, K=4, projection_dims=[32, 32], gru_dim=16).to(device)
        x = torch.randn(2, 20, 40, device=device)
        out = cbhg(x)
        assert out.shape == (2, 20, 32)


class TestLocationSensitiveAttention:
    def test_forward(self, device, batch_size, seq_len):
        attn = LocationSensitiveAttention(query_dim=32, memory_dim=64, attention_dim=16).to(device)
        query = torch.randn(batch_size, 32, device=device)
        memory = torch.randn(batch_size, seq_len, 64, device=device)
        cum_weights = torch.zeros(batch_size, seq_len, device=device)

        context, weights, next_cum = attn(query, memory, cum_weights)
        assert context.shape == (batch_size, 64)
        assert weights.shape == (batch_size, seq_len)
        assert next_cum.shape == (batch_size, seq_len)
        assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size, device=device), atol=1e-5)


class TestPostNet:
    def test_forward(self, device):
        postnet = PostNet(mel_dim=40, linear_dim=100).to(device)
        mel = torch.randn(2, 15, 40, device=device)
        linear = postnet(mel)
        assert linear.shape == (2, 15, 100)


class TestGriffinLimVocoder:
    def test_synthesize_shape(self):
        linear_spec = np.random.randn(50, 20).astype(np.float32)
        with patch("prosody_tacotron.librosa.istft") as mock_istft, \
            patch("prosody_tacotron.librosa.stft") as mock_stft:
            mock_istft.return_value = np.random.randn(512)
            mock_stft.return_value = (
                np.random.randn(20, 50) + 1j * np.random.randn(20, 50)
            )
            waveform = GriffinLimVocoder.synthesize(linear_spec, n_iter=2)
            assert isinstance(waveform, np.ndarray)
            assert waveform.ndim == 1


class TestProsodyAcousticDecoder:
    def test_forward_smoke(self, device, hidden_size, tone_dim, num_tones, batch_size, seq_len):
        mel_dim = 40
        decoder = ProsodyAcousticDecoder(
            mel_dim, hidden_size, tone_dim, num_tones, prosody_feature_dim=3
        ).to(device)

        encoder_memory = torch.randn(batch_size, seq_len, hidden_size, device=device)
        target_style = torch.randint(0, num_tones, (batch_size,), device=device)
        pitch_energy_duration = torch.rand(batch_size, 3, device=device)

        mel_seq, linear_seq, stop_outputs = decoder(
            encoder_memory, target_style, pitch_energy_duration,
            max_mel_steps=10, stop_threshold=0.99
        )
        assert mel_seq.shape[0] == batch_size
        assert mel_seq.shape[2] == mel_dim
        assert linear_seq.shape[0] == batch_size
        assert stop_outputs.shape[0] == batch_size

class TestStyleClassifier:
    def test_forward(self, device, small_vocab_size, embedding_dim, hidden_size, num_tones, batch_size, seq_len):
        clf = StyleClassifier(small_vocab_size, embedding_dim, hidden_size, num_tones).to(device)
        x = torch.randint(0, small_vocab_size, (batch_size, seq_len), device=device)
        lengths = make_sorted_lengths(batch_size, seq_len).to(device)
        logits = clf(x, lengths)
        assert logits.shape == (batch_size, num_tones)
        assert torch.allclose(torch.exp(logits).sum(dim=1), torch.ones(batch_size, device=device), atol=1e-5)


class TestProductionDataFactory:
    def test_get_batch_cpu(self, small_vocab_size, num_tones, rhetoric_dim, batch_size, seq_len):
        factory = ProductionDataFactory(
            small_vocab_size, num_tones, rhetoric_dim,
            batch_size=batch_size, seq_len=seq_len
        )
        batch = factory.get_batch(use_cuda=False)
        assert len(batch) == 8
        input_seqs, input_lengths, target_seqs, target_tones, target_rhetoric, \
            history_seqs, history_lengths, pitch = batch

        assert input_seqs.size() == (batch_size, seq_len)
        assert input_lengths.size() == (batch_size,)
        assert target_tones.size() == (batch_size,)
        assert target_rhetoric.size() == (batch_size, rhetoric_dim)
        assert history_seqs.size() == (batch_size, 3, seq_len)
        assert pitch.size() == (batch_size, 3)

        assert torch.all(input_lengths[:-1] >= input_lengths[1:])

def test_seq2seq_to_discriminator_pipeline(device, small_vocab_size, hidden_size,
                                           embedding_dim, tone_dim, num_tones,
                                           rhetoric_dim, batch_size, seq_len):
    enc = EncoderRNN(small_vocab_size, hidden_size, embedding_dim, n_layers=1)
    dec = ConditionalDecoderRNN(
        small_vocab_size, hidden_size, embedding_dim,
        tone_dim, num_tones, rhetoric_dim, n_layers=1
    )
    model = HarperSeq2Seq(enc, dec).to(device)
    disc = AdversarialDiscriminator(small_vocab_size, embedding_dim).to(device)

    input_seqs = Variable(torch.randint(1, small_vocab_size, (batch_size, seq_len), device=device))
    lengths = Variable(torch.full((batch_size,), seq_len, dtype=torch.long, device=device))
    target_seqs = Variable(torch.randint(1, small_vocab_size, (batch_size, seq_len), device=device))
    tones = Variable(torch.randint(0, num_tones, (batch_size,), device=device))
    rhetoric = Variable(torch.randn(batch_size, rhetoric_dim, device=device))

    log_probs = model(input_seqs, lengths, target_seqs, tones, rhetoric)
    ppl = EvaluationMetrics.calculate_perplexity(log_probs, target_seqs)
    assert ppl > 0

    scores = disc(target_seqs, is_embedding=False)
    assert scores.shape == (batch_size, 1)


def test_hred_to_acoustic_pipeline(device, small_vocab_size, embedding_dim, hidden_size,
                                   tone_dim, num_tones, rhetoric_dim, batch_size, seq_len):
    hred = HarperHRED(
        small_vocab_size, embedding_dim, hidden_size,
        tone_dim, num_tones, rhetoric_dim, n_layers=1
    ).to(device)
    acoustic = ProsodyAcousticDecoder(
        mel_dim=40, encoder_hidden_dim=hidden_size,
        style_dim=tone_dim, num_styles=num_tones
    ).to(device)

    history_seqs = Variable(torch.randint(1, small_vocab_size, (batch_size, 2, seq_len), device=device))
    history_lengths = Variable(torch.randint(4, seq_len + 1, (batch_size, 2), device=device))
    target_seqs = Variable(torch.randint(1, small_vocab_size, (batch_size, seq_len), device=device))
    tones = Variable(torch.randint(0, num_tones, (batch_size,), device=device))
    rhetoric = Variable(torch.randn(batch_size, rhetoric_dim, device=device))

    _ = hred(history_seqs, history_lengths, target_seqs, tones, rhetoric)

    enc_memory = torch.randn(batch_size, seq_len, hidden_size, device=device)
    pitch = torch.rand(batch_size, 3, device=device)
    mel, linear, stop = acoustic(enc_memory, tones, pitch, max_mel_steps=5)
    assert mel.shape[0] == batch_size
    assert linear.shape[0] == batch_size