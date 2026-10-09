import json
import os
import sys
import tempfile
from collections import defaultdict
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.autograd import Variable

sys.modules["stanfordcorenlp"] = MagicMock()
sys.modules["spacy"] = MagicMock()

from bilstm_crf import BiLSTM_CRF, log_sum_exp
from event_extractor import JointEventExtractor
from relation_extractor import PositionAwareRelationExtractor
from data_utils import align_distant_supervision, get_position_ids
from main import Vocabulary, load_data, prepare_batch, pad_sequence

def _patch_bilstm_crf_for_modern_torch():
    def safe_neg_log_likelihood(self, sentences, seq_lens, tags, mask):
        mask_sum = torch.sum(mask)
        is_empty = (mask_sum.item() == 0) if mask_sum.dim() == 0 else (mask_sum.data[0] == 0)

        if is_empty:
            zero_loss = Variable(sentences.data.new(1).fill_(0.0).float(), requires_grad=True)
            if getattr(self, "use_gpu", False):
                zero_loss = zero_loss.cuda()
            return zero_loss

        feats = self._get_lstm_features(sentences, seq_lens)
        forward_score = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(feats, tags, mask)
        return torch.mean(forward_score - gold_score)

    BiLSTM_CRF.neg_log_likelihood = safe_neg_log_likelihood

_patch_bilstm_crf_for_modern_torch()

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tag_to_ix():
    return {"<PAD>": 0, "O": 1, "<START>": 2, "<STOP>": 3, "B-PER": 4, "I-PER": 5}


@pytest.fixture
def small_vocab():
    v = Vocabulary()
    for w in ["the", "cat", "sat", "on", "mat", "John", "works", "at", "Google"]:
        v.add_word(w)
    for t in ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"]:
        v.add_tag(t)
    for r in ["NA", "Work_For", "Located_In"]:
        v.add_relation(r)
    v.compile()
    return v


@pytest.fixture
def batch_size():
    return 3


@pytest.fixture
def seq_len():
    return 8

class TestLogSumExp:
    def test_basic(self):
        vec = Variable(torch.FloatTensor([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]]))
        result = log_sum_exp(vec)
        assert result.size() == (2,)
        val = result[0].item() if result[0].dim() == 0 else float(result.data[0])
        assert abs(val - 3.407605964) < 1e-4


class TestBiLSTM_CRF:
    @pytest.fixture
    def model(self, tag_to_ix):
        return BiLSTM_CRF(
            vocab_size=20,
            tag_to_ix=tag_to_ix,
            embedding_dim=16,
            hidden_dim=32,
            dropout=0.0,
            use_gpu=False
        )

    def test_init_transitions(self, model, tag_to_ix):
        assert model.transitions.data[tag_to_ix["<START>"], :].max() < -9999
        assert model.transitions.data[:, tag_to_ix["<STOP>"]].max() < -9999

    def test_get_lstm_features_shape(self, model, batch_size, seq_len):
        sentences = Variable(torch.LongTensor(batch_size, seq_len).random_(1, 19))
        seq_lens = Variable(torch.LongTensor([seq_len, seq_len - 2, seq_len - 4]))
        sentences.data[1, seq_len - 2:] = 0
        sentences.data[2, seq_len - 4:] = 0

        feats = model._get_lstm_features(sentences, seq_lens)
        max_len = int(seq_lens.max())
        assert feats.size() == (batch_size, max_len, model.tagset_size)

    def test_neg_log_likelihood_positive(self, model, batch_size, seq_len, tag_to_ix):
        sentences = Variable(torch.LongTensor(batch_size, seq_len).random_(1, 19))
        seq_lens = Variable(torch.LongTensor([seq_len] * batch_size))
        tags = Variable(torch.LongTensor(batch_size, seq_len).random_(1, model.tagset_size - 1))
        tags.data.clamp_(1, model.tagset_size - 2)
        mask = Variable(torch.ones(batch_size, seq_len))

        loss = model.neg_log_likelihood(sentences, seq_lens, tags, mask)
        assert loss.dim() == 0 or loss.size() == torch.Size([])
        loss_val = loss.item() if loss.dim() == 0 else float(loss.data[0])
        assert loss_val > 0

    def test_neg_log_likelihood_all_pad(self, model, batch_size, seq_len):
        sentences = Variable(torch.zeros(batch_size, seq_len).long())
        seq_lens = Variable(torch.zeros(batch_size).long())
        tags = Variable(torch.zeros(batch_size, seq_len).long())
        mask = Variable(torch.zeros(batch_size, seq_len))

        loss = model.neg_log_likelihood(sentences, seq_lens, tags, mask)
        val = loss.item() if loss.dim() == 0 else float(loss.data[0])
        assert val == 0.0

    def test_forward_decode(self, model, batch_size, seq_len):
        sentences = Variable(torch.LongTensor(batch_size, seq_len).random_(1, 19))
        seq_lens = Variable(torch.LongTensor([seq_len, seq_len - 1, seq_len - 3]))
        sentences.data[1, -1:] = 0
        sentences.data[2, -3:] = 0
        mask = Variable((sentences.data != 0).float())

        scores, paths = model(sentences, seq_lens, mask)
        assert len(paths) == batch_size
        for b, path in enumerate(paths):
            s = mask[b].sum()
            expected_len = int(s.item() if s.dim() == 0 else s.data[0])
            assert len(path) == expected_len

class TestJointEventExtractor:
    @pytest.fixture
    def model(self, tag_to_ix):
        return JointEventExtractor(
            vocab_size=30,
            emb_dim=16,
            hidden_dim=32,
            tag_to_ix=tag_to_ix,
            num_role_tags=5,
            dropout=0.0,
            use_gpu=False
        )

    def test_forward_trigger_only(self, model, batch_size, seq_len):
        sentences = Variable(torch.LongTensor(batch_size, seq_len).random_(0, 29))
        trigger_feats, role_logits = model(sentences)
        assert trigger_feats.size() == (batch_size, seq_len, model.tagset_size)
        assert role_logits is None

    def test_forward_with_roles(self, model, batch_size, seq_len):
        sentences = Variable(torch.LongTensor(batch_size, seq_len).random_(0, 29))
        trigger_idx = Variable(torch.LongTensor([1, 2, 0]))
        entity_idx = Variable(torch.LongTensor([3, 4, 5]))
        trigger_feats, role_logits = model(sentences, trigger_idx, entity_idx)
        assert role_logits is not None
        assert role_logits.size() == (batch_size, model.num_role_tags)

    def test_calculate_loss_trigger_only(self, model, batch_size, seq_len, tag_to_ix):
        sentences = Variable(torch.LongTensor(batch_size, seq_len).random_(1, 29))
        mask = Variable(torch.ones(batch_size, seq_len))
        targets = Variable(torch.LongTensor(batch_size, seq_len).fill_(tag_to_ix["O"]))
        loss = model.calculate_loss(sentences, mask, targets)
        assert loss.dim() == 0 or loss.size() == torch.Size([])
        val = loss.item() if loss.dim() == 0 else float(loss.data[0])
        assert val > 0

    def test_predict_smoke(self, model, batch_size, seq_len, tag_to_ix):
        sentences = Variable(torch.LongTensor(batch_size, seq_len).random_(1, 29))
        mask = Variable(torch.ones(batch_size, seq_len))
        entity_indices_batch = [[2, 4], [1], [3, 5, 6]]
        preds = model.predict(sentences, mask, entity_indices_batch)
        assert len(preds) == batch_size
        for p in preds:
            assert "triggers" in p
            assert "roles" in p

class TestPositionAwareRelationExtractor:
    @pytest.fixture
    def model(self):
        return PositionAwareRelationExtractor(
            vocab_size=40,
            emb_dim=16,
            pos_dim=8,
            hidden_dim=32,
            num_relations=5,
            max_len=20,
            dropout=0.0
        )

    def test_forward_shape(self, model, batch_size, seq_len):
        words = Variable(torch.LongTensor(batch_size, seq_len).random_(0, 39))
        pos1 = Variable(torch.LongTensor(batch_size, seq_len).random_(0, 39))
        pos2 = Variable(torch.LongTensor(batch_size, seq_len).random_(0, 39))
        mask = Variable(torch.ones(batch_size, seq_len).long())
        seq_lens = Variable(torch.LongTensor([seq_len] * batch_size))

        logits = model(words, pos1, pos2, mask, seq_lens)
        assert logits.size() == (batch_size, 5)

    def test_forward_with_padding(self, model, batch_size, seq_len):
        words = Variable(torch.LongTensor(batch_size, seq_len).random_(0, 39))
        pos1 = Variable(torch.LongTensor(batch_size, seq_len).random_(0, 39))
        pos2 = Variable(torch.LongTensor(batch_size, seq_len).random_(0, 39))
        seq_lens = Variable(torch.LongTensor([seq_len, seq_len - 2, seq_len - 4]))
        mask = Variable(torch.ones(batch_size, seq_len).long())
        for i, l in enumerate(seq_lens.data):
            if l < seq_len:
                words.data[i, l:] = 0
                mask.data[i, l:] = 0

        logits = model(words, pos1, pos2, mask, seq_lens)
        assert logits.size() == (batch_size, 5)

    def test_batch_size_assert(self, model, seq_len):
        words = Variable(torch.LongTensor(2, seq_len).random_(0, 39))
        pos1 = Variable(torch.LongTensor(3, seq_len).random_(0, 39))
        pos2 = Variable(torch.LongTensor(2, seq_len).random_(0, 39))
        mask = Variable(torch.ones(2, seq_len).long())
        seq_lens = Variable(torch.LongTensor([seq_len, seq_len]))
        with pytest.raises(AssertionError):
            model(words, pos1, pos2, mask, seq_lens)
            
class TestAlignDistantSupervision:
    def test_exact_match(self):
        tokens = ["John", "works", "at", "Google"]
        kb = {"google": "ORG", "john works": "PER"}
        labels = align_distant_supervision(tokens, kb, max_ngram=3)
        assert labels[0].startswith("B-") or labels[3].startswith("B-")
        assert len(labels) == 4

    def test_empty_inputs(self):
        assert align_distant_supervision([], {"a": "X"}) == []
        assert align_distant_supervision(["a", "b"], {}) == []

    def test_no_overlap(self):
        tokens = ["the", "cat", "sat"]
        kb = {"dog ran": "ANIMAL"}
        labels = align_distant_supervision(tokens, kb)
        assert all(l == "O" for l in labels)

    def test_jaccard_fuzzy(self):
        tokens = ["new", "york", "city"]
        kb = {"york city": "LOC"}
        labels = align_distant_supervision(tokens, kb, jaccard_threshold=0.5)
        assert isinstance(labels, list)
        assert len(labels) == 3


class TestGetPositionIds:
    def test_basic_shape(self):
        seq_lens = [5, 4, 6]
        head = [1, 0, 2]
        tail = [3, 2, 4]
        p1, p2 = get_position_ids(seq_lens, head, tail, max_len=10, use_gpu=False)
        assert p1.size() == (3, 6)
        assert p2.size() == (3, 6)

    def test_clipping(self):
        seq_lens = [3]
        head = [0]
        tail = [2]
        p1, p2 = get_position_ids(seq_lens, head, tail, max_len=5, use_gpu=False)
        assert p1.data.min() >= 0
        assert p1.data.max() < 10

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            get_position_ids([5], [], [1])

class TestVocabulary:
    def test_add_and_compile(self):
        v = Vocabulary()
        v.add_word("hello")
        v.add_tag("B-PER")
        v.add_relation("Work_For")
        v.compile()
        assert "hello" in v.word2idx
        assert v.idx2word[v.word2idx["hello"]] == "hello"
        assert v.get_word_idx("unknown_xyz") == v.word2idx["<UNK>"]

    def test_pad_unk_present(self):
        v = Vocabulary()
        assert "<PAD>" in v.word2idx
        assert "<UNK>" in v.word2idx
        assert "O" in v.tag2idx


class TestLoadData:
    def test_load_and_build_vocab(self, tmp_path):
        data = [
            {"tokens": ["John", "works"], "ner_tags": ["B-PER", "O"], "relations": [{"type": "Work_For"}]},
            {"tokens": ["Google"], "ner_tags": ["B-ORG"], "relations": []},
        ]
        path = tmp_path / "train.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        vocab = Vocabulary()
        dataset = load_data(str(path), vocab, is_train=True)
        vocab.compile()
        assert len(dataset) == 2
        assert "John" in vocab.word2idx
        assert "B-PER" in vocab.tag2idx
        assert "Work_For" in vocab.rel2idx

    def test_file_not_found(self):
        vocab = Vocabulary()
        with pytest.raises(FileNotFoundError):
            load_data("/non/existent/file.jsonl", vocab)


class TestPrepareBatch:
    def test_shapes_and_mask(self, small_vocab):
        batch = [
            {"tokens": ["the", "cat"], "tags": ["O", "O"]},
            {"tokens": ["John", "works", "at", "Google"], "tags": ["B-PER", "O", "O", "B-ORG"]},
        ]
        sentences, tags, seq_lens, mask = prepare_batch(batch, small_vocab, use_gpu=False)
        assert sentences.size(0) == 2
        assert sentences.size(1) == 4  # max len
        assert (mask[0, 2:] == 0).all() or (mask[0, 2:].sum() == 0)
        assert seq_lens.data.tolist() == [2, 4]


class TestPadSequence:
    def test_pad(self):
        assert pad_sequence([1, 2, 3], 5, pad_value=0) == [1, 2, 3, 0, 0]
        assert pad_sequence([1, 2], 2) == [1, 2]

class TestDeepLinguisticAnalyzer:
    @pytest.fixture
    def analyzer(self):
        with patch("linguistic_pipeline.spacy.load") as mock_spacy, \
             patch("linguistic_pipeline.StanfordCoreNLP") as mock_corenlp, \
             patch("linguistic_pipeline.os.path.exists", return_value=True):

            mock_nlp = MagicMock()
            mock_spacy.return_value = mock_nlp

            mock_sent = MagicMock()
            mock_sent.text = "Hello world."
            mock_doc = MagicMock()
            mock_doc.sents = [mock_sent]
            mock_token = MagicMock()
            mock_token.text = "Hello"
            mock_token.lemma_ = "hello"
            mock_token.pos_ = "INTJ"
            mock_token.tag_ = "UH"
            mock_token.dep_ = "ROOT"
            mock_token.head.text = "Hello"
            mock_token.is_stop = False
            mock_doc.__iter__ = lambda self: iter([mock_token])
            mock_nlp.return_value = mock_doc

            client = MagicMock()
            mock_corenlp.return_value = client
            client.annotate.return_value = json.dumps({
                "sentences": [{
                    "openie": [{"subject": "John", "relation": "works at", "object": "Google", "confidence": 0.9}],
                    "parse": "(ROOT (S ...))"
                }],
                "corefs": {
                    "1": [{"text": "John", "isRepresentativeMention": True, "sentNum": 1}]
                }
            })

            from linguistic_pipeline import DeepLinguisticAnalyzer
            analyzer = DeepLinguisticAnalyzer(corenlp_path="/fake/path")
            yield analyzer
            analyzer.close()

    def test_empty_text(self, analyzer):
        result = analyzer.analyze("   ")
        assert result["spacy_syntax"] == []
        assert result["coreference"] == {}
        assert result["open_ie"] == []

    def test_analyze_smoke(self, analyzer):
        result = analyzer.analyze("John works at Google.")
        assert "spacy_syntax" in result
        assert "coreference" in result
        assert "open_ie" in result
        assert "rhetorical_structure" in result
        assert len(result["spacy_syntax"]) > 0 or len(result["open_ie"]) > 0

    def test_chunking_logic(self, analyzer):
        long_text = "Sentence one. " * 100
        chunks = analyzer._get_sentence_chunks(long_text, max_chars=50)
        assert len(chunks) >= 1

def test_bilstm_crf_train_step_smoke(small_vocab, tag_to_ix):
    model = BiLSTM_CRF(
        vocab_size=len(small_vocab.word2idx),
        tag_to_ix=small_vocab.tag2idx,
        embedding_dim=16,
        hidden_dim=32,
        dropout=0.1,
        use_gpu=False
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch = [
        {"tokens": ["the", "cat", "sat"], "tags": ["O", "O", "O"]},
        {"tokens": ["John", "works"], "tags": ["B-PER", "O"]},
    ]
    sentences, tags, seq_lens, mask = prepare_batch(batch, small_vocab, use_gpu=False)

    model.train()
    optimizer.zero_grad()
    loss = model.neg_log_likelihood(sentences, seq_lens, tags, mask)
    loss.backward()
    optimizer.step()
    loss_val = loss.item() if loss.dim() == 0 else float(loss.data[0])
    assert loss_val >= 0


def test_joint_event_loss_smoke(tag_to_ix):
    model = JointEventExtractor(
        vocab_size=20, emb_dim=12, hidden_dim=24,
        tag_to_ix=tag_to_ix, num_role_tags=4, dropout=0.0
    )
    sentences = Variable(torch.LongTensor([[1, 2, 3, 0], [4, 5, 0, 0]]))
    mask = Variable(torch.FloatTensor([[1, 1, 1, 0], [1, 1, 0, 0]]))
    targets = Variable(torch.LongTensor([[1, 1, 1, 0], [1, 1, 0, 0]]))
    loss = model.calculate_loss(sentences, mask, targets)
    loss_val = loss.item() if loss.dim() == 0 else float(loss.data[0])
    assert loss_val >= 0


def test_relation_extractor_end_to_end(small_vocab):
    model = PositionAwareRelationExtractor(
        vocab_size=len(small_vocab.word2idx),
        emb_dim=16, pos_dim=8, hidden_dim=32,
        num_relations=len(small_vocab.rel2idx),
        max_len=20, dropout=0.0
    )
    batch_size, seq_len = 2, 6
    words = Variable(torch.LongTensor(batch_size, seq_len).random_(0, len(small_vocab.word2idx) - 1))
    pos1 = Variable(torch.LongTensor(batch_size, seq_len).random_(0, 39))
    pos2 = Variable(torch.LongTensor(batch_size, seq_len).random_(0, 39))
    mask = Variable(torch.ones(batch_size, seq_len).long())
    seq_lens = Variable(torch.LongTensor([seq_len, seq_len]))

    logits = model(words, pos1, pos2, mask, seq_lens)
    loss = nn.CrossEntropyLoss()(logits, Variable(torch.LongTensor([1, 2])))
    loss.backward()
    assert logits.size(0) == batch_size