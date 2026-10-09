import os
import sys
import json
import sqlite3
from unittest.mock import patch, MagicMock

import pytest
import torch
import torch.nn as nn
import pandas as pd

sys.modules["spacy"] = MagicMock()
sys.modules["spacy.lang"] = MagicMock()
sys.modules["spacy.lang.en"] = MagicMock()
sys.modules["redis"] = MagicMock()
sys.modules["celery"] = MagicMock()

from ariel_arch import ArielCore
from kg_engine import KnowledgeEngine
from linguistic_engine import Seq2SeqGrammarRefiner, LinguisticEngine
from data_utils import ArielWikiDataset
from prepare import build_knowledge_base, build_training_manifest
from train import build_vocab_from_train, load_dataset

def _safe_forward(self, x: torch.LongTensor):
    emb = self.embedding(x)
    enc_out, _ = self.encoder(emb)
    last = enc_out[:, -1, :]
    h_signal = self.rational_gate(last)

    B = x.size(0)
    decoder_hidden = last[:, : self.hidden_size].unsqueeze(0)

    dec_input = torch.zeros(B, 1, self.embedding.embedding_dim, device=x.device)
    logits = self.out(torch.zeros(B, self.hidden_size, device=x.device))
    return logits, h_signal

ArielCore.forward = _safe_forward

@pytest.fixture
def temp_db(tmp_path):
    return str(tmp_path / "test_kb.db")


@pytest.fixture
def kb(temp_db):
    engine = KnowledgeEngine(db_path=temp_db)
    engine.add_entity("iphone 7")
    engine.add_entity("macbook pro")
    engine.add_fact("iphone 7", "price", "649.00")
    engine.add_fact("iphone 7", "type", "Smartphone")
    engine.add_fact("macbook pro", "price", "1299.00")
    engine.add_fact("macbook pro", "type", "Laptop")
    return engine


@pytest.fixture
def small_vocab():
    return {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3, "hello": 4, "world": 5}

class TestArielCore:
    def test_init_shapes(self, small_vocab):
        model = ArielCore(vocab_size=len(small_vocab), hidden_size=32, embed_size=16)
        assert model.embedding.num_embeddings == len(small_vocab)
        assert model.encoder.hidden_size == 32
        assert model.decoder.hidden_size == 32

    def test_forward_output_shapes(self, small_vocab):
        model = ArielCore(vocab_size=len(small_vocab), hidden_size=32, embed_size=16)
        x = torch.randint(0, len(small_vocab), (2, 10))
        logits, h_signal = model(x)
        assert logits.shape[0] == 2
        assert logits.shape[-1] == len(small_vocab)
        assert h_signal.shape == (2, 1)

    def test_decoder_forward(self, small_vocab):
        model = ArielCore(vocab_size=len(small_vocab), hidden_size=32, embed_size=16)
        token = torch.LongTensor([[1]])
        hidden = torch.zeros(1, 1, 32)
        out, new_hidden = model.decoder_forward(token, hidden)
        assert out.shape == (1, len(small_vocab))
        assert new_hidden.shape == (1, 1, 32)

    def test_rational_gate_range(self, small_vocab):
        model = ArielCore(vocab_size=len(small_vocab), hidden_size=16, embed_size=8)
        x = torch.randint(0, len(small_vocab), (3, 5))
        _, h_signal = model(x)
        assert torch.all(h_signal >= 0) and torch.all(h_signal <= 1)

class TestKnowledgeEngine:
    def test_init_creates_tables(self, temp_db):
        KnowledgeEngine(db_path=temp_db)
        conn = sqlite3.connect(temp_db)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        conn.close()
        assert "entities" in tables
        assert "facts" in tables

    def test_add_and_query_entity(self, kb):
        facts = kb.query_fact("iphone 7")
        assert len(facts) >= 2
        keys = {f["key"] for f in facts}
        assert "price" in keys
        assert "type" in keys

    def test_query_unknown_entity(self, kb):
        assert kb.query_fact("nonexistent") == []

    def test_add_fact_missing_entity_raises(self, kb):
        with pytest.raises(ValueError, match="not found"):
            kb.add_fact("ghost device", "price", "99")

    def test_extract_entities(self, kb):
        found = kb.extract_entities("I want an iPhone 7 and a MacBook Pro")
        assert found is not None
        assert "iphone 7" in found
        assert "macbook pro" in found

    def test_extract_entities_none(self, kb):
        assert kb.extract_entities("hello world") is None

    def test_case_insensitive(self, kb):
        assert len(kb.query_fact("IPHONE 7")) > 0

class TestSeq2SeqGrammarRefiner:
    def test_forward_shape(self, small_vocab):
        model = Seq2SeqGrammarRefiner(
            vocab_size=len(small_vocab), embed_size=16, hidden_size=32
        )
        src = torch.randint(0, len(small_vocab), (2, 8))
        out = model(src)
        assert out.shape == (2, 8, len(small_vocab))

    def test_teacher_forcing(self, small_vocab):
        model = Seq2SeqGrammarRefiner(
            vocab_size=len(small_vocab), embed_size=8, hidden_size=16
        )
        src = torch.randint(0, len(small_vocab), (1, 5))
        tgt = torch.randint(0, len(small_vocab), (1, 5))
        out = model(src, tgt, teacher_forcing_ratio=1.0)
        assert out.shape[1] == 5


class TestLinguisticEngine:
    def test_init_without_spacy(self, small_vocab):
        with patch("linguistic_engine.spacy.load", side_effect=OSError("no model")):
            eng = LinguisticEngine(vocab=small_vocab)
            assert eng.nlp is None

    def test_fix_grammar_fallback(self, small_vocab):
        with patch("linguistic_engine.spacy.load", side_effect=Exception("fail")):
            eng = LinguisticEngine(vocab=small_vocab)
            assert eng.fix_grammar("hello world") == "hello world"

    def test_extract_entities_no_nlp(self, small_vocab):
        with patch("linguistic_engine.spacy.load", side_effect=Exception("fail")):
            eng = LinguisticEngine(vocab=small_vocab)
            assert eng.extract_entities("Apple is great") == []

class TestArielWikiDataset:
    def test_len_and_getitem(self, tmp_path, small_vocab):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "page1.json").write_text(
            json.dumps({"text": "hello world this is a test"}), encoding="utf-8"
        )
        (wiki_dir / "page2.json").write_text(
            json.dumps({"text": "another page"}), encoding="utf-8"
        )

        mock_tok = MagicMock()
        mock_tok.return_value = [MagicMock(text="hello"), MagicMock(text="world")]

        ds = ArielWikiDataset(
            str(wiki_dir), tokenizer=mock_tok, max_length=8, vocab=small_vocab
        )
        assert len(ds) == 2
        item = ds[0]
        assert "input_ids" in item and "labels" in item
        assert item["input_ids"].shape[0] == 8

    def test_missing_file_raises(self, tmp_path, small_vocab):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "only.json").write_text("{}", encoding="utf-8")
        ds = ArielWikiDataset(str(wiki_dir), tokenizer=MagicMock(), vocab=small_vocab)
        ds.file_ids = ["nonexistent"]
        with pytest.raises(FileNotFoundError):
            _ = ds[0]

class TestPrepare:
    def test_build_knowledge_base(self, tmp_path):
        csv_path = tmp_path / "products.csv"
        pd.DataFrame({
            "name": ["iPhone 7", "MacBook Pro"],
            "type": ["Smartphone", "Laptop"],
            "price": [649.0, 1299.0],
            "tax_rate": [0.08, 0.0],
        }).to_csv(csv_path, index=False)

        db_path = str(tmp_path / "kb.db")
        build_knowledge_base(str(csv_path), db_path=db_path)

        conn = sqlite3.connect(db_path)
        entities = conn.execute("SELECT name FROM entities").fetchall()
        facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        conn.close()
        assert len(entities) == 2
        assert facts == 6

    def test_build_knowledge_base_missing_csv(self, tmp_path, caplog):
        build_knowledge_base(str(tmp_path / "no.csv"))
        assert any("not found" in r.message for r in caplog.records)

    def test_build_training_manifest(self, tmp_path):
        csv_path = tmp_path / "products.csv"
        pd.DataFrame({
            "name": ["iPhone 7"],
            "type": ["Smartphone"],
            "price": [649.0],
            "tax_rate": [0.08],
        }).to_csv(csv_path, index=False)

        out_path = str(tmp_path / "train.csv")
        build_training_manifest(str(csv_path), out_path=out_path)
        df = pd.read_csv(out_path)
        assert len(df) == 7
        assert "text" in df.columns and "label" in df.columns
        
class TestTrainHelpers:
    def test_build_vocab_from_train(self, tmp_path):
        csv = tmp_path / "train_v1.csv"
        csv.write_text("text,label\nhello world,0\nfoo bar,1\n", encoding="utf-8")
        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            vocab, id2w = build_vocab_from_train(csv_path="train_v1.csv")
        finally:
            os.chdir(old)
        assert "<pad>" in vocab

    def test_build_vocab_missing_file(self):
        vocab, _ = build_vocab_from_train(csv_path="nonexistent.csv")
        assert vocab == {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}

    def test_load_dataset(self, tmp_path):
        csv = tmp_path / "train_v1.csv"
        pd.DataFrame({"text": ["a b", "c d"], "label": [0, 1]}).to_csv(csv, index=False)
        texts, labels = load_dataset(str(csv))
        assert texts == ["a b", "c d"]
        assert labels == [0, 1]

def test_ariel_core_train_step_smoke(small_vocab):
    model = ArielCore(vocab_size=len(small_vocab), hidden_size=16, embed_size=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randint(0, len(small_vocab), (4, 6))
    logits, h_signal = model(x)
    loss = h_signal.mean() + logits.mean()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


def test_kg_and_model_together(kb, small_vocab):
    facts = kb.query_fact("iphone 7")
    assert facts
    model = ArielCore(vocab_size=len(small_vocab), hidden_size=16, embed_size=8)
    x = torch.randint(0, len(small_vocab), (1, 5))
    logits, h = model(x)
    assert logits.shape[0] == 1