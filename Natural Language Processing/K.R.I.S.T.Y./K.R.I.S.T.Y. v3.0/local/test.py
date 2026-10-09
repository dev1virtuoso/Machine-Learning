import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch_geometric.data import Data

from kristy_arch import KristyConfig, KristyEngine, KristyLogitWarper
from linguistic_engine import LinguisticEngine
from main import KristySystem2019
from prepare import prepare_2019_dataset
from retrieval import BM25Retriever


class TestKristyConfig(unittest.TestCase):

    def test_default_config(self):
        config = KristyConfig()
        self.assertEqual(config.model_name, "bert-base-uncased")
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.logic_threshold, 0.7)
        self.assertEqual(config.graph_layers, 2)

    def test_custom_config(self):
        config = KristyConfig(hidden_size=512, logic_threshold=0.5, graph_layers=3)
        self.assertEqual(config.hidden_size, 512)
        self.assertEqual(config.logic_threshold, 0.5)
        self.assertEqual(config.graph_layers, 3)


class TestKristyEngine(unittest.TestCase):

    @patch("kristy_arch.BertModel.from_pretrained")
    def setUp(self, mock_bert_from_pretrained):
        self.config = KristyConfig(
            hidden_size=64, graph_hidden=32, graph_output_dim=32
        )

        self.mock_bert = MagicMock()
        self.mock_bert.config.vocab_size = 1000
        self.mock_bert.device = torch.device("cpu")
        mock_bert_from_pretrained.return_value = self.mock_bert

        self.engine = KristyEngine(self.config)

    def test_forward_without_knowledge_graph(self):
        batch_size = 2
        seq_len = 10

        fake_last_hidden = torch.randn(batch_size, seq_len, self.config.hidden_size)
        fake_pooled = torch.randn(batch_size, self.config.hidden_size)
        self.mock_bert.return_value = (fake_last_hidden, fake_pooled)

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        logits, logic_score = self.engine(input_ids)

        self.assertEqual(logits.shape, (batch_size, seq_len, 1000))
        self.assertEqual(logic_score.shape, (batch_size, 1))
        self.assertTrue((logic_score >= 0.0).all() and (logic_score <= 1.0).all())

    def test_forward_with_valid_knowledge_graph(self):
        batch_size = 2
        seq_len = 10

        fake_last_hidden = torch.randn(batch_size, seq_len, self.config.hidden_size)
        fake_pooled = torch.randn(batch_size, self.config.hidden_size)
        self.mock_bert.return_value = (fake_last_hidden, fake_pooled)

        x = torch.randn(batch_size, self.config.hidden_size)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        kg = Data(x=x, edge_index=edge_index)

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        logits, logic_score = self.engine(input_ids, knowledge_graph=kg)

        self.assertEqual(logits.shape, (batch_size, seq_len, 1000))
        self.assertEqual(logic_score.shape, (batch_size, 1))

    def test_forward_with_empty_node_features_knowledge_graph(self):
        batch_size = 2
        seq_len = 10

        fake_last_hidden = torch.randn(batch_size, seq_len, self.config.hidden_size)
        fake_pooled = torch.randn(batch_size, self.config.hidden_size)
        self.mock_bert.return_value = (fake_last_hidden, fake_pooled)

        kg = Data(x=None, edge_index=None)

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        logits, logic_score = self.engine(input_ids, knowledge_graph=kg)

        self.assertEqual(logits.shape, (batch_size, seq_len, 1000))


class TestKristyLogitWarper(unittest.TestCase):

    def test_warper_call(self):
        mock_model = MagicMock()
        fake_bert_output = MagicMock()
        fake_bert_output.pooler_output = torch.randn(1, 64)
        mock_model.bert.return_value = fake_bert_output

        mock_model.reasoning_gate.return_value = torch.tensor([[0.8]])

        warper = KristyLogitWarper(model=mock_model, threshold=0.7)

        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        scores = torch.tensor([[2.0, 4.0, 6.0]], dtype=torch.float32)

        updated_scores = warper(input_ids, scores)

        expected_scores = scores * 0.2
        self.assertTrue(torch.allclose(updated_scores, expected_scores, atol=1e-4))


class TestBM25Retriever(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.index_file = os.path.join(self.temp_dir.name, "index.jsonl")

        self.sample_data = [
            {"text": "apple banana fruit", "label_ids": [101, 1, 2]},
            {"text": "apple pie recipe dessert", "label_ids": [101, 1, 3]},
            {"text": "cat dog pet animal", "label_ids": [101, 4, 5]},
        ]
        with open(self.index_file, "w", encoding="utf-8") as f:
            for item in self.sample_data:
                f.write(json.dumps(item) + "\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            BM25Retriever("non_existent_file.jsonl")

    def test_get_top_k_with_scores(self):
        retriever = BM25Retriever(self.index_file)
        results = retriever.get_top_k(
            query="apple fruit", k=2, return_scores=True
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0]), 3)
        self.assertIn("apple", results[0][0])
        self.assertIsInstance(results[0][2], float)

    def test_get_top_k_without_scores(self):
        retriever = BM25Retriever(self.index_file)
        results = retriever.get_top_k(
            query="cat pet", k=1, return_scores=False
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 2)
        self.assertEqual(results[0][0], "cat dog pet animal")


class TestLinguisticEngine(unittest.TestCase):

    @patch("linguistic_engine.pipeline")
    @patch("linguistic_engine.spacy.load")
    def test_extract_entities(self, mock_spacy_load, mock_pipeline):
        mock_nlp = MagicMock()
        mock_ent = MagicMock()
        mock_ent.text = "Apple"
        mock_ent.label_ = "ORG"

        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp

        engine = LinguisticEngine()
        entities = engine.extract_entities("Apple is buying a startup")

        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0], {"text": "Apple", "label": "ORG"})

    @patch("linguistic_engine.pipeline")
    @patch("linguistic_engine.spacy.load")
    def test_polish_response_success(self, mock_spacy_load, mock_pipeline):
        mock_fixer = MagicMock()
        mock_fixer.return_value = [{"generated_text": "  This is a clean sentence.  "}]
        mock_pipeline.return_value = mock_fixer

        engine = LinguisticEngine()
        polished = engine.polish_response("this is a clean sentence")

        self.assertEqual(polished, "This is a clean sentence.")

    @patch("linguistic_engine.pipeline")
    @patch("linguistic_engine.spacy.load")
    def test_polish_response_empty_or_exception(self, mock_spacy_load, mock_pipeline):
        mock_fixer = MagicMock()
        mock_fixer.side_effect = Exception("Model Error")
        mock_pipeline.return_value = mock_fixer

        engine = LinguisticEngine()
        self.assertEqual(engine.polish_response(""), "")
        raw_text = "unparsed text"
        self.assertEqual(engine.polish_response(raw_text), raw_text)


class TestPrepareDataset(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_dir = os.path.join(self.temp_dir.name, "input")
        os.makedirs(self.input_dir, exist_ok=True)
        self.output_file = os.path.join(self.temp_dir.name, "train.jsonl")

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("prepare.BertTokenizer.from_pretrained")
    def test_prepare_2019_dataset(self, mock_tokenizer_cls):
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [101, 2000, 102]
        mock_tokenizer_cls.return_value = mock_tokenizer

        valid_json = os.path.join(self.input_dir, "doc1.json")
        with open(valid_json, "w", encoding="utf-8") as f:
            json.dump({"text": "This is a valid test documentation text."}, f)

        short_json = os.path.join(self.input_dir, "doc2.json")
        with open(short_json, "w", encoding="utf-8") as f:
            json.dump({"text": "short"}, f)

        prepare_2019_dataset(self.input_dir, self.output_file)

        self.assertTrue(os.path.exists(self.output_file))

        with open(self.output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 1)
        data = json.loads(lines[0])
        self.assertIn("text", data)
        self.assertIn("label_ids", data)
        self.assertIn("logic_score", data)
        self.assertEqual(data["logic_score"], 0.0)


class TestKristySystem2019(unittest.TestCase):

    @patch("main.LinguisticEngine")
    @patch("main.BM25Retriever")
    @patch("main.KristyEngine")
    @patch("main.T5ForConditionalGeneration")
    @patch("main.T5Tokenizer")
    @patch("main.os.path.exists")
    def test_build_graph_from_text(
        self,
        mock_exists,
        mock_t5_tok,
        mock_t5_gen,
        mock_kristy_engine,
        mock_bm25,
        mock_ling,
    ):
        mock_exists.return_value = True
        system = KristySystem2019(index_path="fake_index.jsonl")

        graph = system._build_graph_from_text("apple pie apple")
        self.assertIsInstance(graph, Data)
        self.assertEqual(graph.x.shape[0], 3)
        self.assertEqual(graph.edge_index.shape[0], 2)

    @patch("main.LinguisticEngine")
    @patch("main.BM25Retriever")
    @patch("main.KristyEngine")
    @patch("main.T5ForConditionalGeneration")
    @patch("main.T5Tokenizer")
    @patch("main.os.path.exists")
    def test_handle_query_suppression_warning(
        self,
        mock_exists,
        mock_t5_tok,
        mock_t5_gen,
        mock_kristy_engine_cls,
        mock_bm25_cls,
        mock_ling_cls,
    ):
        mock_exists.return_value = True

        mock_engine_inst = MagicMock()
        mock_engine_inst.to.return_value = mock_engine_inst
        mock_logits = torch.randn(1, 10, 1000)
        mock_logic_score = torch.tensor([[0.95]])
        mock_engine_inst.return_value = (mock_logits, mock_logic_score)
        mock_kristy_engine_cls.return_value = mock_engine_inst

        system = KristySystem2019(index_path="fake_index.jsonl")

        response = system.handle_query("Explain financial results.")
        self.assertIn("WARNING: Logic anomaly detected.", response)

    @patch("main.LinguisticEngine")
    @patch("main.BM25Retriever")
    @patch("main.KristyEngine")
    @patch("main.T5ForConditionalGeneration")
    @patch("main.T5Tokenizer")
    @patch("main.os.path.exists")
    def test_handle_query_normal_flow(
        self,
        mock_exists,
        mock_t5_tok_cls,
        mock_t5_gen_cls,
        mock_kristy_engine_cls,
        mock_bm25_cls,
        mock_ling_cls,
    ):
        mock_exists.return_value = True

        mock_engine_inst = MagicMock()
        mock_engine_inst.to.return_value = mock_engine_inst
        mock_engine_inst.return_value = (
            torch.randn(1, 10, 1000),
            torch.tensor([[0.2]]),
        )
        mock_kristy_engine_cls.return_value = mock_engine_inst

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode_plus.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_tokenizer.decode.return_value = "generated answer raw"
        mock_t5_tok_cls.from_pretrained.return_value = mock_tokenizer

        mock_generator = MagicMock()
        mock_generator.generate.return_value = torch.tensor([[10, 20, 30]])
        mock_t5_gen_cls.from_pretrained.return_value = mock_generator

        mock_ling_inst = MagicMock()
        mock_ling_inst.polish_response.return_value = "Generated Answer Polished."
        mock_ling_cls.return_value = mock_ling_inst

        system = KristySystem2019(index_path="fake_index.jsonl")
        response = system.handle_query("What is AI?")

        self.assertEqual(response, "Generated Answer Polished.")


if __name__ == "__main__":
    unittest.main()