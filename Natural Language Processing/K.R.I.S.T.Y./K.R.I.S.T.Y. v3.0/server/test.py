import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch
from fastapi.testclient import TestClient
from torch_geometric.data import Data

from api import app, state, QueryRequest, _build_graph
from kristy_arch import KristyConfig, KristyEngine, KristyLogitWarper
from linguistic_engine import LinguisticEngine
from prepare import prepare_2019_dataset
from retrieval import BM25Retriever
from train import KristyDataset, run_train


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
        self.mock_bert.config.vocab_size = 30522
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

        self.assertEqual(logits.shape, (batch_size, seq_len, 30522))
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

        self.assertEqual(logits.shape, (batch_size, seq_len, 30522))
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

        self.assertEqual(logits.shape, (batch_size, seq_len, 30522))


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


class TestTrainModule(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_file = os.path.join(self.temp_dir.name, "sample.jsonl")
        self.sample_data = [
            {"text": "Sample text for training.", "label_ids": [101, 200, 102], "logic_score": 0.0},
            {"text": "Another training sample text.", "label_ids": [101, 201, 102], "logic_score": 1.0},
        ]
        with open(self.data_file, "w", encoding="utf-8") as f:
            for d in self.sample_data:
                f.write(json.dumps(d) + "\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_kristy_dataset_getitem(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.encode_plus.return_value = {
            "input_ids": torch.tensor([[101, 200, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        dataset = KristyDataset(self.data_file, mock_tokenizer, max_len=16)
        self.assertEqual(len(dataset), 2)

        item = dataset[0]
        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("labels", item)
        self.assertIn("logic_label", item)
        self.assertEqual(item["labels"].shape[0], 16)
        self.assertEqual(item["logic_label"].item(), 0.0)

    @patch("train.AdamW")
    @patch("train.KristyEngine")
    @patch("train.BertTokenizer.from_pretrained")
    def test_run_train_flow(
        self, mock_tok_cls, mock_engine_cls, mock_adam
    ):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.encode_plus.return_value = {
            "input_ids": torch.full((1, 128), 101, dtype=torch.long),
            "attention_mask": torch.ones((1, 128), dtype=torch.long),
        }
        mock_tok_cls.return_value = mock_tokenizer

        mock_engine = MagicMock()
        mock_engine.to.return_value = mock_engine

        mock_engine.state_dict.return_value = {"dummy_weight": torch.tensor([1.0])}

        dummy_param = torch.nn.Parameter(torch.randn(1, requires_grad=True))
        mock_engine.parameters.return_value = [dummy_param]

        def fake_forward(input_ids, attention_mask=None, **kwargs):
            bsize = input_ids.size(0)
            seq_len = input_ids.size(1)
            logits = torch.randn(bsize, seq_len, 30522) + (dummy_param * 0)
            logic_score = torch.sigmoid(torch.randn(bsize, 1) + (dummy_param * 0))
            return logits, logic_score

        mock_engine.side_effect = fake_forward
        mock_engine_cls.return_value = mock_engine

        orig_dir = os.getcwd()
        os.chdir(self.temp_dir.name)
        try:
            with open("train.jsonl", "w", encoding="utf-8") as f:
                for d in self.sample_data * 5:
                    f.write(json.dumps(d) + "\n")

            run_train()
            self.assertTrue(os.path.exists("kristy_v3.0.pth"))
        finally:
            os.chdir(orig_dir)


class TestKristyAPI(unittest.TestCase):

    def setUp(self):
        state.config = KristyConfig(hidden_size=64, logic_threshold=0.7)
        state.device = torch.device("cpu")

        state.tokenizer = MagicMock()
        state.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        state.tokenizer.decode.return_value = "Parsed answer"

        state.retriever = MagicMock()
        state.retriever.get_top_k.return_value = [("Retrieved doc context", [101], 0.9)]

        state.model = MagicMock()
        state.model.to.return_value = state.model

        state.generator = MagicMock()
        state.generator.to.return_value = state.generator
        state.generator.generate.return_value = torch.tensor([[10, 20]])

        state.logit_warper = MagicMock()
        state.ling = MagicMock()
        state.ling.polish_response.side_effect = lambda x: f"Polished: {x}"

        self.client = TestClient(app)

    def test_build_graph(self):
        graph = _build_graph("apple pie dessert")
        self.assertEqual(graph.x.shape[0], 3)
        self.assertEqual(graph.x.shape[1], state.config.hidden_size)

    def test_infer_empty_question(self):
        response = self.client.post("/infer", json={"question": "   "})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Question cannot be empty", response.json()["detail"])

    def test_infer_normal_flow(self):
        state.model.return_value = (torch.randn(1, 10, 30522), torch.tensor([[0.2]]))

        response = self.client.post(
            "/infer", json={"question": "What is AI?", "context": "AI is intelligence."}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["risk_flag"])
        self.assertIn("Polished: Parsed answer", data["answer"])

    def test_infer_risk_suppression(self):
        state.model.return_value = (torch.randn(1, 10, 30522), torch.tensor([[0.95]]))

        response = self.client.post(
            "/infer", json={"question": "Generate anomalous data"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["risk_flag"])
        self.assertIn("Logic anomaly detected", data["answer"])


if __name__ == "__main__":
    unittest.main()