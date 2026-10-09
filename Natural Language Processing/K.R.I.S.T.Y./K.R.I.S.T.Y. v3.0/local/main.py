import logging
import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from kristy_arch import KristyEngine, KristyLogitWarper, KristyConfig
from linguistic_engine import LinguisticEngine
from retrieval import BM25Retriever
from torch_geometric.data import Data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

class KristySystem2019:
    def __init__(self, model_path: str = "kristy_v3.0.pth", index_path: str = "index.jsonl"):
        self.config = KristyConfig()
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.tokenizer_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"The index file {index_path} does not exist.")
        self.retriever = BM25Retriever(index_path, device=self.device)

        self.model = KristyEngine(self.config).to(self.device)
        try:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            logging.info(f"Loaded checkpoint: {model_path}")
        except Exception:
            logging.warning("Checkpoint not found, using pre‑trained BERT weights")

        self.model.eval()

        self.ling = LinguisticEngine()

        self.generator = T5ForConditionalGeneration.from_pretrained(
            self.config.tokenizer_name
        ).to(self.device)
        self.logit_warper = KristyLogitWarper(
            self.model, threshold=self.config.logic_threshold
        )

    def _build_graph_from_text(self, text: str) -> Data:
        tokens = text.split()
        size = len(tokens)
        idx_map = {w: i for i, w in enumerate(tokens)}

        edges = []
        for i, wi in enumerate(tokens):
            for j, wj in enumerate(tokens):
                if i != j and (wi in wj or wj in wi):
                    edges.append([i, j])
        if not edges:
            edges = [[i, i] for i in range(size)]

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.randn((size, self.config.hidden_size), device=self.device)
        return Data(x=x, edge_index=edge_index)

    def handle_query(self, text: str) -> str:
        try:
            entities = self.ling.extract_entities(text)
            logging.debug(f"Entities: {entities}")

            top_k = self.retriever.get_top_k(text, k=self.config.retrieval_k, return_scores=True)
            logging.debug(f"Top‑k retrieved: {top_k}")

            context = " ".join([t[0] for t in top_k])
            knowledge_graph = self._build_graph_from_text(context)

            inputs = self.tokenizer.encode_plus(
                text,
                return_tensors="pt",
                max_length=self.config.max_seq_length,
                truncation=True,
                padding="max_length",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits, logic_score = self.model(
                    **inputs, knowledge_graph=knowledge_graph
                )
            risk = logic_score.squeeze().cpu().item()
            logging.debug(f"Logic score: {risk:.4f}")

            if risk > self.config.logic_threshold:
                return (
                    "WARNING: Logic anomaly detected. "
                    "Output suppressed for factual safety."
                )

            prompt = f"Answer the following question: {text}\nContext: {context}"
            prompt_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.config.max_seq_length,
                truncation=True,
                padding="max_length",
            )
            for k, v in prompt_ids.items():
                prompt_ids[k] = v.to(self.device)

            generated_ids = self.generator.generate(
                input_ids=prompt_ids["input_ids"],
                attention_mask=prompt_ids["attention_mask"],
                logits_processor=[self.logit_warper],
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
            raw_answer = self.tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )

            return self.ling.polish_response(raw_answer)

        except Exception as exc:
            logging.exception(f"handle_query error: {exc}")
            return "ERROR: Unable to process the request."

if __name__ == "__main__":
    kristy = KristySystem2019()
    user_in = "Explain the fiscal results found in the latest JSON reports."
    print(f"USER: {user_in}")
    print(f"KRISTY: {kristy.handle_query(user_in)}")
