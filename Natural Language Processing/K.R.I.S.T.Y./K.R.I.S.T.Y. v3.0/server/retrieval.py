import json
from pathlib import Path
from typing import List, Tuple

import torch
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, index_file: str, device: torch.device = torch.device("cpu")):
        self.index_file = Path(index_file)
        if not self.index_file.exists():
            raise FileNotFoundError(f"Index file {index_file} does not exist")

        self.documents: List[str] = []
        self.label_ids: List[List[int]] = []

        with self.index_file.open("r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.documents.append(data.get("text", ""))
                self.label_ids.append(data.get("label_ids", []))

        self.tokenized_corpus = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def get_top_k(
        self,
        query: str,
        k: int = 5,
        return_scores: bool = False,
    ) -> List[Tuple[str, List[int], float]]:
        
        try:
            tokenized_query = query.split()
            scores = self.bm25.get_scores(tokenized_query)
            top_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            results: List[Tuple[str, List[int], float]] = []
            for idx in top_ids:
                result = (
                    self.documents[idx],
                    self.label_ids[idx],
                    float(scores[idx]),
                )
                if not return_scores:
                    result = result[:2]
                results.append(result)
            return results
        except Exception as exc:
            raise RuntimeError(f"检索失败: {exc}") from exc
