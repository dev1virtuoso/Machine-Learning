import os
import json
from typing import Dict, List

import torch
from torch.utils.data import Dataset

class ArielWikiDataset(Dataset):

    def __init__(
        self,
        json_dir: str,
        tokenizer,
        max_length: int = 512,
        vocab: Dict[str, int] | None = None,
    ) -> None:
        self.json_dir = json_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab = vocab or {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.file_ids: List[str] = [
            f.split(".")[0]
            for f in os.listdir(json_dir)
            if f.endswith(".json")
        ]

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        wiki_id = self.file_ids[idx]
        file_path = os.path.join(self.json_dir, f"{wiki_id}.json")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        content = data.get("text", "")

        tokens = self.tokenizer(content)
        token_ids = [
            self.vocab.get(tok.text, self.vocab["<unk>"]) for tok in tokens
        ]

        if len(token_ids) < self.max_length:
            token_ids += [self.vocab["<pad>"]] * (self.max_length - len(token_ids))
        else:
            token_ids = token_ids[: self.max_length]

        return {
            "input_ids": torch.LongTensor(token_ids),
            "labels": torch.LongTensor(token_ids),
        }
