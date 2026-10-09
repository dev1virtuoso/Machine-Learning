import json
import logging
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer
from kristy_arch import KristyEngine, KristyConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TRAIN] - %(message)s",
)

class KristyDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: BertTokenizer, max_len: int = 128):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer.encode_plus(
            sample["text"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_ids = torch.tensor(
            sample["label_ids"][: self.max_len], dtype=torch.long
        )
        padded = torch.full((self.max_len,), self.tokenizer.pad_token_id, dtype=torch.long)
        padded[: len(target_ids)] = target_ids

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": padded,
            "logic_label": torch.tensor([sample["logic_score"]], dtype=torch.float),
        }

def run_train():
    cfg = KristyConfig()
    tokenizer = BertTokenizer.from_pretrained(cfg.model_name)

    with open("train.jsonl", "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]
    split = int(0.8 * len(all_data))
    train_data = all_data[:split]
    val_data = all_data[split:]

    with open("train.tmp.jsonl", "w", encoding="utf-8") as f:
        for d in train_data:
            f.write(json.dumps(d) + "\n")
    with open("val.tmp.jsonl", "w", encoding="utf-8") as f:
        for d in val_data:
            f.write(json.dumps(d) + "\n")

    train_dataset = KristyDataset("train.tmp.jsonl", tokenizer, cfg.max_seq_length)
    val_dataset = KristyDataset("val.tmp.jsonl", tokenizer, cfg.max_seq_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KristyEngine(cfg).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    criterion_gen = torch.nn.CrossEntropyLoss(ignore_index=cfg.pad_token_id)
    criterion_logic = torch.nn.BCELoss()

    best_val_loss = float("inf")
    patience = 0

    for epoch in range(cfg.train_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logic_targets = batch["logic_label"].to(device)

            logits, logic_score = model(input_ids, attention_mask=mask)

            loss_gen = criterion_gen(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
            loss_logic = criterion_logic(logic_score.squeeze(), logic_targets.squeeze())

            loss = loss_gen + loss_logic
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        logging.info(f"[Epoch {epoch+1}/{cfg.train_epochs}] train loss={avg_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logic_targets = batch["logic_label"].to(device)

                logits, logic_score = model(input_ids, attention_mask=mask)

                loss_gen = criterion_gen(
                    logits.view(-1, logits.size(-1)), labels.view(-1)
                )
                loss_logic = criterion_logic(logic_score.squeeze(), logic_targets.squeeze())

                loss = loss_gen + loss_logic
                val_loss += loss.item()
        val_loss /= len(val_loader)
        logging.info(f"[Epoch {epoch+1}/{cfg.train_epochs}] val loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "kristy_v3.0.pth")
            patience = 0
            logging.info("[+] Best model checkpoint saved.")
        else:
            patience += 1
            if patience >= 2:
                logging.info("[!] Early stopping triggered.")
                break

if __name__ == "__main__":
    run_train()
