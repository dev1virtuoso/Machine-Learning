import os
import logging
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from ariel_arch import ArielCore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [TRAIN] - %(message)s",
)
logger = logging.getLogger("Train")

BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "ariel_v2.0.pth"

def build_vocab_from_train(
    csv_path: str = "train_v1.csv", max_vocab: int = 20000
) -> Tuple[Dict[str, int], Dict[int, str]]:
    if not os.path.exists(csv_path):
        logger.error("Training file not found.")
        return (
            {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3},
            {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"},
        )
    counter: Dict[str, int] = {}
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            txt = line.strip().split(",", 1)[0]
            for tok in txt.split():
                counter[tok] = counter.get(tok, 0) + 1
    vocab: Dict[str, int] = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for tok, _ in sorted(counter.items(), key=lambda x: -x[1]):
        if len(vocab) >= max_vocab:
            break
        vocab[tok] = len(vocab)
    id_to_word = {v: k for k, v in vocab.items()}
    return vocab, id_to_word

def load_dataset(csv_path: str = "train_v1.csv") -> Tuple[list[str], list[int]]:
    df = pd.read_csv(csv_path, dtype={"text": str, "label": int})
    return df["text"].tolist(), df["label"].tolist()

def train() -> None:
    vocab, _ = build_vocab_from_train()
    word_to_id = vocab
    texts, labels = load_dataset()

    if len(texts) < 100:
        logger.warning("Training data is very limited. Consider adding more samples.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ArielCore(vocab_size=len(word_to_id), hidden_size=512).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_gate = nn.BCEWithLogitsLoss()
    criterion_gen = nn.CrossEntropyLoss(ignore_index=word_to_id["<pad>"])

    logger.info("Training on %d samples", len(texts))

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for i in range(0, len(texts), BATCH_SIZE):
            batch_text = texts[i : i + BATCH_SIZE]
            batch_label = labels[i : i + BATCH_SIZE]

            batch_ids = []
            for txt in batch_text:
                ids = [word_to_id.get(tok, word_to_id["<pad>"]) for tok in txt.split()]
                if len(ids) < 32:
                    ids += [word_to_id["<pad>"]] * (32 - len(ids))
                else:
                    ids = ids[:32]
                batch_ids.append(ids)

            input_tensor = torch.LongTensor(batch_ids).to(device)

            target_tensor = torch.cat(
                [
                    torch.full((input_tensor.size(0), 1), word_to_id["<sos>"], device=device),
                    input_tensor[:, :-1],
                ],
                dim=1,
            )

            logits, h_signal = model(input_tensor)

            label_tensor = torch.FloatTensor(batch_label).unsqueeze(1).to(device)
            loss_gate = criterion_gate(h_signal, label_tensor)

            loss_gen = criterion_gen(
                logits.view(-1, logits.size(-1)), target_tensor.view(-1)
            )

            loss = loss_gate + loss_gen

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        logger.info(
            "Epoch %d/%d | Loss: %.4f",
            epoch,
            EPOCHS,
            epoch_loss / (len(texts) / BATCH_SIZE),
        )

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info("Model saved to %s", MODEL_SAVE_PATH)

if __name__ == "__main__":
    train()
