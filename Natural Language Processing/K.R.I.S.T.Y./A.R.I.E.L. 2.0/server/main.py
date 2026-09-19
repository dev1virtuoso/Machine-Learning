import os
import logging
from typing import List

import torch
import spacy

from ariel_arch import ArielCore
from kg_engine import KnowledgeEngine
from linguistic_engine import LinguisticEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [PROD] - %(message)s",
)
logger = logging.getLogger("ArielProd")

word_to_id = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
id_to_word = {v: k for k, v in word_to_id.items()}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ariel_v2.0.pth")

model = ArielCore(vocab_size=len(word_to_id), hidden_size=512).to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
logger.info("Model loaded from %s", MODEL_PATH)

kb = KnowledgeEngine(db_path="knowledge_base.db")

try:
    nlp = spacy.load("en_core_web_md")
except Exception:
    nlp = None
    logger.warning("spaCy model could not be loaded")

linguistic = LinguisticEngine(vocab=word_to_id)

def preprocess_text(text: str, max_len: int = 32) -> torch.LongTensor:
    tokens = text.strip().split()
    ids = [word_to_id.get(tok, word_to_id["<pad>"]) for tok in tokens]
    if len(ids) < max_len:
        ids += [word_to_id["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.LongTensor([ids]).to(DEVICE)

def postprocess_logits(logits: torch.Tensor) -> str:
    pred_ids = logits.argmax(1).tolist()
    pred_ids = [i for i in pred_ids if i != word_to_id["<pad>"]]
    return " ".join([id_to_word.get(i, "<unk>") for i in pred_ids])

def run(user_input: str) -> str:
    if not user_input:
        return "Please say something."

    entity: str | None = None
    if nlp:
        doc = nlp(user_input)
        for ent in doc.ents:
            if ent.label_ in {"PRODUCT", "ORG", "GPE"}:
                entity = ent.text.lower()
                break

    facts = kb.query_fact(entity) if entity else []
    facts_text = f"Facts: {facts}" if facts else "No facts found."

    corrected = linguistic.fix_grammar(user_input) if linguistic.nlp else user_input

    input_tensor = preprocess_text(corrected)

    with torch.no_grad():
        logits, h_signal = model(input_tensor)

    hallucination = h_signal.mean().item()

    if hallucination > 0.6:
        reply = "Sorry, I cannot answer confidently at the moment."
    else:
        decoder_input = torch.LongTensor([[word_to_id["<sos>"]]]).to(DEVICE)
        encoder_output, _ = model.encoder(model.embedding(input_tensor))
        decoder_hidden = encoder_output[:, -1, :].mean(1).unsqueeze(0)

        pred_ids: List[int] = []
        for _ in range(32):
            out, decoder_hidden = model.decoder_forward(decoder_input, decoder_hidden)
            next_token = out.argmax(1).item()
            if next_token == word_to_id["<eos>"]:
                break
            pred_ids.append(next_token)
            decoder_input = torch.LongTensor([[next_token]]).to(DEVICE)

        reply = " ".join([id_to_word.get(i, "<unk>") for i in pred_ids])

    return f"{reply} | {facts_text}"

if __name__ == "__main__":
    print("A.R.I.E.L. v2.0 ONLINE")
    print("-" * 50)
    while True:
        try:
            query = input("User: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if query.lower() in ("bye", "exit", "quit"):
            print("Bye! See you next time.")
            break

        answer = run(query)
        print(f"A.R.I.E.L.: {answer}")
