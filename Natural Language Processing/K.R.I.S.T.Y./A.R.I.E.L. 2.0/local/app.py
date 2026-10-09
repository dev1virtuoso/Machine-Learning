import os
import json
import logging
from typing import Optional

import torch
import redis
from flask import Flask, request, jsonify

from ariel_arch import ArielCore
from kg_engine import KnowledgeEngine
from linguistic_engine import LinguisticEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [APP] - %(message)s'
)
logger = logging.getLogger('ArielAPI')
app = Flask(__name__)

CACHE_DB: int = 0
REDIS_HOST: str = 'localhost'
REDIS_PORT: int = 6379
cache = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=CACHE_DB)

kb = KnowledgeEngine(db_path="knowledge_base.db")

from spacy.lang.en import English
nlp_tokenizer = English()

def build_vocab_from_tokenizer(
    tokenizer: English,
    max_vocab: int = 20000
) -> tuple[dict[str, int], dict[int, str]]:

    train_file = "train_v1.csv"
    if not os.path.exists(train_file):
        logger.warning("Training data not found – using minimal vocab.")
        return (
            {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3},
            {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        )

    counter: dict[str, int] = {}
    with open(train_file, encoding="utf-8") as f:
        for line in f:
            txt = line.strip().split(",", 1)[0]
            for tok in tokenizer(txt).text.split():
                counter[tok] = counter.get(tok, 0) + 1

    vocab: dict[str, int] = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for tok, _ in sorted(counter.items(), key=lambda x: -x[1]):
        if len(vocab) >= max_vocab:
            break
        vocab[tok] = len(vocab)

    id_to_word: dict[int, str] = {v: k for k, v in vocab.items()}
    return vocab, id_to_word

word_to_id, id_to_word = build_vocab_from_tokenizer(nlp_tokenizer)
logger.info("Vocabulary size: %d", len(word_to_id))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ariel_v2.0.pth")
model = ArielCore(vocab_size=len(word_to_id), hidden_size=512).to(DEVICE)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
logger.info("Model loaded from %s", MODEL_PATH)
linguistic = LinguisticEngine(vocab=word_to_id)

def _get_cache(key: str) -> Optional[str]:
    raw = cache.get(key)
    return raw.decode("utf-8") if raw else None

def _set_cache(key: str, value: str, ttl: int = 3600) -> None:
    cache.setex(key, ttl, value.encode("utf-8"))

def _tokenize(text: str, max_len: int = 32) -> torch.LongTensor:
    tokens = [tok.text for tok in nlp_tokenizer(text)]
    ids = [word_to_id.get(tok, word_to_id["<unk>"]) for tok in tokens]
    if len(ids) < max_len:
        ids += [word_to_id["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.LongTensor([ids]).to(DEVICE)

def _detokenize(ids: list[int]) -> str:
    filtered = [i for i in ids if i != word_to_id["<pad>"]]
    return " ".join([id_to_word.get(i, "<unk>") for i in filtered])

@app.route("/ask", methods=["POST"])
def handle_request() -> jsonify:
    data = request.get_json(silent=True) or {}
    user_query = data.get("text", "").strip()
    if not user_query:
        return jsonify({"status": "error", "message": "Empty query"}), 400

    cached = _get_cache(user_query)
    if cached:
        context_str = cached
    else:
        entities = kb.extract_entities(user_query)
        facts = kb.query_fact(entities[0]) if entities else []
        context_str = json.dumps(facts, ensure_ascii=False) if facts else "[]"
        _set_cache(user_query, context_str)

    corrected = linguistic.fix_grammar(user_query) if linguistic.nlp else user_query

    input_tensor = _tokenize(corrected)
    with torch.no_grad():
        logits, h_signal = model(input_tensor)

    hallucination = h_signal.mean().item()

    if hallucination > 0.6:
        reply = "Sorry, I cannot answer confidently at the moment."
    else:
        decoder_input = torch.LongTensor([[word_to_id["<sos>"]]]).to(DEVICE)
        encoder_output, _ = model.encoder(model.embedding(input_tensor))
        decoder_hidden = encoder_output[:, -1, :].mean(1).unsqueeze(0)

        pred_ids = []
        for _ in range(32):
            out, decoder_hidden = model.decoder_forward(
                decoder_input, decoder_hidden
            )
            next_token = out.argmax(1).item()
            if next_token == word_to_id["<eos>"]:
                break
            pred_ids.append(next_token)
            decoder_input = torch.LongTensor([[next_token]]).to(DEVICE)
        reply = _detokenize(pred_ids)

    return jsonify(
        {
            "status": "success",
            "ariel_output": f"{reply} | Context: {context_str}",
            "hallucination_risk": round(hallucination, 4),
        }
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)