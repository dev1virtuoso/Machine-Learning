# Usage

## 1. Overview

The project implements a small retrieval‑augmented language model that can be interacted with in two ways:

1. **HTTP server** (`api.py`) exposes an `/infer` endpoint that accepts a JSON payload with a question and optional context.  
2. **Command‑line client** (`example.py`) sends requests to that endpoint and prints the response.

Both interfaces share the same core model defined in `kristy_arch.py` and the language‑processing utilities in `linguistic_engine.py`.

## 2. Install Requirements

The project relies on the following libraries. Install them with:

```bash
pip install -r requirements.txt
```

If you are using a virtual environment, activate it before running the command above.  
`requirements.txt` contains pinned versions for reproducibility.

## 3. Running the HTTP Server

```bash
python api.py
```

* The FastAPI application will start on `http://0.0.0.0:8000`.  
* The `/infer` endpoint expects a JSON body:

```json
{
  "question": "What is the capital of France?",
  "context": "Optional context text that may help answer the question."
}
```

* If `context` is omitted, the server will automatically retrieve the top‑k relevant passages from the BM25 index (`index.jsonl`).

### 3.1. Testing the Endpoint

```bash
curl -X POST http://127.0.0.1:8000/infer \
     -H "Content-Type: application/json" \
     -d '{"question":"Explain the fiscal results found in the latest JSON reports."}'
```

Expected response:

```json
{
  "answer": "The fiscal results show a 12% increase in revenue...",
  "logic_score": 0.12,
  "risk_flag": false
}
```

If `logic_score` exceeds the threshold defined in `KristyConfig`, the `answer` field will contain a suppression notice.

## 4. Using the Command‑Line Client

```bash
python example.py
```

The client will prompt you for a question, send it to the server, and display the following:

```
--- K.R.I.S.T.Y. RESPONSE ---
Logic score : 0.120
Risk flag   : False

Answer:
The fiscal results show a 12% increase in revenue...
```

You can exit the client by pressing `Ctrl+C` or typing `exit` when prompted for a question.

## 5. Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ImportError: No module named 'fastapi'` | FastAPI not installed | Run `pip install fastapi[all]` or `pip install -r requirements.txt` |
| Server starts but requests return `404 Not Found` | Wrong port or path | Ensure you are hitting `http://<host>:8000/infer` and that the server is running on port 8000 |
| `RuntimeError: BM25 index not found: index.jsonl` | Index file missing or mis‑named | Place a BM25 index file named `index.jsonl` in the same directory or set `BM25_INDEX` env variable |
| `RuntimeError: spaCy model load error` | `en_core_web_md` not downloaded | Run `python -m spacy download en_core_web_md` |
| `CUDA error: device-side assert` | Model weights mismatch | Verify that `kristy_v3.0.pth` matches the architecture in `kristy_arch.py`. Re‑train if necessary |
| `HTTP 500 Internal Server Error` | Unexpected exception in `infer` | Check the server logs for stack trace. Common issues include missing context, malformed JSON, or out‑of‑memory on GPU |

If a problem persists, consult the detailed log output printed by the server or the client. The stack traces typically point to the exact line where the error occurred.