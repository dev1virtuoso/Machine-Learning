# Usage

## 1. Overview  

A.R.I.E.L. is a knowledge‑based question‑answering system that combines:  
* a lightweight SQLite knowledge base,  
* a neural encoder‑decoder model (GRU based) for generating responses,  
* a grammar correction module, and  
* a Redis cache for fast repeated queries.  
The API is exposed through a Flask endpoint and a simple command‑line client is also provided.  

## 2. Install Requirements  

```bash
pip install -r requirements.txt
```  

The required packages are:
* `torch==1.4.0`
* `spacy==2.0.18`
* `flask==0.12.2`
* `redis==2.10.6`
* `pandas==0.25.1`
* `celery==4.2.0`

If the spaCy language model is not present, run:

```bash
python -m spacy download en_core_web_md
```

## 3. Running the HTTP Server  

The Flask server is started by executing:

```bash
python app.py
```

The server listens on `0.0.0.0:8080`.  
Send a JSON payload to `/ask`:

```json
{ "text": "What is the price of Tesla Model 3?" }
```

The response format is:

```json
{
  "status": "success",
  "ariel_output": "<reply> | Context: [...facts...]",
  "hallucination_risk": 0.12
}
```

If the request body is empty or missing the `text` key, a 400 error is returned.

## 4. Using the Command‑Line Client  

The `client.py` module demonstrates how to call the API from the terminal:

```bash
python client.py
```

Alternatively, run the `example.py` script, which sends a single query and prints the JSON result:

```bash
python example.py
```

Both clients use `requests` to POST to `http://localhost:8080/ask`.

## 5. Troubleshooting  

| Issue | Possible Cause | Fix |
|---|---|---|
| Server crashes with `ImportError: cannot import name 'ArielCore'` | The module path is incorrect or the file is missing | Verify that `ariel_arch.py` is in the same directory as `app.py` and that the import statement matches the file name. |
| Redis connection error | Redis server not running or wrong host/port | Start Redis (`redis-server`) or adjust `REDIS_HOST`/`REDIS_PORT` in `app.py`. |
| Model file not found | `ariel_v2.0.pth` is missing | Train the model with `python train.py` or copy an existing checkpoint. |
| Spacy model not loaded | spaCy language pack missing | Install with `python -m spacy download en_core_web_md`. |
| Empty query response | The user input contains no recognized entities | Ensure the knowledge base contains matching entity names; use `prepare.py` to create the database from a CSV. |
| High hallucination risk (>0.6) | The model is uncertain | Retrain with more data or adjust the threshold. |
| Database locked | Multiple processes access the same SQLite file | Run only one process that writes to the DB at a time, or use a proper RDBMS. |
| Timeout or slow response | Large vocabulary or insufficient GPU | Reduce `max_vocab` in `build_vocab_from_tokenizer` or run on CPU if GPU is unavailable. |