# Usage

## 1. Overview

K.R.I.S.T.Y. (Knowledge Retrieval and Inference System for Test Yielding) v1.0 is a small Python program that answers product‑price queries.  
It stores product data in a SQLite database, builds a directed graph with NetworkX, and uses SymPy to compute total prices that include tax.  
The engine can be used through a REST API or a Tkinter desktop client.

## 2. Install Requirements

The project depends only on the Python standard library and three pure‑Python packages.

```bash
pip install -r requirements.txt
```

The `requirements.txt` files include:

* `networkx==1.11`
* `sympy==1.0`
* `nltk==3.2.1`
* (For the server) `Flask==0.12.2`

Make sure the NLTK data packages are available.  If the tokenizer fails, run:

```bash
python -m nltk.downloader punkt
```

## 3. Running the HTTP Server

The server exposes a single endpoint, `/query`, that accepts a JSON body:

```json
{ "text": "What is the total price for iPhone 7?" }
```

Start the service:

```bash
python server.py
```

The service listens on `0.0.0.0:5000`.  A successful call returns:

```json
{ "response": "The total price for iPhone 7 after 8% tax is $701.12." }
```

If the request body is missing or empty, the server returns a 400 error with an explanatory message.

## 4. Using the Command‑Line Client

The `example.py` script demonstrates how to call the server from the command line.

```bash
python example.py
```

It prompts for queries and prints the server’s replies until the user types `exit`.  
The script internally uses `urllib.request` to send POST requests to the server.

## 5. Troubleshooting

| Issue | Likely Cause | Resolution |
|---|---|---|
| **Server returns “Invalid JSON payload.”** | Body not JSON or missing `Content‑Type` header | Ensure the request body is valid JSON and the header is set to `application/json`. |
| **Server returns “Please provide a query.”** | Empty or whitespace query | Provide a non‑empty string in the `text` field. |
| **Database not found** | `prepare.py` not executed or database path incorrect | Run `python prepare.py` to create `data/knowledge_base.db`. |
| **Tkinter window does not appear** | Tkinter not installed or incompatible Python version | Install Tkinter (`sudo apt-get install python3‑tk` on Debian/Ubuntu) or use a compatible Python distribution. |
| **Tax calculation reports “No tax records found.”** | The product type is not “Smartphone” or the edge to `Tax_Rate` is missing | Check that the product type in the database is “Smartphone” and that the graph construction in `kristy.py` has not been altered. |
| **SQLite error “database is locked”** | Multiple processes access the database simultaneously | Ensure only one process (server or client) is writing to the database at a time. |
| **SymPy raises an exception** | Version mismatch or missing dependency | Confirm that SymPy 1.0 is installed and compatible with your Python version. |