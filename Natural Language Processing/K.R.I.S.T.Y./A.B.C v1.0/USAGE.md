# Usage

## 1. Overview

The package implements a small offline chatbot that can be accessed in two ways:

1. HTTP server `server.py` exposes a `/chat` endpoint.  
2. Command‑line client `example.py` sends requests to that endpoint.

The core conversation logic resides in `core/main.py` and is shared by both interfaces.

## 2. Install Requirements

Only Flask is required for the server.

```bash
pip install -r requirements.txt
```

If you are using a virtual environment, activate it before running the above command.

## 3. Running the HTTP Server

```bash
python server.py
```

* Listens on `0.0.0.0:5000`.  
* Debug mode is off.  
* The server will respond to POST requests at `http://<host>:5000/chat`.

### 3.1. Testing the Endpoint

```bash
curl -X POST http://127.0.0.1:5000/chat \
     -H "Content-Type: application/json" \
     -d '{"message":"hello"}'
```

Expected response:

```json
{"reply":"Hi friend! How are you today?"}
```

## 4. Using the Command‑Line Client

```bash
python example.py
```

You will see:

```
Chat with A.B.C. (server). Type 'exit' to quit.
You: 
```

Type any message and press Enter. The bot replies in the same terminal.  
Type `exit` to close the client.

## 5. Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `ImportError: No module named 'Flask'` | Flask not installed | Run `pip install -r requirements.txt` or install Flask manually. |
| Server fails to start with `Address already in use` | Port 5000 is occupied | Stop the other process or change the port in `server.py` (`app.run(port=5001)`). |
| Client cannot connect (`URLError: <urlopen error [Errno 111] Connection refused>`) | Server not running or wrong address | Start the server first; ensure `SERVER_URL` in `example.py` points to the correct host and port. |
| `Invalid JSON payload.` | Malformed JSON or missing content‑type | Ensure `Content-Type: application/json` header and valid JSON body. |
| Empty reply or `Please send a non‑empty message.` | Sending an empty string or only whitespace | Provide at least one non‑whitespace character. |
| Bot returns `{name}` or `{location}` unchanged | Session context never populated | Provide a name or location (`my name is Alice`, `i live in Paris`). |
| `AttributeError: module 'core.main' has no attribute 'abc_respond'` | Incorrect package structure or missing `__init__.py` | Verify that the directory containing `main.py` is named `core` and contains an `__init__.py` file. |
| Tkinter GUI fails to start | Tkinter not installed or missing `tkinter` package | Install the Tkinter package for your platform (`sudo apt-get install python3-tk`, `brew install python-tk@3`, etc.). |

## 6. Notes

* All conversation state is stored in memory for the lifetime of the server process.  
* The bot is intentionally minimal; it does not provide authentication or rate limiting.  
* The `run_local.py` script launches a Tkinter GUI that uses the same core logic (`abc_respond`).  
* The package structure is simple; no additional configuration is required.  

Happy chatting!