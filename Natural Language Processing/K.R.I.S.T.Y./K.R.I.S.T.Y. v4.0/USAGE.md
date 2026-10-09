# Usage Guide

## 1. Overview

This document provides instructions for using K.R.I.S.T.Y. v4.0. The system supports two primary deployment modes: a local web interface and a headless HTTP API server. Both modes allow interaction with the multimodal engine through text prompts, returning text responses, motion data, and generated images.

## 2. Install Requirements

Create and activate a Python virtual environment, then install the dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Ensure PyTorch is compatible with your system hardware (CUDA support is recommended for better performance).

## 3. Running the HTTP Server

### Local Web Interface

Run the local desktop version which includes a browser-based UI:

```bash
python webui.py
```

The application will automatically open a browser window at http://127.0.0.1:5000/. Enter prompts in the input field and click EXECUTE to receive multimodal responses.

### Headless Production Server

For API-only deployment, start the server:

```bash
python server_api.py
```

The server will listen on http://0.0.0.0:8000/. Health check endpoint is available at `/api/health`.

## 4. Using the Command‑Line Client

Use the provided example client to interact with the running server:

```bash
python example_client.py
```

Modify the `test_prompt` variable in `example_client.py` to send custom queries. The client connects to the local server and displays text output, telemetry, motion data summary, and saves the generated image as `kristy_output_latent.jpg`.

You can also send requests directly using tools like curl:

```bash
curl -X POST http://127.0.0.1:8000/api/inference \
  -H "Content-Type: application/json" \
  -d '{"message": "Initialize forward pass sequence for module articulation."}'
```

## 5. Troubleshooting

- **Module import errors**: Verify all files (`main.py`, `module.py`) are in the same directory and the virtual environment is activated.
- **CUDA out of memory**: Reduce batch size or run on CPU by setting device in configuration.
- **Model checkpoint missing**: The system will warn on startup. Place a valid checkpoint at `./checkpoints/Kristy.pt` or run in non-strict mode.
- **Slow inference**: Ensure GPU acceleration is available. First run may take longer due to model warmup.
- **No image or motion output**: Check console logs for tensor shape mismatches or generation exceptions.
- **Port already in use**: Change the port in `webui.py` (5000) or `server_api.py` (8000).
- **Tokenizer errors**: Reinstall `transformers` package if tokenizer loading fails.

Check application logs for detailed error messages. Restart the server after modifying configuration values.