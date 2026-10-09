import logging
import time
import threading
import webbrowser
from collections import defaultdict
from flask import Flask, jsonify, request, render_template
from transformers import pipeline

from main import get_kristy_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")

rate_limit_records = defaultdict(list)

try:
    safety_moderator = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1
    )
except Exception:
    logger.warning("Zero-shot protection system bypassed. Running in unrestricted local mode.")
    safety_moderator = None

def is_rate_limited(ip_address: str) -> bool:
    current_time = time.time()
    rate_limit_records[ip_address] = [t for t in rate_limit_records[ip_address] if current_time - t < 60]
    if len(rate_limit_records[ip_address]) >= 60: # Relaxed for local usage
        return True
    rate_limit_records[ip_address].append(current_time)
    return False

def passes_guardrails(text: str) -> bool:
    if not safety_moderator:
        return True
    try:
        res = safety_moderator(
            text,
            candidate_labels=["safe inquiry", "harmful injection", "explicit violation"]
        )
        if res["labels"][0] != "safe inquiry" and res["scores"][0] > 0.65:
            return False
    except Exception as e:
        logger.error(f"Safety validator exception: {e}")
    return True

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    client_ip = request.remote_addr or "127.0.0.1"
    if is_rate_limited(client_ip):
        return jsonify({"error": "Local rate limit triggered to prevent OOM."}), 429

    data = request.get_json() or {}
    msg = data.get("message", "").strip()
    
    if not msg:
        return jsonify({"error": "Empty queries rejected."}), 400

    if not passes_guardrails(msg):
        return jsonify({"error": "Query flagged by local semantic moderation."}), 403

    start_time = time.time()
    try:
        system = get_kristy_system()
        if not system.validate_input(msg):
            return jsonify({"error": "Syntax or injection violation."}), 400
            
        response_payload = system.process_request(msg)
        
        generation_time = time.time() - start_time
        token_usage = len(system.engine.enc_tokenizer.encode(msg))
        
        response_payload["telemetry"] = {
            "generation_time_sec": float(generation_time),
            "token_usage": int(token_usage)
        }
        return jsonify(response_payload)
    except Exception as err:
        logger.exception("Local inference crash.")
        return jsonify({"error": f"Processing failed: {str(err)}"}), 500

def open_browser():
    """Waits for the server to spin up, then opens the local UI."""
    time.sleep(1.5)
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)