import logging
import time
from collections import defaultdict
from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import pipeline

from main import get_kristy_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

rate_limit_records = defaultdict(list)

try:
    safety_moderator = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1
    )
except Exception:
    logger.warning("Zero-shot protection system dropped. Network pipeline failure.")
    safety_moderator = None

def is_rate_limited(ip_address: str) -> bool:
    current_time = time.time()
    rate_limit_records[ip_address] = [t for t in rate_limit_records[ip_address] if current_time - t < 60]
    if len(rate_limit_records[ip_address]) >= 30:
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
        pass
    return True

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "active", "version": "4.0.4-headless-production"}), 200

@app.route("/api/inference", methods=["POST"])
def inference():
    client_ip = request.remote_addr or "unknown"
    if is_rate_limited(client_ip):
        return jsonify({"error": "Too many requests. Rate limit triggered."}), 429

    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Malformed request. 'message' key required."}), 400

    msg = data.get("message", "").strip()
    if not msg:
        return jsonify({"error": "Empty queries rejected by gateway standard."}), 400

    if not passes_guardrails(msg):
        return jsonify({"error": "Query flagged by semantic moderation pipeline. Rejected."}), 403

    start_time = time.time()
    try:
        system = get_kristy_system()
        if not system.validate_input(msg):
            return jsonify({"error": "Command parameter syntax formatting exception violation."}), 400
            
        response_payload = system.process_request(msg)
        
        generation_time = time.time() - start_time
        response_payload["telemetry"] = {
            "generation_time_sec": float(generation_time),
            "client_ip": client_ip
        }
        
        return jsonify(response_payload), 200
    except Exception as err:
        logger.exception("Inference call initialization dropped at API layer.")
        return jsonify({"error": f"Gateway cluster processing failed: {str(err)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)