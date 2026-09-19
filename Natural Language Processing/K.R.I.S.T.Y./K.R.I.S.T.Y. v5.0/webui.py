import logging
import time
import threading
import sqlite3
import os
import re
from flask import Flask, jsonify, request, render_template
from main import get_kristy_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder=".")
DB_PATH = "./checkpoints/kristy_webui_security.db"

class ThreadSafeSecurityDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self.lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS security_rate_limits (
                    ip_address TEXT, timestamp REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, prompt TEXT, verdict TEXT, timestamp REAL
                )
            """)
            conn.commit()

    def enforce_rate_limit(self, ip: str) -> bool:
        now = time.time()
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM security_rate_limits WHERE timestamp < ?", (now - 60.0,))
            cursor.execute("SELECT COUNT(*) FROM security_rate_limits WHERE ip_address = ?", (ip,))
            if cursor.fetchone()[0] >= 30:
                return True
            cursor.execute("INSERT INTO security_rate_limits VALUES (?, ?)", (ip, now))
            conn.commit()
            return False

    def log_violation(self, text: str, decision: str):
        with self.lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO audit_logs (prompt, verdict, timestamp) VALUES (?, ?, ?)", (text, decision, time.time()))
            conn.commit()

security_db = ThreadSafeSecurityDB(DB_PATH)

def passes_guardrails(text: str) -> bool:
    unsafe_patterns = [
        r"(?i)(override|bypass|ignore)\s+.*(safety|protocol|guardrail)",
        r"(?i)(kinetic|weaponize|destroy|attack)",
        r"(?i)(drop|delete|insert|select)\s+.*(table|database|from|sys)"
    ]
    for pattern in unsafe_patterns:
        if re.search(pattern, text):
            security_db.log_violation(text, "BLOCKED_BY_REGEX_AUDIT")
            return False
    return True

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/v5/inference", methods=["POST"])
def inference_pipeline():
    ip = request.remote_addr or "127.0.0.1"
    if security_db.enforce_rate_limit(ip):
        return jsonify({"error": "Exceeded transactional boundary limits. Restricted access active."}), 429
        
    data = request.get_json() or {}
    msg = data.get("message", "").strip()
    
    if not msg:
        return jsonify({"error": "Empty input strings rejected."}), 400
        
    if not passes_guardrails(msg):
        return jsonify({"error": "Input string contains unsafe physical operational vectors."}), 403
        
    try:
        system = get_kristy_system()
        response_payload = system.process_request(msg)
        return jsonify(response_payload)
    except Exception as err:
        logger.exception("Operational execution failure inside core orchestrator.")
        return jsonify({"error": f"Internal system transaction processing aborted: {str(err)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)