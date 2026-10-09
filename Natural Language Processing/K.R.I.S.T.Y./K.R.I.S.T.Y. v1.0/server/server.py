import json
from flask import Flask, request, jsonify
from core.kristy import KristyGraphMath
from core.prepare import build_database
import os

app = Flask(__name__)

db_path = "./data/knowledge_base.db"
if not os.path.exists(db_path):
    build_database()
engine = KristyGraphMath(db_path)

@app.route("/query", methods=["POST"])
def handle_query():
    try:
        data = request.get_json(force=True)
        user_text = data.get("text", "").strip()
        if not user_text:
            return jsonify({"response": "Please provide a query."}), 400
    except Exception:
        return jsonify({"response": "Invalid JSON payload."}), 400

    reply = engine.handle_query(user_text)
    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
