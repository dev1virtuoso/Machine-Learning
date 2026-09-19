import json
from flask import Flask, request, jsonify
from core.main import abc_respond

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """
    Expected JSON payload:
        { "message": "user text here" }

    Returns:
        { "reply": "bot response" }
    """
    try:
        data = request.get_json(force=True)
        user_msg = data.get("message", "").strip()
        if not user_msg:
            return jsonify({"reply": "Please send a non‑empty message."}), 400
    except Exception:
        return jsonify({"reply": "Invalid JSON payload."}), 400

    bot_reply = abc_respond(user_msg)
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    # Bind to all interfaces so the client can connect from other machines
    app.run(host="0.0.0.0", port=5000, debug=False)
