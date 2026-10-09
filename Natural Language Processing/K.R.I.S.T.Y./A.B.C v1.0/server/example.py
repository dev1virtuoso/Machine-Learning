import json
import urllib.request

SERVER_URL = "http://127.0.0.1:5000/chat"

def send_message(msg):
    payload = json.dumps({"message": msg}).encode("utf-8")
    req = urllib.request.Request(
        SERVER_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8")).get("reply", "")

def main():
    print("Chat with A.B.C. (server). Type 'exit' to quit.")
    while True:
        user_msg = input("You: ").strip()
        if user_msg.lower() == "exit":
            print("Exiting.")
            break
        reply = send_message(user_msg)
        print(f"A.B.C.: {reply}")

if __name__ == "__main__":
    main()
