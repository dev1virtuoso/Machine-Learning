import json
import urllib.request

SERVER_URL = "http://127.0.0.1:5000/query"

def send_query(text):
    payload = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        SERVER_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "")

def main():
    print("K.R.I.S.T.Y. v1.0 – Server demo. Type 'exit' to quit.")
    while True:
        q = input("You: ").strip()
        if q.lower() == "exit":
            print("Bye.")
            break
        r = send_query(q)
        print(f"K.R.I.S.T.Y.: {r}")

if __name__ == "__main__":
    main()
