import requests
import json

API_URL = "http://localhost:8000/infer"

def main():
    question = input("Enter your question: ").strip()
    if not question:
        print("Empty question – aborting.")
        return

    payload = {"question": question}
    resp = requests.post(API_URL, json=payload)

    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        return

    data = resp.json()
    print("\n--- K.R.I.S.T.Y. RESPONSE ---")
    print(f"Logic score : {data['logic_score']:.3f}")
    print(f"Risk flag   : {data['risk_flag']}")
    print("\nAnswer:")
    print(data["answer"])

if __name__ == "__main__":
    main()
