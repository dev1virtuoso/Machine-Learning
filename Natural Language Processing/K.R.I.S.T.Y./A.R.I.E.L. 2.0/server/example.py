import requests
import json

URL = "http://localhost:8080/ask"

def ask(text: str) -> dict:
    r = requests.post(URL, json={"text": text})
    return r.json()

if __name__ == "__main__":
    query = "What is the price of Tesla Model 3?"
    result = ask(query)
    print(json.dumps(result, indent=2, ensure_ascii=False))
