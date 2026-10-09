import requests
import json

API_URL = "http://localhost:8080/ask"

def ask(text: str) -> dict:
    response = requests.post(API_URL, json={"text": text})
    response.raise_for_status()
    return response.json()
