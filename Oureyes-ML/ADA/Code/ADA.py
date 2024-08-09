import time
import speech_recognition as sr
import pyttsx3
import spacy
import requests
import json
import tkinter as tk
from transformers import pipeline
import bluetooth

WAKE_WORDS = ["Hey Ada", "Okay Ada", "Ada"]
DEVICE_ADDRESS = "YOUR_DEVICE_ADDRESS"

r = sr.Recognizer()

recognition_models = [
    {"name": "Google Speech Recognition", "recognizer": sr.Recognizer()},
    {"name": "Wit.ai", "recognizer": sr.Recognizer()},
    {"name": "Microsoft Bing Voice Recognition", "recognizer": sr.Recognizer()},
    {"name": "IBM Speech to Text", "recognizer": sr.Recognizer()},
]

engine = pyttsx3.init()

response_generator = pipeline("text-generation", model="gpt2")

nlp = spacy.load("en_core_web_sm")

languages = ["en_US", "zh_TW"]

socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
socket.connect((DEVICE_ADDRESS, 1))
print("Bluetooth connection established.")

def get_stock_price(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}?p={ticker}&.tsrc=fin-srch"
    response = requests.get(url)

    if response.status_code == 404:
        return "The stock ticker you entered is invalid."

    start_index = response.text.find('"regularMarketPrice"')
    end_index = response.text.find('"', start_index + len('"regularMarketPrice":'))

    return response.text[start_index + len('"regularMarketPrice":'):end_index]

def on_button_click():
    input_text = input_entry.get()
    input_text = input_text.strip()

    response = response_generator(input_text, max_length=100, temperature=0.7)[0]["generated_text"].strip()

    engine.say(response)
    engine.runAndWait()

    output_label.config(text=response)

def on_key_press(event):
    if event.keysym == "Return":
        on_button_click()

root = tk.Tk()
root.title("Ada Assistant")

input_label = tk.Label(root, text="Input:")
input_label.pack()

input_entry = tk.Entry(root, width=50)
input_entry.pack()

output_label = tk.Label(root, text="")
output_label.pack()

button = tk.Button(root, text="Submit", command=on_button_click)
button.pack()

input_entry.bind("<KeyPress>", on_key_press)

while True:
    print("Waiting for wake word...")
    with sr.Microphone() as source:
        audio = r.listen(source)
    try:
        wake_word = r.recognize_google(audio)
        if wake_word.lower() not in [word.lower() for word in WAKE_WORDS]:
            continue
        print("Wake word recognized.")
    except:
        continue

    recognition_model = recognition_models[0]["recognizer"]
    
    print("Listening...")
    with sr.Microphone() as source:
        audio = recognition_model.listen(source)

    input_text = ""
    for model in recognition_models:
        try:
            input_text = model["recognizer"].recognize_sphinx(audio)
            print(f"Using {model['name']} for recognition.")
            break
        except:
            continue

    if not input_text:
        print("Failed to recognize speech.")
        continue

    print(f"You said: {input_text}")

    doc = nlp(input_text)
    for token in doc:
        if token.pos_ == "VERB" and token.text == "stock":
            ticker = input_text.split()[-1]
            response = get_stock_price(ticker)
            engine.say(response)
            engine.runAndWait()
            break

    language = languages[0]

    engine.setProperty("voice", language)

    response = response_generator(input_text, max_length=100, temperature=0.7)[0]["generated_text"].strip()

    engine.say(response)
    engine.runAndWait()

    time.sleep(1)

root.mainloop()
