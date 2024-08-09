import socket
from microbit import display
from microbit import speech
from microbit import Image
import openai
import time
from gtts import gTTS
import os
import bluetooth
import snowboy.snowboydecoder as snowboydecoder

def wake_word_callback():
    print("Kristy detected. How can I help you?")

detector = snowboydecoder.HotwordDetector("kristy.pmdl", sensitivity=0.5)
detector.start(detected_callback=wake_word_callback, interrupt_check=None, sleep_time=0.03)

openai.api_key = "YOUR_OPENAI_API_KEY"

def generate_response(prompt):
    model_engine = "davinci"
    prompt_text = prompt
    max_tokens = 60
    temperature = 0.7
    n = 1
    stop = None

    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt_text,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        stop=stop
    )

    ai_response = response.choices[0].text.strip()

    return ai_response

def get_audio():
    display.show(Image.YES)
    speech.on()
    input_text = speech.recognize()
    speech.off()
    display.clear()
    return input_text

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    os.system("mpg123 response.mp3")

def measure_distance():

    return 0

def main():
  
    devices = bluetooth.discover_devices()

    device_address = None

    for d in devices:
        if "My MacBook" in bluetooth.lookup_name(d):
            device_address = d
            break

    if device_address:
        socket.connect((device_address, 1))
        print("Bluetooth connection established.")
    else:
        print("Could not find device to connect to.")

if __name__ == "__main__":
    main()

while True:
    input_text = get_audio()
    if input_text:
        response = generate_response(input_text)
        speak(response)
        time.sleep(1)
