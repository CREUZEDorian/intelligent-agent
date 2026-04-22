import pytchat
from openai import OpenAI
import json
from pytchat import LiveChat, SpeedCalculator
import time
import requests
from pydub import AudioSegment
from pydub.playback import play
import io
import pyttsx3
import sys
import argparse
import ollama
import random
import os


summarized_context = ""


client = OpenAI()
def initTTS():
    global engine

    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    engine.setProperty('volume', 1)
    voice = engine.getProperty('voices')
    engine.setProperty('voice', voice[1].id)

try:
        with open("config.json", "r") as json_file:
            data = json.load(json_file)
except:
        print("Unable to open JSON file.")
        exit()

def initVar():
    global EL_key
    global OAI_key
    global EL_voice
    global video_id
    global tts_type
    global OAI
    global EL


    

    class OAI:
        key = data["keys"][0]["OAI_key"]
        model = data["OAI_data"][0]["model"]
        prompt = data["OAI_data"][0]["stable3"]
        temperature = data["OAI_data"][0]["temperature"]
        max_tokens = data["OAI_data"][0]["max_tokens"]
        top_p = data["OAI_data"][0]["top_p"]
        frequency_penalty = data["OAI_data"][0]["frequency_penalty"]
        presence_penalty = data["OAI_data"][0]["presence_penalty"]

    class EL:
        key = data["keys"][0]["EL_key"]
        voice = data["EL_data"][0]["voice"]

    tts_list = ["pyttsx3", "EL"]

    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--video_id", type=str)
    parser.add_argument("-tts", "--tts_type", default="EL", choices=tts_list, type=str)

    args = parser.parse_args()

    video_id = args.video_id
    tts_type = args.tts_type

    if tts_type == "pyttsx3":
        initTTS()


def Controller_TTS(message):
    if tts_type == "EL":
        EL_TTS(message)
    elif tts_type == "pyttsx3":
        pyttsx3_TTS(message)

def ask_llm_full(message):

    initVar()
    
    payload = {
        "model": "qwen3:14b",
        "message": message,
        "messages": [
            {"role": "system", "content": OAI.prompt},
            {"role": "user", "content": "Mika:" + message}
        ]
    }
    
    res = requests.post("http://192.168.1.21:18888/chat", json=payload)
    res.raise_for_status()

    return res.json()["reply"]

def chat_with_search(message: str, use_tools: bool = True) -> str:
    print(f"[chat_with_search] called, use_tools={use_tools}")  # ← add this
    payload = {
        "model": "qwen3:14b",
        "messages": [
            {"role": "system", "content": OAI.prompt + " You have access to a web search tool."},
            {"role": "user", "content": message}
        ],
        "use_tools": use_tools
    }
    response = requests.post("http://192.168.1.21:18888/chat", json=payload)
    response.raise_for_status()
    return response.json()["reply"]



def pyttsx3_TTS(message):

    initTTS()
    engine.say(message)
    engine.runAndWait()

def openai_voice(response):
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=response,
    )


def EL_TTS(message):

    url = f'https://api.elevenlabs.io/v1/text-to-speech/{EL.voice}'
    headers = {
        'accept': 'audio/mpeg',
        'xi-api-key': EL.key,
        'Content-Type': 'application/json'
    }
    data = {
        'text': message,
        'voice_settings': {
            'stability': 0.75,
            'similarity_boost': 0.75
        }
    }

    response = requests.post(url, headers=headers, json=data, stream=True)
    audio_content = AudioSegment.from_file(io.BytesIO(response.content), format="mp3")
    play(audio_content)


def read_chat():

    chat = pytchat.create(video_id=video_id)
    schat = pytchat.create(video_id=video_id, processor = SpeedCalculator(capacity = 20))

    while chat.is_alive():
        for c in chat.get().sync_items():
            print(f"\n{c.datetime} [{c.author.name}]- {c.message}\n")
            message = "chat: " + c.author.name +": " +c.message

            response = llm(message)
            print(response)
            Controller_TTS(response)

            if schat.get() >= 20:
                chat.terminate()
                schat.terminate()
                return


            time.sleep(1)




def llm(message):
    global change_prob
    global current_mood_number

    if random.randint(0, 100) >= 50:
        short="You make very short and very simple sentences."
    else:
        short=""

    if random.randint(0, 100) < change_prob:
        current_mood_number = random.randint(0, 38)
    else:
        change_prob += 10

    
    
    max_tokens=random.randint(35, 150)
    top_log = random.randint(0, 5)
    
    response = client.chat.completions.create(
      model= "gpt-3.5-turbo",
      #prompt= message,
      top_p=0.5,
      presence_penalty=1.7,
      top_logprobs = top_log,
      logprobs=True,
      messages=[
        {"role": "system", "content": OAI.prompt + short},
        {"role": "user", "content": message}
      ]
    )
    

    print("mood " + str(current_mood_number))
    return(response.choices[0].message.content)




def test_llm(model_name):
    folder = os.path.join(os.getcwd(), rf"test_prompt")
    questions = open("quesion_test.txt", "r", encoding="utf-8").read().splitlines()

    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, rf"{model_name}shorttext_moderatehistoric_stable3_additioninst_2.txt")
    

    for message in questions:
        r = ask_llm_full(message)

        if os.path.exists(file_path):
            with open(file_path, "a", encoding="utf-8") as f:
                f.write("additional instruction: " + "\n\n")
                f.write(message + "\n\n")
                f.write(r + "\n\n\n")
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("additional instruction: " + "\n\n")
                f.write(message + "\n\n")
                f.write(r + "\n\n\n")



#test_llm("qwen3_14b")