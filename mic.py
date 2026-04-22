import io
import queue
import pyaudio
import threading
import re
import os
import tempfile
import numpy as np
import torch
#from TTS.api import TTS
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import speech_recognition as sr
import pyttsx3
import click
import run
import soundfile as sf  # Add soundfile to write audio data to .wav files
from kokoro_lib import KokoroTTSAgent
import random

#tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC_ph", progress_bar=False)



tts_agent = KokoroTTSAgent()

probable_hotwords = ""
list_of_hotwords = probable_hotwords.split()



def sanitize_response(response):
    """Replaces words based on the replacements dictionary and removes non-alphanumeric characters."""
    # Remove non-alphanumeric characters except spaces
    response = re.sub(r"[^\w\s]", "", response)

    # Replace words efficiently
    return response

# Configure model and audio settings
model_size = "large-v3"
sample_rate = 16000  # Sample rate expected by faster-whisper

# Initialize faster-whisper model on GPU
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# Initialize audio queue and result queue
audio_queue = queue.Queue()
result_queue = queue.Queue()


i=0

run.initVar()

@click.command()
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True, type=bool)
def main(verbose):
    threading.Thread(target=record_audio, args=(audio_queue,), daemon=True).start()
    threading.Thread(target=transcribe_forever, args=(audio_queue, result_queue, model, verbose), daemon=True).start()

    while True:
        search = random.randint(0, 0) <= 0
        if search:
            print("Search enabled for this turn")

        user_spoke = False
        transcribed_text = ""

        try:
            transcribed_text = result_queue.get(timeout=15)
            print("Raw transcription:", transcribed_text)

            # Skip hotword-only captures
            if is_just_combination(transcribed_text.strip(), list_of_hotwords):
                print("Only hotwords detected, skipping...")
                continue

            # Skip empty transcriptions
            if not transcribed_text.strip():
                print("Empty transcription, skipping...")
                continue

            user_spoke = True
            print("User said:", transcribed_text)

        except queue.Empty:
            print("25s of silence detected, generating autonomous message...")
            # user_spoke stays False

        try:
            if user_spoke:
                # Normal reply — use conversation history
                response = run.ask_llm_full(transcribed_text)

            elif search:
                # Silence + search enabled — find something interesting
                print("Using search for autonomous message...")
                response = run.chat_with_search(
                    "Find an interesting, surprising, nerdy fact or recent news topic "
                    "to bring up in conversation. Keep it very short, one sentence max."
                )

            else:
                # Silence, no search — use a direct system-level prompt
                # We call ask_llm_full with a short trigger, NOT the meta-prompt
                response = run.ask_llm_full(
                    "[system: say something short and interesting, change the subject]"
                )

        except Exception as e:
            print(f"LLM error: {e}")
            import traceback
            traceback.print_exc()
            continue

        if not response or not response.strip():
            print("Empty response, skipping TTS...")
            continue

        print("LLM response:", response)
        sanitized_response = sanitize_response(response)
        print("Sanitized response:", sanitized_response)
        tts_agent.synthesize_text(sanitized_response, play_direct=True)
        print("⏳ Attente de la fin de la lecture streaming...")
        tts_agent.wait_for_audio_completion()

        
        

def record_audio(audio_queue):
    r = sr.Recognizer()
    r.energy_threshold = 500
    r.pause_threshold = 0.8
    r.dynamic_energy_threshold = False

    print("Recorder thread started")

    with sr.Microphone(sample_rate=sample_rate) as source:
        print("Listening...")
        while True:
            print("Waiting for speech...")
            audio = r.listen(source)
            print("Speech captured")
            torch_audio = torch.from_numpy(
                np.frombuffer(audio.get_raw_data(), np.int16)
                .astype(np.float32) / 32768.0
            )
            audio_queue.put_nowait(torch_audio)
            print("Audio pushed to queue")

def transcribe_forever(audio_queue, result_queue, model, verbose):
    print("Transcriber thread started")

    while True:
        print("Waiting for audio from queue...")
        def is_silent(audio_tensor, threshold=0.01):
            return audio_tensor.abs().max().item() < threshold

        # In transcribe_forever, after receiving audio:
        audio_data = audio_queue.get()

        # Duration filter: skip if less than 0.8 seconds of audio
        min_samples = int(sample_rate * 0.8)
        if len(audio_data) < min_samples:
            print(f"Audio too short ({len(audio_data)/sample_rate:.2f}s), skipping...")
            continue

        # Silence filter: RMS energy check (more robust than peak)
        rms = audio_data.pow(2).mean().sqrt().item()
        if rms < 0.02:
            print(f"Silent audio (RMS={rms:.4f}), skipping...")
            continue
        # Save audio data to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            audio_np = audio_data.numpy()
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(-1, 1)
            sf.write(temp_audio_file.name, audio_np, sample_rate, subtype='PCM_16')
            temp_audio_file_path = temp_audio_file.name
        
        print("Starting Whisper decode...")

        segments, info = model.transcribe(
            temp_audio_file_path,
            language="en",
            hotwords=probable_hotwords,
            beam_size=5,
            temperature=0.0
        )

        print("Whisper returned generator")

        try:
            segments = list(segments)
            print("Segments materialized:", len(segments))
        except Exception as e:
            print(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()
            result_queue.put_nowait("")
            continue

        print("Segments materialized:", len(segments))

        transcription = " ".join(segment.text for segment in segments)

        print("Transcription complete")
        os.remove(temp_audio_file_path)

        if verbose:
            print("Detailed Transcription:", transcription)
        
        result_queue.put_nowait(transcription)




import re

def is_just_combination(text: str, elements: list[str]) -> bool:
    """
    Returns True if the text is just a combination of elements from the list,
    separated by spaces, commas, or similar separators.
    """
    # Escape elements for regex and sort by length (longest first to avoid partial matches)
    escaped = [re.escape(e) for e in sorted(elements, key=len, reverse=True)]
    
    # Pattern: optional separator between elements
    separator = r'[\s,;/|]+'
    element_pattern = f'(?:{"|".join(escaped)})'
    full_pattern = rf'^\s*{element_pattern}(?:{separator}{element_pattern})*\s*$'
    
    return bool(re.match(full_pattern, text, re.IGNORECASE))


def is_only_combination(text: str, elements: list[str]) -> bool:
    """
    Returns True if the text adds nothing beyond combinations of elements.
    """
    return is_just_combination(text, elements)

if __name__ == "__main__":
    main()
