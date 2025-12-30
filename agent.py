import sounddevice as sd
import numpy as np
import tempfile
from scipy.io.wavfile import write
from groq import Groq
import pyttsx3
import os
import openai
import re
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

groq_client = Groq(api_key=GROQ_API_KEY)

def record_audio(duration=4, sample_rate=16000):
    print("🎤 Speak now...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    audio = np.int16(audio * 32767)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_file.name, sample_rate, audio)
    print("🎤 Recording saved!")
    return temp_file.name

def speech_to_text(audio_file):
    with open(audio_file, "rb") as af:
        transcript = groq_client.audio.transcriptions.create(
            file=af,
            model="whisper-large-v3"
        )
    print("📝 You said (raw):", transcript.text)
    return transcript.text

def clean_transcript(text):
    text = re.sub(r"[*_`#>-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def generate_response(user_text):
    system_prompt = (
        "You are a conversational AI assistant. "
        "Respond exactly as if you are speaking to a human. "
        "Do NOT use bullets, headings, markdown, code blocks, numbers, or special symbols. "
        "Write smooth, flowing paragraphs that sound natural when read aloud."
    )

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ],
        temperature=0.7
    )

    response = completion.choices[0].message.content.strip()
    print("\n🤖 Agent (clean paragraph):", response)
    return response

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    print("🚀 Voice AI Agent Ready!")
    while True:
        audio_path = record_audio()
        user_text = speech_to_text(audio_path)
        user_text = clean_transcript(user_text)

        response = generate_response(user_text)
        speak(response)

        print("\n--- Next Interaction ---\n")
