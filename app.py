import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile
import pyttsx3
from groq import Groq
import os
from dotenv import load_dotenv

# ===== Load API keys =====
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ===== Groq client =====
groq_client = Groq(api_key=GROQ_API_KEY)

# ===== Streamlit Page Settings =====
st.set_page_config(page_title="Real-Time Voice AI Agent", layout="centered")
st.title("🎤 Real-Time Voice AI Agent")

# ====== RECORD AUDIO FUNCTION ======
def record_audio(duration=4, sample_rate=16000):
    st.info("🎤 Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    audio = np.int16(audio * 32767)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_file.name, sample_rate, audio)
    st.success("🎤 Recording complete!")
    return temp_file.name

# ====== SPEECH TO TEXT ======
def speech_to_text(audio_file):
    with open(audio_file, "rb") as af:
        transcript = groq_client.audio.transcriptions.create(
            file=af,
            model="whisper-large-v3"
        )
    return transcript.text

# ====== GENERATE RESPONSE ======
def generate_response(text):
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": text}],
        model="llama-3.3-70b-versatile"
    )
    return chat_completion.choices[0].message.content

# ====== TEXT TO SPEECH ======
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ====== Streamlit UI ======
st.subheader("Press the button to record your question:")

if st.button("🎙️ Record & Send"):
    audio_file = record_audio()
    user_text = speech_to_text(audio_file)
    
    # Updated text area sizes for better GUI
    st.text_area("You said:", user_text, height=150, max_chars=None, key="user_transcript")
    
    response_text = generate_response(user_text)
    st.text_area("AI Agent says:", response_text, height=250, max_chars=None, key="ai_response")
    
    # Speak response
    speak(response_text)
