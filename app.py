import streamlit as st
import tempfile
import pyttsx3
from groq import Groq
import os
from dotenv import load_dotenv
import openai

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

groq_client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="Real-Time Voice AI Agent", layout="centered")
st.title("🎤 Real-Time Voice AI Agent")

st.subheader("Upload your audio (.wav) for the AI to process:")

uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_audio_path = temp_audio.name

    transcript = groq_client.audio.transcriptions.create(
        file=open(temp_audio_path, "rb"),
        model="whisper-large-v3"
    )

    user_text = transcript.text
    st.text_area("You said:", user_text, height=150, key="user_transcript")

    system_prompt = (
        "You are a conversational AI assistant. "
        "Respond exactly as if you are speaking to a human. "
        "Do NOT use bullets, headings, markdown, code blocks, numbers, or special symbols. "
        "Write smooth, flowing paragraphs that sound natural when read aloud."
    )

    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ],
        temperature=0.7
    )

    response_text = completion.choices[0].message["content"].strip()
    st.text_area("AI Agent says:", response_text, height=250, key="ai_response")

    # Generate AI voice
    engine = pyttsx3.init()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_file:
        engine.save_to_file(response_text, audio_file.name)
        engine.runAndWait()
        st.audio(audio_file.name, format="audio/mp3")
