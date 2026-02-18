import streamlit as st
from voice_out import speak_to_wav
from pathlib import Path
from agent_loop import run_prompt

st.set_page_config(page_title="THURSDAY Voice Console", page_icon="🧠")
st.title("⚛️ THURSDAY Voice Console")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio_path" not in st.session_state:
    st.session_state.last_audio_path = None

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])

if st.session_state.last_audio_path:
    p = Path(st.session_state.last_audio_path)
    if p.exists() and p.stat().st_size > 0:
        st.audio(p.read_bytes(), format="audio/aiff")
    else:
        st.warning("Audio File Missing or Empty. Try Again.")

user_text = st.chat_input("Say something to THURSDAY...")

if user_text:
    st.session_state.messages.append({"role": "user","text": user_text})

    reply_text = run_prompt(user_text) or ""
    st.session_state.messages.append({"role": "assistant", "text": reply_text})

    # Generate audio and store path; playback happens on the next run
    st.session_state.last_audio_path = speak_to_wav(reply_text)

    st.rerun()