import streamlit as st
import streamlit.components.v1 as components
import requests
from agent_loop import run_prompt
from agent_schemas import AgentResponse
import os

st.set_page_config(page_title="THURSDAY Voice Console", page_icon="🧠")
st.title("⚛️ THURSDAY Voice Console")

TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://localhost:9000").rstrip("/")


def fetch_tts_audio(text: str) -> bytes | None:
    if not text:
        return None

    resp = requests.post(
        f"{TTS_BASE_URL}/speak",
        json={"text": text},
        timeout=180,
    )
    resp.raise_for_status()

    payload = resp.json()
    audio_url = str(payload.get("audio_url") or "").strip()
    if not audio_url:
        return None

    if audio_url.startswith("/"):
        audio_url = f"{TTS_BASE_URL}{audio_url}"

    audio_resp = requests.get(audio_url, timeout=180)
    audio_resp.raise_for_status()
    return audio_resp.content

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio_bytes" not in st.session_state:
    st.session_state.last_audio_bytes = None
if "audio_id" not in st.session_state:
    st.session_state.audio_id = 0

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])

if st.session_state.last_audio_bytes:
    audio_bytes = st.session_state.last_audio_bytes

    # Hidden, autoplaying audio element with unique id and explicit play()
    import base64
    b64 = base64.b64encode(audio_bytes).decode()

    audio_dom_id = f"tts-audio-{st.session_state.audio_id}"
    components.html(
        f"""
        <audio id="{audio_dom_id}" autoplay preload="auto">
          <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
        <script>
          const el = document.getElementById('{audio_dom_id}');
          if (el) {{
            // Try to force playback in case autoplay is flaky
            el.play().catch(() => {{ /* autoplay may be blocked */ }});
          }}
        </script>
        """,
        height=0,
    )

user_text = st.chat_input("Say something to THURSDAY...")

if user_text:
    st.session_state.messages.append({"role": "user","text": user_text})

    resp: AgentResponse = run_prompt(user_text)
    reply_text = resp.reply or ""
    speak_text = (resp.tts_text or resp.reply or "").strip()

    st.session_state.messages.append({"role": "assistant", "text": reply_text})

    try:
        audio_bytes = fetch_tts_audio(speak_text) if speak_text else None
    except requests.RequestException as exc:
        audio_bytes = None
        st.warning(f"TTS request failed: {exc}")

    if audio_bytes:
        st.session_state.last_audio_bytes = audio_bytes
        st.session_state.audio_id += 1

    st.rerun()
