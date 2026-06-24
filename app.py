import base64
import hashlib
import os

import requests
import streamlit as st
import streamlit.components.v1 as components
from agent_loop import run_prompt
from agent_schemas import AgentResponse

st.set_page_config(page_title="THURSDAY Voice Console", page_icon="🧠")
st.title("⚛️ THURSDAY Voice Console")

TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://localhost:9000").rstrip("/")
STT_BASE_URL = os.getenv("STT_BASE_URL", "http://localhost:9200").rstrip("/")


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


def transcribe_audio(audio_bytes: bytes, content_type: str = "audio/wav") -> str | None:
    """Send recorded mic audio to the STT service and return the transcribed text."""
    if not audio_bytes:
        return None

    files = {"audio": ("input.wav", audio_bytes, content_type or "audio/wav")}
    resp = requests.post(f"{STT_BASE_URL}/transcribe", files=files, timeout=120)
    resp.raise_for_status()

    payload = resp.json()
    if not payload.get("ok"):
        return None
    return str(payload.get("text") or "").strip() or None


def handle_user_message(text: str) -> None:
    """Shared path for both typed and spoken input: log, run the agent, fetch TTS, rerun."""
    text = (text or "").strip()
    if not text:
        return

    st.session_state.messages.append({"role": "user", "text": text})

    resp: AgentResponse = run_prompt(text)
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


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio_bytes" not in st.session_state:
    st.session_state.last_audio_bytes = None
if "audio_id" not in st.session_state:
    st.session_state.audio_id = 0
if "last_stt_hash" not in st.session_state:
    st.session_state.last_stt_hash = None

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])

if st.session_state.last_audio_bytes:
    audio_bytes = st.session_state.last_audio_bytes

    # Hidden, autoplaying audio element with unique id and explicit play()
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

mic_audio = st.audio_input("🎤 Speak to THURSDAY")

if mic_audio is not None:
    raw = mic_audio.getvalue()
    digest = hashlib.md5(raw).hexdigest()

    # st.audio_input keeps returning the same recording across reruns, so only
    # transcribe it once per new recording instead of on every rerun.
    if digest != st.session_state.last_stt_hash:
        st.session_state.last_stt_hash = digest

        with st.spinner("Transcribing..."):
            try:
                transcribed = transcribe_audio(raw, mic_audio.type or "audio/wav")
            except requests.RequestException as exc:
                transcribed = None
                st.warning(f"STT request failed: {exc}")

        if transcribed:
            handle_user_message(transcribed)
        else:
            st.warning("Didn't catch that — try again.")

user_text = st.chat_input("Say something to THURSDAY...")

if user_text:
    handle_user_message(user_text)
