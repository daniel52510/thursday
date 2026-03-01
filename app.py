import streamlit as st
import streamlit.components.v1 as components
from voice_out import speak_to_wav
from pathlib import Path
from agent_loop import run_prompt
from agent_schemas import AgentResponse

st.set_page_config(page_title="THURSDAY Voice Console", page_icon="🧠")
st.title("⚛️ THURSDAY Voice Console")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio_path" not in st.session_state:
    st.session_state.last_audio_path = None
if "audio_id" not in st.session_state:
    st.session_state.audio_id = 0

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])

if st.session_state.last_audio_path:
    p = Path(st.session_state.last_audio_path)
    if p.exists() and p.stat().st_size > 0:
        audio_bytes = p.read_bytes()

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
    else:
        st.warning("Audio file missing or empty. Try again.")

user_text = st.chat_input("Say something to THURSDAY...")

if user_text:
    st.session_state.messages.append({"role": "user","text": user_text})

    resp: AgentResponse = run_prompt(user_text)
    reply_text = resp.reply or ""
    speak_text = (resp.tts_text or resp.reply or "").strip()

    st.session_state.messages.append({"role": "assistant", "text": reply_text})

    audio_path = speak_to_wav(speak_text) if speak_text else None
    if audio_path:
        st.session_state.last_audio_path = audio_path
        st.session_state.audio_id += 1

    st.rerun()