from pathlib import Path
import subprocess
from datetime import datetime
OUT_DIR = Path("out/tts/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def speak_to_wav(text: str) -> str:
    # macOS built-in TTS -> generates AIFF; we’ll save that and Streamlit can still play it
    # (If you want WAV later, we’ll convert once the pipeline works.)
    out_path = OUT_DIR / "last.aiff"
    subprocess.run(["say", text, "-o", str(out_path)], check=True)
    return str(out_path)