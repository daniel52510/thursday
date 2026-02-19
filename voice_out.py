from pathlib import Path
import subprocess

OUT_DIR = Path("out/tts")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def speak_to_wav(text: str) -> str | None:
    """Generate a browser-friendly WAV file from macOS TTS.

    Streamlit's st.audio plays via the browser <audio> element.
    Many browsers won't decode AIFF variants produced by `say`.
    WAV (linear PCM) is the most reliable target.
    """
    if not text or not text.strip():
        return None

    tmp_aiff = OUT_DIR / "last_tmp.aiff"
    out_wav = OUT_DIR / "last.wav"

    # 1) Generate AIFF with macOS TTS
    subprocess.run(["say", text, "-o", str(tmp_aiff)], check=True)

    # 2) Convert to 16-bit linear PCM WAV (browser-friendly)
    subprocess.run(
        ["afconvert", str(tmp_aiff), str(out_wav), "-f", "WAVE", "-d", "LEI16"],
        check=True,
    )

    return str(out_wav)