from pathlib import Path
import subprocess
import torch
import soundfile as sf

# Qwen TTS
from qwen_tts import Qwen3TTSModel

OUT_DIR = Path("out/tts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

_MODEL = None

def _get_model():
    global _MODEL
    if _MODEL is None:
        # Mac-friendly settings (CPU). On Linux RTX later: device_map="cuda:0", dtype=torch.bfloat16
        _MODEL = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            device_map="cpu",
            dtype=torch.float32,
        )
    return _MODEL

def speak_to_wav(text: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None

    out_wav = OUT_DIR / "last.wav"

    try:
        model = _get_model()
        wavs, sr = model.generate_custom_voice(
            text=text,
            language="English",
            speaker="Sohee",
            instruct="Native American English, slight accent. Calm, warm, confident, slightly playful. Roleplay as someone who is from Singapore.",
        )
        sf.write(str(out_wav), wavs[0], sr, subtype="PCM_16")
        return str(out_wav)

    except Exception as e:
        # Fallback: macOS TTS so THURSDAY never goes mute
        tmp_aiff = OUT_DIR / "last_tmp.aiff"
        subprocess.run(["say", text, "-o", str(tmp_aiff)], check=False)
        subprocess.run(
            ["afconvert", str(tmp_aiff), str(out_wav), "-f", "WAVE", "-d", "LEI16"],
            check=False,
        )
        return str(out_wav)