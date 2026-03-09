from pathlib import Path
import subprocess
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from qwen_tts import Qwen3TTSModel
# from pydantic import BaseModel, Field
from pydantic import BaseModel, Field
import os
from uuid import uuid4

# Optional offline mode for local/cached Hugging Face assets.
#os.environ.setdefault("HF_HUB_OFFLINE", "1")
#os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
#os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
TTS_DEVICE = os.getenv("TTS_DEVICE", "cpu")
TTS_DTYPE = os.getenv("TTS_DTYPE", "float32")
_MODEL = None
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_LANGUAGE = "English"
DEFAULT_SPEAKER = "Sohee"
DEFAULT_INSTRUCT = (
    "Native American English Speaker, slight accent. Calm, warm, confident, slightly playful. Roleplay as someone who is from Singapore."
)

class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = DEFAULT_LANGUAGE
    speaker: str = DEFAULT_SPEAKER
    instruct: str = DEFAULT_INSTRUCT
    file_name: str | None = Field(default=None, description="Optional output file name without path")


class SpeakResponse(BaseModel):
    ok: bool
    file_path: str
    sample_rate: int | None = None
    fallback_used: bool = False

app = FastAPI(title="THURSDAY TTS Service")

@app.get("/health")
def healthcheck():
    return {"ok": True, "service": "tts", "model_loaded": _MODEL is not None}

@app.post("/speak", response_model=SpeakResponse)
def speak(req: SpeakRequest):
    wav_path, sample_rate, fallback_used = speak_to_wav(
        text=req.text,
        language=req.language,
        speaker=req.speaker,
        instruct=req.instruct,
        file_name=req.file_name,
    )

    if not wav_path:
        raise HTTPException(status_code=400, detail="Failed to synthesize speech")

    return SpeakResponse(
        ok=True,
        file_path=wav_path,
        sample_rate=sample_rate,
        fallback_used=fallback_used,
    )

@app.post("/speak/file")
def speak_file(req: SpeakRequest):
    wav_path, _, _ = speak_to_wav(
        text=req.text,
        language=req.language,
        speaker=req.speaker,
        instruct=req.instruct,
        file_name=req.file_name,
    )

    if not wav_path:
        raise HTTPException(status_code=400, detail="Failed to synthesize speech")

    return FileResponse(path=wav_path, media_type="audio/wav", filename=Path(wav_path).name)



OUT_DIR = Path("out/tts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _safe_output_path(file_name: str | None) -> Path:
    if file_name:
        cleaned = Path(file_name).name
        if not cleaned.endswith(".wav"):
            cleaned = f"{cleaned}.wav"
        return OUT_DIR / cleaned
    return OUT_DIR / f"tts_{uuid4().hex}.wav"

dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
torch_dtype = dtype_map.get(TTS_DTYPE, torch.float32)
def _get_model():
    global _MODEL
    if _MODEL is None:
        # Mac-friendly settings (CPU). On Linux RTX later: device_map="cuda:0", dtype=torch.bfloat16
        _MODEL = Qwen3TTSModel.from_pretrained(
        MODEL_NAME,
        device_map=TTS_DEVICE,
        dtype=torch_dtype,
        )
    return _MODEL


def speak_to_wav(
    text: str,
    language: str = DEFAULT_LANGUAGE,
    speaker: str = DEFAULT_SPEAKER,
    instruct: str = DEFAULT_INSTRUCT,
    file_name: str | None = None,
) -> tuple[str | None, int | None, bool]:
    text = (text or "").strip()
    if not text:
        return None, None, False

    out_wav = _safe_output_path(file_name)

    try:
        model = _get_model()
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
        )
        sf.write(str(out_wav), wavs[0], sr, subtype="PCM_16")
        return str(out_wav), sr, False

    except Exception:
        # Fallback: macOS TTS so THURSDAY never goes mute
        tmp_aiff = OUT_DIR / f"{out_wav.stem}_tmp.aiff"
        subprocess.run(["say", text, "-o", str(tmp_aiff)], check=False)
        subprocess.run(
            ["afconvert", str(tmp_aiff), str(out_wav), "-f", "WAVE", "-d", "LEI16"],
            check=False,
        )
        if out_wav.exists():
            return str(out_wav), None, True
        return None, None, True
