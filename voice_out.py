from pathlib import Path
import logging
import os
import shutil
import subprocess
from urllib.parse import quote
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from qwen_tts import Qwen3TTSModel
# from pydantic import BaseModel, Field
from pydantic import BaseModel, Field
from uuid import uuid4

# Optional offline mode for local/cached Hugging Face assets.
#os.environ.setdefault("HF_HUB_OFFLINE", "1")
#os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
#os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# "auto" picks Apple GPU (mps) when available, then CUDA, then CPU.
TTS_DEVICE = os.getenv("TTS_DEVICE", "auto")
TTS_DTYPE = os.getenv("TTS_DTYPE", "auto")
_MODEL = None
_VOICE_PROMPT = None
logger = logging.getLogger(__name__)
# The Base model clones the voice in VOICE_REF_AUDIO, which scripts/design_voice.py
# creates from a text description with the VoiceDesign model.
MODEL_NAME = os.getenv("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
VOICE_REF_AUDIO = Path(os.getenv("VOICE_REF_AUDIO", "voices/thursday.wav"))
VOICE_REF_TEXT = Path(os.getenv("VOICE_REF_TEXT", "voices/thursday.txt"))
DEFAULT_LANGUAGE = "English"

class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = DEFAULT_LANGUAGE
    file_name: str | None = Field(default=None, description="Optional output file name without path")


class SpeakResponse(BaseModel):
    ok: bool
    file_path: str
    file_name: str
    audio_url: str
    sample_rate: int | None = None
    fallback_used: bool = False

app = FastAPI(title="THURSDAY TTS Service")

@app.get("/health")
def healthcheck():
    return {"ok": True, "service": "tts", "model_loaded": _MODEL is not None}


@app.get("/audio/{file_name}")
def get_audio_file(file_name: str):
    safe_name = Path(file_name).name
    audio_path = OUT_DIR / safe_name

    if not audio_path.exists() or not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        path=audio_path,
        media_type="audio/wav",
        filename=safe_name,
    )

@app.post("/speak", response_model=SpeakResponse)
def speak(req: SpeakRequest):
    wav_path, sample_rate, fallback_used, error_detail = speak_to_wav(
        text=req.text,
        language=req.language,
        file_name=req.file_name,
    )

    if not wav_path:
        raise HTTPException(status_code=500, detail=error_detail or "Failed to synthesize speech")

    file_name = Path(wav_path).name
    audio_url = f"/audio/{quote(file_name)}"
    return SpeakResponse(
        ok=True,
        file_path=wav_path,
        file_name=file_name,
        audio_url=audio_url,
        sample_rate=sample_rate,
        fallback_used=fallback_used,
    )

@app.post("/speak/file")
def speak_file(req: SpeakRequest):
    wav_path, _, _, error_detail = speak_to_wav(
        text=req.text,
        language=req.language,
        file_name=req.file_name,
    )

    if not wav_path:
        raise HTTPException(status_code=500, detail=error_detail or "Failed to synthesize speech")

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

def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
device = _resolve_device(TTS_DEVICE)
if TTS_DTYPE == "auto":
    # bf16 is native on Apple Silicon (M2+) and Ampere+ GPUs; fp32 on CPU.
    torch_dtype = torch.float32 if device == "cpu" else torch.bfloat16
else:
    torch_dtype = dtype_map.get(TTS_DTYPE, torch.float32)


def _get_model():
    global _MODEL
    if _MODEL is None:
        logger.info("Loading %s on %s (%s)", MODEL_NAME, device, torch_dtype)
        _MODEL = Qwen3TTSModel.from_pretrained(
        MODEL_NAME,
        device_map=device,
        dtype=torch_dtype,
        )
    return _MODEL


def _get_voice_prompt(model):
    # Encoding the reference clip is slow, so do it once and reuse it for every request.
    global _VOICE_PROMPT
    if _VOICE_PROMPT is None:
        if not VOICE_REF_AUDIO.exists() or not VOICE_REF_TEXT.exists():
            raise FileNotFoundError(
                f"No voice at {VOICE_REF_AUDIO}; run scripts/design_voice.py to create one."
            )
        _VOICE_PROMPT = model.create_voice_clone_prompt(
            ref_audio=str(VOICE_REF_AUDIO),
            ref_text=VOICE_REF_TEXT.read_text().strip(),
        )
    return _VOICE_PROMPT


def _run_fallback_tts(text: str, out_wav: Path) -> bool:
    if shutil.which("say") and shutil.which("afconvert"):
        tmp_aiff = OUT_DIR / f"{out_wav.stem}_tmp.aiff"
        subprocess.run(["say", text, "-o", str(tmp_aiff)], check=True)
        subprocess.run(
            ["afconvert", str(tmp_aiff), str(out_wav), "-f", "WAVE", "-d", "LEI16"],
            check=True,
        )
        return out_wav.exists()

    fallback_bin = shutil.which("espeak-ng") or shutil.which("espeak")
    if fallback_bin:
        subprocess.run([fallback_bin, "-w", str(out_wav), text], check=True)
        return out_wav.exists()

    return False


def speak_to_wav(
    text: str,
    language: str = DEFAULT_LANGUAGE,
    file_name: str | None = None,
) -> tuple[str | None, int | None, bool, str | None]:
    text = (text or "").strip()
    if not text:
        return None, None, False, "Text cannot be empty"

    out_wav = _safe_output_path(file_name)

    try:
        model = _get_model()
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=_get_voice_prompt(model),
        )
        sf.write(str(out_wav), wavs[0], sr, subtype="PCM_16")
        return str(out_wav), sr, False, None

    except Exception as exc:
        logger.exception("Primary TTS generation failed")
        try:
            if _run_fallback_tts(text, out_wav):
                return str(out_wav), None, True, f"Primary model failed: {exc}"
        except Exception:
            logger.exception("Fallback TTS generation failed")

        return (
            None,
            None,
            True,
            f"Primary model failed: {exc}. No working fallback TTS engine is available in this container.",
        )
