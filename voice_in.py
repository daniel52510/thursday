from pathlib import Path
import logging
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from faster_whisper import WhisperModel
from pydantic import BaseModel

STT_DEVICE = os.getenv("STT_DEVICE", "cpu")
STT_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "int8")
STT_MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "small")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en")

_MODEL = None
logger = logging.getLogger(__name__)

OUT_DIR = Path("out/stt")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class TranscribeResponse(BaseModel):
    ok: bool
    text: str | None = None
    language: str | None = None
    duration: float | None = None
    error: str | None = None


app = FastAPI(title="THURSDAY STT Service")


@app.get("/health")
def healthcheck():
    return {"ok": True, "service": "stt", "model_loaded": _MODEL is not None}


def _get_model() -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = WhisperModel(
            STT_MODEL_SIZE,
            device=STT_DEVICE,
            compute_type=STT_COMPUTE_TYPE,
        )
    return _MODEL


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: UploadFile = File(...)):
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="No audio data received")

    # Persist the most recent upload so we can inspect/replay it for debugging.
    suffix = Path(audio.filename or "input.wav").suffix or ".wav"
    in_path = OUT_DIR / f"last_input{suffix}"
    in_path.write_bytes(raw)

    try:
        model = _get_model()
        segments, info = model.transcribe(
            str(in_path),
            language=None if STT_LANGUAGE.lower() == "auto" else STT_LANGUAGE,
            vad_filter=True,
        )
        text = "".join(segment.text for segment in segments).strip()
        return TranscribeResponse(
            ok=True,
            text=text,
            language=info.language,
            duration=info.duration,
        )
    except Exception as exc:
        logger.exception("STT transcription failed")
        return TranscribeResponse(ok=False, error=str(exc))
