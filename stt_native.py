"""Native STT on the Mac GPU (MLX): NVIDIA Parakeet TDT 0.6B v2 via parakeet-mlx.

Same API as the Docker STT (voice_in.py), so voice_ws.py only needs STT_BASE_URL:
  GET  /health      {"ok": true, "service": "stt", "model_loaded": true, "engine": "parakeet"}
  POST /transcribe  multipart "audio" (WAV) -> {"ok": true, "text": "...", "duration": s}

Docker on macOS can't use the GPU, so the container ran faster-whisper "small" on
CPU (~0.6-0.7 s per utterance). This runs natively, like Ollama and the voice.
Audio is decoded in memory (no ffmpeg subprocess). One worker thread owns MLX
(safest from the thread that loaded it); requests queue to it.

Binds 127.0.0.1 only; Docker Desktop forwards host.docker.internal to it.

    scripts/run_stt_native.sh          # port 9300; STT_NATIVE_MODEL overrides the model
"""
import asyncio
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile

MODEL_ID = os.getenv("STT_NATIVE_MODEL", "mlx-community/parakeet-tdt-0.6b-v2")
SAMPLE_RATE = 16000

logger = logging.getLogger("stt-native")
app = FastAPI(title="THURSDAY native STT")
_pool = ThreadPoolExecutor(max_workers=1)  # the one thread that touches MLX
_model = None


def _load() -> None:
    global _model
    from parakeet_mlx import from_pretrained
    started = time.monotonic()
    _model = from_pretrained(MODEL_ID)
    _run(np.zeros(SAMPLE_RATE, dtype=np.float32))  # compile kernels before the first real request
    logger.info("%s ready in %.1fs", MODEL_ID, time.monotonic() - started)


def _run(audio: np.ndarray) -> str:
    import mlx.core as mx
    from parakeet_mlx.audio import get_logmel
    mel = get_logmel(mx.array(audio), _model.preprocessor_config)
    return _model.generate(mel)[0].text.strip()


def _decode(raw: bytes) -> np.ndarray:
    audio, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:  # THURSDAY sends 16 kHz; tests may not
        import librosa
        audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE)
    return audio


@app.on_event("startup")
async def startup() -> None:
    await asyncio.get_running_loop().run_in_executor(_pool, _load)


@app.get("/health")
def health():
    return {"ok": True, "service": "stt", "model_loaded": _model is not None, "engine": "parakeet", "model": MODEL_ID}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="No audio data received")
    started = time.monotonic()
    try:
        pcm = _decode(raw)
        if len(pcm) < SAMPLE_RATE // 10:  # under 100 ms: nothing to recognise
            return {"ok": True, "text": "", "duration": len(pcm) / SAMPLE_RATE}
        text = await asyncio.get_running_loop().run_in_executor(_pool, _run, pcm)
    except Exception as exc:
        logger.exception("STT transcription failed")
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    logger.info("STT ok: audio=%.1fs took=%dms", len(pcm) / SAMPLE_RATE, (time.monotonic() - started) * 1000)
    return {"ok": True, "text": text, "language": "en", "duration": len(pcm) / SAMPLE_RATE}
