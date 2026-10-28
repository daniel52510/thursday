from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import unicodedata
from urllib.parse import quote
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
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
# uvicorn only configures its own loggers; without this, INFO lines here are dropped.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s: %(message)s")
logger = logging.getLogger(__name__)
# The Base model clones the voice in VOICE_REF_AUDIO, which scripts/design_voice.py
# creates from a text description with the VoiceDesign model.
MODEL_NAME = os.getenv("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
VOICE_REF_AUDIO = Path(os.getenv("VOICE_REF_AUDIO", "voices/thursday.wav"))
VOICE_REF_TEXT = Path(os.getenv("VOICE_REF_TEXT", "voices/thursday.txt"))
DEFAULT_LANGUAGE = "English"
# The model emits 12 codec frames per second of audio and speech runs ~0.7
# frames per character, so 1.6/char allows ~2x headroom. Without a cap the
# library default (2048 frames, ~170s) lets a run-on generation block for minutes.
FRAMES_PER_SECOND = 12
MAX_TOKENS_PER_CHAR = 1.6
MIN_NEW_TOKENS = 120
MAX_NEW_TOKENS = 2048
# One synthesis at a time: concurrent generate calls on MPS from the threadpool
# compete for the GPU and can stall each other.
_SYNTH_LOCK = threading.Lock()

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

@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Load the model, encode the voice prompt and run one short synthesis before
    # accepting requests, so the first real reply doesn't pay ~8s of cold start.
    if os.getenv("TTS_WARMUP", "1") == "1":
        started = time.monotonic()
        wav_path, _, fallback_used, error = await asyncio.to_thread(
            speak_to_wav, "Ready.", DEFAULT_LANGUAGE, "warmup.wav"
        )
        if wav_path and not fallback_used:
            logger.info("TTS warm-up done in %.1fs", time.monotonic() - started)
        else:
            logger.error("TTS warm-up failed; first request will be slow or fail: %s", error)
    yield


app = FastAPI(title="THURSDAY TTS Service", lifespan=_lifespan)

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

@app.post("/speak", response_model=SpeakResponse, responses={204: {"description": "Nothing speakable"}})
def speak(req: SpeakRequest):
    # Emoji-only or markup-only replies are a normal outcome, not a server error.
    if not clean_for_tts(req.text):
        return Response(status_code=204)
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

@app.post("/speak/file", responses={204: {"description": "Nothing speakable"}})
def speak_file(req: SpeakRequest):
    if not clean_for_tts(req.text):
        return Response(status_code=204)
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
# Every reply is written to a file the client fetches once; keep only the newest.
KEEP_OUTPUTS = int(os.getenv("TTS_KEEP_OUTPUTS", "50"))


def _prune_outputs() -> None:
    files = sorted(OUT_DIR.glob("*.wav"), key=lambda f: f.stat().st_mtime, reverse=True)
    for old in files[KEEP_OUTPUTS:]:
        old.unlink(missing_ok=True)


def _safe_output_path(file_name: str | None) -> Path:
    _prune_outputs()
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


_CODE_FENCE = re.compile(r"```.*?(```|$)", re.S)
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_HEADER = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.M)
_EMPHASIS = re.compile(r"[*_~`]+")
_KEEP_SYMBOLS = set("$€£%°&+=@/'\"")


def clean_for_tts(text: str) -> str:
    """Turn LLM output (often markdown) into plain speakable sentences."""
    text = _CODE_FENCE.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub(" ", text)
    text = _HEADER.sub("", text)
    text = _BULLET.sub("", text)
    text = _EMPHASIS.sub("", text)
    # Drop emoji, box-drawing and other symbols the model can't voice.
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] in "LNPZ" or ch in "\n" or ch in _KEEP_SYMBOLS
    )
    # Headers and list items have no final punctuation; end each line as a
    # sentence so they don't run together into one breathless clause.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = [ln if ln[-1] in ".!?:;," else f"{ln}." for ln in lines]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _max_new_tokens(text: str) -> int:
    return max(MIN_NEW_TOKENS, min(MAX_NEW_TOKENS, int(len(text) * MAX_TOKENS_PER_CHAR)))


def _run_fallback_tts(text: str, out_wav: Path) -> bool:
    if shutil.which("say") and shutil.which("afconvert"):
        tmp_aiff = OUT_DIR / f"{out_wav.stem}_tmp.aiff"
        subprocess.run(["say", text, "-o", str(tmp_aiff)], check=True, timeout=30)
        subprocess.run(
            ["afconvert", str(tmp_aiff), str(out_wav), "-f", "WAVE", "-d", "LEI16"],
            check=True,
            timeout=30,
        )
        return out_wav.exists()

    fallback_bin = shutil.which("espeak-ng") or shutil.which("espeak")
    if fallback_bin:
        subprocess.run([fallback_bin, "-w", str(out_wav), text], check=True, timeout=30)
        return out_wav.exists()

    return False


def speak_to_wav(
    text: str,
    language: str = DEFAULT_LANGUAGE,
    file_name: str | None = None,
) -> tuple[str | None, int | None, bool, str | None]:
    raw_len = len(text or "")
    text = clean_for_tts(text or "")
    if not text:
        return None, None, False, f"No speakable text after cleanup (input was {raw_len} chars)"

    out_wav = _safe_output_path(file_name)
    max_new_tokens = _max_new_tokens(text)
    started = time.monotonic()
    waited = -1.0

    try:
        with _SYNTH_LOCK:
            waited = time.monotonic() - started
            model = _get_model()
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=_get_voice_prompt(model),
                max_new_tokens=max_new_tokens,
            )
        sf.write(str(out_wav), wavs[0], sr, subtype="PCM_16")
        audio_s = len(wavs[0]) / sr
        logger.info(
            "TTS ok: chars=%d audio=%.1fs synth=%.1fs lock_wait=%.1fs",
            len(text), audio_s, time.monotonic() - started - waited, waited,
        )
        frames = audio_s * FRAMES_PER_SECOND
        if frames >= max_new_tokens * 0.98:
            # Hitting the cap means the model never emitted end-of-speech.
            logger.warning(
                "TTS hit the token cap (%d frames) for %d chars; output is likely truncated or run-on: %r",
                max_new_tokens, len(text), text[:120],
            )
        return str(out_wav), sr, False, None

    except Exception as exc:
        logger.exception(
            "TTS failed: chars=%d (raw %d) max_new_tokens=%d lock_wait=%.1fs elapsed=%.1fs",
            len(text), raw_len, max_new_tokens, waited, time.monotonic() - started,
        )
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
