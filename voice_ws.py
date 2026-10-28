"""Real-time voice endpoint. A client (the browser page now, the Pi satellite
later) streams 16 kHz mono int16 PCM and gets THURSDAY's reply back as WAV.

One Session per connection, one turn at a time:

    idle --"start"--> listening --VAD end--> thinking --audio sent--> speaking
      ^                   |                     |                        |
      +---- no speech ----+------ failure ------+---- "playback_ended" --+

Protocol
  client -> server  binary: PCM frames (any size; re-sliced into 32 ms frames)
                    text:   {"type": "start" | "playback_started" | "playback_ended"}
  server -> client  text:   {"type": "state", "state": ...}
                            {"type": "transcript" | "reply", "text": ...}
                            {"type": "error", "stage": ..., "message": ...}
                    binary: one WAV per reply
"""
import asyncio
import hmac
import io
import json
import logging
import os
import re
import time
import wave

import httpx
from fastapi import APIRouter, WebSocket, status

from agent_stream import stream_prompt
from vad import FRAME_BYTES, FRAME_MS, SAMPLE_RATE, Endpointer, VADConfig

STT_BASE_URL = os.getenv("STT_BASE_URL", "http://thursday-stt:9200").rstrip("/")
TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://host.docker.internal:9000").rstrip("/")
VOICE_TOKEN = os.getenv("VOICE_TOKEN", "")
NO_SPEECH_TIMEOUT_S = float(os.getenv("VOICE_NO_SPEECH_TIMEOUT_MS", "8000")) / 1000
STT_TIMEOUT = httpx.Timeout(30, connect=5)
TTS_TIMEOUT = httpx.Timeout(60, connect=5)

logger = logging.getLogger("thursday.voice")
if not logger.handlers:
    # uvicorn only configures its own loggers; give ours a handler so the
    # per-turn timing lines reach `docker logs`.
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class _RedactToken(logging.Filter):
    """uvicorn logs the WebSocket path with its query string; keep the token out of logs."""

    _pattern = re.compile(r"(token=)[^&\s\"']+")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "token=" in msg:
            record.msg, record.args = self._pattern.sub(r"\1***", msg), ()
        return True


for _name in ("uvicorn.error", "uvicorn.access"):
    logging.getLogger(_name).addFilter(_RedactToken())

router = APIRouter()


def _token_ok(token: str | None) -> bool:
    if not VOICE_TOKEN:
        logger.error("VOICE_TOKEN is not set; refusing all /ws/voice connections")
        return False
    return bool(token) and hmac.compare_digest(token.encode(), VOICE_TOKEN.encode())


def _ms(since: float) -> int:
    return round((time.monotonic() - since) * 1000)


def _pcm_to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


class Session:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.state = "idle"
        self.endpointer = Endpointer(VADConfig.from_env())
        self.http = httpx.AsyncClient()
        self._buf = bytearray()
        self._listen_started = 0.0
        self._turn: asyncio.Task | None = None
        # Filled by a turn; logged when the client reports playback started.
        self._timing: dict | None = None
        self._speech_end = 0.0
        self._conversation_id: str | None = None  # sent by the client with "start"; a turn may create one
        self._context: dict | None = None  # {"timezone", "location"}, also sent with "start"

    async def set_state(self, state: str) -> None:
        self.state = state
        await self.ws.send_json({"type": "state", "state": state})

    async def on_control(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "start" and self.state == "idle":
            self.endpointer.reset()
            self._buf.clear()
            self._listen_started = time.monotonic()
            self._conversation_id = msg.get("conversation_id") or self._conversation_id
            self._context = msg.get("context")
            await self.set_state("listening")
        elif kind == "playback_started" and self._timing is not None:
            self._timing["speech_end_to_playback_ms"] = _ms(self._speech_end)
            self._log_timing()
        elif kind == "playback_ended" and self.state == "speaking":
            await self.set_state("idle")
        elif kind == "stop" and self.state == "listening":
            await self.set_state("idle")  # mic muted
        elif kind == "interrupt":
            # Tap-to-interrupt: drop the turn wherever it is. Cancelling the task
            # closes stream_prompt's HTTP stream, so Ollama stops generating too.
            if self._turn and not self._turn.done():
                self._turn.cancel()
                logger.info("turn interrupted in state %s", self.state)
            self._timing = None
            await self.set_state("idle")

    async def on_audio(self, data: bytes) -> None:
        # Outside "listening" the audio is ignored; barge-in (3c) will watch it while speaking.
        if self.state != "listening":
            return
        self._buf.extend(data)
        while len(self._buf) >= FRAME_BYTES:
            frame = bytes(self._buf[:FRAME_BYTES])
            del self._buf[:FRAME_BYTES]
            utterance = self.endpointer.feed(frame)
            if utterance is not None:
                self._speech_end = time.monotonic()
                await self.set_state("thinking")
                self._turn = asyncio.create_task(self._run_turn(utterance))
                return
        if not self.endpointer.in_speech and time.monotonic() - self._listen_started > NO_SPEECH_TIMEOUT_S:
            await self.set_state("idle")

    async def _run_turn(self, pcm: bytes) -> None:
        timing = {
            "speech_ms": len(pcm) // 2 * 1000 // SAMPLE_RATE,
            "endpoint_wait_ms": self.endpointer.cfg.end_silence_ms,
        }
        stage, input_desc, started = "STT", f"{len(pcm) // 1024} KB PCM", time.monotonic()
        try:
            text = await self._transcribe(pcm)
            timing["stt_ms"] = _ms(started)
            if not text:
                await self._fail("STT", "no speech recognized", input_desc, started)
                return
            await self.ws.send_json({"type": "transcript", "text": text})

            stage, input_desc, started = "LLM", f"{len(text)} chars", time.monotonic()
            # Sentence-chunked TTS is 3b; for now collect the streamed reply.
            speak_text = ""
            async for ev in stream_prompt(text, mode="voice", conversation_id=self._conversation_id, context=self._context):
                if ev["type"] == "conversation":
                    self._conversation_id = ev["id"]
                    await self.ws.send_json(ev)
                elif ev["type"] == "delta" and "llm_ttft_ms" not in timing:
                    timing["llm_ttft_ms"] = _ms(started)
                elif ev["type"] == "tool":
                    await self.ws.send_json({"type": "tool", "name": ev["name"]})
                elif ev["type"] == "done":
                    speak_text = ev["reply"]
            timing["llm_ms"] = _ms(started)
            await self.ws.send_json({"type": "reply", "text": speak_text})

            stage, input_desc, started = "TTS", f"{len(speak_text)} chars", time.monotonic()
            audio = await self._synthesize(speak_text)
            timing["tts_ms"] = _ms(started)
            if not audio:
                await self.set_state("idle")
                self._timing = timing
                self._log_timing()
                return

            # Store before sending: the client's playback_started can arrive
            # while send_bytes is still yielding to the event loop.
            self._timing = timing
            await self.set_state("speaking")
            timing["speech_end_to_audio_sent_ms"] = _ms(self._speech_end)
            await self.ws.send_bytes(audio)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail(stage, f"{type(exc).__name__}: {exc}", input_desc, started)

    async def _transcribe(self, pcm: bytes) -> str:
        files = {"audio": ("utterance.wav", _pcm_to_wav(pcm), "audio/wav")}
        resp = await self.http.post(f"{STT_BASE_URL}/transcribe", files=files, timeout=STT_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("ok"):
            raise RuntimeError(payload.get("error") or "STT returned ok=false")
        return str(payload.get("text") or "").strip()

    async def _synthesize(self, text: str) -> bytes | None:
        if not text:
            return None
        resp = await self.http.post(f"{TTS_BASE_URL}/speak", json={"text": text}, timeout=TTS_TIMEOUT)
        resp.raise_for_status()
        if resp.status_code == 204:  # nothing speakable after cleanup
            return None
        audio = await self.http.get(f"{TTS_BASE_URL}{resp.json()['audio_url']}", timeout=TTS_TIMEOUT)
        audio.raise_for_status()
        return audio.content

    async def _fail(self, stage: str, message: str, input_desc: str, started: float) -> None:
        logger.error(
            "%s failed after %.1fs (input: %s) - %s", stage, time.monotonic() - started, input_desc, message
        )
        await self.ws.send_json({"type": "error", "stage": stage, "message": message})
        await self.set_state("idle")

    def _log_timing(self) -> None:
        logger.info("turn timing: %s", " ".join(f"{k}={v}" for k, v in self._timing.items()))
        self._timing = None

    async def close(self) -> None:
        if self._turn and not self._turn.done():
            self._turn.cancel()
        await self.http.aclose()


@router.websocket("/ws/voice")
async def voice_ws(ws: WebSocket) -> None:
    # Accept first so the client receives the 1008 close code; rejecting during
    # the handshake would surface in the browser only as a generic 1006.
    await ws.accept()
    if not _token_ok(ws.query_params.get("token")):
        logger.warning("Rejected /ws/voice from %s: bad or missing token", ws.client.host if ws.client else "?")
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="invalid token")
        return

    session = Session(ws)
    logger.info("voice session opened (frame=%dms, vad=%s)", FRAME_MS, session.endpointer.cfg)
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                await session.on_audio(msg["bytes"])
            elif msg.get("text"):
                await session.on_control(json.loads(msg["text"]))
    finally:
        await session.close()
        logger.info("voice session closed")
