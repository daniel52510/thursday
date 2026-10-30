"""Real-time voice endpoint. A client (the browser page or the Pi satellite)
streams 16 kHz mono int16 PCM and gets THURSDAY's reply back as WAV.

One Session per connection, one turn at a time:

    idle --"start"--> listening --VAD end--> thinking --audio sent--> speaking
      ^                   |                     |                        |
      +---- no speech ----+------ failure ------+---- "playback_ended" --+

Protocol
  client -> server  binary: PCM frames (any size; re-sliced into 32 ms frames)
                    text:   {"type": "start" | "playback_started" | "playback_ended"}
                            "start" may carry "chunked": true (see below)
  server -> client  text:   {"type": "state", "state": ...}
                            {"type": "transcript" | "reply", "text": ...}
                            {"type": "audio_end", "chunks": n}   chunked only, after the last WAV
                            {"type": "error", "stage": ..., "message": ...}
                    binary: one WAV per reply, or with "chunked" one WAV per
                            sentence, sent while later ones are synthesized.
                            Clients play them in order and send
                            playback_started on the first, playback_ended
                            after the last (once audio_end has arrived).
"""
import asyncio
import hmac
import io
import json
import logging
import os
import random
import re
import time
import wave

import httpx
import numpy as np
from fastapi import APIRouter, WebSocket, status

from agent_stream import stream_prompt
from vad import FRAME_BYTES, FRAME_MS, SAMPLE_RATE, Endpointer, VADConfig

STT_BASE_URL = os.getenv("STT_BASE_URL", "http://thursday-stt:9200").rstrip("/")
TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://host.docker.internal:9000").rstrip("/")
VOICE_TOKEN = os.getenv("VOICE_TOKEN", "")
NO_SPEECH_TIMEOUT_S = float(os.getenv("VOICE_NO_SPEECH_TIMEOUT_MS", "8000")) / 1000
# Speculative STT: after this much silence, transcribe the utterance so far in the
# background. If no speech follows before the endpoint (VAD_END_SILENCE_MS), the
# final audio is identical, so the turn reuses that transcript instead of waiting
# ~800 ms for STT after the endpoint. Speech resuming throws it away.
SPECULATE_FRAMES = max(1, round(float(os.getenv("STT_SPECULATE_MS", "250")) / FRAME_MS))
STT_TIMEOUT = httpx.Timeout(30, connect=5)
TTS_TIMEOUT = httpx.Timeout(60, connect=5)
# A TTS server that can stream (the CSM voice from the thursday-voice project)
# sends raw PCM while it is still generating: the first sound comes ~0.3 s after
# the text instead of after the whole piece. Qwen can't; it keeps /speak/file.
TTS_CAPS_TTL_S = 30
STREAM_MIN_BYTES = 7680  # 160 ms of 24 kHz int16: the smallest WAV we forward
_tts_caps = {"at": 0.0, "stream": False, "rate": 24000}

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
SATELLITES: set["Session"] = set()  # connected satellites, for server-initiated speech (alarms)


def _token_ok(token: str | None) -> bool:
    if not VOICE_TOKEN:
        logger.error("VOICE_TOKEN is not set; refusing all /ws/voice connections")
        return False
    return bool(token) and hmac.compare_digest(token.encode(), VOICE_TOKEN.encode())


def _ms(since: float) -> int:
    return round((time.monotonic() - since) * 1000)


# Sentence end: punctuation (optionally closing quotes/brackets) then whitespace.
_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*\s+")
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "jr", "sr", "no", "approx"}
_CLAUSE_BREAK = re.compile(r" (?=(and|but|with|which|because|while|after|before|so|according to) )", re.I)
_SOFT_BREAK = re.compile(r"[,;:](?=\s)")
_INITIALISM = re.compile(r"(?:[A-Za-z]\.)+[A-Za-z]?")  # U.S, e.g, p.m
# Chunked TTS pacing (see Session._speak_chunks). The first piece is short so
# audio starts fast; the guesses are replaced by measurements after one piece.
FIRST_PIECE_CHARS = 60
# Streaming TTS: every break between pieces costs ~1 s (the voice generates its own lead-in
# and trailing silence at about real time), while the LLM finishes a voice reply ~0.5-0.7 s
# after its first token. So the first piece waits up to this long after the first sentence
# is complete for the rest, and short replies go out as one piece with no breaks.
FIRST_PIECE_WAIT_S = 0.6
TTS_RTF_GUESS = 0.85        # synthesis seconds per second of audio (1.7B on MPS)
TTS_CHARS_PER_S_GUESS = 16  # characters of text per second of speech
PIECE_SAFETY = 0.85         # aim to finish each piece with ~15% of the lead to spare


class SentenceChunker:
    """Holds streamed LLM text and hands out speakable pieces for chunked TTS.

    The speaker pulls a piece with a character budget. Qwen3-TTS 1.7B on MPS
    synthesizes at roughly real time, so a piece is only ready before the
    audio queued ahead of it runs out if it is short enough; the speaker sizes
    the budget from that lead (see Session._speak_chunks). Within the budget a
    piece ends at the latest sentence end, else a comma, else before a joining
    word: never mid-phrase, so prosody stays natural. Pieces grow as the lead
    builds, so longer replies settle into whole sentences.
    """

    MIN_CHARS = 12
    MAX_CHARS = 220

    def __init__(self) -> None:
        self._buf = ""
        self.closed = False
        self._changed = asyncio.Event()

    def feed(self, text: str) -> None:
        self._buf += text
        self._changed.set()

    def close(self) -> None:
        """The LLM is done: whatever is left can go out without a boundary."""
        self.closed = True
        self._changed.set()

    @property
    def empty(self) -> bool:
        return not self._buf.strip()

    async def settle(self, wait_s: float) -> None:
        """Wait for a first complete sentence, then up to wait_s more for the LLM to finish,
        so a short reply can go to TTS whole instead of in pieces."""
        loop = asyncio.get_running_loop()
        deadline = None
        while not self.closed:
            if deadline is None and self._sentence_ends():
                deadline = loop.time() + wait_s
            timeout = None if deadline is None else deadline - loop.time()
            if timeout is not None and timeout <= 0:
                return
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), timeout)
            except asyncio.TimeoutError:
                return

    async def next_piece(self, max_chars: int) -> str | None:
        """The next piece within max_chars when possible; None once all text is spoken."""
        while True:
            piece = self._take(max_chars)
            if piece:
                return piece
            if self.closed:
                rest, self._buf = self._buf.strip(), ""
                return rest or None
            self._changed.clear()
            await self._changed.wait()

    def _sentence_ends(self) -> list[int]:
        ends = []
        for m in _SENTENCE_END.finditer(self._buf):
            last_word = self._buf[:m.end()].rstrip().rstrip(".!?…\"')]").rsplit(None, 1)[-1:] or [""]
            # "Dr. Smith", "U.S. Open", "3 p.m. today": not the end of a sentence ("People.com." is).
            if last_word[0].lower() not in _ABBREVIATIONS and not _INITIALISM.fullmatch(last_word[0]):
                ends.append(m.end())
        return ends

    def _take(self, max_chars: int) -> str | None:
        buf = self._buf
        fits = lambda i: self.MIN_CHARS <= len(buf[:i].strip()) <= max_chars  # noqa: E731
        ends = self._sentence_ends()
        if self.closed and buf.strip() and (not ends or ends[-1] < len(buf.rstrip())):
            ends.append(len(buf))  # the reply's end is a sentence end (no whitespace follows it)
        cut = next((i for i in reversed(ends) if fits(i)), None)
        if cut is None and not self.closed and len(buf) < max_chars:
            return None  # the sentence may still end within budget: wait for more text
        if cut is None:
            # The sentence is longer than the budget: break at a comma or before a joining word.
            soft = sorted([m.start() + 1 for m in _SOFT_BREAK.finditer(buf)] + [m.start() for m in _CLAUSE_BREAK.finditer(buf)])
            soft = [i for i in soft if len(buf[:i].strip()) >= self.MIN_CHARS and (not ends or i < ends[0])]
            cut = max((i for i in soft if fits(i)), default=None)
            if cut is None and soft and (ends or self.closed or len(buf) >= max_chars):
                cut = soft[0]  # nothing fits: the earliest natural break overshoots least
        if cut is None and ends:
            cut = ends[0]  # no break inside the sentence at all: the whole sentence, gap or not
        if cut is None:
            return None  # a long run with no break yet; wait (close() flushes it)
        piece, self._buf = buf[:cut].strip(), buf[cut:].lstrip()
        return piece


# Qwen3-TTS pads each clip with 0.1-0.5 s of silence at both ends; between
# chunks that stacks into pauses far longer than the punctuation calls for.
# Trim to the speech (keeping a little margin for soft onsets and breaths)
# and add back a pause sized like natural speech.
SILENCE_FLOOR = 0.02   # fraction of the clip's peak treated as silence
EDGE_MARGIN_MS = 40
PAUSE_SENTENCE_MS = 300
PAUSE_CLAUSE_MS = 150
PAUSE_JOIN_MS = 60


def _pause_after(piece: str) -> int:
    end = piece.rstrip("\"')]")[-1:]
    if end in ".!?…":
        return PAUSE_SENTENCE_MS
    if end in ",;:":
        return PAUSE_CLAUSE_MS
    return PAUSE_JOIN_MS  # cut before a joining word: speech carries straight on


def _trim_wav(wav: bytes, lead_ms: int, pause_ms: int) -> bytes:
    try:
        with wave.open(io.BytesIO(wav)) as w:
            params, frames = w.getparams(), w.readframes(w.getnframes())
    except (wave.Error, EOFError):
        return wav
    if params.sampwidth != 2 or not frames:
        return wav
    samples = np.frombuffer(frames, dtype="<i2").reshape(-1, params.nchannels)
    level = np.abs(samples.astype(np.int32)).max(axis=1)
    loud = np.flatnonzero(level > SILENCE_FLOOR * max(int(level.max()), 1))
    if loud.size == 0:
        return wav
    per_ms = params.framerate / 1000
    start = max(0, int(loud[0] - EDGE_MARGIN_MS * per_ms))
    end = min(len(samples), int(loud[-1] + EDGE_MARGIN_MS * per_ms))
    pad = lambda ms: np.zeros((int(ms * per_ms), params.nchannels), dtype="<i2")  # noqa: E731
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(params.nchannels)
        w.setsampwidth(2)
        w.setframerate(params.framerate)
        w.writeframes(np.concatenate([pad(lead_ms), samples[start:end], pad(pause_ms)]).tobytes())
    return out.getvalue()


# Spoken fillers: when she reaches for a tool, a short line in her own voice
# ("Ooh, let me see.") covers the wait instead of a looping cue, and a soft
# "Hmm." covers a plain reply that's slow to start. Audio only: they never
# appear in the written reply. Rendered once per process (CSM takes ~0.5 s a
# line) and kept in memory, so one plays the instant it's needed.
FILLERS = {
    "web_search": ["Mm, let me look.", "Ooh, one sec.", "Hang on, let me see.", "Okay, checking."],
    "sports_scores": ["Ooh, let me see.", "One sec, checking the game.", "Hang on, pulling up the score."],
    "launch_schedule": ["Ooh, my favorite. One sec.", "Let me check what's flying.", "Hang on, pulling up launches."],
    "get_weather": ["Mm, let me check the sky.", "One sec, checking the forecast."],
    "calendar_events": ["Let me look at your day.", "One sec, checking your schedule."],
    "calendar_create": ["Okay, one sec.", "Sure, let me set that up."],
    "calendar_delete": ["Okay, let me find it.", "Hang on."],
    "_slow": ["Hmm.", "Mm, let me think.", "Hmm, okay."],
}
SLOW_REPLY_S = 1.6  # a plain reply with no audio this long after speech end gets a "Hmm."
_filler_audio: dict[str, bytes] = {}
_filler_task: asyncio.Task | None = None
_last_filler = ""


async def _render_fillers() -> None:
    async with httpx.AsyncClient() as http:
        for line in {line for lines in FILLERS.values() for line in lines}:
            if line in _filler_audio:
                continue
            try:
                wav = await _tts(http, line)
            except Exception as exc:  # voice server down: try again with the next session
                logger.warning("filler %r not rendered: %s", line, exc)
                return
            if wav:
                _filler_audio[line] = _trim_wav(wav, EDGE_MARGIN_MS, PAUSE_CLAUSE_MS)
    logger.info("spoken fillers ready (%d)", len(_filler_audio))


def warm_fillers() -> None:
    global _filler_task
    if (_filler_task is None or _filler_task.done()) and len(_filler_audio) < sum(map(len, FILLERS.values())):
        _filler_task = asyncio.create_task(_render_fillers())


def pick_filler(kind: str) -> bytes | None:
    """A ready line for this tool (or "_slow"), never the one she said last."""
    global _last_filler
    ready = [line for line in FILLERS.get(kind, []) if line in _filler_audio]
    fresh = [line for line in ready if line != _last_filler] or ready
    if not fresh:
        return None
    _last_filler = random.choice(fresh)
    return _filler_audio[_last_filler]


# Clock times as she'd say them: the voice reads "8:15" digit by digit around the
# colon, so every piece of reply text is rewritten before it reaches TTS
# ("8:15" -> "eight fifteen", "8:05" -> "eight oh five", "8:00" -> "eight o'clock").
# "PM" / "p.m." / "pm": only the dotted form's own dots are consumed, so "at 4:25 PM." keeps its full stop.
_TIME = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)(?:\s*(?:([AaPp])\.\s?[Mm]\.|([AaPp])\s?[Mm]\b))?")
_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
          "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty"}


def _number_words(n: int) -> str:
    if n < 20:
        return _UNITS[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + (f"-{_UNITS[ones]}" if ones else "")


def _say_time(m: re.Match) -> str:
    hour, minute, half = int(m.group(1)), int(m.group(2)), (m.group(3) or m.group(4) or "").upper()
    # "p.m." ending a sentence was also its full stop: put it back.
    rest = m.string[m.end():]
    stop = "." if m.group(3) and (not rest.strip() or rest.lstrip()[:1].isupper()) else ""
    if not half and hour >= 13:  # 24-hour clock: "18:30" -> "six thirty PM"
        hour, half = hour - 12, "P"
    elif not half and hour == 0:
        hour, half = 12, "A"
    words = _number_words(hour)
    if minute:
        words += f" oh {_UNITS[minute]}" if minute < 10 else f" {_number_words(minute)}"
    elif not half:
        words += " o'clock"
    return (f"{words} {half}M" if half else words) + stop


def _speakable(text: str) -> str:
    return _TIME.sub(_say_time, text)


async def _tts(http: httpx.AsyncClient, text: str) -> bytes | None:
    text = _speakable(text)
    if not text:
        return None
    # /speak/file returns the WAV itself: one round trip instead of two.
    resp = await http.post(f"{TTS_BASE_URL}/speak/file", json={"text": text}, timeout=TTS_TIMEOUT)
    resp.raise_for_status()
    if resp.status_code == 204:  # nothing speakable after cleanup
        return None
    return resp.content


async def synthesize_ahead(http: httpx.AsyncClient, text: str) -> list[bytes]:
    """All of `text` as trimmed WAVs, one per whole sentence, for audio made
    before anyone is listening (alarm briefings). With no listener to keep
    ahead of, there's no pacing: whole sentences sound the most natural."""
    chunker = SentenceChunker()
    chunker.feed(text)
    chunker.close()
    pieces = []
    while (piece := await chunker.next_piece(SentenceChunker.MAX_CHARS)) is not None:
        pieces.append(piece)
    audio = []
    for i, piece in enumerate(pieces):
        wav = await _tts(http, piece)
        if wav:
            audio.append(_trim_wav(wav, lead_ms=0, pause_ms=0 if i == len(pieces) - 1 else _pause_after(piece)))
    return audio


def _wav_seconds(wav: bytes) -> float:
    try:
        with wave.open(io.BytesIO(wav)) as w:
            return w.getnframes() / w.getframerate()
    except (wave.Error, EOFError, ZeroDivisionError):
        return 0.0


def _pcm_to_wav(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


async def _tts_streams(http: httpx.AsyncClient) -> bool:
    """Whether the TTS server can stream (its /health says so). Re-checked every
    TTS_CAPS_TTL_S, so switching TTS servers needs no restart here."""
    if time.monotonic() - _tts_caps["at"] > TTS_CAPS_TTL_S:
        try:
            info = (await http.get(f"{TTS_BASE_URL}/health", timeout=5)).json()
            _tts_caps.update(stream=bool(info.get("streaming")), rate=int(info.get("sample_rate") or 24000))
        except Exception:
            _tts_caps["stream"] = False
        _tts_caps["at"] = time.monotonic()
    return _tts_caps["stream"]


async def _tts_stream(http: httpx.AsyncClient, text: str):
    """Yield one piece as WAVs of >= STREAM_MIN_BYTES while the server is still
    generating it. Leaving early (barge-in) closes the request, which stops the
    server's generation."""
    text = _speakable(text)
    async with http.stream("POST", f"{TTS_BASE_URL}/speak/stream", json={"text": text}, timeout=TTS_TIMEOUT) as resp:
        if resp.status_code == 204:  # nothing speakable after cleanup
            return
        resp.raise_for_status()
        buf = b""
        async for data in resp.aiter_bytes():
            buf += data
            if len(buf) >= STREAM_MIN_BYTES:
                cut = len(buf) - len(buf) % 2  # whole int16 samples only
                yield _pcm_to_wav(buf[:cut], _tts_caps["rate"])
                buf = buf[cut:]
        if len(buf) >= 2:
            yield _pcm_to_wav(buf[: len(buf) - len(buf) % 2], _tts_caps["rate"])


class Session:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.state = "idle"
        self.endpointer = Endpointer(VADConfig.from_env())
        self.http = httpx.AsyncClient()
        self._buf = bytearray()
        self._listen_started = 0.0
        self._turn: asyncio.Task | None = None
        self._spec: tuple[asyncio.Task, int] | None = None  # (speculative STT, endpointer.voiced at snapshot)
        # Filled by a turn; logged when the client reports playback started.
        self._timing: dict | None = None
        self._speech_end = 0.0
        self._conversation_id: str | None = None  # sent by the client with "start"; a turn may create one
        self._context: dict | None = None  # {"timezone", "location"}, also sent with "start"
        self._chunked = False  # client asked for one WAV per sentence
        self.role = "client"   # "satellite" after its hello

    async def set_state(self, state: str) -> None:
        self.state = state
        await self.ws.send_json({"type": "state", "state": state})

    async def on_control(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "start" and self.state == "idle":
            self._drop_spec()
            self.endpointer.reset()
            self._buf.clear()
            self._listen_started = time.monotonic()
            self._conversation_id = msg.get("conversation_id") or self._conversation_id
            self._chunked = bool(msg.get("chunked"))
            self._context = msg.get("context")
            await self.set_state("listening")
        elif kind == "playback_started" and self._timing is not None:
            self._timing["speech_end_to_playback_ms"] = _ms(self._speech_end)
            if not self._chunked:  # chunked turns are still synthesizing; log at the end
                self._log_timing()
        elif kind == "playback_ended" and self.state == "speaking":
            if self._timing is not None:
                self._log_timing()
            await self.set_state("idle")
        elif kind == "hello":
            # Satellites say so on connect; the alarm scheduler speaks only through them.
            if msg.get("role") == "satellite":
                self.role = "satellite"
                # Timezone/location for briefings, before any spoken turn sets them.
                self._context = msg.get("context") or self._context
                SATELLITES.add(self)
                logger.info("satellite registered (%d connected)", len(SATELLITES))
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
                self._turn = asyncio.create_task(self._run_turn(utterance, self._take_spec()))
                return
            ep = self.endpointer
            # Once per pause, and one at a time: a stale one still running would slow the real STT.
            if ep.in_speech and ep.silence_frames == SPECULATE_FRAMES and (self._spec is None or self._spec[0].done()):
                self._drop_spec()
                self._spec = (asyncio.create_task(self._transcribe(ep.snapshot())), ep.voiced)
        if not self.endpointer.in_speech and time.monotonic() - self._listen_started > NO_SPEECH_TIMEOUT_S:
            await self.set_state("idle")

    def _take_spec(self) -> asyncio.Task | None:
        """The speculative transcript, if the user stayed silent since it was taken."""
        spec, self._spec = self._spec, None
        if spec and spec[1] == self.endpointer.last_voiced:
            return spec[0]
        if spec:
            spec[0].cancel()
        return None

    def _drop_spec(self) -> None:
        if self._spec:
            self._spec[0].cancel()
            self._spec = None

    async def _run_turn(self, pcm: bytes, spec: asyncio.Task | None = None) -> None:
        timing = {
            "speech_ms": len(pcm) // 2 * 1000 // SAMPLE_RATE,
            "endpoint_wait_ms": self.endpointer.cfg.end_silence_ms,
        }
        stage, input_desc, started = "STT", f"{len(pcm) // 1024} KB PCM", time.monotonic()
        try:
            text = None
            if spec:
                try:
                    text = await spec
                    timing["stt_spec"] = "hit"
                except asyncio.CancelledError:
                    raise
                except Exception:  # the speculative call failed: transcribe normally
                    timing["stt_spec"] = "failed"
            if text is None:
                text = await self._transcribe(pcm)
            timing["stt_ms"] = _ms(started)  # from the endpoint; ~0 when the speculative transcript was ready
            if not text:
                await self._fail("STT", "no speech recognized", input_desc, started)
                return
            await self.ws.send_json({"type": "transcript", "text": text})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail(stage, f"{type(exc).__name__}: {exc}", input_desc, started)
            return
        await self._respond(text, timing)

    async def _respond(self, text: str, timing: dict, mode: str = "voice", context: dict | None = None) -> None:
        """LLM + TTS for one user message, or for a prompt the server writes
        itself (the alarm briefing's live fallback)."""
        stage, input_desc, started = "LLM", f"{len(text)} chars", time.monotonic()
        try:
            # Chunked clients get one WAV per piece, synthesized while the LLM
            # is still writing; the rest get the whole reply as one WAV.
            chunker = SentenceChunker() if self._chunked else None
            speaker = asyncio.create_task(self._speak_chunks(chunker, timing)) if chunker else None
            # A soft "Hmm." if the reply is slow to start; never for briefings (alarm audio).
            slow = (asyncio.create_task(self._filler_when_slow(timing))
                    if chunker and mode != "briefing" and self._speech_end else None)
            try:
                speak_text = ""
                async for ev in stream_prompt(text, mode=mode, conversation_id=self._conversation_id,
                                             context=self._context if context is None else context):
                    if ev["type"] == "conversation":
                        self._conversation_id = ev["id"]
                        await self.ws.send_json(ev)
                    elif ev["type"] == "delta":
                        timing.setdefault("llm_ttft_ms", _ms(started))
                        if chunker:
                            chunker.feed(ev["text"])
                    elif ev["type"] == "tool":
                        await self.ws.send_json({"type": "tool", "name": ev["name"]})
                        if chunker and mode != "briefing":
                            await self._say_filler(ev["name"], timing)
                    elif ev["type"] == "done":
                        speak_text = ev["reply"]
                        if ev["timing"].get("prefill"):
                            timing["llm_prefill"] = "+".join(ev["timing"]["prefill"])
                timing["llm_ms"] = _ms(started)
                await self.ws.send_json({"type": "reply", "text": speak_text})
                if speaker:
                    chunker.close()
                    sent = await speaker
                    if sent == 0:
                        await self.set_state("idle")
                        self._timing = timing
                        self._log_timing()
                    return
            finally:
                if slow:
                    slow.cancel()
                if speaker and not speaker.done():
                    speaker.cancel()

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

    async def _speak_chunks(self, chunker: SentenceChunker, timing: dict) -> int:
        """Pull pieces from the chunker, synthesize them in order (the TTS
        server runs one at a time anyway) and send each WAV as soon as it
        exists. Returns how many were sent; -1 after a TTS failure, which it
        has already reported.

        Each piece is sized so it can be synthesized before the audio already
        sent finishes playing: budget = lead seconds / TTS real-time factor,
        in characters. Both rates are measured on every piece.
        """
        sent = 0
        rtf, chars_per_s = TTS_RTF_GUESS, TTS_CHARS_PER_S_GUESS
        max_chars = FIRST_PIECE_CHARS
        play_end = 0.0  # when the client's queue of audio runs out (our clock)
        gap_s = 0.0

        async def send(audio: bytes, started: float) -> None:
            nonlocal sent, play_end, gap_s
            if sent == 0:
                timing["tts_first_ms"] = _ms(started)
                timing["speech_end_to_audio_sent_ms"] = _ms(self._speech_end)
                # Store before sending: playback_started can arrive while send_bytes yields.
                self._timing = timing
                await self.set_state("speaking")
            now = time.monotonic()
            if sent and now > play_end:
                gap_s += now - play_end  # the client ran dry before this audio arrived
            await self.ws.send_bytes(audio)
            sent += 1
            play_end = max(play_end, now) + _wav_seconds(audio)

        try:
            if await _tts_streams(self.http):
                await chunker.settle(FIRST_PIECE_WAIT_S)
                max_chars = SentenceChunker.MAX_CHARS
            while (piece := await chunker.next_piece(max_chars)) is not None:
                started = time.monotonic()
                if await _tts_streams(self.http):
                    # Forward audio while it's generated. The voice ends each piece
                    # with its own short pause, so no trim / added pause here.
                    audio_s = 0.0
                    async for audio in _tts_stream(self.http, piece):
                        audio_s += _wav_seconds(audio)
                        await send(audio, started)
                    synth_s = time.monotonic() - started
                    streamed = True
                else:
                    audio = await self._synthesize(piece)
                    if not audio:
                        continue
                    synth_s = time.monotonic() - started
                    audio_s = _wav_seconds(audio)
                    streamed = False
                    # Swap the clip's padding for the pause its last punctuation calls for.
                    last = chunker.closed and chunker.empty
                    await send(_trim_wav(audio, lead_ms=0, pause_ms=0 if last else _pause_after(piece)), started)
                if audio_s > 0:
                    rtf = (rtf + synth_s / audio_s) / 2
                    chars_per_s = (chars_per_s + len(piece) / audio_s) / 2
                lead_s = play_end - time.monotonic()
                max_chars = int(lead_s * PIECE_SAFETY / rtf * chars_per_s)
                max_chars = max(FIRST_PIECE_CHARS, min(SentenceChunker.MAX_CHARS, max_chars))
                if streamed:
                    # Streaming starts audio fast whatever the piece length, so after the
                    # first piece cut only at sentence ends: each extra cut costs ~0.5 s of
                    # silence and a falling, end-of-sentence tone mid-sentence.
                    max_chars = SentenceChunker.MAX_CHARS
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail("TTS", f"{type(exc).__name__}: {exc}", f"chunk {sent + 1}", time.monotonic())
            return -1
        timing["tts_chunks"] = sent
        timing["gap_ms"] = round(gap_s * 1000)
        timing["tts_rtf"] = round(rtf, 2)
        if sent:
            await self.ws.send_json({"type": "audio_end", "chunks": sent})
        return sent

    async def _say_filler(self, kind: str, timing: dict) -> None:
        """At most one filler a turn, and only before any of the reply's own audio:
        chunked clients queue it ahead of the reply, back to back."""
        if self.state != "thinking" or (filler := pick_filler(kind)) is None:
            return
        self._timing = timing  # before sending: playback_started may arrive first
        timing["filler"] = kind
        timing["filler_ms"] = _ms(self._speech_end)
        await self.set_state("speaking")
        await self.ws.send_bytes(filler)

    async def _filler_when_slow(self, timing: dict) -> None:
        await asyncio.sleep(max(0.0, self._speech_end + SLOW_REPLY_S - time.monotonic()))
        await self._say_filler("_slow", timing)

    async def _synthesize(self, text: str) -> bytes | None:
        return await _tts(self.http, text)

    async def _fail(self, stage: str, message: str, input_desc: str, started: float) -> None:
        logger.error(
            "%s failed after %.1fs (input: %s) - %s", stage, time.monotonic() - started, input_desc, message
        )
        await self.ws.send_json({"type": "error", "stage": stage, "message": message})
        await self.set_state("idle")

    def _log_timing(self) -> None:
        logger.info("turn timing: %s", " ".join(f"{k}={v}" for k, v in self._timing.items()))
        self._timing = None

    async def announce(self, *, audio: list[bytes] | None = None, prompt: str = "",
                       conversation_id: str | None = None, context: dict | None = None) -> bool:
        """Server-initiated speech (alarms): an "alarm" cue, then either audio
        made ahead of time or a live reply to `prompt` (mode "briefing"). The
        session continues in `conversation_id`, so follow-up questions have the
        briefing as context. The client reports playback_ended as usual, and a
        satellite in follow-up mode starts listening right after. Only from
        idle; returns False if the session is busy."""
        if self.state != "idle" or (self._turn and not self._turn.done()):
            return False
        await self.ws.send_json({"type": "alarm"})
        self._speech_end = time.monotonic()
        self._chunked = True
        if conversation_id:
            self._conversation_id = conversation_id
        if audio:
            self._turn = asyncio.create_task(self._send_prepared(audio))
        else:
            self._turn = asyncio.create_task(self._respond(prompt, {"alarm": 1}, mode="briefing", context=context))
        return True

    async def _send_prepared(self, audio: list[bytes]) -> None:
        self._timing = {"alarm": 1, "prepared_chunks": len(audio)}
        await self.set_state("speaking")
        for wav in audio:
            await self.ws.send_bytes(wav)
        await self.ws.send_json({"type": "audio_end", "chunks": len(audio)})

    async def close(self) -> None:
        SATELLITES.discard(self)
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
    warm_fillers()
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
