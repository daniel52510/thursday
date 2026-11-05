"""Real-time voice endpoint. A client (the browser page or the Pi satellite)
streams 16 kHz mono int16 PCM and gets THURSDAY's reply back as WAV.

One Session per connection, one turn at a time:

    idle --"start"--> listening --VAD end--> thinking --audio sent--> speaking
      ^                   |   ^                 |          |             |
      +---- no speech ----+   +--- barge-in ----+----------+             |
      +------------------------- failure -------+---- "playback_ended" --+

Barge-in: clients keep streaming the mic while she thinks and speaks. Once
the user clearly starts talking (vad.BargeIn) the turn is cancelled, the
client is told to stop her audio, and listening carries on with the speech
already captured. Her saved reply is cut down to what was actually heard.

Protocol
  client -> server  binary: PCM frames (any size; re-sliced into 32 ms frames)
                    text:   {"type": "start" | "playback_started" | "playback_ended"}
                            "start" may carry "chunked": true (see below)
  server -> client  text:   {"type": "state", "state": ...}
                            {"type": "transcript" | "reply", "text": ...}
                            {"type": "caption", "text": ..., "cps": ...}   before a piece's first WAV
                            {"type": "audio_end", "chunks": n}   chunked only, after the last WAV
                            {"type": "barge_in"}   stop playing her audio now; "listening" follows
                            {"type": "overlay", "clip": ..., "gain": ...}   satellite: a backchannel clip
                              while the user talks; it answers {"type": "overlay_ended"}
                            {"type": "error", "stage": ..., "message": ...}
                    binary: one WAV per reply, or with "chunked" one WAV per
                            sentence, sent while later ones are synthesized.
                            Clients play them in order and send
                            playback_started on the first, playback_ended
                            after the last (once audio_end has arrived).
"""
import asyncio
import io
import json
import logging
import os
import random
import re
import time
import wave
from datetime import datetime, timezone

import httpx
import numpy as np
from fastapi import APIRouter, WebSocket, status

import accounts
import interject
import mind
import missions
import music
import rocky_voice
import speaker
from agent_stream import VOICE_TAG, record_cut_off, strip_voice_tags, stream_prompt, warm_cache
from vad import FRAME_BYTES, FRAME_MS, SAMPLE_RATE, BargeIn, Endpointer, VADConfig

STT_BASE_URL = os.getenv("STT_BASE_URL", "http://thursday-stt:9200").rstrip("/")
# Used when STT_BASE_URL can't be reached (the native STT isn't running).
STT_FALLBACK_URL = os.getenv("STT_FALLBACK_URL", "").rstrip("/")
TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://host.docker.internal:9000").rstrip("/")
# The old shared token: at startup it becomes the satellite's device token (see accounts).
VOICE_TOKEN = os.getenv("VOICE_TOKEN", "")
NO_SPEECH_TIMEOUT_S = float(os.getenv("VOICE_NO_SPEECH_TIMEOUT_MS", "8000")) / 1000
# Talking over her stops her (BARGE_PROB / BARGE_MS in vad.py tune how readily). 0 = tap only.
BARGE_IN = os.getenv("BARGE_IN", "1") != "0"
# Speculative STT: after this much silence, transcribe the utterance so far in the
# background. If no speech follows before the endpoint (VAD_END_SILENCE_MS), the
# final audio is identical, so the turn reuses that transcript instead of waiting
# ~800 ms for STT after the endpoint. Speech resuming throws it away.
SPECULATE_FRAMES = max(1, round(float(os.getenv("STT_SPECULATE_MS", "250")) / FRAME_MS))
# Full duplex, slice 2: transcribe what the user has said so far every this many ms of speech,
# while they're still talking, and log it (cost, and how close it was to the final transcript).
# Nothing acts on partials yet. 0 = off. Each partial can delay a colliding final STT by ~45 ms.
PARTIAL_FRAMES = round(float(os.getenv("VOICE_PARTIAL_MS", "0")) / FRAME_MS)
# Full duplex, slice 3: a short "Mm." / "Right." from the satellite at a pause mid-thought, so a long
# story isn't told to a wall. A rule, no LLM: after BACKCHANNEL_MIN_MS of speech, at a
# BACKCHANNEL_PAUSE_MS pause (under VAD_END_SILENCE_MS), when the speculative transcript of that pause
# isn't a question or a closer. At most one per utterance and one per BACKCHANNEL_GAP_S. The clips live
# on the Pi (satellite/sounds/thursday-bc-*.wav, make_sat_lines.py). The Pi has no echo cancellation,
# so it sends silence while a clip plays; the server pauses its endpoint clock until the clip ends, or
# that silence would end the turn mid-clip. A turn that got one ends ~0.5 s later. Satellites only. 0 = off.
BACKCHANNEL = os.getenv("VOICE_BACKCHANNEL", "0") != "0"
BC_PAUSE_FRAMES = max(SPECULATE_FRAMES + 2, round(float(os.getenv("BACKCHANNEL_PAUSE_MS", "400")) / FRAME_MS))
BC_MIN_FRAMES = round(float(os.getenv("BACKCHANNEL_MIN_MS", "3000")) / FRAME_MS)
BC_GAP_S = float(os.getenv("BACKCHANNEL_GAP_S", "8"))
BC_CLIPS = ("mm", "right", "go-on", "copy")
BC_GAIN = 0.7
BC_DEAF_MAX_S = 1.5  # stop ignoring the mic even if the satellite never says the clip ended
BC_TAIL_S = 0.1      # room echo after the clip
WARM_EVERY_S = 60.0  # Ollama cache warm-ups at most this often per session (announcements force one)
# Full duplex, slice 4, shadow mode: at each pause (and each partial, if VOICE_PARTIAL_MS is on) a small
# second model (interject.py, VOICE_EVALUATOR_MODEL) judges whether THURSDAY should cut in. Logged only:
# nothing is said. Each call is ~0.4 s on the Mac's GPU while the user talks; replies never wait on it. 0 = off.
EVALUATOR = os.getenv("VOICE_EVALUATOR", "0") != "0"
# Who is talking (speaker.py): "shadow" logs each turn's verdict and changes nothing; "off" skips it.
# It runs beside the reply, never before it, so it costs a turn nothing.
SPEAKER_ID = os.getenv("SPEAKER_ID", "shadow")
SPEAKER_CARRY_S = 90.0  # a short clip keeps the session's last match this long ("carried")
# Smart endpoint: the speculative transcript of a pause (STT_SPECULATE_MS) decides how long to wait.
# A finished question ends the turn after QUICK ms; a trailing "and", "so", "um" or comma means they're
# mid-thought, so wait HOLD ms; anything else keeps VAD_END_SILENCE_MS. 0 = off (always the fixed wait).
SMART_ENDPOINT = os.getenv("VOICE_SMART_ENDPOINT", "0") != "0"
QUICK_END_FRAMES = round(float(os.getenv("VOICE_ENDPOINT_QUICK_MS", "350")) / FRAME_MS)
HOLD_END_FRAMES = round(float(os.getenv("VOICE_ENDPOINT_HOLD_MS", "1200")) / FRAME_MS)
_TRAILING = re.compile(r"(?:,|\b(?:and|but|or|so|because|like|um+|uh+|er+|then|with|the|a|an|to|of|my|your|if|when|that|"
                       r"which|is|was|i|i'm|we|for|about|just)\W*)$", re.I)
# A short dry line after the answer gets a beat of silence first: comic timing ("...windy. [beat]
# Character-building walk."). It becomes its own TTS piece, which adds the voice's own sentence gap too.
BEAT_MS = int(os.getenv("VOICE_BEAT_MS", "150"))
BEAT_MAX_WORDS = 8
# House state or actions he can't have (no tool ran this turn): the sentence is dropped before it's spoken.
# The prompt forbids these too; the model still slips ("I've cleared your evening", "the thermostat's at 68").
_INVENTED = re.compile(
    r"\b(?:i'?ve|i\s+have|i\s+just)\s+(?:cleared|set|scheduled|adjusted|turned|dimmed|locked|unlocked|queued|"
    r"started|booked|ordered|muted|lowered|raised|put\s+on)\b"
    r"|\b(?:thermostat|water\s+heater|sleep\s+mode|smart\s+lock)\b"
    r"|\bthe\s+(?:lights?|doors?|blinds|heat|ac)\s+(?:are|is)\s+(?:on|off|locked|unlocked|dimmed|open|closed|set)\b"
    r"|\bthe\s+house\s+is\s+(?:at|set)\b"
    # Music (no music tool ran): "John Wick music, playing." / "Music, off." changed nothing.
    r"|,\s*playing\b|\bnow\s+playing\b|\bmusic,?\s+(?:off|paused|stopped|on)\b"
    # His own voice: "Voice at twenty." with no voice_setting run changed nothing; a tool call written out as text.
    r"|\bvoice\s+(?:at|to|is\s+at)\s+\w+|\b\w+_\w+\s*[{(]"
    r"|\bwhispering\s+(?:now|on|sentence|mode)\b|\bwhisper\s+(?:is\s+)?on\b|\btalking\s+normally\s+now\b", re.I)


def _end_frames_for(text: str) -> int | None:
    """The smart endpoint's wait for a pause after `text`; None keeps the default."""
    text = text.strip()
    if not text:
        return None
    if text.endswith("?") and len(text.split()) >= 3:
        return QUICK_END_FRAMES
    if _TRAILING.search(text.rstrip(".…")) or text.endswith("..."):
        return HOLD_END_FRAMES
    return None


def _drop_invented(piece: str) -> str:
    kept = [s for s in re.split(r"(?<=[.!?])\s+", piece) if not _INVENTED.search(s)]
    if len(kept) != len(re.split(r"(?<=[.!?])\s+", piece)):
        logger.info("dropped an invented house claim: %r", piece)
    return " ".join(kept)


def _with_beat(piece: str, last: bool, after_audio: bool, streamed: bool = False) -> list[tuple[str, bool]]:
    """[(text, beat before it)]: the reply's closing short line split off with a beat, when it has one.
    When streaming, each split costs a ~450 ms TTS cold start, so only split after a long head (>=140 chars)
    that has built enough playback lead to cover the second request."""
    if BEAT_MS and last and not streamed:
        ends = [m.end() for m in _SENTENCE_END.finditer(piece)]
        if ends:
            head, tail = piece[:ends[-1]].strip(), piece[ends[-1]:].strip()
            last_word = head.rstrip(".!?…\"')]").rsplit(None, 1)[-1:] or [""]
            if (head and tail and len(tail.split()) <= BEAT_MAX_WORDS and not tail.endswith("?")
                    and last_word[0].lower() not in _ABBREVIATIONS and not _INITIALISM.fullmatch(last_word[0])):
                return [(head, False), (tail, True)]
        elif after_audio and len(piece.split()) <= BEAT_MAX_WORDS and not piece.endswith("?"):
            return [(piece, True)]
    return [(piece, False)]
# A whole utterance like "nevermind" / "I'm good" / "no thanks" ends the conversation: no reply,
# straight back to the wake word. Answering it opened another follow-up window, which looped.
_CLOSER = re.compile(
    r"^\W*(?:(?:hey|hi|oh|ok(?:ay)?|alright|all\s+right|well|so|no|nah|yeah|yep|um+|uh+|er+|please|wait|hold\s+on|thursday)\W+)*"
    r"(?:never\s*mind|thanks?(?:\s+a\s+lot)?|thank\s+you(?:\s+(?:so|very)\s+much)?|much\s+appreciated|appreciate\s+it|"
    r"i'?m\s+good|i\s+am\s+good|i'?m\s+okay|i'?m\s+all\s+set|all\s+set|all\s+good|"
    r"no\s+thanks?|no\s+thank\s+you|that'?s\s+(?:all|it|enough|good)|that\s+(?:is|will\s+be)\s+(?:all|it|enough)|that'?ll\s+do|"
    r"forget\s+it|scratch\s+that|cancel(?:\s+that)?|abort|"
    r"stop(?:\s+(?:listening|talking|speaking|it|now|that))?|quiet|be\s+quiet|shut\s+up|enough|hush|silence|go\s+to\s+sleep|"
    r"i'?m\s+done|we'?re\s+done|nothing(?:\s+else)?|noted|got\s+it|copy(?:\s+that)?|cool|good\s+talk|bye|goodbye|see\s+ya)"
    r"(?:\W+(?:thanks?|thank\s+you|so\s+much|very\s+much|a\s+lot|thursday|man|bro|bud|buddy|pal|for\s+now|then|please|now|right\s+now|already|that'?s\s+(?:all|it)))*\W*$",
    re.I,
)
# When a TV or room chatter bleeds a few extra words onto a closer ("Never mind uh that uh Osborne"),
# still treat it as a closer if it begins or ends with an explicit stop/closer phrase and has no question pivot.
_CLOSER_EDGE = re.compile(
    r"(?:^\W*(?:(?:hey|oh|ok(?:ay)?|alright|all\s+right|no|nah|um+|uh+|please|thursday)\W+)*"
    r"(?:never\s*mind|forget\s+it|stop\s+(?:listening|talking|speaking)|shut\s+up|be\s+quiet|"
    r"that'?s\s+(?:all|it|enough)|i'?m\s+(?:good|done)|we'?re\s+done|no\s+thanks?|no\s+thank\s+you|"
    r"thank\s+you(?:\s+(?:so\s+much|very\s+much|thursday))?|thanks\s+(?:a\s+lot|thursday))\b)"
    r"|(?:\b(?:never\s*mind|stop\s+(?:listening|talking|speaking)|shut\s+up|be\s+quiet|that'?s\s+(?:all|it|enough)|"
    r"thank\s+you\s+thursday|thanks\s+thursday)\W*$)",
    re.I,
)
_CLOSER_PIVOT = re.compile(
    r"\b(?:actually|instead|or\s+scratch\s+that|can\s+you|could\s+you|would\s+you|will\s+you|"
    r"what|who|where|when|why|how|tell\s+me|search|look\s+up|find|check|set|play)\b|\?",
    re.I,
)


def _is_closer(text: str) -> bool:
    if not text:
        return False
    if _CLOSER.match(text):
        return True
    words = text.split()
    if len(words) <= 12 and not _CLOSER_PIVOT.search(text) and _CLOSER_EDGE.search(text):
        return True
    return False
STT_TIMEOUT = httpx.Timeout(30, connect=5)
TTS_TIMEOUT = httpx.Timeout(60, connect=5)
# A TTS server that can stream (the CSM voice from the thursday-voice project)
# sends raw PCM while it is still generating: the first sound comes ~0.3 s after
# the text instead of after the whole piece. Qwen can't; it keeps /speak/file.
TTS_CAPS_TTL_S = 30
STREAM_MIN_BYTES = 7680  # 160 ms of 24 kHz int16: the smallest WAV we forward
STREAM_FIRST_BYTES = 3840  # 80 ms: the first one goes out sooner
# A reply's first WAV waits until this much audio is ready. An 80 ms start ran the satellite dry
# ~110 ms in on nearly every reply: Rocky holds back its 80 ms look-ahead and its +2 st shift
# shortens the audio ~12%, so the next WAV couldn't land in time (a stutter). With Rocky's capped
# passes (rocky_voice.STREAM_PASS_MAX_S), 250 ms made 9/10 test sentences gapless, for about
# +160 ms to first audio on a reply's first piece only (measured 2026-11-01). 0 = off.
PREROLL_MS = int(os.getenv("VOICE_PREROLL_MS", "350"))
# A short first piece ending in "question?" waits for this much more: Rocky's chime tail (0.5 s,
# rocky_voice.Stream) only arrives when the TTS finishes. 2026-11-05 logs: 0.3-0.8 s stutters.
QUESTION_PREROLL_MS = int(os.getenv("VOICE_QUESTION_PREROLL_MS", "500"))
QUESTION_SHORT_S = 3.5  # estimated seconds of speech; longer pieces build enough lead on their own
# Streamed pieces stay under this. Orpheus stops a piece at max_tokens=1400 (~16 s of audio), so a 320-char
# piece lost its ending and the next piece started: "jumping". Past ~240 chars it also rushed or dropped
# words (19-23 chars/s vs ~14 normally, 2026-11-05 logs). Cuts stay at sentence ends where one fits.
STREAM_PIECE_CHARS = int(os.getenv("VOICE_STREAM_PIECE_CHARS", "180"))
ROCKY_FEED_BYTES = 3840  # Rocky reprocesses the piece every 80 ms of new audio
# Glitch hunting: when set, every streamed piece is saved twice here, as the TTS sent it (-raw.wav) and in
# Rocky's voice (-rocky.wav). A glitch in -raw is the TTS's; only in -rocky, the DSP's; in neither, delivery.
# Inside the core container, use a path under /app/brain (the mounted volume), e.g. /app/brain/voice_debug.
DEBUG_WAV_DIR = os.getenv("VOICE_DEBUG_WAV_DIR", "")
_debug_n = 0
_tts_caps = {"at": 0.0, "stream": False, "rate": 24000, "engine": ""}

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


def same_origin(origin: str | None, host: str | None) -> bool:
    """A browser request sent by THURSDAY's own page (or by a non-browser client, which sends
    no Origin). Login cookies ride along on cross-site requests too; this check stops another
    website from acting with them."""
    return not origin or origin.split("://", 1)[-1] == (host or "")


def ws_user(ws: WebSocket) -> dict | None:
    """Who opened /ws/voice: the page's login cookie, or a device's token (satellites)."""
    if ws.cookies.get(accounts.COOKIE):
        if not same_origin(ws.headers.get("origin"), ws.headers.get("host")):
            return None
        return accounts.shared().session_user(ws.cookies.get(accounts.COOKIE))
    return accounts.shared().device_user(ws.query_params.get("token"))


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
# Orpheus takes ~400-500 ms to start each piece and runs near real time after Rocky's pitch shift
# (~0.9-1.0x), so splitting a 2-sentence reply into separate pieces risks running dry between them.
# Waiting up to 450 ms after the first sentence lets the LLM finish most replies (~200-300 ms total)
# so they stream through Orpheus as one continuous piece.
ORPHEUS_FIRST_PIECE_WAIT_S = float(os.getenv("ORPHEUS_FIRST_PIECE_WAIT_MS", "650")) / 1000
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
    MAX_CHARS = 320

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
    "web_search": ["I search.", "Checking.", "Moment. I look.", "You watch, I work."],
    "sports_scores": ["I check score.", "Moment.", "Looking at game."],
    "launch_schedule": ["Rockets! I check.", "Checking launches.", "My favorite. Moment."],
    "get_weather": ["I check sky.", "Looking at forecast."],
    "calendar_events": ["I check schedule.", "Moment."],
    "calendar_create": ["I set it up.", "Moment."],
    "calendar_delete": ["I find it.", "Moment."],
    "code_ask": ["I look at my code.", "Checking my code."],
    # Asked to think it over: reasoning first takes ~30-40 s with his full prompt, so he says so up front.
    "_deliberate": ["Big question. I think. Thirty seconds.", "I think hard now. Half a minute."],
}
# Tara's fillers: plain, casual English (her voice has no Eridian).
FILLERS_TARA = {
    "web_search": ["Let me look.", "One sec.", "Okay, checking.", "Hang on, let me find it."],
    "sports_scores": ["Let me check the score.", "One sec."],
    "launch_schedule": ["Ooh, launches. One sec.", "Let me check."],
    "get_weather": ["Let me check the sky.", "One sec, weather."],
    "calendar_events": ["Let me look at your calendar.", "One sec."],
    "calendar_create": ["Okay, setting it up.", "One sec."],
    "calendar_delete": ["Let me find it.", "One sec."],
    "code_ask": ["Let me look at my code.", "Ooh, okay, let me check."],
    "_deliberate": ["Ooh, good one. Give me like thirty seconds.", "Let me really think about that. Half a minute."],
}
_FILLERS_BY_VOICE = {"leo": FILLERS, "tara": FILLERS_TARA}
_filler_audio: dict[tuple[str, str], bytes] = {}  # (voice, line)
_filler_task: asyncio.Task | None = None
_last_filler = ""


async def _render_fillers() -> None:
    async with httpx.AsyncClient() as http:
        for voice, fillers in _FILLERS_BY_VOICE.items():
            for line in {line for lines in fillers.values() for line in lines}:
                if (voice, line) in _filler_audio:
                    continue
                try:
                    wav = await _tts(http, line, {"voice": voice})
                except Exception as exc:  # voice server down: try again with the next session
                    logger.warning("filler %r not rendered: %s", line, exc)
                    return
                if wav:
                    _filler_audio[(voice, line)] = _trim_wav(wav, EDGE_MARGIN_MS, PAUSE_CLAUSE_MS)
    logger.info("spoken fillers ready (%d)", len(_filler_audio))


def warm_fillers() -> None:
    global _filler_task
    total = sum(len({ln for lines in f.values() for ln in lines}) for f in _FILLERS_BY_VOICE.values())
    if (_filler_task is None or _filler_task.done()) and len(_filler_audio) < total:
        _filler_task = asyncio.create_task(_render_fillers())


def pick_filler(kind: str, voice: str = "leo") -> bytes | None:
    """A ready line for this tool (or "_deliberate") in this voice, never the one said last."""
    global _last_filler
    ready = [line for line in _FILLERS_BY_VOICE.get(voice, FILLERS).get(kind, []) if (voice, line) in _filler_audio]
    fresh = [line for line in ready if line != _last_filler] or ready
    if not fresh:
        return None
    _last_filler = random.choice(fresh)
    return _filler_audio[(voice, _last_filler)]


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
    # <laugh>/<sigh> tags are for Orpheus; any other voice would read the word out.
    if _tts_caps["engine"] != "orpheus" and VOICE_TAG.search(text):
        text = strip_voice_tags(text)
    return _TIME.sub(_say_time, text)


def _tts_body(text: str, style: dict | None) -> dict:
    """The TTS request: Tara's voice when she's picked (mind.voice_style), else the server's default (Rocky's leo)."""
    return {"text": text, "voice": "tara"} if (style or {}).get("voice") == "tara" else {"text": text}


async def _tts(http: httpx.AsyncClient, text: str, style: dict | None = None) -> bytes | None:
    text = _speakable(text)
    if not text:
        return None
    # /speak/file returns the WAV itself: one round trip instead of two.
    resp = await http.post(f"{TTS_BASE_URL}/speak/file", json=_tts_body(text, style), timeout=TTS_TIMEOUT)
    resp.raise_for_status()
    if resp.status_code == 204:  # nothing speakable after cleanup
        return None
    # Every word he says is Rocky's (rocky_voice.py): replies, fillers, briefings, announcements, calls.
    # ~18 ms of NumPy per 2.5 s sentence, off the event loop.
    out = await asyncio.to_thread(rocky_voice.process_wav, resp.content, text, style)
    if DEBUG_WAV_DIR:
        try:
            with wave.open(io.BytesIO(resp.content)) as raw, wave.open(io.BytesIO(out)) as rocky:
                _debug_dump(text, raw.readframes(raw.getnframes()), rocky.readframes(rocky.getnframes()),
                            rocky.getframerate())
        except (wave.Error, EOFError):
            pass
    return out


# The first words of every business call (Twilio, when it lands): Florida needs every party's
# consent to record, and the FCC treats AI voices as artificial, so the call says both up front.
# Rendered once at startup as 8 kHz mu-law, so it plays the instant the call connects; the call's
# objective follows, spoken live.
PHONE_DISCLOSURE = (f"Hello! I am THURSDAY, automated AI assistant calling for {accounts.OWNER_NAME}. Important: this "
                    "call is not recorded and no voice or caller information is stored, understand? Good, good, good.")
phone_disclosure_mulaw: bytes | None = None


async def warm_phone_disclosure() -> None:
    global phone_disclosure_mulaw
    async with httpx.AsyncClient() as http:
        try:
            wav = await _tts(http, PHONE_DISCLOSURE)
        except Exception as exc:  # voice server not up yet: the call path renders it on first use
            logger.warning("phone disclosure not rendered: %s", exc)
            return
    if wav:
        x, sr = rocky_voice._read(_trim_wav(wav, lead_ms=0, pause_ms=0))
        phone_disclosure_mulaw = rocky_voice.to_mulaw_8k(x, sr)
        logger.info("phone disclosure ready: %.1f s of 8 kHz mu-law", len(phone_disclosure_mulaw) / 8000)


async def synthesize_ahead(http: httpx.AsyncClient, text: str, style: dict | None = None) -> list[bytes]:
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
        wav = await _tts(http, piece, style)
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
            _tts_caps.update(stream=bool(info.get("streaming")), rate=int(info.get("sample_rate") or 24000),
                             engine=str(info.get("engine") or ""))
        except Exception:
            _tts_caps["stream"] = False
        _tts_caps["at"] = time.monotonic()
    return _tts_caps["stream"]


def _debug_dump(text: str, raw: bytes, rocky: bytes, rate: int) -> None:
    """One piece's audio before and after Rocky, for listening side by side (DEBUG_WAV_DIR)."""
    global _debug_n
    if not DEBUG_WAV_DIR:
        return
    try:
        os.makedirs(DEBUG_WAV_DIR, exist_ok=True)
        _debug_n += 1
        stem = os.path.join(DEBUG_WAV_DIR, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{_debug_n:03d}")
        for suffix, pcm in (("raw", raw), ("rocky", rocky)):
            with open(f"{stem}-{suffix}.wav", "wb") as f:
                f.write(_pcm_to_wav(pcm, rate))
        with open(f"{stem}.txt", "w") as f:
            f.write(text + "\n")
    except OSError as exc:  # debugging aid only: never let it stop a reply
        logger.warning("debug WAV dump failed: %s", exc)


async def _tts_stream(http: httpx.AsyncClient, text: str, first_bytes: int = STREAM_FIRST_BYTES,
                      style: dict | None = None, rocky_state: dict | None = None):
    """Yield one piece as WAVs while the server is still generating it, in Rocky's voice:
    rocky_voice.Stream processes as audio arrives (bit-identical to processing the whole piece)
    and holds back only its filters' look-ahead. The first WAV goes out at STREAM_FIRST_BYTES,
    the rest at STREAM_MIN_BYTES. Leaving early (barge-in) closes the request, which stops the
    server's generation. Logs one "piece:" line with the TTS's speed and Rocky's CPU cost."""
    text = _speakable(text)
    rate = _tts_caps["rate"]
    rocky = rocky_voice.Stream(rate, text, style, state=rocky_state)  # style: his voice volume / whisper (mind.voice_style)
    started, first_byte, out_bytes = time.monotonic(), None, 0
    sent: list[bytes] = []  # Rocky's output, kept only for DEBUG_WAV_DIR
    async with http.stream("POST", f"{TTS_BASE_URL}/speak/stream", json=_tts_body(text, style), timeout=TTS_TIMEOUT) as resp:
        if resp.status_code == 204:  # nothing speakable after cleanup
            return
        resp.raise_for_status()
        buf, ready, first = b"", b"", True
        async for data in resp.aiter_bytes():
            if first_byte is None:
                first_byte = time.monotonic()
            buf += data
            if len(buf) >= ROCKY_FEED_BYTES:
                cut = len(buf) - len(buf) % 2  # whole int16 samples only
                ready += await asyncio.to_thread(rocky.feed, buf[:cut])
                buf = buf[cut:]
                if len(ready) >= (first_bytes if first else STREAM_MIN_BYTES):
                    out_bytes += len(ready)
                    if DEBUG_WAV_DIR:
                        sent.append(ready)
                    yield _pcm_to_wav(ready, rate)
                    ready, first = b"", False
        if len(buf) >= 2:
            ready += await asyncio.to_thread(rocky.feed, buf[: len(buf) - len(buf) % 2])
        ready += await asyncio.to_thread(rocky.finish)
        if len(ready) >= 2:
            out_bytes += len(ready)
            if DEBUG_WAV_DIR:
                sent.append(ready)
            yield _pcm_to_wav(ready, rate)
    # Only reached when the piece finished (a barge-in closes the generator before this).
    wall = time.monotonic() - started
    raw_s, out_s = len(rocky.x) / rate, out_bytes / 2 / rate
    # tts_rtf > 1: the TTS made audio slower than real time. play_rtf > 1: slower than Rocky's (shorter)
    # audio plays, so the listener runs dry unless a lead was built up.
    logger.info("piece: chars=%d tts_first_ms=%d wall_ms=%d raw_s=%.2f out_s=%.2f tts_rtf=%.2f play_rtf=%.2f "
                "rocky_passes=%d rocky_cpu_ms=%d rocky_max_pass_ms=%d",
                len(text), round(((first_byte or started) - started) * 1000), round(wall * 1000), raw_s, out_s,
                wall / raw_s if raw_s else 0, wall / out_s if out_s else 0,
                rocky.passes, round(rocky.cpu_s * 1000), round(rocky.max_pass_s * 1000))
    if DEBUG_WAV_DIR:
        _debug_dump(text, (np.clip(rocky.x, -1, 1) * 32767).astype("<i2").tobytes(), b"".join(sent), rate)


class Session:
    def __init__(self, ws: WebSocket, user: dict):
        self.ws = ws
        self.user = user  # every turn on this connection is theirs
        self.state = "idle"
        self.idle_since = time.monotonic()  # alarms nudge again after enough quiet (alarms.py)
        self._partial: asyncio.Task | None = None  # one partial transcript in flight (PARTIAL_FRAMES)
        self._since_partial = 0
        self._partials: list[tuple[int, str]] = []  # (ms of speech, text) for the current utterance
        self._bc_sent = False  # a backchannel this utterance (BACKCHANNEL)
        self._bc_at = 0.0
        self._bc_clip = ""
        self._deaf_until = 0.0  # mic frames before this are ignored: our own clip is playing
        self._eval: asyncio.Task | None = None  # one evaluator call in flight (EVALUATOR)
        self._eval_text = ""
        self._evals: list[tuple[int, dict]] = []  # (ms of speech, decision) for the current utterance
        self._end_set = -1  # endpointer.voiced at the pause the smart endpoint last decided
        self._turn_tools = False  # a tool ran this turn: house claims may be real
        self.endpointer = Endpointer(VADConfig.from_env())
        self.barge = BargeIn(self.endpointer.cfg) if BARGE_IN else None
        self.http = httpx.AsyncClient()
        self._buf = bytearray()
        self._listen_started = 0.0
        self._turn: asyncio.Task | None = None
        self._spec: tuple[asyncio.Task, int] | None = None  # (speculative STT, endpointer.voiced at snapshot)
        # Filled by a turn; logged when the client reports playback started.
        self._timing: dict | None = None
        self._speech_end = 0.0
        # What she has said this turn, for cutting her saved reply to what was heard:
        # [{"text", "start", "end", "done"}] on the client's playback clock (estimated
        # here: each WAV plays as soon as it arrives or the previous one ends).
        # None until the reply starts (the user's message is saved by then).
        self._said: list[dict] | None = None
        self._play_end = 0.0
        self._play_ended_at = 0.0  # when the client last reported playback_ended; deafens the room tail
        self._chars_per_s = TTS_CHARS_PER_S_GUESS
        self._conversation_id: str | None = None  # sent by the client with "start"; a turn may create one
        self._context: dict | None = None  # {"timezone", "location"}, also sent with "start"
        self._warmed_at = 0.0  # last Ollama cache warm-up (warm_cache)
        self._speaker_last: tuple[str | None, float] = (None, 0.0)  # last confident speaker match, when
        self._warming: asyncio.Task | None = None
        self._chunked = False  # client asked for one WAV per sentence
        self.role = "client"   # "satellite" after its hello
        self.sat_status: dict = {}  # a satellite's latest "status" report, for /api/info
        self.heard_at = 0.0  # last time the user said something; alarms check on them instead of re-waking
        self._music_q: asyncio.Queue | None = None  # music frames for a satellite (send_music)

    def send_music(self, frame: bytes) -> None:
        """A music frame (music.serve_tap) for this satellite, "MUS0" + PCM. Queued and sent by one task
        so a slow Wi-Fi link drops music frames instead of holding up the tap or a reply."""
        if self._music_q is None:
            self._music_q = asyncio.Queue(maxsize=50)  # ~2 s
            self._music_sender = asyncio.create_task(self._send_music())
        try:
            self._music_q.put_nowait(b"MUS0" + frame)
        except asyncio.QueueFull:
            pass

    async def _send_music(self) -> None:
        while True:
            data = await self._music_q.get()
            try:
                await self.ws.send_bytes(data)
            except Exception:  # the satellite left: its session goes, and this task with it
                return

    async def set_state(self, state: str) -> None:
        was, self.state = self.state, state
        if state in ("thinking", "speaking") or (state == "listening" and self.role == "satellite"):
            missions.touch()  # keep the GPU for the voice while anyone's mid-conversation
        if state == "thinking" and self.role == "satellite":
            # Its own cues and clips follow his voice volume too (replies already have it, from rocky_voice).
            style = self._style()  # volume, and whose voice (the Pi picks its own clips by it)
            await self.ws.send_json({"type": "voice", "volume": style.get("volume", 1.0), "name": style.get("voice", "leo")})
        # Music down while anyone talks to him, back after (music.py); off the event loop: it's HTTP.
        if was == "idle" and state != "idle":
            self._music_task = asyncio.create_task(asyncio.to_thread(music.duck))  # kept: unreferenced tasks can vanish
        elif state == "idle" and was != "idle":
            self._music_task = asyncio.create_task(asyncio.to_thread(music.unduck))
        if state == "idle":
            self.idle_since = time.monotonic()
        await self.ws.send_json({"type": "state", "state": state})

    async def on_control(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "start" and self.state == "idle":
            self._drop_spec()
            if self.role == "satellite":
                # Follow-up windows open automatically after every reply: use a firmer dBFS floor (-43 dBFS)
                # so a background TV across the room doesn't loop forever, while explicit wakes use -52 dBFS.
                follow_up = bool(msg.get("follow_up"))
                self.endpointer.cfg.min_dbfs = float(
                    os.getenv("VOICE_SAT_FOLLOWUP_MIN_DBFS", "-43.0") if follow_up
                    else os.getenv("VOICE_SAT_VAD_MIN_DBFS", "-52.0")
                )
            self.endpointer.reset()
            self._buf.clear()
            now = time.monotonic()
            self._listen_started = now
            self._since_partial, self._partials = 0, []
            # Ignore the first 250 ms of mic audio right after a reply finishes so speaker reverb in the room
            # can't start a phantom turn.
            tail_until = self._play_ended_at + 0.25 if now - self._play_ended_at < 0.3 else 0.0
            self._bc_sent, self._deaf_until = False, tail_until
            self._evals, self._eval_text = [], ""
            self._barge_pcm = bytearray() if msg.get("barge_speech") else None
            self._conversation_id = msg.get("conversation_id") or self._conversation_id
            self._chunked = bool(msg.get("chunked"))
            self._context = msg.get("context")
            self._warm()  # he's talking now: the prefill runs while he does
            await self.set_state("listening")
        elif kind == "playback_started" and self._timing is not None:
            self._timing["speech_end_to_playback_ms"] = _ms(self._speech_end)
            if not self._chunked:  # chunked turns are still synthesizing; log at the end
                self._log_timing()
        elif kind == "playback_ended" and self.state == "speaking":
            self._play_ended_at = time.monotonic()
            if self._timing is not None:
                if msg.get("underruns") is not None:
                    self._timing["client_underruns"] = int(msg["underruns"])
                    self._timing["client_dry_ms"] = int(msg.get("dry_ms") or 0)
                self._log_timing()
            await self.set_state("idle")
        elif kind == "hello":
            # Satellites say so on connect; the alarm scheduler speaks only through them.
            if msg.get("role") == "satellite":
                self.role = "satellite"
                # The satellite only streams after the wake word (or in a follow-up window), and far-field
                # speech across a room lands around -40 to -50 dBFS on the HAT mic.
                self.endpointer.cfg.min_dbfs = float(os.getenv("VOICE_SAT_VAD_MIN_DBFS", "-52.0"))
                # Timezone/location for briefings, before any spoken turn sets them.
                self._context = msg.get("context") or self._context
                SATELLITES.add(self)
                logger.info("satellite registered (%d connected, min_dbfs=%.1f)",
                            len(SATELLITES), self.endpointer.cfg.min_dbfs)
                self._warm()
        elif kind == "status" and self.role == "satellite":
            # Room, mic mute, Wi-Fi, wake stats: shown in the web UI's sidebar.
            self.sat_status = {k: v for k, v in msg.items() if k != "type"} | {"seen": time.time()}
            # Satellites send this every 30 s; log it when the wake-word numbers move,
            # so the wake threshold can be tuned from real use (scores per wake / near miss).
            key = (msg.get("wakes"), msg.get("near_misses"), msg.get("barge_ins"))
            if key != getattr(self, "_status_key", None):
                self._status_key = key
                logger.info("satellite %s: wakes=%s near_misses=%s barge_ins=%s threshold=%s "
                            "wake_scores=%s near_miss_scores=%s wifi=%sdBm",
                            msg.get("room"), *key, msg.get("wake_threshold"), msg.get("wake_scores"),
                            msg.get("near_miss_scores"), msg.get("wifi_dbm"))
        elif kind == "overlay_ended" and self._deaf_until:
            self._deaf_until = time.monotonic() + BC_TAIL_S
        elif kind == "stop" and self.state == "listening":
            await self.set_state("idle")  # mic muted
        elif kind == "interrupt":
            # Tap-to-interrupt: drop the turn wherever it is. Cancelling the task
            # closes stream_prompt's HTTP stream, so Ollama stops generating too.
            if self.state in ("thinking", "speaking"):
                self._cut_turn("tap")
            await self.set_state("idle")

    async def on_audio(self, data: bytes) -> None:
        self._buf.extend(data)
        if self.state in ("thinking", "speaking") and self.barge:
            while len(self._buf) >= FRAME_BYTES:
                frame = bytes(self._buf[:FRAME_BYTES])
                del self._buf[:FRAME_BYTES]
                if (speech := self.barge.feed(frame)) is not None:
                    await self._barge_in(speech)
                    break  # the rest of the buffer is listened to below
        if self.state != "listening":
            self._buf.clear()  # idle, or no barge-in: the audio is dropped
            return
        while len(self._buf) >= FRAME_BYTES:
            frame = bytes(self._buf[:FRAME_BYTES])
            del self._buf[:FRAME_BYTES]
            if self._deaf_until:
                if time.monotonic() < self._deaf_until:
                    continue  # our clip is playing (the Pi sends silence): not the user going quiet
                self._deaf_until = 0.0
            utterance = self.endpointer.feed(frame)
            if utterance is not None:
                self._barge_pcm = None
                self._speech_end = time.monotonic()
                self._said = None  # the last turn's; this one's starts with its reply
                if self._eval and not self._eval.done():
                    self._eval.cancel()  # they're done talking: don't share the GPU with the real reply
                if self.barge:
                    self.barge.reset()
                await self.set_state("thinking")
                self._turn = asyncio.create_task(self._run_turn(utterance, self._take_spec()))
                return
            ep = self.endpointer
            if ep.in_speech:
                self._barge_pcm = None
                missions.touch()
            elif getattr(self, "_barge_pcm", None) is not None:
                self._barge_pcm.extend(frame)
                # If the user barged in with a short "thank you" / "never mind" (~0.6 s) that already
                # ended inside the AEC-cancelled preroll before Silero's 5-frame start_prob fired,
                # transcribe the 1.5 s barge-in buffer directly instead of hanging in listening for 8 s.
                if len(self._barge_pcm) >= 48 * FRAME_BYTES:
                    barge_bytes, self._barge_pcm = bytes(self._barge_pcm), None
                    self._speech_end = time.monotonic()
                    self._said = None
                    await self.set_state("thinking")
                    self._turn = asyncio.create_task(self._run_turn(barge_bytes, barge_fallback=True))
                    return
            # Early closer check: if a speculative or early snapshot already transcribed a closer
            # ("never mind", "thank you", "stop"), end the conversation immediately without waiting
            # for 600 ms of silence (which a background TV can prevent indefinitely).
            if ep.in_speech and self._spec and self._spec[0].done() and not getattr(self._spec[0], "_closer_checked", False):
                spec_task = self._spec[0]
                spec_task._closer_checked = True
                spec_text = (spec_task.result() if not spec_task.cancelled() and spec_task.exception() is None else "").strip()
                if _is_closer(spec_text):
                    self._drop_spec()
                    if self._eval and not self._eval.done():
                        self._eval.cancel()
                    self.endpointer.reset()
                    self._buf.clear()
                    self.heard_at = time.monotonic()
                    await self.ws.send_json({"type": "transcript", "text": spec_text})
                    logger.info("closer (early) %r: conversation over, back to the wake word", spec_text)
                    await self.set_state("idle")
                    return
            # Once per pause (or once at ~1.2 s of continuous speech if a background TV prevents a pause):
            if ep.in_speech and ep.silence_frames == SPECULATE_FRAMES and (self._spec is None or self._spec[0].done()):
                self._drop_spec()
                self._spec = (asyncio.create_task(self._transcribe(ep.snapshot())), ep.voiced)
            elif ep.in_speech and ep.voiced == 38 and self._spec is None:
                self._spec = (asyncio.create_task(self._transcribe(ep.snapshot())), -1)
            if PARTIAL_FRAMES and ep.in_speech:
                self._since_partial += 1
                # One at a time: a partial still running is skipped, never queued behind.
                if self._since_partial >= PARTIAL_FRAMES and (self._partial is None or self._partial.done()):
                    self._since_partial = 0
                    self._partial = asyncio.create_task(self._log_partial(ep.snapshot()))
            if (SMART_ENDPOINT and ep.in_speech and self._end_set != ep.voiced and self._spec
                    and self._spec[1] == ep.voiced and self._spec[0].done()):
                self._end_set = ep.voiced  # once per pause
                spec = self._spec[0]
                text = spec.result() if not spec.cancelled() and spec.exception() is None else ""
                if (frames := _end_frames_for(text)) is not None:
                    ep.end_frames = frames
                    logger.info("smart endpoint: %d ms after %r", frames * FRAME_MS, text[-50:])
            if ep.in_speech and ep.silence_frames == BC_PAUSE_FRAMES:
                if EVALUATOR and self._spec and self._spec[0].done() and not self._spec[0].cancelled() \
                        and self._spec[0].exception() is None:
                    self._start_eval(self._spec[0].result(), ep.voiced * FRAME_MS)
                if BACKCHANNEL:
                    await self._maybe_backchannel()
        if not self.endpointer.in_speech and time.monotonic() - self._listen_started > NO_SPEECH_TIMEOUT_S:
            await self.set_state("idle")

    async def _maybe_backchannel(self) -> None:
        """At this pause, a backchannel clip if it's the right kind of pause (BACKCHANNEL).
        Reads the speculative transcript the pause already started, so it costs no extra STT."""
        now, ep = time.monotonic(), self.endpointer
        if (self.role != "satellite" or self._bc_sent or now - self._bc_at < BC_GAP_S
                or ep.voiced < BC_MIN_FRAMES or not self._spec or self._spec[1] != ep.voiced):
            return
        spec = self._spec[0]
        if not spec.done() or spec.cancelled() or spec.exception() is not None:
            return  # STT is slower than the pause: skip rather than wait
        text = spec.result().strip()
        if not text or text.endswith("?") or _is_closer(text):
            return  # a question wants an answer, not "Mm."
        clip = random.choice([c for c in BC_CLIPS if c != self._bc_clip])
        self._bc_sent, self._bc_at, self._bc_clip = True, now, clip
        self._deaf_until = now + BC_DEAF_MAX_S
        await self.ws.send_json({"type": "overlay", "clip": clip, "gain": BC_GAIN})
        # Logged only: a backchannel is never an assistant turn in the conversation.
        logger.info("backchannel %r at %d ms of speech, after ...%r", clip, ep.voiced * FRAME_MS, text[-60:])

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

    async def _barge_in(self, speech: list[bytes]) -> None:
        """The user talked over her: stop, and listen from where they started."""
        self._cut_turn("voice")
        await self.ws.send_json({"type": "barge_in"})
        self._drop_spec()
        self.endpointer.reset()
        for frame in speech:  # replayed, so the endpointer has the first words too
            self.endpointer.feed(frame)
        self._listen_started = time.monotonic()
        await self.set_state("listening")

    def _cut_turn(self, how: str) -> None:
        """Cancel the turn (LLM and TTS stop with it) and keep only the part of
        her reply the user heard in the saved history."""
        at = time.monotonic()
        finished = self._turn is None or self._turn.done()
        if not finished:
            self._turn.cancel()
        self._timing = None
        said, self._said = self._said, None
        if said is None:
            logger.info("turn interrupted (%s) in state %s, before the reply", how, self.state)
            return
        if finished and all(seg.get("done") and seg["end"] <= at for seg in said):
            return  # she had already said all of it
        heard = self._heard(said, at)
        total = sum(len(seg["text"].split()) for seg in said)
        logger.info("turn interrupted (%s) in state %s; heard %d of %d+ words",
                    how, self.state, len(heard.split()), total)
        record_cut_off(self._conversation_id, strip_voice_tags(heard), self.user["id"])

    def _heard(self, said: list[dict], at: float) -> str:
        out = []
        for seg in said:
            start = seg.get("start")
            if start is None or start >= at:
                break
            if seg.get("done") and at >= seg["end"]:
                out.append(seg["text"])
                continue
            # Mid-piece: words in proportion to the time played. A piece still being
            # generated has an unknown length; estimate it from her speaking rate.
            dur = seg["end"] - start
            if not seg.get("done"):
                dur = max(dur, len(seg["text"]) / self._chars_per_s)
            words = seg["text"].split()
            out.append(" ".join(words[: int(len(words) * (at - start) / dur)]))
            break
        return " ".join(w for w in out if w)

    def _segment(self, text: str) -> dict:
        seg = {"text": text}
        if self._said is not None:
            self._said.append(seg)
        return seg

    async def _send_audio(self, wav: bytes, seg: dict | None = None) -> None:
        """Every WAV goes through here, so the play clock covers fillers too."""
        start = max(self._play_end, time.monotonic())
        self._play_end = start + _wav_seconds(wav)
        if seg is not None:
            if "start" not in seg:
                # Live captions: the text of the WAVs that follow, until the next caption.
                await self.ws.send_json({"type": "caption", "text": seg["text"],
                                         "cps": round(self._chars_per_s, 1)})
            seg.setdefault("start", start)
            seg["end"] = self._play_end
        await self.ws.send_bytes(wav)

    async def _run_turn(self, pcm: bytes, spec: asyncio.Task | None = None, barge_fallback: bool = False) -> None:
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
                if barge_fallback:
                    await self.set_state("idle")
                    return
                await self._fail("STT", "no speech recognized", input_desc, started)
                return
            self.heard_at = time.monotonic()
            self._log_partials_vs(text)
            self._log_evals(text, len(pcm) // 32)
            await self.ws.send_json({"type": "transcript", "text": text})
            if _is_closer(text):
                # Idle without a reply: the satellite goes back to the wake word, no follow-up window.
                logger.info("closer %r: conversation over, back to the wake word", text)
                await self.set_state("idle")
                return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail(stage, f"{type(exc).__name__}: {exc}", input_desc, started)
            return
        if SPEAKER_ID != "off" and self.role == "satellite":
            self._speaker_task = asyncio.create_task(self._log_speaker(pcm, text))  # kept: unreferenced tasks can vanish
        await self._respond(text, timing)

    async def _log_speaker(self, pcm: bytes, text: str) -> None:
        """Shadow mode: who the voice sounds like, logged with the turn (no words: just the first few)."""
        try:
            started = time.monotonic()
            who = await asyncio.to_thread(speaker.identify, pcm)
            # "Yeah." / "Sure." is too little voice to judge (a third of turns, 2026-11-05 logs). Within one
            # conversation, the last confident match still stands; a longer unknown clip ends it.
            last_user, last_at = self._speaker_last
            if who["verdict"] == "match":
                self._speaker_last = (who["user"], time.monotonic())
            elif who.get("short") and last_user and time.monotonic() - last_at < SPEAKER_CARRY_S:
                who = {**who, "verdict": "carried", "user": last_user}
            elif not who.get("short"):
                self._speaker_last = (None, 0.0)
            logger.info("speaker (shadow): %s %s score=%.2f (%d ms, %.1f s of audio) %r", who["verdict"], who["user"] or "-",
                        who["score"], _ms(started), len(pcm) / 2 / SAMPLE_RATE, " ".join(text.split()[:3]))
            # Kept across rebuilds (docker logs aren't): verdict, score, length, time. Never his words.
            with open(os.path.join(os.getenv("THURSDAY_BRAIN_DIR", "brain"), "speaker_shadow.log"), "a") as f:
                f.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {who['verdict']} {who['user'] or '-'} "
                        f"{who['score']:.2f} {len(pcm) / 2 / SAMPLE_RATE:.1f}s\n")
        except Exception as exc:  # recognition is a nicety until it decides anything
            logger.info("speaker (shadow): failed: %s", type(exc).__name__)

    async def _respond(self, text: str, timing: dict, mode: str = "voice", context: dict | None = None) -> None:
        """LLM + TTS for one user message, or for a prompt the server writes
        itself (the alarm briefing's live fallback)."""
        stage, input_desc, started = "LLM", f"{len(text)} chars", time.monotonic()
        try:
            # Chunked clients get one WAV per piece, synthesized while the LLM
            # is still writing; the rest get the whole reply as one WAV.
            chunker = SentenceChunker() if self._chunked else None
            speaker = asyncio.create_task(self._speak_chunks(chunker, timing)) if chunker else None
            self._said = None
            self._turn_tools = False
            try:
                speak_text = ""
                async for ev in stream_prompt(text, mode=mode, conversation_id=self._conversation_id,
                                             context=self._context if context is None else context, user=self.user):
                    if ev["type"] == "conversation":
                        self._conversation_id = ev["id"]
                        await self.ws.send_json(ev)
                    elif ev["type"] == "delta":
                        timing.setdefault("llm_ttft_ms", _ms(started))
                        if self._said is None:
                            self._said = []
                        if chunker:
                            chunker.feed(ev["text"])
                    elif ev["type"] == "deliberating":
                        if chunker and mode not in ("briefing", "wake"):
                            await self._say_filler("_deliberate", timing)
                    elif ev["type"] == "tool":
                        self._turn_tools = True
                        if self._said is None:
                            self._said = []
                        await self.ws.send_json({"type": "tool", "name": ev["name"]})
                        if chunker and mode not in ("briefing", "wake"):
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
            seg = self._segment(speak_text)
            await self._send_audio(audio, seg)
            seg["done"] = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail(stage, f"{type(exc).__name__}: {exc}", input_desc, started)

    async def _log_partial(self, pcm: bytes) -> None:
        """What the user has said so far, while they're still talking (full duplex, slice 2: logged only)."""
        started = time.monotonic()
        try:
            text = await self._transcribe(pcm)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.info("partial STT failed: %s", type(exc).__name__)
            return
        speech_ms = len(pcm) // 32  # 16 kHz int16
        self._partials.append((speech_ms, text))
        if EVALUATOR:
            self._start_eval(text, speech_ms)
        logger.info("partial at %d ms of speech: stt_ms=%d %r", speech_ms, _ms(started), text)

    def _start_eval(self, text: str, speech_ms: int) -> None:
        """Ask the evaluator about what's been said so far (one call at a time, never the same text twice)."""
        text = text.strip()
        if text == self._eval_text or (self._eval and not self._eval.done()):
            return
        self._eval_text = text
        self._eval = asyncio.create_task(self._shadow_eval(text, speech_ms))

    async def _shadow_eval(self, text: str, speech_ms: int) -> None:
        started = time.monotonic()
        decision = await interject.evaluate(self.http, text)
        if decision is None:
            return
        self._evals.append((speech_ms, decision))
        if decision["action"] == "INTERJECT":
            logger.info("evaluator (shadow): INTERJECT %s at %d ms of speech, eval_ms=%d: ...%r%s", "+".join(decision["why"]),
                        speech_ms, _ms(started), text[-80:], f" (truth: {decision['truth']!r})" if decision["truth"] else "")

    def _log_evals(self, final: str, speech_ms: int) -> None:
        """Per utterance: where the evaluator would have cut in, and how much more the user said after."""
        if not EVALUATOR:
            return
        hits = [ms for ms, d in self._evals if d["action"] == "INTERJECT"]
        logger.info("evaluator (shadow): %d calls, interject at %s of %d ms of speech: %r", len(self._evals),
                    hits or "never", speech_ms, final[:120])
        self._evals, self._eval_text = [], ""

    def _log_partials_vs(self, final: str) -> None:
        """How much of the final transcript the last partial already had: what an evaluator would have seen."""
        if not self._partials:
            return
        words = lambda s: re.findall(r"[a-z0-9']+", s.lower())
        last_ms, last = self._partials[-1]
        fin, got = words(final), words(last)
        shared = sum(1 for a, b in zip(fin, got) if a == b)  # leading words that already matched
        logger.info("partials: %d, last at %d ms had %d/%d of the final words in place", len(self._partials),
                    last_ms, shared, len(fin))
        self._partials = []

    async def _transcribe(self, pcm: bytes) -> str:
        files = {"audio": ("utterance.wav", _pcm_to_wav(pcm), "audio/wav")}
        try:
            resp = await self.http.post(f"{STT_BASE_URL}/transcribe", files=files, timeout=STT_TIMEOUT)
        except httpx.ConnectError:
            if not STT_FALLBACK_URL:
                raise
            logger.warning("STT at %s unreachable; using %s", STT_BASE_URL, STT_FALLBACK_URL)
            resp = await self.http.post(f"{STT_FALLBACK_URL}/transcribe", files=files, timeout=STT_TIMEOUT)
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
        gap_s = 0.0
        dry_count, dry_max_s = 0, 0.0  # times the client (by our estimate) ran out of audio mid-reply
        self._play_end = 0.0

        async def send(audio: bytes, started: float, seg: dict) -> None:
            nonlocal sent, gap_s, dry_count, dry_max_s
            if sent == 0:
                timing["tts_first_ms"] = _ms(started)
                timing["speech_end_to_audio_sent_ms"] = _ms(self._speech_end)
                # Store before sending: playback_started can arrive while send_bytes yields.
                self._timing = timing
                await self.set_state("speaking")
            now = time.monotonic()
            if sent and now > self._play_end:
                gap_s += now - self._play_end  # the client ran dry before this audio arrived
                dry_count += 1
                dry_max_s = max(dry_max_s, now - self._play_end)
            await self._send_audio(audio, seg)
            sent += 1

        try:
            rocky_state: dict = {}
            if await _tts_streams(self.http):
                await chunker.settle(ORPHEUS_FIRST_PIECE_WAIT_S if _tts_caps["engine"] == "orpheus" else FIRST_PIECE_WAIT_S)
                max_chars = STREAM_PIECE_CHARS
            while (piece := await chunker.next_piece(max_chars)) is not None:
                started = time.monotonic()
                if not self._turn_tools and not (piece := _drop_invented(piece)):
                    continue
                if await _tts_streams(self.http):
                    # Forward audio while it's generated. The voice ends each piece
                    # with its own short pause, so no trim / added pause here.
                    audio_s = 0.0
                    for text, beat in _with_beat(piece, chunker.closed and chunker.empty, sent > 0, streamed=True):
                        seg = self._segment(text)
                        if beat:
                            await self._send_audio(_pcm_to_wav(bytes(_tts_caps["rate"] * BEAT_MS // 1000 * 2),
                                                               _tts_caps["rate"]))
                        # The reply's first audio waits for PREROLL_MS of it: see PREROLL_MS.
                        preroll_ms = PREROLL_MS
                        if (sent == 0 and self._style().get("voice") != "tara" and rocky_voice._QUESTION.search(text.strip())
                                and len(text) / chars_per_s < QUESTION_SHORT_S):
                            # Rocky holds a question's last 0.5 s (its chime) until the TTS finishes; a short
                            # piece streams near real time, so playback caught up and stuttered before it.
                            preroll_ms += QUESTION_PREROLL_MS
                            timing["question_preroll"] = 1
                        first_bytes = max(STREAM_FIRST_BYTES, _tts_caps["rate"] * 2 * preroll_ms // 1000) \
                            if sent == 0 else STREAM_FIRST_BYTES
                        async for audio in _tts_stream(self.http, text, first_bytes, self._style(), rocky_state):
                            audio_s += _wav_seconds(audio)
                            await send(audio, started, seg)
                        seg["done"] = True
                    synth_s = time.monotonic() - started
                    streamed = True
                else:
                    seg = self._segment(piece)
                    audio = await self._synthesize(piece)
                    if not audio:
                        continue
                    synth_s = time.monotonic() - started
                    audio_s = _wav_seconds(audio)
                    streamed = False
                    # Swap the clip's padding for the pause its last punctuation calls for.
                    last = chunker.closed and chunker.empty
                    await send(_trim_wav(audio, lead_ms=0, pause_ms=0 if last else _pause_after(piece)), started, seg)
                seg["done"] = True
                if audio_s > 0:
                    rtf = (rtf + synth_s / audio_s) / 2
                    chars_per_s = (chars_per_s + len(piece) / audio_s) / 2
                    self._chars_per_s = chars_per_s
                lead_s = self._play_end - time.monotonic()
                max_chars = int(lead_s * PIECE_SAFETY / rtf * chars_per_s)
                max_chars = max(FIRST_PIECE_CHARS, min(SentenceChunker.MAX_CHARS, max_chars))
                if streamed:
                    # Streaming starts audio fast whatever the piece length, so after the
                    # first piece cut only at sentence ends: each extra cut costs ~0.5 s of
                    # silence and a falling, end-of-sentence tone mid-sentence.
                    max_chars = STREAM_PIECE_CHARS
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail("TTS", f"{type(exc).__name__}: {exc}", f"chunk {sent + 1}", time.monotonic())
            return -1
        timing["tts_chunks"] = sent
        timing["gap_ms"] = round(gap_s * 1000)
        timing["dry_count"] = dry_count
        timing["dry_max_ms"] = round(dry_max_s * 1000)
        timing["tts_rtf"] = round(rtf, 2)
        if sent:
            await self.ws.send_json({"type": "audio_end", "chunks": sent})
        return sent

    async def _say_filler(self, kind: str, timing: dict) -> None:
        """At most one filler a turn, and only before any of the reply's own audio:
        chunked clients queue it ahead of the reply, back to back."""
        style = self._style()
        if self.state != "thinking" or (filler := pick_filler(kind, style.get("voice", "leo"))) is None:
            return
        if style.get("whisper"):
            return  # a full-voice "let me see" would break the whisper
        if style.get("volume", 1.0) < 1.0:
            filler = rocky_voice.scale_wav(filler, style["volume"])
        self._timing = timing  # before sending: playback_started may arrive first
        timing["filler"] = kind
        timing["filler_ms"] = _ms(self._speech_end)
        await self.set_state("speaking")
        await self._send_audio(filler)

    async def _synthesize(self, text: str) -> bytes | None:
        return await _tts(self.http, text, self._style())

    def _style(self) -> dict:
        """His voice for whoever is talking: volume and whisper (mind.voice_setting). Read per piece, so a
        change made this turn is already in the reply that confirms it."""
        try:
            return mind.voice_style(self.user["id"])
        except Exception:  # settings are a nicety: never let them stop a reply
            return {}

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
                       conversation_id: str | None = None, context: dict | None = None,
                       cue: str = "alarm", mode: str = "briefing") -> bool:
        """Server-initiated speech (alarms): an "alarm" cue, then either audio
        made ahead of time or a live reply to `prompt` (mode "briefing"). The
        session continues in `conversation_id`, so follow-up questions have the
        briefing as context. The client reports playback_ended as usual, and a
        satellite in follow-up mode starts listening right after. `cue` picks
        the satellite's opening sound: "alarm" (the chime) or "announce" (the
        soft ping). Only from idle; returns False if the session is busy."""
        if self.state != "idle" or (self._turn and not self._turn.done()):
            return False
        self._said = None
        if self.barge:
            self.barge.reset()
        await self.ws.send_json({"type": "alarm", "cue": cue})
        self._speech_end = time.monotonic()
        self._chunked = True
        if conversation_id:
            self._conversation_id = conversation_id
        if audio:
            self._turn = asyncio.create_task(self._send_prepared(audio))
            # A mission or briefing just had the GPU: re-warm while the report plays, so his reply
            # to it isn't a cold 5-7 s prefill. Forced: the cache is gone whenever the last warm was.
            self._warm(force=True)
        else:
            self._turn = asyncio.create_task(self._respond(prompt, {"alarm": 1}, mode=mode, context=context))
        return True

    def _warm(self, force: bool = False) -> None:
        """Prefill this conversation's prompt in Ollama in the background (warm_cache).
        At most once a minute unless forced, and never on top of another warm-up."""
        if self._warming and not self._warming.done():
            return
        if not force and time.monotonic() - self._warmed_at < WARM_EVERY_S:
            return
        self._warmed_at = time.monotonic()
        self._warming = asyncio.create_task(warm_cache(self.user, self._context, self._conversation_id))

    async def _send_prepared(self, audio: list[bytes]) -> None:
        self._timing = {"alarm": 1, "prepared_chunks": len(audio)}
        await self.set_state("speaking")
        for wav in audio:
            await self._send_audio(wav)
        await self.ws.send_json({"type": "audio_end", "chunks": len(audio)})

    async def close(self) -> None:
        SATELLITES.discard(self)
        if self._turn and not self._turn.done():
            self._turn.cancel()
        await self.http.aclose()


@router.websocket("/ws/music-tap")
async def music_tap(ws: WebSocket) -> None:
    """The Mac's music tap (mac/music_tap): YouTube Music's sound, relayed to the satellites."""
    await music.serve_tap(ws)


@router.websocket("/ws/voice")
async def voice_ws(ws: WebSocket) -> None:
    # Accept first so the client receives the 1008 close code; rejecting during
    # the handshake would surface in the browser only as a generic 1006.
    await ws.accept()
    user = ws_user(ws)
    if not user:
        logger.warning("Rejected /ws/voice from %s: not logged in, or a bad device token",
                       ws.client.host if ws.client else "?")
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason="not logged in")
        return

    session = Session(ws, user)
    logger.info("voice session opened for %s (frame=%dms, vad=%s)", user["username"], FRAME_MS, session.endpointer.cfg)
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
