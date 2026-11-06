"""Phone calls over Twilio: THURSDAY calls someone and speaks a message, and answers calls to its number.

Calling out is one REST request (Calls.json) whose inline TwiML <Play>s a WAV rendered in his voice ahead of time:
the disclosure (voice_ws.PHONE_DISCLOSURE) and then the message. Twilio fetches that WAV from
/twilio/audio/<token>.wav: a random one-use-per-call token that expires, since Twilio doesn't sign media fetches.

Calls are two-step, enforced here like calendar_confirm: call_place only renders the audio and stores a pending
call; call_confirm dials only when his *next* message is a plain yes. A daily cap bounds any mistake.

The internet reaches only /twilio/* (Tailscale Funnel on :8443 with --set-path=/twilio; the UI stays tailnet-only),
and every webhook POST must carry a valid X-Twilio-Signature for TWILIO_PUBLIC_URL + path, or it gets a 403.
Credentials come from .env (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM) and are never logged or returned.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import io
import logging
import os
import re
import secrets
import time
import wave
from typing import Any, Dict
from urllib.parse import parse_qsl
from xml.sax.saxutils import escape

import httpx
from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect

from accounts import OWNER, OWNER_NAME
from calendar_tool import PENDING_TTL_S, TURN, _HEDGE, _YES
from tools import ToolResult

logger = logging.getLogger("thursday.voice.phone")  # under voice_ws's logger: its handler gets these into docker logs
SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
FROM = os.getenv("TWILIO_FROM", "")
OWNER_NUMBER = os.getenv("TWILIO_OWNER_NUMBER", "")  # "call me"
PUBLIC = os.getenv("TWILIO_PUBLIC_URL", "").rstrip("/")  # e.g. https://<mac>.<tailnet>.ts.net:8443
DAILY_CAP = int(os.getenv("TWILIO_DAILY_CALLS", "10"))
# Call times on disk, one per line, so a restart can't reset the cap.
PLACED_FILE = os.path.join(os.getenv("THURSDAY_BRAIN_DIR", "./brain"), "calls_placed.txt")
# Two-way calls (in progress): the call opens a Media Stream to /twilio/stream instead of <Play>ing a file.
# Slice A only plays the rendered message over it and logs what the line hears; off until it's proven.
TWO_WAY = os.getenv("TWILIO_TWO_WAY", "0") == "1"
LISTEN_AFTER_S = 20  # slices A-B: how long the stream stays open after the message, hearing the other side
CONTACTS_FILE = os.path.join(os.getenv("THURSDAY_BRAIN_DIR", "./brain"), "contacts.txt")
AUDIO_TTL_S = 15 * 60
E164 = re.compile(r"^\+[1-9]\d{7,14}$")
# Toll fraud: only these country prefixes dial (international premium lines bill per minute), never US premium-rate.
ALLOWED = tuple(p.strip() for p in os.getenv("TWILIO_ALLOWED_PREFIXES", "+1").split(",") if p.strip())
PREMIUM = re.compile(r"^\+1(900|976)")

_audio: dict[str, tuple[float, bytes]] = {}  # token -> (expires, wav)
_pending: dict[str, dict] = {}               # conversation -> prepared call


def _calls_today(now: float) -> list[float]:
    try:
        with open(PLACED_FILE) as f:
            times = [float(line) for line in f if line.strip()]
    except (OSError, ValueError):
        times = []
    return [t for t in times if now - t < 86400]


def _record_call(times: list[float]) -> None:
    try:
        with open(PLACED_FILE, "w") as f:
            f.write("".join(f"{t}\n" for t in times))
    except OSError as exc:
        logger.warning("call count not saved: %s", exc)
router = APIRouter(prefix="/twilio")


def configured() -> bool:
    return bool(SID and TOKEN and FROM and PUBLIC)


def signature(url: str, params: dict[str, str]) -> str:
    """Twilio's X-Twilio-Signature: base64 HMAC-SHA1 of the URL plus every POST param, sorted, name then value."""
    data = url + "".join(k + params[k] for k in sorted(params))
    return base64.b64encode(hmac.new(TOKEN.encode(), data.encode(), hashlib.sha1).digest()).decode()


async def _signed_form(request: Request) -> dict[str, str]:
    """The webhook's form params, or a 403 unless Twilio signed them (checked against the public URL Twilio called,
    not the local one Funnel forwards to)."""
    params = dict(parse_qsl((await request.body()).decode(), keep_blank_values=True))
    got = request.headers.get("x-twilio-signature", "")
    if not (TOKEN and PUBLIC and got and hmac.compare_digest(got, signature(PUBLIC + request.url.path, params))):
        logger.warning("twilio webhook refused: bad or missing signature")
        raise HTTPException(status_code=403)
    return params


def _dialable(to: str) -> str | None:
    """A said or written number as E.164, if it's one we'd ever dial (+1 by default, never premium)."""
    digits = re.sub(r"[^\d+]", "", to or "")
    if re.fullmatch(r"\d{10}", digits):  # a US number said without the country code
        digits = "+1" + digits
    ok = E164.match(digits) and digits.startswith(ALLOWED) and not PREMIUM.match(digits)
    return digits if ok else None


def contacts() -> dict[str, str]:
    """The allowlist: brain/contacts.txt, one "Name: number" per line, edited by him only (no tool writes it,
    so nobody on a call or in a web page can add a number). Lines with a number we'd never dial are skipped."""
    out: dict[str, str] = {}
    try:
        with open(CONTACTS_FILE) as f:
            for line in f:
                name, sep, num = line.partition(":")
                if sep and not name.lstrip().startswith("#") and (number := _dialable(num)):
                    out[name.strip().lower()] = number
    except OSError:
        pass
    return out


def _number(to: str) -> str | None:
    """Only "me" or someone in contacts(): a name, or a number that's on the list."""
    to = re.sub(r"^(?:my|our)\s+", "", (to or "").strip(), flags=re.I).lower()
    if to in ("me", "phone", "daniel", OWNER_NAME.lower()):
        return OWNER_NUMBER or None
    book = contacts()
    if to in book:
        return book[to]
    number = _dialable(to)
    return number if number and number in book.values() else None


def _render(text: str) -> bytes:
    """Disclosure + message in his current voice, as one WAV (runs in the tool's worker thread)."""
    import mind
    import voice_ws

    async def go() -> list[bytes]:
        async with httpx.AsyncClient() as http:
            return await voice_ws.synthesize_ahead(http, f"{voice_ws.PHONE_DISCLOSURE} {text}", mind.voice_style(OWNER))
    parts = asyncio.run(go())
    if not parts:
        raise RuntimeError("voice server returned nothing")
    out, rate = io.BytesIO(), None
    with wave.open(out, "wb") as w:
        for part in parts:
            with wave.open(io.BytesIO(part)) as r:
                if rate is None:
                    rate = r.getframerate()
                    w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
                w.writeframes(r.readframes(r.getnframes()))
    # Phone audio is 8 kHz: send that (band-limited like to_mulaw_8k) so Twilio doesn't have to resample a 24 kHz file.
    import numpy as np
    import rocky_voice
    x, sr = rocky_voice._read(out.getvalue())
    y = rocky_voice._lowpass(x, 3400 / sr, taps=191)
    y = y[::sr // 8000] if sr % 8000 == 0 else np.interp(np.arange(0, len(y), sr / 8000), np.arange(len(y)), y)
    return rocky_voice._write(y, 8000)


def _keep_audio(wav: bytes) -> str:
    now = time.time()
    for k in [k for k, (exp, _) in _audio.items() if exp < now]:
        del _audio[k]
    token = secrets.token_urlsafe(24)
    _audio[token] = (now + AUDIO_TTL_S, wav)
    return token


def call_place(args: Dict[str, Any]) -> ToolResult:
    """Prepares a call (renders the audio) and asks him; nothing dials until call_confirm."""
    turn = TURN.get()
    if not turn or turn.get("user") != OWNER:
        return ToolResult(ok=False, tool_name="call_place", error="calls_are_daniels_only")
    if not configured():
        return ToolResult(ok=False, tool_name="call_place", error="phone_not_set_up")
    number = _number(str(args.get("to", "")))
    message = str(args.get("message") or "").strip()
    if not number:
        names = ", ".join(n.title() for n in contacts()) or "nobody yet"
        return ToolResult(ok=False, tool_name="call_place", error=(
            f"not_in_contacts: you can only call him or his saved contacts ({names}). Tell him to add them "
            "to his contacts first; you can't add anyone yourself."))
    if not message:
        return ToolResult(ok=False, tool_name="call_place", error="no_message: ask him what to say")
    try:
        token = _keep_audio(_render(message[:600]))
    except Exception as exc:
        logger.warning("call audio not rendered: %s", exc)
        return ToolResult(ok=False, tool_name="call_place", error="voice_server_down")
    _pending[turn["conversation"]] = {"at": time.monotonic(), "turn": turn["turn"], "to": number, "token": token,
                                      "message": message}
    return ToolResult(ok=True, tool_name="call_place", data={
        "status": "waiting_for_his_ok", "to": number, "will_say": message,
        "note": "Read back who and what in one line and ask if you should call. Only call_confirm after a plain yes."})


def confirms_pending(conversation_id: str, said: str) -> bool:
    """A plain yes to a call still waiting in this conversation: the agent then dials via call_confirm itself."""
    pending = _pending.get(conversation_id)
    return bool(pending) and time.monotonic() - pending["at"] <= PENDING_TTL_S and bool(_YES.match(said)) \
        and not _HEDGE.search(said)


def call_confirm(args: Dict[str, Any]) -> ToolResult:
    """Dials the pending call if his latest message is a plain yes."""
    turn = TURN.get()
    pending = _pending.get(turn["conversation"]) if turn else None
    if not pending or time.monotonic() - pending["at"] > PENDING_TTL_S:
        _pending.pop((turn or {}).get("conversation"), None)
        return ToolResult(ok=False, tool_name="call_confirm",
                          error="nothing_to_confirm: call_place never ran; run call_place now, then ask him again")
    if pending["turn"] == turn["turn"]:
        return ToolResult(ok=False, tool_name="call_confirm", error="not_confirmed: ask him first and wait")
    said = turn["user_said"]
    if not _YES.match(said) or _HEDGE.search(said):
        return ToolResult(ok=False, tool_name="call_confirm",
                          error="not_confirmed: not a plain yes; if he changed details, call_place again")
    del _pending[turn["conversation"]]
    now = time.time()
    placed = _calls_today(now)
    if len(placed) >= DAILY_CAP:
        return ToolResult(ok=False, tool_name="call_confirm", error=f"daily_cap: {DAILY_CAP} calls a day")
    if TWO_WAY:
        twiml = (f'<Response><Connect><Stream url="{escape(_stream_url())}">'
                 f'<Parameter name="token" value="{pending["token"]}"/></Stream></Connect><Hangup/></Response>')
    else:
        twiml = f"<Response><Play>{escape(PUBLIC)}/twilio/audio/{pending['token']}.wav</Play></Response>"
    try:
        r = httpx.post(f"https://api.twilio.com/2010-04-01/Accounts/{SID}/Calls.json", auth=(SID, TOKEN), timeout=10,
                       data={"To": pending["to"], "From": FROM, "Twiml": twiml,
                             "TimeLimit": "180"})  # a hard stop on billed minutes, whatever happens on the line
    except httpx.HTTPError as exc:
        return ToolResult(ok=False, tool_name="call_confirm", error=f"twilio_unreachable: {type(exc).__name__}")
    if r.status_code >= 400:
        detail = (r.json().get("message") if "json" in r.headers.get("content-type", "") else "") or r.status_code
        logger.warning("twilio refused the call: %s", detail)
        return ToolResult(ok=False, tool_name="call_confirm", error=f"twilio_refused: {detail}")
    _record_call(placed + [now])
    logger.info("call placed to %s…%s", pending["to"][:5], pending["to"][-2:])
    return ToolResult(ok=True, tool_name="call_confirm", data={"status": "calling", "to": pending["to"]})


@router.get("/audio/{token}.wav")
def audio(token: str) -> Response:
    item = _audio.get(token)
    if not item or item[0] < time.time():
        raise HTTPException(status_code=404)
    return Response(item[1], media_type="audio/wav")


def _stream_url() -> str:
    return "wss://" + PUBLIC.split("://", 1)[-1] + "/twilio/stream"


def _mulaw(wav: bytes) -> bytes:
    import rocky_voice
    x, sr = rocky_voice._read(wav)
    return rocky_voice.to_mulaw_8k(x, sr)


class _CallEar:
    """Slice B: what the other side says. 8 kHz mu-law in, 16 kHz frames through the same Silero endpointer
    as /ws/voice, each finished utterance to STT. Logs the words and per-stage timings to judge phone-line STT."""

    def __init__(self) -> None:
        from vad import FRAME_BYTES, Endpointer, VADConfig
        self.frame_bytes = FRAME_BYTES
        self.endpointer = Endpointer(VADConfig.from_env())
        self.buf = b""
        self.tasks: set[asyncio.Task] = set()
        self.heard: list[str] = []
        self.http = httpx.AsyncClient()

    def feed(self, mulaw: bytes) -> None:
        import numpy as np
        import rocky_voice
        x = np.asarray(rocky_voice.mulaw_decode(mulaw), dtype=np.float32)
        up = np.interp(np.arange(0, len(x), 0.5), np.arange(len(x)), x)  # 8 -> 16 kHz
        self.buf += up.astype(np.int16).tobytes()
        while len(self.buf) >= self.frame_bytes:
            frame, self.buf = self.buf[:self.frame_bytes], self.buf[self.frame_bytes:]
            if utterance := self.endpointer.feed(frame):
                task = asyncio.create_task(self._transcribe(utterance, time.monotonic()))
                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)

    async def _transcribe(self, pcm: bytes, ended: float) -> None:
        import voice_ws
        try:
            files = {"audio": ("call.wav", voice_ws._pcm_to_wav(pcm), "audio/wav")}
            resp = await self.http.post(f"{voice_ws.STT_BASE_URL}/transcribe", files=files, timeout=voice_ws.STT_TIMEOUT)
            text = str(resp.json().get("text") or "").strip()
        except Exception as exc:
            logger.warning("call ear: STT failed (%s)", type(exc).__name__)
            return
        self.heard.append(text)
        logger.info("call heard: %r (speech %d ms, stt %d ms)", text[:200], len(pcm) // 32,
                    (time.monotonic() - ended) * 1000)

    async def close(self) -> None:
        if self.tasks:
            await asyncio.wait(self.tasks, timeout=10)
        await self.http.aclose()


@router.websocket("/stream")
async def media_stream(ws: WebSocket) -> None:
    """A two-way call's live audio (Twilio Media Streams: JSON frames, base64 8 kHz mu-law, 20 ms each).
    Twilio signs the handshake like a webhook (URL, no params); the first "start" frame must also carry a live
    audio token from call_confirm, so a forged stream can't get anything played or heard."""
    got = ws.headers.get("x-twilio-signature", "")
    if not (TOKEN and PUBLIC and got and hmac.compare_digest(got, signature(_stream_url(), {}))):
        logger.warning("twilio stream refused: bad or missing signature")
        await ws.close(code=1008)
        return
    await ws.accept()
    import json
    import numpy as np
    import rocky_voice
    sid, played, heard, levels = None, False, 0, []
    ear: _CallEar | None = None  # made once the message has played: nothing loads for a stream that never gets there
    try:
        while True:
            frame = json.loads(await ws.receive_text())
            event = frame.get("event")
            if event == "start":
                start = frame.get("start") or {}
                sid = start.get("streamSid")
                item = _audio.pop((start.get("customParameters") or {}).get("token", ""), None)
                if not item or item[0] < time.time():
                    logger.warning("twilio stream refused: no live call token")
                    break
                audio = await asyncio.to_thread(_mulaw, item[1])
                for i in range(0, len(audio), 8000):  # 1 s per frame; Twilio queues and plays them in order
                    await ws.send_json({"event": "media", "streamSid": sid,
                                        "media": {"payload": base64.b64encode(audio[i:i + 8000]).decode()}})
                await ws.send_json({"event": "mark", "streamSid": sid, "mark": {"name": "message_done"}})
                logger.info("call stream: started, %.1f s of message queued", len(audio) / 8000)
            elif event == "media" and sid:
                raw = base64.b64decode(frame["media"]["payload"])
                if played:  # not during the message: an echo of it would be heard as them
                    ear = ear or _CallEar()
                    ear.feed(raw)
                pcm = rocky_voice.mulaw_decode(raw)
                levels.append(float(np.mean(np.asarray(pcm, dtype=np.float32) ** 2)))
                heard += 1
                if heard % 50 == 0:  # each second of the line: its level, to see the other side is heard
                    db = 10 * np.log10(np.mean(levels) / 32768 ** 2 + 1e-12)
                    logger.info("call stream: %ds, line %.0f dBFS%s", heard // 50, db, " (after message)" if played else "")
                    levels.clear()
                if played and time.monotonic() - played > LISTEN_AFTER_S:
                    break
            elif event == "mark" and (frame.get("mark") or {}).get("name") == "message_done":
                played = time.monotonic()
                logger.info("call stream: message played; listening %d s", LISTEN_AFTER_S)
            elif event == "stop":
                break
    except WebSocketDisconnect:
        pass
    if ear:
        await ear.close()
    logger.info("call stream: ended after %d s heard, %d utterance(s)", heard // 50, len(ear.heard) if ear else 0)
    try:
        await ws.close()  # Twilio moves on to <Hangup/>
    except RuntimeError:
        pass


@router.post("/voice")
async def inbound_call(request: Request) -> Response:
    """Someone called THURSDAY's number: say who answered, that he'll see it, and hang up; push him the caller."""
    params = await _signed_form(request)
    caller = params.get("From", "unknown")
    import approvals
    await asyncio.to_thread(approvals.push, "Missed call", f"{caller} called THURSDAY's number.", tags=("telephone",))
    say = (f"Hi, you've reached THURSDAY, {escape(OWNER_NAME)}'s assistant. This line doesn't take calls yet. "
           f"{escape(OWNER_NAME)} will see that you called. Goodbye!")
    return Response(f'<?xml version="1.0" encoding="UTF-8"?><Response><Say>{say}</Say><Hangup/></Response>',
                    media_type="application/xml")


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "call_place",
        "description": "Phone someone for him and speak a message in your voice ('call my mom and tell her I'm "
        "running late', 'call me'). Run it RIGHT AWAY when he asks, before asking him anything: it dials nothing, "
        "it only gets the call ready. Then read back who and what from its result and ask; call_confirm dials "
        "after his yes. Never say a call is prepared unless this tool ran. A one-way message: you can't hear them yet.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string", "description": "A saved contact's name ('Mom'), or 'me' for his own phone"},
            "message": {"type": "string", "description": "What to say, worded for the listener, a few sentences"}},
            "required": ["to", "message"]}}},
    {"type": "function", "function": {
        "name": "call_confirm", "description": "Dial the call call_place prepared, only after his plain yes.",
        "parameters": {"type": "object", "properties": {}}}},
]
