"""What THURSDAY says when he's woken ("Hey THURSDAY" on the satellite, opening voice mode in the web UI), instead
of a ping. A fresh batch every hour: the LLM writes short greetings for the time of day, each is rendered in his
current voice ahead of time, and the clips go to the satellites and the page, so a wake plays one instantly with no
TTS wait. Rendering waits for a quiet house (missions.touch), like the mission worker, so it never competes with a
conversation for the GPU. The clients greet only when he pauses after waking; one breath ("Hey THURSDAY, what's
the weather?") gets no greeting, so none of his words are lost under it.

The batch is kept on disk (brain/greetings/), so a restart has greetings before the first refresh.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import time
from datetime import datetime

import httpx

from accounts import OWNER, OWNER_NAME

logger = logging.getLogger("thursday.voice.greetings")  # under voice_ws's logger: its handler reaches docker logs
DIR = os.path.join(os.getenv("THURSDAY_BRAIN_DIR", "./brain"), "greetings")
REFRESH_S = 3600
COUNT = 10        # asked for; the ones too long to say are dropped
MAX_S = 2.0       # longer is a wait between "Hey THURSDAY" and his request
QUIET_S = 30  # nobody has talked this long before rendering uses the GPU

clips: list[bytes] = []  # WAVs, the current batch
version = 0              # bumped each batch, so the page knows to fetch the new ones


def _part_of_day(now: datetime) -> str:
    h = now.hour
    return "early morning" if h < 8 else "morning" if h < 12 else "afternoon" if h < 17 else "evening" if h < 22 \
        else "late night"


async def _write_lines(http: httpx.AsyncClient, tara: bool) -> list[str]:
    from agent_stream import CHAT_URL, MODEL, OLLAMA_KEEP_ALIVE, OLLAMA_OPTIONS
    import mind
    when = _part_of_day(datetime.now(mind.VOICE_TZ))
    prompt = (f"Write {COUNT} different greetings THURSDAY, {OWNER_NAME}'s AI assistant, says the moment {OWNER_NAME} "
              f"wakes him with \"Hey THURSDAY\". It's {when}. Each 2 to 5 words, casual and warm, like a friend picking "
              f"up: some use his name, some fit the time of day, some are a quick question back (\"Hey {OWNER_NAME}, "
              "what's up?\", \"Hey, how's it going?\"). All different. No emojis or quotes. One per line, nothing else.")
    r = await http.post(CHAT_URL, timeout=60, json={
        "model": MODEL, "stream": False, "think": False, "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {**OLLAMA_OPTIONS, "temperature": 1.0}, "messages": [{"role": "user", "content": prompt}]})
    lines = [re.sub(r"^[\s\-*\d.)\"']+|[\"']+$", "", ln).strip()
             for ln in (r.json().get("message") or {}).get("content", "").splitlines()]
    lines = [ln for ln in lines if 1 <= len(ln.split()) <= 7][:COUNT]
    if not tara:  # Rocky's grammar, like everything else he says
        import eridian
        lines = [eridian.translate(ln) for ln in lines]
    return lines


async def refresh() -> bool:
    """One new batch: write, render, keep, send. False (old batch kept) if anything fails."""
    global clips, version
    import mind
    import voice_ws
    style = {**mind.voice_style(OWNER), "volume": 1.0}  # players apply his volume themselves
    try:
        async with httpx.AsyncClient() as http:
            lines = await _write_lines(http, style.get("voice") == "tara")
            new = []
            for line in lines:
                parts = await voice_ws.synthesize_ahead(http, line, style)
                if parts and voice_ws._wav_seconds(parts[0]) <= MAX_S:
                    new.append(parts[0])
    except Exception as exc:
        logger.warning("greetings not refreshed: %s", type(exc).__name__)
        return False
    if len(new) < 3:
        logger.warning("greetings not refreshed: only %d rendered", len(new))
        return False
    clips, version = new, version + 1
    _save(new)
    logger.info("greetings refreshed: %d clips (%s)", len(new), "; ".join(lines[:3]) + " ...")
    await push_to_satellites()
    return True


def _save(batch: list[bytes]) -> None:
    try:
        os.makedirs(DIR, exist_ok=True)
        for name in os.listdir(DIR):
            if name.endswith(".wav"):
                os.remove(os.path.join(DIR, name))
        for i, wav in enumerate(batch):
            with open(os.path.join(DIR, f"{i}.wav"), "wb") as f:
                f.write(wav)
    except OSError as exc:
        logger.warning("greetings not saved: %s", exc)


def _load_saved() -> None:
    global clips, version
    try:
        names = sorted((n for n in os.listdir(DIR) if n.endswith(".wav")), key=lambda n: int(n[:-4]))
        batch = [open(os.path.join(DIR, n), "rb").read() for n in names]
    except (OSError, ValueError):
        return
    if batch:
        clips, version = batch, 1


def message() -> dict:
    """What a satellite gets: the whole batch, base64 (8 short clips, a few hundred KB, once an hour)."""
    return {"type": "greetings", "clips": [base64.b64encode(c).decode() for c in clips]}


async def push_to_satellites() -> None:
    import voice_ws
    for sat in list(voice_ws.SATELLITES):
        try:
            await sat.ws.send_json(message())
        except Exception as exc:
            logger.warning("greetings not sent to a satellite: %s", type(exc).__name__)


async def run() -> None:
    """Startup: the saved batch at once; then a fresh one each REFRESH_S, when the house is quiet."""
    import missions
    _load_saved()
    last = 0.0 if not clips else time.monotonic()
    while True:
        await asyncio.sleep(20)
        try:
            due = time.monotonic() - last >= (REFRESH_S if clips else 60)
            quiet = time.monotonic() - missions._last_activity >= QUIET_S
            if due and quiet:
                await refresh()
                last = time.monotonic()  # a failure waits for the next slot too, instead of retrying hot
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("greetings")
