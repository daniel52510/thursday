"""Alarm scheduler: rings connected satellites at each alarm's local time
with a spoken morning briefing, then the satellite listens for follow-ups.

Alarms live in the alarms table (memory.py) and are edited from the web UI.
Every POLL_S the scheduler:
  - PREP_LEAD before an alarm, writes the briefing (LLM + calendar/weather/
    news tools) in its own conversation and synthesizes all of its audio, so
    at ring time it plays straight after the chime with no wait and no gaps
    (1.7B TTS runs at about real time, so a 60-90 s briefing needs minutes);
  - at the alarm's time, rings an idle satellite with that audio. If nothing
    was prepared (alarm set at the last minute, or preparation failed), the
    satellite writes it live instead, paced like any other reply.
A busy satellite (mid-conversation) is retried until the alarm is
RING_WINDOW old, so a late poll or a short chat never skips it, but a Mac
that was asleep at 7:00 doesn't ring at noon either.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx

from agent_loop import initalize_db
from agent_stream import stream_prompt
from voice_ws import SATELLITES, synthesize_ahead

POLL_S = 15
RING_WINDOW = timedelta(minutes=2)
PREP_LEAD = timedelta(minutes=5)
PREP_GRACE = timedelta(seconds=45)  # past ring time, stop waiting for preparation and brief live
DEFAULT_BRIEF = "the weather, my calendar for today, and the top three news headlines"

logger = logging.getLogger("thursday.voice")

# (alarm id, ring time, brief) -> prepared briefing / preparation in flight.
# Keyed on the brief too, so editing an alarm after preparation re-prepares it.
_prepared: dict[tuple, dict] = {}
_preparing: dict[tuple, asyncio.Task] = {}


def scheduled_today(alarm: dict, now_utc: datetime) -> datetime | None:
    """When the alarm is due today in its timezone (as UTC), or None if not today."""
    tz = ZoneInfo(alarm["timezone"])
    local = now_utc.astimezone(tz)
    if alarm["days"] and local.weekday() not in alarm["days"]:
        return None
    hour, minute = map(int, alarm["time"].split(":"))
    return local.replace(hour=hour, minute=minute, second=0, microsecond=0).astimezone(timezone.utc)


def _not_fired_since(alarm: dict, at: datetime) -> bool:
    last = alarm.get("last_fired_at")
    return not last or datetime.fromisoformat(last) < at


def is_due(alarm: dict, now_utc: datetime) -> bool:
    if not alarm["enabled"]:
        return False
    at = scheduled_today(alarm, now_utc)
    return at is not None and at <= now_utc < at + RING_WINDOW and _not_fired_since(alarm, at)


def is_upcoming(alarm: dict, now_utc: datetime) -> bool:
    """Within PREP_LEAD of ringing: time to prepare the briefing."""
    if not alarm["enabled"]:
        return False
    at = scheduled_today(alarm, now_utc)
    return at is not None and at - PREP_LEAD <= now_utc < at and _not_fired_since(alarm, at)


def _key(alarm: dict, at: datetime) -> tuple:
    return (alarm["id"], at.isoformat(), alarm["brief"])


def briefing_prompt(alarm: dict, at: datetime) -> str:
    local = at.astimezone(ZoneInfo(alarm["timezone"]))
    when = f"{local:%A, %B} {local.day} at {local:%I:%M %p}".replace(" 0", " ")
    return (
        f"It's {when} and my alarm just went off. Give me my morning briefing covering "
        f"{alarm['brief'] or DEFAULT_BRIEF}. Use your tools for each topic so everything is current. "
        "If my location is unknown, skip the weather and say so in one short line."
    )


def _context(alarm: dict) -> dict:
    # A satellite's own context (sent on connect) knows where the user is.
    location = next((s._context.get("location") for s in SATELLITES if s._context and s._context.get("location")), "")
    return {"timezone": alarm["timezone"], "location": location or os.getenv("THURSDAY_LOCATION", "")}


async def prepare(alarm: dict, at: datetime) -> dict:
    """Write the briefing in its own conversation and synthesize all of it."""
    started = time.monotonic()
    db = initalize_db()
    local = at.astimezone(ZoneInfo(alarm["timezone"]))
    conv = db.create_conversation(f"Morning briefing · {local:%a %b} {local.day}")
    prompt = briefing_prompt(alarm, at)
    reply, tools = "", []
    async for ev in stream_prompt(prompt, mode="briefing", conversation_id=conv["id"], context=_context(alarm)):
        if ev["type"] == "tool":
            tools.append(ev["name"])
        elif ev["type"] == "done":
            reply = ev["reply"]
    llm_ms = round((time.monotonic() - started) * 1000)
    async with httpx.AsyncClient() as http:
        audio = await synthesize_ahead(http, reply)
    logger.info("briefing for %s prepared: llm_ms=%d tts_ms=%d tools=%s chars=%d chunks=%d", alarm["time"], llm_ms,
                round((time.monotonic() - started) * 1000) - llm_ms, ",".join(tools) or "none", len(reply), len(audio))
    return {"conversation_id": conv["id"], "audio": audio, "prompt": prompt}


def _start_prep(alarm: dict, at: datetime) -> None:
    key = _key(alarm, at)
    if key in _prepared or key in _preparing:
        return

    async def run() -> None:
        try:
            _prepared[key] = await prepare(alarm, at)
        except Exception:
            logger.exception("briefing preparation for %s failed; will brief live", alarm["time"])
        finally:
            _preparing.pop(key, None)

    _preparing[key] = asyncio.create_task(run())


async def _ring(alarm: dict, at: datetime, now: datetime) -> bool:
    key = _key(alarm, at)
    if key in _preparing and now < at + PREP_GRACE:
        return False  # nearly done: wait a tick rather than brief live
    prepared = _prepared.get(key)
    for sat in list(SATELLITES):
        if prepared and prepared["audio"]:
            ok = await sat.announce(audio=prepared["audio"], conversation_id=prepared["conversation_id"])
        else:
            conv = initalize_db().create_conversation("Morning briefing")
            ok = await sat.announce(prompt=briefing_prompt(alarm, at), conversation_id=conv["id"], context=_context(alarm))
        if ok:
            _prepared.pop(key, None)
            logger.info("alarm %s rang with %s briefing", alarm["time"], "a prepared" if prepared else "a live")
            return True
    return False


async def run_scheduler() -> None:
    db = initalize_db()
    while True:
        now = datetime.now(timezone.utc)
        try:
            for alarm in db.list_alarms():
                if is_upcoming(alarm, now):
                    _start_prep(alarm, scheduled_today(alarm, now))
                    continue
                if not is_due(alarm, now):
                    continue
                if not SATELLITES:
                    logger.warning("alarm %s %s due but no satellite is connected", alarm["time"], alarm["timezone"])
                    continue
                if not await _ring(alarm, scheduled_today(alarm, now), now):
                    logger.info("alarm %s due; satellite busy or briefing still preparing, retrying", alarm["time"])
                    continue
                # A one-off alarm turns itself off once it has rung.
                db.update_alarm(alarm["id"], last_fired_at=now.isoformat(), **({} if alarm["days"] else {"enabled": False}))
            # Drop briefings whose alarm passed without ringing (no satellite, deleted alarm).
            for key in [k for k in _prepared if datetime.fromisoformat(k[1]) + RING_WINDOW < now]:
                _prepared.pop(key, None)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("alarm scheduler tick failed")
        await asyncio.sleep(POLL_S)
