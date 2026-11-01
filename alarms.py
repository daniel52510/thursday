"""Alarm scheduler: rings connected satellites at each alarm's local time
with a spoken morning briefing, then the satellite listens for follow-ups.

Alarms live in the alarms table (memory.py), set from the web UI ("briefing":
chime + briefing) or by asking her ("wake": a soft ping, then she wakes you
herself, remembering why you wanted to be up; see alarm_tools.py).
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
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx

import accounts
from agent_loop import MODEL, OLLAMA_KEEP_ALIVE, OLLAMA_OPTIONS, _post_ollama, initalize_db
from agent_stream import STYLE, _system_prompt, stream_prompt, strip_voice_tags
from voice_ws import SATELLITES, synthesize_ahead

POLL_S = 15
RING_WINDOW = timedelta(minutes=2)
PREP_LEAD = timedelta(minutes=5)
PREP_GRACE = timedelta(seconds=45)  # past ring time, stop waiting for preparation and brief live
DEFAULT_BRIEF = "the weather, my calendar for today, and the top three news headlines"
NUDGE_GAP_S = 30  # quiet after her last try (she spoke, listened, heard nothing) before she tries again
MAX_NUDGES = 6
CHIME_FROM = 4  # nudges from this one on open with the full alarm chime, not the soft ping
# Rides along in every wake-up prompt, so her replies to what the user mumbles back see it in the history.
ANSWER_RULES = (" When I answer: 'snooze' or 'five more minutes' means alarm_snooze; clearly up ('I'm up', "
                "'I'm awake', 'turn it off') means alarm_dismiss and a short send-off; a mumble or 'mm' isn't awake, "
                "so say so, dryly, and ask again.")

logger = logging.getLogger("thursday.voice")

# (alarm id, ring time, brief) -> prepared briefing / preparation in flight.
# Keyed on the brief too, so editing an alarm after preparation re-prepares it.
_prepared: dict[tuple, dict] = {}
_preparing: dict[tuple, asyncio.Task] = {}
# user id -> her wake-up in progress: she keeps trying until they're up or snooze.
_wakeups: dict[str, asyncio.Task] = {}


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


def wake_prompt(alarm: dict, at: datetime) -> str:
    """For an alarm she set herself: she wakes the user, remembering why they wanted to be up."""
    local = at.astimezone(ZoneInfo(alarm["timezone"]))
    when = f"{local:%A, %B} {local.day}, {local:%I:%M %p}".replace(" 0", " ")
    why = f' When I asked you to set it, I said: "{alarm["brief"]}".' if alarm["brief"] else ""
    return f"(It's {when}: the alarm you set for me. I'm still asleep; wake me up.{why}{ANSWER_RULES})"


def snooze_prompt(alarm: dict, at: datetime) -> str:
    """A snooze running out: she's already woken the user once and let them drift off again."""
    why = f' I wanted to be up because: "{alarm["brief"]}".' if alarm["brief"] else ""
    return (f"(My snooze is up. You woke me a little while ago and let me sleep a few more minutes; "
            f"wake me again.{why}{ANSWER_RULES})")


def nudge_prompt(alarm: dict, tries: int, answered: bool = False) -> str:
    """She tried to wake the user and heard nothing back, or they answered and went quiet."""
    why = f' I wanted to be up because: "{alarm["brief"]}".' if alarm["brief"] else ""
    if answered:
        return ("(I answered you a bit ago but never said I'm up, and I've gone quiet: I may have drifted off. "
                f"Check on me, loud enough that I'd hear it.{why}{ANSWER_RULES})")
    return (f"(I didn't answer; I'm still asleep. You've tried {tries} time{'s' if tries > 1 else ''} already. "
            f"Wake me again, a little more insistent than last time.{why}{ANSWER_RULES})")


def _write_nudges(alarm: dict, user: dict, first_line: str) -> list[str]:
    """MAX_NUDGES escalating tries for if they sleep through her first, in one LLM pass."""
    said = f' You just tried to wake him: "{first_line}".' if first_line else " You just tried to wake him."
    why = f' He wanted to be up because: "{alarm["brief"]}".' if alarm["brief"] else ""
    raw = _post_ollama({
        "model": MODEL,
        "system": f"{_system_prompt(user)}\n\n{STYLE['wake']}",
        "prompt": f"{said} He didn't answer.{why} Write {MAX_NUDGES} more tries in your voice, for if he keeps "
                  "sleeping, each more insistent than the last: first warm, then urgent, then loud, the last ones "
                  "impossible to sleep through. Rocky's Eridian English throughout, never mean. They come about "
                  "a minute apart, so never state a clock time or how much time is left. Each 1 or 2 "
                  "short sentences, each different, each ending so he answers. "
                  'Reply as JSON: {"lines": ["...", "..."]}',
        "format": "json",
        "stream": False,
        "think": False,
        "options": OLLAMA_OPTIONS,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    })
    lines = json.loads(raw).get("lines") or []
    return [" ".join(str(line).split()) for line in lines if str(line).strip()][:MAX_NUDGES]


async def _prepare_nudges(alarm: dict, first_line: str, out: list) -> None:
    """Fill `out` with (line, audio) in order, while her first try plays and she listens:
    the first nudge isn't due for ~50 s, and each plays as soon as it's needed."""
    started = time.monotonic()
    user = accounts.shared().get_user(alarm["user_id"]) or accounts.shared().get_user(accounts.OWNER)
    try:
        lines = await asyncio.to_thread(_write_nudges, alarm, user, first_line)
        async with httpx.AsyncClient() as http:
            for line in lines:
                out.append((line, await synthesize_ahead(http, line)))
        logger.info("wake-up %s: %d nudges prepared in %d ms", alarm["time"], len(out),
                    round((time.monotonic() - started) * 1000))
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("wake-up %s: preparing nudges failed; they'll be written live", alarm["time"])


def ring_prompt(alarm: dict, at: datetime) -> tuple[str, str]:
    """The prompt and stream_prompt mode for this alarm's style."""
    if alarm.get("style") == "wake":
        return wake_prompt(alarm, at), "wake"
    if alarm.get("style") == "snooze":
        return snooze_prompt(alarm, at), "wake"
    return briefing_prompt(alarm, at), "briefing"


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
    prompt, mode = ring_prompt(alarm, at)
    title = "Wake-up" if mode == "wake" else "Morning briefing"
    conv = db.create_conversation(f"{title} · {local:%a %b} {local.day}", alarm["user_id"])
    reply, tools = "", []
    # The briefing is the alarm owner's: their facts, memories and calendar.
    user = accounts.shared().get_user(alarm["user_id"]) or accounts.shared().get_user(accounts.OWNER)
    async for ev in stream_prompt(prompt, mode=mode, conversation_id=conv["id"], context=_context(alarm), user=user):
        if ev["type"] == "tool":
            tools.append(ev["name"])
        elif ev["type"] == "done":
            reply = ev["reply"]
    llm_ms = round((time.monotonic() - started) * 1000)
    async with httpx.AsyncClient() as http:
        audio = await synthesize_ahead(http, reply)
    logger.info("briefing for %s prepared: llm_ms=%d tts_ms=%d tools=%s chars=%d chunks=%d", alarm["time"], llm_ms,
                round((time.monotonic() - started) * 1000) - llm_ms, ",".join(tools) or "none", len(reply), len(audio))
    return {"conversation_id": conv["id"], "audio": audio, "prompt": prompt, "reply": reply}


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
    prompt, mode = ring_prompt(alarm, at)
    # Her own wake-ups start with the soft ping, not the alarm chime.
    cue = "announce" if mode == "wake" else "alarm"
    for sat in list(SATELLITES):
        if prepared and prepared["audio"]:
            conv_id = prepared["conversation_id"]
            ok = await sat.announce(audio=prepared["audio"], conversation_id=conv_id, cue=cue)
        else:
            conv_id = initalize_db().create_conversation("Wake-up" if mode == "wake" else "Morning briefing",
                                                         alarm["user_id"])["id"]
            ok = await sat.announce(prompt=prompt, conversation_id=conv_id, context=_context(alarm), cue=cue, mode=mode)
        if ok:
            _prepared.pop(key, None)
            logger.info("alarm %s rang with %s briefing", alarm["time"], "a prepared" if prepared else "a live")
            if mode == "wake":
                _start_wakeup(alarm, sat, conv_id, prepared["reply"] if prepared else "")
            return True
    return False


def end_wakeup(user_id: str) -> bool:
    """They're up, or snoozed: stop trying to wake them. Safe from a tool's thread."""
    task = _wakeups.pop(user_id, None)
    if task:
        task.get_loop().call_soon_threadsafe(task.cancel)
    return task is not None


def _start_wakeup(alarm: dict, sat, conversation_id: str, first_line: str = "") -> None:
    end_wakeup(alarm["user_id"])
    _wakeups[alarm["user_id"]] = asyncio.create_task(_keep_waking(alarm, sat, conversation_id, first_line))


async def _keep_waking(alarm: dict, sat, conversation_id: str, first_line: str = "") -> None:
    """Try again, in the same conversation, each time the satellite has been quiet
    NUDGE_GAP_S since her last try: she spoke, listened, and heard nothing (or they
    answered and drifted off). Nudges are written and synthesized in the background
    (_prepare_nudges) so they play at once; one that isn't ready, or a check-in after
    they answered, is written live. Ends when they dismiss or snooze (end_wakeup),
    after MAX_NUDGES, or if the satellite goes away or its mic is muted."""
    tried_at, tries = time.monotonic(), 0
    ready: list[tuple[str, list[bytes]]] = []
    writer = asyncio.create_task(_prepare_nudges(alarm, first_line, ready))
    db = initalize_db()
    try:
        while tries < MAX_NUDGES:
            # A fresh idle after her last try, then enough quiet.
            while not (sat.state == "idle" and sat.idle_since > tried_at
                       and time.monotonic() - sat.idle_since >= NUDGE_GAP_S):
                if sat not in SATELLITES:
                    logger.info("wake-up %s: satellite gone, stopping", alarm["time"])
                    return
                await asyncio.sleep(1)
            if sat.sat_status.get("muted"):
                logger.info("wake-up %s: mic muted, stopping", alarm["time"])
                return
            answered = sat.heard_at > tried_at
            prompt = nudge_prompt(alarm, tries + 1, answered)
            cue = "alarm" if tries + 1 >= CHIME_FROM else "announce"
            # The prepared lines assume silence; after an answer she reacts to it live.
            line, audio = ready[tries] if len(ready) > tries and not answered else ("", None)
            if audio:
                ok = await sat.announce(audio=audio, conversation_id=conversation_id, cue=cue)
            else:
                ok = await sat.announce(prompt=prompt, conversation_id=conversation_id,
                                        context=_context(alarm), cue=cue, mode="wake")
            if not ok:
                continue  # busy this instant; a conversation in progress will idle again
            if audio:  # live tries are saved by stream_prompt; prepared ones once they've played
                db.log_message(role="user", content=prompt, conversation_id=conversation_id)
                db.log_message(role="assistant", content=strip_voice_tags(line), conversation_id=conversation_id)
            tried_at, tries = time.monotonic(), tries + 1
            logger.info("wake-up %s: %s, nudge %d/%d (%s)", alarm["time"], "answered then quiet" if answered
                        else "no answer", tries, MAX_NUDGES, "prepared" if audio else "live")
        logger.info("wake-up %s: gave up after %d nudges", alarm["time"], MAX_NUDGES)
    finally:
        writer.cancel()
        if _wakeups.get(alarm["user_id"]) is asyncio.current_task():
            _wakeups.pop(alarm["user_id"], None)


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
