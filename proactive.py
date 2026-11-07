"""THURSDAY speaks first: the FRIDAY part. A loop that watches his calendar, his Apple Watch data and his open
promises, and decides what's worth telling him, and how.

Rules (each at most once per thing, remembered in brain/reminders.db so a restart doesn't repeat them):
  - calendar: an event about to start (10-16 min ahead)
  - workout: a new Apple Watch workout gets a short coach's comment (from health_check data)
  - health: HRV down and resting heart rate up on most of the last three days: an early warning, once a day
  - evening: at 9 pm, a wrap-up: tomorrow's first event, reminders still pending, promises still open

Delivery follows where he is (deliver): a push always (it reaches him anywhere over Tailscale), and out loud
as well when a satellite is free and it's daytime. Between 10 pm and 8 am pushes that aren't urgent are held
and sent at 8 am, so the phone stays quiet at night.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import httpx

logger = logging.getLogger("thursday.voice.proactive")  # under voice_ws's logger: its handler reaches docker logs
QUIET = (22, 8)        # local hours: non-urgent pushes wait for the morning
EVENING_HOUR = 21
TICK_S = 60  # presence changes are noticed within a minute (the helper pings once a minute)


def _zone():
    from zoneinfo import ZoneInfo
    return ZoneInfo(os.getenv("THURSDAY_TZ", "America/New_York"))


def _db() -> sqlite3.Connection:
    import reminders
    conn = reminders._connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS proactive_seen (key TEXT PRIMARY KEY, at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS proactive_held (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL,
            text TEXT NOT NULL, tag TEXT, at TEXT NOT NULL);""")
    return conn


def _first_time(key: str) -> bool:
    """True once per key: the rule's way of never saying the same thing twice."""
    with _db() as conn:
        if conn.execute("SELECT 1 FROM proactive_seen WHERE key = ?", (key,)).fetchone():
            return False
        conn.execute("INSERT INTO proactive_seen VALUES (?, ?)", (key, datetime.now(timezone.utc).isoformat()))
    return True


def _quiet(now: datetime) -> bool:
    return now.hour >= QUIET[0] or now.hour < QUIET[1]


async def deliver(title: str, text: str, *, tag: str = "bell", urgent: bool = False, say: str | None = None) -> None:
    """Reach him the way that fits where he is (presence.channel): out loud in the room when he's home and awake,
    a push when he's out or in a meeting, held for the morning when he's asleep (unless urgent)."""
    import approvals
    import presence
    state = await asyncio.to_thread(presence.now_state)
    how = presence.channel(state, urgent)
    if how == "hold":
        with _db() as conn:
            conn.execute("INSERT INTO proactive_held(title, text, tag, at) VALUES (?,?,?,?)",
                         (title, text, tag, datetime.now(_zone()).isoformat()))
        logger.info("proactive: '%s' held (asleep or night)", title)
        return
    if how == "speak" and await _speak(say or text):
        logger.info("proactive: said '%s' (he's home)", title)
        return
    await asyncio.to_thread(approvals.push, title, text, tags=(tag,), priority=4 if urgent else 3)
    logger.info("proactive: pushed '%s' (%s)", title, "in a meeting" if state["meeting"] else f"phone {state['phone']}")


async def _speak(line: str) -> bool:
    try:
        import mind
        import voice_ws
        from accounts import OWNER
        idle = [s for s in voice_ws.SATELLITES if s.state == "idle"]
        if not idle:
            return False
        style = mind.voice_style(OWNER)
        if style.get("voice") != "tara":
            import eridian
            line = eridian.translate(line)
        async with httpx.AsyncClient() as http:
            audio = await voice_ws.synthesize_ahead(http, line, style)
        for sat in idle:
            if audio and await sat.announce(audio=audio, cue="announce"):
                return True
    except Exception as exc:
        logger.warning("proactive: not spoken (%s)", type(exc).__name__)
    return False


async def _release_held() -> None:
    import approvals
    import presence
    if _quiet(datetime.now(_zone())) or await asyncio.to_thread(presence.asleep):
        return
    with _db() as conn:
        held = conn.execute("SELECT * FROM proactive_held ORDER BY id").fetchall()
        conn.execute("DELETE FROM proactive_held")
    for h in held:
        await asyncio.to_thread(approvals.push, h["title"], h["text"], tags=(h["tag"] or "bell",))
    if held:
        logger.info("proactive: sent %d held overnight", len(held))


# ---------------------------------------------------------------- rules
def _events(day: datetime) -> list[dict]:
    import calendar_tool
    r = calendar_tool.calendar_events({"start_date": day.date().isoformat(), "days": 1, "timezone": str(_zone())})
    return (r.data or {}).get("events", []) if r.ok else []


async def calendar_rule(now: datetime) -> None:
    for ev in await asyncio.to_thread(_events, now):
        if ev.get("all_day"):
            continue
        try:
            start = datetime.fromisoformat(str(ev["start"]))
        except ValueError:
            continue
        start = start if start.tzinfo else start.replace(tzinfo=_zone())
        mins = (start - now).total_seconds() / 60
        if 10 <= mins <= 16 and _first_time(f"cal:{ev.get('title')}:{ev['start']}"):
            where = f" at {ev['location']}" if ev.get("location") else ""
            await deliver("Coming up", f"{ev.get('title', 'Event')} in {round(mins)} minutes{where}.", tag="calendar")


async def workout_rule(now: datetime) -> None:
    import health
    since = (now - timedelta(hours=12)).astimezone(timezone.utc).isoformat()
    with health._connect() as conn:
        rows = conn.execute("SELECT start, end, value, note FROM samples WHERE metric = 'workout' AND end >= ? "
                            "ORDER BY start", (since,)).fetchall()
    for start, end, minutes, kind in rows:
        if minutes < 8 or not _first_time(f"workout:{start}"):
            continue  # a 3-minute accidental start isn't a workout
        line = await _coach_line(kind or "Workout", minutes, health.readiness())
        await deliver(f"{kind or 'Workout'} logged", line, tag="muscle")


async def _coach_line(kind: str, minutes: float, ready: dict | None) -> str:
    from agent_stream import CHAT_URL, MODEL, OLLAMA_KEEP_ALIVE, OLLAMA_OPTIONS
    fallback = f"Nice, {round(minutes)} minutes of {kind.lower()}. Logged."
    prompt = (f"You're his personal trainer. He just finished: {kind}, {round(minutes)} minutes. Today's readiness: "
              f"{ready or 'unknown'}. Write one or two short, warm sentences a coach would text: acknowledge it, and "
              "one useful pointer for recovery or the next session given his readiness. No emojis, no medical claims.")
    try:
        async with httpx.AsyncClient(timeout=30) as http:
            r = await http.post(CHAT_URL, json={"model": MODEL, "stream": False, "think": False,
                                                "keep_alive": OLLAMA_KEEP_ALIVE, "options": OLLAMA_OPTIONS,
                                                "messages": [{"role": "user", "content": prompt}]})
        return (r.json().get("message") or {}).get("content", "").strip() or fallback
    except (httpx.HTTPError, ValueError):
        return fallback


def stressed(days: list[dict]) -> bool:
    """Most of the last three days: HRV 12%+ under his 30-day baseline and resting HR 3+ bpm over it."""
    if len(days) < 10:
        return False
    recent, base = days[-3:], days[:-3]
    import health
    hrv_b, rhr_b = health._avg(base, "hrv"), health._avg(base, "resting_heart_rate")
    if not hrv_b or not rhr_b:
        return False
    bad = sum(1 for d in recent if d.get("hrv") is not None and d.get("resting_heart_rate") is not None
              and d["hrv"] < hrv_b * 0.88 and d["resting_heart_rate"] > rhr_b + 3)
    return bad >= 2


async def health_rule(now: datetime) -> None:
    import health
    if not 8 <= now.hour < 22:  # a morning-or-day message, never a night one
        return
    if stressed(health.daily(31)) and _first_time(f"health:{now.date()}"):
        await deliver("Your body's asking for rest",
                      "HRV has been low and your resting heart rate up for a few days. That often means stress, poor "
                      "sleep or a bug coming on. I'd keep today easy and get to bed early. If it keeps up, worth "
                      "checking with a doctor.", tag="heart")


async def evening_rule(now: datetime) -> None:
    if now.hour != EVENING_HOUR or not _first_time(f"evening:{now.date()}"):
        return
    import reminders
    from accounts import OWNER
    parts = []
    tomorrow = [e for e in await asyncio.to_thread(_events, now + timedelta(days=1)) if not e.get("all_day")]
    if tomorrow:
        t = tomorrow[0]
        when = datetime.fromisoformat(str(t["start"])).strftime("%-I:%M %p") if t.get("start") else ""
        parts.append(f"Tomorrow starts with {t.get('title', 'an event')} at {when}.")
    with reminders._connect() as conn:
        pending = conn.execute("SELECT COUNT(*) FROM reminders WHERE user_id = ? AND status = 'pending'",
                               (OWNER,)).fetchone()[0]
        promises = conn.execute("SELECT text FROM promises WHERE user_id = ? AND status = 'open' "
                                "ORDER BY created DESC LIMIT 2", (OWNER,)).fetchall()
    if pending:
        parts.append(f"{pending} reminder{'s' if pending > 1 else ''} still set.")
    if promises:
        parts.append("Still on my list: " + "; ".join(p[0] for p in promises))
    if parts:
        await deliver("Evening wrap-up", " ".join(parts), tag="crescent_moon", say=" ".join(parts[:2]))


_last_phone = {"state": None, "away_since": None}


async def welcome_rule(now: datetime) -> None:
    """Back home after 30+ minutes out: a short "welcome back" with what happened meanwhile, out loud."""
    import presence
    state = presence.phone()
    was, _last_phone["state"] = _last_phone["state"], state
    if state == "away" and was != "away":
        _last_phone["away_since"] = now
    if not (state == "home" and was == "away" and _last_phone["away_since"]):
        return
    gone = now - _last_phone["away_since"]
    _last_phone["away_since"] = None
    if gone < timedelta(minutes=30) or presence.asleep(now) or not 7 <= now.hour < 23:
        return
    bits = []
    since = (now - gone).astimezone(timezone.utc).isoformat()
    try:
        from agent_stream import initalize_db
        from accounts import OWNER
        done = [m for m in initalize_db().list_missions(OWNER, "done", limit=10) if (m.get("finished_at") or "") >= since]
        if done:
            bits.append(f"Your research on {done[0]['goal'][:60]} is done.")
    except Exception:
        pass
    import reminders
    from accounts import OWNER
    with reminders._connect() as conn:
        nxt = conn.execute("SELECT text, due FROM reminders WHERE user_id = ? AND status = 'pending' ORDER BY due "
                           "LIMIT 1", (OWNER,)).fetchone()
    if nxt:
        bits.append(f"Next reminder: {nxt['text']}, {reminders._say_when(datetime.fromisoformat(nxt['due']))}.")
    hours = gone.total_seconds() / 3600
    hello = "Welcome back." if hours < 3 else "Welcome home."
    if await _speak(" ".join([hello] + bits)):
        logger.info("proactive: welcomed him home after %.1f h", hours)


async def followup_rule(now: datetime) -> None:
    """He's home, it's daytime, and something in his life just happened: ask how it went, out loud, once."""
    import presence
    import reminders
    from accounts import OWNER
    if not 10 <= now.hour < 21 or presence.phone() != "home" or presence.asleep(now):
        return
    for f in reminders.due_followups(OWNER)[:1]:
        if not _first_time(f"followup:{f['id']}"):
            continue
        try:
            import mind
            import voice_ws
            style = mind.voice_style(OWNER)
            line = f"Hey, how did {f['event']} go?"
            if style.get("voice") != "tara":
                import eridian
                line = eridian.translate(line)
            async with httpx.AsyncClient() as http:
                audio = await voice_ws.synthesize_ahead(http, line, style)
            for sat in [s for s in voice_ws.SATELLITES if s.state == "idle"]:
                # In the conversation where he mentioned it, so his answer has the context.
                if audio and await sat.announce(audio=audio, conversation_id=f["conversation"], cue="announce"):
                    with reminders._connect() as conn:
                        conn.execute("UPDATE followups SET status = 'asked' WHERE id = ?", (f["id"],))
                    logger.info("proactive: asked how %s went", f["id"])
                    break
        except Exception as exc:
            logger.warning("proactive: follow-up not asked (%s)", type(exc).__name__)


async def run() -> None:
    """Every couple of minutes, each rule looks once. A rule's failure never stops the others."""
    from accounts import OWNER  # noqa: F401  (owner-only: everything here is his)
    await asyncio.sleep(30)
    while True:
        now = datetime.now(_zone())
        for rule in (_release_held, welcome_rule, followup_rule, calendar_rule, workout_rule, health_rule, evening_rule):
            try:
                await (rule() if rule is _release_held else rule(now))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("proactive rule %s", rule.__name__)
        await asyncio.sleep(TICK_S)
