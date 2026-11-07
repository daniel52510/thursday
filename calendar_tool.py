"""iCloud calendar access over CalDAV, per user.

Each user's credentials (Apple ID + an Apple app-specific password, never the
account password) live encrypted in the vault, saved from Settings. The admin
may still use ICLOUD_USERNAME / ICLOUD_APP_PASSWORD from .env; server startup
moves those into the vault. They are never logged or returned to the model.

Discovering the calendars costs several round trips to iCloud, so it happens
once per process and user; each question after that is one REPORT per calendar.

Changes are two-step and the confirmation is enforced here, not left to the
model: calendar_create / calendar_delete only store a pending change (delete
only accepts an id from this conversation's latest listing); calendar_confirm applies it
only if the user's *next* message is a plain yes. The agent sets TURN for each
user message; the model can't see or set it.
"""
from __future__ import annotations

import os
import re
import threading
import time
import uuid
from contextvars import ContextVar
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import caldav
import icalendar

import vault
from accounts import OWNER
from tools import ToolResult

ICLOUD_URL = os.getenv("ICLOUD_CALDAV_URL", "https://caldav.icloud.com")
DEFAULT_CALENDAR = os.getenv("ICLOUD_CALENDAR", "")
MAX_DAYS = 31
MAX_EVENTS = 25
PENDING_TTL_S = 180

# {"conversation": id, "turn": unique per user message, "user_said": text}
TURN: ContextVar[dict | None] = ContextVar("calendar_turn", default=None)
_YES = re.compile(
    r"^\W*(?:(?:uh+|um+|er+|oh|ah|well|hey|thursday|amaze|good|cool)\W+)*"
    r"(yes|yeah|yep|yup|ya|sure|ok(ay)?|alright|all\s+right|absolutely|definitely|affirmative|correct|perfect|please"
    r"|for\s+sure|works\s+for\s+me|you\s+got\s+it|make\s+it\s+happen"
    r"|go (ahead|for it)|do it|let's do it|please do|confirm(ed)?|that's (right|correct|it)|sounds good"
    r"|(?:start|run|launch|begin)(?:\s+(?:it|now|that|the\s+(?:mission|task|routine)|mission))?"
    r"|(add|save|book|schedule) it)\b",
    re.I,
)
# Words that turn a request into "add something to my calendar".
CREATE_INTENT = re.compile(
    r"\b(add|create|schedule|put|book|set( up)?|make|pencil|block)\b.*\b(calendar|appointment|event|meeting|reservation)s?\b", re.I
)
DELETE_INTENT = re.compile(
    r"\b(delete|remove|cancel|clear|drop|take off)\b.*\b(calendar|appointment|event|meeting|reservation)s?\b", re.I
)
# "Yes, but make it 4" is a change, not a confirmation.
_HEDGE = re.compile(r"\b(but|actually|instead|change|wait|not|don't)\b", re.I)

_lock = threading.Lock()
_calendars: dict[str, list] = {}  # user id -> that user's event calendars
CREDENTIAL = "apple_calendar"
_pending: dict[str, dict] = {}  # conversation id -> prepared change
_listed: dict[str, dict] = {}  # conversation id -> {event id: event} from the latest calendar_events


def env_credentials() -> dict | None:
    user, password = os.getenv("ICLOUD_USERNAME", ""), os.getenv("ICLOUD_APP_PASSWORD", "")
    return {"username": user, "app_password": password} if user and password else None


def discover(username: str, app_password: str) -> list:
    """The account's event calendars (iCloud also lists reminder lists here). Raises on bad credentials."""
    client = caldav.DAVClient(url=ICLOUD_URL, username=username, password=app_password, timeout=10)
    return [c for c in client.principal().calendars() if "VEVENT" in _components(c)]


def forget(user_id: str) -> None:
    """Drop a user's cached calendars (their credentials changed)."""
    with _lock:
        _calendars.pop(user_id, None)


def _get_calendars(user_id: str | None = None) -> list:
    # The agent puts the speaking user in TURN; alarms and warm-up are the admin's.
    user_id = user_id or (TURN.get() or {}).get("user") or OWNER
    with _lock:
        if user_id not in _calendars:
            creds = vault.shared().get(user_id, CREDENTIAL) or (env_credentials() if user_id == OWNER else None)
            if not creds:
                raise LookupError("not_configured")
            t = time.monotonic()
            _calendars[user_id] = discover(creds["username"], creds["app_password"])
            print(f"[calendar] discovered {len(_calendars[user_id])} calendars in {int((time.monotonic() - t) * 1000)} ms")
        return _calendars[user_id]


def warm_up() -> None:
    """Discover the admin's calendars ahead of the first question (~3.5 s against iCloud)."""
    try:
        _get_calendars(OWNER)
    except LookupError:
        pass  # not configured: calendar_events reports it when asked
    except Exception as exc:
        print(f"[calendar] warm-up failed, will retry on first use: {type(exc).__name__}")


def _components(cal) -> list[str]:
    try:
        return cal.get_supported_components()
    except Exception:
        return ["VEVENT"]


def _when(value, tz: ZoneInfo) -> tuple[str, bool]:
    """ISO string in the user's timezone, and whether it is an all-day date."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=tz)  # floating time: read as the user's local time
        return value.astimezone(tz).isoformat(timespec="minutes"), False
    return value.isoformat(), True


def calendar_events(args: Dict[str, Any]) -> ToolResult:
    """
    Args
        - start_date: YYYY-MM-DD, first day to list (default: today in the user's timezone)
        - days: number of days from start_date (default 1, max 31)
        - timezone: IANA timezone of the user (filled in by the agent from client context)
    Returns events sorted by start: title, start, end, all_day, location, calendar.
    """
    try:
        # His home timezone when the request carried none (Siri, the proactive loop): UTC read 8 pm as "1 in the morning".
        tz = ZoneInfo(str(args.get("timezone") or os.getenv("THURSDAY_TIMEZONE") or os.getenv("THURSDAY_TZ") or "UTC"))
    except ZoneInfoNotFoundError:
        return ToolResult(ok=False, tool_name="calendar_events", error="unknown_timezone")
    try:
        first = date.fromisoformat(str(args["start_date"])) if args.get("start_date") else datetime.now(tz).date()
    except ValueError:
        return ToolResult(ok=False, tool_name="calendar_events", error="bad_start_date (use YYYY-MM-DD)")
    days = max(1, min(int(args.get("days") or 1), MAX_DAYS))
    start = datetime.combine(first, datetime.min.time(), tz)
    end = start + timedelta(days=days)

    try:
        calendars = _get_calendars()
    except LookupError:
        return ToolResult(ok=False, tool_name="calendar_events", error="calendar_not_configured")

    t = time.monotonic()
    events = []
    for cal in calendars:
        # expand=True turns recurring events into one entry per occurrence in the range.
        for ev in cal.search(start=start.astimezone(timezone.utc), end=end.astimezone(timezone.utc), event=True, expand=True):
            comp = ev.icalendar_component
            begins, all_day = _when(comp.get("dtstart").dt, tz)
            dtend = comp.get("dtend")
            events.append({
                "title": str(comp.get("summary", "(no title)")),
                "start": begins,
                "end": _when(dtend.dt, tz)[0] if dtend else None,
                "all_day": all_day,
                "location": str(comp.get("location", "")) or None,
                "calendar": cal.get_display_name(),
                "recurring": "rrule" in comp or "recurrence-id" in comp,
                "_ref": (cal, str(ev.url)),
            })
    print(f"[calendar] {len(events)} events over {days} day(s) from {len(calendars)} calendars in "
          f"{int((time.monotonic() - t) * 1000)} ms")

    events.sort(key=lambda e: (not e["all_day"], e["start"]))
    total, events = len(events), events[:MAX_EVENTS]
    # Short ids the model can pass to calendar_delete. Only events in the
    # latest listing for this conversation can be deleted.
    listed = {}
    for i, e in enumerate(events, 1):
        e["id"] = f"e{i}"
        cal, url = e.pop("_ref")
        listed[e["id"]] = {"calendar": cal, "url": url, "event": dict(e)}
    turn = TURN.get()
    if turn:
        _listed[turn["conversation"]] = listed
    return ToolResult(
        ok=True,
        tool_name="calendar_events",
        data={
            "range": f"{first.isoformat()} to {(first + timedelta(days=days - 1)).isoformat()}",
            "timezone": str(tz),
            "events": events,
            "truncated": total > MAX_EVENTS,
        },
    )


def _spoken(dt: datetime | date) -> str:
    if isinstance(dt, datetime):
        return f"{dt:%A, %B} {dt.day} at {dt:%I:%M %p}".replace(" 0", " ")
    return f"{dt:%A, %B} {dt.day}"


def _pick_calendar(name: str):
    calendars = _get_calendars()
    wanted = (name or DEFAULT_CALENDAR).strip().lower()
    for cal in calendars:
        if cal.get_display_name().lower() == wanted:
            return cal
    if name:
        raise LookupError(f"no calendar named {name!r}; have: {', '.join(c.get_display_name() for c in calendars)}")
    return calendars[0]


def prepare_event(args: Dict[str, Any]) -> tuple:
    """(calendar, event, read_back) from calendar_create's args, nothing saved. Raises ValueError,
    ZoneInfoNotFoundError (bad input) or LookupError (no such calendar). Also used by approvals.py."""
    title = str(args.get("title") or "").strip()[:200]
    if not title or not args.get("start"):
        raise ValueError("missing title or start; ask the user")
    tz = ZoneInfo(str(args.get("timezone") or os.getenv("THURSDAY_TIMEZONE") or "UTC"))
    raw = str(args["start"]).strip()
    if "T" in raw or " " in raw:
        start: date | datetime = datetime.fromisoformat(raw)
        if start.tzinfo is None:
            start = start.replace(tzinfo=tz)
        end: date | datetime = start + timedelta(minutes=max(5, min(int(args.get("duration_minutes") or 60), 24 * 60)))
    else:
        start = date.fromisoformat(raw)
        end = start + timedelta(days=1)
    cal = _pick_calendar(str(args.get("calendar") or ""))
    location = str(args.get("location") or "").strip()[:200]
    read_back = f"{title}, {_spoken(start)}" + (f" until {_spoken(end).split(' at ')[-1]}" if isinstance(start, datetime) else ", all day")
    read_back += (f", at {location}" if location else "") + f", on the {cal.get_display_name()} calendar"
    return cal, {"summary": title, "dtstart": start, "dtend": end, "location": location}, read_back


def save_event(cal, ev: dict) -> None:
    vevent = icalendar.Event()
    vevent.add("uid", f"{uuid.uuid4()}@thursday")
    vevent.add("dtstamp", datetime.now(timezone.utc))
    for key in ("summary", "dtstart", "dtend"):
        vevent.add(key, ev[key])
    if ev["location"]:
        vevent.add("location", ev["location"])
    vcal = icalendar.Calendar()
    vcal.add("prodid", "-//THURSDAY//EN")
    vcal.add("version", "2.0")
    vcal.add_component(vevent)
    cal.save_event(ical=vcal.to_ical().decode())


def calendar_create(args: Dict[str, Any]) -> ToolResult:
    """
    Prepares an event; nothing is saved until calendar_confirm.
    Args
        - title (required)
        - start (required): YYYY-MM-DDTHH:MM local time, or YYYY-MM-DD for an all-day event
        - duration_minutes: default 60 (ignored for all-day)
        - location, calendar (display name; default ICLOUD_CALENDAR)
        - timezone: filled in by the agent from client context
    """
    turn = TURN.get()
    if not turn:
        return ToolResult(ok=False, tool_name="calendar_create", error="no_conversation")
    try:
        cal, event, read_back = prepare_event(args)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        return ToolResult(ok=False, tool_name="calendar_create", error=f"bad_input: {exc}")
    except LookupError as exc:
        return ToolResult(ok=False, tool_name="calendar_create", error=str(exc))
    _pending[turn["conversation"]] = {
        "kind": "create", "turn": turn["turn"], "at": time.monotonic(), "calendar": cal, "event": event,
        "read_back": read_back,
    }
    return ToolResult(
        ok=True,
        tool_name="calendar_create",
        data={
            "status": "NOT SAVED YET - awaiting the user's confirmation",
            "read_back": read_back,
            "next": "Read this back and ask the user to confirm. Call calendar_confirm only after they say yes.",
        },
    )


def calendar_delete(args: Dict[str, Any]) -> ToolResult:
    """
    Prepares deleting one event; nothing is removed until calendar_confirm.
    Args
        - event_id (required): an id ("e3") from this conversation's latest calendar_events result
    """
    turn = TURN.get()
    if not turn:
        return ToolResult(ok=False, tool_name="calendar_delete", error="no_conversation")
    listed = _listed.get(turn["conversation"], {})
    target = listed.get(str(args.get("event_id") or "").strip())
    if not target:
        return ToolResult(ok=False, tool_name="calendar_delete",
                          error="unknown_event_id: call calendar_events for that day first and use an id from it")
    ev = target["event"]
    if ev["recurring"]:
        # Deleting the resource would remove every occurrence of the series.
        return ToolResult(ok=False, tool_name="calendar_delete",
                          error="recurring_event: can't delete one occurrence yet; tell the user to remove it in the Calendar app")
    when = ev["start"] if ev["all_day"] else _spoken(datetime.fromisoformat(ev["start"]))
    read_back = f"delete {ev['title']}, {when}{', all day' if ev['all_day'] else ''}, from the {ev['calendar']} calendar"
    _pending[turn["conversation"]] = {
        "kind": "delete", "turn": turn["turn"], "at": time.monotonic(),
        "calendar": target["calendar"], "url": target["url"], "read_back": read_back,
    }
    return ToolResult(
        ok=True,
        tool_name="calendar_delete",
        data={
            "status": "NOT DELETED YET - awaiting the user's confirmation",
            "read_back": read_back,
            "next": "Read this back and ask the user to confirm. It is deleted only after they say yes.",
        },
    )


def plain_yes(said: str) -> bool:
    """A plain yes ("yes", "do it", "book it"), not "yes, but make it 4". Also used by approvals.py."""
    return bool(_YES.match(said or "")) and not _HEDGE.search(said or "")


def confirms_pending(conversation_id: str, said: str) -> bool:
    """True when this message is a plain yes to a change still waiting in this conversation.
    The agent then runs calendar_confirm itself instead of hoping the model does."""
    pending = _pending.get(conversation_id)
    return bool(pending) and time.monotonic() - pending["at"] <= PENDING_TTL_S \
        and bool(_YES.match(said)) and not _HEDGE.search(said)


def calendar_confirm(args: Dict[str, Any]) -> ToolResult:
    """Carries out the pending change if the user's latest message is a plain yes."""
    turn = TURN.get()
    pending = _pending.get(turn["conversation"]) if turn else None
    if not pending or time.monotonic() - pending["at"] > PENDING_TTL_S:
        if pending:
            del _pending[turn["conversation"]]
        return ToolResult(ok=False, tool_name="calendar_confirm", error="nothing_to_confirm")
    if pending["turn"] == turn["turn"]:
        return ToolResult(ok=False, tool_name="calendar_confirm",
                          error="not_confirmed: ask the user first and wait for their answer")
    said = turn["user_said"]
    if not _YES.match(said) or _HEDGE.search(said):
        return ToolResult(ok=False, tool_name="calendar_confirm",
                          error="not_confirmed: the user's reply was not a plain yes; if they changed details, call calendar_create again")
    del _pending[turn["conversation"]]

    t = time.monotonic()
    if pending["kind"] == "delete":
        pending["calendar"].event_by_url(pending["url"]).delete()
        _listed.pop(turn["conversation"], None)  # ids are stale now; list again before another delete
        print(f"[calendar] deleted event in {int((time.monotonic() - t) * 1000)} ms")
        return ToolResult(ok=True, tool_name="calendar_confirm", data={"status": "deleted", "event": pending["read_back"]})

    save_event(pending["calendar"], pending["event"])
    print(f"[calendar] created event in {int((time.monotonic() - t) * 1000)} ms")
    return ToolResult(ok=True, tool_name="calendar_confirm", data={"status": "saved", "event": pending["read_back"]})
