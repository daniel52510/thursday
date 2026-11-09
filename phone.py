"""THURSDAY on his iPhone: "Hey Siri, THURSDAY" runs a Shortcut that dictates what he says, POSTs it to /api/phone,
speaks the reply, and does the one thing the reply asks of the phone:

  call     the Shortcut finds the contact on his iPhone (any contact, not the Twilio allowlist) and calls; iOS asks
           him to confirm before it dials
  message  read back first ("Text Sam: '...'. Send it?"); only his plain yes on the next turn sends it, checked
           here in code (confirm_message), so the Shortcut can send with no compose sheet
  open     a URL: his music app at the song / playlist / artist (one tap to play: no music app lets a Shortcut
           start an arbitrary track), or Apple Maps directions
  timer    an iPhone timer for N minutes
and keeps listening (listen) when THURSDAY asked him something back.

These tools exist only on phone turns (agent_stream adds PHONE_SCHEMAS when context["channel"] == "phone"), so the
satellite never offers to "call from your phone". Each tool only records what the phone should do (ACTION); nothing
here reaches his contacts or his messages: the iPhone does it, in front of him.
"""
from __future__ import annotations

import json
import os
import re
import time
from contextvars import ContextVar
from typing import Any, Dict
from urllib.parse import quote

ACTION: ContextVar[dict | None] = ContextVar("phone_action", default=None)
CONFIRM_S = 120
_pending: dict[str, dict] = {}  # user id -> a text waiting for his yes
PREFS_FILE = os.path.join(os.getenv("THURSDAY_BRAIN_DIR", "./brain"), "phone.json")
MUSIC_APPS = {
    "youtube": ("YouTube Music", "https://music.youtube.com/search?q={q}"),
    "apple": ("Apple Music", "https://music.apple.com/us/search?term={q}"),
    "spotify": ("Spotify", "https://open.spotify.com/search/{q}"),  # https like the rest: the Shortcut opens it only if it contains "http"
}


def prefs() -> dict:
    try:
        with open(PREFS_FILE) as f:
            p = json.load(f)
    except (OSError, ValueError):
        p = {}
    return {"music_app": p.get("music_app") if p.get("music_app") in MUSIC_APPS else "youtube"}


def set_prefs(music_app: str) -> dict:
    if music_app not in MUSIC_APPS:
        raise ValueError("music_app must be one of " + ", ".join(MUSIC_APPS))
    with open(PREFS_FILE, "w") as f:
        json.dump({"music_app": music_app}, f)
    return prefs()


def _record(action: dict) -> bool:
    """One phone action per turn: the last one asked for wins. False when this isn't a phone turn."""
    box = ACTION.get()
    if box is None:
        return False
    box.clear()
    box.update(action)
    return True


def _not_phone(name: str):
    from tools import ToolResult
    return ToolResult(ok=False, tool_name=name, error="only_from_his_iphone: this works when he asks through the "
                                                     "THURSDAY Shortcut on his phone")


def _contact(args: Dict[str, Any]) -> str:
    return re.sub(r"^(?:my|our)\s+", "", " ".join(str(args.get("contact") or "").split()), flags=re.I)[:60]


def phone_call(args: Dict[str, Any]):
    from tools import ToolResult
    who = _contact(args)
    if not who:
        return ToolResult(ok=False, tool_name="phone_call", error="no_contact: ask who to call")
    if not _record({"do": "call", "contact": who}):
        return _not_phone("phone_call")
    return ToolResult(ok=True, tool_name="phone_call", data={
        "status": "his phone is calling", "contact": who,
        "note": f"Say you're calling {who}, in a few words. His iPhone finds the contact and asks him to confirm."})


def phone_message(args: Dict[str, Any]):
    from tools import ToolResult
    who, body = _contact(args), " ".join(str(args.get("text") or "").split())[:600]
    if not who or not body:
        return ToolResult(ok=False, tool_name="phone_message", error="ask who it's to and what to say")
    if ACTION.get() is None:
        return _not_phone("phone_message")
    from calendar_tool import TURN
    _pending[(TURN.get() or {}).get("user", "")] = {"contact": who, "body": body, "at": time.monotonic()}
    return ToolResult(ok=True, tool_name="phone_message", data={
        "status": "NOT SENT YET - waiting for his yes", "to": who, "text": body,
        "note": "Read it back in one line, exactly: who, then the text, then ask 'Send it?'. It sends only after "
                "his yes (handled for you)."})


def confirm_message(user_id: str, said: str) -> dict | None:
    """His answer to "Send it?": the Shortcut's response (send, or not), or None when it isn't one."""
    from calendar_tool import plain_yes
    p = _pending.get(user_id)
    if not p or time.monotonic() - p["at"] > CONFIRM_S:
        _pending.pop(user_id, None)
        return None
    if plain_yes(said):
        del _pending[user_id]
        return {"speak": "Sent.", "do": "message", "contact": p["contact"], "body": p["body"]}
    if re.match(r"^\W*(?:no|nope|nah|don't|do not|cancel|never ?mind|stop)\b", said or "", re.I):
        del _pending[user_id]
        return {"speak": "Okay, not sending it."}
    _pending.pop(user_id, None)  # something else (a change, a new request): the model takes it from here
    return None


_NOT_ASKED = re.compile(r"remix|edit|karaoke|tribute|cover|instrumental|live|sped up|slowed|8-bit|lullaby", re.I)
_ENTITY = {"artist": ("musicArtist", "artistLinkUrl", "artistName"), "album": ("album", "collectionViewUrl", "collectionName"),
           "song": ("song", "trackViewUrl", "trackName")}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s*[(\[].*?[)\]]", "", (text or "").lower())).strip()


def _apple_page(query: str, kind: str, artist: str = "", exact: bool = False) -> str | None:
    """The Apple Music page itself (the artist, album or song, play button right there), from Apple's public
    iTunes API, instead of a search results page. None when nothing fits: then the search page.
    Its text search ranks covers and club edits above the real song (the original "The Less I Know The Better"
    wasn't in its top 8), so with an artist it looks in that artist's own catalog and matches the title there."""
    import difflib
    import httpx
    entity, link, name = _ENTITY.get(kind, _ENTITY["song"])
    api = "https://itunes.apple.com"

    def get(path: str, **params) -> list[dict]:
        try:
            return httpx.get(f"{api}/{path}", params={"country": "us", **params}, timeout=4).json().get("results") or []
        except (httpx.HTTPError, ValueError):
            return []

    who = artist or (query if kind == "artist" else "")
    artist_hit = next((x for x in get("search", term=who, entity="musicArtist", limit=3)), None) if who else None
    if kind == "artist":
        if not artist_hit or not artist_hit.get("artistLinkUrl") or (exact and _norm(artist_hit.get("artistName", "")) != _norm(who)):
            return None  # a guess at "is this an artist?" that didn't name one exactly
        return str(artist_hit["artistLinkUrl"]).replace("?uo=4", "")
    if not artist_hit:
        return None  # a song or album with no artist: its search guessed piano covers; the search page is safer
    title = query
    if artist and artist.lower() in query.lower():
        title = re.sub(re.escape(artist), "", query, flags=re.I).strip()
    pool = [x for x in get("lookup", id=artist_hit["artistId"], entity=entity, limit=200) if x.get(link)]
    asked_for = bool(_NOT_ASKED.search(query))
    want = _norm(title)

    def score(x: dict) -> float:
        got = _norm(str(x.get(name) or ""))
        version_penalty = 0.5 if (not asked_for and _NOT_ASKED.search(str(x.get(name) or ""))) else 0
        return difflib.SequenceMatcher(None, want, got).ratio() + (0.3 if got == want else 0) - version_penalty
    best = max(pool, key=score, default=None)
    if not best or score(best) < 0.6:
        return None
    return str(best[link]).replace("&uo=4", "").replace("?uo=4", "")


def _sounds_like_artist(query: str) -> bool:
    """His own words say it's an artist: "play some X", "music by X", "X's music", "anything by X"."""
    from calendar_tool import TURN
    said = ((TURN.get() or {}).get("user_said") or "").lower()
    q = re.escape(query.lower())
    return bool(re.search(rf"\b(?:some|music by|songs by|anything by|stuff by|something by)\s+{q}\b|\b{q}'?s?\s+(?:music|songs)\b", said))


def phone_music(args: Dict[str, Any]):
    from tools import ToolResult
    query = " ".join(str(args.get("query") or "").split())[:120]
    kind = str(args.get("kind") or "").lower()
    artist = " ".join(str(args.get("artist") or "").split())[:80]
    if not query:
        return ToolResult(ok=False, tool_name="phone_music", error="ask what to play")
    if ACTION.get() is None:
        return _not_phone("phone_music")
    music_app = prefs()["music_app"]
    app, url = MUSIC_APPS[music_app]
    page = None
    if music_app == "apple":
        # The model often sends just his words ("The Less I Know The Better by Tame Impala", "the album Currents"),
        # so the kind and artist are read from them too.
        if m := re.match(r"^(?:the\s+)?album\s+(.+)$", query, re.I):
            query, kind = m[1], "album"
        if not artist and (m := re.match(r"^(.+?)\s+by\s+(.+)$", query, re.I)):
            query, artist = m[1], m[2]
            kind = kind if kind in ("song", "album") else "song"
        if kind in _ENTITY:
            page = _apple_page(query, kind, artist)
        elif not artist and _sounds_like_artist(query):  # "play some Tame Impala", "music by Drake"
            page = _apple_page(query, "artist", exact=True)
        # Otherwise a bare title ("play Blinding Lights"): an artist is literally named that, so no guessing:
        # the search page, where his song is the first hit.
    _record({"do": "open", "url": page or url.format(q=quote(query))})
    return ToolResult(ok=True, tool_name="phone_music", data={
        "opening": query, "app": app, "straight_to_it": bool(page),
        "note": f"Say you're pulling up {query} in {app}" + ("; he taps play." if page else
                "; he picks it and taps play (a Shortcut can't start the track itself).")})


def phone_directions(args: Dict[str, Any]):
    from tools import ToolResult
    where = " ".join(str(args.get("destination") or "").split())[:160]
    if not where:
        return ToolResult(ok=False, tool_name="phone_directions", error="ask where to")
    if not _record({"do": "open", "url": f"https://maps.apple.com/?daddr={quote(where)}&dirflg=d"}):
        return _not_phone("phone_directions")
    return ToolResult(ok=True, tool_name="phone_directions", data={
        "to": where, "note": "Say you're starting directions in Maps, in a few words."})


def phone_timer(args: Dict[str, Any]):
    from tools import ToolResult
    try:
        minutes = float(args.get("minutes") or 0)
    except (TypeError, ValueError):
        minutes = 0
    if not 0 < minutes <= 24 * 60:
        return ToolResult(ok=False, tool_name="phone_timer", error="minutes must be between 1 and 1440")
    if not _record({"do": "timer", "minutes": round(minutes, 2)}):
        return _not_phone("phone_timer")
    return ToolResult(ok=True, tool_name="phone_timer", data={
        "minutes": minutes, "note": "Confirm the timer in a few words."})


def _present(out: dict) -> dict:
    """Only the fields with something in them: a Shortcut's "has any value" is true for an empty string, so a blank
    url made Open URLs fail on every plain question (2026-11-09). speak is always there."""
    return {k: v for k, v in out.items() if k == "speak" or v not in ("", 0, None, False)}


def respond(reply: str) -> dict:
    """What the Shortcut gets: what to say, the phone action (flat keys, simple for Shortcuts), and whether to listen
    again (he was asked something)."""
    action = ACTION.get() or {}
    speak = reply.strip() or "Sorry, I didn't get anything back."
    return _present({"speak": speak, "do": action.get("do", ""), "contact": action.get("contact", ""),
            "body": action.get("body", ""), "url": action.get("url", ""), "minutes": action.get("minutes", 0),
            # "yes" or empty, not true/false: a Shortcut's If can test it with "has any value", no typing.
            "listen": "yes" if speak.rstrip().endswith("?") and not action else ""})


PHONE_SCHEMAS = [
    {"type": "function", "function": {
        "name": "phone_call", "description": "Call someone from his iPhone ('call Mom', 'call Jordana'). His phone "
                                             "finds them in his contacts and confirms. Not Twilio: that's call_place.",
        "parameters": {"type": "object", "properties": {"contact": {"type": "string", "description": "The name as he said it"}},
                       "required": ["contact"]}}},
    {"type": "function", "function": {
        "name": "phone_message", "description": "Text someone from his iPhone ('tell Sam I'm running late', 'text "
                                                "Mom happy birthday'). Write the text as him, short.",
        "parameters": {"type": "object", "properties": {"contact": {"type": "string"}, "text": {"type": "string"}},
                       "required": ["contact", "text"]}}},
    {"type": "function", "function": {
        "name": "phone_music", "description": "Play music on his iPhone: a song, playlist, album or artist ('play "
                                              "some Tame Impala', 'put on my workout playlist').",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "The song, album or artist name ('The Less I Know The Better')"},
            "artist": {"type": "string", "description": "Who it's by, when he said or you know it ('Tame Impala')"},
            "kind": {"type": "string", "enum": ["song", "artist", "album", "playlist"]}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "phone_directions", "description": "Directions in Apple Maps on his iPhone ('take me to Whole Foods', "
                                                   "'directions home').",
        "parameters": {"type": "object", "properties": {"destination": {"type": "string"}}, "required": ["destination"]}}},
    {"type": "function", "function": {
        "name": "phone_timer", "description": "A timer on his iPhone ('timer for 10 minutes').",
        "parameters": {"type": "object", "properties": {"minutes": {"type": "number"}}, "required": ["minutes"]}}},
]
