"""YouTube Music: THURSDAY plays, pauses, skips, finds and suggests music, through YTMDesktop on the Mac.

YTMDesktop (/Applications, v2.0.12, re-signed ad hoc) is YouTube Music's own web player with Daniel's
Premium account; its Companion Server (:9863, /api/v1) takes commands from a paired app. Pairing is
once, by hand: `docker exec -it thursday-core python music.py pair`, then he confirms the code in
the app; the token goes to the vault (never .env, never logged). The server listens on 0.0.0.0, so
"companion authorization" stays off in the app after pairing: nobody else can pair.

Finding things is ytmusicapi (pinned), signed out: a search gives the video or playlist id, and
"changeVideo" plays it. A song plays with its radio queued behind it ("RDAMVM" + id), so it keeps
going with music like it. /state allows one read per 5 s, so the last state is cached.

Where it plays (music_output): on the satellite by default. THURSDAY Music Tap (mac/music_tap, a
launchd agent) taps the app's sound with the Mac muted and streams 24 kHz mono to /ws/music-tap;
serve_tap relays it to every satellite ("MUS0" + PCM frames), which mixes it under his voice, ducks
it locally while anyone talks, and cancels it from the mic, so the wake word works over music. On
the Mac, the tap stops muting and ducking is the app's volume: DUCK_VOLUME while anyone talks, the
old level after the reply (voice_ws.Session.set_state).
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import os
import sys
import threading
import time
from typing import Any, Dict

import httpx
import numpy as np

from accounts import OWNER
from tools import ToolResult

YTMD = os.getenv("YTMD_URL", "http://host.docker.internal:9863/api/v1").rstrip("/")
APP = {"appId": "thursday", "appName": "THURSDAY", "appVersion": "1.0.0"}
CREDENTIAL = "ytmd"
DUCK_VOLUME = int(os.getenv("MUSIC_DUCK_VOLUME", "20"))
STATE_EVERY_S = 5.2  # the app's limit: one /state per 5 s
PLAYING = 1          # player.trackState: -1 unknown, 0 paused, 1 playing, 2 buffering

TAP_TOKEN = os.getenv("MUSIC_TAP_TOKEN", "")
TAP_SILENCE_S = 1.5  # this long under SILENT_RMS and the satellite stops getting frames (its buffer drains)
SILENT_RMS = 30.0
OUTPUT = {"where": os.getenv("MUSIC_OUTPUT", "satellite")}  # "satellite" or "mac"

logger = logging.getLogger("thursday.voice")
_taps: set = set()  # connected music taps (normally one)
# Is music playing, as far as our own commands know: no network, so a request can ask it on the spot
# (agent_stream: a plain "volume to 50%" is music only while music plays). The cached state agrees when fresh.
_playing = {"on": False}


def playing_hint() -> bool:
    st = _state["value"]
    if st is not None and time.monotonic() - _state["at"] < 20:
        return ((st.get("player") or {}).get("trackState")) == PLAYING
    return _playing["on"]
_loop: asyncio.AbstractEventLoop | None = None
_state: dict = {"at": 0.0, "value": None}
_duck: dict = {"restore": None}  # the volume to put back after a turn, if we lowered it
_lock = threading.Lock()
_yt = None


def _token() -> str | None:
    import vault  # on first use
    creds = vault.shared().get(OWNER, CREDENTIAL) if vault.shared().available else None
    return (creds or {}).get("token")


def _cmd(command: str, data: Any = None) -> None:
    token = _token()
    if not token:
        raise RuntimeError("not_paired: run `python music.py pair` in thursday-core")
    body = {"command": command, **({"data": data} if data is not None else {})}
    httpx.post(f"{YTMD}/command", json=body, headers={"Authorization": token}, timeout=3).raise_for_status()


def state(fresh: bool = False) -> dict | None:
    """The player's state, at most STATE_EVERY_S old (or as fresh as the limit allows)."""
    with _lock:
        age = time.monotonic() - _state["at"]
        if _state["value"] is not None and (age < STATE_EVERY_S or not fresh) and age < 60:
            return _state["value"]
        if age < STATE_EVERY_S:
            time.sleep(STATE_EVERY_S - age)
        token = _token()
        if not token:
            return None
        res = httpx.get(f"{YTMD}/state", headers={"Authorization": token}, timeout=3)
        _state["at"] = time.monotonic()
        if res.status_code == 429:
            return _state["value"]
        res.raise_for_status()
        _state["value"] = res.json()
        return _state["value"]


def _now(st: dict | None) -> dict:
    video, player = (st or {}).get("video") or {}, (st or {}).get("player") or {}
    return {"playing": player.get("trackState") == PLAYING, "title": video.get("title"), "artist": video.get("author"),
            "album": video.get("album"), "volume": player.get("volume"), "ad": player.get("adPlaying")}


def _ytmusic():
    global _yt
    if _yt is None:
        from ytmusicapi import YTMusic  # signed out: search and radio only
        _yt = YTMusic()
    return _yt


def _find(query: str, kind: str) -> tuple[dict, str] | None:
    """(changeVideo data, what it is in words) for the best match of kind."""
    yt = _ytmusic()
    if kind == "album":
        hit = next(iter(yt.search(query, filter="albums", limit=3)), None)
        if hit and (pid := yt.get_album(hit["browseId"]).get("audioPlaylistId")):
            return {"playlistId": pid}, f"album {hit['title']} by {', '.join(a['name'] for a in hit.get('artists') or [])}"
        return None
    if kind == "playlist":  # a vibe or mood ("lofi beats for focus") is a playlist too
        hit = next(iter(yt.search(query, filter="playlists", limit=3)), None)
        return ({"playlistId": hit["browseId"].removeprefix("VL")}, f"playlist {hit['title']}") if hit else None
    if kind == "artist":
        hit = next(iter(yt.search(query, filter="artists", limit=3)), None)
        if hit:
            songs = (yt.get_artist(hit["browseId"]).get("songs") or {})
            if pid := (songs.get("browseId") or "").removeprefix("VL"):
                return {"playlistId": pid}, f"songs by {hit['artist']}"
        kind = "song"  # no artist page: their top song, and its radio
    hit = next(iter(yt.search(query, filter="songs", limit=3)), None)
    if not hit:
        return None
    by = ", ".join(a["name"] for a in hit.get("artists") or [])
    return {"videoId": hit["videoId"], "playlistId": f"RDAMVM{hit['videoId']}"}, f"{hit['title']} by {by}"


# --- the voice model's tools ------------------------------------------------------------------

def music_play(args: Dict[str, Any]) -> ToolResult:
    query, kind = " ".join(str(args.get("query") or "").split()), str(args.get("kind") or "song")
    try:
        if not query or query.lower() in ("more like this", "similar", "something like this"):
            now = (state() or {}).get("video") or {}
            if not now.get("id"):
                return ToolResult(ok=False, tool_name="music_play", error="nothing_playing: ask what he wants to hear")
            target, what = {"videoId": now["id"], "playlistId": f"RDAMVM{now['id']}"}, f"more like {now.get('title')}"
        else:
            found = _find(query, kind if kind in ("song", "artist", "album", "playlist") else "song")
            if not found:
                return ToolResult(ok=False, tool_name="music_play", error="not_found: say you couldn't find it")
            target, what = found
        _cmd("changeVideo", target)
        _cmd("play")  # changeVideo keeps a paused player paused (it played ~1 s and stopped)
        _playing["on"] = True
        _duck["restore"] = None
        logger.info("music: playing %s", what)
        return ToolResult(ok=True, tool_name="music_play", data={"playing": what})
    except (httpx.HTTPError, RuntimeError) as exc:
        return ToolResult(ok=False, tool_name="music_play", error=_why(exc))


def music_control(args: Dict[str, Any]) -> ToolResult:
    action = str(args.get("action") or "")
    commands = {"pause": "pause", "resume": "play", "next": "next", "previous": "previous", "volume_up": "volumeUp",
                "volume_down": "volumeDown", "like": "toggleLike", "mute": "mute", "unmute": "unmute"}
    try:
        if action == "set_volume":
            level = max(0, min(100, int(args.get("level") or 50)))
            _cmd("setVolume", level)
        elif action in commands:
            _cmd(commands[action])
            if action in ("pause", "resume"):
                _playing["on"] = action == "resume"
        else:
            return ToolResult(ok=False, tool_name="music_control", error=f"unknown action {action}")
        if action.startswith("volume") or action in ("set_volume", "mute", "unmute", "pause"):
            _duck["restore"] = None  # he set it himself: don't put the old level back after this turn
        return ToolResult(ok=True, tool_name="music_control", data={"done": action})
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        return ToolResult(ok=False, tool_name="music_control", error=_why(exc))


def music_now(args: Dict[str, Any]) -> ToolResult:
    try:
        st = state(fresh=True)
    except httpx.HTTPError as exc:
        return ToolResult(ok=False, tool_name="music_now", error=_why(exc))
    if st is None:
        return ToolResult(ok=False, tool_name="music_now", error=_why(RuntimeError("not_paired")))
    return ToolResult(ok=True, tool_name="music_now", data=_now(st))


def music_output(args: Dict[str, Any]) -> ToolResult:
    """Where music plays: the satellite (the Mac muted) or the Mac's own output."""
    where = str(args.get("where") or "")
    if where not in ("satellite", "mac"):
        return ToolResult(ok=False, tool_name="music_output", error="where must be satellite or mac")
    OUTPUT["where"] = where
    _tell_taps()
    if where == "satellite" and not _taps:
        return ToolResult(ok=True, tool_name="music_output", data={
            "where": where, "warning": "the Mac's music tap isn't connected, so the satellite hears nothing yet"})
    return ToolResult(ok=True, tool_name="music_output", data={"where": where})


def _tell_taps() -> None:
    """Mute the app on the Mac while the satellite plays it; let the Mac play it otherwise."""
    if _loop is None:
        return
    for ws in list(_taps):
        asyncio.run_coroutine_threadsafe(ws.send_json({"mute": OUTPUT["where"] == "satellite"}), _loop)


async def serve_tap(ws) -> None:
    """/ws/music-tap: the Mac's tap streams the app's sound; satellites get it while it isn't silent."""
    global _loop
    if not TAP_TOKEN or not hmac.compare_digest(ws.headers.get("x-music-tap-token", ""), TAP_TOKEN):
        await ws.close(code=4401)
        return
    await ws.accept()
    _loop = asyncio.get_running_loop()
    _taps.add(ws)
    logger.info("music tap connected")
    await ws.send_json({"mute": OUTPUT["where"] == "satellite"})
    from voice_ws import SATELLITES  # on first use: voice_ws imports this module
    loud_at = 0.0
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            frame = msg.get("bytes")
            if not frame or OUTPUT["where"] != "satellite":
                continue
            x = np.frombuffer(frame[: len(frame) // 2 * 2], dtype="<i2").astype(np.float32)
            if len(x) and float(np.sqrt(np.mean(x * x))) > SILENT_RMS:
                loud_at = time.monotonic()
            if time.monotonic() - loud_at > TAP_SILENCE_S:
                continue  # paused or between songs: send nothing, the satellite's music fades out
            for sat in list(SATELLITES):
                sat.send_music(frame)
    finally:
        _taps.discard(ws)
        logger.info("music tap disconnected")


def music_recommend(args: Dict[str, Any]) -> ToolResult:
    """Songs like the one playing (or like a song he names), from YouTube Music's own radio."""
    try:
        seed = " ".join(str(args.get("like") or "").split())
        if seed:
            hit = next(iter(_ytmusic().search(seed, filter="songs", limit=1)), None)
            vid = hit and hit["videoId"]
        else:
            vid = ((state() or {}).get("video") or {}).get("id")
        if not vid:
            return ToolResult(ok=False, tool_name="music_recommend", error="no_seed: ask what he likes")
        tracks = _ytmusic().get_watch_playlist(videoId=vid, limit=12).get("tracks") or []
        picks = [{"title": t.get("title"), "artist": ", ".join(a["name"] for a in t.get("artists") or [])} for t in tracks[1:9]]
        return ToolResult(ok=True, tool_name="music_recommend", data={"like_it": picks})
    except (httpx.HTTPError, RuntimeError, KeyError) as exc:
        return ToolResult(ok=False, tool_name="music_recommend", error=_why(exc))


def _why(exc: Exception) -> str:
    if isinstance(exc, httpx.ConnectError):
        return "music_app_off: YouTube Music Desktop isn't running on the Mac"
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (401, 403):
        return "not_paired: the music app needs pairing again"
    if isinstance(exc, RuntimeError):  # ours, already worded for him ("not_paired: ...")
        return str(exc)[:200]
    return f"{type(exc).__name__}: {exc}"[:200]


# --- ducking (voice_ws) ---------------------------------------------------------------------

def duck() -> None:
    """Someone started talking to him: music down to DUCK_VOLUME, if it's playing louder. On the
    satellite the satellite ducks it itself, at once."""
    if OUTPUT["where"] == "satellite":
        return
    try:
        now = _now(state())
        if now["playing"] and (now["volume"] or 0) > DUCK_VOLUME and _duck["restore"] is None:
            _cmd("setVolume", DUCK_VOLUME)
            _duck["restore"] = now["volume"]
    except Exception:  # music is a nicety: never let it break a turn
        pass


def unduck() -> None:
    """The turn is over: the volume he had, unless he changed it meanwhile."""
    level, _duck["restore"] = _duck["restore"], None
    if level is not None:
        try:
            _cmd("setVolume", level)
        except Exception:
            pass


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "music_play",
        "description": "Play music on YouTube Music (on his satellite speaker, or the Mac: music_output): a song, an artist, an album, or a playlist for a "
        "mood or activity. A song keeps going with its radio. query 'more like this' plays the current song's radio.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to play, e.g. 'Bohemian Rhapsody', 'Daft Punk', 'lofi beats for focus'"},
            "kind": {"type": "string", "enum": ["song", "artist", "album", "playlist"],
                     "description": "playlist for a mood, genre or activity"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "music_control",
        "description": "Pause, resume, skip, go back, change the volume, or like the current song.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["pause", "resume", "next", "previous", "volume_up", "volume_down",
                                                  "set_volume", "mute", "unmute", "like"]},
            "level": {"type": "integer", "description": "0-100, for set_volume"}},
            "required": ["action"]}}},
    {"type": "function", "function": {
        "name": "music_output",
        "description": "Where music plays: 'satellite' (his speaker here, the default) or 'mac' (the Mac's own speakers "
        "or whatever the Mac outputs to).",
        "parameters": {"type": "object", "properties": {"where": {"type": "string", "enum": ["satellite", "mac"]}},
                       "required": ["where"]}}},
    {"type": "function", "function": {
        "name": "music_now", "description": "What's playing on YouTube Music right now, and whether it's paused.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "music_recommend",
        "description": "Songs like the one playing, or like a song he names, from YouTube Music's radio. Suggest two or "
        "three; play one with music_play if he picks it.",
        "parameters": {"type": "object", "properties": {"like": {"type": "string", "description": "A song, if not the current one"}}}}},
]


def pair() -> None:
    """One time, by hand: he turns on companion authorization in the app, runs this, and confirms the code."""
    import vault
    if not vault.shared().available:
        sys.exit("the vault is off (no master key): pairing needs it to store the token")
    code = httpx.post(f"{YTMD}/auth/requestcode", json=APP, timeout=10).json()
    if "code" not in code:
        sys.exit(f"the app refused: {code}. In YTMDesktop: Settings > Integrations > Companion server: enable it, "
                 "and enable companion authorization")
    print(f"In YouTube Music Desktop, a window asks to allow THURSDAY. Check it shows code {code['code']}, then Allow.")
    res = httpx.post(f"{YTMD}/auth/request", json={"appId": APP["appId"], "code": code["code"]}, timeout=120).json()
    if "token" not in res:
        sys.exit(f"not paired: {res}")
    vault.shared().set(OWNER, CREDENTIAL, {"token": res["token"]})
    print("Paired; the token is in the vault. Now turn companion authorization back OFF in the app.")


if __name__ == "__main__" and sys.argv[1:] == ["pair"]:
    pair()
