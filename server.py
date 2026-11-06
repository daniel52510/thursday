import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from starlette.requests import ClientDisconnect
from fastapi.staticfiles import StaticFiles
import requests
from pydantic import BaseModel, Field, field_validator

import accounts
import approvals
import improvements
import calendar_tool
import missions
import diagnostics  # noqa: F401  (starts remembering warnings for the health page)
import greetings
import health
import telephony
import vault
from memory import now_iso as memory_now
from accounts import FAMILY, FAMILY_ROLES, OWNER, LoginThrottle, public
from agent_loop import MODEL, URL as OLLAMA_URL, initalize_db, preload_model
from agent_stream import stream_prompt
from alarms import run_scheduler as run_alarm_scheduler
from episodic import run_memory_writer
from mind import run_reflection, voice_style
from voice_ws import SATELLITES, VOICE_TOKEN, same_origin, synthesize_ahead, router as voice_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The old shared token keeps the satellite working as a device of the admin's.
    accounts.shared().adopt_legacy_token(VOICE_TOKEN)
    # iCloud credentials from .env move into the vault once; then .env no longer needs them.
    box = vault.shared()
    if not box.available:
        logger.warning("vault key not found at %s: saving credentials is off", vault.KEY_FILE)
    elif not box.get(OWNER, calendar_tool.CREDENTIAL) and (creds := calendar_tool.env_credentials()):
        box.set(OWNER, calendar_tool.CREDENTIAL, creds)
        logger.warning("moved ICLOUD_USERNAME / ICLOUD_APP_PASSWORD into the vault; you can remove them from .env")
    # Load the LLM in the background so the first voice turn doesn't pay for it.
    threading.Thread(target=preload_model, daemon=True).start()
    # Same for iCloud calendar discovery, so the first schedule question is one round trip.
    threading.Thread(target=calendar_tool.warm_up, daemon=True).start()
    scheduler = asyncio.create_task(run_alarm_scheduler())
    remembering = asyncio.create_task(run_memory_writer())
    reflecting = asyncio.create_task(run_reflection())
    on_missions = asyncio.create_task(missions.run_worker())
    on_improvements = asyncio.create_task(improvements.run_notifier())
    on_review = asyncio.create_task(improvements.start_self_review())
    on_greetings = asyncio.create_task(greetings.run())  # fresh wake greetings each hour
    yield
    on_greetings.cancel()
    on_missions.cancel()
    on_improvements.cancel()
    on_review.cancel()
    scheduler.cancel()
    remembering.cancel()
    reflecting.cancel()


app = FastAPI(lifespan=lifespan)
app.include_router(voice_router)
app.include_router(telephony.router)  # /twilio/*: the only paths Funnel makes public
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

class StreamChatIn(BaseModel):
    text: str
    conversation_id: str | None = None  # omitted: start a new conversation
    context: dict | None = None  # {"timezone", "location"} from the page


# ---------------- who is asking ----------------
# The page logs in and gets a session cookie; satellites and scripts send a device
# token as "Authorization: Bearer ...". Either way the request acts as one user.

def current_user(request: Request) -> dict:
    session = request.cookies.get(accounts.COOKIE)
    if session:
        if request.method not in ("GET", "HEAD") and not same_origin(request.headers.get("origin"),
                                                                      request.headers.get("host")):
            raise HTTPException(status_code=403, detail="cross-site request")
        user = accounts.shared().session_user(session)
    else:
        user = accounts.shared().device_user((request.headers.get("authorization") or "").removeprefix("Bearer ").strip())
    if not user:
        raise HTTPException(status_code=401, detail="not logged in")
    return user


def require(*roles: str):
    def check(user: dict = Depends(current_user)) -> dict:
        if user["role"] not in roles:
            raise HTTPException(status_code=403, detail="not allowed for your account")
        return user
    return check


admin_only = require("admin")
family_only = require(*FAMILY_ROLES)


@app.get("/")
def chat_page():
    return FileResponse("static/index.html")


@app.get("/sw.js")
def service_worker():
    """At the root, so its scope covers the whole app. Never cached itself: an update reaches phones at once."""
    return FileResponse("static/sw.js", media_type="text/javascript", headers={"Cache-Control": "no-cache"})


# ---------------- login ----------------

class LoginIn(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=256)


_throttle = LoginThrottle()


@app.post("/api/login")
def login(body: LoginIn, request: Request, response: Response):
    if not same_origin(request.headers.get("origin"), request.headers.get("host")):
        raise HTTPException(status_code=403, detail="cross-site request")
    keys = (f"user:{body.username.strip().lower()}", f"ip:{request.client.host if request.client else '?'}")
    if _throttle.blocked(*keys):
        raise HTTPException(status_code=429, detail="too many tries; wait a minute")
    result = accounts.shared().login(body.username, body.password)
    if not result:
        _throttle.fail(*keys)
        raise HTTPException(status_code=401, detail="wrong username or password")
    _throttle.clear(*keys)
    user, token = result
    # HttpOnly: page scripts can't read it. Secure once THURSDAY is served over https.
    response.set_cookie(accounts.COOKIE, token, max_age=accounts.SESSION_DAYS * 86400, httponly=True,
                        samesite="lax", secure=request.url.scheme == "https", path="/")
    return public(user)


@app.post("/api/logout")
def logout(request: Request, response: Response):
    accounts.shared().logout(request.cookies.get(accounts.COOKIE))
    response.delete_cookie(accounts.COOKIE, path="/")
    return {"ok": True}


# ---------------- me ----------------

def _me(user: dict) -> dict:
    acc, box = accounts.shared(), vault.shared()
    persona = acc.get_persona(user.get("persona_id"))
    creds = box.names(user["id"])
    return {
        **public(user),
        "persona": persona["name"] if persona else "THURSDAY",
        "tokens_today": acc.tokens_today(user["id"]),
        "family_memory": user["role"] in FAMILY_ROLES,
        "vault": box.available,
        "credentials": {"apple_calendar": calendar_tool.CREDENTIAL in creds},
    }


@app.get("/api/me")
def me(user: dict = Depends(current_user)):
    return _me(user)


class MePatch(BaseModel):
    display_name: str | None = Field(default=None, max_length=60)
    persona_id: str | None = None  # "" = THURSDAY as she is


@app.patch("/api/me")
def update_me(body: MePatch, user: dict = Depends(current_user)):
    fields: dict[str, Any] = {}
    if body.display_name is not None and body.display_name.strip():
        fields["display_name"] = body.display_name.strip()
    if body.persona_id is not None:
        if body.persona_id and not accounts.shared().get_persona(body.persona_id):
            raise HTTPException(status_code=404, detail="persona not found")
        fields["persona_id"] = body.persona_id or None
    return _me(accounts.shared().update_user(user["id"], **fields))


class VoiceIn(BaseModel):
    voice: Literal["tara", "rocky"]


# Whose voice speaks to this user (mind.voice_style): the UI's Tara/Rocky toggle. "switch to Tara" by voice
# sets the same setting (mind.voice_setting).
@app.get("/api/greetings")
def greetings_index(user: dict = Depends(current_user)):
    """The current batch of wake greetings: the page fetches /api/greetings/<version>/<i>.wav when version changes."""
    return {"version": greetings.version, "count": len(greetings.clips)}


@app.get("/api/greetings/{version}/{i}.wav")
def greeting_clip(version: int, i: int, user: dict = Depends(current_user)):
    if version != greetings.version or not 0 <= i < len(greetings.clips):
        raise HTTPException(status_code=404)
    return Response(greetings.clips[i], media_type="audio/wav", headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/voice")
def get_voice(user: dict = Depends(current_user)):
    return {"voice": "tara" if voice_style(user["id"])["voice"] == "tara" else "rocky"}


@app.put("/api/voice")
def set_voice(body: VoiceIn, user: dict = Depends(current_user)):
    initalize_db().set_setting(user["id"], "voice_tara", int(body.voice == "tara"))
    return {"voice": body.voice}


class PasswordIn(BaseModel):
    current: str = Field(max_length=256)
    new: str = Field(max_length=256)


@app.post("/api/me/password")
def change_password(body: PasswordIn, user: dict = Depends(current_user)):
    if not accounts.check_password(body.current, user.get("password_hash")):
        raise HTTPException(status_code=403, detail="current password is wrong")
    try:
        accounts.shared().update_user(user["id"], password=body.new)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"ok": True}  # every session is logged out, this one included


@app.get("/api/me/facts")
def my_facts(user: dict = Depends(current_user)):
    return {k: v for k, v in initalize_db().list_facts(user["id"]).items() if not k.startswith("_")}


class FactIn(BaseModel):
    value: str = Field(min_length=1, max_length=500)


@app.put("/api/me/facts/{key:path}")
def set_my_fact(key: str, body: FactIn, user: dict = Depends(current_user)):
    """Add or correct something THURSDAY knows about you (the memory page)."""
    key = "_".join(key.lower().split())[:60]
    if not key or key.startswith("_"):
        raise HTTPException(status_code=422, detail="name it in a few words")
    initalize_db().upsert_fact(user["id"], key, body.value.strip(), source="settings")
    return {"key": key, "value": body.value.strip()}


class InstructionsIn(BaseModel):
    text: str = Field(max_length=1500)


@app.get("/api/me/instructions")
def get_instructions(user: dict = Depends(current_user)):
    return {"text": initalize_db().instructions(user["id"])}


@app.put("/api/me/instructions")
def put_instructions(body: InstructionsIn, user: dict = Depends(current_user)):
    initalize_db().set_instructions(user["id"], body.text.strip())
    return {"text": body.text.strip()}


@app.get("/api/me/memories")
def my_memories(user: dict = Depends(current_user)):
    """What he remembers from past conversations (one summary each), newest first."""
    return initalize_db().recent_memories(None, user_id=user["id"], limit=100)


@app.delete("/api/me/memories/{conversation_id}")
def forget_memory(conversation_id: str, user: dict = Depends(current_user)):
    if not initalize_db().delete_memory(conversation_id, user["id"]):
        raise HTTPException(status_code=404, detail="memory not found")
    return {"deleted": conversation_id}


@app.delete("/api/me/facts/{key:path}")
def forget_my_fact(key: str, user: dict = Depends(current_user)):
    if not initalize_db().delete_fact(user["id"], key):
        raise HTTPException(status_code=404, detail="fact not found")
    return {"deleted": key}


class AppleCalendarIn(BaseModel):
    username: str = Field(min_length=3, max_length=200)      # Apple ID email
    app_password: str = Field(min_length=8, max_length=64)   # appleid.apple.com > App-Specific Passwords


@app.put("/api/me/credentials/apple_calendar")
async def save_apple_calendar(body: AppleCalendarIn, user: dict = Depends(current_user)):
    box = vault.shared()
    if not box.available:
        raise HTTPException(status_code=503, detail="the vault isn't set up on the server")
    try:  # check them against iCloud before keeping them
        calendars = await asyncio.wait_for(asyncio.to_thread(
            calendar_tool.discover, body.username.strip(), body.app_password.strip()), timeout=20)
    except Exception as exc:
        logger.info("apple calendar check failed for %s: %s", user["username"], type(exc).__name__)
        raise HTTPException(status_code=400, detail="iCloud didn't accept those; check the Apple ID and app-specific password")
    box.set(user["id"], calendar_tool.CREDENTIAL, {"username": body.username.strip(), "app_password": body.app_password.strip()})
    calendar_tool.forget(user["id"])
    return {"ok": True, "calendars": len(calendars)}


@app.delete("/api/me/credentials/apple_calendar")
def remove_apple_calendar(user: dict = Depends(current_user)):
    vault.shared().delete(user["id"], calendar_tool.CREDENTIAL)
    calendar_tool.forget(user["id"])
    return {"ok": True}


# ---------------- family memory (admin + family) ----------------

class FactIn(BaseModel):
    key: str = Field(min_length=1, max_length=80)
    value: str = Field(min_length=1, max_length=500)


@app.get("/api/family/facts")
def family_facts(user: dict = Depends(family_only)):
    return initalize_db().list_facts(FAMILY)


@app.put("/api/family/facts")
def save_family_fact(body: FactIn, user: dict = Depends(family_only)):
    key = " ".join(body.key.split()).lower().replace(" ", "_")
    initalize_db().upsert_fact(FAMILY, key, body.value.strip(), source=f"user:{user['username']}")
    return {"key": key, "value": body.value.strip()}


@app.delete("/api/family/facts/{key:path}")
def forget_family_fact(key: str, user: dict = Depends(family_only)):
    if not initalize_db().delete_fact(FAMILY, key):
        raise HTTPException(status_code=404, detail="fact not found")
    return {"deleted": key}


# ---------------- personas ----------------

class PersonaIn(BaseModel):
    name: str = Field(min_length=1, max_length=40)
    prompt: str = Field(min_length=1, max_length=8000)


@app.get("/api/personas")
def list_personas(user: dict = Depends(current_user)):
    rows = accounts.shared().list_personas()
    # Everyone can pick one; only the admin sees (and edits) the prompts.
    return [r if user["role"] == "admin" else {"id": r["id"], "name": r["name"]} for r in rows]


@app.post("/api/personas", status_code=201)
def create_persona(body: PersonaIn, user: dict = Depends(admin_only)):
    try:
        return accounts.shared().save_persona(body.name, body.prompt)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.put("/api/personas/{persona_id}")
def update_persona(persona_id: str, body: PersonaIn, user: dict = Depends(admin_only)):
    if not accounts.shared().get_persona(persona_id):
        raise HTTPException(status_code=404, detail="persona not found")
    return accounts.shared().save_persona(body.name, body.prompt, persona_id)


@app.delete("/api/personas/{persona_id}")
def delete_persona(persona_id: str, user: dict = Depends(admin_only)):
    if not accounts.shared().delete_persona(persona_id):
        raise HTTPException(status_code=404, detail="persona not found")
    return {"deleted": persona_id}


# ---------------- people and devices (admin) ----------------

class UserIn(BaseModel):
    username: str = Field(max_length=32)
    display_name: str = Field(default="", max_length=60)
    role: str
    password: str = Field(max_length=256)
    daily_token_limit: int | None = Field(default=None, ge=0)


class UserPatch(BaseModel):
    display_name: str | None = Field(default=None, max_length=60)
    role: str | None = None
    password: str | None = Field(default=None, max_length=256)
    daily_token_limit: int | None = Field(default=None, ge=0)
    clear_limit: bool = False   # daily_token_limit can't say "no limit" by itself (None = unchanged)
    disabled: bool | None = None


def _person(user: dict) -> dict:
    return {**public(user), "tokens_today": accounts.shared().tokens_today(user["id"]),
            "can_log_in": bool(user.get("password_hash"))}


@app.get("/api/users")
def list_users(user: dict = Depends(admin_only)):
    return [_person(u) for u in accounts.shared().list_users()]


@app.post("/api/users", status_code=201)
def create_user(body: UserIn, user: dict = Depends(admin_only)):
    try:
        return _person(accounts.shared().create_user(body.username, body.display_name, body.role,
                                                     body.password, body.daily_token_limit))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.patch("/api/users/{user_id}")
def update_user(user_id: str, body: UserPatch, user: dict = Depends(admin_only)):
    if not accounts.shared().get_user(user_id):
        raise HTTPException(status_code=404, detail="user not found")
    fields = body.model_dump(exclude_none=True, exclude={"clear_limit"})
    if body.clear_limit:
        fields["daily_token_limit"] = None
    try:
        return _person(accounts.shared().update_user(user_id, **fields))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


class DeviceIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    user_id: str = OWNER


@app.get("/api/devices")
def list_devices(user: dict = Depends(admin_only)):
    return accounts.shared().list_devices()


@app.post("/api/devices", status_code=201)
def create_device(body: DeviceIn, user: dict = Depends(admin_only)):
    if not accounts.shared().get_user(body.user_id):
        raise HTTPException(status_code=404, detail="user not found")
    device, token = accounts.shared().create_device(body.name, body.user_id)
    return {**device, "token": token}  # shown once; only its hash is kept


@app.delete("/api/devices/{device_id}")
def revoke_device(device_id: str, user: dict = Depends(admin_only)):
    if not accounts.shared().revoke_device(device_id):
        raise HTTPException(status_code=404, detail="device not found")
    return {"revoked": device_id}


# ---------------- the rest of THURSDAY ----------------

_models_seen: dict = {"at": 0.0, "value": None}


def _model_states() -> dict:
    """Each model's state from Ollama itself: voice ready (loaded) / asleep (loads on the
    first turn); missions working / loaded / standby (pulled, unloaded between missions); either
    missing (not pulled) or offline (Ollama down). Cached 3 s: two quick local calls, at most every 3 s."""
    if time.monotonic() - _models_seen["at"] < 3 and _models_seen["value"]:
        return _models_seen["value"]
    base = OLLAMA_URL.rsplit("/api/", 1)[0]
    try:
        loaded = {m["name"] for m in requests.get(f"{base}/api/ps", timeout=1.5).json().get("models", [])}
        pulled = {m["name"] for m in requests.get(f"{base}/api/tags", timeout=1.5).json().get("models", [])}
    except (requests.RequestException, ValueError):
        value = {"voice": {"name": MODEL, "state": "offline"}, "missions": {"name": missions.MODEL, "state": "offline"}}
    else:
        def state(name: str, up: str, down: str) -> str:
            return up if name in loaded else down if name in pulled else "missing"
        working = bool(initalize_db().list_missions(status="running", limit=1))
        value = {"voice": {"name": MODEL, "state": state(MODEL, "ready", "asleep")},
                 "missions": {"name": missions.MODEL,
                              "state": "working" if working else state(missions.MODEL, "loaded", "standby")}}
    _models_seen.update(at=time.monotonic(), value=value)
    return value


@app.get("/api/info")
def api_info(user: dict = Depends(current_user)):
    # Connected satellites and their latest status report (room, mute, Wi-Fi, wake stats).
    models = dict(_model_states())  # a copy: the cache is shared
    if user["id"] != OWNER:
        models.pop("missions")  # missions are his alone
    return {"model": MODEL, "models": models, "satellites": [s.sat_status or {"room": "Satellite"} for s in SATELLITES]}


class AnnounceIn(BaseModel):
    text: str = Field(min_length=1, max_length=300)


@app.post("/api/announce")
async def api_announce(body: AnnounceIn, user: dict = Depends(family_only)):
    """Say `text` on the satellite in THURSDAY's voice ("Dinner's ready"), after a soft ping."""
    if not SATELLITES:
        raise HTTPException(503, "No satellite is connected")
    for sat in list(SATELLITES):
        audio = await synthesize_ahead(sat.http, body.text.strip())
        if not audio:
            raise HTTPException(502, "The voice server didn't return any audio")
        if await sat.announce(audio=audio, cue="announce"):
            return {"ok": True}
    raise HTTPException(409, "The satellite is busy right now")


# Automatic location: the page sends its position, rounded to ~1 km (2 decimals),
# and gets back a place name like "Austin, TX". Only those rounded coordinates leave
# the house (to OpenStreetMap's Nominatim); results are cached per rounded point.
_NOMINATIM = "https://nominatim.openstreetmap.org/reverse"
_place_cache: dict[tuple[float, float], str] = {}


@app.get("/api/whereami")
def where_am_i(lat: float, lon: float, user: dict = Depends(current_user)):
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(status_code=400, detail="bad coordinates")
    key = (round(lat, 2), round(lon, 2))
    if key not in _place_cache:
        try:
            resp = requests.get(_NOMINATIM, params={"format": "jsonv2", "lat": key[0], "lon": key[1], "zoom": 10},
                                headers={"User-Agent": "THURSDAY-self-hosted-assistant"}, timeout=8)
            resp.raise_for_status()
            addr = resp.json().get("address") or {}
        except requests.exceptions.RequestException as exc:
            raise HTTPException(status_code=502, detail=f"reverse geocoding failed: {exc}") from exc
        city = next((addr[k] for k in ("city", "town", "village", "hamlet", "suburb", "county") if addr.get(k)), "")
        region = (addr.get("ISO3166-2-lvl4") or "").split("-")[-1] if addr.get("country_code") == "us" else addr.get("country", "")
        if not city:
            raise HTTPException(status_code=404, detail="no place found")
        _place_cache[key] = f"{city}, {region}" if region else city
    return {"location": _place_cache[key]}


@app.get("/api/conversations")
def list_conversations(user: dict = Depends(current_user)):
    return initalize_db().list_conversations(user["id"])


@app.get("/api/conversations/{conversation_id}/messages")
def conversation_messages(conversation_id: str, limit: int = 500, user: dict = Depends(current_user)):
    db = initalize_db()
    if not db.get_conversation(conversation_id, user["id"]):
        raise HTTPException(status_code=404, detail="conversation not found")
    rows = db.recent_messages(min(limit, 2000), conversation_id=conversation_id)
    return [
        {"role": m["role"], "content": m["content"], "created_at": m["created_at"]}
        for m in rows
        if not m["tool_name"] and m["role"] in ("user", "assistant") and m["content"]
    ]


class RenameIn(BaseModel):
    title: str | None = None
    pinned: bool | None = None


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, body: RenameIn, user: dict = Depends(current_user)):
    """Rename and/or pin."""
    db = initalize_db()
    out: dict = {"id": conversation_id}
    if body.pinned is not None:
        if not db.pin_conversation(conversation_id, body.pinned, user["id"]):
            raise HTTPException(status_code=404, detail="conversation not found")
        out["pinned"] = body.pinned
    if body.title is not None:
        title = " ".join(body.title.split())[:100]
        if not title or not db.rename_conversation(conversation_id, title, user["id"]):
            raise HTTPException(status_code=404 if title else 422, detail="conversation not found" if title else "empty title")
        out["title"] = title
    return out


@app.get("/api/conversations/search")
def search_conversations(q: str, user: dict = Depends(current_user)):
    q = " ".join(q.split())[:100]
    return initalize_db().search_conversations(user["id"], q) if len(q) >= 2 else []


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, user: dict = Depends(current_user)):
    if not initalize_db().delete_conversation(conversation_id, user["id"]):
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"deleted": conversation_id}


class TruncateIn(BaseModel):
    from_user_turn: int


@app.post("/api/conversations/{conversation_id}/truncate")
def truncate_conversation(conversation_id: str, body: TruncateIn, user: dict = Depends(current_user)):
    """Regenerate and edit-and-resend: drop his Nth message and all after it; the page then sends again."""
    deleted = initalize_db().truncate_conversation(conversation_id, user["id"], body.from_user_turn)
    if deleted is None:
        raise HTTPException(status_code=404, detail="no such message")
    return {"deleted": deleted}


class SpeakIn(BaseModel):
    text: str


@app.post("/api/speak")
async def speak(body: SpeakIn, user: dict = Depends(current_user)):
    """Read aloud: a reply in his current voice, as one WAV (sentence-sized TTS pieces joined)."""
    import io, wave
    import mind
    text = " ".join(body.text.split())[:4000]
    if not text:
        raise HTTPException(status_code=422, detail="nothing to say")
    async with httpx.AsyncClient() as http:
        parts = await synthesize_ahead(http, text, mind.voice_style(user["id"]))
    if not parts:
        raise HTTPException(status_code=503, detail="the voice server isn't answering")
    out, rate = io.BytesIO(), None
    with wave.open(out, "wb") as w:
        for part in parts:
            with wave.open(io.BytesIO(part)) as r:
                if rate is None:
                    rate = r.getframerate()
                    w.setnchannels(r.getnchannels()); w.setsampwidth(r.getsampwidth()); w.setframerate(rate)
                w.writeframes(r.readframes(r.getnframes()))
    return Response(out.getvalue(), media_type="audio/wav")


class AlarmIn(BaseModel):
    time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")  # "HH:MM" in the alarm's timezone
    days: list[int] = Field(default_factory=list)            # Mon=0..Sun=6; empty = rings once
    timezone: str
    brief: str = Field(default="", max_length=500)           # what the morning briefing should cover
    enabled: bool = True

    @field_validator("days")
    @classmethod
    def _days(cls, v: list[int]) -> list[int]:
        if any(d not in range(7) for d in v):
            raise ValueError("days must be 0 (Mon) to 6 (Sun)")
        return sorted(set(v))

    @field_validator("timezone")
    @classmethod
    def _tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError("unknown IANA timezone")
        return v

    @field_validator("brief")
    @classmethod
    def _brief(cls, v: str) -> str:
        return " ".join(v.split())


class AlarmPatch(BaseModel):
    enabled: bool


def owner_only(user: dict = Depends(current_user)) -> dict:
    """His own things, by account id whatever the role: Apple Health, missions (his GPU, his calendar)."""
    if user["id"] != OWNER:
        raise HTTPException(status_code=403, detail="that's the owner's only")
    return user


# His Apple Health data (health.py): the Health Auto Export app on his iPhone POSTs here over the tailnet, with a
# device token from Settings → Devices.


@app.post("/api/health/ingest")
async def health_ingest(request: Request, user: dict = Depends(owner_only)):
    try:
        body = await request.body()
    except ClientDisconnect:  # the phone gave up mid-upload (a huge first sync): nothing to store, no traceback
        logging.getLogger("thursday.voice.health").warning(
            "health upload dropped by the phone mid-transfer: too big? Turn on batching / hourly grouping in the app")
        return Response(status_code=400)
    if len(body) > 50_000_000:
        raise HTTPException(status_code=413, detail="send less at once (50 MB max)")
    try:
        rows = health.parse_payload(json.loads(body))
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(status_code=400, detail=f"not a Health Auto Export / THURSDAY payload: {type(exc).__name__}")
    new = await asyncio.to_thread(health.store, rows)
    # What arrived, by metric and date range (never values), so a sync that adds nothing explains itself.
    from collections import Counter
    kinds = ", ".join(f"{m} {n}" for m, n in Counter(r[0] for r in rows).most_common(8))
    span = f"{min(r[1] for r in rows)[:16]} -> {max(r[1] for r in rows)[:16]} UTC" if rows else "no dates"
    logging.getLogger("thursday.voice.health").info("health: %d samples received, %d new (%s; %s; payload keys %s)",
                                                    len(rows), new, kinds or "none recognized", span,
                                                    sorted((json.loads(body).get("data") or {}).keys()))
    return {"received": len(rows), "new": new}


@app.get("/api/health/data")
def health_data(user: dict = Depends(owner_only)):
    return health.summary()


@app.get("/api/health/daily")
def health_daily(days: int = 30, user: dict = Depends(owner_only)):
    return health.daily(max(1, min(days, 365)))


@app.get("/api/health/readiness")
def health_readiness(user: dict = Depends(owner_only)):
    return health.readiness() or {}


@app.delete("/api/health/data")
def health_delete(user: dict = Depends(owner_only)):
    return {"deleted": health.delete_all()}


@app.get("/api/latency")
def latency(hours: float = 24, user: dict = Depends(admin_only)):
    """Each voice turn's time to first audio (and its parts), for the health page's chart."""
    import voice_ws
    keys = ("speech_end_to_playback_ms", "stt_ms", "llm_ttft_ms", "tts_first_ms")
    return [{"at": r["at"], "role": r.get("role", ""), **{k: r[k] for k in keys if k in r}}
            for r in voice_ws.turn_history(min(max(hours, 1), 24 * 14)) if "speech_end_to_playback_ms" in r]


@app.get("/api/health")
def health_page(user: dict = Depends(admin_only)):
    """The health page: each service's light and detail, current problems, the last warnings, reply speed."""
    return diagnostics.snapshot()


class ContactIn(BaseModel):
    name: str
    number: str


# The call allowlist (brain/contacts.txt): only he edits it, here or by hand. No tool can, so nothing said to
# THURSDAY (on a call, in a web page) can add a number.
@app.get("/api/contacts")
def list_contacts(user: dict = Depends(admin_only)):
    return telephony.contact_list()


@app.post("/api/contacts", status_code=201)
def save_contact(body: ContactIn, user: dict = Depends(admin_only)):
    saved = telephony.add_contact(body.name, body.number)
    if "error" in saved:
        raise HTTPException(status_code=400, detail=saved["error"])
    return saved


@app.delete("/api/contacts/{name}", status_code=204)
def delete_contact(name: str, user: dict = Depends(admin_only)):
    if not telephony.remove_contact(name):
        raise HTTPException(status_code=404)


# Alarms ring on the satellite, which is the admin's: they're the admin's too.
@app.get("/api/alarms")
def list_alarms(user: dict = Depends(admin_only)):
    return initalize_db().list_alarms(user["id"])


@app.post("/api/alarms", status_code=201)
def create_alarm(body: AlarmIn, user: dict = Depends(admin_only)):
    return initalize_db().create_alarm(body.time, body.days, body.timezone, body.brief, body.enabled, user_id=user["id"])


@app.put("/api/alarms/{alarm_id}")
def replace_alarm(alarm_id: str, body: AlarmIn, user: dict = Depends(admin_only)):
    alarm = initalize_db().update_alarm(alarm_id, user_id=user["id"], **body.model_dump())
    if not alarm:
        raise HTTPException(status_code=404, detail="alarm not found")
    return alarm


@app.patch("/api/alarms/{alarm_id}")
def toggle_alarm(alarm_id: str, body: AlarmPatch, user: dict = Depends(admin_only)):
    alarm = initalize_db().update_alarm(alarm_id, user_id=user["id"], enabled=body.enabled)
    if not alarm:
        raise HTTPException(status_code=404, detail="alarm not found")
    return alarm


@app.delete("/api/alarms/{alarm_id}")
def delete_alarm(alarm_id: str, user: dict = Depends(admin_only)):
    if not initalize_db().delete_alarm(alarm_id, user["id"]):
        raise HTTPException(status_code=404, detail="alarm not found")
    return {"deleted": alarm_id}


class MissionIn(BaseModel):
    goal: str = Field(min_length=8, max_length=1000)
    timezone: str | None = None


@app.get("/api/missions")
def list_missions(user: dict = Depends(owner_only)):
    # Watch checks that found nothing new stay off the page; their watch shows what they found.
    return [m for m in initalize_db().list_missions(user["id"], limit=60) if m["announced_at"] != "quiet"][:30]


@app.get("/api/approvals")
def list_approvals(user: dict = Depends(owner_only)):
    return approvals.pending(user["id"])


@app.post("/api/approvals/{approval_id}/{answer}")
async def decide_approval(approval_id: str, answer: str, user: dict = Depends(owner_only)):
    # His click on the Agentic page is the OK (owner session, same-origin checked in current_user).
    if answer not in ("approve", "decline"):
        raise HTTPException(status_code=404, detail="unknown answer")
    done = await asyncio.to_thread(approvals.decide, approval_id, user["id"], answer == "approve")
    if not done:
        raise HTTPException(status_code=404, detail="nothing waiting with that id")
    return done


@app.get("/api/watches")
def list_watches(user: dict = Depends(owner_only)):
    return initalize_db().list_watches(user["id"], limit=20)


@app.post("/api/watches/{watch_id}/stop")
def stop_watch(watch_id: str, user: dict = Depends(owner_only)):
    db = initalize_db()
    if not db.get_watch(watch_id, user["id"]):
        raise HTTPException(status_code=404, detail="watch not found")
    db.update_watch(watch_id, status="stopped")
    return db.get_watch(watch_id, user["id"])


@app.get("/api/improvements")
def list_improvements(user: dict = Depends(owner_only)):
    return improvements.jobs()


@app.get("/api/improvements/{job}")
def review_improvement(job: str, user: dict = Depends(owner_only)):
    try:
        return improvements.review(job)
    except improvements.GateError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/deploy")
def get_deploy(user: dict = Depends(owner_only)):
    return improvements.deploy_status()


@app.post("/api/improvements/{job}/deploy")
def deploy_improvement(job: str, user: dict = Depends(owner_only)):
    try:
        return improvements.deploy_request(job)
    except improvements.GateError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/api/improvements/{job}/{answer}")
async def decide_improvement(job: str, answer: str, user: dict = Depends(owner_only)):
    """His Approve merges the job's branch into his checkout (improvements.py says when it won't)."""
    if answer not in ("approve", "decline"):
        raise HTTPException(status_code=404, detail="approve or decline")
    try:
        return await asyncio.to_thread(improvements.approve if answer == "approve" else improvements.decline, job)
    except improvements.GateError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/routines")
def list_routines(user: dict = Depends(owner_only)):
    return [{**r, "when": missions._when(r["time"], r["days"])} for r in initalize_db().list_routines(user["id"], limit=20)]


@app.post("/api/routines/{routine_id}/stop")
def stop_routine(routine_id: str, user: dict = Depends(owner_only)):
    db = initalize_db()
    if not db.get_routine(routine_id, user["id"]):
        raise HTTPException(status_code=404, detail="routine not found")
    db.update_routine(routine_id, status="stopped")
    return db.get_routine(routine_id, user["id"])


@app.post("/api/missions", status_code=201)
def start_mission(body: MissionIn, user: dict = Depends(owner_only)):
    db = initalize_db()
    if len(db.list_missions(user["id"], "queued")) >= missions.MAX_QUEUED:
        raise HTTPException(status_code=429, detail="too many missions queued")
    return db.create_mission(user["id"], body.goal.strip(), body.timezone)


@app.post("/api/missions/{mission_id}/abort")
def abort_mission(mission_id: str, user: dict = Depends(owner_only)):
    db = initalize_db()
    mission = db.get_mission(mission_id, user["id"])
    if not mission:
        raise HTTPException(status_code=404, detail="mission not found")
    if mission["status"] in ("queued", "running"):
        db.update_mission(mission_id, status="aborted", finished_at=memory_now())
    return db.get_mission(mission_id, user["id"])


@app.post("/api/chat")
async def api_chat(body: StreamChatIn, user: dict = Depends(current_user)):
    # Server-sent events: one `data: {...}` line per stream_prompt event.
    # If the client disconnects (stop button), the generator is closed and Ollama stops.
    async def events():
        started = time.monotonic()
        try:
            async for ev in stream_prompt(body.text, mode="chat", conversation_id=body.conversation_id,
                                          context=body.context, user=user):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("api_chat failed: stage=LLM chars=%d elapsed_ms=%d",
                             len(body.text), round((time.monotonic() - started) * 1000))
            yield f"data: {json.dumps({'type': 'error', 'message': f'{type(exc).__name__}: {exc}'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
