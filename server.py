import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import requests
from pydantic import BaseModel, Field, field_validator

from agent_loop import MODEL, initalize_db, preload_model
from agent_stream import stream_prompt
from alarms import run_scheduler as run_alarm_scheduler
from episodic import run_memory_writer
from calendar_tool import warm_up as calendar_warm_up
from voice_ws import _token_ok, router as voice_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the LLM in the background so the first voice turn doesn't pay for it.
    threading.Thread(target=preload_model, daemon=True).start()
    # Same for iCloud calendar discovery, so the first schedule question is one round trip.
    threading.Thread(target=calendar_warm_up, daemon=True).start()
    scheduler = asyncio.create_task(run_alarm_scheduler())
    remembering = asyncio.create_task(run_memory_writer())
    yield
    scheduler.cancel()
    remembering.cancel()


app = FastAPI(lifespan=lifespan)
app.include_router(voice_router)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

class StreamChatIn(BaseModel):
    text: str
    conversation_id: str | None = None  # omitted: start a new conversation
    context: dict | None = None  # {"timezone", "location"} from the page


def require_token(authorization: str | None = Header(default=None)) -> None:
    # Same VOICE_TOKEN as /ws/voice: the pages are public, the API is not.
    token = (authorization or "").removeprefix("Bearer ").strip()
    if not _token_ok(token):
        raise HTTPException(status_code=401, detail="invalid token")


@app.get("/")
def chat_page():
    return FileResponse("static/index.html")


@app.get("/api/info", dependencies=[Depends(require_token)])
def api_info():
    return {"model": MODEL}


# Automatic location: the page sends its position, rounded to ~1 km (2 decimals),
# and gets back a place name like "Austin, TX". Only those rounded coordinates leave
# the house (to OpenStreetMap's Nominatim); results are cached per rounded point.
_NOMINATIM = "https://nominatim.openstreetmap.org/reverse"
_place_cache: dict[tuple[float, float], str] = {}


@app.get("/api/whereami", dependencies=[Depends(require_token)])
def where_am_i(lat: float, lon: float):
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


@app.get("/api/conversations", dependencies=[Depends(require_token)])
def list_conversations():
    return initalize_db().list_conversations()


@app.get("/api/conversations/{conversation_id}/messages", dependencies=[Depends(require_token)])
def conversation_messages(conversation_id: str, limit: int = 500):
    db = initalize_db()
    if not db.get_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    rows = db.recent_messages(min(limit, 2000), conversation_id=conversation_id)
    return [
        {"role": m["role"], "content": m["content"], "created_at": m["created_at"]}
        for m in rows
        if not m["tool_name"] and m["role"] in ("user", "assistant") and m["content"]
    ]


class RenameIn(BaseModel):
    title: str


@app.patch("/api/conversations/{conversation_id}", dependencies=[Depends(require_token)])
def rename_conversation(conversation_id: str, body: RenameIn):
    title = " ".join(body.title.split())[:100]
    if not title or not initalize_db().rename_conversation(conversation_id, title):
        raise HTTPException(status_code=404 if title else 422, detail="conversation not found" if title else "empty title")
    return {"id": conversation_id, "title": title}


@app.delete("/api/conversations/{conversation_id}", dependencies=[Depends(require_token)])
def delete_conversation(conversation_id: str):
    if not initalize_db().delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"deleted": conversation_id}


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


@app.get("/api/alarms", dependencies=[Depends(require_token)])
def list_alarms():
    return initalize_db().list_alarms()


@app.post("/api/alarms", dependencies=[Depends(require_token)], status_code=201)
def create_alarm(body: AlarmIn):
    return initalize_db().create_alarm(body.time, body.days, body.timezone, body.brief, body.enabled)


@app.put("/api/alarms/{alarm_id}", dependencies=[Depends(require_token)])
def replace_alarm(alarm_id: str, body: AlarmIn):
    alarm = initalize_db().update_alarm(alarm_id, **body.model_dump())
    if not alarm:
        raise HTTPException(status_code=404, detail="alarm not found")
    return alarm


@app.patch("/api/alarms/{alarm_id}", dependencies=[Depends(require_token)])
def toggle_alarm(alarm_id: str, body: AlarmPatch):
    alarm = initalize_db().update_alarm(alarm_id, enabled=body.enabled)
    if not alarm:
        raise HTTPException(status_code=404, detail="alarm not found")
    return alarm


@app.delete("/api/alarms/{alarm_id}", dependencies=[Depends(require_token)])
def delete_alarm(alarm_id: str):
    if not initalize_db().delete_alarm(alarm_id):
        raise HTTPException(status_code=404, detail="alarm not found")
    return {"deleted": alarm_id}


@app.post("/api/chat", dependencies=[Depends(require_token)])
async def api_chat(body: StreamChatIn):
    # Server-sent events: one `data: {...}` line per stream_prompt event.
    # If the client disconnects (stop button), the generator is closed and Ollama stops.
    async def events():
        started = time.monotonic()
        try:
            async for ev in stream_prompt(body.text, mode="chat", conversation_id=body.conversation_id, context=body.context):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("api_chat failed: stage=LLM chars=%d elapsed_ms=%d",
                             len(body.text), round((time.monotonic() - started) * 1000))
            yield f"data: {json.dumps({'type': 'error', 'message': f'{type(exc).__name__}: {exc}'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
