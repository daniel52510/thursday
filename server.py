import json
import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_loop import MODEL, initalize_db, preload_model
from agent_stream import stream_prompt
from voice_ws import _token_ok, router as voice_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the LLM in the background so the first voice turn doesn't pay for it.
    threading.Thread(target=preload_model, daemon=True).start()
    yield


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
