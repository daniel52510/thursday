import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent_loop import run_prompt

logger = logging.getLogger(__name__)

app = FastAPI()

class ChatIn(BaseModel):
    text: str

@app.post("/chat")
def chat(body: ChatIn):
    try:
        resp = run_prompt(body.text)
    except RuntimeError as exc:
        logger.exception("run_prompt failed")
        raise HTTPException(status_code=503, detail=str(exc))
    return resp.model_dump()