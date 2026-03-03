from fastapi import FastAPI
from pydantic import BaseModel

from agent_loop import run_prompt

app = FastAPI()

class ChatIn(BaseModel):
    text: str

@app.post("/chat")
def chat(body: ChatIn):
    resp = run_prompt(body.text)
    return resp.model_dump()