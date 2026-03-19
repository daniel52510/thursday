from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class ChatOut(BaseModel):
    text: str
    language: str = "en"
    duration: float

@app.post("/transcribe",model=ChatOut)
def transcribe():
    print("Continue ENDPOINT Work Here")