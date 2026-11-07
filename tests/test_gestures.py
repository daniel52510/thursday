"""Body language: <nod> etc. become gesture events and never reach speech, screen or memory (agent_stream)."""
import asyncio

import agent_stream


def _run(deltas, monkeypatch):
    async def fake(*a, **k):
        for d in deltas:
            yield {"type": "delta", "text": d}
        yield {"type": "done", "reply": "".join(deltas)}
    monkeypatch.setattr(agent_stream, "_stream_with_grammar", fake)

    async def go():
        return [ev async for ev in agent_stream.stream_prompt("x", user={"id": "owner"})]
    return asyncio.run(go())


def test_gestures_are_lifted_out_even_when_split(monkeypatch):
    evs = _run(["Good news! <cele", "brate> The run was fast.", " <nod>Keep going."], monkeypatch)
    text = "".join(e["text"] for e in evs if e["type"] == "delta")
    assert [e["name"] for e in evs if e["type"] == "gesture"] == ["celebrate", "nod"]
    assert "<" not in text and "The run was fast." in text
    done = next(e for e in evs if e["type"] == "done")
    assert "<" not in done["reply"]


def test_other_angle_text_stays(monkeypatch):
    evs = _run(["2 < 3 and <chuckle> fine", " <b>bold</b>"], monkeypatch)
    text = "".join(e["text"] for e in evs if e["type"] == "delta")
    assert "2 < 3" in text and "<chuckle>" in text and not [e for e in evs if e["type"] == "gesture"]
