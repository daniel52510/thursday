"""THURSDAY on his iPhone (phone.py, /api/phone): the phone_ tools only record what the phone should do, only on a
phone turn; the Shortcut gets flat keys; and the phone tools are offered only to phone turns."""
import pytest

import agent_stream
import phone


@pytest.fixture(autouse=True)
def prefs(monkeypatch, tmp_path):
    monkeypatch.setattr(phone, "PREFS_FILE", str(tmp_path / "phone.json"))


def _turn():
    box = {}
    return box, phone.ACTION.set(box)


def test_call_and_message_go_to_his_phone():
    box, t = _turn()
    try:
        assert phone.phone_call({"contact": "my Mom"}).ok and box == {"do": "call", "contact": "Mom"}
        r = phone.phone_message({"contact": "Sam", "text": "Running   10 late"})
        assert r.ok and box == {"do": "message", "contact": "Sam", "body": "Running 10 late"}   # the last one wins
        out = phone.respond("Texting Sam.")
        assert (out["do"], out["contact"], out["body"], out["listen"]) == ("message", "Sam", "Running 10 late", False)
    finally:
        phone.ACTION.reset(t)


def test_music_opens_in_his_app_and_directions_and_timer():
    box, t = _turn()
    try:
        phone.phone_music({"query": "Tame Impala"})
        assert box["url"] == "https://music.youtube.com/search?q=Tame%20Impala"
        phone.set_prefs("apple")
        phone.phone_music({"query": "workout playlist"})
        assert box["url"].startswith("https://music.apple.com/us/search?term=workout")
        phone.phone_directions({"destination": "Whole Foods"})
        assert box["url"] == "https://maps.apple.com/?daddr=Whole%20Foods&dirflg=d"
        phone.phone_timer({"minutes": 10})
        assert box == {"do": "timer", "minutes": 10}
        assert not phone.phone_timer({"minutes": 0}).ok
    finally:
        phone.ACTION.reset(t)


def test_not_a_phone_turn_does_nothing():
    r = phone.phone_call({"contact": "Mom"})
    assert not r.ok and r.error.startswith("only_from_his_iphone")


def test_listen_again_when_he_was_asked_something():
    box, t = _turn()
    try:
        assert phone.respond("Which Sam, question?")["listen"] is True
        assert phone.respond("Done.")["listen"] is False
    finally:
        phone.ACTION.reset(t)


@pytest.mark.parametrize("said,hit", [("call Mom", True), ("text Sam I'm late", True), ("play some Tame Impala", True),
                                      ("tell Jordana I'm on my way", True), ("tell me a joke", False),
                                      ("what's the weather", False), ("set a timer for 5 minutes", True)])
def test_phone_requests_are_recognised(said, hit):
    assert bool(agent_stream._PHONE_ASK.search(said)) is hit


def test_phone_tools_are_only_for_phone_turns():
    names = {t["function"]["name"] for t in agent_stream.TOOLS}
    assert not names & {"phone_call", "phone_message", "phone_music"}


@pytest.mark.parametrize("kwargs", [
    {"json": {"text": "what time is it"}}, {"json": {"Text": "what time is it"}}, {"json": {" TEXT ": "what time is it"}}, {"json": {"text =": "what time is it"}},
    {"data": {"text": "what time is it"}}, {"data": {"Text": "what time is it"}}])
def test_shortcut_body_any_shape(kwargs, monkeypatch):
    """JSON or Form, "text" in any case: a Shortcut's field boxes make all of these."""
    from fastapi.testclient import TestClient
    import server
    monkeypatch.setattr(server, "stream_prompt", _fake_stream)
    server.app.dependency_overrides[server.current_user] = lambda: {"id": "owner"}
    try:
        r = TestClient(server.app).post("/api/phone", **kwargs)
        assert r.status_code == 200 and r.json()["speak"] == "It's noon."
        assert TestClient(server.app).post("/api/phone", json={"words": "hi"}).status_code == 422
    finally:
        server.app.dependency_overrides.clear()


async def _fake_stream(text, mode, conversation_id=None, context=None, user=None):
    assert text == "what time is it" and context["channel"] == "phone"
    yield {"type": "conversation", "id": "c1"}
    yield {"type": "done", "reply": "It's noon."}
