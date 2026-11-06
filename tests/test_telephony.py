"""Phone calls (telephony.py): only signed webhooks get in, and nothing dials without his yes."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import telephony
from calendar_tool import TURN

PUBLIC = "https://mac.example.ts.net:8443"


@pytest.fixture
def phone(monkeypatch, tmp_path):
    for k, v in {"SID": "AC1", "TOKEN": "secret", "FROM": "+15550000000", "PUBLIC": PUBLIC,
                 "OWNER_NUMBER": "+15551112222"}.items():
        monkeypatch.setattr(telephony, k, v)
    monkeypatch.setattr(telephony, "_render", lambda text: b"RIFFwav")
    monkeypatch.setattr(telephony, "_opening", lambda message, goal, name: message)  # the LLM rewrite
    monkeypatch.setattr("approvals.push", lambda *a, **k: True)
    telephony._pending.clear()
    monkeypatch.setattr(telephony, "PLACED_FILE", str(tmp_path / "calls_placed.txt"))
    app = FastAPI()
    app.include_router(telephony.router)
    return TestClient(app)


def test_webhook_needs_twilios_signature(phone):
    form = {"From": "+15553334444", "CallSid": "CA9"}
    assert phone.post("/twilio/voice", data=form).status_code == 403
    assert phone.post("/twilio/voice", data=form, headers={"X-Twilio-Signature": "forged"}).status_code == 403
    good = telephony.signature(PUBLIC + "/twilio/voice", form)
    r = phone.post("/twilio/voice", data=form, headers={"X-Twilio-Signature": good})
    assert r.status_code == 200 and "<Hangup/>" in r.text


def test_numbers_only_from_contacts(phone, tmp_path, monkeypatch):
    book = tmp_path / "contacts.txt"
    book.write_text("# family\nMom: (512) 555-0123\nUK friend: +44 20 7946 0958\nHotline: +1 900 555 0123\n")
    monkeypatch.setattr(telephony, "CONTACTS_FILE", str(book))
    assert telephony._number("Mom") == "+15125550123"
    assert telephony._number("my mom") == "+15125550123"
    assert telephony._number("512 555 0123") == "+15125550123"  # a number that's on the list
    assert telephony._number("me") == "+15551112222"
    assert telephony._number("+1 (305) 555-0123") is None       # a fine number, but not a contact
    assert telephony._number("UK friend") is None               # toll fraud: +1 only, even when saved
    assert telephony._number("Hotline") is None                 # premium rate, even when saved
    assert telephony._number("call the plumber") is None
    _turn(1, "call the plumber")
    err = telephony.call_place({"to": "the plumber", "message": "hi"}).error
    assert err.startswith("not_in_contacts") and "Mom" in err


def _turn(n, said, user="daniel"):
    from accounts import OWNER
    TURN.set({"user": OWNER if user == "daniel" else user, "conversation": "c1", "turn": n, "user_said": said})


def test_dials_only_after_a_plain_yes_next_turn(phone, monkeypatch):
    sent = []

    class R:
        status_code, headers = 201, {"content-type": "application/json"}

    monkeypatch.setattr(telephony.httpx, "post", lambda url, **kw: sent.append(kw["data"]) or R())
    _turn(1, "call me and say the oven is on")
    assert telephony.call_place({"to": "me", "message": "The oven is on."}).ok
    assert not telephony.call_confirm({}).ok                       # same turn: he hasn't answered
    _turn(2, "yes but later")
    assert not telephony.call_confirm({}).ok and not sent           # hedged
    _turn(3, "yes")
    assert telephony.confirms_pending("c1", "yes")
    assert telephony.call_confirm({}).ok
    assert sent[0]["To"] == "+15551112222" and f"{PUBLIC}/twilio/audio/" in sent[0]["Twiml"]
    token = sent[0]["Twiml"].split("/twilio/audio/")[1].split(".wav")[0]
    assert phone.get(f"/twilio/audio/{token}.wav").content == b"RIFFwav"
    assert phone.get("/twilio/audio/guess.wav").status_code == 404


def test_not_his_voice_account_no_call(phone):
    _turn(1, "call 5125550123", user="guest")
    assert telephony.call_place({"to": "5125550123", "message": "hi"}).error == "calls_are_daniels_only"


def test_stream_needs_signature_and_a_live_token(phone, monkeypatch):
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):  # forged handshake: closed before anything is accepted
        with phone.websocket_connect("/twilio/stream", headers={"X-Twilio-Signature": "forged"}) as ws:
            ws.receive_text()
    sig = {"X-Twilio-Signature": telephony.signature(telephony._stream_url(), {})}
    with phone.websocket_connect("/twilio/stream", headers=sig) as ws:  # signed, but no call token: nothing played
        ws.send_json({"event": "start", "start": {"streamSid": "MZ1", "customParameters": {"token": "guess"}}})
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()
    monkeypatch.setattr(telephony, "_mulaw", lambda wav: b"\xff" * 12000)
    token = telephony._keep_audio(b"RIFFwav")
    with phone.websocket_connect("/twilio/stream", headers=sig) as ws:
        ws.send_json({"event": "start", "start": {"streamSid": "MZ1", "customParameters": {"token": token}}})
        frames = [ws.receive_json() for _ in range(3)]
        assert [f["event"] for f in frames] == ["media", "media", "mark"]
        ws.send_json({"event": "stop"})
    with phone.websocket_connect("/twilio/stream", headers=sig) as ws:  # one stream per call
        ws.send_json({"event": "start", "start": {"streamSid": "MZ2", "customParameters": {"token": token}}})
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_phone_encoder_has_no_seams_between_pieces():
    import numpy as np
    import rocky_voice
    x = (0.3 * np.sin(np.arange(24000) * 2 * np.pi * 440 / 24000)).astype(np.float32)
    whole = telephony._PhoneEncoder()
    one = whole.feed(rocky_voice._write(x, 24000)) + whole.flush()
    pieced = telephony._PhoneEncoder()
    parts = b"".join(pieced.feed(rocky_voice._write(x[i:i + 1901], 24000)) for i in range(0, len(x), 1901))
    assert parts + pieced.flush() == one


def test_fillers_wait_for_real_words():
    for filler in ("But uh", "uh", "Um...", "so", "and, uh,", "Hmm."):
        assert telephony._FILLER.match(filler), filler
    for words in ("Dinner is at 8.", "Okay", "No", "So dinner is at 8", "Who is this?"):
        assert not telephony._FILLER.match(words), words


def _signed_post(phone, path, form):
    sig = telephony.signature(PUBLIC + path, form)
    return phone.post(path, data=form, headers={"X-Twilio-Signature": sig})


def test_answer_picks_stream_or_voicemail_and_needs_signature(phone):
    token = telephony._keep_audio(b"RIFFwav")
    telephony._briefs[token] = {"who": "Mom", "message": "hi", "transcript": []}
    path = f"/twilio/answer?token={token}"
    assert phone.post(path, data={"AnsweredBy": "human"}).status_code == 403
    assert "<Stream" in _signed_post(phone, path, {"AnsweredBy": "human"}).text
    vm = _signed_post(phone, path, {"AnsweredBy": "machine_end_beep"}).text
    assert f"/twilio/audio/{token}.wav" in vm and "<Stream" not in vm
    assert "<Hangup/>" in _signed_post(phone, "/twilio/answer?token=guess", {"AnsweredBy": "human"}).text


def test_status_starts_one_report(phone, monkeypatch):
    seen = []

    async def fake_report(brief):
        seen.append(brief)
    monkeypatch.setattr(telephony, "_report", fake_report)
    telephony._briefs["T1"] = {"who": "Mom", "message": "hi", "transcript": [], "answered_by": ""}
    assert phone.post("/twilio/status?token=T1", data={"CallStatus": "no-answer"}).status_code == 403
    assert _signed_post(phone, "/twilio/status?token=T1", {"CallStatus": "no-answer", "CallDuration": "0"}).status_code == 204
    assert seen and seen[0]["status"] == "no-answer" and "T1" not in telephony._briefs


def test_outcomes_without_a_conversation():
    base = {"who": "Mom", "transcript": [], "answered_by": ""}
    assert "nobody picked up" in telephony._outcome({**base, "status": "no-answer"})
    assert "voicemail" in telephony._outcome({**base, "status": "completed", "answered_by": "machine_end_beep"})
    talked = [("assistant", "hi"), ("user", "(the other person said) dinner is at 8")]
    assert telephony._outcome({**base, "status": "completed", "answered_by": "human", "transcript": talked}) is None


def test_share_only_his_approved_details_from_his_account(phone):
    assert telephony._shareable(["name", "phone"]) == {
        "Name for the order or message": telephony.OWNER_NAME, "His callback number": "555-111-2222"}
    assert telephony._shareable(["address", "card", "+1 305 555 0000"]) == {}  # nothing else, no values from the model
    prompt = telephony.call_prompt({"message": "hi", "goal": "order", "share": telephony._shareable(["phone"])})
    assert "555-111-2222" in prompt and "Never invent" in prompt
    assert "555-111-2222" not in telephony.call_prompt({"message": "hi", "goal": "order"})


def test_share_comes_from_his_words(phone):
    _turn(1, "call me and order a pizza, you can give them my name and number")
    r = telephony.call_place({"to": "me", "message": "Hi, an order for pickup please.", "goal": "order a pizza"})
    assert set(r.data["will_share_if_asked"]) == {"Name for the order or message", "His callback number"}
    _turn(2, "call me and order a pizza")
    assert "will_share_if_asked" not in telephony.call_place({"to": "me", "message": "Hi.", "goal": "order"}).data


def test_signature_with_or_without_the_port(phone):
    token = telephony._keep_audio(b"RIFFwav")
    telephony._briefs[token] = {"who": "Mom", "message": "hi", "transcript": []}
    path, form = f"/twilio/answer?token={token}", {"AnsweredBy": "human"}
    no_port = PUBLIC.replace(":8443", "")
    r = phone.post(path, data=form, headers={"X-Twilio-Signature": telephony.signature(no_port + path, form)})
    assert r.status_code == 200 and "<Stream" in r.text
    other = "https://evil.example.com" + path
    assert phone.post(path, data=form, headers={"X-Twilio-Signature": telephony.signature(other, form)}).status_code == 403


def test_contact_book_add_update_remove(phone, tmp_path, monkeypatch):
    monkeypatch.setattr(telephony, "CONTACTS_FILE", str(tmp_path / "contacts.txt"))
    for raw in ("(305) 555-0123", "305.555.0123", "1 305 555 0123", "+1 305 555 0123"):
        assert telephony.add_contact("Mom", raw)["number"] == "+13055550123", raw
    assert [c["name"] for c in telephony.contact_list()] == ["Mom"]          # updated, not duplicated
    assert telephony._number("mom") == "+13055550123"                        # callable right away
    assert "+1" in telephony.add_contact("UK", "+44 20 7946 0958")["error"]
    assert "premium" in telephony.add_contact("Psychic", "900 555 0123")["error"]
    assert "10-digit" in telephony.add_contact("Short", "555 0123")["error"]
    assert "you" in telephony.add_contact("me", "305 555 0123")["error"]
    assert telephony.remove_contact("MOM") and telephony._number("Mom") is None


def test_same_opener_twice_is_dropped():
    m = telephony._OPENER.match("Oh, perfect, a large it is.")
    assert m and m.group(1) == "Oh"
    assert not telephony._OPENER.match("A large, please.")
    assert not telephony._OPENER.match("Ohio is far.")


def test_amd_only_a_sure_voicemail_switches_a_live_call(phone, monkeypatch):
    posted = []

    class FakeClient:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, **k):
            posted.append((url, k["data"]["Twiml"]))
            return type("R", (), {"status_code": 200})()
    monkeypatch.setattr(telephony.httpx, "AsyncClient", FakeClient)
    token = telephony._keep_audio(b"RIFFwav")
    telephony._briefs[token] = {"who": "Ivanna", "message": "hi", "transcript": []}
    sid = "CA" + "0" * 32
    path = f"/twilio/amd?token={token}"
    assert phone.post(path, data={"AnsweredBy": "machine_end_beep", "CallSid": sid}).status_code == 403
    _signed_post(phone, path, {"AnsweredBy": "machine_end_other", "CallSid": sid})  # "Hello? Hello?": leave it
    _signed_post(phone, path, {"AnsweredBy": "human", "CallSid": sid})
    assert posted == []
    _signed_post(phone, path, {"AnsweredBy": "machine_end_beep", "CallSid": sid})
    assert len(posted) == 1 and sid in posted[0][0] and f"/twilio/audio/{token}.wav" in posted[0][1]


def test_report_summarizes_a_real_conversation_whatever_the_machine_verdict():
    talked = [("assistant", "Hi Ivanna!"), ("user", "(the other person said) So do the joke.")]
    assert telephony._outcome({"who": "Ivanna", "status": "completed", "answered_by": "machine_end_other",
                               "transcript": talked}) is None  # summarize it
    assert "no beep" in telephony._outcome({"who": "Ivanna", "status": "completed", "answered_by": "machine_start",
                                            "transcript": [("assistant", "Hi")]})
