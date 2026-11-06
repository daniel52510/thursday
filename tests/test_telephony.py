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
