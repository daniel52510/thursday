"""Email conversations (email_threads.py): only their replies to our emails count, replies stay in the thread, and
the guardrails (no new links or numbers, a reply cap, stop and steer) hold whatever the model says."""
import asyncio
from email.message import EmailMessage

import pytest

import email_threads
import gmail_tool
from calendar_tool import TURN


class Inbox:
    """A fake Gmail INBOX: SEARCH returns every uid; FETCH the raw email."""
    def __init__(self): self.msgs = {}
    def add(self, uid, frm, body, in_reply_to="", mid=None):
        m = EmailMessage()
        m["From"], m["Subject"], m["Message-ID"] = frm, "Re: Sink", mid or f"<them{uid}@x>"
        if in_reply_to:
            m["In-Reply-To"] = in_reply_to
        m.set_content(body)
        self.msgs[str(uid)] = m.as_bytes()
    def select(self, box, readonly=False): return "OK", [b"1"]
    def uid(self, cmd, *args):
        if cmd == "SEARCH":
            return "OK", [" ".join(self.msgs).encode()]
        return "OK", [(b"1 (UID 1)", self.msgs[args[0].decode() if isinstance(args[0], bytes) else args[0]])]
    def logout(self): pass


@pytest.fixture
def world(monkeypatch, tmp_path):
    TURN.set({"user": "owner", "conversation": "c", "turn": 1, "user_said": "x"})
    monkeypatch.setattr(email_threads, "FILE", str(tmp_path / "threads.json"))
    monkeypatch.setattr(gmail_tool, "EMAIL_CONTACTS_FILE", str(tmp_path / "contacts.txt"))
    monkeypatch.setattr(gmail_tool, "SENT_LOG", str(tmp_path / "sent.jsonl"))
    monkeypatch.setattr(gmail_tool.vault, "shared", lambda: type("V", (), {
        "get": lambda s, u, n: {"username": "me@gmail.com", "app_password": "x"}})())
    gmail_tool.add_email_contact("Landlord", "lord@example.com")
    inbox, sent, told, decisions = Inbox(), [], [], []
    monkeypatch.setattr(gmail_tool, "_connect", lambda user: inbox)
    monkeypatch.setattr(gmail_tool, "smtp_send", lambda msg: sent.append(msg))
    monkeypatch.setattr(email_threads, "_decide", lambda t: decisions.pop(0))

    async def tell(title, text, say=None): told.append(title)
    monkeypatch.setattr(email_threads, "_tell", tell)
    first = EmailMessage()
    first["From"], first["To"], first["Subject"], first["Message-ID"] = "me@gmail.com", "lord@example.com", "Sink", "<us1@gmail.com>"
    first.set_content("Hi, the sink leaks. When could you fix it? I'm free Tue or Wed after 5.")
    t = email_threads.start(first, "Find a time to fix the sink; I'm free Tue or Wed after 5")
    return {"inbox": inbox, "sent": sent, "told": told, "decisions": decisions, "id": t["id"]}


def _thread(tid):
    return next(t for t in email_threads._load() if t["id"] == tid)


def test_only_their_replies_to_us_count_and_the_reply_stays_in_thread(world):
    w = world
    w["inbox"].add(1, "Scammer <bad@evil.com>", "Re: Sink. Pay here", in_reply_to="<us1@gmail.com>")
    w["inbox"].add(2, "lord@example.com", "Unrelated newsletter")  # them, but not answering us
    w["inbox"].add(3, "lord@example.com", "Wednesday at 6 works?\n\nOn Mon, me wrote: Hi, the sink leaks",
                   in_reply_to="<us1@gmail.com>")
    w["decisions"].append({"action": "reply", "reply": "Wednesday at 6 is perfect. Thanks! Daniel", "summary": "He offered Wed 6."})
    asyncio.run(email_threads.tick())
    assert len(w["sent"]) == 1
    msg = w["sent"][0]
    assert msg["To"] == "lord@example.com" and msg["In-Reply-To"] == "<them3@x>" and msg["Subject"] == "Re: Sink"
    t = _thread(w["id"])
    assert t["replies"] == 1 and t["transcript"][1]["text"] == "Wednesday at 6 works?"  # quoted history cut
    assert sorted(t["seen"]) == ["1", "2", "3"] and w["told"] == ["Replied to Landlord"]
    asyncio.run(email_threads.tick())  # nothing new: no second reply
    assert len(w["sent"]) == 1


def test_a_reply_with_a_new_link_or_number_goes_to_him(world):
    w = world
    w["inbox"].add(4, "lord@example.com", "Can you confirm?", in_reply_to="<us1@gmail.com>")
    w["decisions"].append({"action": "reply", "reply": "Sure, call me at 305-555-0199 or see http://x.co", "summary": "s"})
    asyncio.run(email_threads.tick())
    assert not w["sent"] and _thread(w["id"])["status"] == "needs_you" and w["told"] == ["Landlord needs you"]


def test_done_needs_you_and_the_reply_cap(world, monkeypatch):
    w = world
    monkeypatch.setattr(email_threads, "MAX_REPLIES", 0)
    w["inbox"].add(5, "lord@example.com", "Thursday?", in_reply_to="<us1@gmail.com>")
    asyncio.run(email_threads.tick())  # cap reached before any model call
    assert not w["sent"] and _thread(w["id"])["status"] == "needs_you" and not w["decisions"]


def test_stop_and_steer(world):
    w = world
    w["inbox"].add(6, "lord@example.com", "What's your budget?", in_reply_to="<us1@gmail.com>")
    w["decisions"].append({"action": "needs_you", "reply": "", "summary": "He asks your budget."})
    asyncio.run(email_threads.tick())
    assert _thread(w["id"])["status"] == "needs_you" and not w["sent"]
    r = email_threads.email_thread_steer({"id": "landlord", "guidance": "Up to $150"})
    assert r.ok and _thread(w["id"])["pending"]
    w["decisions"].append({"action": "reply", "reply": "Up to $150 works. Daniel", "summary": "Told him $150."})
    asyncio.run(email_threads.tick())  # answers their last email with his note, no new email needed
    assert len(w["sent"]) == 1 and _thread(w["id"])["notes"] == ["Up to $150"]
    assert email_threads.email_thread_stop({}).ok and _thread(w["id"])["status"] == "stopped"
    w["inbox"].add(7, "lord@example.com", "Great, see you", in_reply_to=w["sent"][0]["Message-ID"])
    asyncio.run(email_threads.tick())  # stopped: not read, nothing sent
    assert len(w["sent"]) == 1


def test_thread_tools_are_owner_only(world):
    TURN.set({"user": "guest", "conversation": "c", "turn": 1, "user_said": "x"})
    for fn in (email_threads.email_threads, email_threads.email_thread_stop):
        assert fn({}).error == "email_is_daniels_only"
    assert email_threads.email_thread_steer({"guidance": "x"}).error == "email_is_daniels_only"
