"""Self-review evidence (improvements._evidence): his corrections and claims no tool backed, newest first."""
from datetime import datetime, timedelta, timezone

import improvements as g
from agent_loop import initalize_db


def _conversation(turns):
    db = initalize_db()
    conv = db.create_conversation("Music", "owner")
    for role, text in turns:
        db.log_message(role=role, content=text, conversation_id=conv["id"])


def _since():
    return (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()


def test_claim_without_a_tool_is_evidence():
    _conversation([("user", "Turn off the music."), ("assistant", "Music, off.")])
    ev = g._evidence(_since())
    assert len(ev) == 1 and ev[0].startswith("CLAIM WITHOUT A TOOL") and "Music, off." in ev[0]


def test_claim_after_a_tool_is_not():
    _conversation([("user", "Turn off the music."), ("assistant", "TOOL_RESULT:music_control"), ("assistant", "Music, off.")])
    assert g._evidence(_since()) == []


def test_his_correction_is_evidence_with_context():
    _conversation([("user", "Play Impact by S. G. Lewis."), ("assistant", "TOOL_RESULT:music_play"),
                   ("assistant", "Impact, by S. G. Lewis. Good beat!"), ("user", "Nope, still not changing it."),
                   ("assistant", "Bad, bad, bad.")])
    ev = g._evidence(_since())
    assert any(e.startswith("CORRECTION") and "still not changing" in e and "Impact" in e for e in ev)


def test_ordinary_no_is_not_a_correction():
    _conversation([("user", "Do you like jazz?"), ("assistant", "Yes. Chords!"), ("user", "No, no, I'm kidding."),
                   ("assistant", "Okay.")])
    assert g._evidence(_since()) == []


def test_reviews_and_missions_are_not_read():
    db = initalize_db()
    conv = db.create_conversation("Self-review · Nov 1", "owner")
    db.log_message(role="assistant", content="Music, off.", conversation_id=conv["id"])
    assert g._evidence(_since()) == []
