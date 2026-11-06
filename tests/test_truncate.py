"""Edit-and-resend / regenerate: the conversation rewinds to his Nth message on the server."""
import memory


def test_truncate_from_his_nth_message(tmp_path):
    db = memory.MemoryDB(str(tmp_path / "m.db"))
    cid = db.create_conversation("t", "owner")["id"]
    for role, text in [("user", "one"), ("assistant", "a1"), ("user", "two"), ("assistant", "TOOL_RESULT:x"),
                       ("assistant", "a2"), ("user", "three"), ("assistant", "a3")]:
        db.log_message(role, text, conversation_id=cid)
    assert db.truncate_conversation(cid, "owner", 1) == 5          # "two" and everything after, tool rows too
    assert [m["content"] for m in db.recent_messages(50, conversation_id=cid)] == ["one", "a1"]
    assert db.truncate_conversation(cid, "owner", 9) is None        # no such turn
    assert db.truncate_conversation(cid, "someone-else", 0) is None  # not his conversation


def test_search_and_pin(tmp_path):
    db = memory.MemoryDB(str(tmp_path / "m.db"))
    a = db.create_conversation("Pizza order", "owner")["id"]
    b = db.create_conversation("Weather", "owner")["id"]
    db.log_message("user", "what's 100% of my sleep_goal?", conversation_id=b)
    assert [c["id"] for c in db.search_conversations("owner", "pizza")] == [a]          # title
    assert db.search_conversations("owner", "100%")[0]["snippet"].startswith("what's")   # content, % is literal
    assert db.search_conversations("owner", "sleep_goal")[0]["id"] == b                   # _ is literal
    assert db.search_conversations("owner", "nothing here") == []
    assert db.search_conversations("someone-else", "pizza") == []
    assert db.pin_conversation(a, True, "owner")
    assert db.list_conversations("owner")[0]["id"] == a and db.list_conversations("owner")[0]["pinned"] == 1


def test_instructions_are_not_facts(tmp_path):
    db = memory.MemoryDB(str(tmp_path / "m.db"))
    db.upsert_fact("owner", "favorite_food", "sushi")
    db.set_instructions("owner", "Keep it short. Call me boss.")
    assert db.instructions("owner") == "Keep it short. Call me boss."
    assert db.prompt_facts("owner") == {"favorite_food": "sushi"}  # never listed as a fact
    db.set_instructions("owner", "")
    assert db.instructions("owner") == "" and "_instructions" not in db.list_facts("owner")
