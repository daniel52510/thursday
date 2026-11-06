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
