"""Requests acted on in code (agent_stream): music and work on THURSDAY's own code.

The phrasings are Daniel's own, from turns where the model claimed an action no tool did
("Music, off.") or said it couldn't reach its own code (2026-11-01).
"""
import pytest

import agent_stream as a

MUSIC = [
    ("Turn off the music.", ("music_control", {"action": "pause"})),
    ("Stop the music.", ("music_control", {"action": "pause"})),
    ("Hey Thursday, pause the music", ("music_control", {"action": "pause"})),
    ("pause", ("music_control", {"action": "pause"})),
    ("Who sings this song?", ("music_now", {})),
    ("And the song name?", ("music_now", {})),
    ("skip", ("music_control", {"action": "next"})),
    ("next song please", ("music_control", {"action": "next"})),
    ("play the previous song", ("music_control", {"action": "previous"})),
    ("turn it up", ("music_control", {"action": "volume_up"})),
    ("quieter", ("music_control", {"action": "volume_down"})),
    ("set the volume to 30", ("music_control", {"action": "set_volume", "level": 30})),
    ("resume the music", ("music_control", {"action": "resume"})),
    ("Play John Wick music. This isn't John Wick music.", ("music_play", {"query": "John Wick music", "kind": "playlist"})),
    ("Play Impact by S. G. Lewis.", ("music_play", {"query": "Impact by S. G. Lewis", "kind": "song"})),
    ("play some Daft Punk", ("music_play", {"query": "Daft Punk", "kind": "artist"})),
    ("play some jazz", ("music_play", {"query": "jazz", "kind": "playlist"})),
    ("play the album Random Access Memories", ("music_play", {"query": "Random Access Memories", "kind": "album"})),
]
NOT_MUSIC = ["play a game with me", "stop", "what's the next game?", "turn off the lights", "I'm going to skip lunch",
             "who is the president", "are you playing with me", "play it"]


@pytest.mark.parametrize("said,want", MUSIC)
def test_music_requests_run_in_code(said, want):
    assert a._music_intent(said) == want


@pytest.mark.parametrize("said", NOT_MUSIC)
def test_other_requests_left_to_the_model(said):
    assert a._music_intent(said) is None


SELF = [
    ("Can you review your internal code and tell me if there are any fixes you propose?", "code_review_start"),
    ("Can you review code and find potential bugs and propose fixes? Sorry I mean your internal code", "code_review_start"),
    ("improvements to your code.", "code_review_start"),
    ("look for bugs in your music code", "code_review_start"),
    ("fix yourself so the music actually stops", "improve_start"),
    ("Can you improve how you answer weather questions?", "improve_start"),
    ("debug yourself", "improve_start"),
]
NOT_SELF = ["fix my code", "can you review my essay", "improve my resume", "How are you?", "we will fix it",
            "change the lights to red"]


@pytest.mark.parametrize("said,tool", SELF)
def test_own_code_requests_reach_the_tools(said, tool):
    got = a._self_code_intent(said)
    assert got and got[0] == tool


@pytest.mark.parametrize("said", NOT_SELF)
def test_other_fix_requests_left_alone(said):
    assert a._self_code_intent(said) is None


def test_an_offer_is_recognized_and_a_claim_is_caught():
    from agent_loop import initalize_db
    db = initalize_db()
    conv = db.create_conversation("x", "owner")
    db.log_message(role="assistant", content="Classical music is good logic. I keep it ready, question?", conversation_id=conv["id"])
    assert a._last_offer(db, conv["id"]) and a.plain_yes("Yeah, sure.")
    assert a._CLAIMED.search("Music, playing.") and a._CLAIMED.search("Okay, now playing jazz")
    assert not a._CLAIMED.search("Okay. Later then.") and not a._CLAIMED.search("I like playing golf.")


def test_a_statement_is_not_an_offer():
    from agent_loop import initalize_db
    db = initalize_db()
    conv = db.create_conversation("x", "owner")
    db.log_message(role="assistant", content="Jupiter is best planet.", conversation_id=conv["id"])
    assert a._last_offer(db, conv["id"]) is None
