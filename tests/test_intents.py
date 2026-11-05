"""Requests acted on in code (agent_stream): music and work on THURSDAY's own code.

The phrasings are Daniel's own, from turns where the model claimed an action no tool did
("Music, off.") or said it couldn't reach its own code (2026-11-01).
"""
from pathlib import Path

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
    ("Initialize code_review_start on your satellite implementation", "code_review_start"),
    ("fix yourself so the music actually stops", "improve_start"),
    ("Can you improve how you answer weather questions?", "improve_start"),
    ("debug yourself", "improve_start"),
    ("Run improve_start tab", "improve_start"),
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


MISSIONS = [
    ("Can you do research on uh on exactly what it takes to get me where I am currently to uh SpaceX, let's say, uh, in terms of career growth?",
     "exactly what it takes to get me where I am currently to SpaceX, let's say, in terms of career growth"),
    ("Can you perform a task to analyze your satellite code to see whether switching to C++ Instead of Python will be faster for you and more efficient?",
     "analyze your satellite code to see whether switching to C++ Instead of Python will be faster for you and more efficient"),
    ("Start a mission to find the best robot vacuum under $400 for a dog",
     "find the best robot vacuum under $400 for a dog"),
    ("Run an agentic task to compare home espresso machines under $600",
     "compare home espresso machines under $600"),
]
NOT_MISSIONS = [
    "Hey Thursday, can I task you to do a mission?",
    "Can you start a mission? Question?",
    "What's the weather tomorrow?",
    "Can you give me the latest in the US-Iran conflict?",
]


@pytest.mark.parametrize("said,goal", MISSIONS)
def test_mission_requests_run_in_code(said, goal):
    assert a._mission_intent(said) == ("mission_start", {"goal": goal})


@pytest.mark.parametrize("said", NOT_MISSIONS)
def test_non_mission_or_goal_less_requests_left_alone(said):
    assert a._mission_intent(said) is None


def test_two_turn_mission_goal_after_thursday_asks():
    history = [{"role": "user", "content": "Hey Thursday, can I task you to do a mission?"},
               {"role": "assistant", "content": "Mission, question? I like missions, exclamation! Tell me goal and any constraints. I am ready, question?"}]
    got = a._mission_intent("Find the best robot vacuum under 400 dollars for a dog", history)
    assert got == ("mission_start", {"goal": "Find the best robot vacuum under 400 dollars for a dog"})
    assert a._mission_intent("Never mind", history) is None


def test_an_offer_is_recognized_and_a_claim_is_caught():
    from agent_loop import initalize_db
    db = initalize_db()
    conv = db.create_conversation("x", "owner")
    db.log_message(role="assistant", content="Classical music is good logic. I keep it ready, question?", conversation_id=conv["id"])
    assert a._last_offer(db, conv["id"]) and a.plain_yes("Yeah, sure.")
    assert a._CLAIMED.search("Music, playing.") and a._CLAIMED.search("Okay, now playing jazz")
    assert a._CLAIMED.search("I start mission now.") and a._CLAIMED.search("Improvement task is running in sandbox")
    assert not a._CLAIMED.search("Okay. Later then.") and not a._CLAIMED.search("I like playing golf.")


def test_a_statement_is_not_an_offer():
    from agent_loop import initalize_db
    db = initalize_db()
    conv = db.create_conversation("x", "owner")
    db.log_message(role="assistant", content="Jupiter is best planet.", conversation_id=conv["id"])
    assert a._last_offer(db, conv["id"]) is None


def test_code_ask_finds_the_code_a_plain_question_means(monkeypatch):
    import improvements
    monkeypatch.setattr(improvements, "TURN", type("T", (), {"get": staticmethod(lambda: {"user": improvements.OWNER})}))
    monkeypatch.setattr(improvements, "REPO", Path(__file__).resolve().parent.parent)
    names = lambda q: [c["name"] for c in improvements.code_ask({"question": q}).data["code"]]  # noqa: E731
    assert any("Endpointer" in n or "end_silence" in n for n in names("how do you decide when I stopped talking?"))
    assert any("barge" in n.lower() for n in names("how does barge-in work?"))
