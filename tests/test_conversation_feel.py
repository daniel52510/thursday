"""Human-feeling voice turns (2026-11-10): a spoken reply's length budget and the reasoning-leak scrub."""
import pytest

from agent_stream import _ThinkScrub, strip_think
from voice_ws import REPLY_BUDGET, REPLY_BUDGET_DETAIL, REPLY_BUDGET_LONG, budget_end, reply_budget


@pytest.mark.parametrize("asked,want", [
    ("I'm taking a lunch after work.", REPLY_BUDGET),
    ("What do you think about drones?", REPLY_BUDGET),
    ("Explain how a jet engine works", REPLY_BUDGET_DETAIL),
    ("why does my squat feel off", REPLY_BUDGET_DETAIL),
    ("tell me the whole story", REPLY_BUDGET_LONG),
    ("keep going", REPLY_BUDGET_LONG),
])
def test_budget_by_what_he_asked(asked, want):
    assert reply_budget(asked, "voice") == want


def test_no_budget_for_alarms_and_briefings():
    assert reply_budget("wake him up", "wake") is None and reply_budget("x", "briefing") is None


def test_stops_at_a_sentence_end_past_the_budget():
    a = "Oh, nice. " + "Drones are hard because the control loop has to run fast and never miss a beat. " * 4
    end = budget_end(a, 300)
    assert end is not None and a[:end].endswith(".") and 180 <= end <= 340


def test_short_reply_runs_to_its_end():
    assert budget_end("Yeah. Keep going, what else?", 300) is None


def test_runaway_sentence_stops_at_a_comma():
    a = ("word " * 30 + ", ") * 4
    end = budget_end(a, 300)
    assert end is not None and end <= len(a)


def test_think_leak_dropped_from_stream():
    s = _ThinkScrub()
    out = "".join(s.feed(t) for t in ("own thought ", "</thi", "nk>That's a big ", "one. Here's my take on it.")) + s.flush()
    assert out == "That's a big one. Here's my take on it."


def test_clean_reply_passes_through():
    s = _ThinkScrub()
    out = "".join(s.feed(t) for t in ("Yeah.", " Go on.")) + s.flush()
    assert out == "Yeah. Go on."


def test_strip_think_on_saved_reply():
    assert strip_think("own thought </think>That's heavy.") == "That's heavy."
    assert strip_think("Fine. </think>") == "Fine. "  # nothing after the tag: keep the reply
    s = _ThinkScrub()
    assert (s.feed("Fine. </think>") + s.flush()) == "Fine."


# ---------- how he said it (voice_ws.delivery) ----------
import numpy as np

from voice_ws import _LISTENER_NOISE, delivery


def _speech(seconds: float, db: float) -> bytes:
    t = np.arange(int(16000 * seconds)) / 16000
    amp = 10 ** (db / 20) * np.sqrt(2)
    return (np.sin(2 * np.pi * 180 * t) * amp * 32767).astype(np.int16).tobytes()


def test_thinking_out_loud():
    text = ("so um I was thinking like you know maybe I could build something uh that flies on its own "
            "and like I don't know you know it just seems um kind of cool")
    note, level = delivery(_speech(12, -25), text, None)
    assert note and "thinking out loud" in note and level is not None


def test_quiet_and_slow_against_his_own_level():
    note, _ = delivery(_speech(4, -38), "I'm just tired today", baseline_db=-25)
    assert note and "quietly and slowly" in note


def test_fired_up():
    note, _ = delivery(_speech(3, -18), "Dude we actually did it the whole thing works first try", baseline_db=-26)
    assert note and "fired up" in note


def test_ordinary_turn_has_no_note():
    note, _ = delivery(_speech(3, -25), "What's the weather like tomorrow", baseline_db=-25)
    assert note is None


@pytest.mark.parametrize("said,noise", [("Yeah.", True), ("Mm-hmm.", True), ("Right, right.", True), ("Haha", True),
                                        ("Oh, wow.", True), ("Wait, no.", False), ("Stop.", False),
                                        ("Yeah but what about the cost?", False)])
def test_listener_noise(said, noise):
    assert bool(_LISTENER_NOISE.fullmatch(said)) is noise


from voice_ws import _unmenu


@pytest.mark.parametrize("said,want", [
    ('That sounds incredible. You thinking about the hardware side first, or more the software for the "brain" part?',
     "That sounds incredible. You thinking about the hardware side first?"),
    ("Are you thinking more about the hardware build, or the autonomy side of things?",
     "Are you thinking more about the hardware build?"),
    ("You want hardware first, or software for the brain, question?", "You want hardware first, question?"),
    ("Want me to play it now or later?", "Want me to play it now or later?"),
    ("Tea or coffee?", "Tea or coffee?"),
    ("Would you rather build the frame or write the flight code first?", "Would you rather build the frame or write the flight code first?"),
    ("That works. Nice.", "That works. Nice."),
])
def test_one_question_not_a_menu(said, want):
    assert _unmenu(said) == want


from voice_ws import HOLD_END_FRAMES, QUICK_END_FRAMES, _end_frames_for


@pytest.mark.parametrize("said,want", [
    ("What do you think about that?", QUICK_END_FRAMES),
    ("Honestly I'm kind of wiped out today.", None),  # a second sentence may follow: default wait
    ("So I was thinking about building a drone and um", HOLD_END_FRAMES),
    ("Okay.", None),
])
def test_end_of_turn_waits(said, want):
    assert _end_frames_for(said) == want


def test_voiced_ms():
    from voice_ws import _voiced_ms
    assert _voiced_ms(_speech(1.0, -25)) >= 900
    assert _voiced_ms(b"\x00\x00" * 16000) == 0


def test_question_gate_sees_whole_sentences():
    from voice_ws import _QuestionGate
    g = _QuestionGate()
    out = "".join(g.feed(t) for t in ("It was a wild time. Does that make sense, ", "or did you want to dive into ",
                                       "the actual mechanics of how they worked?")) + g.flush()
    assert out == "It was a wild time. Does that make sense?" and g.changed


@pytest.mark.parametrize("said,want", [
    ("It was a wild time. Does that make sense?", "It was a wild time."),
    ("Whittle patented it in 1930. Hope that helps!", "Whittle patented it in 1930."),
    ("Done. Let me know if you need anything else.", "Done."),
    ("Anything else?", "Anything else?"),
    ("That's a great question. Short answer: 1939.", "Short answer: 1939."),
    ("You want jacket, question? Does that make sense, question?", "You want jacket, question?"),
])
def test_no_help_desk_sign_offs(said, want):
    from voice_ws import humanize
    assert humanize(said) == want


def test_opener_dropped_while_streaming():
    from voice_ws import _QuestionGate
    g = _QuestionGate()
    out = "".join(g.feed(t) for t in ("That's a great ", "question. Short ", "answer: 1939. ", "Whittle first.")) + g.flush()
    assert out == "Short answer: 1939. Whittle first."
    g = _QuestionGate()
    assert g.feed("That's a great question.") + g.flush() == "That's a great question." and not g.changed


@pytest.mark.parametrize("said,want", [
    ("That sounds ambitious. Are you thinking of a hobbyist build?", "Are you thinking of a hobbyist build?"),
    ("That sounds like a massive engineering challenge. Where would you start?", "Where would you start?"),
    ("That sounds rough. Want to talk about it?", "That sounds rough. Want to talk about it?"),
    ("That sounds amazing!", "That sounds amazing!"),
])
def test_stock_praise_opener(said, want):
    from voice_ws import humanize
    assert humanize(said) == want
