"""Rocky's grammar (eridian.py): the tag rules Daniel approved on 2026-11-01."""
import pytest

import eridian

CASES = [  # (reply, statements, expected)
    ("Do you want a jacket?", False, "You want jacket, question?"),
    ("It is sunny, question. You want jacket?", False, "It is sunny. You want jacket, question?"),
    ("Launch is tonight! Do you want to watch?", False, "Launch is tonight, exclamation! You want to watch, question?"),
    ("We plan, statement.", False, "We plan."),
    ("Good, good, good! Engine works!", False, "Good, good, good! Engine works, exclamation!"),
    ("Morning! Amaze! Amaze! Amaze!", False, "Morning! Amaze! Amaze! Amaze!"),
    ("See https://x.com/a!b now.", False, "See https://x.com/a!b now."),
    ("I find best collar, question?", True, "I find best collar."),
    ("Thanks.", False, "Gracious."),
    ("Awesome!", False, "Amaze!"),
]


@pytest.mark.parametrize("text,statements,want", CASES)
def test_whole_reply(text, statements, want):
    assert eridian.translate(text, statements) == want


@pytest.mark.parametrize("text,statements,want", CASES)
def test_streamed_token_by_token(text, statements, want):
    """Ollama sends a few characters at a time: the stream must give the same result."""
    stream = eridian.Stream()
    stream.statements = statements
    assert "".join(stream.feed(ch) for ch in text) + stream.flush() == want


def test_code_blocks_untouched():
    text = "Here.\n```\nx = 1  # is it?\n```\nDone!"
    assert "x = 1  # is it?" in eridian.translate(text)
