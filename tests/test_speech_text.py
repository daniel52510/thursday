"""How THURSDAY says symbols (speech_text.say): the misreads measured on 2026-11-10."""
import pytest

from speech_text import say


@pytest.mark.parametrize("written,spoken", [
    ("It's 72°F now.", "It's 72 degrees now."),
    ("It's 22°C.", "It's 22 degrees Celsius."),
    ("2,450 kcal and 187 g of protein", "2,450 calories and 187 grams of protein"),
    ("around $100k–$180k", "around 100 to 180 thousand dollars"),
    ("a $2.5M round", "a 2.5 million dollars round"),
    ("on 11/18 with the team", "on November 18th with the team"),
    ("due 3/1/2027", "due March 1st, 2027"),
    ("Mix 1/2 cup", "Mix 1/2 cup"),
    ("on Nov. 14th", "on November 14th"),
    ("Version 2.0.12 is out", "Version 2 point 0 point 12 is out"),
    ("e.g. no tears", "for example, no tears"),
    ("The DevSecOps role", "The Dev Sec Ops role"),
    ("HRV is 62 ms", "H R V is 62 milliseconds"),
    ("bench at 225 lb, curl 1 lb", "bench at 225 pounds, curl 1 pound"),
    ("5–10 minutes", "5 to 10 minutes"),
    ("Plain words stay plain.", "Plain words stay plain."),
])
def test_said_like_a_person(written, spoken):
    assert say(written) == spoken
