"""Text the way a person would say it, just before it's spoken (voice_ws._speakable).

Orpheus reads plain words beautifully and symbols badly. Measured on 2026-11-10 (live voice, transcribed back):
"72°F" came out "72 for self ice", "2,450 kcal and 187 g" as "2,450 callus and 187 grimasdreine", "DevSecOps" as
"Death Stack Ops", "on 11/18" as "on 1118", "2.0.12" as "2.012". So units, money, dates, versions, abbreviations
and a little jargon are written out first. Only what's spoken changes: the chat and the saved reply keep the
original text.
"""
from __future__ import annotations

import re

_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]
_ABBR_MONTH = {m[:3].lower(): m for m in _MONTHS} | {"sept": "September"}


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


_SCALE = {"k": "thousand", "m": "million", "b": "billion"}
_MONEY_RANGE = re.compile(r"\$(\d[\d,]*(?:\.\d+)?)\s*([kKmMbB])?\s*(?:[–—-]|to)\s*\$?(\d[\d,]*(?:\.\d+)?)\s*([kKmMbB])\b")
_MONEY_SCALED = re.compile(r"\$(\d[\d,]*(?:\.\d+)?)\s*([kKmMbB])\b")
_NUM_RANGE = re.compile(r"(?<=\d)\s*[–—]\s*(?=\d)")
_TEMP = re.compile(r"(-?\d+(?:\.\d+)?)\s*°\s*([FC])?\b")
# Units after a number. Longest first, so "kcal" is not read as "k" + "cal".
_UNITS = [("kcal", "calories"), ("cal", "calories"), ("kg", "kilograms"), ("lbs", "pounds"), ("lb", "pounds"),
          ("oz", "ounces"), ("ml", "milliliters"), ("km", "kilometers"), ("mi", "miles"), ("mph", "miles per hour"),
          ("bpm", "beats per minute"), ("ms", "milliseconds"), ("hrs", "hours"), ("hr", "hours"), ("mins", "minutes"),
          ("min", "minutes"), ("secs", "seconds"), ("sec", "seconds"), ("g", "grams"), ("GB", "gigabytes"),
          ("MB", "megabytes"), ("TB", "terabytes")]
_UNIT = re.compile(r"(\d)\s?(" + "|".join(sorted((re.escape(u) for u, _ in _UNITS), key=len, reverse=True)) + r")\b")
_UNIT_WORD = {u: w for u, w in _UNITS}
_DATE_FULL = re.compile(r"\b(1[0-2]|0?[1-9])/(3[01]|[12]\d|0?[1-9])/(\d{4})\b")
# A bare "3/4" may be a fraction: only after a word that introduces a date.
_DATE = re.compile(r"\b(on|by|until|till|before|after|from|since|due|through|thru)\s+(1[0-2]|0?[1-9])/(3[01]|[12]\d|0?[1-9])\b", re.I)
_MONTH_ABBR = re.compile(r"\b(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.(?=\s*\d)", re.I)
_VERSION = re.compile(r"\b\d+(?:\.\d+){2,}\b")
_PHRASES = [
    (re.compile(r"\be\.g\.,?\s*", re.I), "for example, "),
    (re.compile(r"\bi\.e\.,?\s*", re.I), "that is, "),
    (re.compile(r"\betc\.", re.I), "et cetera"),
    (re.compile(r"\bvs\.?\s", re.I), "versus "),
    (re.compile(r"\bDr\.\s+(?=[A-Z])"), "Doctor "),
    (re.compile(r"\bw/\s*"), "with "),
    (re.compile(r"\s&\s"), " and "),
    (re.compile(r"~\s*(?=\d)"), "about "),
    (re.compile(r"\bDevSecOps\b"), "Dev Sec Ops"),
    (re.compile(r"\bDevOps\b"), "Dev Ops"),
    (re.compile(r"\bHRV\b"), "H R V"),
]


def _money_range(m: re.Match) -> str:
    a, sa, b, sb = m.group(1), (m.group(2) or m.group(4)).lower(), m.group(3), m.group(4).lower()
    if sa == sb:
        return f"{a} to {b} {_SCALE[sb]} dollars"
    return f"{a} {_SCALE[sa]} to {b} {_SCALE[sb]} dollars"


def _unit(m: re.Match) -> str:
    word = _UNIT_WORD[m.group(2)]
    whole = re.search(r"(?:^|[^\d.,])(\d+)$", m.string[:m.end(1)])
    if whole and whole.group(1) == "1" and word.endswith("s") and word not in ("calories",):
        word = word[:-1]  # "1 lb": "1 pound"
    return f"{m.group(1)} {word}"


def say(text: str) -> str:
    text = _MONEY_RANGE.sub(_money_range, text)
    text = _MONEY_SCALED.sub(lambda m: f"{m.group(1)} {_SCALE[m.group(2).lower()]} dollars", text)
    text = _TEMP.sub(lambda m: f"{m.group(1)} degrees" + (" Celsius" if m.group(2) == "C" else ""), text)
    text = _DATE_FULL.sub(lambda m: f"{_MONTHS[int(m.group(1)) - 1]} {_ordinal(int(m.group(2)))}, {m.group(3)}", text)
    text = _DATE.sub(lambda m: f"{m.group(1)} {_MONTHS[int(m.group(2)) - 1]} {_ordinal(int(m.group(3)))}", text)
    text = _MONTH_ABBR.sub(lambda m: _ABBR_MONTH[m.group(1).lower()], text)
    text = _VERSION.sub(lambda m: " point ".join(m.group(0).split(".")), text)
    text = _NUM_RANGE.sub(" to ", text)
    text = _UNIT.sub(_unit, text)
    for pat, rep in _PHRASES:
        text = pat.sub(rep, text)
    return text
