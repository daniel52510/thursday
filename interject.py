"""Interjection evaluator (full duplex, slice 4): while the user is still talking, should
THURSDAY cut in? A small second model on the Mac (gemma3:4b, ~0.4 s) judges what they've
said so far, never the main model: calling gemma4:26b every pause would evict its prompt
cache and slow every real reply.

It only decides. What THURSDAY would say is the main model's job (slice 5), in his voice.
Backchannels stay a rule (voice_ws.BACKCHANNEL): the model over-fires on "mid-story".

Measured 2026-10-31 on 15 hand-made partials: 14/15 right (every interjection case caught;
one false "wrong fact" on a true claim), median 393 ms. A wrong fact only counts when the
model quotes a false claim that really is in the transcript: that removed 2 of 3 false alarms.
"""
import json
import logging
import os

import httpx

from agent_stream import CHAT_URL

MODEL = os.getenv("VOICE_EVALUATOR_MODEL", "gemma3:4b")
TIMEOUT = httpx.Timeout(5, connect=2)
MIN_WORDS = 5  # fewer and there's nothing to judge yet

SYSTEM = """Someone is speaking, mid-sentence. Read what they've said so far and answer each question with true or false.
- wrong_fact: they stated something factually false (science, geography, history, numbers).
- risky_plan: they announced a plan that is clearly reckless (money, health, safety).
- big_news: they shared major personal news (a new job, engagement, a birth, a big win, a loss).
Most speech is none of these: questions, requests, plans for dinner and everyday stories are all false.
If wrong_fact is true, "claim" quotes the false statement in their exact words, and "truth" says what is actually true; otherwise both are "".
JSON only: {"wrong_fact": false, "claim": "", "truth": "", "risky_plan": false, "big_news": false}"""

logger = logging.getLogger("thursday.voice")


async def evaluate(http: httpx.AsyncClient, text: str) -> dict | None:
    """{"action": "LISTEN" | "INTERJECT", "why": [...], "truth": str}, or None (too short, or it failed)."""
    if len(text.split()) < MIN_WORDS:
        return None
    body = {"model": MODEL, "stream": False, "format": "json", "keep_alive": "30m",
            "options": {"temperature": 0, "num_predict": 60},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f'Said so far: "{text}"'}]}
    try:
        res = await http.post(CHAT_URL, json=body, timeout=TIMEOUT)
        res.raise_for_status()
        flags = json.loads(res.json()["message"]["content"])
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        logger.info("evaluator failed: %s", type(exc).__name__)
        return None
    claim = str(flags.get("claim") or "").lower().strip(' .,"')
    if flags.get("wrong_fact") and (len(claim) < 8 or claim not in text.lower()):
        flags["wrong_fact"] = False  # no quotable false claim in what they said: not an error
    why = [k for k in ("wrong_fact", "risky_plan", "big_news") if flags.get(k) is True]
    return {"action": "INTERJECT" if why else "LISTEN", "why": why, "truth": str(flags.get("truth") or "")}
