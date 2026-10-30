"""THURSDAY's episodic memory: what she remembers of past conversations.

Writer: once every conversation has been quiet for IDLE_MIN, a background pass
turns each unsummarised one into a short first-person memory (what mattered,
how they seemed, how it felt to her, what's still open). It only runs while
nobody is talking, so its Ollama call never evicts a live conversation's
prompt cache or delays a turn.

Recall: on a conversation's first turn, the most relevant memories are
phrased into a short note for her system prompt and frozen on the
conversation (see MemoryDB.set_memory_note), so later turns keep a stable prompt.
"""
import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone

from agent_loop import MODEL, OLLAMA_KEEP_ALIVE, OLLAMA_OPTIONS, _post_ollama, initalize_db
from agent_schemas import MemoryNote
from memory import MemoryDB

logger = logging.getLogger("thursday.memory")

IDLE_MIN = float(os.getenv("MEMORY_IDLE_MIN", "10"))   # quiet this long before she writes memories
POLL_S = 120
PER_TICK = 3             # conversations summarised per pass (the backfill trickles in)
TRANSCRIPT_MESSAGES = 40  # the end of a long conversation is what she'd remember
MESSAGE_CHARS = 600
RECALL_COUNT = 4

MEMORY_SYSTEM = """
You are THURSDAY, writing a private memory of a conversation you just had with the person you live with.
Write it the way you'd remember it yourself: what mattered, not a transcript.

Return ONLY JSON:
{"summary": string, "their_mood": string, "her_feeling": string, "open_threads": [string], "importance": integer}

- summary: 1-3 sentences, first person, past tense ("We talked about...", "They told me..."). Keep the specific
  details they'd expect you to remember (names, plans, dates, what they decided) over generalities.
- their_mood: a few words on how they seemed.
- her_feeling: a few words on how the conversation felt to you (amused, close, worried for them, delighted...).
- open_threads: only things that will have news later and that they'd be glad you asked about: a decision
  they were weighing, something coming up for them, a problem or worry they were dealing with. Not curiosities
  about what they said, not things you could just look up. At most 2, each under 15 words. Usually empty.
- importance 1-5: 1 = small talk or a quick lookup (weather, a fact), 3 = something personal or a real plan,
  5 = something big in their life.
- A lookup (weather, time, a search) is not a memory by itself; mention it only if it mattered to them.
- Never record passwords, tokens, keys or financial account details.
""".strip()

RECALL_GUIDE = (
    "What you remember from earlier conversations (your own memories, newest first; not facts to recite). "
    "Let them shape how you talk, like someone who knows them. When there's a natural opening, especially early on, "
    "you can bring up one open thread in passing (\"Wait, did you ever decide...?\"). At most one per conversation, "
    "and let it go if they're in the middle of something. Never list memories or mention having a record."
)


def _transcript(db: MemoryDB, conversation_id: str) -> str:
    rows = [m for m in db.recent_messages(TRANSCRIPT_MESSAGES * 3, conversation_id=conversation_id)
            if not m["tool_name"] and m["role"] in ("user", "assistant") and m["content"]][-TRANSCRIPT_MESSAGES:]
    lines = []
    for m in rows:
        text = " ".join(m["content"].split())
        if len(text) > MESSAGE_CHARS:
            text = text[:MESSAGE_CHARS] + "..."
        lines.append(f"{'Them' if m['role'] == 'user' else 'You'}: {text}")
    return "\n".join(lines)


def write_memory(db: MemoryDB, conv: dict) -> dict:
    """One Ollama pass: the conversation's transcript -> her memory of it."""
    raw = _post_ollama({
        "model": MODEL,
        "system": MEMORY_SYSTEM,
        "prompt": f"The conversation:\n{_transcript(db, conv['id'])}",
        "format": "json",
        "stream": False,
        "think": False,
        "options": OLLAMA_OPTIONS,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    })
    note = MemoryNote.model_validate(json.loads(raw)).model_dump()
    db.upsert_memory(conv["id"], note, covered_until=conv["last_id"], happened_at=conv["last_at"])
    return note


async def run_memory_writer() -> None:
    db = initalize_db()
    while True:
        try:
            for _ in range(PER_TICK):
                idle_before = (datetime.now(timezone.utc) - timedelta(minutes=IDLE_MIN)).isoformat()
                last = db.last_message_at()
                if last and last > idle_before:
                    break  # someone is talking: never compete with a live turn
                todo = db.conversations_to_remember(idle_before, limit=1)
                if not todo:
                    break
                note = await asyncio.to_thread(write_memory, db, todo[0])
                logger.info("remembered conversation %s (importance %s, %d open threads)",
                            todo[0]["id"][:8], note["importance"], len(note["open_threads"]))
        except Exception:
            logger.exception("memory writer pass failed")
        await asyncio.sleep(POLL_S)


def _ago(when: str, now: datetime) -> str:
    then = datetime.fromisoformat(when)
    days = (now.date() - then.astimezone(now.tzinfo).date()).days
    if days <= 0:
        hours = (now - then).total_seconds() / 3600
        return "earlier today" if hours >= 2 else "a little while ago"
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"{days} days ago"
    if days < 60:
        return f"{days // 7} weeks ago"
    return f"{days // 30} months ago"


def _lower_first(text: str) -> str:
    text = text.strip().rstrip(".")
    return text[:1].lower() + text[1:]


def _score(m: dict, now: datetime) -> float:
    age_days = (now - datetime.fromisoformat(m["happened_at"])).total_seconds() / 86400
    recency = 3 if age_days < 1 else 2 if age_days < 3 else 1 if age_days < 10 else 0
    return m["importance"] + recency + (1.5 if m["open_threads"] else 0)


def recall_note(db: MemoryDB, conversation_id: str | None, tz: timezone | None = None) -> str:
    """The memories worth carrying into this conversation, phrased for her prompt ("" if none)."""
    now = datetime.now(tz or timezone.utc)
    memories = [m for m in db.recent_memories(conversation_id) if m["importance"] >= 2]
    if not memories:
        return ""
    picked = sorted(memories, key=lambda m: _score(m, now), reverse=True)[:RECALL_COUNT]
    picked.sort(key=lambda m: m["happened_at"], reverse=True)
    lines = []
    for m in picked:
        line = f"- {_ago(m['happened_at'], now)}: {m['summary']}"
        if m["their_mood"]:
            line += f" They seemed {_lower_first(m['their_mood'])}."
        if m["her_feeling"]:
            line += f" You felt {_lower_first(m['her_feeling'])}."
        if m["open_threads"]:
            line += " Still open: " + "; ".join(t.rstrip(".") for t in m["open_threads"]) + "."
        lines.append(line)
    return RECALL_GUIDE + "\n" + "\n".join(lines)
