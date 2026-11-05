"""The project brief: what each module does in one sentence, and how the whole thing fits together,
written by the local model and cached by each file's hash so only changed files are redone.

The runner puts it at the top of every job instead of a bare file list: the job starts knowing where
things live and how a voice turn flows, instead of paging through files to find out (benchmark t4,
2026-11-05). Tests get their docstring, not a model call. Cache: /out/brief/cache.json.

    python /runner/brief.py              # build or refresh it from the base branch (first build ~10-15 min)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from codeindex import CodeIndex

MODEL = os.getenv("RUNNER_MODEL", "gemma4:31b")
CHAT_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434").rstrip("/") + "/api/chat"
# Always the shared /out, never RUNNER_OUT: the benchmark points that at runner-out/bench, and its first
# job rebuilt the whole brief (~12 min) inside the job (t2, 2026-11-05).
CACHE = Path(os.getenv("BRIEF_CACHE", "/out/brief/cache.json"))
FILE_CHARS = 9000      # of a file's source, after its outline: enough to see what it really does
ROLLUP_WORDS = 300

FILE_PROMPT = """You write one line of a code map for an engineer about to change this codebase.
In ONE sentence (at most 30 words), say what this file does and its main entry points or data, concretely
(name the functions or tools that matter). No preamble, no "This file". Plain text."""
# Checked against the code by hand (2026-11-05): the model's first overview got exactly these wrong.
# They go into the overview prompt as ground truth and sit at the top of the brief. Update them when
# the architecture changes.
FACTS = """- Live speech: voice_ws.py streams each reply sentence by sentence from the Orpheus TTS server (port 9001,
  ~/thursday-voice, outside this repo) through _tts_stream. voice_out.py is the old Qwen TTS server (port 9000),
  a fallback only. STT is voice_in.py / stt_native.py (faster-whisper).
- A new tool needs both: its function registered in tools.py's TOOLS dict (or a domain module's tool list),
  and its JSON schema in agent_stream.py's TOOLS list (which also adds missions/notes/approvals/lights schemas).
  Clear requests can be routed in code before the model (agent_stream._voice_intent, _music_intent, ...).
- Persona: agent_stream.py holds both personalities: SYSTEM_PROMPT (Rocky) and TARA_PROMPT (Tara), plus STYLE /
  STYLE_TARA and the example banks. mind.py holds settings (honesty/humor/discretion, voice) and stances.
  The voice setting picks the persona (mind.voice_style "voice": leo = Rocky, tara = Tara).
- eridian.py (Rocky's grammar) and rocky_voice.py (Rocky's sound) apply only when Rocky is the voice.
- runner/ (this agent's own code) is protected: changes there are refused."""

ROLLUP_PROMPT = f"""You write the architecture overview an engineer reads before changing this codebase.
From the one-line file summaries below, in at most {ROLLUP_WORDS} words of plain text: the main parts and
what each owns; how one spoken request flows through them, step by step, naming files; and where the
usual changes go (a new tool, the persona, voice timing, the satellite, tests). Only what the summaries
support. No Markdown headings; short paragraphs. These facts are verified and override anything the
summaries suggest; never contradict them:
{FACTS}"""


def _ask(http: httpx.Client, system: str, user: str) -> str:
    r = http.post(CHAT_URL, json={"model": MODEL, "stream": False, "think": False, "keep_alive": "2m",
                                  "options": {"num_ctx": 16384, "temperature": 0.2},
                                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                  timeout=600)
    r.raise_for_status()
    return " ".join(r.json()["message"]["content"].split())


def _load() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except (OSError, ValueError):
        return {}


def _save(cache: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=1))


def refresh(root: Path, log=lambda m: print(m, flush=True)) -> str:
    """Bring the brief up to date with the code in root; returns it as text for a prompt."""
    cache = _load()
    files, idx = cache.get("files", {}), CodeIndex.build(root)
    paths = sorted(idx.imports)  # every tracked .py file that parsed
    rows, changed = [], 0
    with httpx.Client() as http:
        for rel in paths:
            text = (root / rel).read_text(errors="replace")
            sha = hashlib.sha1(text.encode()).hexdigest()
            if files.get(rel, {}).get("sha") != sha:
                started = time.monotonic()
                if rel.startswith("tests/"):
                    doc = text.split('"""')[1].strip().splitlines()[0] if text.lstrip().startswith('"""') else ""
                    summary = doc or "Tests."
                else:
                    outline = "\n".join(idx.outline(rel)[:80])
                    summary = _ask(http, FILE_PROMPT, f"File: {rel}\n\nOutline:\n{outline}\n\nSource (start):\n{text[:FILE_CHARS]}")
                files[rel] = {"sha": sha, "summary": summary}
                cache["files"] = files
                _save(cache)  # after every file: a crash halfway keeps what's done
                changed += 1
                log(f"brief: {rel} ({time.monotonic() - started:.1f} s)")
            rows.append(f"- {rel}: {files[rel]['summary']}")
        files = {k: v for k, v in files.items() if k in paths}  # deleted files drop out
        listing = "\n".join(rows)
        key = hashlib.sha1((FACTS + listing).encode()).hexdigest()  # new facts redo the overview
        if cache.get("rollup_key") != key:
            started = time.monotonic()
            cache["rollup"] = _ask(http, ROLLUP_PROMPT, listing)
            cache["rollup_key"] = key
            log(f"brief: architecture overview ({time.monotonic() - started:.1f} s)")
    cache["files"] = files
    _save(cache)
    log(f"brief: {len(paths)} files, {changed} refreshed")
    return f"Verified facts:\n{FACTS}\n\nHow it fits together:\n{cache['rollup']}\n\nThe files:\n{listing}"


if __name__ == "__main__":  # a full build from the base branch, outside any job
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "-q", "--branch", os.getenv("RUNNER_BASE", "container-branch"),
                        "/repo.git", tmp], check=True)
        print(refresh(Path(tmp)), file=sys.stderr)
