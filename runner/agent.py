"""THURSDAY's self-improvement runner: one coding job, in a sandbox, ending on a branch for Daniel.

    docker compose --profile runner run --rm -e JOB_ID=<id> runner "<task>"   (scripts/improve.sh does this)

The repo arrives as its .git folder only, read-only (/repo.git): git never holds .env, brain/ or
any ignored file, so no secret or memory reaches the sandbox. The job clones it into /work, branches
thursday/improve-<id> off RUNNER_BASE, and a local model (RUNNER_MODEL, Ollama on the Mac) edits
with a few tools. Protected paths can't be touched. Checks (runner/checks.py) must pass before it
commits; the commits leave as a git bundle in /out with a JSON report. Nothing is merged, pushed or
deployed here: that's Daniel's call, after he reads the diff.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

MODEL = os.getenv("RUNNER_MODEL", "gemma4:31b")
CHAT_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434").rstrip("/") + "/api/chat"
BASE = os.getenv("RUNNER_BASE", "container-branch")
JOB = os.getenv("JOB_ID") or time.strftime("%Y%m%d-%H%M%S")
NUM_CTX = 16384
MAX_STEPS = 40
MAX_WORK_S = 20 * 60
READ_LINES = 200
RESULT_CHARS = 6000
WORK, OUT, SRC = Path("/work"), Path("/out"), "/repo.git"
BRANCH = f"thursday/improve-{JOB}"

# What a job may never change: its own sandbox, the approval gate, secrets and deploy config.
PROTECTED = re.compile(r"^(runner/|approvals\.py$|vault\.py$|improvements\.py$|docker-compose\.ya?ml$|Dockerfile|\.env|\.git|"
                       r"\.gitignore$|\.dockerignore$|scripts/improve\.sh$|scripts/deploy_sat\.sh$|brain/|infra/)")


def _git(*args: str, check: bool = True) -> str:
    res = subprocess.run(["git", *args], cwd=WORK, capture_output=True, text=True)
    if check and res.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {res.stderr.strip()[:300]}")
    return res.stdout


def _path(rel: str) -> Path:
    """A path inside the checkout; anything outside it (../, absolute) is refused."""
    p = (WORK / str(rel or ".").lstrip("/")).resolve()
    if p != WORK and WORK not in p.parents:
        raise ValueError("outside the repo")
    return p


def _rel(p: Path) -> str:
    return str(p.relative_to(WORK)) if p != WORK else "."


# --- tools ------------------------------------------------------------------------------

def list_files(args: dict) -> dict:
    tracked = _git("ls-files", "--", _rel(_path(args.get("dir", ".")))).split()
    return {"files": tracked[:300], "more": max(0, len(tracked) - 300)}


def read_file(args: dict) -> dict:
    p = _path(args["path"])
    lines = p.read_text(errors="replace").splitlines()
    start = max(1, int(args.get("start") or 1))
    chunk = lines[start - 1:start - 1 + READ_LINES]
    return {"path": _rel(p), "lines": f"{start}-{start + len(chunk) - 1} of {len(lines)}",
            "text": "\n".join(f"{start + i}\t{line}" for i, line in enumerate(chunk))}


def search(args: dict) -> dict:
    res = subprocess.run(["git", "grep", "-n", "-I", "-E", str(args["pattern"]), "--", _rel(_path(args.get("dir", ".")))],
                         cwd=WORK, capture_output=True, text=True)
    hits = res.stdout.splitlines()
    return {"hits": [h[:200] for h in hits[:60]], "more": max(0, len(hits) - 60)}


def _writable(rel: str) -> Path:
    p = _path(rel)
    if PROTECTED.match(_rel(p)):
        raise PermissionError(f"protected: {_rel(p)} can't be changed by a job; say so in finish instead")
    return p


def edit_file(args: dict) -> dict:
    p = _writable(args["path"])
    text, old, new = p.read_text(), str(args["old"]), str(args["new"])
    if text.count(old) != 1:
        return {"ok": False, "error": f"old text found {text.count(old)} times; it must match exactly once (read the file again)"}
    p.write_text(text.replace(old, new))
    return {"ok": True, "path": _rel(p)}


def create_file(args: dict) -> dict:
    p = _writable(args["path"])
    if p.exists():
        return {"ok": False, "error": "exists: use edit_file"}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(args["content"]))
    return {"ok": True, "path": _rel(p)}


def run_checks(_: dict | None = None) -> dict:
    res = subprocess.run([sys.executable, "/runner/checks.py", str(WORK)], capture_output=True, text=True, timeout=300)
    return {"passed": res.returncode == 0, "output": (res.stdout + res.stderr)[-3000:]}


TOOLS = {"list_files": list_files, "read_file": read_file, "search": search, "edit_file": edit_file,
         "create_file": create_file, "run_checks": run_checks}


def _schema(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


_S = {"type": "string"}
SCHEMAS = [
    _schema("list_files", "Tracked files under a folder.", {"dir": _S}, []),
    _schema("read_file", f"Read {READ_LINES} lines of a file, numbered, from start (default 1).",
            {"path": _S, "start": {"type": "integer"}}, ["path"]),
    _schema("search", "git grep -E across the repo (or a folder): find where something is.", {"pattern": _S, "dir": _S},
            ["pattern"]),
    _schema("edit_file", "Replace old with new in a file. old must match the file exactly once, whitespace included.",
            {"path": _S, "old": _S, "new": _S}, ["path", "old", "new"]),
    _schema("create_file", "Create a new file.", {"path": _S, "content": _S}, ["path", "content"]),
    _schema("run_checks", "Compile every Python file and run THURSDAY's quick tests.", {}, []),
    _schema("finish", "Done: what you changed and why, for Daniel's review. Checks must pass first.",
            {"summary": {"type": "string", "description": "2-4 sentences: what changed, why, anything he should test"},
             "title": {"type": "string", "description": "Commit title, under 60 characters"}}, ["summary", "title"]),
]


def _system() -> str:
    guide = (WORK / "CLAUDE.md").read_text()[:5000] if (WORK / "CLAUDE.md").exists() else ""
    return f"""You are THURSDAY's engineer, improving THURSDAY's own code in a sandbox. Daniel reviews every change before it goes anywhere; nothing you do here is live.

How to work:
- Find the code first (search, then read_file around the hits). Read before you edit; edit_file needs the exact current text.
- Make the smallest change that does the task. Match the surrounding style, comments and naming. Don't refactor or touch unrelated code.
- Protected files (runner/, approvals.py, vault.py, docker-compose, Dockerfiles, .env, .gitignore, .dockerignore, infra/, deploy scripts) can't be changed. If the task needs one, say so in finish without changing it.
- After your edits, run_checks. If they fail, fix what you broke. Then call finish once.
- If the task is unclear or can't be done safely, change nothing and call finish explaining why.
- You have {MAX_STEPS} tool calls and about {MAX_WORK_S // 60} minutes.

The project's guide (CLAUDE.md):
{guide}"""


def _trim(messages: list[dict]) -> None:
    """Stay inside the context: shorten the oldest tool results first."""
    for m in messages:
        if sum(len(x.get("content") or "") for x in messages) <= NUM_CTX * 3:
            return
        if m["role"] == "tool" and len(m["content"]) > 500:
            m["content"] = m["content"][:500] + " [trimmed]"


def _chat(http: httpx.Client, messages: list[dict], tools: list[dict]) -> dict:
    res = http.post(CHAT_URL, json={"model": MODEL, "messages": messages, "tools": tools, "stream": False, "think": True,
                                    "keep_alive": "2m", "options": {"num_ctx": NUM_CTX}})
    res.raise_for_status()
    return res.json()["message"]


def _report(**fields) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{JOB}.json").write_text(json.dumps({"job": JOB, "branch": BRANCH, "base": BASE, "model": MODEL, **fields},
                                                indent=1))
    print(json.dumps(fields.get("status")), fields.get("summary", ""), sep="\n")


def run(task: str) -> None:
    subprocess.run(["git", "clone", "-q", "--branch", BASE, SRC, str(WORK)], check=True)
    _git("checkout", "-q", "-b", BRANCH)
    base_sha = _git("rev-parse", "HEAD").strip()
    messages = [{"role": "system", "content": _system()}, {"role": "user", "content": f"Task: {task}"}]
    steps, started, nudges = [], time.monotonic(), 0
    with httpx.Client(timeout=httpx.Timeout(600, connect=5)) as http:
        while True:
            over = len(steps) >= MAX_STEPS or time.monotonic() - started > MAX_WORK_S
            if over and nudges >= 2:
                return _report(status="failed", error="out_of_budget", task=task, steps=steps)
            if over:
                nudges += 1
                messages.append({"role": "user", "content": "Budget's spent: run_checks if you haven't, then finish."})
            msg = _chat(http, messages, SCHEMAS)
            calls = msg.get("tool_calls") or []
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            if not calls:
                nudges += 1
                if nudges > 3:
                    return _report(status="failed", error="no_finish", task=task, steps=steps)
                messages.append({"role": "user", "content": "Keep going with the tools, or call finish."})
                continue
            for call in calls:
                name, args = call["function"]["name"], call["function"].get("arguments") or {}
                if name == "finish":
                    return _finish(task, args, steps, base_sha, started)
                try:
                    result = TOOLS[name](args) if name in TOOLS else {"error": "unknown tool"}
                except Exception as exc:  # a bad path or argument is the model's to route around
                    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
                steps.append({"tool": name, "args": {k: (str(v)[:120]) for k, v in args.items()},
                              "ok": result.get("ok", True) is not False and result.get("passed", True) is not False})
                print(f"step {len(steps)}: {name} {json.dumps(steps[-1]['args'])[:140]}", flush=True)
                messages.append({"role": "tool", "tool_name": name,
                                 "content": json.dumps(result, ensure_ascii=False)[:RESULT_CHARS]})
            _trim(messages)


def _finish(task: str, args: dict, steps: list, base_sha: str, started: float) -> None:
    summary, title = str(args.get("summary") or "").strip(), str(args.get("title") or "THURSDAY improvement").strip()[:72]
    changed = [line[3:] for line in _git("status", "--porcelain").splitlines()]
    if not changed:
        return _report(status="no_change", task=task, summary=summary, steps=steps)
    checks = run_checks()  # the gate is ours, not the model's word
    if not checks["passed"]:
        return _report(status="checks_failed", task=task, summary=summary, files=changed, checks=checks["output"],
                       steps=steps, diff=_git("diff")[:20000])
    _git("add", "-A")
    _git("-c", "user.name=THURSDAY runner", "-c", "user.email=runner@thursday.local", "commit", "-q", "-m",
         f"{title}\n\n{summary}\n\nTask: {task}\nModel: {MODEL} (THURSDAY runner job {JOB})")
    _git("bundle", "create", str(OUT / f"{JOB}.bundle"), f"{base_sha}..{BRANCH}")
    _report(status="ready", task=task, summary=summary, title=title, files=changed, checks=checks["output"][-800:],
            steps=steps, minutes=round((time.monotonic() - started) / 60, 1))


if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        sys.exit("usage: agent.py '<task>'")
    run(sys.argv[1].strip())
