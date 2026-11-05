"""The approval gate for THURSDAY's self-improvement jobs (runner/agent.py, scripts/improve.sh).

A job leaves runner-out/<id>.json (its report) and <id>.bundle (its commits). Daniel reviews it on
the Agentic page; nothing reaches his code without his Approve:

  review   the diff is rebuilt here from the bundle with git, never taken from the report: the
           runner's word about what it changed doesn't count
  approve  merges thursday/improve-<id> into his checkout, only when it's on the job's base branch
           with no uncommitted changes to tracked files (his work in progress is never touched),
           and only if no protected file changed. A conflict aborts the merge and says so.
  decline  deletes the branch; the report stays as a record

Going live (rebuild) and publishing (git push) stay his, by hand. The repo is mounted at REPO_DIR.

Out loud: improve_start queues a job for the runner service (runner-out/queue), read back first
and started on his plain yes like a mission (missions.confirms_pending); improve_status says how
jobs went. run_notifier pushes each finished job to his phone, the tap opening its review card.

Self-review (run_self_review): once a day at SELF_REVIEW_HOUR (his time, quiet hours), or when he
asks (self_review), THURSDAY reads the last REVIEW_DAYS of their conversations for where it fell
short: his corrections ("this isn't John Wick music") and claims no tool backed ("Music, off."
with nothing run). The mission model turns that evidence, and a map of the code, into at most 3
small fixes, posted as a "Self-review" conversation in the chat with a phone push. They discuss it
there; a fix he settles on goes to improve_start like any other, read back first.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from typing import Any, Dict

from accounts import OWNER
from calendar_tool import TURN
from tools import ToolResult

REPO = Path(os.getenv("REPO_DIR", "/repo"))
OUT = REPO / "runner-out"
PROJECTS = Path(os.getenv("PROJECTS_DIR", "/projects"))  # his GitHub Projects folder, mounted read-only


def _project_root(name: str | None) -> tuple[Path | None, str]:
    """(root, name) for a project in GitHub Projects; THURSDAY's own repo for none or "thursday"."""
    name = (name or "").strip().strip("/")
    if not name or name.lower() == "thursday":
        return REPO, "thursday"
    match = [p for p in PROJECTS.iterdir() if p.is_dir() and p.name.lower() == name.lower()] if PROJECTS.exists() else []
    return (match[0], match[0].name) if match else (None, name)
GIT_ID = ["-c", "user.name=THURSDAY (approved by Daniel)", "-c", "user.email=runner@thursday.local",
          "-c", "safe.directory=*"]
# Same list as the runner's (runner/agent.py PROTECTED), checked again here on the real diff.
PROTECTED = re.compile(r"^(runner/|approvals\.py$|vault\.py$|improvements\.py$|docker-compose\.ya?ml$|Dockerfile|\.env|"
                       r"\.git|\.gitignore$|\.dockerignore$|scripts/improve\.sh$|scripts/deploy_sat\.sh$|brain/|infra/)")
_ID = re.compile(r"^\d{8}-\d{6}$")
MAX_QUEUED = 3
logger = logging.getLogger("thursday.voice")


class GateError(Exception):
    """Why a decision couldn't be carried out, in words for the page."""


# His other projects: secrets and git internals are never merged (THURSDAY's own list is PROTECTED).
PROJECT_PROTECTED = re.compile(r"(^|/)(\.env[^/]*|\.git(/|$)|[^/]*secret[^/]*|[^/]*\.(pem|key|p12)$|id_rsa[^/]*)", re.I)


def _repo_of(report: dict) -> Path:
    """Where a job's change goes: THURSDAY's checkout, or his project (GitHub Projects, writable only for this)."""
    root, name = _project_root(report.get("project"))
    if root is None or not (root / ".git").exists():
        raise GateError(f"the project {name} isn't a git repo in GitHub Projects any more")
    return root


def _protected(report: dict, path: str) -> bool:
    return bool(PROTECTED.match(path)) if report.get("project", "thursday") == "thursday" else bool(PROJECT_PROTECTED.search(path))


def _git(*args: str, check: bool = True, cwd: Path | None = None) -> str:
    res = subprocess.run(["git", *GIT_ID, *args], cwd=cwd or REPO, capture_output=True, text=True, timeout=60)
    if check and res.returncode:
        raise GateError((res.stderr or res.stdout).strip()[:400] or f"git {args[0]} failed")
    return res.stdout


def _report(job: str) -> dict:
    if not _ID.match(job) or not (OUT / f"{job}.json").exists():
        raise GateError("no such job")
    return json.loads((OUT / f"{job}.json").read_text())


def _save(job: str, report: dict) -> None:
    (OUT / f"{job}.json").write_text(json.dumps(report, indent=1))


def _branch(job: str, report: dict) -> str:
    """The job's branch, fetched from its bundle into its repo (THURSDAY's or the project's) if it isn't there yet."""
    branch, repo = report["branch"], _repo_of(report)
    if not _git("branch", "--list", branch, cwd=repo).strip():
        if not (OUT / f"{job}.bundle").exists():
            raise GateError("the job's bundle is missing")
        _git("fetch", "-q", str(OUT / f"{job}.bundle"), f"{branch}:{branch}", cwd=repo)
    return branch


def jobs(limit: int = 20) -> list[dict]:
    """Newest first: each report, trimmed for the page."""
    if not OUT.exists():
        return []
    out = []
    for path in sorted(OUT.glob("*.json"), reverse=True)[:limit]:
        try:
            r = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.append({k: r.get(k) for k in ("job", "status", "decision", "title", "summary", "task", "files", "base", "project",
                                           "branch", "model", "minutes", "decided_at", "merged", "error")})
    return out


def review(job: str) -> dict:
    """The real diff, from git: what approving would merge."""
    report = _report(job)
    if report.get("status") != "ready":
        return {"job": job, "status": report.get("status"), "diff": report.get("diff") or "", "files": report.get("files")}
    branch, repo = _branch(job, report), _repo_of(report)
    files = _git("diff", "--name-only", f"{report['base']}...{branch}", cwd=repo).split()
    return {"job": job, "status": "ready", "project": report.get("project", "thursday"), "files": files,
            "protected": [f for f in files if _protected(report, f)],
            "stat": _git("diff", "--stat", f"{report['base']}...{branch}", cwd=repo).strip(),
            "diff": _git("diff", f"{report['base']}...{branch}", cwd=repo)[:200_000]}


def approve(job: str) -> dict:
    report = _report(job)
    if report.get("status") != "ready" or report.get("decision"):
        raise GateError("this job isn't waiting for a decision")
    branch, repo = _branch(job, report), _repo_of(report)
    if (head := _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo).strip()) != report["base"]:
        raise GateError(f"your checkout is on {head}, not {report['base']}: switch back, then approve")
    if _git("status", "--porcelain", "--untracked-files=no", cwd=repo).strip():
        raise GateError("you have uncommitted changes: commit or stash them first, so the merge can't mix with them")
    if bad := [f for f in _git("diff", "--name-only", f"{report['base']}...{branch}", cwd=repo).split() if _protected(report, f)]:
        raise GateError(f"it changes protected files ({', '.join(bad)}): decline it")
    res = subprocess.run(["git", *GIT_ID, "merge", "--no-edit", branch], cwd=repo, capture_output=True, text=True,
                         timeout=60)
    if res.returncode:
        _git("merge", "--abort", check=False, cwd=repo)
        raise GateError("it doesn't merge cleanly with your current code (conflict): decline it and run the job again")
    report.update(decision="approved", decided_at=datetime.now(timezone.utc).isoformat(),
                  merged=_git("rev-parse", "--short", "HEAD", cwd=repo).strip())
    _save(job, report)
    return report


def deploy_request(job: str) -> dict:
    """His Deploy on a merged change: a request for the deployer (deploy/deployer.sh), which tests,
    rebuilds, checks health and rolls back on its own. One at a time."""
    report = _report(job)
    if report.get("decision") != "approved":
        raise GateError("only a merged change can be deployed")
    if report.get("project", "thursday") != "thursday":
        raise GateError("deploy is for THURSDAY's own code; his projects run however he runs them")
    state = deploy_status()
    if state.get("state") in ("requested", "testing", "building") or (OUT / "deploy" / "request.json").exists():
        raise GateError("a deploy is already running")
    (OUT / "deploy").mkdir(parents=True, exist_ok=True)
    (OUT / "deploy" / "request.json").write_text(json.dumps({"job": job, "at": datetime.now(timezone.utc).isoformat()}))
    (OUT / "deploy" / "status.json").write_text(json.dumps({"state": "requested", "detail": "waiting for the deployer",
                                                            "job": job, "at": datetime.now(timezone.utc).isoformat()}))
    logger.info("deploy of %s requested", job)
    return deploy_status()


def deploy_status() -> dict:
    try:
        return json.loads((OUT / "deploy" / "status.json").read_text())
    except (OSError, ValueError):
        return {}


def decline(job: str) -> dict:
    report = _report(job)
    if report.get("decision"):
        raise GateError("already decided")
    try:
        repo = _repo_of(report)
    except GateError:
        repo = None
    if repo and report.get("branch") and _git("branch", "--list", report["branch"], cwd=repo).strip():
        _git("branch", "-D", report["branch"], cwd=repo)
    report.update(decision="declined", decided_at=datetime.now(timezone.utc).isoformat())
    _save(job, report)
    return report


# --- out loud --------------------------------------------------------------------------------

def improve_start(args: Dict[str, Any]) -> ToolResult:
    import missions  # on first use: the read-back helpers live there, shared with missions and routines
    turn = TURN.get() or {}
    if turn.get("user") != OWNER:
        return ToolResult(ok=False, tool_name="improve_start", error="self_improvement_is_daniels_only")
    if args.get("confirmed"):
        held = missions._take_confirmed(turn, "improve_start")
        if held is None:
            started = missions._started.get((turn.get("conversation") or "", "improve_start"))
            if started and time.monotonic() - started[1] <= missions.CONFIRM_TTL_S:
                return started[0]
            return ToolResult(ok=False, tool_name="improve_start",
                              error="not_confirmed: read the task back (call without confirmed) and wait for his yes")
        args = held
    task = " ".join(str(args.get("task") or "").split())[:1000]
    if len(task.split()) < 4:
        return ToolResult(ok=False, tool_name="improve_start", error="task_too_vague: ask what he wants changed")
    root, project = _project_root(args.get("project"))
    if root is None:
        return ToolResult(ok=False, tool_name="improve_start", error=f"no_project_named_{project}: project_list shows them")
    if project != "thursday" and not (root / ".git").exists():
        return ToolResult(ok=False, tool_name="improve_start", error=f"{project}_is_not_a_git_repo: changes need git "
                          "(approving merges a branch); he can run git init there first")
    if len(list((OUT / "queue").glob("*.json"))) >= MAX_QUEUED:
        return ToolResult(ok=False, tool_name="improve_start", error=f"already_{MAX_QUEUED}_jobs_queued")
    if not args.get("confirmed"):
        held = {"task": task, "confirmed": True} | ({"project": project} if project != "thursday" else {})
        return missions._read_back(turn, "improve_start", held, f"In {project}: {task}" if project != "thursday" else task)
    job = _queue(task, project=None if project == "thursday" else project)
    result = ToolResult(ok=True, tool_name="improve_start", data={
        "job": job, "status": "STARTED",
        "note": "Confirm in one short statement, no question: you work on it when the house is quiet and his phone "
                "gets the change to review. Nothing changes until he approves it."})
    missions._started[(turn.get("conversation") or "", "improve_start")] = (result, time.monotonic())
    return result


def _queue(task: str, kind: str = "change", project: str | None = None) -> str:
    queue = OUT / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    job = datetime.now().strftime("%Y%m%d-%H%M%S")
    while (queue / f"{job}.json").exists() or (OUT / f"{job}.json").exists():
        time.sleep(1)
        job = datetime.now().strftime("%Y%m%d-%H%M%S")
    (queue / f"{job}.json").write_text(json.dumps({"job": job, "task": task, "kind": kind, "project": project,
                                                   "queued_at": datetime.now(timezone.utc).isoformat()}))
    logger.info("%s %s queued: %s", "code review" if kind == "review" else "improvement", job, task[:120])
    return job


def code_review_start(args: Dict[str, Any]) -> ToolResult:
    """A read-only look through his code for bugs: no read-back needed, it changes nothing."""
    if (TURN.get() or {}).get("user") != OWNER:
        return ToolResult(ok=False, tool_name="code_review_start", error="self_improvement_is_daniels_only")
    if len(list((OUT / "queue").glob("*.json"))) >= MAX_QUEUED:
        return ToolResult(ok=False, tool_name="code_review_start", error=f"already_{MAX_QUEUED}_jobs_queued")
    focus = " ".join(str(args.get("focus") or "").split())[:500] or "the code that handles his requests"
    root, project = _project_root(args.get("project"))
    if root is None:
        return ToolResult(ok=False, tool_name="code_review_start", error=f"no_project_named_{project}: project_list shows them")
    job = _queue(focus, "review", None if project == "thursday" else project)
    return ToolResult(ok=True, tool_name="code_review_start", data={"job": job, "status": "STARTED", "focus": focus, "note":
        "Confirm in one short statement, no question: you read your own code when the house is quiet, change "
        "nothing, and post what you find in the chat (and his phone) to talk through."})


def _review_write_up(report: dict) -> str:
    findings = report.get("findings") or []
    parts = [report.get("summary") or "I read through my code.",
             f"{len(findings)} finding(s), most serious first:" if findings else "I found nothing worth fixing."]
    for n, f in enumerate(findings, 1):
        where = f"`{f.get('file')}`" + (f", line {f['line']}" if f.get("line") else "")
        parts.append(f"**{n}. [{str(f.get('severity', '')).upper()}] {where}**\n- Problem: {f.get('problem')}\n- Fix: {f.get('fix')}")
    if findings:
        parts.append("Tell me which you want fixed, or ask about any of them. When we agree, I make the change in my "
                     "sandbox and you review it before anything is live.")
    return "\n\n".join(parts)


def improve_status(args: Dict[str, Any]) -> ToolResult:
    if (TURN.get() or {}).get("user") != OWNER:
        return ToolResult(ok=False, tool_name="improve_status", error="self_improvement_is_daniels_only")
    queued = [json.loads(p.read_text()) for p in sorted((OUT / "queue").glob("*.json"))] if (OUT / "queue").exists() else []
    said = {"ready": "waiting for his review on the Agentic page", "no_change": "changed nothing", "failed": "failed",
            "checks_failed": "its change failed the checks, so nothing was kept"}
    return ToolResult(ok=True, tool_name="improve_status", data={
        "queued": [q["task"][:120] for q in queued],
        "working_now": (OUT / "runner-busy").exists() and time.time() - (OUT / "runner-busy").stat().st_mtime < 120,
        "recent": [{"task": (j.get("task") or "")[:120], "result": j.get("decision") or said.get(j["status"], j["status"]),
                    "summary": (j.get("summary") or j.get("error") or "")[:300]} for j in jobs(5)]})


async def run_notifier() -> None:
    """Each finished job, once, to his phone; the tap opens its review card. Jobs finished before
    this process started were already announced (improve.sh, or the last run)."""
    import approvals  # on first use
    started = time.time()
    deployed = (deploy_status().get("state"), deploy_status().get("at"))
    while True:
        try:
            now = deploy_status()
            if (now.get("state"), now.get("at")) != deployed:  # a deploy moved on: his phone hears the outcome
                deployed = (now.get("state"), now.get("at"))
                if now.get("state") in ("live", "rolled_back", "refused", "broken"):
                    title = {"live": "THURSDAY's change is live", "rolled_back": "Deploy rolled back",
                             "refused": "Deploy refused", "broken": "Deploy broke: check THURSDAY"}[now["state"]]
                    await asyncio.to_thread(approvals.push, title, str(now.get("detail") or "")[:300], click=approvals.link(),
                                            tags=("rocket",) if now["state"] == "live" else ("warning",))
            for path in sorted(OUT.glob("*.json")) if OUT.exists() else []:
                report = json.loads(path.read_text())
                if report.get("notified"):
                    continue
                if path.stat().st_mtime >= started and report.get("kind") == "nightly":
                    from agent_loop import initalize_db
                    db = initalize_db()
                    conv = db.create_conversation(f"Engineer pass · {datetime.now(ZoneInfo(REVIEW_TZ)):%b %-d}", OWNER)
                    db.log_message(role="assistant", content=_nightly_write_up(report), conversation_id=conv["id"])
                    if not report.get("passed"):  # quiet when all is well
                        cause = (report.get("analysis") or {}).get("cause") or "see the chat"
                        await asyncio.to_thread(approvals.push, "THURSDAY's checks are failing", cause[:400],
                                                click=approvals.link(conv["id"]), tags=("warning",))
                    logger.info("engineer pass %s: %s", report.get("job"), "pass" if report.get("passed") else "FAIL")
                elif path.stat().st_mtime >= started and report.get("kind") == "review":
                    # A code review becomes a conversation to talk through, like a self-review.
                    from agent_loop import initalize_db
                    db = initalize_db()
                    conv = db.create_conversation(f"Code review · {datetime.now(ZoneInfo(REVIEW_TZ)):%b %-d}", OWNER)
                    db.log_message(role="user", content=f"Review your code: {report.get('task')}", conversation_id=conv["id"])
                    db.log_message(role="assistant", content=_review_write_up(report), conversation_id=conv["id"])
                    n = len(report.get("findings") or [])
                    await asyncio.to_thread(approvals.push, f"THURSDAY reviewed its code: {n} finding(s)",
                                            (report.get("summary") or "")[:400], click=approvals.link(conv["id"]), tags=("mag",))
                    logger.info("code review %s finished: %d finding(s)", report.get("job"), n)
                elif path.stat().st_mtime >= started:
                    where = "" if report.get("project", "thursday") == "thursday" else f" in {report['project']}"
                    title = f"THURSDAY has a change{where} for you to review" if report.get("status") == "ready" \
                        else f"THURSDAY's improvement job ended: {report.get('status')}"
                    await asyncio.to_thread(approvals.push, title, (report.get("summary") or report.get("error") or "")[:500],
                                            click=approvals.link(), tags=("hammer_and_wrench",))
                    logger.info("improvement %s finished: %s", report.get("job"), report.get("status"))
                report["notified"] = True
                path.write_text(json.dumps(report, indent=1))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("improvement notifier")
        await asyncio.sleep(10)


_STOP = {"the", "and", "how", "does", "what", "where", "when", "why", "you", "your", "with", "for", "that", "this",
         "work", "works", "code", "about", "from", "into", "can", "are", "is", "do", "decide", "handle", "handles", "tell",
         "explain", "show", "file", "files", "function", "thursday", "it", "its", "of", "to", "in", "on", "a", "an", "me"}


# How he says it -> the words the code uses (names and docstrings), so a plain question finds the code.
_SAYS = [(r"stop(?:ped)?\s+talking|done\s+talking|finish(?:ed)?\s+speaking|end\s+of\s+(?:my\s+)?speech",
          "endpoint silence vad end_silence smart"),
         (r"interrupt|cut\s+(?:you|her|him)\s+off|talk\s+over", "barge"),
         (r"wake\s*word|hey\s+thursday", "wake"),
         (r"remember|memory|forget", "memory fact recall"),
         (r"\bslow\b|\bfast\b|latency|lag", "timing latency"),
         (r"listen", "listening vad"),
         (r"wi-?fi|connection|disconnect|offline|internet|network", "connect reconnect connection offline wifi"),
         (r"(?:hear|hearing)\s+(?:your|yourself|my)|own\s+voice|echo|feedback", "aec echo cancel reference barge"),
         (r"\bgpu\b|memory|ram\b", "touch busy gpu memory"),
         (r"who(?:'s|\s+is)\s+(?:talking|speaking)|recogni[sz]e\s+(?:me|my\s+voice)", "speaker identify voiceprint"),
         (r"finish|is\s+done|complete", "finish announce report")]


def _stem(w: str) -> str:
    for end in ("ing", "es", "ed", "s"):
        if w.endswith(end) and len(w) - len(end) >= 4:
            return w[: -len(end)]
    return w


def _terms(text: str, question: bool = False) -> set[str]:
    """Searchable words; the phrase map (_SAYS) expands only his question: on a name it made the echo tool
    look like echo cancellation (2026-11-05)."""
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text).lower()  # CodeIndex -> code index
    extra = " ".join(add for pat, add in _SAYS if re.search(pat, text)) if question else ""
    words = re.findall(r"[a-z][a-z0-9]+", f"{text} {extra}".replace("_", " "))
    return {_stem(w) for w in words if len(w) > 2 and w not in _STOP}


EMBED_MODEL = os.getenv("CODE_EMBED_MODEL", "nomic-embed-text")
EMBED_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434/api/chat").split("/api/")[0] + "/api/embed"


def _embed(texts: list[str]) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), 64):
        r = httpx_post(EMBED_URL, {"model": EMBED_MODEL, "input": texts[i:i + 64], "keep_alive": "5m"})
        out += r["embeddings"]
    return out


def httpx_post(url: str, body: dict) -> dict:
    import httpx
    r = httpx.post(url, json=body, timeout=120)
    r.raise_for_status()
    return r.json()


def _meaning(root: Path, project: str, symbols: list, question: str) -> dict[int, float]:
    """Cosine similarity of the question to each definition, by meaning (nomic-embed-text): finds code that
    never uses his words ("Wi-Fi drops" -> the reconnect loop). Vectors are cached by each definition's text
    in runner-out/embed/<project>.json (a subfolder: runner-out/*.json are job reports). {} when it fails."""
    import math
    cache_path = OUT / "embed" / f"{project}.json"
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}
    texts, keys, sources = [], [], {}
    for i, sym in enumerate(symbols):
        if sym.path not in sources:
            sources[sym.path] = (root / sym.path).read_text(errors="replace").splitlines()
        body = "\n".join(sources[sym.path][sym.line - 1:min(sym.end, sym.line + 14)])
        text = f"{sym.path} {sym.name}\n{sym.doc}\n{body}"[:1500]
        texts.append(text)
        keys.append(hashlib.sha1(text.encode()).hexdigest())
    try:
        missing = sorted({k for k in keys if k not in cache})
        if missing:
            todo = {k: t for k, t in zip(keys, texts) if k in missing}
            for k, v in zip(todo, _embed([f"search_document: {t}" for t in todo.values()])):
                cache[k] = [round(x, 5) for x in v]
            cache = {k: cache[k] for k in set(keys)}  # changed definitions drop out
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache))
        q = _embed([f"search_query: {question}"])[0]
    except Exception as exc:  # the embedding model is down: words alone still work
        logger.info("code_ask: no meaning search (%s)", type(exc).__name__)
        return {}
    qn = math.sqrt(sum(x * x for x in q)) or 1.0
    out = {}
    for i, k in enumerate(keys):
        v = cache[k]
        out[i] = sum(a * b for a, b in zip(q, v)) / (qn * (math.sqrt(sum(x * x for x in v)) or 1.0))
    return out


def code_ask(args: Dict[str, Any]) -> ToolResult:
    """A question about how his own code works, answered from the code: the parts that match it (runner/codeindex.py:
    names, docs, constants) with their source, plus the project brief's one-liners. Read-only, ~0.2 s, no extra model:
    the chat model answers from what this returns."""
    if (TURN.get() or {}).get("user") != OWNER:
        return ToolResult(ok=False, tool_name="code_ask", error="own_code_is_daniels_only")
    question = " ".join(str(args.get("question") or "").split())[:300]
    terms = _terms(question, question=True)
    # Words his phrasing maps to ("own voice" -> echo cancellation) count double: they're what he means, while
    # the literal word ("voice") matches dozens of names (2026-11-05).
    meant = {_stem(w) for pat, add in _SAYS if re.search(pat, question.lower()) for w in add.split()}
    if not terms:
        return ToolResult(ok=False, tool_name="code_ask", error="ask_what_part_of_the_code")
    root, project = _project_root(args.get("project"))
    if root is None:
        return ToolResult(ok=False, tool_name="code_ask", error=f"no_project_named_{project}: project_list shows them")
    sys.path.insert(0, str(REPO / "runner"))
    from codeindex import CodeIndex
    idx = CodeIndex.build(root)
    tests = "test" in terms

    def words(sym) -> float:
        name, doc = _terms(sym.name), _terms(sym.doc)
        # A term counts in a name when it starts a word of it: "endpoint" finds Endpointer, "barge" _barge_in.
        named = sum(1 + (t in meant) for t in terms if any(w.startswith(t) for w in name))
        # A matching module explains the whole mechanism (satellite/aec.py's docstring for "your own voice").
        return 3 * named + len(terms & doc) + {"constant": 0, "module": 2 if named else 0}.get(sym.kind, 0.5)

    pool = [x for x in idx.symbols if tests or not x.path.startswith("tests/")]
    near = _meaning(root, project, pool, question)
    # Words and meaning together: an exact name still wins; meaning finds what his words don't name.
    # Rank fusion: a definition scores by its place in each list (words, meaning), not by raw numbers: meaning's
    # similarities sit in a narrow band (0.58-0.68), so adding them to word points let words drown them out.
    by_words = sorted((i for i, x in enumerate(pool) if words(x) >= 3), key=lambda i: -words(pool[i]))
    by_meaning = sorted(near, key=lambda i: -near[i])[:30]
    fused: dict[int, float] = {}
    for ranks in (by_words, by_meaning):
        for r, i in enumerate(ranks):
            fused[i] = fused.get(i, 0.0) + 1 / (10 + r)
    ranked = [pool[i] for i in sorted(fused, key=lambda i: -fused[i])][:3]
    # Plus meaning's own top two: on vague questions ("repeating the same joke") words pick noise and meaning
    # alone had the answer first (recent_lines_note); the chat model picks the relevant one of five.
    ranked += [pool[i] for i in by_meaning[:2] + by_words[:1] if pool[i] not in ranked]  # and words' own best
    code, used = [], 0
    for sym in ranked[:6]:
        lines = (root / sym.path).read_text(errors="replace").splitlines()[sym.line - 1:min(sym.end, sym.line + 39)]
        snippet = "\n".join(lines)[:1600]
        code.append({"at": f"{sym.path}:{sym.line}", "name": sym.name, "source": snippet})
        used += len(snippet)
    try:
        if project != "thursday":
            raise OSError("the brief is THURSDAY's own")
        cache = json.loads((OUT / "brief" / "cache.json").read_text())
        files = sorted(((len(terms & _terms(v["summary"] + " " + k)), k, v["summary"]) for k, v in cache.get("files", {}).items()),
                       reverse=True)
        about = [f"{k}: {v}" for n, k, v in files if n and (tests or not k.startswith("tests/"))][:4]
    except (OSError, ValueError):
        about = []
    if not code and not about:
        return ToolResult(ok=True, tool_name="code_ask", data={"found": False, "note":
            "Nothing in your code matches that. Say so in one line and ask which part he means."})
    return ToolResult(ok=True, tool_name="code_ask", data={"found": True, "project": project, "files": about, "code": code, "note":
        ("This is your own code." if project == "thursday" else f"This is his {project} project, not your code.") + " Explain how it works in your own voice from what's here only: 2-4 plain sentences, "
        "the mechanism and the numbers that matter, naming a file or two in plain words (no backticks or code formatting, "
        "no line numbers out loud). If it doesn't "
        "answer the question, say what you can see and that you'd need to look further."})


def project_list(args: Dict[str, Any]) -> ToolResult:
    """The projects in his GitHub Projects folder: language, size, last commit. Read-only."""
    if (TURN.get() or {}).get("user") != OWNER:
        return ToolResult(ok=False, tool_name="project_list", error="projects_are_daniels_only")
    rows = []
    for p in sorted(PROJECTS.iterdir()) if PROJECTS.exists() else []:
        if not p.is_dir() or p.name.startswith("."):
            continue
        res = subprocess.run(["git", "-C", str(p), "ls-files"], capture_output=True, text=True)
        files = res.stdout.splitlines() if res.returncode == 0 else []
        exts: Dict[str, int] = {}
        for name in files:
            if "." in name.rsplit("/", 1)[-1]:
                ext = name.rsplit(".", 1)[-1].lower()
                exts[ext] = exts.get(ext, 0) + 1
        last = subprocess.run(["git", "-C", str(p), "log", "-1", "--format=%cr: %s"], capture_output=True, text=True)
        rows.append({"name": p.name, "git": res.returncode == 0, "files": len(files),
                     "languages": [e for e, _ in sorted(exts.items(), key=lambda kv: -kv[1])[:3]],
                     "last_commit": last.stdout.strip()[:100] if last.returncode == 0 else None})
    return ToolResult(ok=True, tool_name="project_list", data={"projects": rows, "note":
        "His GitHub Projects folder. Say the names (and what they look like) in a sentence. You can explain any of "
        "them (code_ask with project) or review one for bugs (code_review_start with project); you never change them."})


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "improve_start",
        "description": "Change your own code or behavior when he asks you to improve yourself ('make your weather "
        "answers shorter', 'fix how you say times'). The first call only holds it for his yes (read it back). The "
        "change is made in a sandbox and waits for his review; nothing changes until he approves it.",
        "parameters": {"type": "object", "properties": {
            "task": {"type": "string", "description": "What to change and why, in full, with any example he gave"},
            "project": {"type": "string", "description": "One of his projects (project_list) to change instead of your "
                        "own code; it waits for his approval like any change"},
            "confirmed": {"type": "boolean", "description": "True only right after he said yes to your read-back"}},
            "required": ["task"]}}},
    {"type": "function", "function": {
        "name": "improve_status", "description": "How his self-improvement jobs went: queued, working, waiting for review.",
        "parameters": {"type": "object", "properties": {}}}},
]


# --- self-review ------------------------------------------------------------------------------

REVIEW_HOUR = int(os.getenv("SELF_REVIEW_HOUR", "3"))
ENGINEER_HOUR = int(os.getenv("ENGINEER_PASS_HOUR", "2"))  # the nightly engineer pass (runner kind "nightly")
REVIEW_DAYS = 2
REVIEW_TZ = os.getenv("SELF_REVIEW_TZ", "America/New_York")
REVIEW_STATE = Path(os.getenv("THURSDAY_BRAIN_DIR", "/app/brain")) / "self_review.json"
EVIDENCE_CHARS = 7000
_review_now = asyncio.Event()
# He's correcting THURSDAY: what came before it is worth a look.
_CORRECTION = re.compile(r"\b(?:nope|that'?s\s+(?:not|wrong)|this\s+isn'?t|isn'?t\s+(?:working|right|it)|still\s+(?:not|isn'?t)|"
                         r"not\s+working|didn'?t\s+(?:work|do|play|change|stop|turn)|you\s+didn'?t|doesn'?t\s+work|"
                         r"why\s+did\s+you|not\s+what\s+i\s+(?:asked|said|meant)|wrong\s+(?:song|one|answer|music|thing))\b", re.I)
# A reply that says something happened. With no tool result before it this turn, nothing did.
_CLAIM = re.compile(r",\s*(?:playing|added|saved|set|done|booked|deleted|off|on|paused|stopped)\s*[.!]|\bi\s+(?:added|set|saved|"
                    r"booked|turned|played|paused|stopped)\b|\bit'?s\s+(?:on|off|playing|set|saved)\s*[.!]", re.I)
_SKIP_TITLES = ("Mission ·", "Watch ·", "Self-review ·")


def _evidence(since: str) -> list[str]:
    """Snippets from his recent conversations where THURSDAY fell short."""
    from agent_loop import initalize_db  # on first use
    db = initalize_db()
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT m.role, m.content, m.conversation_id, m.created_at, c.title FROM messages m JOIN conversations c "
            "ON c.id = m.conversation_id WHERE c.user_id = ? AND m.created_at >= ? ORDER BY m.id", (OWNER, since)).fetchall()
    by_conv: dict[str, list] = {}
    for r in rows:
        if not str(r["title"]).startswith(_SKIP_TITLES):
            by_conv.setdefault(r["conversation_id"], []).append(
                (r["role"], " ".join(str(r["content"]).split())[:300], str(r["created_at"])[:16].replace("T", " ")))
    out = []
    for convo in by_conv.values():
        msgs = [(role, text) for role, text, _ in convo]
        tooled = False  # a tool result since his last message
        for i, (role, text) in enumerate(msgs):
            when = f" (at {convo[i][2]} UTC)"
            if role == "user":
                tooled = False
                if i and _CORRECTION.search(text):
                    before = [f"{'Daniel' if r == 'user' else 'THURSDAY'}: {t}" for r, t in msgs[max(0, i - 3):i]
                              if not t.startswith("TOOL_RESULT:")]
                    after = next((t for r, t in msgs[i + 1:] if r == "assistant" and not t.startswith("TOOL_RESULT:")), "")
                    out.append(f"CORRECTION{when}\n" + "\n".join(before + [f"Daniel: {text}", f"THURSDAY: {after}"]))
            elif text.startswith("TOOL_RESULT:"):
                tooled = True
            elif _CLAIM.search(text) and not tooled:
                said = next((t for r, t in reversed(msgs[:i]) if r == "user"), "")
                out.append(f"CLAIM WITHOUT A TOOL{when}\nDaniel: {said}\nTHURSDAY: {text}")
    seen, kept, size = set(), [], 0
    for e in reversed(out):  # newest first: what's still going wrong gets the room
        if e not in seen and size + len(e) <= EVIDENCE_CHARS:
            seen.add(e); kept.append(e); size += len(e)
    return kept


def _code_map() -> str:
    """Each Python file and the first line of what it says it does: enough to name where a fix goes."""
    lines = []
    for path in sorted(Path("/app").glob("*.py")) + sorted(Path("/app/satellite").glob("*.py")):
        try:
            doc = (ast.get_docstring(ast.parse(path.read_text())) or "").strip().splitlines()
        except (SyntaxError, OSError):
            continue
        lines.append(f"- {path.relative_to('/app')}: {doc[0][:140] if doc else ''}")
    return "\n".join(lines)


_REVIEW_SYSTEM = """You are THURSDAY's engineer, reviewing the last {days} days of THURSDAY's conversations with Daniel to find where THURSDAY fell short, and proposing fixes to its own code.

The evidence below is only the moments that looked wrong: his corrections, and replies that claimed an action with no tool run that turn (so nothing happened). Read it, find the real patterns, and propose at most 3 fixes. Each fix must be:
- grounded in the evidence (quote it), about a pattern not a one-off,
- small and concrete: one change to code or the prompt, in one or two files from the map below,
- something a careful engineer could do in an hour, with how to check it worked.
Skip anything already working, anything vague ("be smarter"), and anything about the voice model choice or hardware.

What's already done, so you don't propose it again:
- The commits below are fixes made since the evidence began, each with its time (UTC) and files. Evidence from before a commit that fixed that kind of thing is already handled: skip it, unless the same thing happened again after the commit.
- THURSDAY's current prompt is below. Don't propose a prompt rule it already has: if the model broke a rule that's there, the fix belongs in code (acting on the request in code, a guard, a retry), not in the prompt again.

Reply with JSON only, no prose: {{"ideas": [{{"title": "...", "problem": "...", "evidence": "...", "fix": "...", "files": ["..."], "check": "..."}}]}}. If nothing is worth fixing, {{"ideas": []}}.

The code (file: what it does):
{code}

Commits since the evidence began (UTC time | message | files):
{commits}

THURSDAY's current prompt:
{prompt}"""


def _recent_commits(since: str) -> str:
    """Fixes since the evidence began: UTC time, subject, files. Read from his checkout (read only)."""
    try:
        log = _git("log", f"--since={since}", "--date=iso-strict-local", "--format=@@%cd|%s", "--name-only",
                   "--", ".", ":!runner-out")
    except GateError:
        return "(no git history available)"
    rows = []
    for block in log.split("@@")[1:]:
        head, *files = [line for line in block.splitlines() if line.strip()]
        stamp, _, subject = head.partition("|")
        try:
            stamp = datetime.fromisoformat(stamp).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass
        rows.append(f"- {stamp} | {subject[:110]} | {', '.join(files[:6])}")
    return "\n".join(rows[:40]) or "(none)"


def _current_prompt() -> str:
    import agent_stream  # on first use: agent_stream imports this module
    return agent_stream.SYSTEM_PROMPT[:9000]


def _parse_ideas(text: str) -> list[dict]:
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1].removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        ideas = json.loads(text[start:end + 1]).get("ideas") or []
    except (ValueError, AttributeError):
        return []
    return [i for i in ideas if isinstance(i, dict) and i.get("title") and i.get("fix")][:3]


def _write_up(ideas: list[dict], evidence_count: int) -> str:
    parts = [f"I read our last {REVIEW_DAYS} days and found {evidence_count} moment(s) where I fell short. "
             f"{len(ideas)} thing(s) I'd change in myself:"]
    for n, i in enumerate(ideas, 1):
        files = ", ".join(f"`{f}`" for f in i.get("files") or [])
        parts.append(f"**{n}. {i['title']}**\n- Problem: {i.get('problem', '')}\n- Evidence: {i.get('evidence', '')}\n"
                     f"- Fix: {i['fix']}" + (f"\n- Files: {files}" if files else "") +
                     (f"\n- How we'd know: {i['check']}" if i.get("check") else ""))
    parts.append("Tell me what you think of these. When we agree on one, I make it in my sandbox and you review the "
                 "change before anything is live.")
    return "\n\n".join(parts)


async def self_review() -> str:
    """One review: evidence, the model's ideas, a Self-review conversation and a push. Returns what happened."""
    import approvals
    import missions
    from agent_loop import initalize_db
    since = (datetime.now(timezone.utc) - timedelta(days=REVIEW_DAYS)).isoformat()
    evidence = await asyncio.to_thread(_evidence, since)
    digest = hashlib.sha256("\n".join(evidence).encode()).hexdigest()
    state = json.loads(REVIEW_STATE.read_text()) if REVIEW_STATE.exists() else {}
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    if not evidence or digest == state.get("digest"):
        REVIEW_STATE.write_text(json.dumps(state))
        logger.info("self-review: %s", "nothing to review" if not evidence else "no new evidence")
        return "nothing new"
    system = _REVIEW_SYSTEM.format(days=REVIEW_DAYS, code=_code_map(), commits=await asyncio.to_thread(_recent_commits, since),
                                   prompt=_current_prompt())
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Evidence:\n\n" + "\n\n".join(evidence)}]
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(900, connect=5)) as http:
            msg, secs = await missions._call_model(http, messages, [])
    finally:
        missions._flag("mission-busy", False)
    ideas = _parse_ideas(msg.get("content") or "")
    state["digest"] = digest
    REVIEW_STATE.write_text(json.dumps(state))
    logger.info("self-review: %d evidence, %d idea(s), %.0f s", len(evidence), len(ideas), secs)
    if not ideas:
        return "no ideas"
    db = initalize_db()
    conv = db.create_conversation(f"Self-review · {datetime.now(ZoneInfo(REVIEW_TZ)):%b %-d}", OWNER)
    db.log_message(role="assistant", content=_write_up(ideas, len(evidence)), conversation_id=conv["id"])
    await asyncio.to_thread(approvals.push, "THURSDAY has ideas to improve itself",
                            "; ".join(i["title"] for i in ideas)[:400], click=approvals.link(conv["id"]), tags=("bulb",))
    return f"{len(ideas)} idea(s)"


async def run_self_review() -> None:
    """Daily at REVIEW_HOUR his time (or now, when he asks), at most once a day."""
    while True:
        try:
            await asyncio.wait_for(_review_now.wait(), timeout=60)
        except asyncio.TimeoutError:
            pass
        try:
            asked = _review_now.is_set()
            _review_now.clear()
            now = datetime.now(ZoneInfo(REVIEW_TZ))
            state = json.loads(REVIEW_STATE.read_text()) if REVIEW_STATE.exists() else {}
            ran_today = state.get("last_run", "")[:10] == now.astimezone(timezone.utc).isoformat()[:10]
            if asked or (now.hour == REVIEW_HOUR and not ran_today):
                await self_review()
            today = now.date().isoformat()
            if now.hour == ENGINEER_HOUR and state.get("engineer_queued") != today:
                _queue("nightly engineer pass", "nightly")
                state = json.loads(REVIEW_STATE.read_text()) if REVIEW_STATE.exists() else {}
                REVIEW_STATE.write_text(json.dumps(state | {"engineer_queued": today}))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("self-review")


def engineer_pass_now(args: Dict[str, Any]) -> ToolResult:
    """The nightly engineer pass, now: checks on the latest code, and the likely cause if anything fails."""
    if (TURN.get() or {}).get("user") != OWNER:
        return ToolResult(ok=False, tool_name="engineer_pass_now", error="self_improvement_is_daniels_only")
    if any(json.loads(p.read_text()).get("kind") == "nightly" for p in (OUT / "queue").glob("*.json")):
        return ToolResult(ok=True, tool_name="engineer_pass_now", data={"status": "ALREADY_QUEUED", "note": "Say it's already on the way."})
    job = _queue("engineer pass (asked)", "nightly")
    return ToolResult(ok=True, tool_name="engineer_pass_now", data={"job": job, "status": "STARTED", "note":
        "Confirm in one short statement: you run your checks when the house is quiet, and the result lands in the "
        "chat (his phone too if anything's broken)."})


def _nightly_write_up(report: dict) -> str:
    commits = report.get("commits") or []
    parts = [("All checks pass" if report.get("passed") else "**Checks are failing.**") +
             f" on {report.get('head')}, after {len(commits)} new commit(s)."]
    if not report.get("passed"):
        a = report.get("analysis") or {}
        parts.append(f"- Cause: {a.get('cause') or 'unclear from the output'}\n- Likely commit: {a.get('commit') or 'unknown'}"
                     f"\n- Fix: {a.get('fix') or 'look at the output below'}")
        parts.append(f"```\n{(report.get('checks') or '')[-1200:]}\n```")
    if report.get("listen"):
        parts.append("Voice-timing code changed (" + ", ".join(report["listen"]) + "): worth a quick listen on the satellite.")
    if commits:
        parts.append("New commits:\n" + "\n".join(f"- {c}" for c in commits[:10]))
    return "\n\n".join(parts)


def self_review_now(args: Dict[str, Any]) -> ToolResult:
    if (TURN.get() or {}).get("user") != OWNER:
        return ToolResult(ok=False, tool_name="self_review_now", error="self_improvement_is_daniels_only")
    if _review_loop is None:
        return ToolResult(ok=False, tool_name="self_review_now", error="review_not_running")
    _review_loop.call_soon_threadsafe(_review_now.set)
    return ToolResult(ok=True, tool_name="self_review_now", data={
        "note": "It runs when the house is quiet; the ideas show up as a Self-review conversation in the chat and on "
                "his phone. Say so in one short statement."})


_review_loop: asyncio.AbstractEventLoop | None = None


async def start_self_review() -> None:
    global _review_loop
    _review_loop = asyncio.get_running_loop()
    await run_self_review()


TOOL_SCHEMAS.append({"type": "function", "function": {
    "name": "code_ask",
    "description": "Look at code to answer how something works: your own ('how does barge-in work?', 'where do you "
    "decide when I've stopped talking?') or one of his projects (project). Read-only and instant.",
    "parameters": {"type": "object", "properties": {
        "question": {"type": "string", "description": "His question, with the part of the system it's about"},
        "project": {"type": "string", "description": "One of his projects (project_list); leave out for your own code"}},
        "required": ["question"]}}})
TOOL_SCHEMAS.append({"type": "function", "function": {
    "name": "project_list",
    "description": "The projects in his GitHub Projects folder (names, languages, last commit): when he asks what "
    "projects he has, or names one you don't know.",
    "parameters": {"type": "object", "properties": {}}}})
TOOL_SCHEMAS.append({"type": "function", "function": {
    "name": "code_review_start",
    "description": "Read through your own code for bugs and report them (changes nothing): when he asks you to review, "
    "audit or find bugs in your code, or in one of his projects (project). Findings arrive as a Code review "
    "conversation in the chat. His projects are only ever reviewed, never changed.",
    "parameters": {"type": "object", "properties": {
        "focus": {"type": "string", "description": "What to look at, if he named it: 'the music code', 'alarms'"},
        "project": {"type": "string", "description": "One of his projects (project_list); leave out for your own code"}}}}})
TOOL_SCHEMAS.append({"type": "function", "function": {
    "name": "engineer_pass_now",
    "description": "Run your nightly engineer check now: the full tests on your latest code, the commits since the "
    "last check, and the likely cause if anything fails. For 'run your checks', 'are your tests passing?'.",
    "parameters": {"type": "object", "properties": {}}}})
TOOL_SCHEMAS.append({"type": "function", "function": {
    "name": "self_review_now",
    "description": "Review your recent conversations with him for where you fell short, and propose fixes to yourself "
    "(also runs every night by itself). The ideas arrive as a Self-review conversation in the chat.",
    "parameters": {"type": "object", "properties": {}}}})
