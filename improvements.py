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


def _git(*args: str, check: bool = True) -> str:
    res = subprocess.run(["git", *GIT_ID, *args], cwd=REPO, capture_output=True, text=True, timeout=60)
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
    """The job's branch, fetched from its bundle if it isn't in the repo yet."""
    branch = report["branch"]
    if not _git("branch", "--list", branch).strip():
        if not (OUT / f"{job}.bundle").exists():
            raise GateError("the job's bundle is missing")
        _git("fetch", "-q", str(OUT / f"{job}.bundle"), f"{branch}:{branch}")
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
        out.append({k: r.get(k) for k in ("job", "status", "decision", "title", "summary", "task", "files", "base",
                                           "branch", "model", "minutes", "decided_at", "merged", "error")})
    return out


def review(job: str) -> dict:
    """The real diff, from git: what approving would merge."""
    report = _report(job)
    if report.get("status") != "ready":
        return {"job": job, "status": report.get("status"), "diff": report.get("diff") or "", "files": report.get("files")}
    branch = _branch(job, report)
    files = _git("diff", "--name-only", f"{report['base']}...{branch}").split()
    return {"job": job, "status": "ready", "files": files, "protected": [f for f in files if PROTECTED.match(f)],
            "stat": _git("diff", "--stat", f"{report['base']}...{branch}").strip(),
            "diff": _git("diff", f"{report['base']}...{branch}")[:200_000]}


def approve(job: str) -> dict:
    report = _report(job)
    if report.get("status") != "ready" or report.get("decision"):
        raise GateError("this job isn't waiting for a decision")
    branch = _branch(job, report)
    if (head := _git("rev-parse", "--abbrev-ref", "HEAD").strip()) != report["base"]:
        raise GateError(f"your checkout is on {head}, not {report['base']}: switch back, then approve")
    if _git("status", "--porcelain", "--untracked-files=no").strip():
        raise GateError("you have uncommitted changes: commit or stash them first, so the merge can't mix with them")
    if bad := [f for f in _git("diff", "--name-only", f"{report['base']}...{branch}").split() if PROTECTED.match(f)]:
        raise GateError(f"it changes protected files ({', '.join(bad)}): decline it")
    res = subprocess.run(["git", *GIT_ID, "merge", "--no-edit", branch], cwd=REPO, capture_output=True, text=True,
                         timeout=60)
    if res.returncode:
        _git("merge", "--abort", check=False)
        raise GateError("it doesn't merge cleanly with your current code (conflict): decline it and run the job again")
    report.update(decision="approved", decided_at=datetime.now(timezone.utc).isoformat(),
                  merged=_git("rev-parse", "--short", "HEAD").strip())
    _save(job, report)
    return report


def deploy_request(job: str) -> dict:
    """His Deploy on a merged change: a request for the deployer (deploy/deployer.sh), which tests,
    rebuilds, checks health and rolls back on its own. One at a time."""
    report = _report(job)
    if report.get("decision") != "approved":
        raise GateError("only a merged change can be deployed")
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
    if report.get("branch") and _git("branch", "--list", report["branch"]).strip():
        _git("branch", "-D", report["branch"])
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
    if len(list((OUT / "queue").glob("*.json"))) >= MAX_QUEUED:
        return ToolResult(ok=False, tool_name="improve_start", error=f"already_{MAX_QUEUED}_jobs_queued")
    if not args.get("confirmed"):
        return missions._read_back(turn, "improve_start", {"task": task, "confirmed": True}, task)
    job = _queue(task)
    result = ToolResult(ok=True, tool_name="improve_start", data={
        "job": job, "status": "STARTED",
        "note": "Confirm in one short statement, no question: you work on it when the house is quiet and his phone "
                "gets the change to review. Nothing changes until he approves it."})
    missions._started[(turn.get("conversation") or "", "improve_start")] = (result, time.monotonic())
    return result


def _queue(task: str, kind: str = "change") -> str:
    queue = OUT / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    job = datetime.now().strftime("%Y%m%d-%H%M%S")
    while (queue / f"{job}.json").exists() or (OUT / f"{job}.json").exists():
        time.sleep(1)
        job = datetime.now().strftime("%Y%m%d-%H%M%S")
    (queue / f"{job}.json").write_text(json.dumps({"job": job, "task": task, "kind": kind,
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
    job = _queue(focus, "review")
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
                if path.stat().st_mtime >= started and report.get("kind") == "review":
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
                    title = "THURSDAY has a change for you to review" if report.get("status") == "ready" \
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


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "improve_start",
        "description": "Change your own code or behavior when he asks you to improve yourself ('make your weather "
        "answers shorter', 'fix how you say times'). The first call only holds it for his yes (read it back). The "
        "change is made in a sandbox and waits for his review; nothing changes until he approves it.",
        "parameters": {"type": "object", "properties": {
            "task": {"type": "string", "description": "What to change and why, in full, with any example he gave"},
            "confirmed": {"type": "boolean", "description": "True only right after he said yes to your read-back"}},
            "required": ["task"]}}},
    {"type": "function", "function": {
        "name": "improve_status", "description": "How his self-improvement jobs went: queued, working, waiting for review.",
        "parameters": {"type": "object", "properties": {}}}},
]


# --- self-review ------------------------------------------------------------------------------

REVIEW_HOUR = int(os.getenv("SELF_REVIEW_HOUR", "3"))
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
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("self-review")


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
    "name": "code_review_start",
    "description": "Read through your own code for bugs and report them (changes nothing): when he asks you to review, "
    "audit or find bugs in your code. Findings arrive as a Code review conversation in the chat.",
    "parameters": {"type": "object", "properties": {
        "focus": {"type": "string", "description": "What to look at, if he named it: 'the music code', 'alarms'"}}}}})
TOOL_SCHEMAS.append({"type": "function", "function": {
    "name": "self_review_now",
    "description": "Review your recent conversations with him for where you fell short, and propose fixes to yourself "
    "(also runs every night by itself). The ideas arrive as a Self-review conversation in the chat.",
    "parameters": {"type": "object", "properties": {}}}})
