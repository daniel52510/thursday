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
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
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
    queue = OUT / "queue"
    if len(list(queue.glob("*.json"))) >= MAX_QUEUED:
        return ToolResult(ok=False, tool_name="improve_start", error=f"already_{MAX_QUEUED}_jobs_queued")
    if not args.get("confirmed"):
        return missions._read_back(turn, "improve_start", {"task": task, "confirmed": True}, task)
    queue.mkdir(parents=True, exist_ok=True)
    job = datetime.now().strftime("%Y%m%d-%H%M%S")
    while (queue / f"{job}.json").exists() or (OUT / f"{job}.json").exists():
        time.sleep(1)
        job = datetime.now().strftime("%Y%m%d-%H%M%S")
    (queue / f"{job}.json").write_text(json.dumps({"job": job, "task": task, "queued_at": datetime.now(timezone.utc).isoformat()}))
    logger.info("improvement %s queued: %s", job, task[:120])
    result = ToolResult(ok=True, tool_name="improve_start", data={
        "job": job, "status": "STARTED",
        "note": "Confirm in one short statement, no question: you work on it when the house is quiet and his phone "
                "gets the change to review. Nothing changes until he approves it."})
    missions._started[(turn.get("conversation") or "", "improve_start")] = (result, time.monotonic())
    return result


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
    while True:
        try:
            for path in sorted(OUT.glob("*.json")) if OUT.exists() else []:
                report = json.loads(path.read_text())
                if report.get("notified"):
                    continue
                if path.stat().st_mtime >= started:
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
