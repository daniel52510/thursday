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
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(os.getenv("REPO_DIR", "/repo"))
OUT = REPO / "runner-out"
GIT_ID = ["-c", "user.name=THURSDAY (approved by Daniel)", "-c", "user.email=runner@thursday.local",
          "-c", "safe.directory=*"]
# Same list as the runner's (runner/agent.py PROTECTED), checked again here on the real diff.
PROTECTED = re.compile(r"^(runner/|approvals\.py$|vault\.py$|improvements\.py$|docker-compose\.ya?ml$|Dockerfile|\.env|"
                       r"\.git|\.gitignore$|\.dockerignore$|scripts/improve\.sh$|scripts/deploy_sat\.sh$|brain/|infra/)")
_ID = re.compile(r"^\d{8}-\d{6}$")


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
