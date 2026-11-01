#!/usr/bin/env python3
"""Scores the self-improvement runner on runner/bench/tasks.py: one job per task, in the sandbox, then
each task's behavior check on the job's result (in thursday-core, which has the dependencies).

    python3 scripts/bench_runner.py [task-id ...]     # all tasks by default

Results print as a table and land in runner-out/bench/results-<time>.json. Nothing touches the
repo's branches, the review queue or his phone: reports go to runner-out/bench/, and each job's
result is checked in a throwaway clone. Each check must fail on the unchanged code first, or its
task is reported as a broken check. Uses the GPU like any job (quiet-gated), ~1-5 min a task.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "runner-out" / "bench"
sys.path.insert(0, str(REPO / "runner" / "bench"))
from tasks import TASKS  # noqa: E402


def sh(*args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


def check_in_core(tree: Path, code: str) -> tuple[bool, str]:
    """Run a task's check against `tree` inside thursday-core; (passed, why)."""
    dest = f"/tmp/bench-{tree.name}"
    sh("docker", "exec", "thursday-core", "rm", "-rf", dest)
    if sh("docker", "cp", f"{tree}/.", f"thursday-core:{dest}").returncode:
        return False, "copy failed"
    script = f"import sys, os\nsys.path.insert(0, {dest!r})\nos.chdir({dest!r})\n{code}\nprint('CHECK OK')"
    res = sh("docker", "exec", "-i", "-w", dest, "thursday-core", "python", "-", input=script, timeout=120)
    sh("docker", "exec", "thursday-core", "rm", "-rf", dest)
    out = (res.stdout + res.stderr).strip()
    return "CHECK OK" in res.stdout, out.splitlines()[-1][:200] if out else ""


def clone(ref: str, bundle: Path | None = None) -> Path:
    tree = Path(tempfile.mkdtemp(prefix="bench-", dir="/private/tmp"))
    sh("git", "clone", "-q", "--branch", "container-branch", str(REPO), str(tree), check=True)
    if bundle:
        sh("git", "-C", str(tree), "fetch", "-q", str(bundle), f"{ref}:{ref}", check=True)
        sh("git", "-C", str(tree), "checkout", "-q", ref, check=True)
    return tree


def main() -> None:
    wanted = set(sys.argv[1:])
    tasks = [t for t in TASKS if not wanted or t["id"] in wanted]
    BENCH.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rows = []
    base = clone("container-branch")
    for t in tasks:
        ok, _ = check_in_core(base, t["check"])
        if ok:
            rows.append({"id": t["id"], "result": "BROKEN CHECK (passes on unchanged code)"})
            continue
        job = f"bench-{t['id']}-{stamp}"
        started = time.monotonic()
        sh("docker", "compose", "run", "--rm", "-e", f"JOB_ID={job}", "-e", "RUNNER_OUT=/out/bench", "runner", t["task"],
           cwd=REPO, timeout=3600)
        wall = round((time.monotonic() - started) / 60, 1)
        report = json.loads((BENCH / f"{job}.json").read_text()) if (BENCH / f"{job}.json").exists() else {"status": "crashed"}
        row = {"id": t["id"], "level": t["level"], "status": report.get("status"), "steps": len(report.get("steps", [])),
               "work_min": report.get("minutes"), "wall_min": wall, "files": report.get("files")}
        if report.get("status") == "ready":
            tree = clone(f"thursday/improve-{job}", BENCH / f"{job}.bundle")
            row["passed"], row["why"] = check_in_core(tree, t["check"])
            row["lines"] = sh("git", "-C", str(tree), "diff", "--shortstat", "container-branch...HEAD").stdout.strip()
            shutil.rmtree(tree, ignore_errors=True)
        else:
            row["passed"], row["why"] = False, report.get("error") or report.get("status")
        rows.append(row)
        print(json.dumps(row), flush=True)
    shutil.rmtree(base, ignore_errors=True)
    (BENCH / f"results-{stamp}.json").write_text(json.dumps(rows, indent=1))
    passed = sum(bool(r.get("passed")) for r in rows)
    print(f"\n{passed}/{len(rows)} passed  (results: runner-out/bench/results-{stamp}.json)")
    for r in rows:
        print(f"  {r['id']:13s} {'PASS' if r.get('passed') else 'FAIL'}  {r.get('status', ''):14s} steps={r.get('steps', '-')!s:3s} "
              f"work={r.get('work_min', '-')}m  {r.get('lines', '')}  {'' if r.get('passed') else r.get('why', '')}")


if __name__ == "__main__":
    main()
