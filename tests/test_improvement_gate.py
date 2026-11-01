"""The approval gate for self-improvement jobs (improvements.py), on a throwaway git repo."""
import json
import subprocess

import pytest

import improvements as g


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repo on container-branch, plus a job branch that changes one file, bundled like the runner does."""
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "container-branch")
    git(r, "config", "user.email", "t@t")
    git(r, "config", "user.name", "t")
    (r / "eridian.py").write_text("WORDS = ('thanks',)\n")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "base")
    git(r, "checkout", "-qb", "thursday/improve-20260101-000001")
    (r / "eridian.py").write_text("WORDS = ('thanks', 'cheers')\n")
    git(r, "commit", "-qam", "job")
    (r / "runner-out").mkdir()
    git(r, "bundle", "create", str(r / "runner-out" / "20260101-000001.bundle"), "container-branch..thursday/improve-20260101-000001")
    git(r, "checkout", "-q", "container-branch")
    git(r, "branch", "-D", "thursday/improve-20260101-000001")
    (r / "runner-out" / "20260101-000001.json").write_text(json.dumps({
        "job": "20260101-000001", "status": "ready", "base": "container-branch",
        "branch": "thursday/improve-20260101-000001", "files": ["eridian.py"]}))
    monkeypatch.setattr(g, "REPO", r)
    monkeypatch.setattr(g, "OUT", r / "runner-out")
    return r


def test_review_rebuilds_the_diff_from_the_bundle(repo):
    rv = g.review("20260101-000001")
    assert rv["files"] == ["eridian.py"] and rv["protected"] == [] and "cheers" in rv["diff"]


def test_uncommitted_work_blocks_the_merge(repo):
    (repo / "eridian.py").write_text("WORDS = ('mine',)\n")
    with pytest.raises(g.GateError, match="uncommitted"):
        g.approve("20260101-000001")


def test_approve_merges_once(repo):
    assert g.approve("20260101-000001")["decision"] == "approved"
    assert "cheers" in (repo / "eridian.py").read_text()
    with pytest.raises(g.GateError):
        g.approve("20260101-000001")


def test_decline_deletes_the_branch(repo):
    g.review("20260101-000001")  # fetches the branch
    assert g.decline("20260101-000001")["decision"] == "declined"
    assert git(repo, "branch", "--list", "thursday/improve-20260101-000001").strip() == ""


@pytest.mark.parametrize("job", ["../../etc/passwd", "20260101", "x" * 15])
def test_bad_job_ids_refused(repo, job):
    with pytest.raises(g.GateError):
        g.review(job)


def test_protected_files_refused(repo):
    git(repo, "checkout", "-qb", "thursday/improve-20260101-000002")
    (repo / "approvals.py").write_text("# sneaky\n")
    git(repo, "add", "approvals.py")
    git(repo, "commit", "-qm", "touch the gate")
    git(repo, "checkout", "-q", "container-branch")
    (repo / "runner-out" / "20260101-000002.json").write_text(json.dumps({
        "job": "20260101-000002", "status": "ready", "base": "container-branch",
        "branch": "thursday/improve-20260101-000002"}))
    with pytest.raises(g.GateError, match="protected"):
        g.approve("20260101-000002")
