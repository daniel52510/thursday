"""THURSDAY's self-improvement runner: one coding job, in a sandbox, ending on a branch for Daniel.

    docker compose run --rm -e JOB_ID=<id> runner "<task>"   (scripts/improve.sh does this)
    agent.py --worker   (the runner service: jobs THURSDAY queues in /out/queue, improvements.improve_start)

The repo arrives as its .git folder only, read-only (/repo.git): git never holds .env, brain/ or
any ignored file, so no secret or memory reaches the sandbox. The job clones it into /work, branches
thursday/improve-<id> off RUNNER_BASE, and a local model (RUNNER_MODEL, Ollama on the Mac) edits
with a few tools. Protected paths can't be touched. Checks (runner/checks.py) must pass before it
commits; the commits leave as a git bundle in /out with a JSON report. Nothing is merged, pushed or
deployed here: that's Daniel's call, after he reads the diff.

It shares the Mac's GPU with the voice and the missions, through files in /out (written by
missions.py): every model call waits until nobody has talked for QUIET_S (/out/activity) and is
dropped the moment someone does; a job starts only when no mission is running (/out/mission-busy),
and while one runs /out/runner-busy tells missions to wait. The time budget counts work, not waiting.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
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
WORK, SRC = Path("/work"), "/repo.git"
PROJECTS = Path("/projects")  # his GitHub Projects folder, read-only: other projects are reviewed, never changed
PROJECT: str | None = None    # this job's other project, or None for THURSDAY's own code
BASE_DEFAULT = BASE           # THURSDAY's branch; a project job uses the project's current branch instead
# In his other projects: secrets and git internals are never touched (THURSDAY's own list is PROTECTED).
PROJECT_PROTECTED = re.compile(r"(^|/)(\.env[^/]*|\.git(/|$)|[^/]*secret[^/]*|[^/]*\.(pem|key|p12)$|id_rsa[^/]*)", re.I)
FLAGS = Path("/out")                             # activity, busy flags, the queue: shared with core
OUT = Path(os.getenv("RUNNER_OUT", "/out"))      # reports and bundles (the benchmark keeps its own)
BRANCH = f"thursday/improve-{JOB}"
QUIET_S = 30         # like missions.QUIET_S: nobody has talked for this long
FRESH_S = 120        # a busy flag older than this is a crashed holder's, not a live one

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
    context = max(0, min(5, int(args.get("context") or 0)))  # lines around each hit
    res = subprocess.run(["git", "grep", "-n", "-I", "-E", *(["-C", str(context)] if context else []), str(args["pattern"]),
                          "--", _rel(_path(args.get("dir", ".")))], cwd=WORK, capture_output=True, text=True)
    hits = res.stdout.splitlines()
    return {"hits": [h[:200] for h in hits[:60]], "more": max(0, len(hits) - 60)}


def _writable(rel: str) -> Path:
    p = _path(rel)
    if (PROJECT_PROTECTED.search(_rel(p)) if PROJECT else PROTECTED.match(_rel(p))):
        raise PermissionError(f"protected: {_rel(p)} can't be changed by a job; say so in finish instead")
    return p


def edit_file(args: dict) -> dict:
    p = _writable(args["path"])
    text, old, new = p.read_text(), str(args["old"]), str(args["new"])
    if text.count(old) != 1:
        return {"ok": False, "error": f"old text found {text.count(old)} times; it must match exactly once (read the file again)"}
    p.write_text(text.replace(old, new))
    return {"ok": True, "path": _rel(p)}


def _node_span(tree: ast.AST, qualified: str) -> tuple[int, int, int] | None:
    """(first line incl. decorators, last line, indent column) of the def/class/constant named qualified."""
    def walk(body, prefix):
        for node in body:
            names = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = [node.target.id]
            for n in names:
                if f"{prefix}{n}" == qualified:
                    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
                    return first, node.end_lineno or node.lineno, node.col_offset
            if isinstance(node, ast.ClassDef) and (hit := walk(node.body, f"{prefix}{node.name}.")):
                return hit
        return None
    return walk(tree.body, "")


def replace_symbol(args: dict) -> dict:
    """Replace a whole definition by name: its current lines come from the syntax tree, so earlier edits
    never leave it pointing at the wrong place (benchmark t4: 9 line edits for 3 changes, 2026-11-05)."""
    from codeindex import CodeIndex
    hits = CodeIndex.build(WORK).find(str(args["name"]), limit=200)
    if args.get("path"):  # a named file narrows first: TOOL_SCHEMAS is in 5 files (benchmark t3, 2026-11-05)
        want = _rel(_path(args["path"]))
        hits = [h for h in hits if h["at"].split(":")[0] == want]
    exact = [h for h in hits if h["name"] == args["name"]] or [h for h in hits if h["name"].endswith("." + str(args["name"]))]
    exact = exact or [h for h in hits if h["name"].rsplit(".", 1)[-1] == args["name"]]
    if len(exact) != 1:
        return {"ok": False, "error": f"{len(exact)} definitions match; use the qualified name (Class.method) or the file",
                "matches": [f"{h['name']} at {h['at']}" for h in hits[:10]]}
    hit = exact[0]
    rel = hit["at"].split(":")[0]
    p = _writable(rel)
    text = p.read_text()
    span = _node_span(ast.parse(text), hit["name"])
    if span is None:
        return {"ok": False, "error": f"couldn't locate {hit['name']} in {rel}"}
    first, last, col = span
    new = textwrap.dedent(str(args["new"]).strip("\n"))
    new = "\n".join((" " * col + line) if line.strip() else "" for line in new.splitlines())
    lines = text.splitlines(keepends=True)
    out = "".join(lines[:first - 1]) + new + "\n" + "".join(lines[last:])
    try:
        ast.parse(out)
    except SyntaxError as exc:
        return {"ok": False, "error": f"not applied, the file wouldn't parse: {exc.msg} at line {exc.lineno}"}
    p.write_text(out)
    return {"ok": True, "path": rel, "replaced": f"{hit['name']} (was lines {first}-{last})"}


def edit_lines(args: dict) -> dict:
    """Replace lines start..end (inclusive, as read_file numbers them): for when old text won't match."""
    p = _writable(args["path"])
    lines = p.read_text().splitlines(keepends=True)
    start, end = int(args["start"]), int(args["end"])
    if not 1 <= start <= end <= len(lines):
        return {"ok": False, "error": f"lines {start}-{end} aren't in the file (it has {len(lines)})"}
    new = str(args["new"])
    lines[start - 1:end] = [new if new.endswith("\n") or not new else new + "\n"]
    p.write_text("".join(lines))
    after = p.read_text().splitlines()
    lo = max(0, start - 3)
    return {"ok": True, "path": _rel(p), "now": "\n".join(f"{lo + i + 1}\t{x}" for i, x in enumerate(after[lo:start + 3 + new.count("\n")]))}


def show_diff(_: dict | None = None) -> dict:
    return {"diff": _git("diff")[:6000] or "no changes yet"}


def undo_file(args: dict) -> dict:
    """Put a file back as it was (a new file is removed)."""
    p = _writable(args["path"])
    if _git("ls-files", "--", _rel(p)).strip():
        _git("checkout", "--", _rel(p))
    elif p.exists():
        p.unlink()
    return {"ok": True, "path": _rel(p)}


def run_tests(args: dict) -> dict:
    """Some of the tests, fast: pytest -k <match> (all without match)."""
    match = str(args.get("match") or "")
    res = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *(["-k", match] if match else []),
                          "tests"], cwd=WORK, capture_output=True, text=True, timeout=300)
    return {"passed": res.returncode in (0, 5), "output": (res.stdout + res.stderr)[-2500:]}


def create_file(args: dict) -> dict:
    p = _writable(args["path"])
    if p.exists():
        return {"ok": False, "error": "exists: use edit_file"}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(args["content"]))
    return {"ok": True, "path": _rel(p)}


def run_checks(_: dict | None = None) -> dict:
    if PROJECT:  # his other project: every tracked Python file compiles. Its own tests need its own dependencies.
        files = [f for f in _git("ls-files", "*.py").splitlines() if f]
        bad = []
        for rel in files:
            res = subprocess.run([sys.executable, "-m", "py_compile", rel], cwd=WORK, capture_output=True, text=True)
            if res.returncode:
                bad.append((res.stderr or res.stdout).strip()[-400:])
        out = f"compiled {len(files) - len(bad)}/{len(files)} Python files (the project's own tests aren't run here)"
        return {"passed": not bad, "output": out + ("\n" + "\n".join(bad) if bad else "")}
    res = subprocess.run([sys.executable, "/runner/checks.py", str(WORK)], capture_output=True, text=True, timeout=300)
    return {"passed": res.returncode == 0, "output": (res.stdout + res.stderr)[-3000:]}


def _index():
    # Rebuilt per call (~0.1 s for this repo): it always reflects the edits made so far.
    from codeindex import CodeIndex
    return CodeIndex.build(WORK)


def find_symbol(args: dict) -> dict:
    return {"definitions": _index().find(str(args["name"]))}


def who_calls(args: dict) -> dict:
    return {"callers": _index().callers(str(args["name"]))}


def outline(args: dict) -> dict:
    rel = _rel(_path(args["path"]))
    return {"outline": _index().outline(rel)[:150]}


TOOLS = {"project_map": lambda a: project_map(a), "replace_symbol": replace_symbol, "find_symbol": find_symbol, "who_calls": who_calls, "outline": outline, "list_files": list_files, "read_file": read_file, "search": search, "edit_file": edit_file,
         "edit_lines": edit_lines, "show_diff": show_diff, "undo_file": undo_file, "run_tests": run_tests,
         "create_file": create_file, "run_checks": run_checks}


def _schema(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


_S = {"type": "string"}
SCHEMAS = [
    _schema("project_map", "Every file with one line on what it does (optionally only lines containing match): "
            "where to look first.", {"match": _S}, []),
    _schema("find_symbol", "Where a function, class, method or constant (a regex, schema, table) is defined: file:lines, "
            "signature or value, what it does. Faster than search for a name.", {"name": {"type": "string", "description": "e.g. warm_cache or Session._warm"}}, ["name"]),
    _schema("who_calls", "Every function that calls this name, with the line of each call: what a change would affect.",
            {"name": _S}, ["name"]),
    _schema("outline", "A file's classes and functions with their line ranges and one-line docs, without reading it all.",
            {"path": _S}, ["path"]),
    _schema("list_files", "Tracked files under a folder.", {"dir": _S}, []),
    _schema("read_file", f"Read {READ_LINES} lines of a file, numbered, from start (default 1).",
            {"path": _S, "start": {"type": "integer"}}, ["path"]),
    _schema("search", "git grep -E across the repo (or a folder): find where something is.",
            {"pattern": _S, "dir": _S, "context": {"type": "integer", "description": "Lines around each hit, 0-5"}}, ["pattern"]),
    _schema("replace_symbol", "Replace a whole function, method, class or constant by name with new code (the full "
            "definition, decorators included). Best for changing a function or a regex/schema: no line numbers or exact "
            "matching needed, it finds the current lines itself and keeps the indentation.",
            {"name": {"type": "string", "description": "e.g. _music_intent, Session._warm, _MUSIC_PLAY"},
             "new": _S, "path": {"type": "string", "description": "optional: the file, to be sure"}}, ["name", "new"]),
    _schema("edit_lines", "Replace lines start..end (inclusive, numbered as read_file shows) with new text. Use it when "
            "edit_file's old text won't match; read the lines first.",
            {"path": _S, "start": {"type": "integer"}, "end": {"type": "integer"}, "new": _S}, ["path", "start", "end", "new"]),
    _schema("show_diff", "Your changes so far, as a diff.", {}, []),
    _schema("undo_file", "Put a file back as it was before your changes.", {"path": _S}, ["path"]),
    _schema("run_tests", "Run some tests fast (pytest -k match), or all of them without match.", {"match": _S}, []),
    _schema("edit_file", "Replace old with new in a file. old must match the file exactly once, whitespace included.",
            {"path": _S, "old": _S, "new": _S}, ["path", "old", "new"]),
    _schema("create_file", "Create a new file.", {"path": _S, "content": _S}, ["path", "content"]),
    _schema("run_checks", "Compile every Python file and run THURSDAY's quick tests.", {}, []),
    _schema("finish", "Done: what you changed and why, for Daniel's review. Checks must pass first.",
            {"summary": {"type": "string", "description": "2-4 sentences: what changed, why, anything he should test"},
             "title": {"type": "string", "description": "Commit title, under 60 characters"}}, ["summary", "title"]),
]


# Review mode (improvements.code_review_start): read the code, change nothing, report findings.
READ_ONLY = ("project_map", "find_symbol", "who_calls", "outline", "list_files", "read_file", "search")
# A task in these words is a bug fix: it must come with a test (in code, not just the prompt: the
# benchmark's t5 fix came back without one, 2026-11-01).
BUG_TASK = re.compile(r"\b(?:bug|fix|broken|breaks?|wrong|crash(?:es)?|error|fails?|instead|doesn'?t|isn'?t|won'?t)\b", re.I)
CRITIC = """You review one change to THURSDAY's code before it's committed. You see the task and the diff only.
Reply JSON only: {"ok": true} if the diff does exactly what the task asks, no more; or {"ok": false, "problems": ["..."]}
listing what's wrong: the task not (fully) done, a behavior it breaks, a bug it adds, changes the task didn't ask for,
a bug fix without a test that would have caught it. Only real problems: not where a line sits, naming, or how you'd
have written it. A correct, minimal change that does the task is ok. Be concrete."""
VERIFY = """You check findings from a code review against the code they quote. For each finding, decide whether the
problem is real given that code: the code must actually do what the finding says goes wrong, for a concrete input.
Drop anything that depends on a guess about what a library returns, or that the code already handles.
Reply JSON only: {"keep": [indexes of the findings that are real]}."""


_FILE_MAP = ""  # the brief's one line per file, served by project_map


def _brief() -> str:
    """The project brief (brief.py) for this checkout: its verified facts and overview go in the prompt,
    the one-line-per-file map stays behind project_map. All of it in every step (~3k tokens) made jobs
    slower: t2 12 -> 19 steps, t4 21 -> 26 (benchmark, 2026-11-05). The plain code map if it fails."""
    global _FILE_MAP
    try:
        from brief import refresh
        text = refresh(WORK, log=lambda m: print(m, flush=True))
        head, _, _FILE_MAP = text.partition("\n\nThe files:\n")
        # Only the verified facts: the model-written overview didn't make jobs faster either (t2 14 steps,
        # t4 26 vs 12 and 21 without it, 2026-11-05). It stays cached for answering questions about the code.
        facts = head.partition("\n\nHow it fits together:")[0]
        return facts + "\n\n(project_map lists every file with one line on what it does.)"
    except Exception as exc:  # Ollama busy or down: a job still runs on the plain map
        print(f"brief unavailable ({type(exc).__name__}): using the code map", flush=True)
        _FILE_MAP = _code_map()
        return "The code (file: what it does):\n" + _FILE_MAP


def project_map(args: dict) -> dict:
    match = str(args.get("match") or "").lower()
    rows = [r for r in _FILE_MAP.splitlines() if match in r.lower()]
    return {"files": rows[:80]}


def _code_map() -> str:
    """Each Python file and the first line of what it says it does: where to look first."""
    rows = []
    for p in sorted(WORK.glob("*.py")) + sorted(WORK.glob("satellite/*.py")) + sorted(WORK.glob("tests/*.py")):
        try:
            doc = (ast.get_docstring(ast.parse(p.read_text())) or "").strip().splitlines()
        except (SyntaxError, OSError):
            continue
        rows.append(f"- {_rel(p)}: {doc[0][:130] if doc else ''}")
    return "\n".join(rows)


def _json(text: str) -> dict:
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[start:end + 1]) if start >= 0 else {}
    except ValueError:
        return {}
REVIEW_SCHEMAS = [t for t in SCHEMAS if t["function"]["name"] in READ_ONLY] + [_schema(
    "report_review", "Done reading: your findings, most serious first (none is a fine answer).",
    {"summary": {"type": "string", "description": "2-3 sentences: what you looked at and the overall state"},
     "findings": {"type": "array", "items": {"type": "object", "properties": {
         "file": _S, "line": {"type": "integer"}, "severity": {"type": "string", "enum": ["high", "medium", "low"]},
         "problem": {"type": "string", "description": "What goes wrong, for whom, when: a concrete input or situation"},
         "quote": {"type": "string", "description": "The exact line(s) of code the problem is in, copied from read_file"},
         "fix": {"type": "string", "description": "The smallest change that would fix it"}},
         "required": ["file", "severity", "problem", "fix", "quote"]}}},
    ["summary", "findings"])]


def _other_map() -> str:
    """Another project's files (tracked ones in a git repo), for its review: no brief, it's not THURSDAY's."""
    res = subprocess.run(["git", "ls-files"], cwd=WORK, capture_output=True, text=True)
    files = res.stdout.splitlines() if res.returncode == 0 else [_rel(p) for p in WORK.rglob("*") if p.is_file()]
    files = [f for f in files if not any(x in f for x in ("node_modules/", ".venv/", "__pycache__/"))]
    return f"The project's files ({len(files)}):\n" + "\n".join(files[:200])


def _review_system() -> str:
    guide = (WORK / "CLAUDE.md").read_text()[:3000] if (WORK / "CLAUDE.md").exists() else ""
    if PROJECT:
        guide = guide or ((WORK / "README.md").read_text()[:3000] if (WORK / "README.md").exists() else "")
        what, focus, context = (f"the {PROJECT} project in Daniel's GitHub Projects folder", "the code the task names, "
                                "otherwise its main entry points", _other_map())
    else:
        what, focus, context = ("THURSDAY's own code", "the code that handles his requests (agent_stream.py, "
                                "voice_ws.py, the tools)", _brief())
    return f"""You are THURSDAY's engineer, reviewing {what} for real bugs. You can only read; nothing you do changes the code. Daniel reads your report and decides what to fix.

How to work:
- Focus on what the task names; otherwise on {focus}.
- Read the actual code (find_symbol / outline / search, then read_file around the hits; who_calls to follow a value to where it's used). Only report what you saw in the code: a file and line, and a concrete situation where it goes wrong.
- Real bugs over style: wrong behavior, crashes, races, a request that can silently do nothing, a claim made without the action, security holes, latency costs. Skip naming, formatting and taste.
- At most 6 findings, most serious first. Fewer good findings beat many weak ones; none is fine if you found none.
- Quote the exact code line(s) for each finding. Don't guess what a library returns: if a finding depends on that, leave it out.
- You have {MAX_STEPS} tool calls. Then call report_review once.

{context}

The project's guide (CLAUDE.md):
{guide}"""


def _project_system() -> str:
    guide = next(((WORK / n).read_text()[:5000] for n in ("CLAUDE.md", "README.md") if (WORK / n).exists()), "")
    return f"""You are THURSDAY's engineer, changing Daniel's {PROJECT} project (in his GitHub Projects folder) in a
sandbox. He reviews every change before it's merged into his project; nothing you do here touches it until then.

How to work:
- Start with a short plan (2-4 lines), then use the tools. Find the code first (list_files, search, outline), read
  before you edit, make the smallest change that does the task, and match the style of the code around it.
- Secrets (.env files, keys) and .git can't be changed. If the task needs them, say so in finish.
- run_checks only compiles Python; the project's own tests can't run here, so say in finish what he should test.
- You have {MAX_STEPS} tool calls and about {MAX_WORK_S // 60} minutes.

{_other_map()}

The project's guide:
{guide}"""


def _system() -> str:
    if PROJECT:
        return _project_system()
    guide = (WORK / "CLAUDE.md").read_text()[:5000] if (WORK / "CLAUDE.md").exists() else ""
    return f"""You are THURSDAY's engineer, improving THURSDAY's own code in a sandbox. Daniel reviews every change before it goes anywhere; nothing you do here is live.

How to work:
- Start with a short plan in your first reply's text (2-4 lines: which files, what change, how you'll check it), then use the tools.
- Find the code first: find_symbol for a name, outline for a file's layout, then read_file just those lines (search for text that isn't a name). Before changing a function, who_calls shows everything it affects. Read before you edit. To change a function or constant, replace_symbol with its whole new definition; edit_file for a small change inside one (exact text); edit_lines only as a last resort.
- Make the smallest change that does the task. Match the surrounding style, comments and naming. Don't refactor or touch unrelated code.
- Protected files (runner/, approvals.py, vault.py, docker-compose, Dockerfiles, .env, .gitignore, .dockerignore, infra/, deploy scripts) can't be changed. If the task needs one, say so in finish without changing it.
- A bug fix comes with a test in tests/ that fails without your fix (run_tests to see it pass). Match the tests already there.
- After your edits, show_diff to check them, then run_checks. If they fail, fix what you broke. Then call finish.
- If the task is unclear or can't be done safely, change nothing and call finish explaining why.
- You have {MAX_STEPS} tool calls and about {MAX_WORK_S // 60} minutes.

{_brief()}

The project's guide (CLAUDE.md):
{guide}"""


def _trim(messages: list[dict]) -> None:
    """Stay inside the context: shorten the oldest tool results first."""
    for m in messages:
        if sum(len(x.get("content") or "") for x in messages) <= NUM_CTX * 3:
            return
        if m["role"] == "tool" and len(m["content"]) > 500:
            m["content"] = m["content"][:500] + " [trimmed]"


def _age(name: str) -> float:
    """Seconds since /out/<name> was last touched (inf when it isn't there)."""
    try:
        return time.time() - (FLAGS / name).stat().st_mtime
    except OSError:
        return float("inf")


def _flag(name: str, on: bool = True) -> None:
    try:
        (FLAGS / name).touch() if on else (FLAGS / name).unlink(missing_ok=True)
    except OSError:
        pass


# Memory: on 2026-11-05 the 31b (~30 GB) loaded next to the 26b chat model pushed the Mac's swap to 99%,
# Ollama hung for everyone (THURSDAY's voice too) and a job waited 47 min on a dead call. Now a call
# waits until the Mac has room for the 31b (scripts/host_mem.sh writes /hostmem/host-mem.json; swap "used"
# is ignored: macOS keeps it allocated long after), and no call may run longer than CALL_LIMIT_S.
# The 26b chat model stays loaded on purpose (OLLAMA_KEEP_ALIVE=-1: voice never waits for a cold load), so
# 34 GB free never came and a job waited forever. The 31b ran fine today with 22-26 GB available; the hang came
# with memory pressure up. So: 20 GB available AND normal kernel pressure, plus the call limit as the backstop.
NEED_GB = float(os.getenv("RUNNER_NEED_GB", "20"))
CALL_LIMIT_S = 300
CALL_TRIES = 3


class CallTimeout(Exception):
    pass


def _model_loaded() -> bool:
    try:
        ps = httpx.get(CHAT_URL.rsplit("/api/", 1)[0] + "/api/ps", timeout=5).json()
        return any(m.get("name", "").startswith(MODEL) for m in ps.get("models", []))
    except (httpx.HTTPError, ValueError):
        return False


def _wait_for_memory() -> None:
    """Until the Mac has room for the model (skipped when it's already loaded, or the host file is stale)."""
    said = False
    while not _model_loaded():
        try:
            mem = json.loads(Path("/hostmem/host-mem.json").read_text())
        except (OSError, ValueError):
            return  # no helper running: the call limit still protects us
        if time.time() - mem.get("at", 0) > 60:
            return
        if mem.get("available_gb", 0) >= NEED_GB and mem.get("pressure", 1) == 1:
            return
        if not said:
            print(f"waiting for memory: {mem.get('available_gb')} GB free, need {NEED_GB:.0f}", flush=True)
            said = True
        time.sleep(15)


def _alarm(*_):
    raise CallTimeout(f"model call over {CALL_LIMIT_S} s")


def _chat(http: httpx.Client, messages: list[dict], tools: list[dict]) -> tuple[dict, float]:
    """_chat_once with a memory wait before it and a hard time limit on it, retried CALL_TRIES times."""
    for attempt in range(1, CALL_TRIES + 1):
        _wait_for_memory()
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(CALL_LIMIT_S)
        try:
            return _chat_once(http, messages, tools)
        except (CallTimeout, httpx.HTTPError) as exc:
            print(f"model call failed ({exc}); try {attempt} of {CALL_TRIES}", flush=True)
            if attempt == CALL_TRIES:
                raise
        finally:
            signal.alarm(0)
    raise CallTimeout("unreachable")


def _chat_once(http: httpx.Client, messages: list[dict], tools: list[dict]) -> tuple[dict, float]:
    """One model call: after a quiet spell, streamed, and dropped (then retried) the moment anyone
    talks, the way missions do. Returns the message and the seconds of actual work."""
    body = {"model": MODEL, "messages": messages, "tools": tools, "stream": True, "think": True,
            "keep_alive": "2m", "options": {"num_ctx": NUM_CTX}}
    while True:
        while _age("activity") < QUIET_S:
            time.sleep(1)
        _flag("runner-busy")
        started, content, calls, dropped = time.monotonic(), [], [], False
        with http.stream("POST", CHAT_URL, json=body) as res:
            res.raise_for_status()
            for line in res.iter_lines():
                if _age("activity") < time.monotonic() - started:  # someone talked since this call began
                    dropped = True
                    break
                if not line:
                    continue
                chunk = json.loads(line)
                msg = chunk.get("message") or {}
                content.append(msg.get("content") or "")
                calls.extend(msg.get("tool_calls") or [])
                if chunk.get("done"):
                    break
        if not dropped:
            return {"content": "".join(content), "tool_calls": calls}, time.monotonic() - started
        print("paused: someone's talking", flush=True)


def _report(**fields) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{JOB}.json").write_text(json.dumps({"job": JOB, "branch": BRANCH, "base": BASE, "model": MODEL,
                                                 "project": PROJECT or "thursday", **fields},
                                                indent=1))
    print(json.dumps(fields.get("status")), fields.get("summary", ""), sep="\n")


def run(task: str, kind: str = "change") -> None:
    global BASE
    BASE = BASE_DEFAULT
    if PROJECT:  # another project: work on a copy; a change comes back as a bundle he approves into the project
        src = PROJECTS / PROJECT
        if kind != "review" and not (src / ".git").exists():
            return _report(status="failed", error="changes need a git repo (approving merges a branch)", task=task,
                           project=PROJECT)
        if (src / ".git").exists():
            subprocess.run(["git", "clone", "-q", str(src), str(WORK)], check=True)
            BASE = _git("rev-parse", "--abbrev-ref", "HEAD").strip()  # his project's current branch
        else:
            shutil.copytree(src, WORK, ignore=shutil.ignore_patterns(".venv", "node_modules", "__pycache__", ".env*"))
            for cmd in (["init", "-q"], ["add", "-A"], ["-c", "user.email=r@t", "-c", "user.name=r", "commit", "-qm", "snapshot"]):
                _git(*cmd)
    else:
        subprocess.run(["git", "clone", "-q", "--branch", BASE, SRC, str(WORK)], check=True)
    _git("checkout", "-q", "-b", BRANCH)
    base_sha = _git("rev-parse", "HEAD").strip()
    review = kind == "review"
    schemas = REVIEW_SCHEMAS if review else SCHEMAS
    messages = [{"role": "system", "content": _review_system() if review else _system()},
                {"role": "user", "content": f"{'Review' if review else 'Task'}: {task}"}]
    steps, started, nudges, work, rounds, critiqued, tested_back = [], time.monotonic(), 0, 0.0, 0, False, False
    with httpx.Client(timeout=httpx.Timeout(600, connect=5)) as http:
        if EDIT_FIRST and not review:
            meta, work = _edit_first(http, task, steps)
            if meta is not None:
                return _finish(task, {"title": meta["title"] or task[:58], "summary": meta["summary"] or task},
                               steps, base_sha, started)
            _reset_work()
            messages.append({"role": "user", "content": "(A one-shot attempt didn't pass; the files are back as they "
                             "were. Work through it with the tools.)"})
        while True:
            over = len(steps) >= MAX_STEPS or work > MAX_WORK_S
            if over and nudges >= 2:
                return _report(status="failed", error="out_of_budget", task=task, steps=steps)
            if over:
                nudges += 1
                messages.append({"role": "user", "content": "Budget's spent: report_review now." if review else
                                 "Budget's spent: run_checks if you haven't, then finish."})
            msg, secs = _chat(http, messages, schemas)
            work += secs
            calls = msg.get("tool_calls") or []
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            if not calls:
                nudges += 1
                if nudges > 3:
                    return _report(status="failed", error="no_finish", task=task, steps=steps)
                messages.append({"role": "user", "content": "Keep going with the tools, or call "
                                 + ("report_review." if review else "finish.")})
                continue
            for call in calls:
                name, args = call["function"]["name"], call["function"].get("arguments") or {}
                if name == "finish" and not review:
                    # Our gate, not the model's word: checks first, then one review of the diff by a fresh pass.
                    # What fails goes back to it to fix (twice at most) instead of ending the job.
                    back = None
                    if _git("status", "--porcelain").strip() and rounds < 2:
                        checks = run_checks()
                        changed = [line[3:] for line in _git("status", "--porcelain").splitlines()]
                        if not checks["passed"]:
                            back = f"Checks failed:\n{checks['output'][-1500:]}\nFix it, then call finish again."
                        elif BUG_TASK.search(task) and not any(f.startswith("tests/") for f in changed) and not tested_back:
                            tested_back = True
                            back = ("This is a bug fix, so it needs a test: add one to the matching file in tests/ that fails "
                                    "without your fix and passes with it (run_tests to check), then call finish again.")
                        elif not critiqued:
                            critiqued = True
                            verdict, secs = _chat(http, [{"role": "system", "content": CRITIC}, {"role": "user", "content":
                                                  f"Task: {task}\n\nDiff:\n{_git('diff')[:8000]}"}], [])
                            work += secs
                            v = _json(verdict.get("content"))
                            if v.get("ok") is False and v.get("problems"):
                                back = ("A review of your diff found:\n- " + "\n- ".join(map(str, v["problems"][:5])) +
                                        "\nFix what's right about it (or explain in finish why not), then call finish again.")
                    if back:
                        rounds += 1
                        steps.append({"tool": "finish", "args": {"sent_back": back[:120]}, "ok": False})
                        print(f"finish sent back: {back[:120]}", flush=True)
                        messages.append({"role": "tool", "tool_name": "finish", "content": json.dumps({"ok": False, "error": back})})
                        continue
                    return _finish(task, args, steps, base_sha, started)
                if name == "report_review" and review:
                    findings = [f for f in (args.get("findings") or []) if isinstance(f, dict) and f.get("problem")][:6]
                    findings, secs = _verify_findings(http, findings)
                    work += secs
                    return _report(status="review_ready", kind="review", task=task,
                                   summary=str(args.get("summary") or ""),
                                   findings=findings, steps=steps, minutes=round((time.monotonic() - started) / 60, 1))
                if review and name not in READ_ONLY:
                    result = {"ok": False, "error": "review mode: read only. Call report_review when done."}
                    messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
                    continue
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


# --- edit-first: the runner gathers the code, the model writes the change in one reply --------------------
# The tool loop's weak spot with gemma4:31b is many steps of tool juggling (benchmark t4: 26-35 steps, line edits,
# double-escaped quotes, a junk file). Here the runner picks the definitions the task means and shows their exact
# source; the model answers once with whole new definitions in fenced code blocks (nothing to escape); the runner
# applies, checks and reviews them. What still fails falls back to the tool loop, so the pass rate can't drop.
EDIT_FIRST = os.getenv("RUNNER_EDIT_FIRST", "1") != "0"
EDIT_TRIES = 3
CONTEXT_CHARS = 16000
EDIT_SYSTEM = """You change THURSDAY's code. You get a task and the exact current source of the code it's about.
Answer with the change only, in these sections (any number of each; nothing else outside them):

### REPLACE <file>::<name>
```python
<the whole new definition of that function, method, class or constant, decorators included>
```

### ADD <file> AFTER <name>      (or: ### ADD <file> AT END)
```python
<a new definition to insert there>
```

### TEST <test file under tests/>
```python
<new test functions to append to that file, with any imports they need>
```

### TITLE <commit title, under 60 characters>
### SUMMARY <2-4 sentences for Daniel: what changed, why, anything to test>

Rules: the smallest change that does the task, matching the style around it; names exactly as shown
(Class.method for methods); never add files the task doesn't need. A bug fix comes with a TEST that fails
without the fix. Protected files (runner/, approvals.py, vault.py, docker-compose, Dockerfiles, .env, infra/,
deploy scripts) can't be changed: if the task needs one, change nothing and say so in SUMMARY."""
_SECTION = re.compile(r"^###\s+(REPLACE|ADD|TEST|TITLE|SUMMARY)\b[ \t]*(.*)$", re.M)
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def _task_context(task: str) -> str:
    """The definitions the task means (exact names first, then the best word matches), with their source,
    and the outline of any file it names."""
    from codeindex import CodeIndex
    idx = CodeIndex.build(WORK)
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", task))
    files = sorted({w for w in words if w.endswith(".py") and (WORK / w).exists()})
    terms = {w.lower() for w in re.findall(r"[a-z][a-z0-9]+", task.lower().replace("_", " ")) if len(w) > 3}

    def score(sym) -> float:
        if sym.name in words or sym.name.rsplit(".", 1)[-1] in words:
            return 100.0
        parts = set(re.findall(r"[a-z][a-z0-9]+", sym.name.lower().replace("_", " ")))
        bonus = 2 if sym.path in files else 0
        return 3 * len(terms & parts) + bonus if terms & parts else 0

    ranked = sorted((x for x in idx.symbols if not x.path.startswith(("tests/", "runner/")) and score(x) > 0),
                    key=score, reverse=True)
    # Constants in the same file that name a picked definition (TOOL_SCHEMAS for a tool function, a table of
    # handlers): a change to the function often needs them too. Benchmark t3 left the schema out: the model
    # never saw it (2026-11-05).
    top = ranked[:6]
    for sym in list(top):
        short = sym.name.rsplit(".", 1)[-1]
        for c in idx.symbols:
            if c.kind == "constant" and c.path == sym.path and c not in top and len(top) < 9:
                src = "\n".join((WORK / c.path).read_text().splitlines()[c.line - 1:c.end])
                if re.search(rf"\b{re.escape(short)}\b", src):
                    top.append(c)
    ranked = top + [x for x in ranked if x not in top]
    out, used = [], 0
    for name in files:
        outline = "\n".join(idx.outline(name)[:60])
        out.append(f"Outline of {name}:\n{outline}")
        used += len(outline)
    for sym in ranked[:9]:
        src = "\n".join((WORK / sym.path).read_text().splitlines()[sym.line - 1:sym.end][:150])
        if used + len(src) > CONTEXT_CHARS:
            break
        out.append(f"{sym.path}::{sym.name} (lines {sym.line}-{sym.end}):\n```python\n{src}\n```")
        used += len(src)
    return "\n\n".join(out)


def _apply_sections(reply: str) -> tuple[dict, list[str]]:
    """Apply the model's sections; returns ({title, summary}, errors)."""
    meta, errors = {"title": "", "summary": ""}, []
    marks = list(_SECTION.finditer(reply))
    for i, m in enumerate(marks):
        kind, head = m.group(1), m.group(2).strip()
        body = reply[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(reply)]
        if kind in ("TITLE", "SUMMARY"):
            meta[kind.lower()] = (head + " " + body.strip()).strip()[:600]
            continue
        code = _FENCE.search(body)
        if not code:
            errors.append(f"{kind} {head}: no ```python block")
            continue
        code = code.group(1).rstrip() + "\n"
        try:
            if kind == "REPLACE":
                path, _, name = head.partition("::")
                res = replace_symbol({"name": name.strip(), "path": path.strip(), "new": code})
                if not res.get("ok"):
                    errors.append(f"REPLACE {head}: {res.get('error')}")
            elif kind == "ADD":
                at = re.match(r"(\S+)\s+(?:AFTER\s+(\S+)|AT\s+END)", head)
                if not at:
                    errors.append(f"ADD {head}: write '<file> AFTER <name>' or '<file> AT END'")
                    continue
                p = _writable(at.group(1))
                text = p.read_text() if p.exists() else ""
                if at.group(2):
                    span = _node_span(ast.parse(text), at.group(2))
                    if span is None:
                        errors.append(f"ADD {head}: no {at.group(2)} in {at.group(1)}")
                        continue
                    lines = text.splitlines(keepends=True)
                    block = "\n".join((" " * span[2] + ln) if ln.strip() else "" for ln in textwrap.dedent(code).splitlines())
                    out = "".join(lines[:span[1]]) + "\n\n" + block + "\n" + "".join(lines[span[1]:])
                else:
                    out = text.rstrip("\n") + "\n\n\n" + code
                ast.parse(out)
                p.write_text(out)
            elif kind == "TEST":
                p = _writable(head)
                if not _rel(p).startswith("tests/"):
                    errors.append(f"TEST {head}: tests go under tests/")
                    continue
                text = p.read_text() if p.exists() else ""
                out = (text.rstrip("\n") + "\n\n\n" if text else "") + code
                ast.parse(out)
                p.write_text(out)
        except (SyntaxError, PermissionError, OSError, ValueError) as exc:
            errors.append(f"{kind} {head}: {type(exc).__name__}: {exc}"[:300])
    if not marks:
        errors.append("no sections: answer with ### REPLACE / ADD / TEST / TITLE / SUMMARY")
    return meta, errors


def _reset_work() -> None:
    _git("checkout", "-q", "--", ".")
    _git("clean", "-qfd")


def _edit_first(http: httpx.Client, task: str, steps: list) -> tuple[dict | None, float]:
    """Up to EDIT_TRIES single-reply attempts; ({title, summary}, work seconds) when one passes every gate."""
    context, work = _task_context(task), 0.0
    system = EDIT_SYSTEM if not PROJECT else EDIT_SYSTEM.replace("You change THURSDAY's code.", f"You change Daniel's {PROJECT} project.").replace(
        "Protected files (runner/, approvals.py, vault.py, docker-compose, Dockerfiles, .env, infra/,\ndeploy scripts) can't be changed",
        "Secrets (.env files, keys) and .git can't be changed")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"Task: {task}\n\nThe code it's about:\n\n{context}"}]
    critiqued = 0  # the critic sees every attempt after it sends one back (t3's second try went unchecked)
    for attempt in range(1, EDIT_TRIES + 1):
        msg, secs = _chat(http, messages, [])
        work += secs
        reply = msg.get("content") or ""
        meta, errors = _apply_sections(reply)
        changed = [line[3:] for line in _git("status", "--porcelain").splitlines()]
        back = None
        if errors:
            back = "These didn't apply:\n- " + "\n- ".join(errors[:6])
        elif not changed:
            back = "Nothing changed: give the REPLACE/ADD sections that do the task."
        else:
            checks = run_checks()
            if not checks["passed"]:
                back = f"Checks failed:\n{checks['output'][-1500:]}"
            elif BUG_TASK.search(task) and not any(c.startswith("tests/") for c in changed):
                back = "This is a bug fix: add a TEST section that fails without the fix."
            elif critiqued < EDIT_TRIES:
                critiqued += 1
                verdict, secs = _chat(http, [{"role": "system", "content": CRITIC}, {"role": "user", "content":
                                      f"Task: {task}\n\nDiff:\n{_git('diff')[:8000]}"}], [])
                work += secs
                v = _json(verdict.get("content"))
                if v.get("ok") is False and v.get("problems"):
                    back = "A review of your diff found:\n- " + "\n- ".join(map(str, v["problems"][:5]))
        steps.append({"tool": "edit_first", "args": {"attempt": attempt, "back": (back or "passed")[:120]}, "ok": back is None})
        print(f"edit-first {attempt}: {(back or 'passed').splitlines()[0][:120]}", flush=True)
        if back is None:
            return meta, work
        _reset_work()
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": back + "\n\nThe files are back as they were. Answer again with the "
                      "complete corrected sections (all of them, not just the fixes)."}]
    return None, work


def _quoted(f: dict) -> str | None:
    """The code around a finding's quote, or None when the quote isn't in that file (a misread)."""
    try:
        text = _path(str(f.get("file") or "")).read_text()
    except (OSError, ValueError):
        return None
    squash = lambda t: " ".join(t.split())
    quote = squash(str(f.get("quote") or ""))
    lines = text.splitlines()
    for i in range(len(lines)):
        if quote and quote[:80] in squash(" ".join(lines[i:i + 6])):
            return "\n".join(f"{n + 1}\t{x}" for n, x in enumerate(lines[max(0, i - 8):i + 14], start=max(0, i - 8)))
    return None


def _verify_findings(http: httpx.Client, findings: list[dict]) -> tuple[list[dict], float]:
    """Keep only findings whose quote is really in the file and that a second look agrees are real."""
    quoted = [(f, code) for f in findings if (code := _quoted(f))]
    if len(quoted) < len(findings):
        print(f"review: dropped {len(findings) - len(quoted)} finding(s) whose quote isn't in the file", flush=True)
    if not quoted:
        return [], 0.0
    shown = "\n\n".join(f"[{i}] {f.get('file')}: {f.get('problem')}\nCode:\n{code}" for i, (f, code) in enumerate(quoted))
    verdict, secs = _chat(http, [{"role": "system", "content": VERIFY}, {"role": "user", "content": shown[:12000]}], [])
    keep = _json(verdict.get("content")).get("keep")
    if not isinstance(keep, list):
        return [f for f, _ in quoted], secs  # no verdict: keep what's quoted
    kept = [f for i, (f, _) in enumerate(quoted) if i in keep]
    print(f"review: second look kept {len(kept)} of {len(quoted)}", flush=True)
    return kept, secs


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


def _fresh_work() -> None:
    """An empty /work for the next clone."""
    for p in WORK.iterdir():
        shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()


def worker() -> None:
    """The runner service: one queued job at a time, oldest first, never alongside a mission."""
    global JOB, BRANCH, PROJECT
    queue = FLAGS / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    print(f"runner worker: {MODEL}, base {BASE}", flush=True)
    while True:
        jobs = sorted(queue.glob("*.json"))
        if not jobs or _age("mission-busy") < FRESH_S:
            time.sleep(3)
            continue
        path, job = jobs[0], {}
        try:
            job = json.loads(path.read_text())
            JOB, BRANCH = job["job"], f"thursday/improve-{job['job']}"
            PROJECT = job.get("project") if job.get("project") not in (None, "", "thursday") else None
            path.unlink()
            _fresh_work()
            print(f"job {JOB}: {job['task'][:120]}", flush=True)
            run(job["task"], job.get("kind", "change"))
        except Exception as exc:  # one bad job never stops the worker
            _report(status="failed", error=f"{type(exc).__name__}: {exc}"[:300], task=str(job.get("task", "")))
        finally:
            _flag("runner-busy", False)


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        worker()
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        sys.exit("usage: agent.py '<task>' | agent.py --worker")
    run(sys.argv[1].strip())
