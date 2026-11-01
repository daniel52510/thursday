"""The runner's gate (runner/agent.py): a job's change commits only if this exits 0.

    python checks.py /work

1. Every tracked Python file compiles.
2. Rocky's grammar (eridian.py, pure stdlib) still does what Daniel approved: questions end
   ", question?", exclamations ", exclamation!", statements stay bare, a confirmation turn asks
   nothing. Add a case here whenever he approves a new rule.
3. The test suite (tests/, pytest): the logic that has broken before. The image has core's pinned
   dependencies but no network, Ollama or devices; the tests need none of them.
"""
from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path

ROCKY = [  # (reply, statements, expected)
    ("Do you want a jacket?", False, "You want jacket, question?"),
    ("It is sunny, question. You want jacket?", False, "It is sunny. You want jacket, question?"),
    ("Launch is tonight! Do you want to watch?", False, "Launch is tonight, exclamation! You want to watch, question?"),
    ("We plan, statement.", False, "We plan."),
    ("Good, good, good! Engine works!", False, "Good, good, good! Engine works, exclamation!"),
    ("Morning! Amaze! Amaze! Amaze!", False, "Morning! Amaze! Amaze! Amaze!"),
    ("See https://x.com/a!b now.", False, "See https://x.com/a!b now."),
    ("I find best collar, question?", True, "I find best collar."),
    ("Thanks.", False, "Gracious."),
]


def main(root: Path) -> int:
    failed = 0
    files = subprocess.run(["git", "ls-files", "*.py"], cwd=root, capture_output=True, text=True).stdout.split()
    for f in files:
        try:
            py_compile.compile(str(root / f), doraise=True)
        except py_compile.PyCompileError as exc:
            failed += 1
            print(f"COMPILE FAIL {f}: {exc.msg.strip()[:300]}")
    print(f"compiled {len(files) - failed}/{len(files)} Python files")

    sys.path.insert(0, str(root))
    try:
        import eridian
    except Exception as exc:
        print(f"FAIL importing eridian: {type(exc).__name__}: {exc}")
        return 1
    for text, statements, want in ROCKY:
        got = eridian.translate(text, statements)
        stream = eridian.Stream()
        stream.statements = statements
        streamed = "".join(stream.feed(ch) for ch in text) + stream.flush()  # token by token, as Ollama sends it
        if got != want or streamed != want:
            failed += 1
            print(f"ROCKY FAIL {text!r}: want {want!r}, got {got!r}" + (f", streamed {streamed!r}" if streamed != got else ""))
    print(f"rocky grammar: {len(ROCKY)} cases, {'all pass' if not failed else 'see failures above'}")
    if (root / "tests").is_dir():
        res = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=root,
                             capture_output=True, text=True, timeout=600)
        tail = (res.stdout + res.stderr).strip().splitlines()
        print("tests: " + (tail[-1] if tail else "no output"))
        if res.returncode:
            failed += 1
            print("\n".join(line for line in tail if line.startswith(("FAILED", "ERROR", "E ")))[-2500:])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else ".")))
