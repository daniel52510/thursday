"""A map of a Python codebase from its syntax tree: every function, class and method (where it is,
its signature, the first line of its docstring), what each one calls, and what each file imports.

Built in memory in about a second for a repo this size, so a job always sees the code it checked out.
No dependencies beyond the standard library: the runner uses it on THURSDAY's own code, and the same
index works on any folder of Python (the GitHub Projects review, later).

    idx = CodeIndex.build(Path("/work"))
    idx.find("warm_cache")       # definitions named (or ending in) warm_cache
    idx.callers("warm_cache")    # who calls it, with the line of each call
    idx.outline("voice_ws.py")   # the file's classes and functions, in order
"""
from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "build", "dist", ".mypy_cache", ".pytest_cache"}
MAX_FILE_BYTES = 1_000_000  # generated or vendored files: not worth parsing


@dataclass
class Symbol:
    name: str          # qualified within its file: "Session._warm", "warm_cache"
    kind: str          # "function", "class", "method"
    path: str          # relative to the root
    line: int
    end: int
    signature: str
    doc: str           # first line of the docstring
    calls: list[tuple[str, int]] = field(default_factory=list)  # (called name, line)


def _signature(node: ast.AST) -> str:
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix} {node.name}({ast.unparse(node.args)}){ret}"


def _called(node: ast.Call) -> str | None:
    """The name a call uses: foo(), self.foo(), mod.foo() -> "foo"."""
    f = node.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


class _Walker(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path, self.stack, self.symbols, self.imports = path, [], [], []

    def _def(self, node, kind: str) -> None:
        parent = self.stack[-1] if self.stack else None
        if kind == "function" and parent is not None and parent.kind == "class":
            kind = "method"
        name = f"{parent.name}.{node.name}" if parent is not None else node.name
        doc = (ast.get_docstring(node) or "").strip().splitlines()
        sym = Symbol(name, kind, self.path, node.lineno, node.end_lineno or node.lineno, _signature(node),
                     doc[0][:160] if doc else "")
        self.symbols.append(sym)
        self.stack.append(sym)
        self.generic_visit(node)
        self.stack.pop()

    def visit_ClassDef(self, node):
        self._def(node, "class")

    def visit_FunctionDef(self, node):
        self._def(node, "function")

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node):
        # Module and class constants (regexes, schemas, tables): a task often means changing one, and
        # without them a job paged through files to find _MUSIC_PLAY (benchmark t4, 2026-11-05).
        parent = self.stack[-1] if self.stack else None
        if parent is None or parent.kind == "class":
            for t in node.targets:
                if isinstance(t, ast.Name):
                    name = f"{parent.name}.{t.id}" if parent is not None else t.id
                    value = ast.unparse(node.value).replace("\n", " ")
                    self.symbols.append(Symbol(name, "constant", self.path, node.lineno, node.end_lineno or node.lineno,
                                               f"{t.id} = {value[:80]}{'...' if len(value) > 80 else ''}", ""))
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self.visit_Assign(ast.Assign(targets=[node.target], value=node.value, lineno=node.lineno,
                                         end_lineno=node.end_lineno))
        else:
            self.generic_visit(node)

    def visit_Call(self, node):
        if self.stack and (name := _called(node)):
            self.stack[-1].calls.append((name, node.lineno))
        self.generic_visit(node)

    def visit_Import(self, node):
        self.imports += [a.name for a in node.names]

    def visit_ImportFrom(self, node):
        self.imports += [f"{node.module or ''}.{a.name}".lstrip(".") for a in node.names]


def _python_files(root: Path) -> list[Path]:
    """Tracked .py files in a git repo (never virtualenvs or build output); otherwise a walk that skips them."""
    if (root / ".git").exists():
        res = subprocess.run(["git", "ls-files", "-z", "--", "*.py"], cwd=root, capture_output=True, text=True)
        if res.returncode == 0:
            return [root / f for f in sorted(res.stdout.split("\0")) if f and (root / f).is_file()]
    out = []
    for p in sorted(root.rglob("*.py")):
        parts = p.relative_to(root).parts[:-1]
        if any(x in SKIP_DIRS or x.startswith(".") or x == "site-packages" for x in parts):
            continue
        if any((root.joinpath(*parts[:i]) / "pyvenv.cfg").exists() for i in range(1, len(parts) + 1)):
            continue  # inside a virtualenv
        out.append(p)
    return out


class CodeIndex:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.symbols: list[Symbol] = []
        self.imports: dict[str, list[str]] = {}
        self.errors: dict[str, str] = {}  # files that didn't parse, and why

    @classmethod
    def build(cls, root: Path) -> "CodeIndex":
        idx = cls(root)
        for p in _python_files(root):
            rel = p.relative_to(root)
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
                tree = ast.parse(p.read_text(errors="replace"), filename=str(rel))
                w = _Walker(str(rel))
                if doc := ast.get_docstring(tree):  # the module's own description: what the file is for
                    first = tree.body[0]
                    w.symbols.append(Symbol(rel.stem, "module", str(rel), 1, first.end_lineno or 1, f"module {rel}",
                                            doc.strip().splitlines()[0][:160]))
                w.visit(tree)
            except (SyntaxError, ValueError, OSError, RecursionError) as exc:  # Recursion: huge generated literals
                idx.errors[str(rel)] = f"{type(exc).__name__}: {str(exc)[:150]}"
                continue
            idx.symbols += w.symbols
            idx.imports[str(rel)] = sorted(set(w.imports))
        return idx

    def find(self, name: str, limit: int = 20) -> list[dict]:
        """Definitions whose qualified name is `name`, ends in ".name", or (failing those) contains it."""
        key = name.strip().lower()
        exact = [s for s in self.symbols if s.name.lower() == key or s.name.lower().endswith("." + key)]
        hits = exact or [s for s in self.symbols if key in s.name.lower()]
        return [{"name": s.name, "kind": s.kind, "at": f"{s.path}:{s.line}-{s.end}", "signature": s.signature,
                 "doc": s.doc} for s in hits[:limit]]

    def callers(self, name: str, limit: int = 40) -> list[dict]:
        """Every definition that calls `name` (by its last part: self.x(), mod.x() and x() all count)."""
        short = name.strip().rsplit(".", 1)[-1]
        out = []
        for s in self.symbols:
            for called, line in s.calls:
                if called == short:
                    out.append({"caller": s.name, "at": f"{s.path}:{line}"})
        return out[:limit]

    def outline(self, path: str) -> list[str]:
        """The file's definitions in order, indented by nesting: "  12-40 def foo(x) - what it does"."""
        rows = []
        for s in self.symbols:
            if s.path == path and s.kind != "module":
                depth = s.name.count(".")
                rows.append(f"{'  ' * depth}{s.line}-{s.end} {s.signature}" + (f"  # {s.doc}" if s.doc else ""))
        return rows

    def stats(self) -> dict:
        return {"files": len(self.imports), "symbols": len(self.symbols),
                "calls": sum(len(s.calls) for s in self.symbols), "unparsed": len(self.errors)}
