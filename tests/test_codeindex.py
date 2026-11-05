"""The runner's code index (runner/codeindex.py): definitions, callers and outlines from the syntax tree."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runner"))
from codeindex import CodeIndex  # noqa: E402

SRC = '''
def helper(x):
    """Doubles x."""
    return x * 2


class Box:
    """Holds a value."""

    def grow(self):
        return helper(self.v)

    async def fetch(self):
        self.grow()
'''


def _index(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(SRC)
    (tmp_path / "broken.py").write_text("def nope(:\n")
    venv = tmp_path / "env" / "lib"
    venv.mkdir(parents=True)
    (tmp_path / "env" / "pyvenv.cfg").write_text("")
    (venv / "vendored.py").write_text("def vendored(): pass\n")
    return CodeIndex.build(tmp_path)


def test_finds_definitions_by_short_or_qualified_name(tmp_path):
    idx = _index(tmp_path)
    assert idx.find("grow")[0]["at"] == "pkg/mod.py:10-11"
    assert idx.find("Box.fetch")[0]["signature"] == "async def fetch(self)"
    assert idx.find("helper")[0]["doc"] == "Doubles x."


def test_callers_follow_plain_and_method_calls(tmp_path):
    idx = _index(tmp_path)
    assert idx.callers("helper") == [{"caller": "Box.grow", "at": "pkg/mod.py:11"}]
    assert idx.callers("grow") == [{"caller": "Box.fetch", "at": "pkg/mod.py:14"}]


def test_outline_nests_methods_and_skips_virtualenvs_and_bad_files(tmp_path):
    idx = _index(tmp_path)
    assert [r.strip().split(" ", 1)[1].split("  #")[0] for r in idx.outline("pkg/mod.py")] == [
        "def helper(x)", "class Box", "def grow(self)", "async def fetch(self)"]
    assert not idx.find("vendored")
    assert "broken.py" in idx.errors
