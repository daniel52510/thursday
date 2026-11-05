"""The runner's replace_symbol tool: a whole definition swapped by name, indentation kept, never a broken file."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runner"))
pytest.importorskip("httpx")
import agent  # noqa: E402

SRC = "class A:\n    def go(self):\n        return 1\n\n\nclass B:\n    @staticmethod\n    def go():\n        return 2\n"


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "WORK", tmp_path)
    (tmp_path / "m.py").write_text(SRC)
    return tmp_path


def test_replaces_a_decorated_method_at_its_indentation(work):
    assert agent.replace_symbol({"name": "B.go", "new": "@staticmethod\ndef go():\n    return 20"})["ok"]
    assert (work / "m.py").read_text().endswith("class B:\n    @staticmethod\n    def go():\n        return 20\n")


def test_ambiguous_name_lists_matches_and_changes_nothing(work):
    res = agent.replace_symbol({"name": "go", "new": "def go(): pass"})
    assert not res["ok"] and len(res["matches"]) == 2
    assert (work / "m.py").read_text() == SRC


def test_code_that_would_not_parse_is_not_applied(work):
    assert not agent.replace_symbol({"name": "A.go", "new": "def go(self) return"})["ok"]
    assert (work / "m.py").read_text() == SRC


def test_edit_first_sections_apply_and_bad_names_are_reported(work):
    reply = ("### REPLACE m.py::A.go\n```python\ndef go(self):\n    return 5\n```\n"
             "### ADD m.py AT END\n```python\ndef extra():\n    return 1\n```\n"
             "### REPLACE m.py::missing\n```python\ndef missing(): pass\n```\n"
             "### TITLE t\n### SUMMARY s\n")
    meta, errors = agent._apply_sections(reply)
    text = (work / "m.py").read_text()
    assert "    def go(self):\n        return 5" in text and text.endswith("def extra():\n    return 1\n")
    assert meta["title"] == "t" and len(errors) == 1 and "missing" in errors[0]


def test_a_named_file_settles_a_name_defined_in_several(work):
    (work / "n.py").write_text("TOOLS = [1]\n")
    (work / "o.py").write_text("TOOLS = [2]\n")
    assert agent.replace_symbol({"name": "TOOLS", "path": "o.py", "new": "TOOLS = [3]"})["ok"]
    assert (work / "o.py").read_text() == "TOOLS = [3]\n" and (work / "n.py").read_text() == "TOOLS = [1]\n"
