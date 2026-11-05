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
