"""THURSDAY's tests: `python -m pytest -q tests` (runner/checks.py runs them on every self-improvement job).

They cover the logic that has broken before, with no network, no GPU, no devices and no real data:
every test runs in its own temp folder, so the database the code opens (./brain/...) is a fresh one.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NTFY_URL", "")  # no phone pushes from tests
os.environ.setdefault("MUSIC_TAP_TOKEN", "test-token")


@pytest.fixture(autouse=True)
def _own_folder(tmp_path, monkeypatch):
    """A fresh working folder: ./brain/thursday_memory.db is a new, empty database per test."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "brain").mkdir()


@pytest.fixture
def owner_turn():
    """Set the turn as Daniel's, in one conversation; call with what he said."""
    from calendar_tool import TURN

    def set_turn(said: str = "", conversation: str = "c1", turn: str = "t1"):
        TURN.set({"user": "owner", "conversation": conversation, "turn": turn, "user_said": said,
                  "timezone": "America/New_York"})
    return set_turn
