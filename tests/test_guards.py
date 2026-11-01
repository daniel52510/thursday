"""Claims with no tool behind them are dropped before they're spoken (voice_ws._drop_invented), and
music tools fail softly (music.py) instead of raising into a turn."""
import pytest

import music
import voice_ws


@pytest.mark.parametrize("said", ["John Wick music, playing.", "Music, off.", "Heavy metal, playing. Loud and fast!"])
def test_invented_music_claims_dropped(said):
    assert "playing" not in voice_ws._drop_invented(said) and "Music, off" not in voice_ws._drop_invented(said)


@pytest.mark.parametrize("said", ["I like playing golf too.", "Bohemian Rhapsody is a classic.", "Jupiter is best planet."])
def test_ordinary_sentences_kept(said):
    assert voice_ws._drop_invented(said) == said


def test_music_play_not_paired_says_so(monkeypatch):
    monkeypatch.setattr(music, "_token", lambda: None)
    monkeypatch.setattr(music, "_find", lambda q, k: ({"videoId": "x", "playlistId": "RDAMVMx"}, "x by y"))
    r = music.music_play({"query": "Bohemian Rhapsody"})
    assert not r.ok and r.error.startswith("not_paired")


def test_music_play_sends_play_after_loading(monkeypatch):
    """changeVideo keeps a paused player paused: music_play must send play too."""
    sent = []
    monkeypatch.setattr(music, "_cmd", lambda command, data=None: sent.append(command))
    monkeypatch.setattr(music, "_find", lambda q, k: ({"videoId": "x", "playlistId": "RDAMVMx"}, "x by y"))
    assert music.music_play({"query": "anything"}).ok and sent == ["changeVideo", "play"]


def test_unknown_music_action_refused(monkeypatch):
    monkeypatch.setattr(music, "_cmd", lambda *a, **k: None)
    assert not music.music_control({"action": "explode"}).ok
