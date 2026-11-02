"""His voice settings: volume and whisper (mind.voice_setting, rocky_voice style, agent_stream intents)."""
from datetime import datetime

import numpy as np
import pytest

import agent_stream as a
import mind
import rocky_voice as rv
from test_rocky_voice import SR, _speechish

NOON = datetime(2026, 11, 2, 12, 0, tzinfo=mind.VOICE_TZ)
NIGHT = datetime(2026, 11, 2, 23, 30, tzinfo=mind.VOICE_TZ)

VOICE = [
    ("whisper", {"whisper": True}), ("Hey Thursday, whisper to me", {"whisper": True}),
    ("speak softly", {"whisper": True}), ("stop whispering", {"whisper": False}), ("speak normally", {"whisper": False}),
    ("talk louder", {"change": "louder"}), ("I can't hear you", {"change": "louder"}),
    ("talk a bit quieter", {"change": "quieter"}), ("you're too loud", {"change": "quieter"}),
    ("set your voice to 40 percent", {"volume": 40}), ("your volume to 70", {"volume": 70}),
]


@pytest.mark.parametrize("said,args", VOICE)
def test_voice_requests_run_in_code(said, args):
    assert a._voice_intent(said) == ("voice_setting", args)


@pytest.mark.parametrize("said", ["turn it up", "louder", "set the volume to 30", "what is whisper ai", "whisper is a model"])
def test_music_and_other_things_left_alone(said):
    assert a._voice_intent(said) is None


def test_turn_it_up_is_still_music():
    assert a._music_intent("turn it up") == ("music_control", {"action": "volume_up"})


def test_settings_persist_and_clamp(owner_turn):
    owner_turn("talk louder")
    assert mind.voice_style("owner", NOON) == {"volume": 1.0, "whisper": False}
    assert mind.voice_setting({"change": "louder"}).data["voice_volume"] == 100  # already at the top
    for _ in range(6):
        mind.voice_setting({"change": "quieter"})
    assert mind.voice_style("owner", NOON)["volume"] == 0.01 ** 2  # 1%: near silence, not none
    mind.voice_setting({"volume": 60, "whisper": True})
    assert mind.voice_style("owner", NOON) == {"volume": pytest.approx(0.36), "whisper": True}


def test_volume_curve_is_perceptual(owner_turn):
    """Percent squared: 50% is about -12 dB (linear 50% sounded nearly full)."""
    owner_turn("")
    mind.voice_setting({"volume": 50})
    assert 20 * np.log10(mind.voice_style("owner", NOON)["volume"]) == pytest.approx(-12.04, abs=0.1)


FOLLOW_UPS = [("Try to go to 20%.", 20), ("Alright, go to five.", 5), ("What about 1%?", 1), ("Go a hundred.", 100),
              ("50", 50), ("how about 30 percent", 30)]


@pytest.mark.parametrize("said,level", FOLLOW_UPS)
def test_a_bare_level_follows_the_last_voice_change(said, level):
    a._volume_target["cv"] = ("voice", a.time.monotonic())
    assert a._bare_level("cv", said) == ("voice_setting", {"volume": level})


def test_a_bare_level_follows_music_when_music_changed_last():
    a._volume_target["cm"] = ("music", a.time.monotonic())
    assert a._bare_level("cm", "go to 20") == ("music_control", {"action": "set_volume", "level": 20})


def test_no_recent_volume_change_no_bare_level():
    assert a._bare_level("fresh", "go to 20") is None
    a._volume_target["old"] = ("voice", a.time.monotonic() - a.VOLUME_CONTEXT_S - 1)
    assert a._bare_level("old", "what about 1%") is None


@pytest.mark.parametrize("said", ["Voice at twenty. Very quiet now.", "Voice_setting{volume:1}Voice"])
def test_invented_voice_changes_dropped(said):
    import voice_ws
    assert voice_ws._drop_invented(said) == "" or "twenty" not in voice_ws._drop_invented(said)
    assert not mind.voice_setting({}).ok


def test_volume_scales_the_voice():
    x = np.frombuffer(_speechish(1.0), "<i2").astype(np.float32) / 32768.0
    full, half = rv.process(x, SR, "Hello there."), rv.process(x, SR, "Hello there.", style={"volume": 0.5})
    assert np.allclose(half, full * 0.5, atol=1e-6)


def test_whisper_is_quieter_and_different():
    x = np.frombuffer(_speechish(1.0), "<i2").astype(np.float32) / 32768.0
    normal, whisper = rv.process(x, SR, "Hello there."), rv.process(x, SR, "Hello there.", style={"whisper": True})
    rms = lambda y: float(np.sqrt(np.mean(y ** 2)))
    assert rms(whisper) < rms(normal) and not np.allclose(whisper, normal)


def test_whisper_streams_like_the_whole_piece():
    pcm, style = _speechish(2.0), {"whisper": True, "volume": 0.7}
    whole = rv.process(np.frombuffer(pcm, "<i2").astype(np.float32) / 32768.0, SR, "Quiet now.", final=True, style=style)
    whole = (np.clip(whole, -1, 1) * 32767).astype("<i2").tobytes()
    s = rv.Stream(SR, "Quiet now.", style)
    assert b"".join(s.feed(pcm[i:i + 1920]) for i in range(0, len(pcm), 1920)) + s.finish() == whole


def test_whispers_at_night_on_its_own(owner_turn):
    owner_turn("")
    assert mind.voice_style("owner", NIGHT)["whisper"] and not mind.voice_style("owner", NOON)["whisper"]


def test_speak_normally_at_night_holds_until_morning(owner_turn, monkeypatch):
    owner_turn("speak normally")
    real = mind.datetime
    monkeypatch.setattr(mind, "datetime", type("D", (), {"now": staticmethod(lambda tz=None: NIGHT)}))
    mind.voice_setting({"whisper": False})
    monkeypatch.setattr(mind, "datetime", real)
    assert not mind.voice_style("owner", NIGHT)["whisper"]  # tonight: normal, as he asked
    next_night = datetime(2026, 11, 3, 23, 30, tzinfo=mind.VOICE_TZ)
    assert mind.voice_style("owner", next_night)["whisper"]  # the next night: whispering again


def test_night_whisper_can_be_turned_off(owner_turn):
    owner_turn("don't whisper at night")
    assert a._voice_intent("don't whisper at night") == ("voice_setting", {"night_whisper": False})
    assert a._voice_intent("whisper at night") == ("voice_setting", {"night_whisper": True})
    mind.voice_setting({"night_whisper": False})
    assert not mind.voice_style("owner", NIGHT)["whisper"]


@pytest.mark.parametrize("said,args", [("volume to 50%.", {"volume": 50}), ("Can you lower the volume to 10%?", {"volume": 10}),
                                       ("turn the volume down", {"change": "quieter"}), ("volume up", {"change": "louder"})])
def test_plain_volume_is_his_voice_when_no_music_plays(said, args, monkeypatch):
    import music
    monkeypatch.setattr(music, "playing_hint", lambda: False)
    assert a._voice_intent(said) == ("voice_setting", args)


def test_plain_volume_is_the_music_while_it_plays(monkeypatch):
    import music
    monkeypatch.setattr(music, "playing_hint", lambda: True)
    assert a._voice_intent("volume to 50%") is None
    assert a._music_intent("set the volume to 50") == ("music_control", {"action": "set_volume", "level": 50})


def test_music_volume_is_always_music(monkeypatch):
    import music
    monkeypatch.setattr(music, "playing_hint", lambda: False)
    assert a._voice_intent("music volume to 30") is None


def test_whats_your_volume_reads_the_setting(owner_turn):
    owner_turn("what's your volume at?")
    assert a._voice_intent("What's your volume at?") == ("voice_setting", {"report": True})
    mind.voice_setting({"volume": 40})
    assert mind.voice_setting({"report": True}).data["voice_volume"] == 40


def test_pause_and_play_keep_the_playing_hint(monkeypatch):
    import music
    monkeypatch.setattr(music, "_cmd", lambda *a, **k: None)
    monkeypatch.setattr(music, "_find", lambda q, k: ({"videoId": "x", "playlistId": "RDAMVMx"}, "x"))
    music._state["value"] = None
    music.music_play({"query": "jazz"})
    assert music.playing_hint()
    music.music_control({"action": "pause"})
    assert not music.playing_hint()
