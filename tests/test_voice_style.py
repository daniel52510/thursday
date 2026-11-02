"""His voice settings: volume and whisper (mind.voice_setting, rocky_voice style, agent_stream intents)."""
import numpy as np
import pytest

import agent_stream as a
import mind
import rocky_voice as rv
from test_rocky_voice import SR, _speechish

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
    assert mind.voice_style("owner") == {"volume": 1.0, "whisper": False}
    assert mind.voice_setting({"change": "louder"}).data["voice_volume"] == 100  # already at the top
    for _ in range(6):
        mind.voice_setting({"change": "quieter"})
    assert mind.voice_style("owner")["volume"] == 0.1  # never silent
    mind.voice_setting({"volume": 60, "whisper": True})
    assert mind.voice_style("owner") == {"volume": 0.6, "whisper": True}
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
