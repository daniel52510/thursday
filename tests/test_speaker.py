"""Speaker recognition logic (speaker.py), without the model: what counts as speech, and the verdicts."""
import numpy as np
import pytest

import speaker

SR = speaker.SR
rng = np.random.default_rng(3)


def _speech(seconds: float, level: float = 900, floor: float = 30) -> np.ndarray:
    """Syllable-like bursts over a quiet room, at the satellite mic's low level."""
    t = np.arange(int(SR * seconds)) / SR
    bursts = (np.sin(2 * np.pi * 3 * t) > 0).astype(float)
    x = bursts * level * np.sin(2 * np.pi * 150 * t) + rng.normal(0, floor, len(t))
    return x.astype("<i2")


def test_quiet_speech_is_kept_silence_and_noise_are_not():
    assert speaker.speech_only(_speech(2.0)) is not None
    assert speaker.speech_only(np.zeros(SR * 2, dtype="<i2")) is None
    assert speaker.speech_only(rng.normal(0, 40, SR * 2).astype("<i2")) is None  # steady noise: no contrast


def test_too_short_is_not_judged():
    assert speaker.speech_only(_speech(0.6)) is None


def test_fbank_shape_and_mean_normalized():
    f = speaker.fbank(_speech(1.0))
    assert f.shape == (98, 80) and np.allclose(f.mean(axis=0), 0, atol=1e-3)


@pytest.fixture
def fake_embed(monkeypatch):
    """embed() returns whatever unit vector the test planted for that audio's length."""
    planted = {}
    monkeypatch.setattr(speaker, "embed", lambda x: planted[len(x) // 1000])
    return planted


def _unit(*v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("vec,verdict,user", [((1, 0), "match", "owner"), ((0.3, 0.95), "unsure", None),
                                              ((0, 1), "unknown", None)])
def test_verdicts(fake_embed, vec, verdict, user):
    audio = _speech(2.0)
    fake_embed[len(speaker.speech_only(audio)) // 1000] = _unit(*vec)
    got = speaker.identify(audio.tobytes(), {"owner": _unit(1, 0)})
    assert got["verdict"] == verdict and got["user"] == user


def test_short_utterance_is_unsure_not_a_stranger():
    got = speaker.identify(_speech(0.5).tobytes(), {"owner": _unit(1, 0)})
    assert got["verdict"] == "unsure" and got["short"]  # voice_ws may carry the session's last match


def test_nobody_enrolled_is_unknown():
    assert speaker.identify(_speech(2.0).tobytes(), {})["verdict"] == "unknown"


def test_enroll_drops_clips_that_dont_match(monkeypatch):
    mine, odd = _unit(1, 0.1), _unit(-0.2, 1)
    clips = [np.full(SR * 2, i, dtype="<i2") for i in range(10)]
    monkeypatch.setattr(speaker, "embed", lambda x: odd if x[0] == 9 else mine)
    v, used = speaker.enroll(clips)
    assert used == 9 and float(v @ mine) > 0.99
