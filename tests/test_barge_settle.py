"""Web barge-in: the start of each reply only learns her echo, so her own first words can't cut her off."""
import numpy as np

import vad


def _frame(db):
    amp = 32768 * 10 ** (db / 20) * np.sqrt(2)
    return (amp * np.sin(np.arange(vad.FRAME_SAMPLES) / 4)).astype(np.int16).tobytes()


class Always:  # a VAD that always says "speech": only the levels and the settle window decide
    def __call__(self, frame): return 0.99
    def reset(self): pass


def test_echo_at_the_start_never_triggers(monkeypatch):
    monkeypatch.setattr(vad, "SileroVAD", Always)
    b = vad.BargeIn(vad.VADConfig.from_env())
    echo = _frame(-36)  # her leaked voice: above the -40 floor
    assert not any(b.feed(echo) for _ in range(vad._frames(600)))   # settle: learned, not counted
    assert not any(b.feed(echo) for _ in range(30))                  # same echo after: under floor + margin
    loud = _frame(-20)                                              # him, talking over her
    assert any(b.feed(loud) for _ in range(30))
