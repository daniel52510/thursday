"""Rocky's voice DSP (rocky_voice.py): streaming must sound exactly like processing the whole piece, or
the satellite hears seams; the stream must release audio steadily (capped passes), not in bursts."""
import numpy as np
import pytest

import rocky_voice as rv

SR = 24000


def _speechish(seconds: float) -> bytes:
    t = np.arange(int(SR * seconds)) / SR
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # syllable-ish loudness
    x = env * (0.3 * np.sin(2 * np.pi * 140 * t) + 0.1 * np.random.default_rng(1).standard_normal(len(t)))
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


@pytest.mark.parametrize("text", ["Forty-one degrees and windy. Jacket.", "You want rain jacket too, question?"])
def test_streamed_equals_whole(text):
    pcm = _speechish(2.5)
    whole = rv.process(np.frombuffer(pcm, "<i2").astype(np.float32) / 32768.0, SR, text, final=True)
    whole = (np.clip(whole, -1, 1) * 32767).astype("<i2").tobytes()
    s = rv.Stream(SR, text)
    out = b"".join(s.feed(pcm[i:i + 1920]) for i in range(0, len(pcm), 1920)) + s.finish()
    assert out == whole


def test_passes_never_wait_for_more_than_the_cap():
    s = rv.Stream(SR, "Plain sentence here.")
    pcm, waits, last = _speechish(5.0), [], 0
    for i in range(0, len(pcm), 960):
        if s.feed(pcm[i:i + 960]):
            n = (i + 960) // 2
            waits.append((n - last) / SR)
            last = n
    assert max(waits[2:]) <= rv.STREAM_PASS_MAX_S + 0.05, waits
