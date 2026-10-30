"""Generates assets/sounds/thursday-alarm.wav: a soft rising C-E-G chime,
played twice, for the satellite alarm. Pure stdlib, so the asset can be
rebuilt anywhere and carries no third-party licence.

    python3 scripts/make_alarm_chime.py
"""
import math
import struct
import wave
from pathlib import Path

RATE = 48000
NOTES = (523.25, 659.25, 783.99)  # C5, E5, G5
NOTE_GAP_S = 0.22                 # onset spacing within one phrase
RING_S = 1.4                      # each note's decay tail
PHRASE_GAP_S = 0.9                # silence-ish between the two phrases
PEAK = 0.32                       # ~-10 dBFS: wakes you without a jolt
OUT = Path(__file__).resolve().parent.parent / "assets" / "sounds" / "thursday-alarm.wav"


def bell(freq: float, t: float) -> float:
    # Soft attack, exponential decay, a little of the octave for a bell colour.
    attack = min(1.0, t / 0.015)
    env = attack * math.exp(-3.2 * t)
    return env * (math.sin(2 * math.pi * freq * t) + 0.25 * math.sin(2 * math.pi * 2 * freq * t))


def main() -> None:
    phrase_len = NOTE_GAP_S * (len(NOTES) - 1) + RING_S
    onsets = [p * (phrase_len - RING_S + PHRASE_GAP_S) + i * NOTE_GAP_S for p in range(2) for i in range(len(NOTES))]
    freqs = [NOTES[i] for _ in range(2) for i in range(len(NOTES))]
    total = onsets[-1] + RING_S
    samples = []
    for n in range(int(total * RATE)):
        t = n / RATE
        samples.append(sum(bell(f, t - o) for f, o in zip(freqs, onsets) if 0 <= t - o < RING_S))
    scale = PEAK / max(abs(s) for s in samples)
    with wave.open(str(OUT), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"".join(struct.pack("<h", int(s * scale * 32767)) for s in samples))
    print(f"wrote {OUT} ({total:.1f} s)")


if __name__ == "__main__":
    main()
