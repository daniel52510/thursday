"""Generates THURSDAY's UI sounds: the listening ping and the thinking loop.
(The alarm chime is scripts/make_alarm_chime.py.)

Synthesized from scratch (one soft A major pad, blooming for the ping and
breathing for the thinking loop), pure stdlib, so they can be rebuilt or tweaked anywhere
and carry no third-party licence.

    python3 scripts/make_sounds.py [out_dir]     # default: assets/sounds
"""
import math
import struct
import sys
import wave
from pathlib import Path

RATE = 48000
TAU = 2 * math.pi


def finish(sig: list[float], peak: float, fade_s: float = 0.01) -> list[float]:
    top = max(abs(s) for s in sig) or 1.0
    out = [s * peak / top for s in sig]
    n = int(fade_s * RATE)
    for i in range(min(n, len(out))):
        out[i] *= i / n
        out[-1 - i] *= i / n
    return out


# A major pad shared by the listening ping and the thinking loop, so the two sound like a pair.
CHORD = ((220.0, 0.55), (329.6, 0.35), (440.0, 0.28), (554.4, 0.18), (880.0, 0.06))


def ping() -> list[float]:
    # "I'm listening": the thinking chord blooms open in 60 ms, then lets go (0.38 s).
    # Daniel picked it over sharper pings: heard many times a day, it never grates.
    dur, rise = 0.38, 0.06
    out, phases = [], [0.0] * len(CHORD)
    for i in range(int(dur * RATE)):
        t = i / RATE
        if t < rise:
            env = 0.5 - 0.5 * math.cos(math.pi * t / rise)
        else:
            env = 0.5 + 0.5 * math.cos(math.pi * (t - rise) / (dur - rise))
        s = 0.0
        for k, (f, a) in enumerate(CHORD):
            phases[k] += TAU * f / RATE  # advance first, as in the prototype Daniel picked
            s += a * math.sin(phases[k])
        out.append(env ** 1.2 * s)
    return finish(out, peak=0.14, fade_s=0.004)


def thinking() -> list[float]:
    # A major pad breathing twice per loop. Every frequency completes a whole
    # number of cycles in LOOP_S and the swell bottoms out at the loop point,
    # so it repeats seamlessly (and the Pi's restart gap falls in the quiet).
    loop_s, breaths = 2.5, 2
    out = []
    for i in range(int(loop_s * RATE)):
        t = i / RATE
        swell = 0.5 - 0.5 * math.cos(TAU * breaths * t / loop_s)
        out.append(swell ** 1.5 * sum(a * math.sin(TAU * f * t) for f, a in CHORD))
    top = max(abs(s) for s in out)
    return [s * 0.07 / top for s in out]  # quiet: it sits under the room, it doesn't announce


def write(path: Path, sig: list[float]) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in sig))
    print(f"wrote {path} ({len(sig) / RATE:.2f} s)")


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "assets" / "sounds"
    out_dir.mkdir(parents=True, exist_ok=True)
    write(out_dir / "thursday-ping.wav", ping())
    write(out_dir / "thursday-thinking.wav", thinking())


if __name__ == "__main__":
    main()
