"""Enroll a person's voice for speaker recognition (speaker.py) and measure how well it tells them apart.

    docker exec -w /app thursday-core python scripts/enroll_voice.py [--user owner] [--save]

Their clips: the satellite's own recordings in /repo/wake-data (positive = "Hey THURSDAY" at every
distance, negative/talk = normal talking), made on the same mic and room as real use. It enrolls on
half, scores the other half against other voices (synthetic "Hey THURSDAY" in many voices from
wake-train, same words so only the voice differs, and the TV clips), and prints false accepts /
false rejects at each threshold, then the thresholds it suggests. --save enrolls on every clip and
stores the voiceprint in the vault. No audio leaves the machine or is kept.
"""
import argparse
import glob
import random
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app")
import speaker  # noqa: E402

DATA = Path("/repo/wake-data")
OTHERS = Path("/repo/wake-train/run/hey_thursday/positive_train")


def load(path: str) -> np.ndarray | None:
    with wave.open(path) as w:
        if w.getframerate() != speaker.SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
            return None
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return speaker.speech_only(x)


def clips(pattern: str, limit: int | None = None) -> list[np.ndarray]:
    paths = sorted(glob.glob(pattern, recursive=True))
    if limit and len(paths) > limit:
        paths = random.Random(7).sample(paths, limit)
    return [c for p in paths if (c := load(p)) is not None and len(c) >= speaker.SR * speaker.MIN_SECONDS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="owner")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    # His enrollment recording (record_wake.py negative enroll): consecutive 2.16 s windows, joined in threes
    # (~6.5 s, a long request). Enrolled on half; the other half and his wake clips are the test.
    raw = []
    for p in sorted(glob.glob(str(DATA / "negative/enroll/*.wav"))):
        with wave.open(p) as w:
            raw.append(np.frombuffer(w.readframes(w.getnframes()), dtype="<i2"))
    talk = [s for i in range(0, len(raw) - 2, 3) if (s := speaker.speech_only(np.concatenate(raw[i:i + 3]))) is not None]
    wake = clips(str(DATA / "positive/**/*.wav"))
    tv = clips(str(DATA / "negative/tv/*.wav"))
    others = clips(str(OTHERS / "*.wav"), limit=300)
    print(f"clips: {len(talk)} talking chunks + {len(wake)} wake clips of him, {len(others)} other voices "
          f"(synthetic), {len(tv)} TV")
    random.Random(1).shuffle(talk)
    enroll_half, test_half = talk[::2], talk[1::2]
    v, used = speaker.enroll(enroll_half)
    print(f"enrolled on {used} of {len(enroll_half)} chunks (the rest didn't sound like him)")
    score = lambda cs: np.array([float(speaker.embed(c) @ v) for c in cs]) if cs else np.zeros(0)
    him, him_wake, other, tvs = score(test_half), score(wake), score(others), score(tv)
    print(f"his held-out talking: median {np.median(him):.2f} min {him.min():.2f} | his wake phrase: median "
          f"{np.median(him_wake):.2f} 5th pct {np.percentile(him_wake, 5):.2f}")
    him = np.concatenate([him, him_wake])
    mine = talk + wake
    print(f"other voices median {np.median(other):.2f} (max {other.max():.2f})"
          + (f" | TV median {np.median(tvs):.2f} (max {tvs.max():.2f})" if len(tvs) else " | TV: no speech in those clips"))
    impostors = np.concatenate([other, tvs])
    print("threshold | rejects him | accepts someone else")
    for t in np.arange(0.30, 0.80, 0.05):
        print(f"   {t:.2f}   |   {np.mean(him < t) * 100:5.1f}%    |   {np.mean(impostors >= t) * 100:5.1f}%")
    # Match: the lowest threshold that lets no impostor through (with a margin). Unknown: under nearly all of him.
    high = max(float(impostors.max()) + 0.03, float(np.percentile(him, 2)))
    low = min(float(np.percentile(him, 1)), high - 0.05)
    print(f"suggested: SPEAKER_THRESHOLD={high:.2f} (match), SPEAKER_LOW={low:.2f} (under it: someone else); "
          f"at those, he's rejected {np.mean(him < high) * 100:.1f}% (as unsure, not as a stranger: "
          f"{np.mean(him < low) * 100:.1f}%)")
    if args.save:
        full, used = speaker.enroll(talk)  # the talking: long, natural speech
        speaker.save_voiceprint(args.user, full, used)
        print(f"saved {args.user}'s voiceprint from {used} clips (in the vault)")


if __name__ == "__main__":
    main()
