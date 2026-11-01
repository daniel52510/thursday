"""Builds openWakeWord's four training folders from the synthetic clips and the
recordings made on the Pi (satellite/record_wake.py). Runs in wake-train/venv.

    wake-train/run/hey_thursday/positive_train  synthetic + your clips (x REAL_REPEAT)
                               /positive_test   synthetic (picks the best checkpoint)
                               /negative_train  synthetic near-misses + your talk/TV (x 3)
                               /negative_test   synthetic near-misses
    wake-train/run/background                   AudioSet + your talk/TV, mixed under clips
    wake-train/run/real_test                    10% of your recordings, never trained on:
                                                evaluate.py scores old vs new model on them

Everything is 16 kHz mono int16 WAV. Your positive clips are cut tight around the
phrase (the augmenter left-pads clips and expects the phrase to end the clip).
"""
import random
import shutil
import sys
from math import gcd
from pathlib import Path

import io
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000
ROOT = Path(__file__).resolve().parents[2]
DATA, RUN = ROOT / "wake-train" / "data", ROOT / "wake-train" / "run"
REC = ROOT / "wake-data"
BEEP_TAIL_S = 0.35  # clips recorded before the recorder skipped the beep's echo start with it
REAL_REPEAT = 8     # each copy gets different augmentation, so your voice/room weigh in
NEG_REPEAT = 3
TEST_SHARE = 0.1
random.seed(1234)


def load(path: Path) -> np.ndarray:
    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        a = resample_poly(a, SR // g, sr // g).astype(np.float32)
    return a


def save(path: Path, a: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(a, -1, 1), SR, subtype="PCM_16")


def phrase_span(a: np.ndarray) -> np.ndarray | None:
    """The loudest stretch of speech in a recorded window, by frame energy over the
    noise floor; gaps under 300 ms are bridged ("Hey ... THURSDAY")."""
    fr = SR // 50  # 20 ms
    e = np.sqrt(np.mean(a[: len(a) // fr * fr].reshape(-1, fr) ** 2, axis=1))
    floor = np.percentile(e, 20)
    voiced = e > max(floor * 3, floor + 4e-4)
    idx = np.flatnonzero(voiced)
    if not len(idx):
        return None
    runs, start = [], idx[0]
    for prev, cur in zip(idx, idx[1:]):
        if cur - prev > 15:
            runs.append((start, prev))
            start = cur
    runs.append((start, idx[-1]))
    s, t = max(runs, key=lambda r: e[r[0]:r[1] + 1].sum())
    if not 0.25 <= (t - s + 1) * fr / SR <= 1.8:
        return None
    return a[max(0, s * fr - int(0.15 * SR)): min(len(a), (t + 1) * fr + int(0.1 * SR))]


def split(files: list) -> tuple[list, list]:
    files = sorted(files)
    random.shuffle(files)
    n = max(1, round(len(files) * TEST_SHARE))
    return files[n:], files[:n]


def synthetic(src: str, dst: Path, prefix: str) -> int:
    n = 0
    for d in sorted((RUN / "synth").glob(f"{src}*")):
        for f in sorted(d.glob("*.wav")):
            save(dst / f"{prefix}_{d.name}_{f.stem}.wav", load(f))
            n += 1
    return n


def audioset(dst: Path, limit_s: float = 4 * 3600) -> int:
    """AudioSet rows as 16 kHz WAVs (10 s each), up to limit_s of audio."""
    n, total = 0, 0.0
    for shard in sorted(DATA.glob("audioset_bal_*.parquet")):
        table = pq.read_table(shard, columns=["audio"])
        for row in table.column("audio").to_pylist():
            try:
                a, sr = sf.read(io.BytesIO(row["bytes"]), dtype="float32", always_2d=True)
            except Exception:
                continue  # a codec libsndfile can't read: skip it
            a = a.mean(axis=1)
            if sr != SR:
                g = gcd(sr, SR)
                a = resample_poly(a, SR // g, sr // g).astype(np.float32)
            save(dst / f"audioset_{shard.stem}_{n:05d}.wav", a)
            n, total = n + 1, total + len(a) / SR
            if total >= limit_s:
                return n
    return n


def main() -> None:
    model = RUN / "hey_thursday"
    for d in ["positive_train", "positive_test", "negative_train", "negative_test"]:
        shutil.rmtree(model / d, ignore_errors=True)
    shutil.rmtree(RUN / "real_test", ignore_errors=True)

    counts = {
        "synthetic positive_train": synthetic("pos_train", model / "positive_train", "syn"),
        "synthetic positive_test": synthetic("pos_test", model / "positive_test", "syn"),
        "synthetic negative_train": synthetic("neg_train", model / "negative_train", "syn"),
        "synthetic negative_test": synthetic("neg_test", model / "negative_test", "syn"),
    }

    # Your "Hey THURSDAY" clips: split per distance so every distance is in the test set.
    kept = dropped = 0
    for label_dir in sorted((REC / "positive").glob("*")):
        train, test = split(list(label_dir.glob("*.wav")))
        for f in train:
            span = phrase_span(load(f)[int(BEEP_TAIL_S * SR):])
            if span is None:
                dropped += 1
                continue
            kept += 1
            for k in range(REAL_REPEAT):
                save(model / "positive_train" / f"real_{label_dir.name}_{f.stem}_{k}.wav", span)
        for f in test:  # whole window: evaluate.py streams it like the satellite would
            save(RUN / "real_test" / "positive" / label_dir.name / f.name, load(f)[int(BEEP_TAIL_S * SR):])
    counts[f"your positive clips (x{REAL_REPEAT})"] = kept
    counts["your positive clips dropped (no clear phrase)"] = dropped

    # Your talk/TV: near-miss negatives, background noise, and held-out false-wake test.
    neg = 0
    for label_dir in sorted((REC / "negative").glob("*")):
        train, test = split(list(label_dir.glob("*.wav")))
        for f in train:
            a = load(f)
            save(RUN / "background" / f"real_{label_dir.name}_{f.name}", a)
            for k in range(NEG_REPEAT):
                save(model / "negative_train" / f"real_{label_dir.name}_{f.stem}_{k}.wav", a)
            neg += 1
        for f in test:
            save(RUN / "real_test" / "negative" / label_dir.name / f.name, load(f))
    counts[f"your negative clips (x{NEG_REPEAT})"] = neg

    if not list((RUN / "background").glob("audioset_*.wav")):
        counts["AudioSet background clips"] = audioset(RUN / "background")

    for k, v in counts.items():
        print(f"  {k:48s} {v}")
    if kept < 50 or counts["synthetic positive_train"] < 1000:
        sys.exit("too little positive data; check the synthetic step and wake-data/")


if __name__ == "__main__":
    main()
