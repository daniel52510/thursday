"""Scores the old and the new wake-word model on recordings neither was trained on
(wake-train/run/real_test, held out by prepare.py). Runs in wake-train/venv.

Each clip is streamed through openWakeWord in 80 ms chunks, the way the satellite
does it, after 1 s of the clip's own background noise so the model's context is full.
Positive clips report the peak score; negative clips count as a false wake when
their peak passes the threshold.

    wake-train/venv/bin/python scripts/wake_train/evaluate.py [new.onnx]
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from openwakeword.model import Model

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "wake-train" / "run" / "real_test"
OLD = ROOT / "satellite" / "models" / "hey_thursday.onnx"
NEW = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "wake-train" / "run" / "hey_thursday.onnx"
THRESHOLDS = (0.3, 0.4, 0.5)


def peak(model: Model, path: Path) -> float:
    a, _ = sf.read(str(path), dtype="int16")
    quiet = a[: 1600] if len(a) >= 1600 else np.zeros(1600, np.int16)
    audio = np.concatenate([np.tile(quiet, 10), a, np.zeros(8000, np.int16)])
    model.reset()
    name = next(iter(model.models))
    return max(model.predict(audio[i:i + 1280])[name] for i in range(0, len(audio) - 1279, 1280))


def main() -> None:
    models = {"old": Model(wakeword_models=[str(OLD)], inference_framework="onnx"),
              "new": Model(wakeword_models=[str(NEW)], inference_framework="onnx")}
    print(f"{'positive':10s} {'n':>3s}  " + "  ".join(f"{m}: median  " + " ".join(f">={t}" for t in THRESHOLDS)
                                               for m in models))
    for label in sorted((TEST / "positive").iterdir()):
        clips = sorted(label.glob("*.wav"))
        cells = []
        for m in models.values():
            p = np.array([peak(m, c) for c in clips])
            cells.append(f"{np.median(p):10.2f}  " + " ".join(f"{(p >= t).mean():4.0%}" for t in THRESHOLDS))
        print(f"{label.name:10s} {len(clips):3d}  " + "    ".join(cells))
    neg = sorted((TEST / "negative").glob("*/*.wav"))
    if neg:
        print(f"\nfalse wakes on {len(neg)} held-out talk/TV clips ({len(neg) * 2.2 / 60:.1f} min):")
        for name, m in models.items():
            p = np.array([peak(m, c) for c in neg])
            print(f"  {name}: " + "  ".join(f">={t}: {(p >= t).sum()}" for t in THRESHOLDS))


if __name__ == "__main__":
    main()
