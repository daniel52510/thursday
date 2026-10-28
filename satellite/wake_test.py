"""Wake-word test for the Pi satellite: listens on the ReSpeaker HAT and prints
a line every time "Hey THURSDAY" scores above the threshold.

Audio is captured with arecord (no PortAudio needed) as 16 kHz mono int16 PCM
and fed to openWakeWord in 1280-sample (80 ms) chunks, its native step size.

    venv/bin/python wake_test.py [--threshold 0.5] [--debug]
"""
import argparse
import subprocess
import sys
import time
import types

import numpy as np

# openwakeword/__init__.py imports its verifier-training module, which pulls in
# scipy and scikit-learn (100+ MB, slow import on a Zero 2W). We never train
# here, so stand in an empty module before the import.
_stub = types.ModuleType("openwakeword.custom_verifier_model")
_stub.train_custom_verifier = None
sys.modules["openwakeword.custom_verifier_model"] = _stub

from openwakeword.model import Model  # noqa: E402

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280
CHUNK_BYTES = CHUNK_SAMPLES * 2
DEVICE = "plughw:CARD=wm8960soundcard"
MODEL_PATH = "models/hey_thursday.onnx"
# After a detection, ignore further hits for this long so one phrase prints once.
COOLDOWN_S = 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--debug", action="store_true", help="print the peak score every second")
    args = ap.parse_args()

    model = Model(wakeword_models=[MODEL_PATH], inference_framework="onnx")
    name = next(iter(model.models))
    mic = subprocess.Popen(
        ["arecord", "-q", "-D", DEVICE, "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw"],
        stdout=subprocess.PIPE,
    )
    print(f"listening for '{name}' (threshold {args.threshold}); Ctrl+C to stop", flush=True)

    last_hit = 0.0
    peak, peak_since = 0.0, time.monotonic()
    try:
        while True:
            data = mic.stdout.read(CHUNK_BYTES)
            if len(data) < CHUNK_BYTES:
                sys.exit("arecord stopped")
            started = time.monotonic()
            score = model.predict(np.frombuffer(data, dtype=np.int16))[name]
            infer_ms = (time.monotonic() - started) * 1000

            now = time.monotonic()
            if score >= args.threshold and now - last_hit > COOLDOWN_S:
                last_hit = now
                print(f"{time.strftime('%H:%M:%S')}  WAKE  score={score:.2f}  infer={infer_ms:.0f}ms", flush=True)
            if args.debug:
                peak = max(peak, score)
                if now - peak_since >= 1.0:
                    print(f"  peak={peak:.3f}  infer={infer_ms:.0f}ms", flush=True)
                    peak, peak_since = 0.0, now
    except KeyboardInterrupt:
        pass
    finally:
        mic.terminate()


if __name__ == "__main__":
    main()
