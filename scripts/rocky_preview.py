"""Hear Rocky before he's wired in: one TTS sentence through RockyVoiceProcessor (rocky_voice.py),
saved at speaker quality and at phone quality.

    python scripts/rocky_preview.py [--text "..."] [--out rocky_preview] [--tts http://localhost:9001]

Writes tts_original.wav (for comparison), rocky_24khz.wav and rocky_8khz_mulaw_preview.wav (real
G.711 mu-law, what a Twilio call carries), and prints how long the processing took.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rocky_voice  # noqa: E402

TEXT = "Amaze! Amaze! Amaze! System metrics look good, good, good. We test phone call now, question?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--out", default="rocky_preview")
    ap.add_argument("--tts", default=os.getenv("TTS_BASE_URL", "http://localhost:9001"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(f"{args.tts}/speak/file", json.dumps({"text": args.text}).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    wav = urllib.request.urlopen(req, timeout=60).read()
    tts_ms = (time.perf_counter() - t0) * 1000
    (out / "tts_original.wav").write_bytes(wav)

    x, sr = rocky_voice._read(wav)
    t0 = time.perf_counter()
    y = rocky_voice.process(x, sr, args.text)
    dsp_ms = (time.perf_counter() - t0) * 1000
    (out / "rocky_24khz.wav").write_bytes(rocky_voice._write(y, sr))
    t0 = time.perf_counter()
    mulaw = rocky_voice.to_mulaw_8k(y, sr)
    phone_ms = (time.perf_counter() - t0) * 1000
    (out / "rocky_8khz_mulaw_preview.wav").write_bytes(rocky_voice.mulaw_wav(mulaw))

    print(f"text: {args.text}")
    print(f"tts: {len(x) / sr:.2f} s of speech at {sr} Hz in {tts_ms:.0f} ms")
    print(f"rocky: {len(y) / sr:.2f} s (pitch {rocky_voice.PITCH_SEMITONES:+.1f} st, chords {rocky_voice.CHORD_MIX:.0%}), "
          f"dsp {dsp_ms:.1f} ms = {dsp_ms / (len(x) / sr) / 10:.2f}% of real time; peak {abs(y).max():.2f}")
    print(f"phone: {len(mulaw) / 8000:.2f} s mu-law 8 kHz in {phone_ms:.1f} ms")
    print(f"files: {out.resolve()}/")


if __name__ == "__main__":
    main()
