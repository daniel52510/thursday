"""Records wake-word training clips on the Pi's own mic, in its own room, so a
retrained "Hey THURSDAY" model learns the distance, echo and noise it will hear.

Stops the satellite and holds the mic while it runs. systemd keeps restarting
the satellite, which exits at once while the mic is busy, and comes back by
itself a few seconds after this script ends. No sudo needed.

    venv/bin/python record_wake.py positive 1m          # 40 clips, one per beep
    venv/bin/python record_wake.py positive 4m --count 60
    venv/bin/python record_wake.py negative talk --seconds 120

positive: a beep, then a WINDOW_S window to say "Hey THURSDAY"; repeats. Each
window is saved whole (16 kHz mono WAV); trimming happens at training time.
negative: one long recording of anything that is NOT the wake word (talking,
"Thursday" alone, "hey there", TV), saved in WINDOW_S pieces.
A double beep means the round is done.

Clips land in ~/wake-data/<kind>/<label>/. Pull them to the Mac with:
    rsync -a -e "ssh -i ~/.ssh/thursday_pi" <user>@<pi>:wake-data/ wake-data/
"""
import argparse
import subprocess
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280
CHUNK_BYTES = CHUNK_SAMPLES * 2
DEVICE = "plughw:CARD=wm8960soundcard"
OUT = Path("~/wake-data").expanduser()
WINDOW_S = 2.2  # time to react to the beep and say the phrase
BEEP_S = 0.12
AFTER_BEEP_S = 0.4  # the beep echoes off the room for ~0.3 s: skipped, not recorded


def tone(freq: float, seconds: float) -> bytes:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    fade = np.minimum(1, np.minimum(t, seconds - t) / 0.01)  # no clicks
    return (np.sin(2 * np.pi * freq * t) * fade * 0.3 * 32767).astype(np.int16).tobytes()


def play(pcm: bytes) -> subprocess.Popen:
    p = subprocess.Popen(["aplay", "-q", "-D", DEVICE, "-t", "raw", "-f", "S16_LE", "-c", "1",
                          "-r", str(SAMPLE_RATE), "-"], stdin=subprocess.PIPE)
    p.stdin.write(pcm)
    p.stdin.close()
    return p


def read(mic: subprocess.Popen, seconds: float) -> bytes:
    n = int(seconds * SAMPLE_RATE / CHUNK_SAMPLES)
    return b"".join(mic.stdout.read(CHUNK_BYTES) for _ in range(n))


def save(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


def level(pcm: bytes) -> str:
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return f"peak {int(np.abs(a).max()):5d}  rms {int(np.sqrt((a ** 2).mean())):4d}"


def open_mic() -> subprocess.Popen:
    """Stop the satellite and take the mic before systemd starts it again."""
    subprocess.run(["pkill", "-f", "[p]ython satellite.py"])
    for _ in range(20):
        time.sleep(0.25)
        mic = subprocess.Popen(["arecord", "-q", "-D", DEVICE, "-f", "S16_LE", "-r", str(SAMPLE_RATE),
                                "-c", "1", "-t", "raw"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        time.sleep(0.3)
        if mic.poll() is None:
            return mic
    raise SystemExit("couldn't get the mic (still busy)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["positive", "negative"])
    ap.add_argument("label", help="where you stand or what you say, e.g. 1m, 4m-away, talk, tv")
    ap.add_argument("--count", type=int, default=40, help="positive: clips to record")
    ap.add_argument("--seconds", type=int, default=120, help="negative: how long to record")
    args = ap.parse_args()

    folder = OUT / args.kind / args.label
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    beep, done = tone(880, BEEP_S), tone(660, 0.15) + bytes(3200) + tone(660, 0.15)
    mic = open_mic()
    try:
        read(mic, 0.5)  # let the HAT settle
        if args.kind == "positive":
            print(f"Say \"Hey THURSDAY\" after each beep ({args.count} times). Vary tone and volume.")
            time.sleep(2)
            for i in range(1, args.count + 1):
                play(beep)
                read(mic, BEEP_S + AFTER_BEEP_S)
                pcm = read(mic, WINDOW_S)
                save(folder / f"{args.label}_{stamp}_{i:03d}.wav", pcm)
                print(f"{i:3d}/{args.count}  {level(pcm)}")
        else:
            print(f"Recording {args.seconds} s of anything but the wake word.")
            play(beep).wait()
            read(mic, AFTER_BEEP_S)
            for i in range(1, int(args.seconds / WINDOW_S) + 1):
                save(folder / f"{args.label}_{stamp}_{i:03d}.wav", read(mic, WINDOW_S))
        play(done).wait()
    finally:
        mic.terminate()
    print(f"saved to {folder}; the satellite restarts by itself in a few seconds")


if __name__ == "__main__":
    main()
