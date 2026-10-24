"""Design THURSDAY's voice from a text description with Qwen3-TTS VoiceDesign.

VoiceDesign samples a new voice every run, so this renders a few candidates to
listen to; promote the one you like and the TTS service clones it from then on.

    .venv-tts/bin/python scripts/design_voice.py            # render candidates
    .venv-tts/bin/python scripts/design_voice.py --pick 2   # use candidate 2
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

VOICES_DIR = ROOT / "voices"
CANDIDATES_DIR = VOICES_DIR / "candidates"
DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

DEFAULT_DESCRIPTION = (
    "A Peruvian man around 50 years old speaking fluent English with a warm, natural "
    "Peruvian Spanish accent. Deep, slightly gravelly baritone. Calm, friendly and "
    "confident, like a seasoned mentor. Measured pace, clear diction, relaxed warmth."
)
# The clone prompt conditions on this exact transcript, so keep it natural and ~10-15s long.
DEFAULT_SAMPLE_TEXT = (
    "Good morning. I checked your schedule and the weather for today. "
    "You have two meetings this afternoon, and it looks like rain after five, "
    "so you might want to bring an umbrella. Let me know what else you need."
)


def render(description: str, sample_text: str, count: int) -> None:
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel
    from voice_out import device, torch_dtype

    print(f"Loading {DESIGN_MODEL} on {device} ({torch_dtype})...")
    model = Qwen3TTSModel.from_pretrained(DESIGN_MODEL, device_map=device, dtype=torch_dtype)

    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    (CANDIDATES_DIR / "transcript.txt").write_text(sample_text + "\n")
    for i in range(1, count + 1):
        wavs, sr = model.generate_voice_design(
            text=sample_text, instruct=description, language="English"
        )
        out = CANDIDATES_DIR / f"candidate_{i}.wav"
        sf.write(str(out), wavs[0], sr, subtype="PCM_16")
        print(f"  wrote {out.relative_to(ROOT)}")

    print(f"\nListen with:  afplay {CANDIDATES_DIR.relative_to(ROOT)}/candidate_1.wav")
    print("Then run:     .venv-tts/bin/python scripts/design_voice.py --pick N")


def pick(n: int) -> None:
    src = CANDIDATES_DIR / f"candidate_{n}.wav"
    if not src.exists():
        sys.exit(f"{src.relative_to(ROOT)} not found; render candidates first.")
    shutil.copy(src, VOICES_DIR / "thursday.wav")
    shutil.copy(CANDIDATES_DIR / "transcript.txt", VOICES_DIR / "thursday.txt")
    print("Saved voices/thursday.wav + thursday.txt. Restart ./scripts/run_tts_native.sh to use it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION, help="What the voice should sound like")
    parser.add_argument("--text", default=DEFAULT_SAMPLE_TEXT, help="Sample line the candidates speak")
    parser.add_argument("--count", type=int, default=3, help="How many candidates to render")
    parser.add_argument("--pick", type=int, help="Promote candidate N to the active voice")
    args = parser.parse_args()

    if args.pick:
        pick(args.pick)
    else:
        render(args.description, args.text, args.count)
