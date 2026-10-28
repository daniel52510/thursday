"""Design THURSDAY's voice from a text description with Qwen3-TTS VoiceDesign.

VoiceDesign samples a new voice every run, so this renders a few candidates to
listen to; promote the one you like and the TTS service clones it from then on.

    .venv-tts/bin/python scripts/design_voice.py                       # render candidates
    .venv-tts/bin/python scripts/design_voice.py --pick 2              # use candidate 2
    .venv-tts/bin/python scripts/design_voice.py --dir candidates-x    # separate batch
    .venv-tts/bin/python scripts/design_voice.py --speaker sohee       # start from a preset voice
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
# Preset speakers (vivian, serena, sohee, ...). Only the 1.7B CustomVoice model
# follows --description; the 0.6B one silently ignores instructions.
CUSTOM_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# "Slight" or "soft" accents come out close to neutral American; name the
# accent and its features plainly.
DEFAULT_DESCRIPTION = (
    "A woman in her early thirties from Dublin, Ireland, speaking English with a clear, "
    "unmistakable Irish accent: Irish vowels, a musical rising-and-falling Irish lilt, "
    "lightly tapped r's, and a softened 'th'. Warm, quick-witted and confident like a "
    "capable AI assistant, with a dry, playful sense of humour. Natural conversational "
    "pace, clear diction, relaxed and friendly."
)
# The clone prompt conditions on this exact transcript, so keep it natural and ~10-15s long.
# Irish phrasing ("you've two meetings", "you'd want to") pulls the accent along with it.
DEFAULT_SAMPLE_TEXT = (
    "Morning! I've had a look at your day for you. You've two meetings this afternoon, "
    "and there's rain coming in after five, so you'd want to bring a jacket with you. "
    "Right so, is there anything else I can do for you?"
)


def render(description: str, sample_text: str, count: int, candidates_dir: Path,
           speaker: str | None = None, custom_model: str = CUSTOM_MODEL) -> None:
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel
    from voice_out import device, torch_dtype

    model_id = custom_model if speaker else DESIGN_MODEL
    print(f"Loading {model_id} on {device} ({torch_dtype})...")
    model = Qwen3TTSModel.from_pretrained(model_id, device_map=device, dtype=torch_dtype)

    candidates_dir.mkdir(parents=True, exist_ok=True)
    (candidates_dir / "transcript.txt").write_text(sample_text + "\n")
    (candidates_dir / "description.txt").write_text(description + "\n")
    for i in range(1, count + 1):
        if speaker:
            wavs, sr = model.generate_custom_voice(
                text=sample_text, speaker=speaker, instruct=description, language="English"
            )
        else:
            wavs, sr = model.generate_voice_design(
                text=sample_text, instruct=description, language="English"
            )
        out = candidates_dir / f"candidate_{i}.wav"
        sf.write(str(out), wavs[0], sr, subtype="PCM_16")
        print(f"  wrote {out.relative_to(ROOT)}")

    rel = candidates_dir.relative_to(ROOT)
    print(f"\nListen with:  afplay {rel}/candidate_1.wav")
    print(f"Then run:     .venv-tts/bin/python scripts/design_voice.py --dir {candidates_dir.name} --pick N")


def pick(n: int, candidates_dir: Path) -> None:
    src = candidates_dir / f"candidate_{n}.wav"
    if not src.exists():
        sys.exit(f"{src.relative_to(ROOT)} not found; render candidates first.")
    # Keep the voice being replaced, so a bad pick is one copy away from undone.
    for name in ("thursday.wav", "thursday.txt"):
        if (VOICES_DIR / name).exists():
            shutil.copy(VOICES_DIR / name, VOICES_DIR / name.replace("thursday", "thursday.prev"))
    shutil.copy(src, VOICES_DIR / "thursday.wav")
    shutil.copy(candidates_dir / "transcript.txt", VOICES_DIR / "thursday.txt")
    print("Saved voices/thursday.wav + thursday.txt (previous voice in voices/thursday.prev.*).")
    print("Restart ./scripts/run_tts_native.sh to use it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION, help="What the voice should sound like")
    parser.add_argument("--text", default=DEFAULT_SAMPLE_TEXT, help="Sample line the candidates speak")
    parser.add_argument("--count", type=int, default=3, help="How many candidates to render")
    parser.add_argument("--pick", type=int, help="Promote candidate N to the active voice")
    parser.add_argument("--dir", default=CANDIDATES_DIR.name, help="Candidates folder under voices/")
    parser.add_argument("--speaker", help="Preset speaker to start from (CustomVoice model), e.g. sohee")
    parser.add_argument("--custom-model", default=CUSTOM_MODEL, help="CustomVoice model id for --speaker")
    args = parser.parse_args()

    candidates_dir = VOICES_DIR / args.dir
    if args.pick:
        pick(args.pick, candidates_dir)
    else:
        render(args.description, args.text, args.count, candidates_dir, args.speaker, args.custom_model)
