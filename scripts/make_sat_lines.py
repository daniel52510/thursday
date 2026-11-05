"""Renders the satellite's own spoken lines in THURSDAY's voice, for the moments
the Mac can't speak for him: it's unreachable, a turn failed, or the mic was
muted with the button. Played from the Pi as plain WAVs. Also the backchannel
clips ("Mm." "Right.") the Pi plays over the user, mid-sentence (voice_ws.BACKCHANNEL).

Every clip goes through Rocky's voice (rocky_voice.py), the same as everything the Mac says,
so it needs NumPy: run it in the container, then deploy with scripts/deploy_sat.sh.
Needs the voice server running (THURSDAY_TTS_URL, default the Orpheus voice on :9001).

    docker cp scripts/make_sat_lines.py thursday-core:/app/scripts/
    docker exec -e THURSDAY_TTS_URL=http://host.docker.internal:9001 thursday-core \
        python scripts/make_sat_lines.py /tmp/sounds && docker cp thursday-core:/tmp/sounds/. satellite/sounds/
"""
import array
import io
import json
import os
import sys
import wave
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rocky_voice  # noqa: E402

TTS_URL = os.getenv("THURSDAY_TTS_URL", "http://127.0.0.1:9001")
LINES = {
    "offline": "Mac does not answer. Bad, bad, bad. Wait one minute, then try again.",
    "error": "Something breaks on my side. Try again, please.",
    "mic-off": "Microphone off. I do not listen now.",
    "mic-on": "Listening again. Good, good.",
}
# Short, never a question: they must not sound like he's taking the floor.
BACKCHANNELS = {"bc-mm": "Mm.", "bc-right": "Yes.", "bc-go-on": "Go on.", "bc-copy": "Understand."}
# Tara's takes (thursday-tara-*.wav; the Pi plays them while she's the voice): plain English, her own voice,
# no Rocky processing. Render only these with: make_sat_lines.py <out_dir> tara
TARA_LINES = {
    "offline": "Hm, I can't reach the Mac right now. Give it a minute and try me again?",
    "error": "Sorry, something went wrong on my end. Try that again?",
    "mic-off": "Okay, mic's off. I'm not listening.",
    "mic-on": "I'm back, listening.",
}
TARA_BACKCHANNELS = {"bc-mm": "Mm-hm.", "bc-right": "Right.", "bc-go-on": "Go on.", "bc-copy": "Okay."}
BC_TAKES = 3      # takes vary a lot in length and level; keep the median-length one
BC_PEAK = 12000   # every clip at the same level (the Pi plays them at 0.7 of that)
BC_TAIL_MS = 60   # trailing silence kept: the server stops listening while a clip plays


def _speak(text: str, voice: str = "leo") -> bytes:
    body = {"text": text} if voice == "leo" else {"text": text, "voice": voice}
    req = urllib.request.Request(f"{TTS_URL}/speak/file", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as res:
        wav = res.read()
    return rocky_voice.process_wav(wav, text) if voice == "leo" else wav


def _clip(wav: bytes) -> tuple[float, bytes]:
    """(voiced seconds, the take trimmed to its voice + a short tail, peak-normalized, faded out)."""
    with wave.open(io.BytesIO(wav)) as w:
        rate, pcm = w.getframerate(), array.array("h", w.readframes(w.getnframes()))
    loud = [i for i, v in enumerate(pcm) if abs(v) > 600] or [0, len(pcm) - 1]
    start, end = max(0, loud[0] - rate // 100), min(len(pcm), loud[-1] + rate * BC_TAIL_MS // 1000)
    pcm = pcm[start:end]
    scale = BC_PEAK / max(1, max(abs(v) for v in pcm))
    fade = rate // 100  # 10 ms each end: no click
    out = array.array("h", (int(v * scale * min(1, i / fade, (len(pcm) - i) / fade)) for i, v in enumerate(pcm)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(out.tobytes())
    return (loud[-1] - loud[0]) / rate, buf.getvalue()


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "satellite" / "sounds"
    out_dir.mkdir(parents=True, exist_ok=True)
    voice = sys.argv[2] if len(sys.argv) > 2 else "leo"
    lines, bcs, prefix = (TARA_LINES, TARA_BACKCHANNELS, "thursday-tara-") if voice == "tara" else (LINES, BACKCHANNELS, "thursday-")
    for name, text in lines.items():
        wav = _speak(text, voice)
        (out_dir / f"{prefix}{name}.wav").write_bytes(wav)
        print(f"wrote {prefix}{name}.wav ({len(wav) // 1024} KB): {text}")
    for name, text in bcs.items():
        takes = sorted(_clip(_speak(text, voice)) for _ in range(BC_TAKES))
        voiced, wav = takes[len(takes) // 2]
        (out_dir / f"{prefix}{name}.wav").write_bytes(wav)
        print(f"wrote {prefix}{name}.wav ({voiced:.2f} s voiced): {text}")


if __name__ == "__main__":
    main()
