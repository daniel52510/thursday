"""Who is talking: speaker recognition from a voiceprint, for ambient mode and for who may use what.

A pretrained speaker-embedding model (WeSpeaker ResNet34-LM, VoxCeleb, ONNX, CC-BY-4.0: "WeSpeaker,
Wang et al.") turns speech into a 256-number voiceprint. Nothing is trained here: a person is
*enrolled* (enroll()) by averaging voiceprints of their clips, after dropping clips that don't
sound like the rest (someone else's voice in the set), and each utterance is scored against the
enrolled ones by cosine similarity: over THRESHOLD is them, under LOW is nobody enrolled, between
is unsure. Thresholds come from scripts/enroll_voice.py on his own clips against other voices.

Voiceprints live in the vault (encrypted, per user, name "voiceprint"); no audio is kept. The model
runs on the CPU in ~20 ms for a 2 s utterance. Features are Kaldi-style 80-bin log mel filterbanks
(25 ms / 10 ms, Povey window, mean-normalized), as the model was trained on.
"""
from __future__ import annotations

import logging
import os
import threading

import numpy as np

MODEL = os.getenv("SPEAKER_MODEL", "/app/models/wespeaker_resnet34_lm.onnx")
SR = 16000
CREDENTIAL = "voiceprint"
# From his enrollment recording vs 15 other voices, at real request lengths (2026-11-02): him 5th pct 0.40
# (1.5 s) to 0.50 (3 s), others at most 0.24. Cautious, since real people in a room score higher than
# those clean synthetic voices; between LOW and THRESHOLD is "unsure". Re-run scripts/enroll_voice.py.
THRESHOLD = float(os.getenv("SPEAKER_THRESHOLD", "0.35"))
LOW = float(os.getenv("SPEAKER_LOW", "0.25"))
MIN_SECONDS = 1.0  # less speech than this ("skip") says too little about the voice: unsure
SPEECH_DB = 8.0    # a 20 ms frame this far over the clip's own noise floor is speech. The satellite mic is
                   # quiet (his voice -45..-50 dBFS, 10-15 dB over the room), so a fixed level threw speech away.
MIN_SNR_DB = 8.0   # less contrast than this is noise (near-silent clips scored like anyone, 2026-11-02)
OUTLIER = 0.25     # an enrollment clip this far under the median similarity is someone else's

logger = logging.getLogger("thursday.voice")
_session = None
_lock = threading.Lock()


def _mel_banks(n_fft: int = 512, n_mels: int = 80, low: float = 20.0, high: float = SR / 2) -> np.ndarray:
    """Kaldi's triangular mel filters (mel = 1127 ln(1 + f/700)) over the FFT bins."""
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)
    edges = np.linspace(mel(low), mel(high), n_mels + 2)
    bins = mel(np.arange(n_fft // 2) * SR / n_fft)
    banks = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = edges[m], edges[m + 1], edges[m + 2]
        up, down = (bins - left) / (center - left), (right - bins) / (right - center)
        banks[m, :n_fft // 2] = np.maximum(0.0, np.minimum(up, down))
    return banks


_BANKS = _mel_banks()
_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 399)) ** 0.85  # Povey


def fbank(pcm: np.ndarray) -> np.ndarray:
    """int16-scale 16 kHz mono -> (frames, 80) log mel filterbank, mean-normalized over time."""
    x = pcm.astype(np.float64)
    if len(x) < 400:
        x = np.pad(x, (0, 400 - len(x)))
    n = 1 + (len(x) - 400) // 160
    frames = x[np.arange(400)[None, :] + 160 * np.arange(n)[:, None]]
    frames = frames - frames.mean(axis=1, keepdims=True)                      # remove DC
    frames = np.concatenate([frames[:, :1] * 0.03, frames[:, 1:] - 0.97 * frames[:, :-1]], axis=1)  # pre-emphasis
    power = np.abs(np.fft.rfft(frames * _WINDOW, 512)) ** 2
    feats = np.log(np.maximum(power @ _BANKS.T, np.finfo(np.float32).eps)).astype(np.float32)
    return feats - feats.mean(axis=0, keepdims=True)


def _model():
    global _session
    with _lock:
        if _session is None:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2  # leave the cores to everything else
            _session = ort.InferenceSession(MODEL, opts, providers=["CPUExecutionProvider"])
        return _session


def speech_only(x: np.ndarray) -> np.ndarray | None:
    """The span with speech in it, trimmed by each 20 ms frame's level against the loudest (a fixed level
    cut quiet talking away); None when there's too little, or it's too quiet to be speech."""
    hop = SR // 50
    n = len(x) // hop
    if n < 5:
        return None
    rms = np.sqrt(np.mean(x[:n * hop].astype(np.float64).reshape(n, hop) ** 2, axis=1))
    db = 20 * np.log10(np.maximum(rms, 1.0))
    floor = np.percentile(db, 10)
    speech = np.flatnonzero(db > floor + SPEECH_DB)
    span = x[max(0, speech[0] - 5) * hop:min(n, speech[-1] + 6) * hop] if len(speech) else x[:0]
    # Natural speech is only ~half loud frames (pauses, soft consonants): 0.5 s of them across a 1 s span.
    if np.percentile(db, 99) - floor < MIN_SNR_DB or len(speech) * hop < SR * 0.5 or len(span) < SR * MIN_SECONDS:
        return None
    return span


def embed(pcm16: bytes | np.ndarray) -> np.ndarray:
    """A unit-length voiceprint for 16 kHz mono int16 speech."""
    x = np.frombuffer(pcm16, dtype="<i2") if isinstance(pcm16, (bytes, bytearray)) else pcm16
    v = _model().run(None, {"feats": fbank(x)[None]})[0][0]
    return v / (np.linalg.norm(v) + 1e-9)


def enroll(clips: list[np.ndarray]) -> tuple[np.ndarray, int]:
    """A voiceprint from someone's clips: (unit vector, clips used). Clips that don't match the rest
    (OUTLIER under the median similarity to the first average) are left out."""
    vs = np.stack([embed(c) for c in clips if len(c) >= SR * MIN_SECONDS])
    first = vs.mean(axis=0)
    sims = vs @ (first / np.linalg.norm(first))
    keep = vs[sims >= np.median(sims) - OUTLIER]
    v = keep.mean(axis=0)
    return v / np.linalg.norm(v), len(keep)


def save_voiceprint(user_id: str, v: np.ndarray, clips: int) -> None:
    import vault
    vault.shared().set(user_id, CREDENTIAL, {"v": [round(float(a), 6) for a in v], "clips": clips})


def voiceprints() -> dict[str, np.ndarray]:
    """Everyone enrolled: {user_id: voiceprint}."""
    import accounts
    import vault
    box, out = vault.shared(), {}
    if not box.available:
        return out
    for user in accounts.shared().list_users():
        if (rec := box.get(user["id"], CREDENTIAL)) and rec.get("v"):
            out[user["id"]] = np.asarray(rec["v"], dtype=np.float32)
    return out


def identify(pcm16: bytes, prints: dict[str, np.ndarray] | None = None) -> dict:
    """{"user": id or None, "score": best similarity, "verdict": "match" | "unsure" | "unknown"}."""
    prints = voiceprints() if prints is None else prints
    if not prints:
        return {"user": None, "score": 0.0, "verdict": "unknown"}
    x = speech_only(np.frombuffer(pcm16, dtype="<i2"))
    if x is None:  # too short or too quiet to judge the voice: the caller may carry an earlier match
        return {"user": None, "score": 0.0, "verdict": "unsure", "short": True}
    v = embed(x)
    user, score = max(((u, float(p @ v)) for u, p in prints.items()), key=lambda t: t[1])
    verdict = "match" if score >= THRESHOLD else "unknown" if score < LOW else "unsure"
    return {"user": user if verdict == "match" else None, "score": round(score, 3), "verdict": verdict}
