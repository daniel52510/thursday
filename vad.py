"""Silero VAD (ONNX) and an endpointer that turns per-frame speech
probabilities into finished utterances.

Audio everywhere here is 16 kHz mono int16 PCM, fed in 512-sample (32 ms)
frames, the only frame size Silero accepts at 16 kHz.
"""
import os
from collections import deque
from dataclasses import dataclass

import numpy as np
import onnxruntime

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
FRAME_BYTES = FRAME_SAMPLES * 2
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE
# Silero expects the last 64 samples of the previous frame prepended to each frame.
CONTEXT_SAMPLES = 64
MODEL_PATH = os.getenv("VAD_MODEL_PATH", "models/silero_vad.onnx")

_SESSION = None


def _get_session() -> onnxruntime.InferenceSession:
    # One shared session; per-connection state lives in SileroVAD, not here.
    global _SESSION
    if _SESSION is None:
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        _SESSION = onnxruntime.InferenceSession(
            MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"]
        )
    return _SESSION


class SileroVAD:
    def __init__(self):
        self._session = _get_session()
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    def __call__(self, frame: bytes) -> float:
        """Speech probability (0-1) for one 512-sample int16 frame."""
        pcm = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        x = np.concatenate([self._context, pcm.reshape(1, -1)], axis=1)
        out, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr}
        )
        self._context = x[:, -CONTEXT_SAMPLES:]
        return float(out[0][0])


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _frame_dbfs(frame: bytes) -> float:
    """RMS level in dBFS (-96..0) for one int16 frame."""
    x = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
    return float(10.0 * np.log10(np.mean(x * x) + 1e-10))


@dataclass
class VADConfig:
    start_prob: float = 0.6        # frame counts as speech at or above this
    end_prob: float = 0.35         # frame counts as silence below this (hysteresis)
    min_speech_ms: int = 160       # this much consecutive speech starts an utterance
    end_silence_ms: int = 600      # this much silence ends it; adds directly to latency
    preroll_ms: int = 300          # audio kept from before the start so the first word isn't clipped
    max_utterance_ms: int = 90000  # thinking out loud runs long; 30 s chopped him mid-thought (2026-11-10)
    min_dbfs: float = -48.0        # quiet bleed / room noise below this cannot start an utterance

    @classmethod
    def from_env(cls) -> "VADConfig":
        return cls(
            start_prob=_env_float("VAD_START_PROB", cls.start_prob),
            end_prob=_env_float("VAD_END_PROB", cls.end_prob),
            min_speech_ms=int(_env_float("VAD_MIN_SPEECH_MS", cls.min_speech_ms)),
            end_silence_ms=int(_env_float("VAD_END_SILENCE_MS", cls.end_silence_ms)),
            preroll_ms=int(_env_float("VAD_PREROLL_MS", cls.preroll_ms)),
            max_utterance_ms=int(_env_float("VAD_MAX_UTTERANCE_MS", cls.max_utterance_ms)),
            min_dbfs=_env_float("VAD_MIN_DBFS", cls.min_dbfs),
        )


def _frames(ms: int) -> int:
    return max(1, round(ms / FRAME_MS))


class BargeIn:
    """Watches the mic while THURSDAY thinks or speaks. Stricter than the
    endpointer's start: her own voice leaking past echo cancellation, a cough
    or a quick "mm" must not cut her off; a real interjection must.

    Silero VAD is scale-invariant (faint residual speaker bleed still scores >0.9),
    so barge-in also tracks the mic's residual echo floor in dBFS and requires the
    user's voice to rise BARGE_SPEECH_DB above it and above BARGE_MIN_DBFS.
    """

    def __init__(self, cfg: VADConfig):
        self.vad = SileroVAD()
        self.prob = _env_float("BARGE_PROB", 0.85)
        self._need = _frames(int(_env_float("BARGE_MS", 352)))
        self.min_dbfs = _env_float("BARGE_MIN_DBFS", -40.0)
        self.margin_db = _env_float("BARGE_SPEECH_DB", 8.0)
        # The start of each reply only measures her echo, never counts as him: with no echo floor learned yet the bar
        # sat at BARGE_MIN_DBFS, and on a phone's loudspeaker her own first words cleared it (cut off 6 words in,
        # 2026-11-06). The satellite has the same settle (satellite.py BARGE_SETTLE_S).
        self._settle_frames = _frames(int(_env_float("BARGE_SETTLE_MS", 600)))
        # Everything since just before the speech began, handed to the endpointer on a trigger.
        self._recent: deque[bytes] = deque(maxlen=_frames(cfg.preroll_ms) + self._need)
        self._residual: deque[float] = deque(maxlen=40)
        self.reset()

    def reset(self) -> None:
        self.vad.reset()
        self._recent.clear()
        self._residual.clear()
        self._run = 0
        self._miss = 0
        self._settle = self._settle_frames

    def feed(self, frame: bytes) -> list[bytes] | None:
        """The buffered speech once the user has clearly started talking."""
        self._recent.append(frame)
        db = _frame_dbfs(frame)
        if self._settle > 0:
            self._settle -= 1
            self._residual.append(db)
            return None
        bar = max(
            self.min_dbfs,
            float(np.percentile(self._residual, 85)) + self.margin_db if len(self._residual) >= 5 else self.min_dbfs,
        )
        if db >= bar and self.vad(frame) >= self.prob:
            self._run += 1
            self._miss = 0
        else:
            if db < bar:
                self._residual.append(db)
            if self._run:
                # One weak frame inside a word doesn't restart the count; two do.
                self._miss += 1
                if self._miss > 1:
                    self._run = self._miss = 0
        if self._run >= self._need:
            frames = list(self._recent)
            self.reset()
            self.trigger_bar = bar
            return frames
        return None

    trigger_bar: float = -40.0

    def probe(self, frame: bytes) -> bool:
        """After a trigger: is he still talking in this frame? Judged against the echo bar at the trigger (a reset
        would spend 600 ms relearning it), with the speech score a little lower, since he's mid-word now."""
        return _frame_dbfs(frame) >= self.trigger_bar and self.vad(frame) >= self.prob * 0.8


class Endpointer:
    """Feed 32 ms frames; get back the utterance's PCM once speech has ended."""

    def __init__(self, cfg: VADConfig):
        self.cfg = cfg
        self.vad = SileroVAD()
        self._start_frames = _frames(cfg.min_speech_ms)
        self._end_frames = _frames(cfg.end_silence_ms)
        self._max_frames = _frames(cfg.max_utterance_ms)
        self._preroll: deque[bytes] = deque(maxlen=_frames(cfg.preroll_ms) + self._start_frames)
        self.last_voiced = 0  # voiced count of the last finished utterance (reset() clears voiced)
        self.reset()

    def reset(self) -> None:
        self.vad.reset()
        self._preroll.clear()
        self._frames_in: list[bytes] = []
        self._speech_run = 0
        self._silence_run = 0
        self.end_frames = self._end_frames  # this pause's endpoint; voice_ws's smart endpoint moves it
        self.voiced = 0  # speech frames in the current utterance; unchanged across a pause
        self.in_speech = False

    @property
    def silence_frames(self) -> int:
        return self._silence_run

    def snapshot(self) -> bytes:
        """The utterance so far, trimmed exactly as feed() trims a finished one,
        so audio snapshotted during a pause equals the final utterance if no
        speech follows."""
        # Drop most of the trailing silence: STT time scales with audio length.
        keep = len(self._frames_in) - max(0, self._silence_run - _frames(150))
        return b"".join(self._frames_in[:keep])

    def feed(self, frame: bytes) -> bytes | None:
        prob = self.vad(frame)
        if not self.in_speech:
            self._preroll.append(frame)
            loud_enough = _frame_dbfs(frame) >= self.cfg.min_dbfs
            self._speech_run = self._speech_run + 1 if (loud_enough and prob >= self.cfg.start_prob) else 0
            if self._speech_run >= self._start_frames:
                self.in_speech = True
                self._frames_in = list(self._preroll)
            return None

        self._frames_in.append(frame)
        if prob < self.cfg.end_prob:
            self._silence_run += 1
        else:
            self._silence_run = 0
            self.voiced += 1
            self.end_frames = self._end_frames  # speech again: the last pause's decision is void
        if self._silence_run >= self.end_frames or len(self._frames_in) >= self._max_frames:
            utterance = self.snapshot()
            self.last_voiced = self.voiced
            self.reset()
            return utterance
        return None
