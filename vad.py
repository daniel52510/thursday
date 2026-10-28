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


@dataclass
class VADConfig:
    start_prob: float = 0.5        # frame counts as speech at or above this
    end_prob: float = 0.35         # frame counts as silence below this (hysteresis)
    min_speech_ms: int = 96        # this much consecutive speech starts an utterance
    end_silence_ms: int = 600      # this much silence ends it; adds directly to latency
    preroll_ms: int = 300          # audio kept from before the start so the first word isn't clipped
    max_utterance_ms: int = 30000

    @classmethod
    def from_env(cls) -> "VADConfig":
        return cls(
            start_prob=_env_float("VAD_START_PROB", cls.start_prob),
            end_prob=_env_float("VAD_END_PROB", cls.end_prob),
            min_speech_ms=int(_env_float("VAD_MIN_SPEECH_MS", cls.min_speech_ms)),
            end_silence_ms=int(_env_float("VAD_END_SILENCE_MS", cls.end_silence_ms)),
            preroll_ms=int(_env_float("VAD_PREROLL_MS", cls.preroll_ms)),
            max_utterance_ms=int(_env_float("VAD_MAX_UTTERANCE_MS", cls.max_utterance_ms)),
        )


def _frames(ms: int) -> int:
    return max(1, round(ms / FRAME_MS))


class Endpointer:
    """Feed 32 ms frames; get back the utterance's PCM once speech has ended."""

    def __init__(self, cfg: VADConfig):
        self.cfg = cfg
        self.vad = SileroVAD()
        self._start_frames = _frames(cfg.min_speech_ms)
        self._end_frames = _frames(cfg.end_silence_ms)
        self._max_frames = _frames(cfg.max_utterance_ms)
        self._preroll: deque[bytes] = deque(maxlen=_frames(cfg.preroll_ms) + self._start_frames)
        self.reset()

    def reset(self) -> None:
        self.vad.reset()
        self._preroll.clear()
        self._frames_in: list[bytes] = []
        self._speech_run = 0
        self._silence_run = 0
        self.in_speech = False

    def feed(self, frame: bytes) -> bytes | None:
        prob = self.vad(frame)
        if not self.in_speech:
            self._preroll.append(frame)
            self._speech_run = self._speech_run + 1 if prob >= self.cfg.start_prob else 0
            if self._speech_run >= self._start_frames:
                self.in_speech = True
                self._frames_in = list(self._preroll)
            return None

        self._frames_in.append(frame)
        self._silence_run = self._silence_run + 1 if prob < self.cfg.end_prob else 0
        if self._silence_run >= self._end_frames or len(self._frames_in) >= self._max_frames:
            # Drop most of the trailing silence: STT time scales with audio length.
            keep = len(self._frames_in) - max(0, self._silence_run - _frames(150))
            utterance = b"".join(self._frames_in[:keep])
            self.reset()
            return utterance
        return None
