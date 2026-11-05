"""Echo cancellation for the satellite (full duplex, slice 1). The HAT has none, so the
mic hears THURSDAY as loudly as the user. Uses Speex's echo canceller from the system
libspeexdsp (Debian's libspeexdsp1) through ctypes: nothing to pip install.

The canceller needs the reference (what was played) lined up with the echo in the mic:

  Player --(PCM it hands aplay, + when it should reach the speaker)--> Reference
  mic chunk --(the reference for the same 80 ms, by the mic's sample clock)--> Canceller

The Player runs one aplay with its pipeline always full, so the time from write to speaker
is constant; the filter is long enough for what's left, and each playback logs the lag it found.
"""
import ctypes
import ctypes.util
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

RATE = 16000
FRAME = 160  # 10 ms: Speex works frame by frame
_ECHO_SET_SAMPLING_RATE = 24
_PP_SET_DENOISE = 0
_PP_SET_ECHO_STATE = 24

log = logging.getLogger("satellite.aec")
DUMP = os.getenv("THURSDAY_AEC_DUMP", "")  # a folder: each playback's mic + reference saved there (diagnosis)


def _load_core() -> ctypes.CDLL | None:
    so = Path(__file__).resolve().parent / "_sat_core.so"
    if not so.exists():
        return None
    try:
        c = ctypes.CDLL(str(so))
        vp, i32, i64, f32, f64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_float, ctypes.c_double
        c.sat_boost_pcm.argtypes = [vp, vp, i32, f32, vp]
        c.sat_dbfs.argtypes = [vp, i32]
        c.sat_dbfs.restype = f64
        c.sat_ref_create.restype = vp
        c.sat_ref_destroy.argtypes = [vp]
        c.sat_ref_mic_arrived.argtypes = [vp, i32, f64]
        c.sat_ref_mic_arrived.restype = i64
        c.sat_ref_index_at.argtypes = [vp, f64, vp]
        c.sat_ref_index_at.restype = i64
        c.sat_ref_add_at.argtypes = [vp, vp, i32, i32, i64]
        c.sat_ref_add_time.argtypes = [vp, vp, i32, i32, f64]
        c.sat_ref_cut.argtypes = [vp, f64]
        c.sat_ref_read.argtypes = [vp, i64, i32, vp]
        c.sat_aec_create.argtypes = [vp, i32]
        c.sat_aec_create.restype = vp
        c.sat_aec_destroy.argtypes = [vp]
        c.sat_aec_process.argtypes = [vp, vp, vp, vp, i32, f64, vp]
        c.sat_aec_process.restype = i32
        return c
    except OSError:
        return None


_CORE = _load_core()


def _lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(ctypes.util.find_library("speexdsp") or "libspeexdsp.so.1")
    vp, i = ctypes.c_void_p, ctypes.c_int
    lib.speex_echo_state_init.restype = vp
    lib.speex_echo_state_init.argtypes = [i, i]
    lib.speex_echo_cancellation.argtypes = [vp, vp, vp, vp]
    lib.speex_echo_ctl.argtypes = [vp, i, vp]
    lib.speex_preprocess_state_init.restype = vp
    lib.speex_preprocess_state_init.argtypes = [i, i]
    lib.speex_preprocess_ctl.argtypes = [vp, i, vp]
    lib.speex_preprocess_run.argtypes = [vp, vp]
    return lib


def _to_16k(pcm: bytes, rate: int) -> np.ndarray:
    x = np.frombuffer(pcm, dtype=np.int16)
    if rate == RATE or not len(x):
        return x
    n = int(len(x) * RATE / rate)
    return np.interp(np.arange(n) * (rate / RATE), np.arange(len(x)), x).astype(np.int16)


def _dbfs(x: np.ndarray) -> float:
    return 10 * np.log10(float(np.mean(np.square(x.astype(np.float32)))) / 32768 ** 2 + 1e-12)


class Reference:
    """What the speaker is playing, on the mic's sample clock. A ring of RING_S seconds:
    the Player writes ahead of time (aplay buffers), the mic reads and clears behind. The Player
    writes from its feeder thread, the mic reads on the event loop: one lock for both."""

    RING_S = 30

    def __init__(self) -> None:
        self._c = _CORE.sat_ref_create() if _CORE else None
        self._ring = np.zeros(self.RING_S * RATE, dtype=np.int16) if not self._c else None
        self._mic_n = 0  # mic samples read so far
        self._anchors: deque = deque(maxlen=64)  # arrival time - samples/RATE, last ~5 s
        self.t0 = None   # monotonic time of mic sample 0 (the smallest anchor: least delayed)
        self.active_until = 0  # mic index after which nothing more has been scheduled
        self._lock = threading.Lock()

    def __del__(self) -> None:
        if getattr(self, "_c", None) and _CORE:
            _CORE.sat_ref_destroy(self._c)
            self._c = None

    def mic_arrived(self, samples: int) -> int:
        """The mic delivered `samples` more; returns the index of the chunk's first sample."""
        if self._c:
            return int(_CORE.sat_ref_mic_arrived(self._c, samples, time.monotonic()))
        start = self._mic_n
        self._mic_n += samples
        self._anchors.append(time.monotonic() - self._mic_n / RATE)
        self.t0 = min(self._anchors)  # the clocks drift ~50 ppm; the window follows it
        return start

    def _index(self, t: float) -> int:
        return round((t - self.t0) * RATE)

    def add(self, pcm: bytes, rate: int, at: float) -> None:
        """`pcm` (int16 mono at `rate`) reaches the speaker at monotonic time `at`."""
        if self._c:
            _CORE.sat_ref_add_time(self._c, pcm, len(pcm) // 2, rate, at)
            return
        if self.t0 is None:
            return
        x = _to_16k(pcm, rate)
        with self._lock:
            self._add(x, self._index(at))

    def add_at(self, pcm: bytes, rate: int, index: int) -> None:
        """`pcm` reaches the speaker at mic sample `index`: placed by sample count, not by the clock,
        since the HAT's mic and speaker share one codec clock (no drift, no jitter)."""
        if self._c:
            _CORE.sat_ref_add_at(self._c, pcm, len(pcm) // 2, rate, index)
            return
        x = _to_16k(pcm, rate)
        with self._lock:
            self._add(x, index)

    def index_at(self, t: float) -> int | None:
        """The mic sample index for monotonic time `t` (None until the mic has started)."""
        if self._c:
            ok = ctypes.c_int(0)
            idx = int(_CORE.sat_ref_index_at(self._c, t, ctypes.byref(ok)))
            return idx if ok.value else None
        return None if self.t0 is None else self._index(t)

    def _add(self, x: np.ndarray, i: int) -> None:
        n = len(self._ring)
        if i < self._mic_n:  # already in the past for the mic: drop the late part
            x, i = x[self._mic_n - i:], self._mic_n
        x = x[: n - RATE]  # never wrap onto what the mic hasn't read
        j = i % n
        k = min(len(x), n - j)
        self._ring[j:j + k] = x[:k]
        self._ring[:len(x) - k] = x[k:]
        self.active_until = max(self.active_until, i + len(x))

    def cut(self, at: float) -> None:
        """Playback stopped at `at`: clear everything scheduled after it."""
        if self._c:
            _CORE.sat_ref_cut(self._c, at)
            return
        if self.t0 is None:
            return
        with self._lock:
            self._cut(max(self._index(at), self._mic_n))

    def _cut(self, i: int) -> None:
        if i < self.active_until:
            idx = np.arange(i, self.active_until) % len(self._ring)
            self._ring[idx] = 0
        self.active_until = min(self.active_until, i)

    def read(self, start: int, n: int) -> np.ndarray:
        if self._c:
            out = np.empty(n, dtype=np.int16)
            _CORE.sat_ref_read(self._c, start, n, out.ctypes.data)
            return out
        idx = np.arange(start, start + n) % len(self._ring)
        with self._lock:
            out = self._ring[idx].copy()
            self._ring[idx] = 0
        return out


class Canceller:
    """Speex AEC + its residual echo suppressor, on 80 ms mic chunks. Runs only while
    something is playing (plus the filter's tail); between, the mic passes through."""

    def __init__(self, ref: Reference, filter_ms: int = 300) -> None:
        self.ref = ref
        self._c = _CORE.sat_aec_create(ref._c, filter_ms) if (_CORE and getattr(ref, "_c", None)) else None
        if self._c:
            self.tail = filter_ms * RATE // 1000
            self._ep: dict | None = None
            return
        self._lib = lib = _lib()
        self.tail = filter_ms * RATE // 1000
        self._echo = lib.speex_echo_state_init(FRAME, self.tail)
        rate = ctypes.c_int(RATE)
        lib.speex_echo_ctl(self._echo, _ECHO_SET_SAMPLING_RATE, ctypes.byref(rate))
        self._pp = lib.speex_preprocess_state_init(FRAME, RATE)
        off = ctypes.c_int(0)
        lib.speex_preprocess_ctl(self._pp, _PP_SET_DENOISE, ctypes.byref(off))  # echo only: STT does its own
        lib.speex_preprocess_ctl(self._pp, _PP_SET_ECHO_STATE, ctypes.c_void_p(self._echo))
        self._out = np.zeros(FRAME, dtype=np.int16)
        self._ep: dict | None = None  # stats for the current playback, logged when it ends

    def __del__(self) -> None:
        if getattr(self, "_c", None) and _CORE:
            _CORE.sat_aec_destroy(self._c)
            self._c = None

    def process(self, chunk: bytes) -> bytes:
        if self._c:
            n = len(chunk) // 2
            out = np.empty(n, dtype=np.int16)
            ref = np.empty(n, dtype=np.int16)
            stats = (ctypes.c_double * 5)()
            cpu = time.process_time()
            active = _CORE.sat_aec_process(self._c, chunk, out.ctypes.data, ref.ctypes.data,
                                           n, time.monotonic(), stats)
            if not active:
                if self._ep:
                    self._log_episode()
                return chunk
            ep = self._ep or {"chunks": 0, "mic": 0.0, "out": 0.0, "cpu": 0.0, "mics": [], "refs": [], "levels": []}
            self._ep = ep
            ep["chunks"] += 1
            ep["cpu"] += time.process_time() - cpu
            if stats[0] > 0.5:
                ep["mic"] += float(stats[1])
                ep["out"] += float(stats[2])
                ep["levels"].append((float(stats[3]), float(stats[4])))
            if len(ep["mics"]) < (150 if DUMP else 25):
                ep["mics"].append(np.frombuffer(chunk, dtype=np.int16).copy())
                ep["refs"].append(ref)
            return out.tobytes()
        mic = np.frombuffer(chunk, dtype=np.int16)
        start = self.ref.mic_arrived(len(mic))
        if start > self.ref.active_until + self.tail:
            if self._ep:
                self._log_episode()
            return chunk
        ref = self.ref.read(start, len(mic))
        cpu = time.process_time()
        out = np.empty_like(mic)
        mic_c, ref_c = np.ascontiguousarray(mic), np.ascontiguousarray(ref)
        for f in range(0, len(mic) - FRAME + 1, FRAME):
            self._lib.speex_echo_cancellation(self._echo, mic_c[f:].ctypes.data, ref_c[f:].ctypes.data,
                                              self._out.ctypes.data)
            self._lib.speex_preprocess_run(self._pp, self._out.ctypes.data)
            out[f:f + FRAME] = self._out
        ep = self._ep or {"chunks": 0, "mic": 0.0, "out": 0.0, "cpu": 0.0, "mics": [], "refs": [], "levels": []}
        self._ep = ep
        ep["chunks"] += 1
        ep["cpu"] += time.process_time() - cpu
        if ref.any():
            ep["mic"] += float(np.square(mic.astype(np.float32)).sum())
            ep["out"] += float(np.square(out.astype(np.float32)).sum())
            ep["levels"].append((_dbfs(mic), _dbfs(out)))
        if len(ep["mics"]) < (150 if DUMP else 25):  # the first 2 s: enough to find the lag
            ep["mics"].append(mic.copy())
            ep["refs"].append(ref)
        return out.tobytes()

    def _log_episode(self) -> None:
        ep, self._ep = self._ep, None
        secs = ep["chunks"] * 0.08
        lag = self._lag(np.concatenate(ep["mics"])[:2 * RATE], np.concatenate(ep["refs"])[:2 * RATE])
        if DUMP and secs > 3:
            np.savez(os.path.join(DUMP, f"aec_{int(time.time())}.npz"),
                     mic=np.concatenate(ep["mics"]), ref=np.concatenate(ep["refs"]))
        db = lambda e: 10 * np.log10(max(e, 1.0) / max(1, ep["chunks"] * 1280) / 32768 ** 2)
        # The loud parts too (90th percentile of 80 ms chunks): speech barge-in has to clear them.
        p90 = np.percentile(np.array(ep["levels"]), 90, axis=0) if ep["levels"] else (0.0, 0.0)
        log.info("aec: %.1f s played, mic %.0f dBFS -> %.0f dBFS (%.0f dB removed; loud parts %.0f -> %.0f), "
                 "echo lag %s, cpu %.0f%% of a core",
                 secs, db(ep["mic"]), db(ep["out"]), db(ep["mic"]) - db(ep["out"]), p90[0], p90[1],
                 f"{lag} ms" if lag is not None else "unclear", 100 * ep["cpu"] / max(secs, 0.01))

    @staticmethod
    def _lag(mic: np.ndarray, ref: np.ndarray, max_ms: int = 600) -> int | None:
        """How much later the echo reached the mic than the reference said, by cross-correlation.
        Positive: the reference is early (fine up to the filter length). None: too quiet to tell."""
        if not ref.any() or len(mic) < RATE // 2:
            return None
        m, r = mic.astype(np.float32), ref.astype(np.float32)
        n = 1 << int(np.ceil(np.log2(len(m) + len(r))))
        xc = np.fft.irfft(np.fft.rfft(m, n) * np.conj(np.fft.rfft(r, n)), n)
        span = max_ms * RATE // 1000
        cand = np.concatenate([xc[-span:], xc[:span]])  # lags -span..span-1
        best = int(np.argmax(np.abs(cand)))
        if np.abs(cand[best]) < 4 * np.median(np.abs(cand)):
            return None
        return (best - span) * 1000 // RATE
