"""THURSDAY Pi satellite: waits for "Hey THURSDAY", streams the question to
the Mac's /ws/voice endpoint, and plays the spoken reply on the HAT speaker.

    wake --"Hey THURSDAY"--> listening --server "thinking"--> thinking --WAV--> speaking
      ^                          |                               |                |
      +------ server "idle" -----+-------- error / idle ---------+-- playback end -+
    "Hey THURSDAY" (or the button) while thinking/speaking: barge-in, back to listening.
    Talking over a reply (THURSDAY_BARGE_SPEECH=1, with THURSDAY_AEC=on) does the same.

Audio in: arecord, 16 kHz mono int16, 1280-sample (80 ms) chunks, the step
size openWakeWord wants; /ws/voice re-slices whatever it gets. One mic loop
runs for the life of the process; connections come and go around it, so the
wake word still works (and she says so) while the Mac is unreachable.
Audio out: one aplay for the life of the process (Player), fed silence between
sounds, so every sound reaches the speaker a fixed time after it's written (what
the echo canceller needs) and chunks, cues and loops join without gaps.
Button (GPIO17 on the ReSpeaker 2-Mic HAT, via gpiomon): press to talk or
interrupt, hold to mute the mic.

    THURSDAY_WS=ws://<mac-ip>:8000/ws/voice venv/bin/python satellite.py
"""
import asyncio
import ctypes
import fcntl
import io
import math
import json
import logging
import os
import struct
import subprocess
import sys
import threading
import time
import types
import wave
from collections import deque
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

# openwakeword/__init__.py imports its verifier-training module, which pulls in
# scipy and scikit-learn (100+ MB, slow import on a Zero 2W). We never train
# here, so stand in an empty module before the import.
_stub = types.ModuleType("openwakeword.custom_verifier_model")
_stub.train_custom_verifier = None
sys.modules["openwakeword.custom_verifier_model"] = _stub

from openwakeword.model import Model  # noqa: E402

HERE = Path(__file__).resolve().parent


def _load_core() -> ctypes.CDLL | None:
    so = HERE / "_sat_core.so"
    if not so.exists():
        return None
    try:
        c = ctypes.CDLL(str(so))
        vp, i32, f32, f64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_double
        c.sat_boost_pcm.argtypes = [vp, vp, i32, f32, vp]
        c.sat_dbfs.argtypes = [vp, i32]
        c.sat_dbfs.restype = f64
        c.sat_resample_scale.argtypes = [vp, i32, i32, vp, i32, i32, f32]
        c.sat_resample_scale.restype = i32
        c.sat_soften_block.argtypes = [vp, i32, i32, i32, i32]
        c.sat_mix_music_block.argtypes = [vp, vp, i32, vp, i32, i32, f32, f32]
        c.sat_mix_music_block.restype = f32
        c.sat_wake_create.argtypes = [i32]
        c.sat_wake_create.restype = vp
        c.sat_wake_destroy.argtypes = [vp]
        c.sat_wake_set_init_features.argtypes = [vp, vp]
        c.sat_wake_reset.argtypes = [vp]
        c.sat_wake_prepare_melspec.argtypes = [vp, vp]
        c.sat_wake_prepare_melspec.restype = ctypes.POINTER(f32)
        c.sat_wake_push_mels.argtypes = [vp, vp, i32]
        c.sat_wake_push_mels.restype = ctypes.POINTER(f32)
        c.sat_wake_push_emb.argtypes = [vp, vp, vp]
        c.sat_wake_push_emb.restype = ctypes.POINTER(f32)
        return c
    except OSError:
        return None


_CORE = _load_core()
_HP_STATE = (ctypes.c_float * 2)(0.0, 0.0)
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280
CHUNK_BYTES = CHUNK_SAMPLES * 2
SILENCE = bytes(CHUNK_BYTES)
DEVICE = os.getenv("THURSDAY_ALSA_DEVICE", "plughw:CARD=wm8960soundcard")
SERVER_URL = os.getenv("THURSDAY_WS", "")
TOKEN_FILE = Path(os.getenv("THURSDAY_TOKEN_FILE", "~/.config/thursday/token")).expanduser()
WAKE_MODEL = os.getenv("THURSDAY_WAKE_MODEL", str(HERE / "models" / "hey_thursday.onnx"))
# 0.5 woke reliably only for Daniel; others likely land just under it. Lowered together with
# the VAD gate and noise suppression below, which keep TV/music false wakes down.
WAKE_THRESHOLD = float(os.getenv("THURSDAY_WAKE_THRESHOLD", "0.4"))
# openWakeWord's Silero VAD gate: wake scores only count while someone is actually speaking.
# Off by default: on this mic, far speech is quiet enough that Silero stays unsure and the
# gate zeroed real wakes (0.00 gated vs 0.97 ungated on the same clip). False wakes are
# handled by the retrained model's threshold instead.
WAKE_VAD = float(os.getenv("THURSDAY_WAKE_VAD", "0"))  # 0 = off
# Digital gain on the mic, before the wake word and the server. The WM8960 captures
# far speech quietly, which scores low; raise this if wakes from across the room miss.
MIC_GAIN = float(os.getenv("THURSDAY_MIC_GAIN", "2.0"))
# Echo cancellation (aec.py, full duplex): "shadow" cancels and logs how well it works per playback
# without changing what's sent; "on" also uses the cancelled mic while THURSDAY talks. Off by default.
AEC_MODE = os.getenv("THURSDAY_AEC", "off")
# Where the reference sits against the echo: the log's "echo lag" tunes it. Measured 2026-11-01 with the
# steady Player: echo 86 ms after the reference; +65 ms leaves ~20 ms, and a 120 ms filter then cancels
# down to the room's noise floor (best of delay 0-75 ms x filter 60-300 ms, offline on a real reply).
AEC_DELAY_S = float(os.getenv("THURSDAY_AEC_DELAY_MS", "65")) / 1000
AEC_FILTER_MS = int(os.getenv("THURSDAY_AEC_FILTER_MS", "120"))
WAKE_SPEEX = os.getenv("THURSDAY_WAKE_SPEEX", "1") == "1"  # Speex noise suppression (needs speexdsp_ns)
# Barge-in listens for the wake word over her own voice from the speaker next
# to the mics, so it needs a higher bar than waking from quiet. 0 turns it off.
BARGE_THRESHOLD = float(os.getenv("THURSDAY_BARGE_THRESHOLD", "0.7"))
# Speech barge-in: talking over THURSDAY cuts him off, no wake word. Needs THURSDAY_AEC=on: it
# listens to the echo-cancelled mic for sound well above the echo the canceller leaves behind.
BARGE_SPEECH = os.getenv("THURSDAY_BARGE_SPEECH", "0") == "1"
# Above the loud parts of the echo left after cancelling: those sit ~-61 dBFS, his voice at the mic -43 to -46.
BARGE_SPEECH_DB = float(os.getenv("THURSDAY_BARGE_SPEECH_DB", "8"))
# And never below the raw echo: at -58 the canceller's brief slips (echo -38 to -42 uncancelled) cut him
# off mid-reply; the false barge-ins logged 2026-11-03..05 all sat 0-3 dB over a bar near -42..-48.
BARGE_FLOOR_DBFS = float(os.getenv("THURSDAY_BARGE_FLOOR_DBFS", "-44"))
# How many of the last BARGE_WINDOW chunks (~80 ms each) must be loud: 4 of 6 is ~320 ms of talking.
BARGE_LOUD_CHUNKS = int(os.getenv("THURSDAY_BARGE_LOUD_CHUNKS", "4"))
BARGE_WINDOW = int(os.getenv("THURSDAY_BARGE_WINDOW", "6"))
BARGE_SETTLE_S = 0.5  # the start of each reply: the canceller re-settles, nothing counts
NEAR_MISS = 0.2  # wake scores logged below the threshold, to tune it from data
# After a reply, listen again without the wake word. The server's no-speech
# timeout (VOICE_NO_SPEECH_TIMEOUT_MS) ends the conversation if nobody talks.
FOLLOW_UP = os.getenv("THURSDAY_FOLLOW_UP", "1") == "1"
ROOM = os.getenv("THURSDAY_ROOM", "Satellite")
# Night mode: "22-7" plays everything but the alarm at NIGHT_GAIN between 22:00 and 07:00.
NIGHT = os.getenv("THURSDAY_NIGHT", "22-7")
NIGHT_GAIN = float(os.getenv("THURSDAY_NIGHT_GAIN", "0.4"))
BUTTON_CHIP = os.getenv("THURSDAY_BUTTON_CHIP", "gpiochip0")
BUTTON_LINE = os.getenv("THURSDAY_BUTTON_LINE", "17")  # "" disables the button
LONG_PRESS_S = 1.2
# Sent with every "start" like the web page does, so "today" and calendar
# answers use the satellite's local day. Defaults to the Pi's system timezone.
CONTEXT = {
    "timezone": os.getenv("THURSDAY_TIMEZONE") or (Path("/etc/timezone").read_text().strip() if Path("/etc/timezone").exists() else ""),
    "location": os.getenv("THURSDAY_LOCATION", ""),
}
SOUNDS = HERE / "sounds"
PING_WAV = SOUNDS / "thursday-ping.wav"
THINKING_WAV = SOUNDS / "thursday-thinking.wav"
ALARM_WAV = SOUNDS / "thursday-alarm.wav"
CLOSE_WAV = SOUNDS / "thursday-close.wav"      # a follow-up window ended
OFFLINE_WAV = SOUNDS / "thursday-offline.wav"  # woken while the Mac is unreachable
ERROR_WAV = SOUNDS / "thursday-error.wav"      # a turn failed on the server
MIC_OFF_WAV = SOUNDS / "thursday-mic-off.wav"
MIC_ON_WAV = SOUNDS / "thursday-mic-on.wav"
BACKCHANNEL_CLIPS = {"mm", "right", "go-on", "copy"}


def voice_clip(path: Path, voice: str) -> Path:
    """Tara's take of a spoken clip (sounds/thursday-tara-*.wav) when she's the voice, else Rocky's."""
    if voice == "tara" and (tara := path.with_name(path.name.replace("thursday-", "thursday-tara-", 1))).exists():
        return tara
    return path  # sounds/thursday-bc-<clip>.wav, the server's "overlay"
RECONNECT_S = 3.0
STATUS_EVERY_S = 30.0
MIC_STALL_S = 10.0  # no audio from arecord this long: exit, and systemd restarts us
# The HAT's three APA102 LEDs, written straight to spidev (no Python package needed). "" disables.
LED_DEVICE = os.getenv("THURSDAY_LED_SPI", "/dev/spidev0.1")

log = logging.getLogger("satellite")


def _ms(since: float) -> int:
    return int((time.monotonic() - since) * 1000)


def _wav_pcm(wav: bytes) -> tuple[int, bytes]:
    """(sample rate, raw frames) of a mono 16-bit WAV, as the server sends them."""
    with wave.open(io.BytesIO(wav)) as w:
        return w.getframerate(), w.readframes(w.getnframes())


def night_gain() -> float:
    try:
        start, end = (int(h) for h in NIGHT.split("-"))
    except ValueError:
        return 1.0
    hour = datetime.now().hour
    at_night = start <= hour or hour < end if start > end else start <= hour < end
    return NIGHT_GAIN if at_night else 1.0


def _scaled(pcm: bytes, gain: float) -> bytes:
    if gain == 1.0 or not pcm:
        return pcm
    if _CORE:
        n = len(pcm) // 2
        out = ctypes.create_string_buffer(n * 2)
        _CORE.sat_resample_scale(pcm, n, 24000, out, n, 24000, ctypes.c_float(gain))
        return out.raw
    return (np.frombuffer(pcm, dtype=np.int16) * gain).astype(np.int16).tobytes()


def _boosted(pcm: bytes) -> bytes:
    """Mic audio with 80 Hz rumble cut and soft-knee MIC_GAIN applied."""
    if _CORE and pcm:
        n = len(pcm) // 2
        out = ctypes.create_string_buffer(n * 2)
        _CORE.sat_boost_pcm(pcm, out, n, ctypes.c_float(MIC_GAIN), ctypes.byref(_HP_STATE))
        return out.raw
    if MIC_GAIN == 1.0:
        return pcm
    boosted = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) * MIC_GAIN
    return np.clip(boosted, -32768, 32767).astype(np.int16).tobytes()


def _configure_alsa_mic() -> None:
    """Raise the WM8960 hardware capture gain and ADC gain on startup (+10.25 dB total over default)
    so far-field speech across the room has clean analog/ADC SNR without clipping on speaker playback."""
    card = "wm8960soundcard"
    capture = os.getenv("THURSDAY_ALSA_CAPTURE", "46")   # 0-63 (39=+12dB, 46=+17.25dB)
    adc = os.getenv("THURSDAY_ALSA_ADC", "205")          # 0-255 (195=0dB, 205=+5.0dB)
    boost = os.getenv("THURSDAY_ALSA_BOOST", "0")        # 0-3 (0=0dB: keeps speaker echo linear for AEC)
    controls = [
        ("Capture", f"{capture},{capture}"),
        ("ADC PCM", f"{adc},{adc}"),
        ("Left Input Boost Mixer LINPUT1", boost),
        ("Right Input Boost Mixer RINPUT1", boost),
    ]
    applied = []
    for name, val in controls:
        try:
            res = subprocess.run(["amixer", "-q", "-c", card, "sset", name, val],
                                 capture_output=True, timeout=2)
            if res.returncode == 0:
                applied.append(f"{name}={val}")
        except (OSError, subprocess.SubprocessError):
            return
    if applied:
        log.info("alsa mic gain configured: %s", ", ".join(applied))


def _wifi_dbm() -> int | None:
    try:
        line = Path("/proc/net/wireless").read_text().splitlines()[-1].split()
        return int(float(line[3]))
    except (OSError, IndexError, ValueError):
        return None


class Leds:
    """The HAT's 3 APA102 RGB LEDs, in the voice figure's colours (DESIGN.md):
    off while waiting for the wake word, blue listening, amber turning while
    thinking, red pulsing while speaking, dim red when muted, grey when the Mac
    is unreachable. Raw SPI frames: start frame, one 4-byte frame per LED, end frame."""

    COUNT = 3
    BLUE, AMBER, RED, ORANGE, GREY = (45, 107, 214), (233, 166, 42), (224, 60, 49), (240, 122, 43), (140, 135, 124)
    _SPI_IOC_WR_MAX_SPEED_HZ = 0x40046B04

    def __init__(self, device: str) -> None:
        self._fd = None
        self._last = b""
        if not device:
            return
        try:
            self._fd = os.open(device, os.O_WRONLY)
            fcntl.ioctl(self._fd, self._SPI_IOC_WR_MAX_SPEED_HZ, struct.pack("<I", 8_000_000))
        except OSError as exc:
            log.warning("LEDs unavailable (%s); continuing without them", exc)
            self._fd = None

    def show(self, colors: list[tuple[int, int, int]], brightness: int) -> None:
        if self._fd is None:
            return
        frame = b"\x00" * 4 + b"".join(bytes([0xE0 | brightness, b, g, r]) for r, g, b in colors) + b"\xff" * 4
        if frame != self._last:  # write only on change
            self._last = frame
            try:
                os.write(self._fd, frame)
            except OSError:
                pass

    def frame(self, state: str, muted: bool, online: bool, t: float) -> tuple[list, int]:
        """The colours for this moment, and a global brightness (0-31)."""
        off = (0, 0, 0)
        if muted:
            return [self.RED] * self.COUNT, 2
        if not online:
            return [self.GREY] * self.COUNT, 2
        if state in ("listening", "interrupting"):
            return [self.BLUE] * self.COUNT, 8
        if state == "thinking":  # one LED leads, turning every 150 ms
            lead = int(t / 0.15) % self.COUNT
            return [self.AMBER if i == lead else tuple(c // 6 for c in self.AMBER) for i in range(self.COUNT)], 8
        if state in ("speaking", "alarm"):
            k = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(t * math.tau / (1.4 if state == "speaking" else 0.9)))
            base = self.RED if state == "speaking" else self.ORANGE
            return [tuple(int(c * k) for c in base)] * self.COUNT, 8
        return [off] * self.COUNT, 0  # waiting for the wake word: dark


class Player:
    """One aplay for the life of the process, fed 20 ms blocks by a thread: sound when there is
    some, silence between. The pipeline (a 4 KB pipe, aplay's 100 ms ALSA buffer) stays full, so
    audio reaches the speaker a constant time after it's written, and the echo canceller knows
    when each sample played. (An aplay per sound started 50-300 ms late, and a different amount
    each time, so the canceller never locked on: 0-2 dB removed on real replies.)
    Starting a sound replaces the current one; stop() cuts it off at once.

    Music (YouTube Music from the Mac's tap, music()) is its own channel, mixed under everything
    else: it waits for MUSIC_PREROLL_S before it starts (and again after running dry), so Wi-Fi
    hiccups don't stutter it; it's ducked to MUSIC_DUCK while he talks or speaks; and it goes into
    the echo canceller's reference like any other sound, so the wake word works over it. It never
    counts as "playing": that's still only his voice and cues."""

    RATE = 24000                 # the replies' rate; other sounds are resampled to it
    BLOCK = 480                  # 20 ms
    BUFFER_US = 100_000          # aplay's ALSA buffer
    FADE = 96                    # 4 ms: the ramp at either side of a gap (_soften)
    MUSIC_PREROLL_S = 0.25       # buffered before music (re)starts
    MUSIC_MAX_S = 1.0            # more than this queued (clock drift): the oldest goes
    MUSIC_GAIN = float(os.getenv("THURSDAY_MUSIC_GAIN", "0.5"))
    MUSIC_DUCK = 0.15            # of MUSIC_GAIN while anyone talks or he speaks
    # Samples between the write and the speaker: pipe + ALSA buffer + aplay's period in hand.
    # Placed 40 ms early on purpose: an early reference is absorbed by the filter, a late one isn't.
    PIPELINE_S = (2048 + BUFFER_US * RATE // 1_000_000 + BLOCK) / RATE - 0.04

    def __init__(self) -> None:
        self._cache: dict[Path, bytes] = {}
        self._lock = threading.Lock()
        self._pending = bytearray()   # audio not yet written to aplay
        self._loop: bytes | None = None  # refills _pending when it runs dry (the thinking cue)
        self._proc = None
        self._written = 0             # samples written to this aplay
        self._anchors: deque = deque(maxlen=250)  # write time - samples/RATE, last ~5 s
        self._audible_until = 0.0     # when the last real (non-silent) block leaves the speaker
        self._dry = True              # the last block written ended in silence: the next one fades in
        self._music = bytearray()     # music not yet mixed in (music())
        self._music_live = False      # past the pre-roll: mixing
        self._mgain = 0.0             # music gain now, ramped toward its target block by block
        self.ducking = lambda: False  # set by the satellite: is anyone talking to him?
        self.voice_gain = 1.0         # his voice volume for the cues and clips (server "voice"); replies have it already
        self._ref_base: int | None = None  # mic sample index where this aplay's sample 0 plays
        self.gain = 1.0  # for the current reply stream
        self.rate = self.RATE  # of the current reply stream
        self.ref = None  # aec.Reference: everything played, for echo cancellation
        # Dropouts inside a reply (start_stream to finish_stream): each time the reply's audio ran out
        # and came back (a fade-out/fade-in dip, _soften), and how long it was silent. Logged per turn.
        self.streaming = False
        self._draining = False
        self._stream_heard = False
        self._silent_run = 0
        self.underruns = 0
        self.underrun_ms = 0
        threading.Thread(target=self._feed, name="player", daemon=True).start()

    # ---- the feeder thread ----

    def _spawn(self):
        proc = subprocess.Popen(
            ["aplay", "-q", "-D", DEVICE, "-t", "raw", "-f", "S16_LE", "-c", "1", "-r", str(self.RATE),
             f"--buffer-time={self.BUFFER_US}", f"--period-time={self.BUFFER_US // 5}", "-"],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        try:
            fcntl.fcntl(proc.stdin.fileno(), 1031, 4096)  # F_SETPIPE_SZ: the smallest pipe there is
        except OSError:
            pass
        self._written = 0
        self._anchors.clear()
        self._ref_base = None
        return proc

    def _feed(self) -> None:
        rebuffer = 5 * self.BLOCK * 2  # 100 ms cushion after an underrun so late chunks don't stutter-chain
        while True:
            if self._proc is None or self._proc.poll() is not None:
                try:
                    self._proc = self._spawn()
                except OSError as exc:
                    log.error("aplay failed to start: %s", exc)
                    time.sleep(1)
                    continue
            with self._lock:
                if not self._pending and self._loop:
                    self._pending += self._loop
                if (self.streaming and self._stream_heard and self._dry and not self._draining
                        and len(self._pending) < rebuffer):
                    block, runs_dry = b"", True
                else:
                    block = bytes(self._pending[:self.BLOCK * 2])
                    del self._pending[:self.BLOCK * 2]
                    runs_dry = not self._pending  # nothing after this block (yet): it's the edge of a gap
            real = bool(block)
            if self.streaming:
                self._count_underrun(real)
            if real:
                block = self._soften(block, runs_dry)
            else:
                self._dry = True
            block = block.ljust(self.BLOCK * 2, b"\0")
            mixed = self._mix_music(block, real)
            heard = real or mixed is not None
            block = mixed or block
            proc = self._proc
            try:
                proc.stdin.write(block)
            except (BrokenPipeError, ValueError, OSError):  # stop() killed it: start a fresh one
                time.sleep(0.005)
                continue
            if proc is not self._proc:
                continue
            self._written += self.BLOCK
            # Writes only return once aplay has room, so the least-delayed one dates the stream.
            self._anchors.append(time.monotonic() - self._written / self.RATE)
            first = self._written - self.BLOCK  # this block's first sample
            at = min(self._anchors) + first / self.RATE + self.PIPELINE_S + AEC_DELAY_S
            if self.ref:
                self._place(at, first)
            if real:
                self._audible_until = at + self.BLOCK / self.RATE + 0.04
            if heard and self.ref and self._ref_base is not None:
                self.ref.add_at(block, self.RATE, self._ref_base + first * SAMPLE_RATE // self.RATE)

    def music(self, pcm: bytes) -> None:
        """A frame of music (24 kHz mono int16) from the server."""
        with self._lock:
            self._music += pcm
            excess = len(self._music) - int(self.MUSIC_MAX_S * self.RATE) * 2
            if excess > 0:
                del self._music[:excess + int(0.5 * self.RATE) * 2]  # back to ~0.5 s queued

    def _mix_music(self, block: bytes, voice: bool) -> bytes | None:
        """The block with this block's music mixed under it, or None when no music plays."""
        n = self.BLOCK * 2
        with self._lock:
            if not self._music_live:
                if len(self._music) < int(self.MUSIC_PREROLL_S * self.RATE) * 2:
                    return None
                self._music_live, self._mgain = True, 0.0  # starts from silence: the ramp is its fade-in
            m = bytes(self._music[:n])
            del self._music[:n]
            if len(self._music) < n:  # ran dry: fade out what's here, then pre-roll again
                self._music_live = False
        target = self.MUSIC_GAIN * night_gain() * (self.MUSIC_DUCK if voice or self.ducking() else 1.0)
        if _CORE:
            out = ctypes.create_string_buffer(n)
            self._mgain = float(_CORE.sat_mix_music_block(
                block, m, len(m) // 2, out, self.BLOCK, 0 if self._music_live else 1,
                ctypes.c_float(self._mgain), ctypes.c_float(target)))
            return out.raw
        x = np.frombuffer(m.ljust(n, b"\0"), dtype=np.int16).astype(np.float32)
        if not self._music_live:
            x[:len(m) // 2] *= np.linspace(1.0, 0.0, len(m) // 2, dtype=np.float32)
        new = self._mgain + max(-0.1, min(0.1, target - self._mgain))  # ~100 ms to duck or fade in: no clicks
        ramp = np.linspace(self._mgain, new, len(x), dtype=np.float32)
        self._mgain = new
        out = np.frombuffer(block, dtype=np.int16).astype(np.float32) + x * ramp
        return np.clip(out, -32768, 32767).astype(np.int16).tobytes()

    def _count_underrun(self, real: bool) -> None:
        """Called per block while a reply streams. A real block after a dry edge (self._dry, before
        _soften clears it) is a dropout the listener heard: a 4 ms dip, plus any silent blocks between."""
        if not real:
            if self._stream_heard:
                self._silent_run += 1
            return
        if self._stream_heard and self._dry:
            self.underruns += 1
            self.underrun_ms += self._silent_run * self.BLOCK * 1000 // self.RATE
        self._stream_heard, self._silent_run = True, 0

    def _soften(self, block: bytes, runs_dry: bool) -> bytes:
        """No hard edges at a gap. The voice's stream can arrive late (its start and sentence breaks run
        slower than real time), and the speaker then went from mid-waveform to zeros and back: a click
        each time. Now the last 4 ms before a gap fade out and the first 4 ms after one fade in. A gap
        that closes within the same 20 ms only dips for 4 ms."""
        if _CORE and (self._dry or runs_dry):
            out = ctypes.create_string_buffer(block, len(block))
            _CORE.sat_soften_block(out, len(block) // 2, self.FADE, 1 if self._dry else 0, 1 if runs_dry else 0)
            self._dry = runs_dry
            return out.raw
        if not self._dry and not runs_dry:
            return block
        x = np.frombuffer(block, dtype=np.int16).astype(np.float32)
        n = min(self.FADE, len(x))
        if self._dry:
            x[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
        if runs_dry:
            x[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
        self._dry = runs_dry
        return x.astype(np.int16).tobytes()

    def _place(self, at: float, first: int) -> None:
        """Line this aplay's samples up with the mic's once, after the pipeline has filled; from then on
        by sample count, so the canceller's filter stays converged from one reply to the next. The clock
        estimate itself wanders +-40 ms, so it re-places only past 80 ms (an underrun)."""
        if len(self._anchors) < 25:  # still filling: the first writes return before anything plays
            return
        index = self.ref.index_at(at)
        if index is None:
            return
        counted = None if self._ref_base is None else self._ref_base + first * SAMPLE_RATE // self.RATE
        if counted is None or abs(counted - index) > 0.08 * SAMPLE_RATE:
            if counted is not None:
                log.info("player: re-lined up with the mic (%+d ms)", (index - counted) * 1000 // SAMPLE_RATE)
            self._ref_base = index - first * SAMPLE_RATE // self.RATE

    # ---- the event loop's side ----

    def _to_rate(self, pcm: bytes, rate: int) -> bytes:
        if rate == self.RATE or not pcm:
            return pcm
        if _CORE:
            in_n = len(pcm) // 2
            out_n = int(in_n * self.RATE / rate)
            out = ctypes.create_string_buffer(out_n * 2)
            got = _CORE.sat_resample_scale(pcm, in_n, rate, out, out_n, self.RATE, ctypes.c_float(1.0))
            return out.raw[:got * 2]
        x = np.frombuffer(pcm, dtype=np.int16)
        n = int(len(x) * self.RATE / rate)
        return np.interp(np.arange(n) * (rate / self.RATE), np.arange(len(x)), x).astype(np.int16).tobytes()

    def _load(self, path: Path) -> bytes:
        if path not in self._cache:
            rate, pcm = _wav_pcm(path.read_bytes())
            self._cache[path] = self._to_rate(pcm, rate)
        return self._cache[path]

    def _replace(self, pcm: bytes = b"", loop: bytes | None = None) -> None:
        """Swap what's queued for `pcm`, fading out the old sound over 10 ms so the switch doesn't click."""
        fade_n = self.RATE // 100
        with self._lock:
            tail = np.frombuffer(bytes(self._pending[:fade_n * 2]), dtype=np.int16)
            if len(tail):
                tail = (tail * np.linspace(1, 0, len(tail))).astype(np.int16).tobytes()
            self._pending = bytearray(tail) if len(tail) else bytearray()
            self._pending += pcm
            self._loop = loop

    async def stop(self) -> None:
        """Silence now: drop what's queued and restart aplay, so what it had buffered never plays."""
        with self._lock:
            self._pending.clear()
            self._loop = None
        self.streaming = False
        if time.monotonic() < self._audible_until and self._proc and self._proc.poll() is None:
            if self.ref:
                self.ref.cut(time.monotonic())
            self._audible_until = 0.0
            self._proc.kill()

    async def play_file(self, path: Path, gain: float | None = None) -> None:
        """Start a sound without waiting for it to finish."""
        self._replace(_scaled(self._load(path), night_gain() * self.voice_gain if gain is None else gain))

    async def play_file_and_wait(self, path: Path, gain: float | None = None) -> None:
        await self.play_file(path, gain)
        await self._drained()

    async def loop_file(self, path: Path) -> None:
        """The sound over and over, joined sample to sample: no gap at the loop point."""
        pcm = _scaled(self._load(path), night_gain() * self.voice_gain)
        self._replace(pcm, loop=pcm)

    async def start_stream(self, rate: int) -> None:
        """A reply follows in pieces (write); they join with no gap."""
        self._replace()
        self.gain = night_gain()
        self.rate = rate
        self._stream_heard, self._silent_run = False, 0
        self.underruns = self.underrun_ms = 0
        self._draining = False
        self.streaming = True

    def mark(self) -> None:
        """The reply's own audio starts here: the wait after a filler (a tool running) isn't a dropout."""
        self._stream_heard, self._silent_run = False, 0
        self.underruns = self.underrun_ms = 0

    async def write(self, pcm: bytes) -> None:
        with self._lock:
            self._pending += _scaled(self._to_rate(pcm, self.rate), self.gain)
        while len(self._pending) > 2 * self.RATE * 2:  # backpressure: ~2 s queued is plenty
            await asyncio.sleep(0.05)

    async def finish_stream(self) -> None:
        """No more audio: return once what's queued has played."""
        self._draining = True
        await self._drained()
        self.streaming = False

    async def _drained(self) -> None:
        while self._pending or self._loop or time.monotonic() < self._audible_until:
            await asyncio.sleep(0.02)

    def is_playing(self) -> bool:
        return bool(self._pending or self._loop) or time.monotonic() < self._audible_until


class FastWakeModel:
    """C++-accelerated openWakeWord pipeline (_sat_core.so + direct ONNX Runtime sessions).
    Eliminates Python list/deque copies on every 80 ms chunk and replaces openWakeWord's
    1,500 ms random-audio reset() with a <0.01 ms C++ buffer restore."""

    fast_reset = True

    def __init__(self, wake_model_path: str, speex: bool = True) -> None:
        import onnxruntime as ort
        import openwakeword
        res_dir = Path(openwakeword.__file__).resolve().parent / "resources" / "models"
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        prov = ["CPUExecutionProvider"]
        self._mel_sess = ort.InferenceSession(str(res_dir / "melspectrogram.onnx"), sess_options=opts, providers=prov)
        self._emb_sess = ort.InferenceSession(str(res_dir / "embedding_model.onnx"), sess_options=opts, providers=prov)
        self._wake_sess = ort.InferenceSession(wake_model_path, sess_options=opts, providers=prov)
        self._mel_in = self._mel_sess.get_inputs()[0].name
        self._emb_in = self._emb_sess.get_inputs()[0].name
        self._wake_in = self._wake_sess.get_inputs()[0].name
        self.name = Path(wake_model_path).stem
        self.models = {self.name: self._wake_sess}
        self._c = _CORE.sat_wake_create(1 if speex else 0)
        # Compute baseline 16x96 features once at startup so every reset() is a 6 KB memcpy (<0.01 ms).
        rng = np.random.default_rng(0)
        noise = rng.integers(-1000, 1000, size=(1, 16000 * 2), dtype=np.int16).astype(np.float32)
        mels = np.squeeze(self._mel_sess.run(None, {self._mel_in: noise})[0]) * 0.1 + 2.0
        wins = np.array([mels[i:i + 76] for i in range(0, mels.shape[0] - 76 + 1, 8)][-16:], dtype=np.float32)[:, :, :, None]
        init_feats = np.ascontiguousarray(np.squeeze(self._emb_sess.run(None, {self._emb_in: wins})[0]), dtype=np.float32)
        _CORE.sat_wake_set_init_features(self._c, init_feats.ctypes.data)

    def __del__(self) -> None:
        if getattr(self, "_c", None) and _CORE:
            _CORE.sat_wake_destroy(self._c)
            self._c = None

    def reset(self) -> None:
        _CORE.sat_wake_reset(self._c)

    def predict(self, x: np.ndarray) -> dict[str, float]:
        pcm = np.ascontiguousarray(x, dtype=np.int16)
        mel_in_ptr = _CORE.sat_wake_prepare_melspec(self._c, pcm.ctypes.data)
        mel_in = np.ctypeslib.as_array(mel_in_ptr, shape=(1, 1760))
        mels = np.ascontiguousarray(np.squeeze(self._mel_sess.run(None, {self._mel_in: mel_in})[0]), dtype=np.float32)
        mel_win_ptr = _CORE.sat_wake_push_mels(self._c, mels.ctypes.data, mels.shape[0])
        mel_win = np.ctypeslib.as_array(mel_win_ptr, shape=(1, 76, 32, 1))
        emb = np.ascontiguousarray(np.squeeze(self._emb_sess.run(None, {self._emb_in: mel_win})[0]), dtype=np.float32)
        ready = ctypes.c_int(0)
        feat_win_ptr = _CORE.sat_wake_push_emb(self._c, emb.ctypes.data, ctypes.byref(ready))
        if not ready.value:
            return {self.name: 0.0}
        feat_win = np.ctypeslib.as_array(feat_win_ptr, shape=(1, 16, 96))
        out = self._wake_sess.run(None, {self._wake_in: feat_win})[0]
        return {self.name: float(np.squeeze(out))}


def _wake_model():
    """The wake-word model with the VAD gate and noise suppression where the Pi has
    what they need (Silero's model file, the speexdsp_ns package); each one that
    can't load is turned off with a log line instead of stopping the satellite."""
    vad, speex = WAKE_VAD, WAKE_SPEEX
    if _CORE and not vad:
        try:
            model = FastWakeModel(WAKE_MODEL, speex=speex)
            log.info("wake word: C++ core (_sat_core.so), threshold %.2f, noise suppression %s, mic gain %.1f",
                     WAKE_THRESHOLD, "on" if speex else "off", MIC_GAIN)
            return model
        except Exception as exc:
            log.warning("C++ wake model init failed (%s); falling back to Python openWakeWord", exc)
    while True:
        try:
            model = Model(wakeword_models=[WAKE_MODEL], inference_framework="onnx",
                          vad_threshold=vad, enable_speex_noise_suppression=speex)
            # Cache initial feature_buffer so Python fallback reset() doesn't re-run 4 s of random audio
            init_fb = model.preprocessor.feature_buffer.copy()
            orig_reset = model.reset

            def _fast_py_reset() -> None:
                model.prediction_buffer.clear()
                model.preprocessor.raw_data_buffer.clear()
                model.preprocessor.melspectrogram_buffer = np.ones((76, 32))
                model.preprocessor.accumulated_samples = 0
                model.preprocessor.raw_data_remainder = np.empty(0)
                model.preprocessor.feature_buffer = init_fb.copy()

            model.reset = _fast_py_reset
            model.fast_reset = True
            log.info("wake word: threshold %.2f, VAD gate %s, noise suppression %s, mic gain %.1f",
                     WAKE_THRESHOLD, vad or "off", "on" if speex else "off", MIC_GAIN)
            return model
        except Exception as exc:  # missing package or model file
            if speex:
                log.warning("noise suppression unavailable (%s); continuing without it", exc)
                speex = False
            elif vad:
                log.warning("VAD gate unavailable (%s); continuing without it", exc)
                vad = 0
            else:
                raise


class Satellite:
    def __init__(self, token: str) -> None:
        self.token = token
        self.wake = _wake_model()
        self.wake_name = next(iter(self.wake.models))
        self.player = Player()
        self.player.ducking = lambda: self.state != "wake"  # music down while anyone talks to him
        self.aec = None
        if AEC_MODE in ("shadow", "on"):
            try:
                import aec
                self.player.ref = aec.Reference()
                self.aec = aec.Canceller(self.player.ref, filter_ms=AEC_FILTER_MS)
                log.info("echo cancellation: %s (Speex, reference delay %d ms, filter %d ms), speech barge-in %s",
                         AEC_MODE, AEC_DELAY_S * 1000, AEC_FILTER_MS, "on" if BARGE_SPEECH else "off")
            except OSError as exc:  # no libspeexdsp
                log.warning("echo cancellation unavailable (%s); continuing without it", exc)
        self.ws = None
        self.state = "wake"  # wake | listening | thinking | speaking | interrupting | alarm
        self.muted = False
        self.voice = "leo"  # whose voice the server speaks in ("voice" events): picks the spoken clips
        self._woke_at = 0.0
        self._thinking_at = 0.0
        self._follow_up = False  # this listening window opened by itself after a reply
        self._failed = False     # the server reported an error this turn
        self._chunks: asyncio.Queue = asyncio.Queue()  # reply WAVs, then None at audio_end
        self._playback: asyncio.Task | None = None
        self._chime: asyncio.Task | None = None  # alarm chime; the reply waits for it
        self._captioned = False  # this turn's first caption seen (Player.mark)
        self._overlay: asyncio.Task | None = None  # a backchannel clip; the thinking cue waits for it
        # How far behind real time the mic is being read (arecord's pipe backs up when we fall behind):
        # the mic's sample count against the clock, worst per state, logged at each turn.
        self._mic_samples = 0
        self._mic_anchors: deque = deque(maxlen=64)
        self._mic_lag: dict[str, int] = {}
        self._wake_lock = threading.Lock()  # predict() and reset() run in threads: never both at once
        self._wake_reset: asyncio.Task | None = None
        self._last_mic = time.monotonic()
        self._last_offline_note = 0.0
        self._near_logged = 0.0
        self._barge_max = 0.0  # loudest wake score heard over her own voice this turn
        # Speech barge-in (BARGE_SPEECH): the cancelled mic's level per chunk while he talks.
        self._speaking_at = 0.0
        self._residual: deque = deque(maxlen=25)  # quiet chunks of the last ~2 s: the echo that's left
        self._loud: deque = deque(maxlen=BARGE_WINDOW)  # the last chunks: above the bar or not
        self._preroll: deque = deque(maxlen=16)   # the last ~1.3 s sent on after a speech barge-in
        self._barge_speech = False                # the current barge-in was speech, not the wake word
        self.stats = {"started": time.time(), "wakes": 0, "near_misses": 0, "barge_ins": 0,
                      "reconnects": 0, "last_wake": None, "last_turn_ms": None}
        # Recent scores, sent with every status so the threshold can be tuned from the Mac's log.
        self._wake_scores: deque = deque(maxlen=20)
        self._near_scores: deque = deque(maxlen=20)

    async def run(self) -> None:
        _configure_alsa_mic()
        mic = await asyncio.create_subprocess_exec(
            "arecord", "-q", "-D", DEVICE, "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw",
            stdout=asyncio.subprocess.PIPE,
        )
        tasks = [self._mic_loop(mic), self._connection_loop(), self._watchdog(), self._status_loop(), self._led_loop()]
        if BUTTON_LINE:
            tasks.append(self._button_loop())
        try:
            await asyncio.gather(*tasks)
        finally:
            if mic.returncode is None:
                mic.terminate()

    # ---- connection ----

    async def _connection_loop(self) -> None:
        down_since: float | None = None
        while True:
            try:
                async with connect(f"{SERVER_URL}?token={quote(self.token)}", max_size=None) as ws:
                    if down_since is not None:
                        self.stats["reconnects"] += 1
                        log.info("reconnected after %.1f s offline", time.monotonic() - down_since)
                    down_since = None
                    log.info("connected to %s; waiting for wake word", SERVER_URL)
                    # Lets the server start speaking here (alarms), not just answer.
                    await ws.send(json.dumps({"type": "hello", "role": "satellite", "context": CONTEXT}))
                    self.ws = ws
                    await self._send_status()
                    await self._recv_loop(ws)
            except (OSError, ConnectionClosed) as exc:
                if down_since is None:
                    log.warning("connection lost (%s); retrying every %.0fs", exc, RECONNECT_S)
            self.ws = None
            down_since = down_since or time.monotonic()
            if self.state != "wake":
                await self.stop_playback()
                self._enter_wake()
            await asyncio.sleep(RECONNECT_S)

    async def _send(self, msg: dict | bytes) -> None:
        """Send if connected; a dead socket is the connection loop's to notice."""
        ws = self.ws
        if ws is None:
            return
        try:
            await ws.send(msg if isinstance(msg, bytes) else json.dumps(msg))
        except ConnectionClosed:
            pass

    async def _send_status(self) -> None:
        await self._send({"type": "status", "room": ROOM, "muted": self.muted, "wifi_dbm": _wifi_dbm(),
                          "night": night_gain() != 1.0, "uptime_s": int(time.time() - self.stats["started"]),
                          "wake_threshold": WAKE_THRESHOLD, "wake_scores": list(self._wake_scores),
                          "near_miss_scores": list(self._near_scores),
                          **{k: v for k, v in self.stats.items() if k != "started"}})

    async def _status_loop(self) -> None:
        while True:
            await asyncio.sleep(STATUS_EVERY_S)
            await self._send_status()

    async def _led_loop(self) -> None:
        leds = Leds(LED_DEVICE)
        while True:
            colors, brightness = leds.frame(self.state, self.muted, self.ws is not None, time.monotonic())
            if brightness and night_gain() != 1.0:
                brightness = max(1, brightness // 3)  # night mode: dimmer too
            leds.show(colors, brightness)
            await asyncio.sleep(0.05)

    async def _watchdog(self) -> None:
        # arecord can wedge (ALSA xrun, HAT glitch) without exiting; a crash
        # systemd sees, a silent stall it doesn't. Exiting turns one into the other.
        while True:
            await asyncio.sleep(MIC_STALL_S / 2)
            if time.monotonic() - self._last_mic > MIC_STALL_S:
                log.error("no audio from the mic for %.0f s; exiting so systemd restarts us", MIC_STALL_S)
                os._exit(1)

    # ---- mic ----

    def _enter_wake(self) -> None:
        # Clear openWakeWord's rolling buffers so audio from the last turn
        # (including our own speaker) can't trigger a stale detection.
        self._reset_wake()
        self.state = "wake"

    def _reset_wake(self) -> None:
        """Clear openWakeWord's buffers. With C++ FastWakeModel (or cached feature_buffer) reset() takes
        <0.01 ms inline instead of burning 1.5 s of CPU on 4 s of random noise."""
        if getattr(self.wake, "fast_reset", False):
            with self._wake_lock:
                self.wake.reset()
            return

        def reset() -> None:
            with self._wake_lock:
                self.wake.reset()
        if self._wake_reset is None or self._wake_reset.done():
            self._wake_reset = asyncio.create_task(asyncio.to_thread(reset))

    async def _wake_score(self, chunk: bytes) -> float:
        # ~50 ms of CPU per chunk; off the event loop so websocket traffic keeps flowing.
        if self._wake_reset and not self._wake_reset.done():
            return 0.0  # buffers still clearing (~1.5 s): nothing to score yet

        def predict(x: np.ndarray) -> dict:
            with self._wake_lock:
                return self.wake.predict(x)
        scores = await asyncio.to_thread(predict, np.frombuffer(chunk, dtype=np.int16))
        return float(scores[self.wake_name])

    async def _mic_loop(self, mic) -> None:
        while True:
            try:
                raw = await mic.stdout.readexactly(CHUNK_BYTES)
            except asyncio.IncompleteReadError:
                raise SystemExit("arecord stopped")
            self._mic_samples += CHUNK_SAMPLES
            now = time.monotonic()
            self._mic_anchors.append(now - self._mic_samples / SAMPLE_RATE)
            lag = int((now - self._mic_samples / SAMPLE_RATE - min(self._mic_anchors)) * 1000)
            self._mic_lag[self.state] = max(self._mic_lag.get(self.state, 0), lag)
            # Every chunk, muted or not: the canceller keeps its clock from the mic's sample count.
            cancelled = self.aec.process(raw) if self.aec else raw
            chunk = _boosted(cancelled if AEC_MODE == "on" else raw)
            self._last_mic = time.monotonic()
            self._preroll.append(chunk)
            if self.muted:
                continue
            if self.state == "wake":
                score = await self._wake_score(chunk)
                if score >= WAKE_THRESHOLD:
                    log.info("wake (score %.2f)", score)
                    self._wake_scores.append(round(score, 2))
                    await self._wake_up()
                elif score >= NEAR_MISS and time.monotonic() - self._near_logged > 2:
                    self._near_logged = time.monotonic()
                    self.stats["near_misses"] += 1
                    self._near_scores.append(round(score, 2))
                    log.info("near miss (score %.2f, threshold %.2f)", score, WAKE_THRESHOLD)
            elif self.state == "listening":
                if min(self._mic_anchors) + self._mic_samples / SAMPLE_RATE < self._woke_at:
                    continue  # recorded before this window opened (a backlog): may be his own voice
                # The speaker sits next to the mics: send silence while the ping
                # plays so the server's VAD doesn't take the chime for speech.
                await self._send(SILENCE if self.player.is_playing() else chunk)
            elif self.state == "speaking" and self._heard_over_reply(cancelled):
                await self._barge_in(speech=True)
            elif self.state in ("thinking", "speaking") and BARGE_THRESHOLD > 0:
                score = await self._wake_score(chunk)
                self._barge_max = max(self._barge_max, score)
                if score >= BARGE_THRESHOLD:
                    log.info("barge-in (score %.2f)", score)
                    await self._barge_in()

    async def _wake_up(self) -> None:
        self.stats["wakes"] += 1
        self.stats["last_wake"] = datetime.now().isoformat(timespec="seconds")
        if self.ws is None:
            # She can't answer, but she can say why (at most once a minute).
            self._reset_wake()
            if time.monotonic() - self._last_offline_note > 60:
                self._last_offline_note = time.monotonic()
                log.warning("woken while offline")
                await self.player.play_file(voice_clip(OFFLINE_WAV, self.voice))
            return
        await self._start_listening(follow_up=False)

    async def _start_listening(self, follow_up: bool, preroll: list[bytes] | None = None) -> None:
        self._woke_at = time.monotonic()
        self._follow_up = follow_up
        self._failed = False
        self.state = "listening"
        if not follow_up:
            await self.player.play_file(PING_WAV)
        # chunked: one WAV per sentence, so the first sentence plays while the rest is synthesized.
        await self._send({"type": "start", "chunked": True, "context": CONTEXT,
                          "follow_up": follow_up, "barge_speech": bool(preroll)})
        for chunk in preroll or ():  # what they'd already said when they cut in
            await self._send(chunk)

    def _heard_over_reply(self, pcm: bytes) -> bool:
        """Someone talking over the reply: 4 of the last 6 chunks (~320 ms) of the cancelled mic
        BARGE_SPEECH_DB above the loud parts of the echo the canceller leaves."""
        if not (BARGE_SPEECH and self.aec and AEC_MODE == "on"):
            return False
        if self._chime:
            return False  # the alarm chime / announce ping is still playing: nothing counts yet
        if _CORE:
            db = float(_CORE.sat_dbfs(pcm, len(pcm) // 2))
        else:
            x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
            db = 10 * math.log10(float(np.mean(x * x)) / 32768 ** 2 + 1e-12)
        if time.monotonic() - self._speaking_at < BARGE_SETTLE_S:
            self._residual.append(db)
            return False
        bar = max(float(np.percentile(self._residual, 90)) + BARGE_SPEECH_DB if self._residual else -40.0,
                  BARGE_FLOOR_DBFS)
        self._loud.append(db > bar)
        if db <= bar:
            self._residual.append(db)
        if sum(self._loud) >= BARGE_LOUD_CHUNKS:
            log.info("barge-in (speech, %.0f dBFS over a %.0f dBFS bar)", db, bar)
            return True
        return False

    async def _barge_in(self, speech: bool = False) -> None:
        """Stop talking now; listen again once the server confirms the turn is cancelled."""
        self.stats["barge_ins"] += 1
        self._barge_speech = speech
        await self.stop_playback()
        self.state = "interrupting"
        await self._send({"type": "interrupt"})

    # ---- button ----

    async def _button_loop(self) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "gpiomon", "-c", BUTTON_CHIP, "-b", "pull-up", "-p", "30ms", "-e", "both", "-F", "%E", BUTTON_LINE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            log.warning("gpiomon not found; button disabled")
            return
        pressed_at = None
        while line := await proc.stdout.readline():
            edge = line.decode().strip()
            if edge == "falling":  # active low: pressed
                pressed_at = time.monotonic()
            elif edge == "rising" and pressed_at is not None:
                held = time.monotonic() - pressed_at
                pressed_at = None
                await (self._toggle_mute() if held >= LONG_PRESS_S else self._button_press())
        err = (await proc.stderr.read()).decode().strip()
        log.warning("button monitor exited (%s); button disabled", err or proc.returncode)

    async def _button_press(self) -> None:
        if self.muted:
            await self._toggle_mute()
        elif self.state in ("thinking", "speaking"):
            log.info("button: interrupt")
            await self._barge_in()
        elif self.state == "wake":
            log.info("button: talk")
            await self._wake_up()
        elif self.state == "listening":
            log.info("button: stop listening")
            await self._send({"type": "stop"})

    async def _toggle_mute(self) -> None:
        self.muted = not self.muted
        log.info("mic %s", "muted" if self.muted else "unmuted")
        if self.muted:
            if self.state == "listening":
                await self._send({"type": "stop"})
            elif self.state in ("thinking", "speaking"):
                await self._barge_in()
            self._enter_wake()
        else:
            self._reset_wake()
        await self.player.play_file(voice_clip(MIC_OFF_WAV if self.muted else MIC_ON_WAV, self.voice))
        await self._send_status()

    # ---- server events ----

    async def _recv_loop(self, ws) -> None:
        async for msg in ws:
            if isinstance(msg, bytes):
                if msg[:4] == b"MUS0":  # music from the Mac's tap: its own channel (Player.music)
                    self.player.music(msg[4:])
                else:
                    self._on_audio(msg)
                continue
            ev = json.loads(msg)
            kind = ev.get("type")
            if kind == "state":
                await self._on_state(ev["state"])
            elif kind == "voice":  # his voice volume, for the cues and clips (alarms stay at full volume)
                self.player.voice_gain = max(0.1, min(1.0, float(ev.get("volume", 1.0))))
                self.voice = str(ev.get("name") or self.voice)
            elif kind == "alarm":
                # The server is about to speak unprompted: a cue first, speech waits for it.
                # Alarms ring at full volume, night or not; announcements get the soft ping.
                cue = ev.get("cue", "alarm")
                log.info(cue)
                await self.stop_playback()
                self.state = "alarm"
                self._chime = asyncio.create_task(
                    self.player.play_file_and_wait(PING_WAV) if cue == "announce"
                    else self.player.play_file_and_wait(ALARM_WAV, gain=1.0))
            elif kind == "overlay":
                # A backchannel while the user talks: only when nothing else is playing. The server
                # stops listening until overlay_ended (no echo cancellation here), so always answer.
                clip = ev.get("clip")
                if self.state == "listening" and clip in BACKCHANNEL_CLIPS and not self.player.is_playing():
                    self._overlay = asyncio.create_task(self._play_overlay(
                        voice_clip(SOUNDS / f"thursday-bc-{clip}.wav", self.voice), float(ev.get("gain", 0.7)) * self.player.voice_gain))
                else:
                    await self._send({"type": "overlay_ended"})
            elif kind == "caption" and not self._captioned:
                # The first caption comes just before the reply's first WAV (after any filler), as in the
                # web client: underruns count from here, not across the tool's wait (logged as 2-7 s "dry").
                self._captioned = True
                self.player.mark()
            elif kind == "audio_end":
                self._chunks.put_nowait(None)
            elif kind == "transcript":
                log.info("heard: %s", ev.get("text", ""))
            elif kind == "reply":
                log.info("reply: %s", ev.get("text", ""))
            elif kind == "error":
                self._failed = True
                log.error("%s error: %s", ev.get("stage"), ev.get("message"))

    async def _on_state(self, state: str) -> None:
        if state == "thinking":
            self._thinking_at = time.monotonic()
            self._captioned = False
            # Barge-in scoring starts now. The wake model wasn't fed while listening, so its
            # buffers still end with the "Hey THURSDAY" that woke us: cancelling every turn.
            self._reset_wake()
            self.state = "thinking"
            log.info("thinking (listened %d ms)", _ms(self._woke_at))
            if self._overlay and not self._overlay.done():
                await asyncio.wait({self._overlay}, timeout=1.0)  # starting the cue would cut the clip off
            await self.player.loop_file(THINKING_WAV)
        elif state == "idle" and self.state == "interrupting":
            # The cancelled turn is gone: listen for what the user wants instead. Cut off by
            # speech, they're mid-sentence: no ping, and what they've said so far goes first.
            speech, self._barge_speech = self._barge_speech, False
            await self._start_listening(follow_up=speech, preroll=list(self._preroll) if speech else None)
        elif state == "idle" and self.state == "speaking" and FOLLOW_UP and not self.muted:
            # The server's answer to our playback_ended: keep the conversation going.
            log.info("reply done; listening for a follow-up")
            await self._start_listening(follow_up=True)
        elif state == "idle" and self.state != "wake":
            # No speech, an error, or the end of a reply: back to the wake word.
            was = self.state
            await self.stop_playback()
            self._enter_wake()
            if self._failed:
                await self.player.play_file(voice_clip(ERROR_WAV, self.voice))
            # From "thinking" with no reply and no error: the server heard "nevermind" / "I'm good"
            # and ended the conversation (voice_ws._CLOSER). Same sound either way.
            elif was in ("listening", "thinking") and not self.muted:
                await self.player.play_file(CLOSE_WAV)  # "I've stopped listening"
            log.info("idle; waiting for wake word")

    async def _play_overlay(self, path: Path, gain: float) -> None:
        try:
            await self.player.play_file_and_wait(path, gain * night_gain())
        except (OSError, ValueError) as exc:  # a missing or bad clip: the server must still hear back
            log.warning("backchannel %s: %s", path.name, exc)
        finally:
            await self._send({"type": "overlay_ended"})
        log.info("backchannel %s", path.name)

    def _on_audio(self, wav: bytes) -> None:
        # Never await playback here: the receive loop must keep reading the
        # next sentences while this one plays.
        if self.state == "interrupting":
            return  # the tail of a cancelled turn
        if self._playback is None or self._playback.done():
            if self._thinking_at:
                log.info("reply audio received %d ms after thinking", _ms(self._thinking_at))
            self.state = "speaking"
            self._speaking_at = time.monotonic()
            self._residual.clear()
            self._loud.clear()
            self._chunks = asyncio.Queue()
            self._playback = asyncio.create_task(self._play_reply(self._chunks))
        self._chunks.put_nowait(wav)

    async def _play_reply(self, chunks: asyncio.Queue) -> None:
        """Play the reply's WAVs back to back through one continuous player until
        audio_end, then report playback_ended."""
        played, rate, audio_s = 0, None, 0.0
        while (wav := await chunks.get()) is not None:
            if played == 0 and self._chime:
                await self._chime  # let the alarm chime finish before THURSDAY speaks
                self._chime = None
                # The settle window and echo floor start with her voice, not with the chime:
                # learned during the chime, the first loud syllable read as someone cutting in.
                self._speaking_at = time.monotonic()
                self._residual.clear()
                self._loud.clear()
            chunk_rate, pcm = _wav_pcm(wav)
            if chunk_rate != rate:  # first chunk (or a format change): (re)start the player
                if rate is not None:
                    await self.player.finish_stream()
                await self.player.start_stream(chunk_rate)
                rate = chunk_rate
            if played == 0:
                # Speech end (the server's "thinking") to the first PCM handed to aplay.
                # aplay's own start-up adds a little on top; logged per turn to compare.
                first_ms = _ms(self._thinking_at) if self._thinking_at else None
                self.stats["last_turn_ms"] = first_ms
                log.info("first reply audio %d KB, %s ms after speech end", len(wav) // 1024, first_ms)
            await self.player.write(pcm)
            if played == 0:
                await self._send({"type": "playback_started"})
            played += 1
            audio_s += len(pcm) / 2 / chunk_rate
        await self.player.finish_stream()
        log.info("turn: listened %d ms, first audio %s ms, %d chunks, %.1f s of speech, underruns %d (%d ms silent), "
                 "barge score max %.2f, mic read lag max %s",
                 int((self._thinking_at - self._woke_at) * 1000) if self._thinking_at > self._woke_at else 0,
                 self.stats["last_turn_ms"], played, audio_s, self.player.underruns, self.player.underrun_ms,
                 self._barge_max, " ".join(f"{k}={v}ms" for k, v in self._mic_lag.items()))
        self._mic_lag = {}
        self._thinking_at = 0.0
        self._barge_max = 0.0
        await self._send({"type": "playback_ended", "underruns": self.player.underruns,
                          "dry_ms": self.player.underrun_ms})

    async def stop_playback(self) -> None:
        for task in (self._playback, self._chime):
            if task and not task.done():
                task.cancel()
        self._playback = self._chime = None
        await self.player.stop()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not SERVER_URL:
        sys.exit("set THURSDAY_WS, e.g. ws://192.168.1.10:8000/ws/voice")
    token = TOKEN_FILE.read_text().strip()
    if not token:
        sys.exit(f"{TOKEN_FILE} is empty")
    asyncio.run(Satellite(token).run())


if __name__ == "__main__":
    main()
