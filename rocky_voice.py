"""Rocky's voice: every chunk of THURSDAY's speech passes through here on its way out (voice_ws._tts),
so the satellite, the browser and phone calls all hear the same Eridian voice. NumPy only.

  TTS sentence (int16 WAV, 24 kHz)
    -> +ROCKY_PITCH_SEMITONES pitch and formant shift (resampling: words stay crisp, ~12% quicker)
    -> envelope and f0 tracking, 10 ms hops
    -> optional (ROCKY_ALIEN > 0, off by default since 2026-11-01: too alien to live with),
       an alien speaking through a translator, ROCKY_ALIEN of it alien:
         alien: a vocoder. The voice's formants (what makes the words) shape a carrier of Eridian
                chords (sawtooth triads snapped to D Dorian; breath noise on consonants), so the
                chords themselves say the words
         translator: the voice band-limited to a small device speaker, with a metallic ring mod
    -> granite-slab resonance (a short comb: sound ringing in stone), ROCKY_GRANITE of it (off by default)
    -> chord layer under every syllable: ocarina triad (root, fifth, octave) snapped to D Dorian,
       jug sub-bass on stressed syllables, clarinet reed on "Amaze!" / "Bad, bad, bad!",
       a rising ocarina glide (+5th -> +octave) on a closing "question?"
    -> mix (1 - ROCKY_CHORD_MIX) voice / ROCKY_CHORD_MIX chords, peak-limited below clipping

The chunk's text tells it about exclamations and questions. Phone audio (8 kHz G.711 mu-law)
comes from to_mulaw_8k after processing, so calls hear exactly what the speaker does.
"""
from __future__ import annotations

import io
import os
import re
import struct
import wave

import numpy as np

PITCH_SEMITONES = float(os.getenv("ROCKY_PITCH_SEMITONES", "2.0"))
CHORD_MIX = float(os.getenv("ROCKY_CHORD_MIX", "0.20"))
ALIEN = float(os.getenv("ROCKY_ALIEN", "0"))  # 0 = off; above 0, vocoded chords vs a translator voice
# The stone-chamber comb and reflections, 0..1. Off since 2026-11-01: heard as reverb on the satellite,
# whose room adds its own; 1 is the original chamber.
GRANITE = float(os.getenv("ROCKY_GRANITE", "0"))
LAYER_GAIN = 0.85         # the ocarina layer, before CHORD_MIX: about the voice's own level
DEVICE_GAIN = 1.4         # the band-limited translator voice loses its lows; this gives them back
STREAM_MARGIN_S = 0.08    # streaming holds back this much: the filters' look-ahead (~50 ms) and some
# Most new audio a streaming pass waits for (Stream). The old x1.4 growth made passes ever further
# apart, so audio left in growing bursts and the satellite ran dry between them (2026-11-01: dry
# median 232 ms per sentence; capped at 0.4 s plus VOICE_PREROLL_MS 250: 0 ms, 9/10 runs clean).
STREAM_PASS_MAX_S = float(os.getenv("ROCKY_STREAM_PASS_MAX_S", "0.4"))
HOP_S = 0.010                  # analysis hop
DORIAN = (0, 2, 3, 5, 7, 9, 10)  # D Dorian, as semitones above D
OCARINA = (294.0, 700.0)       # the chord root lives here (D4 up to about F5)
PEAK = 0.93                    # of full scale: headroom for the mu-law encoder

_EXCLAIM = re.compile(r"\bamaze\b|\b(bad|happy|good),? \1,? \1\b|!", re.I)
_QUESTION = re.compile(r"\bquestion\W*$", re.I)


# --- WAV in, WAV out -------------------------------------------------------------------------

def _read(wav: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav)) as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"expected 16-bit PCM, got {8 * width}-bit")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return (x.reshape(-1, ch).mean(axis=1) if ch > 1 else x), sr


def _write(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


# --- building blocks -------------------------------------------------------------------------

def _lowpass(x: np.ndarray, cutoff: float, taps: int = 127) -> np.ndarray:
    """Windowed-sinc FIR; cutoff as a fraction of the sample rate (0.5 = Nyquist)."""
    n = np.arange(taps) - (taps - 1) / 2
    h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.blackman(taps)
    return np.convolve(x, h / h.sum(), mode="same")


def _shift(x: np.ndarray, semitones: float) -> np.ndarray:
    """Pitch and formants up together by resampling: the cheapest shift with no phasey artifacts.
    Speech gets 2^(st/12) quicker (12% at +2), which suits an excited engineer."""
    r = 2 ** (semitones / 12)
    if abs(r - 1) < 1e-3:
        return x
    y = _lowpass(x, 0.5 / r * 0.92) if r > 1 else x  # no aliasing from the squeeze
    return np.interp(np.arange(int(len(y) / r)) * r, np.arange(len(y)), y).astype(np.float32)


def _delay(x: np.ndarray, samples: int) -> np.ndarray:
    return np.concatenate([np.zeros(samples, x.dtype), x[:-samples]]) if 0 < samples < len(x) else np.zeros_like(x)


def _granite(x: np.ndarray, sr: int) -> np.ndarray:
    """Sound in a small stone chamber: a tight feedback comb (5.3 ms, unrolled so it's
    vectorized) plus a few early reflections. Short and light, so consonants stay sharp.
    Scaled by GRANITE; at 0 the voice passes through dry."""
    if GRANITE <= 0:
        return x
    d = int(sr * 0.0053)
    comb = sum(0.32 ** k * _delay(x, k * d) for k in range(1, 6))
    early = sum(g * _delay(x, int(sr * ms / 1000)) for ms, g in ((2.9, 0.18), (7.1, 0.12), (11.3, 0.07)))
    return x + GRANITE * (0.55 * comb + early)


def _stft(x: np.ndarray, n: int = 1024, hop: int = 256) -> tuple[np.ndarray, np.ndarray]:
    win = np.hanning(n).astype(np.float32)
    pad = np.pad(x, (n // 2, n // 2 + n))
    count = 1 + (len(pad) - n) // hop
    return np.fft.rfft(pad[np.arange(n)[None, :] + hop * np.arange(count)[:, None]] * win, axis=1), win


def _istft(spec: np.ndarray, win: np.ndarray, hop: int, length: int) -> np.ndarray:
    n = len(win)
    frames = np.fft.irfft(spec, n, axis=1) * win
    out, norm = np.zeros(hop * (len(frames) - 1) + n), np.zeros(hop * (len(frames) - 1) + n)
    for i, f in enumerate(frames):
        out[i * hop:i * hop + n] += f
        norm[i * hop:i * hop + n] += win ** 2
    return (out / np.maximum(norm, 1e-6))[n // 2:n // 2 + length].astype(np.float32)


def _smooth_bins(mag: np.ndarray, width: int) -> np.ndarray:
    """Moving average across frequency (odd width): the spectral envelope, formants without pitch."""
    half = width // 2
    c = np.cumsum(np.pad(mag, ((0, 0), (half + 1, half)), mode="edge"), axis=1)
    return (c[:, width:] - c[:, :-width]) / width


def _vocode(voice: np.ndarray, carrier: np.ndarray) -> np.ndarray:
    """The carrier, shaped by the voice's spectral envelope frame by frame (~11 ms, ~250 Hz), each
    frame at the voice's own level. Frame-local only, so it streams (see Stream)."""
    sv, win = _stft(voice)
    sc, _ = _stft(carrier)
    ev, ec = _smooth_bins(np.abs(sv), 11), _smooth_bins(np.abs(sc), 11)
    out = sc * ev / (ec + 1e-3 * ec.max(axis=1, keepdims=True) + 1e-9)
    gain = np.sqrt((np.abs(sv) ** 2).sum(axis=1) / np.maximum((np.abs(out) ** 2).sum(axis=1), 1e-12))
    return _istft(out * gain[:, None], win, 256, len(voice))


def _saw(freq: np.ndarray, sr: int) -> np.ndarray:
    return 2 * ((np.cumsum(freq) / sr) % 1.0) - 1


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) or 1e-9


def _frames(x: np.ndarray, hop: int) -> np.ndarray:
    win = 4 * hop
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    return x[np.arange(win)[None, :] + hop * np.arange(n)[:, None]]


def _envelope(frames: np.ndarray) -> np.ndarray:
    """Syllable envelope: frame RMS, quick attack (~10 ms), slower release (~60 ms)."""
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    out, level = np.empty_like(rms), 0.0
    for i, v in enumerate(rms):
        level += (v - level) * (0.65 if v > level else 0.15)
        out[i] = level
    return out


def _pitch(frames: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """f0 per frame by autocorrelation (FFT), 70-500 Hz, and whether the frame is voiced."""
    win = frames.shape[1]
    spec = np.fft.rfft(frames * np.hanning(win), 2 * win)
    ac = np.fft.irfft(np.abs(spec) ** 2)[:, :win]
    lo, hi = int(sr / 500), min(int(sr / 70), win - 1)
    lag = lo + np.argmax(ac[:, lo:hi], axis=1)
    strength = ac[np.arange(len(lag)), lag] / np.maximum(ac[:, 0], 1e-9)
    return sr / lag, strength > 0.35


def _snap(freq: float) -> float:
    """Nearest D Dorian note, folded into the ocarina's range."""
    while freq < OCARINA[0]:
        freq *= 2
    while freq > OCARINA[1]:
        freq /= 2
    midi = 69 + 12 * np.log2(freq / 440)
    cands = [m for m in range(int(midi) - 2, int(midi) + 3) if (m - 62) % 12 in DORIAN]
    best = min(cands, key=lambda m: abs(m - midi))
    return float(440 * 2 ** ((best - 69) / 12))


def _osc(freq: np.ndarray, sr: int, harmonics: dict[int, float]) -> np.ndarray:
    """Continuous-phase additive oscillator following a per-sample frequency; partials
    above 0.45 x sr are dropped so it stays clean down the 8 kHz phone path too."""
    phase = 2 * np.pi * np.cumsum(freq) / sr
    out = np.zeros_like(freq)
    for h, amp in harmonics.items():
        out += amp * np.sin(h * phase) * (h * freq < 0.45 * sr)
    return out


_TRIANGLE = {1: 1.0, 3: -1 / 9, 5: 1 / 25}             # warm ocarina / flute
_SINE = {1: 1.0}
_JUG = {1: 1.0, 2: 0.35, 3: 0.12}                      # earthenware jug, didgeridoo-ish
_REED = {1: 1.0, 3: 1 / 3, 5: 1 / 5, 7: 1 / 7, 9: 1 / 9}  # clarinet: odd harmonics


# --- the processor -----------------------------------------------------------------------------

def process(x: np.ndarray, sr: int, text: str = "", final: bool = True) -> np.ndarray:
    """One chunk of speech (float, -1..1) in Rocky's voice.

    Prefix-stable: every stage is causal or looks ahead under STREAM_MARGIN_S, so processing
    more of the same chunk never changes the start of the output. That's what lets Stream send
    audio while the TTS is still generating. `final` adds what needs the end in sight: the
    question chime and the closing fade."""
    voice = _shift(x, PITCH_SEMITONES)
    if len(voice) < sr // 20:  # a click of audio: nothing to sing under
        return _limit(_granite(voice, sr))
    hop = int(sr * HOP_S)
    frames = _frames(voice, hop)
    env = _envelope(frames)
    f0, voiced = _pitch(frames, sr)

    # The chord root follows his pitch, snapped to the scale; held through consonants and gaps,
    # with a 30 ms glide between notes so the layer never steps or clicks.
    root, note = np.empty(len(env)), _snap(220.0)
    for i in range(len(env)):
        if voiced[i] and env[i] > 0.02:
            note = _snap(f0[i])
        root[i] = note if i == 0 else root[i - 1] * (note / root[i - 1]) ** 0.3
    t = np.arange(len(voice))
    centers = np.arange(len(env)) * hop + 2 * hop
    root_s = np.interp(t, centers, root)
    level = env / np.maximum(np.maximum.accumulate(env), 0.02)   # running peak: causal
    env_s = np.interp(t, centers, level)
    loud_s = np.interp(t, centers, env)                          # the voice's own level

    exclaim = bool(_EXCLAIM.search(text))
    speech = _granite(voice if ALIEN <= 0 else  # ROCKY_ALIEN=0: his own shifted voice, no vocoder or device
                      _translate(voice, root_s / 2, np.interp(t, centers, voiced.astype(np.float32)), exclaim, sr), sr)
    breath = _lowpass(np.random.default_rng(7).standard_normal(len(voice)).astype(np.float32), 0.12)
    breath -= _lowpass(breath, 0.02)                    # band-passed air through the ocarina
    layer = (_osc(root_s, sr, _TRIANGLE) + 0.7 * _osc(root_s * 1.5, sr, _TRIANGLE)
             + 0.5 * _osc(root_s * 2, sr, _SINE) + 0.6 * breath) * env_s
    # Jug on stressed syllables: near this chunk's loudest so far.
    layer += 0.9 * _osc(root_s / 4, sr, _JUG) * np.interp(t, centers, np.clip((level - 0.6) / 0.4, 0, 1))
    if exclaim:  # "Amaze!" / "Bad, bad, bad!": the reed swells in
        layer += 0.7 * _osc(root_s, sr, _REED) * env_s ** 0.7
    layer *= LAYER_GAIN * loud_s / np.maximum(env_s, 0.05)       # at the voice's level, sample by sample

    if final and _QUESTION.search(text.strip()):
        layer += _question_chime(level, root, hop, len(voice), sr) * float(np.sqrt(np.mean(voice ** 2))) * 2.2
    edge = min(len(layer) // 4, int(sr * 0.01))
    if edge:  # the layer fades in (and, at the end, out) with the chunk: no clicks at sentence joins
        ramp = np.linspace(0, 1, edge)
        layer[:edge] *= ramp
        if final:
            layer[-edge:] *= ramp[::-1]
    return _limit((1 - CHORD_MIX) * speech + CHORD_MIX * layer)


def _limit(x: np.ndarray) -> np.ndarray:
    """Soft knee above 80% of PEAK, sample by sample (a whole-chunk normalize isn't streamable)."""
    knee = 0.8 * PEAK
    over = np.abs(x) > knee
    if over.any():
        x = x.copy()
        x[over] = np.sign(x[over]) * (knee + (PEAK - knee) * np.tanh((np.abs(x[over]) - knee) / (PEAK - knee)))
    return x


def _translate(voice: np.ndarray, root: np.ndarray, voiced: np.ndarray, exclaim: bool, sr: int) -> np.ndarray:
    """The words as an alien through a translator: Eridian chords vocoded with the voice (ALIEN),
    plus the translator's own thin, metallic rendering of it (1 - ALIEN)."""
    chord = (_saw(root, sr) + 0.6 * _saw(root * 1.5, sr) + 0.45 * _saw(root * 2, sr)) / 0.72  # ~unit RMS
    if exclaim:  # reedier: square-ish odd harmonics under "Amaze!"
        chord += 0.5 * np.sign(np.sin(2 * np.pi * np.cumsum(root) / sr))
    noise = np.random.default_rng(11).standard_normal(len(voice)).astype(np.float32)
    v = np.clip(_lowpass(voiced, 0.002, taps=63), 0, 1)  # ~25 ms voicing crossfade
    # Chords on vowels, breath on consonants (s, t, f): the noise is what keeps the words crisp.
    carrier = chord * (0.25 + 0.75 * v) + noise * (0.35 + 0.65 * (1 - v))
    alien = _vocode(voice, carrier)                       # already at the voice's level, frame by frame
    device = _lowpass(voice, 3600 / sr) - _lowpass(voice, 280 / sr, taps=255)   # small speaker: 280-3600 Hz
    device *= 0.7 + 0.3 * np.sin(2 * np.pi * 55 * np.arange(len(voice)) / sr)    # metallic ring mod
    return ALIEN * alien + (1 - ALIEN) * DEVICE_GAIN * device


def _question_chime(level: np.ndarray, root: np.ndarray, hop: int, n: int, sr: int) -> np.ndarray:
    """A quick rising ocarina glide (+5th -> +octave) rising out of the last syllable into the
    pause after it, with a little cluck at its start: Rocky's question mark. It starts 80 ms
    before the syllable ends, so a stream only has to hold back the pause (Stream)."""
    active = np.nonzero(level > 0.2)[0]
    out = np.zeros(n, dtype=np.float32)
    if not len(active):
        return out
    start = max(0, (active[-1] + 4) * hop - int(sr * 0.08))
    end = min(n, start + int(sr * 0.28))
    span = end - start
    base = float(root[active[-1]])
    freq = base * 1.5 * (2 / 1.5) ** (np.arange(span) / span) ** 0.6   # fifth up to octave, fast then settling
    amp = np.minimum(1, np.arange(span) / (sr * 0.015)) * np.minimum(1, (span - np.arange(span)) / (sr * 0.06))
    out[start:end] = _osc(freq, sr, _TRIANGLE) * amp
    cluck = int(sr * 0.03)
    k = np.arange(min(cluck, span))
    out[start:start + len(k)] += 0.8 * np.sin(2 * np.pi * base * 4 * k / sr) * np.exp(-k / (sr * 0.006))
    return out


class Stream:
    """process() over a piece that's still being generated (voice_ws._tts_stream): it reprocesses
    everything so far and returns only the new audio, minus the last STREAM_MARGIN_S whose
    look-ahead isn't settled yet. A piece ending "question?" holds back enough for its chime.
    finish() returns the rest, chime and fade included.

    Reprocessing runs often at first (the first words go out fast), then each time the piece has
    grown 40%, but never more than STREAM_PASS_MAX_S of new audio between passes: bigger batches
    starved the satellite between them. ~75 ms of CPU per second of audio."""

    def __init__(self, sr: int, text: str) -> None:
        self.sr, self.text = sr, text
        self.x = np.zeros(0, np.float32)
        self.sent = 0
        self.next = sr // 10        # samples needed before the next pass
        hold = 0.5 if _QUESTION.search(text.strip()) else STREAM_MARGIN_S  # the chime lives in the last ~0.4 s
        self.margin = int(sr * hold)

    def feed(self, pcm16: bytes) -> bytes:
        self.x = np.concatenate([self.x, np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0])
        if len(self.x) < self.next:
            return b""
        n = len(self.x)
        self.next = n + max(int(self.sr * 0.08), min(int(n * 0.4), int(self.sr * STREAM_PASS_MAX_S)))
        y = process(self.x, self.sr, self.text, final=False)
        return self._take(y, len(y) - self.margin)

    def finish(self) -> bytes:
        return self._take(process(self.x, self.sr, self.text, final=True), None) if len(self.x) else b""

    def _take(self, y: np.ndarray, upto: int | None) -> bytes:
        upto = len(y) if upto is None else upto
        if upto <= self.sent:
            return b""
        out, self.sent = y[self.sent:upto], upto
        return (np.clip(out, -1, 1) * 32767).astype("<i2").tobytes()


def process_wav(wav: bytes, text: str = "") -> bytes:
    """voice_ws._tts output in, the same WAV format out."""
    x, sr = _read(wav)
    return _write(process(x, sr, text), sr)


# --- phone audio (Twilio media streams: 8 kHz G.711 mu-law) ---------------------------------------

def to_mulaw_8k(x: np.ndarray, sr: int) -> bytes:
    """Processed speech for a phone line: band-limited to 3.4 kHz, 8 kHz, mu-law encoded."""
    y = _lowpass(x, 3400 / sr, taps=191)
    y = y[::sr // 8000] if sr % 8000 == 0 else np.interp(np.arange(0, len(y), sr / 8000), np.arange(len(y)), y)
    return mulaw_encode((np.clip(y, -1, 1) * 32767).astype(np.int16))


def mulaw_encode(pcm: np.ndarray) -> bytes:
    """G.711 mu-law, the standard 0x84-bias segment encoding."""
    s = pcm.astype(np.int32)
    sign = (s < 0).astype(np.int32) << 7
    m = np.minimum(np.abs(s), 32635) + 0x84
    exp = np.clip(np.floor(np.log2(m)).astype(np.int32) - 7, 0, 7)
    mant = (m >> (exp + 3)) & 0x0F
    return (~(sign | (exp << 4) | mant) & 0xFF).astype(np.uint8).tobytes()


def mulaw_decode(data: bytes) -> np.ndarray:
    u = ~np.frombuffer(data, dtype=np.uint8).astype(np.int32) & 0xFF
    exp, mant = (u >> 4) & 0x07, u & 0x0F
    mag = ((mant << 3) + 0x84) << exp
    return np.where(u & 0x80, -(mag - 0x84), mag - 0x84).astype(np.int16)


def mulaw_wav(data: bytes) -> bytes:
    """A playable .wav (format 7, mu-law, 8 kHz mono) around raw mu-law bytes."""
    fmt = struct.pack("<HHIIHHH", 7, 1, 8000, 8000, 1, 8, 0)
    body = (b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"fact" + struct.pack("<II", 4, len(data))
            + b"data" + struct.pack("<I", len(data)) + data + (b"\0" if len(data) % 2 else b""))
    return b"RIFF" + struct.pack("<I", len(body)) + body
