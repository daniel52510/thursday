"""THURSDAY Pi satellite: waits for "Hey THURSDAY", streams the question to
the Mac's /ws/voice endpoint, and plays the spoken reply on the HAT speaker.

    wake --"Hey THURSDAY"--> listening --server "thinking"--> thinking --WAV--> speaking
      ^                          |                               |                |
      +------ server "idle" -----+-------- error / idle ---------+-- playback end -+

Audio in: arecord, 16 kHz mono int16, 1280-sample (80 ms) chunks, the step
size openWakeWord wants; /ws/voice re-slices whatever it gets.
Audio out: aplay, one process at a time (the HAT is a single ALSA device).

    THURSDAY_WS=ws://<mac-ip>:8000/ws/voice venv/bin/python satellite.py
"""
import asyncio
import json
import logging
import os
import sys
import time
import types
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
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280
CHUNK_BYTES = CHUNK_SAMPLES * 2
SILENCE = bytes(CHUNK_BYTES)
DEVICE = os.getenv("THURSDAY_ALSA_DEVICE", "plughw:CARD=wm8960soundcard")
SERVER_URL = os.getenv("THURSDAY_WS", "")
TOKEN_FILE = Path(os.getenv("THURSDAY_TOKEN_FILE", "~/.config/thursday/token")).expanduser()
WAKE_MODEL = os.getenv("THURSDAY_WAKE_MODEL", str(HERE / "models" / "hey_thursday.onnx"))
WAKE_THRESHOLD = float(os.getenv("THURSDAY_WAKE_THRESHOLD", "0.5"))
# After a reply, listen again without the wake word. The server's no-speech
# timeout (VOICE_NO_SPEECH_TIMEOUT_MS) ends the conversation if nobody talks.
FOLLOW_UP = os.getenv("THURSDAY_FOLLOW_UP", "1") == "1"
PING_WAV = HERE / "sounds" / "thursday-ping.wav"
THINKING_WAV = HERE / "sounds" / "thursday-thinking.wav"
RECONNECT_S = 3.0

log = logging.getLogger("satellite")


def _ms(since: float) -> int:
    return int((time.monotonic() - since) * 1000)


class Player:
    """One aplay at a time; starting a new sound stops the current one."""

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._loop_task: asyncio.Task | None = None

    async def _spawn(self, source: Path | None) -> asyncio.subprocess.Process:
        args = ["aplay", "-q", "-D", DEVICE]
        if source is None:
            return await asyncio.create_subprocess_exec(*args, "-", stdin=asyncio.subprocess.PIPE)
        return await asyncio.create_subprocess_exec(*args, str(source))

    async def stop(self) -> None:
        if self._loop_task:
            self._loop_task.cancel()
            self._loop_task = None
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            await self._proc.wait()
        self._proc = None

    async def play_file(self, path: Path) -> None:
        """Start a sound without waiting for it to finish."""
        await self.stop()
        self._proc = await self._spawn(path)

    async def loop_file(self, path: Path) -> None:
        await self.stop()

        async def _loop() -> None:
            while True:
                self._proc = await self._spawn(path)
                await self._proc.wait()

        self._loop_task = asyncio.create_task(_loop())

    async def play_wav_bytes(self, wav: bytes, on_start) -> None:
        """Play a WAV from memory and wait until it has finished."""
        await self.stop()
        self._proc = proc = await self._spawn(None)
        await on_start()
        try:
            proc.stdin.write(wav)
            await proc.stdin.drain()
            proc.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            pass
        await proc.wait()

    def is_playing(self) -> bool:
        return self._proc is not None and self._proc.returncode is None


class Satellite:
    def __init__(self, token: str) -> None:
        self.token = token
        self.wake = Model(wakeword_models=[WAKE_MODEL], inference_framework="onnx")
        self.wake_name = next(iter(self.wake.models))
        self.player = Player()
        self.state = "wake"
        self._woke_at = 0.0
        self._thinking_at = 0.0

    async def run(self) -> None:
        mic = await asyncio.create_subprocess_exec(
            "arecord", "-q", "-D", DEVICE, "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw",
            stdout=asyncio.subprocess.PIPE,
        )
        try:
            while True:
                try:
                    async with connect(f"{SERVER_URL}?token={quote(self.token)}", max_size=None) as ws:
                        log.info("connected to %s; waiting for wake word", SERVER_URL)
                        await asyncio.gather(self._mic_loop(mic, ws), self._recv_loop(ws))
                except (OSError, ConnectionClosed) as exc:
                    log.warning("connection lost (%s); retrying in %.0fs", exc, RECONNECT_S)
                await self.player.stop()
                self._enter_wake()
                await asyncio.sleep(RECONNECT_S)
        finally:
            if mic.returncode is None:
                mic.terminate()

    def _enter_wake(self) -> None:
        # Clear openWakeWord's rolling buffers so audio from the last turn
        # (including our own speaker) can't trigger a stale detection.
        self.wake.reset()
        self.state = "wake"

    async def _mic_loop(self, mic, ws) -> None:
        while True:
            try:
                chunk = await mic.stdout.readexactly(CHUNK_BYTES)
            except asyncio.IncompleteReadError:
                raise SystemExit("arecord stopped")
            if self.state == "wake":
                # ~50 ms of CPU per chunk; off the event loop so websocket traffic keeps flowing.
                scores = await asyncio.to_thread(self.wake.predict, np.frombuffer(chunk, dtype=np.int16))
                score = scores[self.wake_name]
                if score >= WAKE_THRESHOLD:
                    log.info("wake (score %.2f)", score)
                    await self._start_listening(ws)
            elif self.state == "listening":
                # The speaker sits next to the mics: send silence while the ping
                # plays so the server's VAD doesn't take the chime for speech.
                await ws.send(SILENCE if self.player.is_playing() else chunk)
            # thinking / speaking: keep draining arecord, send nothing (barge-in is a later slice).

    async def _start_listening(self, ws) -> None:
        self._woke_at = time.monotonic()
        self.state = "listening"
        await self.player.play_file(PING_WAV)
        await ws.send(json.dumps({"type": "start"}))

    async def _recv_loop(self, ws) -> None:
        async for msg in ws:
            if isinstance(msg, bytes):
                await self._speak(ws, msg)
                continue
            ev = json.loads(msg)
            kind = ev.get("type")
            if kind == "state":
                await self._on_state(ws, ev["state"])
            elif kind == "transcript":
                log.info("heard: %s", ev.get("text", ""))
            elif kind == "reply":
                log.info("reply: %s", ev.get("text", ""))
            elif kind == "error":
                log.error("%s error: %s", ev.get("stage"), ev.get("message"))

    async def _on_state(self, ws, state: str) -> None:
        if state == "thinking":
            self._thinking_at = time.monotonic()
            self.state = "thinking"
            log.info("thinking (listened %d ms)", _ms(self._woke_at))
            await self.player.loop_file(THINKING_WAV)
        elif state == "idle" and self.state == "speaking" and FOLLOW_UP:
            # The server's answer to our playback_ended: keep the conversation going.
            log.info("reply done; listening for a follow-up")
            await self._start_listening(ws)
        elif state == "idle" and self.state != "wake":
            # No speech, an error, or the end of a reply: back to the wake word.
            await self.player.stop()
            self._enter_wake()
            log.info("idle; waiting for wake word")

    async def _speak(self, ws, wav: bytes) -> None:
        self.state = "speaking"
        log.info("reply audio %d KB after %d ms thinking", len(wav) // 1024, _ms(self._thinking_at))

        async def started() -> None:
            await ws.send(json.dumps({"type": "playback_started"}))

        await self.player.play_wav_bytes(wav, started)
        await ws.send(json.dumps({"type": "playback_ended"}))


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
