"""
Round-trip smoke test for the THURSDAY voice pipeline.

Sends a known phrase to the TTS service, feeds the resulting audio into the
STT service, and compares the transcribed text against the original phrase
using a simple word-overlap ratio.

Run this on thursdayserver after building/starting both the thursday-tts and
thursday-stt containers:

    python voice_loop_test.py
"""
import os
import sys

import requests

TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://localhost:9000").rstrip("/")
STT_BASE_URL = os.getenv("STT_BASE_URL", "http://localhost:9200").rstrip("/")

TEST_PHRASE = "The quick brown fox jumps over the lazy dog near the river."

PASS_THRESHOLD = 0.8
WARN_THRESHOLD = 0.5


def fetch_tts_audio(text: str) -> bytes:
    resp = requests.post(f"{TTS_BASE_URL}/speak", json={"text": text}, timeout=180)
    resp.raise_for_status()
    payload = resp.json()
    audio_url = payload["audio_url"]
    if audio_url.startswith("/"):
        audio_url = f"{TTS_BASE_URL}{audio_url}"
    audio_resp = requests.get(audio_url, timeout=180)
    audio_resp.raise_for_status()
    return audio_resp.content


def transcribe_audio(audio_bytes: bytes) -> str:
    files = {"audio": ("roundtrip.wav", audio_bytes, "audio/wav")}
    resp = requests.post(f"{STT_BASE_URL}/transcribe", files=files, timeout=120)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error") or "Transcription failed")
    return (payload.get("text") or "").strip()


def word_overlap_ratio(expected: str, actual: str) -> float:
    expected_words = set(expected.lower().split())
    actual_words = set(actual.lower().split())
    if not expected_words:
        return 0.0
    return len(expected_words & actual_words) / len(expected_words)


def main() -> int:
    print(f"TTS:    {TTS_BASE_URL}")
    print(f"STT:    {STT_BASE_URL}")
    print(f"Phrase: {TEST_PHRASE!r}\n")

    print("Generating TTS audio...")
    audio_bytes = fetch_tts_audio(TEST_PHRASE)
    print(f"Got {len(audio_bytes)} bytes of audio.\n")

    print("Transcribing audio back...")
    transcribed = transcribe_audio(audio_bytes)
    print(f"Transcribed: {transcribed!r}\n")

    ratio = word_overlap_ratio(TEST_PHRASE, transcribed)
    print(f"Word overlap ratio: {ratio:.2f}")

    if ratio >= PASS_THRESHOLD:
        print("PASS: voice loop round-trip looks healthy.")
        return 0
    elif ratio >= WARN_THRESHOLD:
        print("WARN: round-trip works but transcription quality is degraded.")
        return 0
    else:
        print("FAIL: transcribed text does not match the original phrase.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
