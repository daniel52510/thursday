#!/usr/bin/env bash
# Run the TTS service natively on macOS so it can use the Apple GPU (MPS).
# Docker Desktop on Mac has no GPU passthrough, so the container would be CPU-only.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=.venv-tts
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install -r requirements-tts.txt
fi

# Ops that MPS doesn't implement yet fall back to CPU instead of erroring.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TTS_DEVICE="${TTS_DEVICE:-auto}"
export TTS_DTYPE="${TTS_DTYPE:-auto}"

exec "$VENV/bin/uvicorn" voice_out:app --host 0.0.0.0 --port 9000
