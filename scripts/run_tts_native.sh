#!/usr/bin/env bash
# Run the TTS service natively on macOS so it can use the Apple GPU (MPS).
# Docker Desktop on Mac has no GPU passthrough, so the container would be CPU-only.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=.venv-tts
# macOS's /usr/bin/python3 is 3.9, too old for qwen-tts; prefer Homebrew's 3.12.
PYTHON="${PYTHON:-$(command -v python3.12 || command -v python3)}"

# Check for uvicorn rather than the folder so a half-finished install gets redone.
if [ ! -x "$VENV/bin/uvicorn" ]; then
  rm -rf "$VENV"
  "$PYTHON" -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install -r requirements-tts.txt
fi

# Ops that MPS doesn't implement yet fall back to CPU instead of erroring.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TTS_DEVICE="${TTS_DEVICE:-auto}"
export TTS_DTYPE="${TTS_DTYPE:-auto}"

# Don't let Ctrl+C wait forever on an open /speak request.
exec "$VENV/bin/uvicorn" voice_out:app --host 0.0.0.0 --port 9000 --timeout-graceful-shutdown 5
