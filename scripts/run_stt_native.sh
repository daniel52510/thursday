#!/usr/bin/env bash
# Run speech-to-text natively on macOS so it can use the Apple GPU (MLX), on
# http://127.0.0.1:9300. Docker Desktop on Mac has no GPU passthrough, so the
# thursday-stt container is CPU-only (kept as the fallback).
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=.venv-stt
PYTHON="${PYTHON:-$(command -v python3.12 || command -v python3)}"

# Check for uvicorn rather than the folder so a half-finished install gets redone.
if [ ! -x "$VENV/bin/uvicorn" ]; then
  rm -rf "$VENV"
  "$PYTHON" -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install -r requirements-stt-native.txt
fi

exec "$VENV/bin/uvicorn" stt_native:app --host 127.0.0.1 --port "${STT_NATIVE_PORT:-9300}" --log-level info
