#!/usr/bin/env bash
# Installs launch agents so the native voice (:9001) and speech-to-text (:9300)
# servers start at login and restart if they exit. Stops any copies started by
# hand first, so the ports are free. Undo: scripts/launchd/install.sh --remove
# Re-run after changing stt_native.py: the STT server runs a copy (see STT below).
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd)
AGENTS=~/Library/LaunchAgents
# Launch agents can't read ~/Documents (macOS privacy), where the repo lives, so the STT
# server runs from its own folder: stt_native.py + a venv built from the lock file.
STT=~/thursday-stt
mkdir -p "$AGENTS" ~/Library/Logs/thursday
if [ "${1:-}" != "--remove" ]; then
  mkdir -p "$STT"
  cp "$REPO/stt_native.py" "$REPO/requirements-stt-native.lock.txt" "$STT/"
  if [ ! -x "$STT/.venv-stt/bin/uvicorn" ]; then
    rm -rf "$STT/.venv-stt"
    /opt/homebrew/bin/python3.12 -m venv "$STT/.venv-stt"
    "$STT/.venv-stt/bin/pip" install -q --upgrade pip
    "$STT/.venv-stt/bin/pip" install -q -r "$STT/requirements-stt-native.lock.txt"
  fi
fi
for plist in com.thursday.voice.plist com.thursday.stt.plist; do
  label=${plist%.plist}
  launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
  rm -f "$AGENTS/$plist"
  [ "${1:-}" = "--remove" ] && { echo "removed $label"; continue; }
  sed -e "s|__HOME__|$HOME|g" -e "s|__REPO__|$REPO|g" -e "s|__STT__|$STT|g" "$plist" > "$AGENTS/$plist"
  plutil -lint -s "$AGENTS/$plist"
done
[ "${1:-}" = "--remove" ] && exit 0
pkill -f "scripts/serve_orpheus.py" || true
pkill -f "uvicorn stt_native:app" || true
sleep 2
for plist in com.thursday.voice.plist com.thursday.stt.plist; do
  launchctl bootstrap "gui/$(id -u)" "$AGENTS/$plist"
done
for check in "voice http://127.0.0.1:9001/health" "stt http://127.0.0.1:9300/health"; do
  set -- $check
  for i in $(seq 1 120); do curl -sf -m 3 -o /dev/null "$2" && { echo "$1 up (${i}s)"; break; }; sleep 1; done
done
