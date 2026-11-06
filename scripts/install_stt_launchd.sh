#!/usr/bin/env bash
# Puts native speech-to-text (Parakeet, :9300) under launchd so it starts at login and restarts if it dies.
# launchd can't run anything inside ~/Documents (macOS privacy, TCC: the job exits 126), so the server and its
# venv are copied to ~/thursday-stt, like the voice server in ~/thursday-voice. Re-run after changing
# stt_native.py or requirements-stt-native.txt. Installs no packages: it copies the existing .venv-stt.
#
#   scripts/install_stt_launchd.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="$HOME/thursday-stt"
PLIST="$HOME/Library/LaunchAgents/com.thursday.stt.plist"
LOGS="$HOME/Library/Logs/thursday"
[ -x .venv-stt/bin/python ] || { echo "no .venv-stt here: run scripts/run_stt_native.sh once first" >&2; exit 1; }

mkdir -p "$DEST" "$LOGS"
rsync -a --delete .venv-stt/ "$DEST/venv/"
cp stt_native.py "$DEST/"
cat > "$DEST/run.sh" <<EOF
#!/usr/bin/env bash
# Written by scripts/install_stt_launchd.sh in the THURSDAY repo; re-run that instead of editing here.
cd "$DEST"
exec "$DEST/venv/bin/python" -m uvicorn stt_native:app --host 127.0.0.1 --port \${STT_NATIVE_PORT:-9300} --log-level info
EOF
chmod +x "$DEST/run.sh"

plutil -replace ProgramArguments -json "[\"$DEST/run.sh\"]" "$PLIST"
plutil -replace WorkingDirectory -string "$DEST" "$PLIST"
plutil -replace StandardOutPath -string "$LOGS/stt.log" "$PLIST"
plutil -replace StandardErrorPath -string "$LOGS/stt.log" "$PLIST"

# The hand-started server holds the port: stop it, then let launchd own the job.
pkill -f "uvicorn stt_native:app" || true
launchctl bootout "gui/$(id -u)/com.thursday.stt" 2>/dev/null || true
sleep 1
launchctl bootstrap "gui/$(id -u)" "$PLIST"
for i in $(seq 1 60); do
  sleep 1
  if curl -sf http://127.0.0.1:9300/health >/dev/null; then echo "stt is up under launchd ($i s)"; exit 0; fi
done
tail -20 "$LOGS/stt.log"; echo "stt did not come up" >&2; exit 1
