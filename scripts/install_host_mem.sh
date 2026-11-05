#!/bin/bash
# Installs scripts/host_mem.sh as a launchd job (com.thursday.hostmem). The script runs from
# ~/Library/Application Support/thursday: launchd can't read ~/Documents. Re-run after changing it.
set -euo pipefail
cd "$(dirname "$0")/.."
H="$HOME/Library/Application Support/thursday"
mkdir -p "$H" "$HOME/Library/Logs/thursday"
cp scripts/host_mem.sh "$H/host_mem.sh"
cp scripts/launchd/com.thursday.hostmem.plist "$HOME/Library/LaunchAgents/"
launchctl bootout "gui/$(id -u)/com.thursday.hostmem" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.thursday.hostmem.plist"
