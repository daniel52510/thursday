#!/usr/bin/env bash
# Builds THURSDAY Music Tap (mac/music_tap) into an app bundle outside ~/Documents (launch agents
# can't read it), signs it ad hoc, makes its token if there isn't one, and (re)starts its launch
# agent. macOS asks once for "System Audio Recording"; a rebuild changes the ad-hoc signature, so
# it asks again. Undo: scripts/build_music_tap.sh --remove
set -euo pipefail
cd "$(dirname "$0")/.."
APP="$HOME/Library/Application Support/thursday/THURSDAY Music Tap.app"
PLIST=com.thursday.musictap.plist
AGENTS=~/Library/LaunchAgents
launchctl bootout "gui/$(id -u)/com.thursday.musictap" 2>/dev/null || true
rm -f "$AGENTS/$PLIST"
if [ "${1:-}" = "--remove" ]; then rm -rf "$APP"; echo "removed"; exit 0; fi

mkdir -p "$APP/Contents/MacOS" ~/Library/Logs/thursday ~/.config/thursday
cp mac/music_tap/Info.plist "$APP/Contents/Info.plist"
swiftc -O -swift-version 5 mac/music_tap/main.swift -o "$APP/Contents/MacOS/music-tap" \
  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker mac/music_tap/Info.plist
codesign --force --sign - --identifier com.thursday.musictap "$APP"

# The token core checks (MUSIC_TAP_TOKEN in .env) and the tap sends. Made once, never printed.
TOKEN_FILE=~/.config/thursday/music_tap_token
if [ ! -s "$TOKEN_FILE" ]; then (umask 077; openssl rand -hex 24 > "$TOKEN_FILE"); fi
if ! grep -q '^MUSIC_TAP_TOKEN=' .env 2>/dev/null; then
  printf '\n# THURSDAY Music Tap (scripts/build_music_tap.sh): same value as ~/.config/thursday/music_tap_token\nMUSIC_TAP_TOKEN=%s\n' "$(cat "$TOKEN_FILE")" >> .env
  echo "added MUSIC_TAP_TOKEN to .env: recreate thursday-core to load it"
fi

sed -e "s|__APP__|$APP|g" -e "s|__HOME__|$HOME|g" "scripts/launchd/$PLIST" > "$AGENTS/$PLIST"
plutil -lint -s "$AGENTS/$PLIST"
launchctl bootstrap "gui/$(id -u)" "$AGENTS/$PLIST"
echo "started; log: ~/Library/Logs/thursday/music-tap.log"
