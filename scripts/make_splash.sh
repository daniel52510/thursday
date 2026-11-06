#!/usr/bin/env bash
# Renders the iPhone launch screens for the installed web app (assets/splash/*.png): the app icon centered on
# the dark background, one PNG per screen size, so opening THURSDAY from the home screen shows its own splash
# instead of a white flash. Uses Playwright's headless Chromium (already on this Mac); re-run if the icon changes.
#   scripts/make_splash.sh
set -euo pipefail
cd "$(dirname "$0")/.."
HS=$(ls -d ~/Library/Caches/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-mac-arm64/chrome-headless-shell | tail -1)
OUT=assets/splash
TMP=$(mktemp -d)
mkdir -p "$OUT"
ICON="file://$PWD/assets/icons/icon.svg"
# width x height in device pixels (portrait): the iPhones since the X, newest first
for size in 1320x2868 1206x2622 1290x2796 1179x2556 1284x2778 1170x2532 1125x2436 1242x2688 828x1792 750x1334; do
  w=${size%x*}; h=${size#*x}; icon=$((w * 32 / 100))
  cat > "$TMP/s.html" <<EOF
<!doctype html><html><body style="margin:0;background:#12110E;width:${w}px;height:${h}px;display:flex;align-items:center;justify-content:center">
<img src="$ICON" style="width:${icon}px;height:${icon}px;border-radius:$((icon * 22 / 100))px"></body></html>
EOF
  rm -f "$OUT/splash-$size.png"
  "$HS" --disable-gpu --hide-scrollbars --force-device-scale-factor=1 --window-size="$w,$h" \
    --screenshot="$OUT/splash-$size.png" "file://$TMP/s.html" >/dev/null 2>&1 &
  pid=$!
  for _ in $(seq 1 30); do sleep 0.5; [ -s "$OUT/splash-$size.png" ] && break; done
  kill "$pid" 2>/dev/null || true
  echo "$OUT/splash-$size.png"
done
rm -rf "$TMP"
