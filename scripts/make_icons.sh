#!/bin/sh
# Render the app icons (home screen / PWA) from assets/icons/icon.svg.
# macOS only: uses Quick Look (qlmanage) and sips, no extra installs.
# After regenerating, bump ?v= on the icon links in static/index.html and
# assets/manifest.json so browsers and iOS stop showing the cached icon.
set -e
cd "$(dirname "$0")/../assets/icons"
tmp=$(mktemp -d)
qlmanage -t -s 512 -o "$tmp" icon.svg >/dev/null
cp "$tmp/icon.svg.png" icon-512.png
sips -z 192 192 icon-512.png --out icon-192.png >/dev/null
sips -z 180 180 icon-512.png --out apple-touch-icon.png >/dev/null
rm -rf "$tmp"
echo "wrote icon-512.png icon-192.png apple-touch-icon.png"
