#!/usr/bin/env bash
# THURSDAY on a fresh Apple Silicon Mac, in one go (also: `make setup`). Safe to re-run: it never overwrites your
# .env or vault key, and skips what's already done. Needs: Docker Desktop, Ollama, Python 3.12 (Homebrew).
#
#   scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
step() { printf '\n\033[1m▸ %s\033[0m\n' "$*"; }
need() { command -v "$1" >/dev/null || { echo "missing: $1 ($2)"; exit 1; }; }

step "Checking prerequisites"
need docker "install Docker Desktop: https://www.docker.com/products/docker-desktop"
need ollama "install Ollama: https://ollama.com/download"
need python3.12 "brew install python@3.12"
docker info >/dev/null 2>&1 || { echo "Docker Desktop isn't running: open it and re-run"; exit 1; }
[ "$(uname -m)" = arm64 ] || echo "note: built for Apple Silicon; the native GPU services won't be fast elsewhere"

step "Config (.env)"
if [ -f .env ]; then
  echo ".env exists: leaving it alone"
else
  cp .env.example .env
  rnd() { python3 -c "import secrets; print(secrets.token_urlsafe($1))"; }
  sed -i '' "s|^SEARXNG_SECRET=.*|SEARXNG_SECRET=$(rnd 32)|; s|^NTFY_TOPIC=.*|NTFY_TOPIC=thursday-$(rnd 18)|" .env
  echo "wrote .env with fresh random secrets: open it and set your name, username and timezone"
fi

step "Vault key (encrypts saved credentials; outside the repo)"
KEY="$HOME/.config/thursday/master.key"
if [ -s "$KEY" ]; then echo "exists"; else
  mkdir -p "$(dirname "$KEY")" && umask 077 && python3 -c "import os, base64; print(base64.b64encode(os.urandom(32)).decode())" > "$KEY"
  echo "created $KEY (back it up: without it, saved credentials can't be read)"
fi
mkdir -p brain

step "Language model (Ollama, on the GPU)"
MODEL=$(grep -E '^OLLAMA_MODEL=' .env | cut -d= -f2)
ollama list | grep -q "^${MODEL:-gemma4:26b}" && echo "${MODEL} already pulled" || ollama pull "${MODEL:-gemma4:26b}"

step "Native speech-to-text (GPU) under launchd"
[ -x .venv-stt/bin/uvicorn ] || PYTHON=python3.12 bash -c 'scripts/run_stt_native.sh & pid=$!; sleep 90; kill $pid' || true
scripts/install_stt_launchd.sh || echo "STT didn't start: the CPU container takes over (slower). Check ~/Library/Logs/thursday/stt.log"

step "Mac status helper (memory, CPU, GPU for the System page)"
scripts/install_host_mem.sh

step "Containers (core, STT fallback, search, ntfy)"
docker compose up -d --build

step "Done"
cat <<EOF
Next:
  1. Text-to-speech: scripts/run_tts_native.sh (Qwen on :9000), or point THURSDAY_TTS_PORT at your own voice server.
  2. Set your password:   docker exec -it thursday-core python accounts.py passwd $(grep -E '^THURSDAY_OWNER_USERNAME=' .env | cut -d= -f2)
  3. Open http://localhost:8000 (or from your phone: http://<this Mac's LAN IP>:8000, or over Tailscale).
  4. Optional: satellite (notes at the top of satellite/satellite.py), phone calls and Apple Health (README), Tailscale for away-from-home.
Health: open the System page in the sidebar; everything should be green.
EOF
