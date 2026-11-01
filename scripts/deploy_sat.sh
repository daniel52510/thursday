#!/usr/bin/env bash
# Copies the satellite (code, cue sounds, spoken lines, wake model) to the Pi,
# restarts it and waits for it to reconnect. No sudo: stopping the process is
# enough, systemd (Restart=always) starts the new code a few seconds later.
# The previous satellite.py is kept on the Pi as satellite.py.prev.
#
#   THURSDAY_SAT_SSH=user@pi-host [THURSDAY_SAT_KEY=~/.ssh/key] scripts/deploy_sat.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${THURSDAY_SAT_SSH:?set THURSDAY_SAT_SSH, e.g. pi@192.168.1.20}"
DIR=${THURSDAY_SAT_DIR:-thursday-sat}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=5${THURSDAY_SAT_KEY:+ -i $THURSDAY_SAT_KEY}"

python3 -m py_compile satellite/satellite.py satellite/aec.py
$SSH "$THURSDAY_SAT_SSH" "cd $DIR && cp satellite.py satellite.py.prev && mkdir -p sounds models"
rsync -a --checksum -e "$SSH" satellite/satellite.py satellite/aec.py "$THURSDAY_SAT_SSH:$DIR/"
rsync -a --checksum -e "$SSH" satellite/models/ "$THURSDAY_SAT_SSH:$DIR/models/"
rsync -a --checksum -e "$SSH" satellite/sounds/ assets/sounds/thursday-{ping,thinking,alarm,close}.wav "$THURSDAY_SAT_SSH:$DIR/sounds/"

echo "restarting the satellite"
$SSH "$THURSDAY_SAT_SSH" 'since=$(date "+%F %T"); pkill -f "[p]ython satellite.py" || true  # [p]: never matches this shell itself
for i in $(seq 1 45); do
  sleep 1
  if journalctl -u thursday-satellite --since "$since" --no-pager | grep -q "connected to"; then
    journalctl -u thursday-satellite --since "$since" --no-pager | tail -4; echo "satellite is up"; exit 0
  fi
done
journalctl -u thursday-satellite --since "$since" --no-pager | tail -20; echo "satellite did not reconnect" >&2; exit 1'
