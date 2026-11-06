#!/usr/bin/env bash
# The satellite's tuning numbers from its own logs: how often it woke for nothing, how close the false wakes and
# barge-ins sat to their thresholds, and how often the wake greeting played or was skipped. Read-only.
# Run after a few days of normal use, then tune THURSDAY_WAKE_THRESHOLD / _FRAMES, THURSDAY_BARGE_FLOOR_DBFS,
# THURSDAY_GREET_WAIT_MS in the Pi's ~/.config/thursday/satellite.env from what it shows.
#
#   THURSDAY_SAT_SSH=user@pi-host [THURSDAY_SAT_KEY=~/.ssh/key] scripts/sat_report.sh [since, default "7 days ago"]
set -euo pipefail
SINCE="${1:-7 days ago}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=5${THURSDAY_SAT_KEY:+ -i $THURSDAY_SAT_KEY}"
$SSH "$THURSDAY_SAT_SSH" "journalctl -u thursday-satellite --since '$SINCE' --no-pager -o short-iso" | python3 -c '
import re, sys
from datetime import datetime
lines = sys.stdin.read().splitlines()
ts = lambda l: datetime.fromisoformat(l.split()[0][:19])
wakes, near, barge, greet, skip = [], [], [], 0, 0
for i, l in enumerate(lines):
    if m := re.search(r"INFO wake \(score ([\d.]+)", l):
        t0, asked = ts(l), False
        for later in lines[i + 1:]:  # a turn within 40 s of the wake = he asked something
            if (ts(later) - t0).total_seconds() > 40 or "INFO wake (score" in later:
                break
            if "INFO turn:" in later:
                asked = True
                break
        wakes.append((t0, float(m[1]), asked))
    elif m := re.search(r"near miss \(score ([\d.]+)", l):
        near.append(float(m[1]))
    elif m := re.search(r"barge-in \(speech, (-?\d+) dBFS over a (-?\d+) dBFS bar\)", l):
        barge.append(int(m[1]) - int(m[2]))
    elif "kept talking after the wake word" in l:
        skip += 1
    elif "INFO greetings:" in l:
        pass
wakes_with_turn = sum(w[2] for w in wakes)
night = [w for w in wakes if w[0].hour < 7 and not w[2]]
print(f"since: {sys.argv[1]}")
print(f"wakes: {len(wakes)}  with a request: {wakes_with_turn}  for nothing: {len(wakes) - wakes_with_turn}  (overnight 00-07, for nothing: {len(night)})")
nothing = sorted(round(w[1], 2) for w in wakes if not w[2])
if nothing:
    print(f"  false-wake scores: min {nothing[0]}  median {nothing[len(nothing)//2]}  max {nothing[-1]}")
real = sorted(round(w[1], 2) for w in wakes if w[2])
if real:
    print(f"  real-wake scores:  min {real[0]}  median {real[len(real)//2]}  max {real[-1]}")
if near:
    print(f"near misses: {len(near)}  top scores: {sorted(near)[-5:]}")
print(f"speech barge-ins: {len(barge)}" + (f"  margin over the bar (dB): {sorted(barge)}" if barge else ""))
print(f"  within 3 dB of the bar (likely echo, not him): {sum(1 for b in barge if b <= 3)}")
print(f"greetings skipped because he kept talking: {skip}")
' "$SINCE"
