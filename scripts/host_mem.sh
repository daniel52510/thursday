#!/bin/bash
# Writes the Mac's memory, CPU and GPU state to ~/Library/Application Support/thursday/host-mem.json for the runner (in Docker's VM,
# it can't see them). The runner waits before loading gemma4:31b while memory is short: on 2026-11-05 the
# 31b (~30 GB) next to the 26b chat model pushed swap to 99% and Ollama hung for everyone.
# Run every 15 s by launchd (com.thursday.hostmem). Read-only: it only reads vm_stat.
set -euo pipefail
# Outside ~/Documents: launchd jobs can't touch it (macOS privacy), and runner-out/*.json are job reports
# (improvements.py would push it to his phone as one). Install: scripts/install_host_mem.sh.
OUT="$HOME/Library/Application Support/thursday/host-mem.json"
page=$(sysctl -n hw.pagesize)
read -r free inactive spec < <(vm_stat | awk '/Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} END {gsub("\\.","",f); gsub("\\.","",i); gsub("\\.","",s); print f, i, s}')
swap_used=$(sysctl -n vm.swapusage | awk '{for (i=1;i<=NF;i++) if ($i=="used") {v=$(i+2); sub("M","",v); print v}}')
swap_total=$(sysctl -n vm.swapusage | awk '{for (i=1;i<=NF;i++) if ($i=="total") {v=$(i+2); sub("M","",v); print v}}')
pressure=$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 0)  # 1 normal, 2 warn, 4 critical
avail_gb=$(echo "($free + $inactive + $spec) * $page / 1073741824" | bc -l)
total_gb=$(echo "$(sysctl -n hw.memsize) / 1073741824" | bc -l)
# CPU: top's second sample (the first is since boot), all cores together. Load: the 1-minute average.
cpu_idle=$(top -l 2 -s 1 -n 0 | awk '/CPU usage/ {v=$7} END {sub("%","",v); print v+0}')
load1=$(sysctl -n vm.loadavg | awk '{print $2}')
ncpu=$(sysctl -n hw.ncpu)
# GPU: Apple's own counters, no sudo (Device Utilization is what Activity Monitor's GPU history shows).
stats=$(ioreg -r -d 1 -c IOAccelerator)
gpu_pct=$(grep -o '"Device Utilization %"=[0-9]*' <<<"$stats" | head -1 | cut -d= -f2)
gpu_mem=$(grep -o '"In use system memory"=[0-9]*' <<<"$stats" | head -1 | cut -d= -f2)
gpu_pct=${gpu_pct:-0}; gpu_mem=${gpu_mem:-0}
tmp="$OUT.tmp"
printf '{"at": %d, "available_gb": %.1f, "total_gb": %.0f, "swap_used_mb": %.0f, "swap_total_mb": %.0f, "pressure": %d, "cpu_pct": %.0f, "load1": %.2f, "cores": %d, "gpu_pct": %d, "gpu_mem_gb": %.1f}\n' \
  "$(date +%s)" "$avail_gb" "$total_gb" "$swap_used" "$swap_total" "$pressure" "$(echo "100 - $cpu_idle" | bc -l)" \
  "$load1" "$ncpu" "$gpu_pct" "$(echo "$gpu_mem / 1073741824" | bc -l)" > "$tmp"
mv "$tmp" "$OUT"

# Presence (presence.py): is his iPhone home? Once a minute, a Tailscale ping to it: a home-network address, or
# the same public address as this Mac, means home; another address means away; no reply means offline. Only the
# verdict is written (presence.json), never an address. Read-only: it changes nothing on the tailnet.
PRES="$(dirname "$OUT")/presence.json"
if [ ! -f "$PRES" ] || [ $(( $(date +%s) - $(stat -f %m "$PRES") )) -ge 55 ]; then
  TS=$(command -v tailscale || echo /usr/local/bin/tailscale)
  phone=$("$TS" status --json 2>/dev/null | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); print(next((p['TailscaleIPs'][0] for p in (d.get('Peer') or {}).values() if p.get('OS')=='iOS' and p.get('Online')), ''))" 2>/dev/null)
  state=offline
  if [ -n "$phone" ]; then
    via=$("$TS" ping --c 2 --timeout 3s "$phone" 2>/dev/null | grep -oE "via [0-9.]+" | tail -1 | cut -d' ' -f2)
    mine=$("$TS" netcheck --format=json 2>/dev/null | /usr/bin/python3 -c "import json,sys; print(json.load(sys.stdin).get('GlobalV4','').split(':')[0])" 2>/dev/null)
    if [ -z "$via" ]; then state=unknown
    elif [[ "$via" =~ ^(192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[01])\.) ]] || [ "$via" = "$mine" ]; then state=home
    else state=away; fi
  fi
  # Keep the last five readings: presence.py smooths them (a waking phone sometimes takes the internet route
  # for a moment while home, so one "away" reading isn't enough).
  /usr/bin/python3 - "$PRES" "$state" <<'PY'
import json, sys, time
path, state = sys.argv[1], sys.argv[2]
try:
    recent = json.load(open(path)).get("recent", [])
except Exception:
    recent = []
recent = (recent + [state])[-5:]
json.dump({"at": int(time.time()), "phone": state, "recent": recent}, open(path + ".tmp", "w"))
PY
  mv "$PRES.tmp" "$PRES"
fi
