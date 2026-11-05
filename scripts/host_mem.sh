#!/bin/bash
# Writes the Mac's memory state to runner-out/host-mem.json for the runner, which lives in Docker's VM
# and can't see it. The runner waits before loading gemma4:31b while memory is short: on 2026-11-05 the
# 31b (~30 GB) next to the 26b chat model pushed swap to 99% and Ollama hung for everyone.
# Run every 15 s by launchd (scripts/launchd/com.thursday.hostmem.plist). Read-only: it only reads vm_stat.
set -euo pipefail
OUT="$(cd "$(dirname "$0")/.." && pwd)/runner-out/host-mem.json"
page=$(sysctl -n hw.pagesize)
read -r free inactive spec < <(vm_stat | awk '/Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} END {gsub("\\.","",f); gsub("\\.","",i); gsub("\\.","",s); print f, i, s}')
swap_used=$(sysctl -n vm.swapusage | awk '{for (i=1;i<=NF;i++) if ($i=="used") {v=$(i+2); sub("M","",v); print v}}')
swap_total=$(sysctl -n vm.swapusage | awk '{for (i=1;i<=NF;i++) if ($i=="total") {v=$(i+2); sub("M","",v); print v}}')
pressure=$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 0)  # 1 normal, 2 warn, 4 critical
avail_gb=$(echo "($free + $inactive + $spec) * $page / 1073741824" | bc -l)
tmp="$OUT.tmp"
printf '{"at": %d, "available_gb": %.1f, "swap_used_mb": %.0f, "swap_total_mb": %.0f, "pressure": %d}\n' \
  "$(date +%s)" "$avail_gb" "$swap_used" "$swap_total" "$pressure" > "$tmp"
mv "$tmp" "$OUT"
