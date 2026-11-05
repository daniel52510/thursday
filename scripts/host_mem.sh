#!/bin/bash
# Writes the Mac's memory state to ~/Library/Application Support/thursday/host-mem.json for the runner (in Docker's VM,
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
tmp="$OUT.tmp"
printf '{"at": %d, "available_gb": %.1f, "swap_used_mb": %.0f, "swap_total_mb": %.0f, "pressure": %d}\n' \
  "$(date +%s)" "$avail_gb" "$swap_used" "$swap_total" "$pressure" > "$tmp"
mv "$tmp" "$OUT"
