#!/usr/bin/env bash
# Host-side resource logger for HAMS container runs.
#
# The 2026-07-13 crashes were host RAM exhaustion (kernel OOM log showed
# ~29 GB/30 GB in anon pages, swap fully exhausted) that froze the desktop
# before anyone could inspect what was ballooning. This script runs OUTSIDE
# any container, samples system + per-container memory every INTERVAL_SEC,
# and fsyncs after every line so the log survives a hard freeze / forced
# reboot — the whole point is to have data from the seconds BEFORE a crash,
# not just the kernel's post-mortem OOM dump (which only names one victim,
# not the actual hog).
#
# Usage: docker/scripts/monitor_resources.sh [interval_sec] [logfile]
set -euo pipefail
cd "$(dirname "$0")/../.."

INTERVAL="${1:-3}"
LOG="${2:-docker/diagnostics/resources_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"

echo "[monitor_resources] logging every ${INTERVAL}s to $LOG (Ctrl-C to stop)"

trap 'echo "[monitor_resources] stopped"; exit 0' INT TERM

while true; do
    {
        echo "=== $(date -Iseconds) ==="
        free -m | awk 'NR==1{print "mem:", $0; next} /^Mem:/{print "mem:", $0} /^Swap:/{print "swap:", $0}'
        # Global memory pressure (PSI) — rising "full avg10" means processes are
        # stalled waiting on memory RIGHT NOW, the earliest signal before a freeze.
        [ -r /proc/pressure/memory ] && awk '{print "psi_mem:", $0}' /proc/pressure/memory
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
            --format=csv,noheader 2>/dev/null | awk '{print "gpu:", $0}'
        docker stats --no-stream --format \
            'container: {{.Name}}\tmem: {{.MemUsage}}\tmem%: {{.MemPerc}}\tcpu%: {{.CPUPerc}}' \
            2>/dev/null
        echo "top_rss:"
        ps -e --no-headers -o rss,pid,comm --sort=-rss | head -8 \
            | awk '{printf "  %8.1fMB  pid=%-8s %s\n", $1/1024, $2, $3}'
    } >> "$LOG"
    sync "$LOG" 2>/dev/null || true
    sleep "$INTERVAL"
done
