#!/usr/bin/env bash
# Hard safety net for HAMS container runs on this 30 GB / 2 GB-swap box.
#
# Two hard freezes (2026-07-13) came from the SAME shape: total ACTUAL usage
# (containers + desktop) outran RAM, MemAvailable collapsed, the 2 GB swap
# exhausted, and the kernel thrashed before its OOM killer could get ahead of it.
# Per-container mem_limits don't prevent that (no container exceeded its cap).
#
# This watches the only two signals that actually predict the freeze —
# MemAvailable and memory PSI ("full avg10" = % of time processes are STALLED
# waiting on memory) — and force-removes the stack the moment either crosses a
# danger line, while the box is still responsive enough to do so.
#
# Usage: docker/scripts/mem_watchdog.sh [avail_mb_floor] [psi_full_ceiling]
set -uo pipefail

FLOOR_MB="${1:-1800}"     # kill if MemAvailable drops below this
PSI_MAX="${2:-20}"        # kill if PSI full avg10 exceeds this (freeze hit 36)
CONTAINERS="hams_ros hams_sim_robocasa"

echo "[mem_watchdog] armed: kill stack if MemAvailable < ${FLOOR_MB}MB or PSI full avg10 > ${PSI_MAX}"

while true; do
    avail_mb=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    psi=$(awk -F'avg10=' '/^full/{split($2,a," "); print a[1]}' /proc/pressure/memory 2>/dev/null)
    psi=${psi:-0}

    trip=""
    (( avail_mb < FLOOR_MB )) && trip="MemAvailable=${avail_mb}MB < ${FLOOR_MB}MB"
    awk -v p="$psi" -v m="$PSI_MAX" 'BEGIN{exit !(p>m)}' && trip="${trip:+$trip; }PSI full avg10=${psi} > ${PSI_MAX}"

    if [ -n "$trip" ]; then
        echo "[mem_watchdog] *** TRIPPED: $trip — killing stack NOW ***"
        docker rm -f $CONTAINERS >/dev/null 2>&1 || true
        echo "[mem_watchdog] stack removed; host protected."
        exit 1
    fi
    sleep 2
done
