#!/bin/bash
# Paper-grid driver: {skill, graspgenx, topdown_irl} x {hanging, standing,
# standing_rand}, N=30 per cell (PAPER_N overrides). Runs each cell through
# two_method_sweep_host.sh (all its guards/QA/telemetry apply per cell).
#
# The plain-STANDING skill and graspgenx cells are NOT collected here — they
# are REUSED from the campaign's sweep_almi (identical ALMI protocol, N=30),
# per Adam. So this driver collects the 7 remaining cells, in this order:
#   1. hanging:        skill graspgenx topdown_irl   (band mode, ~3.5 h)
#   2. standing:       topdown_irl                   (ALMI; other two reused, ~2.5 h)
#   3. standing_rand:  skill graspgenx topdown_irl   (ALMI, randomized spawn,
#                      fresh env per trial, ~17 h) — LAST on purpose: the
#                      sampling sigmas (UNFROZEN_RAND_SIG_LONG/LAT, default
#                      11.43/6.83 cm long/lat per the plotted Fig. 3 scatter)
#                      can still be corrected once the raw endpoints arrive.
# Everything is resumable: re-running skips existing trial_NN.json per cell.
#
# Output tree (host: Downloads/HAMS-test-grasping/core_ws/benchmark_results/):
#   sweep_paper/{hanging,standing,standing_rand}/<method>/trial_NN.*
# Aggregate per tier with aggregate_2method.py; point the standing tier's
# skill/graspgenx at sweep_almi/.
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N="${PAPER_N:-30}"
OUTBASE=/home/code/core_ws/benchmark_results/sweep_paper

exec 8>/tmp/hams_paper_grid.lock
if ! flock -n 8; then
  echo "[grid] REFUSING to start — another paper_grid_host.sh holds the lock."
  exit 1
fi

run_cell() {  # $1=tier-name $2=policy $3=rand $4=methods
  # Re-entrancy fast-path: if EVERY method in this cell already has N trial
  # JSONs (host view), skip the cell outright — a resumed grid should not pay
  # 3 env rebuilds just to discover there is nothing to do.
  local HOST_TIER="/home/guest/Downloads/HAMS-test-grasping/core_ws/benchmark_results/sweep_paper/$1"
  local done_all=1 m
  for m in $4; do
    [ "$(ls "$HOST_TIER/$m"/trial_*.json 2>/dev/null | wc -l)" -lt "$N" ] && { done_all=0; break; }
  done
  if [ "$done_all" = 1 ]; then
    echo "[grid] cell '$1' already complete (N=$N per method) — skipping"
    return 0
  fi
  echo "[grid] === cell tier=$1 methods='$4' N=$N $(date +%H:%M:%S) ==="
  UNFROZEN_N=$N UNFROZEN_POLICY="$2" UNFROZEN_RAND="$3" \
  UNFROZEN_METHODS="$4" UNFROZEN_OUT="$OUTBASE/$1" \
    bash "$HERE/two_method_sweep_host.sh"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[grid] cell '$1' FAILED (rc=$rc) — stopping the grid (fix, then re-run; done trials are skipped)"
    exit $rc
  fi
}

run_cell hanging       ""     0 "skill graspgenx topdown_irl"
run_cell standing      "almi" 0 "topdown_irl"
# Randomized tier: mechanism A (fresh env per trial on the LIVE sim). The warm
# path (B) needs /hams/place_base, which lives only in this worktree's sim —
# but the worktree's core_ws is NOT colcon-built, so its sim container crashes
# at import (magpie_msgs). Rather than build the worktree workspace mid-run,
# run the validated fresh-env randomization on the live sim: guaranteed-good,
# same sim every other cell uses, position sed into HAMS_SPAWN_FORWARD/LATERAL.
# Cost: ~12 min/trial. skill+graspgenx are the paper-critical 2-method pair,
# so they run FIRST; topdown_irl (0/30 on the fridge everywhere) runs last.
run_cell standing_rand "almi" 1 "skill graspgenx topdown_irl"
echo "[grid] ALL CELLS COMPLETE $(date +%H:%M:%S)"
