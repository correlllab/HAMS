#!/bin/bash
# ==========================================================================
# Tethered (elastic-band, swaying-base) battery screw-grasp sweep.
# Methods run here (PHASE 1 = validated battery-native harness):
#   centroid     = battery_bench --method noservo  (open-loop GT: base sways after
#                  the pose is committed -> stale target -> the misses live here)
#   centroid_vs  = battery_bench --method servo    (closed-loop GT re-read each iter
#                  -> tracks the sway -> the "visual servoing" arm of the comparison)
# N trials each (default 30). Per trial: telemetry CSV (pelvis sway + grip traj) +
# result JSON, saved host-persistent under core_ws/benchmark_results (bind-mounted),
# RESUMABLE (skips a trial whose JSON already exists). Validated recipe: tethered
# reset+band re-engage, 6 N grip, 80deg top-down, 12 mm sway-budget weld gate.
# ==========================================================================
set -u
ROOT=/home/guest/HAMS-grasp-2method
N=${SWEEP_N:-30}
SCREW=${SWEEP_SCREW:-screw_27}
OUTC=/home/code/core_ws/benchmark_results/sweep_battery_tethered
LOG=$ROOT/sweep_tethered.log
PW=Unitreeh12
GENV="BATT_RELEASE_AFTER_RESET=1 BATT_WELD_MAX_ERR_MM=12 BATT_GRIP_FORCE_N=6 BATT_TOPDOWN_PITCH=80 BATT_HOVER_M=0.10 BATT_FINGER_SINK=-0.002"

SUDO(){ echo "$PW" | sudo -S "$@" 2>/dev/null; }
RX(){ SUDO docker exec hams_ros bash -lc "$1"; }                       # raw exec
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

say "=== TETHERED SWEEP START: N=$N screw=$SCREW methods=centroid,centroid_vs ==="
# stage helpers into the container
SUDO docker cp "$ROOT/battery_bench.py" hams_ros:/tmp/battery_bench.py >/dev/null 2>&1
SUDO docker cp "$ROOT/trial_recorder.py" hams_ros:/tmp/trial_recorder.py >/dev/null 2>&1

run_method(){   # $1=label  $2=battery_bench --method value
  local M=$1 MODE=$2 d="$OUTC/$1"
  RX "mkdir -p $d"
  local T
  for T in $(seq 1 "$N"); do
    local TT=$(printf '%02d' "$T")
    if RX "test -f $d/trial_$TT.json" ; then say "$M trial $TT exists -> skip"; continue; fi
    # start telemetry recorder (detached), give it a beat to open the CSV
    SUDO docker exec -d hams_ros bash -lc "source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; python3 /tmp/trial_recorder.py > /tmp/tel_$M.csv 2>/dev/null"
    sleep 0.6
    # run the grasp (self-resets scene + re-engages band each trial)
    SUDO timeout 340 docker exec hams_ros bash -lc "source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; cd /tmp && $GENV python3 battery_bench.py --method $MODE --arm right --screw $SCREW --out $d/trial_$TT.json > $d/trial_$TT.log 2>&1"
    # stop telemetry, save it
    RX "pkill -TERM -f trial_recorder.py" >/dev/null 2>&1
    sleep 0.4
    RX "cp -f /tmp/tel_$M.csv $d/trial_${TT}_telemetry.csv" >/dev/null 2>&1
    local r=$(RX "cat $d/trial_$TT.json 2>/dev/null" | python3 -c "import sys,json
try:
 d=json.load(sys.stdin);print('success=%s good=%s closed=%s post_err=%s dz=%s'%(d.get('success'),d.get('good_grasp'),d.get('closed_on_object'),d.get('post_grasp_err_mm'),d.get('screw_lift_dz')))
except: print('NO-JSON')" 2>/dev/null)
    say "$M trial $TT: $r"
  done
  # quick tally
  local tot=$(RX "ls $d/trial_*.json 2>/dev/null | wc -l")
  local succ=$(RX "cat $d/trial_*.json 2>/dev/null" | python3 -c "import sys,json;n=0
for l in sys.stdin:
 l=l.strip()
 if l.startswith('{'):
  try:
   n+= 1 if json.loads(l).get('success') else 0
  except: pass
print(n)" 2>/dev/null)
  say "=== $M DONE: $succ/$tot success ==="
}

run_method centroid     noservo
run_method centroid_vs  servo
say "=== PHASE 1 (battery-native methods) COMPLETE ==="
