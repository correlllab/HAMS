#!/bin/bash
# ==========================================================================
# NAV-RANDOM tier battery screw-grasp sweep (for the SECOND computer, run in
# parallel with the standing tier). Identical ALMI standing grasp to
# sweep_standing.sh, EXCEPT each trial teleports the base to a RANDOM standing
# position (Gaussian around the nominal spawn) via /hams/place_base, so the robot
# reaches the FIXED screw from a different relative pose each time. Same validated
# grasp config, same 10 Hz pelvis-sway telemetry, resumable, host-persistent.
#
# Structure: 6 RANDOM POSITIONS x 5 repeats = 30 trials/method. The 6 positions are
# sampled ONCE (Gaussian, sigma_fwd=0.1143, sigma_lat=0.0683, clipped +/-3 sigma) and
# are IDENTICAL for both centroid + centroid_vs (fair comparison at the same 6 spots).
# Trials 01-05 = position 1, 06-10 = position 2, ... 26-30 = position 6.
# Per-trial spawn + position index is logged to trial_NN_spawn.json.
#
# PREREQ: the sim must be up in the SAME config as the standing tier — frozen
# BatteryWorkcell (HAMS_FREEZE_BODY=1, HAMS_STANCE=almi, HAMS_SPAWN_LOWER=0.07),
# bringup running, ALMI lowerbody controller engaged. See RANDOM_TIER_README.md.
# VALIDATE 1-2 trials visually before the full run.
# ==========================================================================
set -u
ROOT=/home/guest/HAMS-grasp-2method
N=${SWEEP_N:-30}
SCREW=${SWEEP_SCREW:-screw_27}
SIG_FWD=${RAND_SIG_FWD:-0.1143}
SIG_LAT=${RAND_SIG_LAT:-0.0683}
OUTC=/home/code/core_ws/benchmark_results/sweep_battery_random
LOG=$ROOT/sweep_random.log
POSFILE=$ROOT/RANDOM_POSITIONS.json
PW=Unitreeh12
# EXACT validated grasp config (same as the standing tier that gave clean grasps)
AENV="ALMI_HOVER_FROZEN=0 ALMI_ENGAGE_RETRY=10 STAB_SIM=30 ALMI_RELEASE_SETTLE=6 ALMI_PELVIS_MIN=0.92 ALMI_REACH_SETTLE=6 ALMI_DESCEND_STEPS=6 ALMI_MOVE_SEC=8 ALMI_DESCEND_SETTLE=3 ALMI_SERVO_ITER=4 ALMI_CONV_MM=6 BATT_WELD_MAX_ERR_MM=12 BATT_GRIP_FORCE_N=6 BATT_TOPDOWN_PITCH=80 BATT_HOVER_M=0.10 BATT_FINGER_SINK=-0.002 BATT_HOLD_SEC=1.5"

SUDO(){ echo "$PW" | sudo -S "$@" 2>/dev/null; }
RX(){ SUDO docker exec hams_ros bash -lc "$1"; }
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

say "=== NAV-RANDOM SWEEP START: N=$N screw=$SCREW sig=($SIG_FWD,$SIG_LAT) ==="
SUDO docker cp "$ROOT/almi_grasp.py"     hams_ros:/tmp/almi_grasp.py     >/dev/null 2>&1
SUDO docker cp "$ROOT/battery_bench.py"  hams_ros:/tmp/battery_bench.py  >/dev/null 2>&1
SUDO docker cp "$ROOT/trial_recorder.py" hams_ros:/tmp/trial_recorder.py >/dev/null 2>&1
if [ "$(RX 'pgrep -c -f lib/h12_lowerbody_rl/lowerbody_controller_node')" -lt 1 ]; then
  say "ALMI controller not running -> starting it"
  SUDO docker exec -d hams_ros bash -lc 'source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; ros2 run h12_lowerbody_rl lowerbody_controller_node --ros-args -p use_sim_time:=true -p active_policy:=almi -p engage_wait_for_confirm:=false -p disable_elastic_band:=false > /tmp/lowerbody.log 2>&1'
  sleep 12
fi

run_method(){   # $1=label  $2=ALMI_SERVO_ITER
  local M=$1 SI=$2 d="$OUTC/$1"
  RX "mkdir -p $d"
  local T
  for T in $(seq 1 "$N"); do
    local TT=$(printf '%02d' "$T")
    RX "test -f $d/trial_$TT.json" && { say "$M trial $TT exists -> skip"; continue; }
    # 6 FIXED positions (picked once, up front, same for both methods) x 5 repeats:
    # trial T -> position (T-1)/5, so 01-05=pos0, 06-10=pos1, ... 26-30=pos5.
    local P=$(( (T-1)/5 ))
    local PB=$(python3 -c "import json;pos=json.load(open('$POSFILE'));print('%.4f,%.4f'%tuple(pos[$P]))")
    RX "echo '{\"position_index\": $P, \"place_base_fwd_lat\": \"$PB\"}' > $d/trial_${TT}_spawn.json"
    SUDO docker exec -d hams_ros bash -lc "source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; python3 /tmp/trial_recorder.py > /tmp/tel_$M.csv 2>/dev/null"
    sleep 0.6
    SUDO timeout 600 docker exec hams_ros bash -lc "source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; cd /tmp && HAMS_PLACE_BASE=$PB $AENV ALMI_SERVO_ITER=$SI OUT=$d/trial_$TT.json python3 almi_grasp.py $SCREW > $d/trial_$TT.log 2>&1"
    RX "pkill -TERM -f trial_recorder.py" >/dev/null 2>&1; sleep 0.4
    RX "cp -f /tmp/tel_$M.csv $d/trial_${TT}_telemetry.csv" >/dev/null 2>&1
    local r=$(RX "cat $d/trial_$TT.json 2>/dev/null" | python3 -c "import sys,json
try:
 d=json.load(sys.stdin);print('success=%s good=%s dz=%s'%(d.get('success'),d.get('good_grasp'),d.get('screw_lift_dz')))
except: print('NO-JSON')" 2>/dev/null)
    say "$M trial $TT (spawn $PB): $r"
  done
  local tot=$(RX "ls $d/trial_*.json 2>/dev/null | wc -l")
  say "=== $M DONE: $tot trials ==="
}

# Pick the 6 random positions ONCE, up front — fixed for the whole run + BOTH methods.
if [ ! -f "$POSFILE" ]; then
  python3 -c "import random,json
r=random.Random('rand6-v2')
pos=[[round(max(-3,min(3,r.gauss(0,1)))*$SIG_FWD,4), round(max(-3,min(3,r.gauss(0,1)))*$SIG_LAT,4)] for _ in range(6)]
json.dump(pos, open('$POSFILE','w'))"
fi
say "6 fixed positions (picked once, fwd,lat metres): $(cat $POSFILE)"

run_method centroid     0
run_method centroid_vs  4
say "=== NAV-RANDOM SWEEP COMPLETE ==="
