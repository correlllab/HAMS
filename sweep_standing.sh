#!/bin/bash
# ==========================================================================
# STANDING (ALMI) tier battery screw-grasp sweep. The robot balances on the ALMI
# RL policy the ENTIRE grasp (band OFF, pin released) — verified genuinely standing.
# Two methods (the ones that work on the screw):
#   centroid     = almi_grasp open-loop  (ALMI_SERVO_ITER=0)
#   centroid_vs  = almi_grasp servo      (ALMI_SERVO_ITER=4, re-reads GT each iter)
# almi_grasp.py is self-contained per trial: pin -> reset -> warm ALMI 30 sim-s ->
# release -> confirm solid (retry) -> reach -> slow 6-step descent -> servo -> close
# -> weld -> lift -> regrade standing_after. Settings are the 2/2-validated recipe.
# N trials each (default 30); telemetry (pelvis sway) + JSON per trial, host-
# persistent, resumable.
# ==========================================================================
set -u
ROOT=/home/guest/HAMS-grasp-2method
N=${SWEEP_N:-30}
SCREW=${SWEEP_SCREW:-screw_27}
OUTC=/home/code/core_ws/benchmark_results/sweep_battery_standing
LOG=$ROOT/sweep_standing.log
PW=Unitreeh12
# fail-fast: after 3 consecutive topple/crash trials (not legit misses) the stack is broken
# (bad heartbeat/TF) -> auto-restart up to MAXRESTARTS, then HALT rather than waste hours.
MAXRESTARTS=${MAXRESTARTS:-2}; RESTARTS=0
# validated ALMI recipe (spawn-lower drop-fix is in the sim env; these are the run knobs)
# EXACT 2/2-validated config (no shortcuts). The aggressive faster config left the arm
# in a pose where the pad-TF read failed, the servo loop bailed, and it closed 115mm
# off on air. These are the settings that gave the two clean +62/+60mm standing grasps
# (post_grasp_err ~2.5mm). ~6-7 min/trial -> 60 = ~6.5h, fits the 11h window.
AENV="ALMI_HOVER_FROZEN=0 ALMI_ENGAGE_RETRY=10 STAB_SIM=30 ALMI_RELEASE_SETTLE=6 ALMI_PELVIS_MIN=0.92 ALMI_REACH_SETTLE=6 ALMI_DESCEND_STEPS=6 ALMI_MOVE_SEC=8 ALMI_DESCEND_SETTLE=3 ALMI_SERVO_ITER=4 ALMI_CONV_MM=6 BATT_WELD_MAX_ERR_MM=12 BATT_GRIP_FORCE_N=6 BATT_TOPDOWN_PITCH=80 BATT_HOVER_M=0.10 BATT_FINGER_SINK=-0.002 BATT_HOLD_SEC=1.5"

SUDO(){ echo "$PW" | sudo -S "$@" 2>/dev/null; }
RX(){ SUDO docker exec hams_ros bash -lc "$1"; }
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

say "=== STANDING (ALMI) SWEEP START: N=$N screw=$SCREW methods=centroid,centroid_vs ==="
SUDO docker cp "$ROOT/almi_grasp.py"     hams_ros:/tmp/almi_grasp.py     >/dev/null 2>&1
SUDO docker cp "$ROOT/battery_bench.py"  hams_ros:/tmp/battery_bench.py  >/dev/null 2>&1
SUDO docker cp "$ROOT/trial_recorder.py" hams_ros:/tmp/trial_recorder.py >/dev/null 2>&1

# make sure the ALMI lowerbody controller is up before we start
if [ "$(RX 'pgrep -c -f lib/h12_lowerbody_rl/lowerbody_controller_node')" -lt 1 ]; then
  say "ALMI controller not running -> starting it"
  # /safety/heartbeat remapped away: a stray/stale False publisher latches the legs silent
  # and the robot collapses on release; in this sim nothing legitimately publishes it, so
  # ALMI is meant to run heartbeat-free. (Safety relay still clips/estops on real limits.)
  SUDO docker exec -d hams_ros bash -lc 'source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; ros2 run h12_lowerbody_rl lowerbody_controller_node --ros-args -p use_sim_time:=true -p active_policy:=almi -p engage_wait_for_confirm:=false -p disable_elastic_band:=false -r /safety/heartbeat:=/safety/heartbeat_ignored > /tmp/lowerbody.log 2>&1'
  sleep 12
fi

run_method(){   # $1=label  $2=ALMI_SERVO_ITER
  local M=$1 SI=$2 d="$OUTC/$1"
  RX "mkdir -p $d"
  local T consec=0
  for T in $(seq 1 "$N"); do
    local TT=$(printf '%02d' "$T")
    RX "test -f $d/trial_$TT.json" && { say "$M trial $TT exists -> skip"; continue; }
    SUDO docker exec -d hams_ros bash -lc "source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; python3 /tmp/trial_recorder.py > /tmp/tel_$M.csv 2>/dev/null"
    sleep 0.6
    SUDO timeout 600 docker exec hams_ros bash -lc "source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; cd /tmp && $AENV ALMI_SERVO_ITER=$SI OUT=$d/trial_$TT.json python3 almi_grasp.py $SCREW > $d/trial_$TT.log 2>&1"
    RX "pkill -TERM -f trial_recorder.py" >/dev/null 2>&1; sleep 0.4
    RX "cp -f /tmp/tel_$M.csv $d/trial_${TT}_telemetry.csv" >/dev/null 2>&1
    # TOPPLE = crash/no-record, gross miss (>100mm), or robot not standing after — i.e. the
    # stack broke. A legit open-loop MISS still writes a valid JSON (good=False) and does NOT
    # count, so honest centroid misses never trip the fail-fast.
    local r=$(RX "cat $d/trial_$TT.json 2>/dev/null" | python3 -c "import sys,json
try:
 d=json.load(sys.stdin)
 pe=d.get('post_grasp_err_mm'); sa=d.get('standing_after')
 hf=(pe is None or (isinstance(pe,(int,float)) and pe>100) or sa is False)
 print('success=%s good=%s stand_after=%s post_err=%s dz=%s%s'%(d.get('success'),d.get('good_grasp'),sa,pe,d.get('screw_lift_dz'),' TOPPLE' if hf else ''))
except Exception as e: print('NO-JSON/'+str(e)+' TOPPLE')" 2>/dev/null)
    say "$M trial $TT: $r"
    case "$r" in *TOPPLE*) consec=$((consec+1));; *) consec=0;; esac
    if [ "$consec" -ge 3 ]; then
      say "!! $M: 3 consecutive topple/crash trials -> stack broke (bad heartbeat/TF, robot not standing)"
      if [ "$RESTARTS" -lt "$MAXRESTARTS" ]; then
        RESTARTS=$((RESTARTS+1)); say "-> recovery restart #$RESTARTS/$MAXRESTARTS via restart_standing.sh"
        bash "$ROOT/restart_standing.sh" >/dev/null 2>&1
        SUDO docker cp "$ROOT/almi_grasp.py"     hams_ros:/tmp/almi_grasp.py     >/dev/null 2>&1
        SUDO docker cp "$ROOT/battery_bench.py"  hams_ros:/tmp/battery_bench.py  >/dev/null 2>&1
        SUDO docker cp "$ROOT/trial_recorder.py" hams_ros:/tmp/trial_recorder.py >/dev/null 2>&1
        consec=0
      else
        say "!! HALT: still toppling after $RESTARTS restarts. Stopping so we don't waste hours — fix ALMI/heartbeat then resume (sweep is resumable)."
        exit 1
      fi
    fi
  done
  local tot=$(RX "ls $d/trial_*.json 2>/dev/null | wc -l")
  local succ=$(RX "cat $d/trial_*.json 2>/dev/null" | python3 -c "import sys,json;n=0
for l in sys.stdin:
 l=l.strip()
 if l.startswith('{'):
  try: n+= 1 if json.loads(l).get('success') else 0
  except: pass
print(n)" 2>/dev/null)
  say "=== $M DONE: $succ/$tot success ==="
}

run_method centroid     0
run_method centroid_vs  4
say "=== STANDING SWEEP COMPLETE ==="
