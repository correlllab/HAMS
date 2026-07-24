#!/bin/bash
# ==========================================================================
# PHASE 2 of the tethered battery sweep: the 3 grasp-SYNTHESIS methods that run
# through the fridge harness (grasp_benchmark), pointed at the standing screw:
#   pca   = --method topdown_antipodal   (top-layer PCA antipodal)
#   ggx   = --method graspgenx           (GraspGenX)
#   skill = --method skill               (deployed /skill/grasp: ggx + re-rank + servo)
# grasp_benchmark has NO weld and resets only the arm, so: (a) we reset the SCENE +
# re-engage the band ourselves each trial, and (b) we grade on CONTACT (gripper closes
# on the screw, centred, on_handle) — the same criterion the fridge study used and the
# common cross-method metric vs battery_bench's good_grasp. Success-mode lift is
# unreliable here (no weld -> a lifted screw slips), so contact is the honest grade.
#
# Waits for phase 1 to finish (shared robot), then SMOKES each method once; if the
# smoke CRASHES (no JSON / error), the method is skipped and logged (not silently
# zero-filled). A method that merely misses still runs its 30 — a low contact rate
# is real data. N trials each (default 30).
# ==========================================================================
set -u
ROOT=/home/guest/HAMS-grasp-2method
N=${SWEEP_N:-30}
SCREW=${SWEEP_SCREW:-screw_27}
OUTC=/home/code/core_ws/benchmark_results/sweep_battery_tethered
LOG=$ROOT/phase2_tethered.log
PW=Unitreeh12
SUDO(){ echo "$PW" | sudo -S "$@" 2>/dev/null; }
RX(){ SUDO docker exec hams_ros bash -lc "$1"; }
RXS(){ SUDO docker exec hams_ros bash -lc "source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; $1"; }
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

say "=== PHASE 2 waiting for phase 1 (sweep_tethered.sh) to finish ==="
while pgrep -f 'sweep_tethered.sh' >/dev/null 2>&1; do sleep 60; done
say "=== phase 1 done -> PHASE 2 START: N=$N screw=$SCREW methods=pca,ggx,skill ==="
SUDO docker cp "$ROOT/trial_recorder.py" hams_ros:/tmp/trial_recorder.py >/dev/null 2>&1

# tethered scene reset + band re-engage before a grasp_benchmark trial
reset_tethered(){
  RXS "ros2 topic pub --once /hams/grasp_release std_msgs/msg/Empty '{}' >/dev/null 2>&1
       for i in \$(seq 1 8); do ros2 topic pub --once /hams/reset_scene std_msgs/msg/Empty '{}' >/dev/null 2>&1; sleep 0.15; done
       sleep 2
       for i in \$(seq 1 6); do ros2 topic pub --once /hams/freeze_body std_msgs/msg/Bool '{data: false}' >/dev/null 2>&1; sleep 0.1; done
       sleep 2.5"
}

# build the grasp_benchmark command for a method (contact grading, GT-cropped cloud)
gb_cmd(){   # $1=grasp_benchmark method  $2=out path
  local M=$1 OUT=$2
  local base="HAMS_GRASP_GT_NAME=$SCREW HAMS_GRASP_BOX_SOURCE=gt ros2 run h12_skills grasp_benchmark --method $M --object 'screw' --gt-name $SCREW --box-source gt --success-mode contact --success-dz 0.03 --arm right --out $OUT"
  if [ "$M" = "skill" ]; then echo "$base"; else echo "$base --max-attempts 20 --no-plan"; fi
}

run_trial(){  # $1=label $2=gb-method $3=trial-num
  local L=$1 M=$2 T=$(printf '%02d' "$3") d="$OUTC/$1"
  RX "mkdir -p $d"
  RX "test -f $d/trial_$T.json" && { say "$L trial $T exists -> skip"; return; }
  reset_tethered
  SUDO docker exec -d hams_ros bash -lc "source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; python3 /tmp/trial_recorder.py > /tmp/tel_$L.csv 2>/dev/null"
  sleep 0.6
  SUDO timeout 420 docker exec hams_ros bash -lc "source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; cd /tmp && $(gb_cmd "$M" "$d/trial_$T.json") > $d/trial_$T.log 2>&1"
  RX "pkill -TERM -f trial_recorder.py" >/dev/null 2>&1; sleep 0.4
  RX "cp -f /tmp/tel_$L.csv $d/trial_${T}_telemetry.csv" >/dev/null 2>&1
  local r=$(RX "cat $d/trial_$T.json 2>/dev/null" | python3 -c "import sys,json
try:
 d=json.load(sys.stdin);print('success=%s holding=%s contact=%s tip_mm=%s dz=%s'%(d.get('success'),d.get('holding'),d.get('grip_contact_flag'),d.get('tip_to_handle_mm'),d.get('lift_dz_m')))
except Exception as e: print('NO-JSON/'+str(e))" 2>/dev/null)
  say "$L trial $T: $r"
}

run_method(){  # $1=label $2=gb-method
  local L=$1 M=$2 d="$OUTC/$1"
  # SMOKE first — skip the method only if it CRASHES (no json / error)
  say "$L: smoke test"
  reset_tethered
  SUDO timeout 420 docker exec hams_ros bash -lc "source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; cd /tmp && $(gb_cmd "$M" /tmp/p2_smoke.json) > /tmp/p2_smoke.log 2>&1"
  local ok=$(RX "cat /tmp/p2_smoke.json 2>/dev/null" | python3 -c "import sys,json
try:
 d=json.load(sys.stdin); print('OK' if (d.get('executed') is not None and not d.get('error')) else 'BAD')
except: print('BAD')" 2>/dev/null)
  if [ "$ok" != "OK" ]; then
    say "$L: SMOKE FAILED (crash/no-json) -> SKIP method. tail:"; RX "tail -5 /tmp/p2_smoke.log 2>/dev/null" | while read -r l; do say "   $l"; done
    return
  fi
  say "$L: smoke OK -> running $N trials"
  local T; for T in $(seq 1 "$N"); do run_trial "$L" "$M" "$T"; done
  local tot=$(RX "ls $d/trial_*.json 2>/dev/null | wc -l")
  say "=== $L DONE: $tot trials ==="
}

run_method pca   topdown_antipodal
run_method ggx   graspgenx
run_method skill skill
say "=== PHASE 2 COMPLETE ==="
