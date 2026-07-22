#!/bin/bash
# Grasp sweep for the paper graph: new(skill) vs graspgenx-normal vs the real
# robot's top-down.
#   skill       -> the deployed /skill/grasp path (heuristic matching selection:
#                  envelope -> Y-up re-roll -> tiers -> diversity -> IK fallback)
#   graspgenx   -> the same GraspGenX server/candidates executed strictly
#                  best-first by raw model score (no heuristic selection)
#   topdown_irl -> HAMS main's DEPLOYED goal.top_down grasp, verbatim (one
#                  synthetic 80-deg-below-horizontal candidate on the cloud
#                  mean; see grasp_benchmark_topdown.py) — runs via this
#                  branch's wrapper, NO rebuild of the live checkout.
# skill/graspgenx hit the identical server + sampling config, so that delta
# isolates the heuristic-matching layer; topdown_irl is the as-deployed IRL
# baseline. pca(topdown_antipodal)/centroid deliberately omitted for this graph
# (ablation data exists in sweep_almi_ablation / sweep_unfrozen).
#
# This is a faithful snapshot of the validated unfrozen_sweep_host.sh protocol
# (ALMI standing tier by default: strict grading, telemetry CSV + rosbag, QA,
# wander watchdog, warm retether), restricted to the two methods. It runs FROM
# this branch worktree but drives the LIVE checkout's docker setup — the live
# repo needs no branch switch and no rebuild (no package code differs).
#
# Run it ONLY after the current campaign's sweep has finished (the guard below
# refuses to start while another orchestrator or a grasp_benchmark is alive).
#   bash two_method_sweep_host.sh            # 20 trials x {skill, graspgenx}
# Knobs (same env-var names as the unfrozen script):
#   UNFROZEN_N=30 UNFROZEN_OUT=/home/code/core_ws/benchmark_results/sweep_almi \
#     bash two_method_sweep_host.sh          # e.g. top-up sweep_almi to N=30
#   UNFROZEN_POLICY=""  -> hanging tier instead of the ALMI standing tier
#   UNFROZEN_RAND=1     -> randomized-START standing tier: fresh env per trial
#                          at a spawn sampled around the tuned position (see
#                          the RAND block below); ~11-12 min/trial
set -o pipefail

HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this branch worktree
LIVE_REPO="${HAMS_LIVE_REPO:-/home/guest/Downloads/HAMS-test-grasping}"
cd "$LIVE_REPO" || { echo "[2method] live repo not found at $LIVE_REPO"; exit 1; }

# --- start guard: never race the running campaign -----------------------------
# Single-instance lock (flock: race-free, auto-released when the script exits).
exec 9>/tmp/hams_two_method_sweep.lock
if ! flock -n 9; then
  echo "[2method] REFUSING to start — another two_method_sweep_host.sh holds the lock."
  exit 1
fi
# Other orchestrators / an in-flight trial: refuse rather than wreck their env
# (fresh_env force-removes the containers they are driving).
if [ "${1:-}" != "--force" ]; then
  OTHERS=$(pgrep -af 'unfrozen_sweep_host|skill_sweep_host|baselines_host|skill_sweep\.sh|run_sweep\.sh|grasp_benchmar[k]' 2>/dev/null)
  if [ -n "$OTHERS" ]; then
    echo "[2method] REFUSING to start — an experiment appears to be running:"
    echo "$OTHERS" | sed 's/^/    /'
    echo "[2method] wait for it to finish, then re-run (or use --force)."
    exit 1
  fi
fi

OUTROOT="${UNFROZEN_OUT:-/home/code/core_ws/benchmark_results/sweep_2method}"
PW="${HAMS_SUDO_PW:-Unitreeh12}"
N="${UNFROZEN_N:-20}"
METHODS="${UNFROZEN_METHODS:-skill graspgenx topdown_irl}"
POLICY="${UNFROZEN_POLICY-almi}"

# UNFROZEN_RAND=1 (standing tier only): randomized-START condition. EVERY trial
# gets its own FRESH env spawned at a position sampled around the tuned spawn —
# /hams/reset_scene restores a fixed snapshot, so warm resets cannot move the
# base; a rebuild per trial is the only spawn-randomizing mechanism the sim
# offers without code changes (which the live checkout forbids). Sampling
# (spec reviewed against the paper 2026-07-22): the paper (Sec IV-A / Fig 3)
# states NO initialization spec — its "noisy initialization" is the
# localization prior, not a spawn perturbation. These sigmas are the MEASURED
# endpoint scatter of its 6 m walk-to-goal on this platform (n=19), used here
# as a post-navigation stance-uncertainty prior. Axis assignment follows the
# PLOTTED data, not the figure annotation (which is very likely swapped: the
# plotted longitudinal spread ~50 cm ~ sigma 11.4, lateral ~21 cm ~ sigma 6.8):
# sigma_long 11.43 cm, sigma_lat 6.83 cm. Clip +/-3 sigma (2-sigma truncation
# would shrink realized sigma ~12%). POSITION ONLY — no spawn-yaw knob exists;
# flag as a limitation in the paper. Gaussian, independent axes; if Adam gets
# the raw 19 endpoints from the authors, switch to the full sample covariance.
# Deterministic per (method, trial) so the sweep is replicable; each trial's
# sampled spawn is logged next to its JSON as trial_NN_spawn.json.
RAND="${UNFROZEN_RAND:-0}"
RAND_SIG_LONG="${UNFROZEN_RAND_SIG_LONG:-0.1143}"   # sigma along the approach [m]
RAND_SIG_LAT="${UNFROZEN_RAND_SIG_LAT:-0.0683}"     # sigma lateral [m]
if [ "$RAND" = "1" ] && [ -z "$POLICY" ]; then
  echo "[2method] UNFROZEN_RAND=1 requires the standing tier (UNFROZEN_POLICY=almi)"; exit 1
fi
BASE_FWD=0.18
BASE_LAT=$(grep '^HAMS_SPAWN_LATERAL=' docker/.env 2>/dev/null | head -1 | cut -d= -f2)
BASE_LAT="${BASE_LAT:-0.10}"
SPAWN_FWD_VAL="$BASE_FWD"; SPAWN_LAT_VAL=""
HOST_OUT="${OUTROOT/\/home\/code/$LIVE_REPO}"   # host view of the results dir

# Tier knobs (defaults keep each tier byte-identical to the campaign's runs)
if [ -n "$POLICY" ]; then
  ATTEMPT_PAUSE=3; BRIGHT_MIN=140; DEPTH_MAX=0.75; MAXATT=8; SKILL_ATT=12
  FF_LIN=0.08; FF_ANG=0.35; SRV_ITER=4; SRV_DUR=15
else
  ATTEMPT_PAUSE=0; BRIGHT_MIN=0; DEPTH_MAX=0; MAXATT=20; SKILL_ATT=40
  FF_LIN=0.05; FF_ANG=0.20; SRV_ITER=6; SRV_DUR=15
fi
GT_WORLD="3.5365,-2.889,1.235"   # fridge-handle bar centre (world frame)
SIM_STATE=/home/code/h1_robocasa/.almi_ready.npz     # sim qpos/qvel snapshot (container path)
LSTM_STATE=/home/code/core_ws/.almi_lstm.pt          # policy LSTM snapshot (container path)
SIM_STATE_HOST="$LIVE_REPO/h1_robocasa/.almi_ready.npz"
LSTM_STATE_HOST="$LIVE_REPO/core_ws/.almi_lstm.pt"
# NO_PIN=1 on the policy tier (v7-validated): a trial-time sim pin — even
# ramped — hands off mid-named_config with a PD setpoint mismatch that yanks
# the free base (2.3 m catapult). The arm is instead PRE-HOMED UNDER FREEZE in
# fresh_env/retether, so go_home's named_config starts from home = trivial.
if [ -n "$POLICY" ]; then NO_PIN=1; G_OFF=0.06; else NO_PIN=0; G_OFF=0.0; fi
if [ -n "$POLICY" ]; then W_SRV=1; else W_SRV=0; fi
TRIAL_ENV="HAMS_BENCH_WORLD_SERVO=$W_SRV HAMS_GRASP_OFFSET=$G_OFF HAMS_NO_SIM_ARM_RESET=$NO_PIN HAMS_GRASP_GT_WORLD=$GT_WORLD HAMS_ATTEMPT_PAUSE=$ATTEMPT_PAUSE HAMS_MASK_BRIGHTNESS_MIN=$BRIGHT_MIN HAMS_TARGET_MAX_DEPTH=$DEPTH_MAX HAMS_SERVO_FASTFAIL_LIN=$FF_LIN HAMS_SERVO_FASTFAIL_ANG=$FF_ANG HAMS_SERVO_MAX_ITER=$SRV_ITER HAMS_SERVO_DURATION=$SRV_DUR"

sudo_do() { echo "$PW" | sudo -S "$@" 2>/dev/null; }
in_ros()  { sudo_do docker exec hams_ros bash -lc "source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; $1"; }
in_ros_d(){ sudo_do docker exec -d hams_ros bash -lc "source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; $1"; }

fell_count() { c=$(sudo_do docker logs hams_sim_robocasa 2>&1 | grep -c "ROBOT FELL"); echo "${c:-0}"; }
FELL_BASE=0
robot_fell() { [ "$(fell_count)" -gt "$FELL_BASE" ]; }
mark_fell_baseline() { FELL_BASE=$(fell_count); }

_drift_val() {  # prints drift/yaw vs /tmp/almi_ref.txt (fallback: station); exit 1 if > ($1 m, $2 deg)
  in_ros "python3 - << 'PY'
import rclpy, json, time, math, sys
try:
    rx, ry = [float(v) for v in open('/tmp/almi_ref.txt').read().split()[:2]]
except Exception:
    rx, ry = 3.02, -2.787
rclpy.init()
from rclpy.node import Node
from std_msgs.msg import String
n=Node('driftchk'); box={'d':None}
n.create_subscription(String,'/robocasa/object_poses',lambda m: box.__setitem__('d',m.data),10)
t0=time.time()
while box['d'] is None and time.time()-t0<8: rclpy.spin_once(n,timeout_sec=0.2)
rclpy.shutdown()
if box['d'] is None: sys.exit(0)
p=json.loads(box['d']).get('__pelvis__')
if not p: sys.exit(0)
d=math.hypot(p[0]-rx,p[1]-ry)
w,x,y,z=p[3:7]
yaw=abs(math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))))
print(f'drift={d:.3f}m yaw={yaw:.1f}deg (ref {rx:.2f},{ry:.2f})')
sys.exit(1 if (d>$1 or yaw>$2) else 0)
PY"
}
robot_drifted() { _drift_val 0.12 12.0; [ $? -eq 1 ]; }
# With WORLD-ANCHORED execution a drifted-but-standing robot still aims at the
# true handle; out-of-reach self-reports as "no reachable candidate". Only kill
# trials that are truly hopeless (walked far away / spun around) — drift below
# that is DATA (recorded in telemetry), not a failure sentence.
hard_drift()    { _drift_val 0.60 60.0; [ $? -eq 1 ]; }

door_closed() {
  in_ros "python3 - << 'PY'
import rclpy, json, time, math, sys
rclpy.init()
from rclpy.node import Node
from std_msgs.msg import String
n=Node('doorchk'); box={'d':None}
n.create_subscription(String,'/robocasa/object_poses',lambda m: box.__setitem__('d',m.data),10)
t0=time.time()
while box['d'] is None and time.time()-t0<8: rclpy.spin_once(n,timeout_sec=0.2)
rclpy.shutdown()
if box['d'] is None: sys.exit(0)
d=json.loads(box['d']).get('door_obj')
if not d: sys.exit(0)
sys.exit(1 if math.dist(d[:3],[4.0018,-2.9283,1.2337])>0.01 else 0)
PY"
  [ $? -eq 0 ]
}

gripper_ok() {
  in_ros "ros2 service call /right/gripper/open std_srvs/srv/Trigger >/dev/null 2>&1; sleep 3
timeout 6 ros2 topic echo --once /right/gripper/state 2>/dev/null | python3 -c '
import sys, yaml
d = yaml.safe_load(sys.stdin.read() or \"{}\") or {}
p = float(d.get(\"position\", 0)); f = d.get(\"finger_positions\") or [0, 0]
sys.exit(0 if (p >= 95.0 and abs(float(f[0])-float(f[1])) <= 6.0) else 1)'"
}

capture_ref() {
  in_ros "python3 - << 'PY' > /tmp/almi_ref.txt
import rclpy, json, time
rclpy.init()
from rclpy.node import Node
from std_msgs.msg import String
n=Node('refcap'); box={'d':None}
n.create_subscription(String,'/robocasa/object_poses',lambda m: box.__setitem__('d',m.data),10)
t0=time.time()
while box['d'] is None and time.time()-t0<8: rclpy.spin_once(n,timeout_sec=0.2)
rclpy.shutdown()
p=json.loads(box['d'])['__pelvis__'] if box['d'] else [3.02,-2.787]
print(f'{p[0]:.4f} {p[1]:.4f}')
PY"
  echo "[env] settled ref: $(sudo_do docker exec hams_ros cat /tmp/almi_ref.txt 2>/dev/null | tail -1)"
}

engage_policy() {  # start lowerbody node, wait engaged, release the pin, verify
  sudo_do docker exec hams_ros bash -c "rm -f /tmp/lowerbody.log /tmp/almi_ref.txt" 2>/dev/null
  LSTM_ARG=""
  STAB=40
  # SPAWN-AT-READY DISABLED: restoring qpos mid-session desyncs the robosuite
  # controller stack (validated 2026-07-23: restored envs show ~150mm unreachable
  # servo errors + wanders; cold envs are healthy). Cold engage always.
  READY_SNAP=0
  if [ "$READY_SNAP" = "1" ] && [ -f "$LSTM_STATE_HOST" ] && [ -f "$SIM_STATE_HOST" ]; then
    LSTM_ARG="-p lstm_state_path:=$LSTM_STATE"
    STAB=4          # spawn-at-ready: sim + LSTM restored, warmup is a formality
    echo "[env] spawn-at-ready snapshots found — fast engage"
  fi
  in_ros_d "ros2 run h12_lowerbody_rl lowerbody_controller_node --ros-args -p use_sim_time:=true -p active_policy:=$POLICY -p engage_wait_for_confirm:=false -p disable_elastic_band:=false $LSTM_ARG > /tmp/lowerbody.log 2>&1"
  for k in $(seq 1 40); do sudo_do docker exec hams_ros bash -c "grep -q 'committed None' /tmp/lowerbody.log 2>/dev/null" && break; sleep 3; done
  # SIM-TIME settle-and-release: wait 40 SIM-sec pinned (verified: policy reaches
  # upright=1.0, holds after release), verify vertical, release, confirm it stays
  # up. Wall-clock sleeps failed because the ~5%-realtime sim gave <4 sim-sec to
  # stabilize before release -> topple. Returns nonzero if it can't hold.
  if ! in_ros "STAB_SIM=$STAB CONFIRM_SIM=10 python3 /tmp/almi_engage.py"; then return 1; fi
  # first successful COLD engage: capture the ready snapshots for all future starts
  if [ "$STAB" = "40" ]; then
    in_ros "ros2 topic pub --times 8 --rate 5 /hams/save_state std_msgs/msg/Empty '{}' >/dev/null 2>&1
ros2 topic pub --times 8 --rate 5 /hams/save_lstm std_msgs/msg/Empty '{}' >/dev/null 2>&1"
    sleep 3
    [ -f "$SIM_STATE_HOST" ] && echo "[env] ready-state snapshots captured (next starts are instant)"
  fi
  # no ref file yet -> drift is measured vs the STATION, catching settle-walks
  if robot_fell || robot_drifted; then return 1; fi
  capture_ref
  return 0
}

fresh_env() {
  echo "[env] recreating env $(date +%H:%M:%S)"
  sudo_do docker rm -f hams_ros hams_sim_robocasa >/dev/null; sleep 4
  sed -i "s/^HAMS_SPAWN_FORWARD=.*/HAMS_SPAWN_FORWARD=${SPAWN_FWD_VAL:-0.18}/" docker/.env
  # SPAWN_LAT_VAL is only ever non-empty in RAND mode — the plain tiers leave
  # the tuned .env lateral untouched, keeping their behavior byte-identical.
  [ -n "$SPAWN_LAT_VAL" ] && sed -i "s/^HAMS_SPAWN_LATERAL=.*/HAMS_SPAWN_LATERAL=$SPAWN_LAT_VAL/" docker/.env
  if [ -n "$POLICY" ]; then
    sed -i 's/^HAMS_STANCE=.*/HAMS_STANCE=almi/' docker/.env
    sed -i 's/^HAMS_FREEZE_BODY=.*/HAMS_FREEZE_BODY=1/' docker/.env
    sed -i 's/^HAMS_SIM_ODOM=.*/HAMS_SIM_ODOM=1/' docker/.env
    sed -i 's/^HAMS_ARM_RESET_RAMP=.*/HAMS_ARM_RESET_RAMP=1.5/' docker/.env
    sed -i 's/^HAMS_CAMERAS=.*/HAMS_CAMERAS=head/' docker/.env
    sed -i 's/^HAMS_LIDAR=.*/HAMS_LIDAR=0/' docker/.env
    sed -i "s|^HAMS_STATE_PATH=.*|HAMS_STATE_PATH=|" docker/.env   # restore disabled (desync)
  else
    sed -i 's/^HAMS_STANCE=.*/HAMS_STANCE=/' docker/.env
    sed -i 's/^HAMS_FREEZE_BODY=.*/HAMS_FREEZE_BODY=0/' docker/.env
    sed -i 's/^HAMS_SIM_ODOM=.*/HAMS_SIM_ODOM=0/' docker/.env
    sed -i 's/^HAMS_ARM_RESET_RAMP=.*/HAMS_ARM_RESET_RAMP=0/' docker/.env
    sed -i 's/^HAMS_CAMERAS=.*/HAMS_CAMERAS=1/' docker/.env
    sed -i 's/^HAMS_LIDAR=.*/HAMS_LIDAR=1/' docker/.env
    sed -i 's|^HAMS_STATE_PATH=.*|HAMS_STATE_PATH=|' docker/.env
  fi
  echo "$PW" | sudo -S -E bash docker/scripts/docker_run.sh robocasa --task OpenFridge --seed 42 > /tmp/sim_batch.log 2>&1 &
  for k in $(seq 1 60); do sudo_do docker logs hams_sim_robocasa 2>&1 | grep -q "ROS bridges up" && break; sleep 3; done
  echo "$PW" | sudo -S -E bash docker/scripts/docker_run.sh ros sleep infinity > /tmp/ros_batch.log 2>&1 &
  for k in $(seq 1 40); do sudo_do docker ps --format '{{.Names}}' | grep -q hams_ros && break; sleep 2; done
  sudo_do docker cp "$HELPER_DIR/grab_head.py" hams_ros:/tmp/grab_head.py
  sudo_do docker cp "$HELPER_DIR/trial_recorder.py" hams_ros:/tmp/trial_recorder.py
  sudo_do docker cp "$HELPER_DIR/almi_engage.py" hams_ros:/tmp/almi_engage.py
  sudo_do docker cp "$HELPER_DIR/grasp_benchmark_topdown.py" hams_ros:/tmp/grasp_benchmark_topdown.py
  sudo_do docker cp "$HELPER_DIR/aggregate_2method.py" hams_ros:/tmp/aggregate_2method.py
  # HAMS_GRASP_BOX_SOURCE=gt makes the skill path (/skill/grasp) skip Gemini+SAM
  # and crop the object cloud around the GT centroid (mentor's quota-free GT
  # perception) -> removes the perception confound so grasps are fast/reliable
  # and can finish inside ALMI's ~25 s stable window before the base wanders.
  sudo_do docker exec -d hams_ros bash -lc "export HAMS_MAX_GRASP_ATTEMPTS=${SKILL_ATT:-40} HAMS_SKIP_VLM_SERVERS=$W_SRV HAMS_GRASP_BOX_SOURCE=gt HAMS_GRASP_GT_NAME=door_obj HAMS_GRASP_GT_WORLD=$GT_WORLD $TRIAL_ENV; source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash && ros2 launch h1_bringup h1_sim_bringup.launch.py use_rviz:=false use_nav:=false use_mjpc:=false > /tmp/bringup.log 2>&1"
  # Wait for ALL model servers + skills (GPU model loads can take minutes; a
  # trial started mid-load starves the sim -> ALMI control-loop instability ->
  # spontaneous wander). Abort + retry the env rather than proceed unready.
  READY=0
  for k in $(seq 1 200); do
    if [ "${W_SRV:-0}" = "1" ]; then
      if sudo_do docker exec hams_ros bash -c "grep -q 'h12_skills ready' /tmp/bringup.log 2>/dev/null && grep -q 'graspgen_server ready' /tmp/bringup.log 2>/dev/null"; then READY=1; break; fi
    else
      if sudo_do docker exec hams_ros bash -c "grep -q 'h12_skills ready' /tmp/bringup.log 2>/dev/null && grep -q 'graspgen_server ready' /tmp/bringup.log 2>/dev/null && grep -q 'sam_server ready' /tmp/bringup.log 2>/dev/null && grep -q 'gemini_server ready' /tmp/bringup.log 2>/dev/null"; then READY=1; break; fi
    fi
    sleep 3
  done
  if [ $READY -ne 1 ]; then echo "[env] bringup NOT ready after 600s — retry env"; fresh_env; return; fi
  sleep 3
  if [ -n "$POLICY" ]; then
    # world anchor for drift-compensated servoing (camera_init -> odom identity)
    in_ros_d "ros2 run tf2_ros static_transform_publisher --frame-id camera_init --child-frame-id odom --x 0 --y 0 --z 0 > /dev/null 2>&1"
    # Pre-home the arm + open the gripper UNDER THE FREEZE PIN, so the FIRST
    # trial starts with the arm already home and the benchmark go_home is a
    # no-op — un-pinned mid-trial homing is a stochastic fall hazard (flaky
    # named_config from awkward poses destabilizes the standing policy).
    in_ros "ros2 topic pub --times 8 --rate 5 /hams/reset_arm std_msgs/msg/Empty '{}' >/dev/null 2>&1; sleep 4
ros2 service call /right/gripper/open std_srvs/srv/Trigger >/dev/null 2>&1; sleep 2"
    if ! engage_policy; then echo "[env] engage FAILED (fell/displaced) — retry"; fresh_env; return; fi
  fi
  echo "[env] ready (policy='${POLICY:-band}') $(date +%H:%M:%S)"
}

retether_and_verify() {
  # WARM CYCLE: the policy stays engaged. Freeze -> in-sim full scene reset
  # (door+objects+robot teleported back to spawn UNDER the pin — recovers
  # falls/wanders too, no container rebuild) -> arm home + gripper open under
  # the pin -> short pinned re-conditioning + release + upright confirm (warm
  # policy: no LSTM cold-start settle) -> verify. Cold re-engage only if the
  # warm confirm fails; env rebuild only if THAT fails (caller handles).
  in_ros "ros2 topic pub --times 8 --rate 5 /hams/freeze_body std_msgs/msg/Bool '{data: true}' >/dev/null 2>&1"
  sleep 2
  in_ros "ros2 topic pub --times 8 --rate 5 /hams/reset_scene std_msgs/msg/Empty '{}' >/dev/null 2>&1"
  sleep 2
  in_ros "ros2 topic pub --times 8 --rate 5 /hams/reset_arm std_msgs/msg/Empty '{}' >/dev/null 2>&1
# Latch the upper-body controller's setpoint HOME while the sim pin holds the
# arm there (mentor's go_home trick): otherwise the stale frame_task target
# (last grasp pose) drags the arm off home the moment the pin releases, and
# the next trial starts from a stranded pose (measured: 547mm unreachable).
timeout 25 ros2 action send_goal /named_config custom_ros_messages/action/NamedConfig \"{config_name: 'home', duration: {sec: 3, nanosec: 0}}\" >/dev/null 2>&1
ros2 service call /right/gripper/open std_srvs/srv/Trigger >/dev/null 2>&1; sleep 2"
  mark_fell_baseline
  sudo_do docker exec hams_ros bash -c "rm -f /tmp/almi_ref.txt" 2>/dev/null
  WARM_TRY=0
  while [ $WARM_TRY -lt 2 ]; do
    WARM_TRY=$((WARM_TRY+1))
    if in_ros "STAB_SIM=6 CONFIRM_SIM=8 python3 /tmp/almi_engage.py"; then
      W_ENG=ok
    else
      W_ENG=fail
    fi
    W_FELL=no; robot_fell && W_FELL=YES
    W_DRIFT=no; robot_drifted && W_DRIFT=YES
    W_DOOR=ok; door_closed || W_DOOR=OPEN
    if [ "$W_ENG" = "ok" ] && [ "$W_FELL" = "no" ] && [ "$W_DRIFT" = "no" ] && [ "$W_DOOR" = "ok" ]; then
      capture_ref
      return 0
    fi
    echo "[2method] warm reset attempt $WARM_TRY failed (engage=$W_ENG fell=$W_FELL drift=$W_DRIFT door=$W_DOOR)"
    sudo_do docker exec hams_ros bash -c "tail -5 /tmp/lowerbody.log 2>/dev/null" | sed 's/^/[lb] /'
    # second attempt: re-freeze + re-reset the scene, then loop retries the engage
    in_ros "ros2 topic pub --times 8 --rate 5 /hams/freeze_body std_msgs/msg/Bool '{data: true}' >/dev/null 2>&1"
    sleep 2
    in_ros "ros2 topic pub --times 8 --rate 5 /hams/reset_scene std_msgs/msg/Empty '{}' >/dev/null 2>&1"
    sleep 2
    mark_fell_baseline
  done
  # in-session COLD re-engage reliably falls (0.55 upright, unexplained) — skip
  # the doomed rung and rebuild directly; the sweep resumes via trial-skip.
  echo "[2method] warm resets exhausted -> fresh env"
  return 1
}

run_trial() {  # $1=method $2=trial-id
  local M=$1 T=$2
  local EXTRA=""
  [ "$M" != "skill" ] && EXTRA="--max-attempts ${MAXATT:-20} --no-plan"
  # topdown_irl is this branch's verbatim port of main's goal.top_down; it runs
  # through the wrapper fresh_env copied to /tmp (same CLI, defers to the
  # INSTALLED grasp_benchmark for everything else -> no live-checkout rebuild).
  local RUNNER="ros2 run h12_skills grasp_benchmark"
  [ "$M" = "topdown_irl" ] && RUNNER="python3 /tmp/grasp_benchmark_topdown.py"
  in_ros "mkdir -p $OUTROOT/$M
for attempt in 1 2 3; do
  ros2 service call /right/gripper/open std_srvs/srv/Trigger >/dev/null 2>&1; sleep 2
  env $TRIAL_ENV timeout 400 $RUNNER --method $M \
    --object 'vertical fridge handle' --gt-name door_obj --arm right --box-source gt \
    --success-mode contact $EXTRA --out $OUTROOT/$M/trial_$T.json > $OUTROOT/$M/trial_$T.log 2>&1
  [ \$? -eq 124 ] && [ ! -s $OUTROOT/$M/trial_$T.json ] && python3 -c \"import json;json.dump({'method':'$M','success':False,'executed':False,'error':'harness timeout'},open('$OUTROOT/$M/trial_$T.json','w'),indent=2)\"
  P=\$(python3 -c \"import json;d=json.load(open('$OUTROOT/$M/trial_$T.json'));l=(str(d.get('chosen_label',''))+' '+str(d.get('error',''))).lower();print(1 if any(k in l for k in ('no mask','no box','no grasp planned','no candidate','synthesis produced no')) else 0)\" 2>/dev/null)
  [ \"\$P\" != \"1\" ] && break
  echo \"[trial $M/$T] perception no-op; retrying\"
done
python3 /tmp/grab_head.py >/dev/null 2>&1 && cp -f /tmp/head_view.jpg $OUTROOT/$M/trial_${T}_head.jpg 2>/dev/null
python3 -c \"import json;d=json.load(open('$OUTROOT/$M/trial_$T.json'));print('  -> $M succ=%s gfinal=%s tip=%s | %s'%(d.get('success'),d.get('grip_final_mm'),d.get('tip_to_handle_mm'),str(d.get('chosen_label'))[:34]))\" 2>/dev/null; true"
}

trial_qa_ok() {
  local out=$1 tel=$2 bag=$3
  in_ros "python3 - << 'PY'
import json, os, sys
ok = True
try:
    d = json.load(open('$out'))
    ok &= 'success' in d and 'method' in d
except Exception:
    ok = False
try:
    ok &= (sum(1 for _ in open('$tel')) - 1) >= 50
except Exception:
    ok = False
ok &= os.path.isdir('$bag') and any(fn.endswith(('.db3', '.mcap')) for fn in (os.listdir('$bag') if os.path.isdir('$bag') else []))
sys.exit(0 if ok else 1)
PY"
  [ $? -eq 0 ]
}

echo "[2method] START methods=[$METHODS] N=$N policy='${POLICY:-band}' out=$OUTROOT $(date +%H:%M:%S)"
BENCH_DONE=0
for M in $METHODS; do
  FIRST=1
  fresh_env
  if [ -n "$POLICY" ] && [ $BENCH_DONE -eq 0 ]; then
    sudo_do docker cp "$HELPER_DIR/almi_inference_bench.py" hams_ros:/tmp/almi_inference_bench.py
    in_ros "mkdir -p $OUTROOT; python3 /tmp/almi_inference_bench.py > $OUTROOT/almi_inference.json 2>/dev/null" && BENCH_DONE=1
  fi
  i=1
  while [ $i -le $N ]; do
    T=$(printf "%02d" $i)
    if sudo_do docker exec hams_ros bash -c "test -f $OUTROOT/$M/trial_$T.json" 2>/dev/null; then
      echo "[2method] skip $M/$T"; i=$((i+1)); continue
    fi
    # --- controlled start ---
    if [ "$RAND" = "1" ]; then
      # Randomized-start: sample this trial's spawn (deterministic seed = the
      # cell/trial id, so a resumed or re-run sweep reproduces the same spawns)
      # and rebuild the env there. QA re-runs of the same trial re-sample the
      # SAME spawn by construction.
      read -r SPAWN_FWD_VAL SPAWN_LAT_VAL <<< "$(python3 -c "
import random
r = random.Random('$M/$T/rand-v2')
s1, s2 = $RAND_SIG_LONG, $RAND_SIG_LAT
f = max(-3*s1, min(3*s1, r.gauss(0.0, s1)))
l = max(-3*s2, min(3*s2, r.gauss(0.0, s2)))
print(round($BASE_FWD+f, 4), round($BASE_LAT+l, 4))")"
      echo "[2method] $M/$T randomized spawn: fwd=$SPAWN_FWD_VAL lat=$SPAWN_LAT_VAL (base $BASE_FWD/$BASE_LAT)"
      FIRST=0
      fresh_env
      mkdir -p "$HOST_OUT/$M"
      printf '{"spawn_forward_m": %s, "spawn_lateral_m": %s, "base_forward_m": %s, "base_lateral_m": %s, "sigma_long_m": %s, "sigma_lat_m": %s, "clip": "3sigma", "seed": "%s"}\n' \
        "$SPAWN_FWD_VAL" "$SPAWN_LAT_VAL" "$BASE_FWD" "$BASE_LAT" "$RAND_SIG_LONG" "$RAND_SIG_LAT" "$M/$T/rand-v2" \
        > "$HOST_OUT/$M/trial_${T}_spawn.json"
    elif [ -n "$POLICY" ]; then
      if [ $FIRST -eq 1 ]; then
        FIRST=0
      elif ! retether_and_verify; then
        echo "[2method] re-tether verify FAILED -> rebuild"; fresh_env
      fi
    else
      if robot_fell; then echo "[2method] fell -> rebuild"; fresh_env; fi
      if ! gripper_ok; then echo "[2method] gripper damaged -> rebuild"; fresh_env; fi
    fi
    if [ -n "$POLICY" ]; then
      # settle gate: don't start a trial on a still-transient base
      SETTLED=0
      for g in 1 2 3; do
        if _drift_val 0.05 8.0; then SETTLED=1; break; fi
        echo "[2method] settle-gate: transient drift, waiting"; sleep 12
      done
      if [ $SETTLED -ne 1 ]; then
        if ! retether_and_verify; then echo "[2method] settle-gate rebuild"; fresh_env; fi
      fi
    fi
    mark_fell_baseline
    echo "[2method] === $M trial $T $(date +%H:%M:%S) ==="
    # --- telemetry + playback bag ---
    in_ros_d "python3 /tmp/trial_recorder.py > /tmp/trial_telemetry.csv 2>/dev/null"
    in_ros_d "rm -rf /tmp/trial_bag; ros2 bag record -o /tmp/trial_bag /robocasa/object_poses /cmd_vel /right/gripper/state /realsense/head/color/image_raw/compressed /joint_states > /dev/null 2>&1"
    # --- the trial (wander-watchdogged on the standing tier) ---
    if [ -n "$POLICY" ]; then
      run_trial $M $T &
      TPID=$!
      WANDERED=0
      while kill -0 $TPID 2>/dev/null; do
        sleep 20
        if kill -0 $TPID 2>/dev/null && hard_drift; then
          WANDERED=1
          sudo_do docker exec hams_ros bash -c "pkill -9 -f 'grasp_benchmar[k]'" 2>/dev/null
        fi
      done
      wait $TPID 2>/dev/null
      if [ $WANDERED -eq 1 ]; then
        in_ros "python3 -c \"import json,os;p='$OUTROOT/$M/trial_$T.json';d=json.load(open(p)) if os.path.exists(p) else {'method':'$M'};d['success']=False;d['base_wandered']=True;json.dump(d,open(p,'w'),indent=2)\""
        echo "[2method] $M/$T BASE WANDERED mid-trial — graded fail"
      fi
    else
      run_trial $M $T
    fi
    # --- stop + stash telemetry ---
    sudo_do docker exec hams_ros bash -c "pkill -TERM -f trial_recorder.py; pkill -INT -f 'ros2 bag record'" 2>/dev/null
    # snapshot the skills-node log (servo/gt-point diagnostics live there, and
    # /tmp/bringup.log dies with every env rebuild)
    in_ros "grep -E 'gt-point|servo\[|camera_init err|unreachable|grasp' /tmp/bringup.log 2>/dev/null | tail -400 > $OUTROOT/$M/trial_${T}_skillslog.txt; true"
    sleep 2
    in_ros "cp -f /tmp/trial_telemetry.csv $OUTROOT/$M/trial_${T}_telemetry.csv 2>/dev/null; rm -rf $OUTROOT/$M/trial_${T}_bag; cp -r /tmp/trial_bag $OUTROOT/$M/trial_${T}_bag 2>/dev/null; true"
    if robot_fell; then
      in_ros "python3 -c \"import json;p='$OUTROOT/$M/trial_$T.json';d=json.load(open(p));d['robot_fell_during_trial']=True;json.dump(d,open(p,'w'),indent=2)\"" 2>/dev/null
      echo "[2method] fell during $M/$T (annotated)"
    fi
    # --- QA gate ---
    if ! trial_qa_ok "$OUTROOT/$M/trial_$T.json" "$OUTROOT/$M/trial_${T}_telemetry.csv" "$OUTROOT/$M/trial_${T}_bag"; then
      REDO=$(( $(cat /tmp/redo_${M}_${T} 2>/dev/null || echo 0) + 1 ))
      echo $REDO > /tmp/redo_${M}_${T}
      if [ $REDO -le 2 ]; then
        echo "[2method] QA FAILED $M/$T (attempt $REDO) — re-running"
        sudo_do docker exec hams_ros bash -c "rm -rf $OUTROOT/$M/trial_$T.json $OUTROOT/$M/trial_${T}_telemetry.csv $OUTROOT/$M/trial_${T}_bag $OUTROOT/$M/trial_$T.log" 2>/dev/null
        continue
      fi
      echo "[2method] QA failed 3x for $M/$T — flagged, keeping"
    fi
    i=$((i+1))
  done
done
if [ "$RAND" = "1" ]; then   # leave the TUNED spawn in .env, not the last sample
  sed -i "s/^HAMS_SPAWN_FORWARD=.*/HAMS_SPAWN_FORWARD=$BASE_FWD/" docker/.env
  sed -i "s/^HAMS_SPAWN_LATERAL=.*/HAMS_SPAWN_LATERAL=$BASE_LAT/" docker/.env
fi
echo "[2method] COMPLETE $(date +%H:%M:%S)"
echo "[2method] aggregate with (the study copy — the live aggregate.py ignores topdown_irl):"
echo "  sudo docker cp $HELPER_DIR/aggregate_2method.py hams_ros:/tmp/aggregate_2method.py && \\"
echo "  sudo docker exec hams_ros bash -lc 'source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; python3 /tmp/aggregate_2method.py $OUTROOT'"
