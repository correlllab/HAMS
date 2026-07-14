#!/usr/bin/env bash
# Grasp-synthesis benchmark matrix, reusing ONE long-lived hams_ros container.
#
# Why not run_benchmark.sh: that script tears down and recreates hams_ros, and the
# fresh container re-runs launch_ros.sh (MJPC ninja rebuild + colcon) whose parallel
# compiles spike host RAM hard enough to trip the memory watchdog on this 30 GB box.
# Here hams_ros is started once and kept; only the SIM and the bringup PROCESS are
# recycled per episode (the sim clock restarts at 0, so bringup must restart with it).
#
# Usage: benchmarks/grasp_synthesis/run_matrix.sh [-m "methods"] [-s "seeds"]
set -uo pipefail
cd "$(dirname "$0")/../.."

METHODS="centroid topdown_antipodal graspgenx vlm_judge"
SEEDS="42 43 44"
TASK="CheesyBread"; OBJECT="cheese"; GT_NAME="cheese"; ARM="right"; LAYOUT=1; STYLE=1
while getopts "m:s:t:o:g:a:" opt; do case "$opt" in
  m) METHODS="$OPTARG";; s) SEEDS="$OPTARG";; t) TASK="$OPTARG";;
  o) OBJECT="$OPTARG";; g) GT_NAME="$OPTARG";; a) ARM="$OPTARG";; *) exit 2;; esac; done

set -a; [ -f docker/.env ] && source docker/.env; set +a
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
COMPOSE="docker compose -f docker/docker-compose.yml"
IN_ROS="docker exec hams_ros bash -lc"
SRC="source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash && export ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
RESULTS=core_ws/benchmark_results
mkdir -p "$RESULTS"

mem(){ awk '/MemAvailable/{printf "      [mem] avail=%.1fGB\n", $2/1048576}' /proc/meminfo; }
wait_for(){ local to=$1 d=$2; shift 2; local t=0
  until "$@" >/dev/null 2>&1; do sleep 3; t=$((t+3)); [ $t -ge "$to" ] && { echo "      TIMEOUT: $d"; return 1; }; done
  echo "      ready: $d (${t}s)"; }

# --- one long-lived ros container (models load once per episode's bringup) --------
# NOTE: never pass --remove-orphans to the per-episode robocasa run below. With only
# the robocasa profile active, compose treats the ros-profile hams_ros as an orphan
# and DELETES it mid-matrix, after which every docker exec in the loop fails.
ensure_ros(){
  docker ps --format '{{.Names}}' | grep -qx hams_ros && return 0
  echo "== starting hams_ros (kept for the whole matrix)"
  $COMPOSE --profile ros run -d --rm --name hams_ros ros \
      /home/code/h12_sim_scripts/launch_ros.sh >/dev/null
  wait_for 900 "core_ws install" $IN_ROS "test -d /home/code/core_ws/install/model_server"
}
ensure_ros || exit 1

for seed in $SEEDS; do
  for method in $METHODS; do
    stamp="${TASK}_${method}_seed${seed}"
    echo; echo "=== $stamp ==="
    ensure_ros || { echo "      hams_ros gone and could not restart"; continue; }
    $IN_ROS "pkill -INT -f h1_sim_bringup" 2>/dev/null || true; sleep 3
    docker rm -f hams_sim_robocasa >/dev/null 2>&1 || true
    $COMPOSE --profile robocasa run -d --rm --name hams_sim_robocasa robocasa \
        /home/code/h12_sim_scripts/launch_robocasa.sh --headless \
        --task "$TASK" --layout "$LAYOUT" --style "$STYLE" --seed "$seed" >/dev/null
    wait_for 300 "sim" bash -c "docker logs hams_sim_robocasa 2>&1 | grep -q 'ROS bridges up'" || continue
    docker exec -d hams_ros bash -lc "$SRC && ros2 launch h1_bringup h1_sim_bringup.launch.py \
        use_rviz:=false use_sliders:=false use_nav:=false use_mjpc:=false \
        model_visualization:=false > /tmp/bringup_${stamp}.log 2>&1"
    wait_for 300 "graspgen"    $IN_ROS "$SRC && ros2 service list | grep -q graspgen"   || continue
    wait_for 180 "named_config" $IN_ROS "$SRC && ros2 action list | grep -q named_config" || continue
    sleep 5
    $IN_ROS "$SRC && ros2 run h12_skills grasp_benchmark --method $method \
        --object '$OBJECT' --gt-name $GT_NAME --arm $ARM \
        --out /home/code/${RESULTS}/${stamp}.json" >/dev/null 2>&1
    python3 - "$RESULTS/${stamp}.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"      -> NO RESULT ({e})"); raise SystemExit
print("      -> executed={} label={} lift_dz={} SUCCESS={} {}".format(
    d.get('executed'), d.get('chosen_label'), d.get('lift_dz_m'), d.get('success'),
    ('err=' + d['error']) if d.get('error') else ''))
PY
    mem
  done
done

echo; echo "================ SUMMARY ================"
python3 benchmarks/grasp_synthesis/summarize.py "$RESULTS"
