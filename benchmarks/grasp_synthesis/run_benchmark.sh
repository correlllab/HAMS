#!/usr/bin/env bash
# Benchmark parallel-jaw grasp-synthesis methods on a RoboCasa task.
#
# For every (method, seed) pair this script brings up a FRESH sim episode and a
# fresh bringup launch (sim time restarts at 0 when the sim restarts, so bringup
# nodes must restart with it), runs one grasp_benchmark episode inside the
# hams_ros container, and collects the JSON result. Results land in
# core_ws/benchmark_results/ (bind-mounted, so visible on the host).
#
# Usage:
#   benchmarks/grasp_synthesis/run_benchmark.sh                       # all methods, seeds 42..44
#   benchmarks/grasp_synthesis/run_benchmark.sh -m graspgenx -s "42 43 44 45 46"
#   benchmarks/grasp_synthesis/run_benchmark.sh -t CheesyBread -o "wedge of cheese" -g cheese
#
# Prerequisites: hams images built, NVIDIA container toolkit installed,
# docker/.env populated (GEMINI_API_KEY), SAM3 weights at
# core_ws/src/model_server/weights/sam3.pt.
set -euo pipefail
cd "$(dirname "$0")/../.."

METHODS="centroid topdown_antipodal graspgenx vlm_judge"
SEEDS="42 43 44"
TASK="CheesyBread"
# Plain "cheese" — the gemini detector (gemini-robotics-er-1.6-preview) returns
# an empty box for the compound "wedge of cheese" on the RoboCasa CheesyBread
# asset (it reads the object as plain cheese / the bread as a bagel), which
# starved SAM of a box and produced zero grasp candidates. "cheese" yields a
# clean box -> mask -> 17 GraspGenX candidates.
OBJECT="cheese"
GT_NAME="cheese"
ARM="right"
LAYOUT=1
STYLE=1

while getopts "m:s:t:o:g:a:l:y:" opt; do
    case "$opt" in
        m) METHODS="$OPTARG" ;;
        s) SEEDS="$OPTARG" ;;
        t) TASK="$OPTARG" ;;
        o) OBJECT="$OPTARG" ;;
        g) GT_NAME="$OPTARG" ;;
        a) ARM="$OPTARG" ;;
        l) LAYOUT="$OPTARG" ;;
        y) STYLE="$OPTARG" ;;
        *) exit 2 ;;
    esac
done

COMPOSE="docker compose -f docker/docker-compose.yml"
RESULTS_DIR="core_ws/benchmark_results"
mkdir -p "$RESULTS_DIR"
# Trial videos come from the robocasa (sim) container, which only bind-mounts
# h1_robocasa — not core_ws — so they land alongside the sim under
# h1_robocasa/benchmark_videos/ (host path); the JSON results stay in core_ws.
VIDEOS_DIR="h1_robocasa/benchmark_videos"
mkdir -p "$VIDEOS_DIR"

# docker/.env drives ROS_DOMAIN_ID + GEMINI_API_KEY for every container.
if [ -f docker/.env ]; then set -a; source docker/.env; set +a; fi
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"

cleanup() {
    docker rm -f hams_sim_robocasa >/dev/null 2>&1 || true
    docker rm -f hams_ros >/dev/null 2>&1 || true
}
trap cleanup EXIT

wait_for() {   # wait_for <timeout_s> <description> <command...>
    local timeout=$1 desc=$2; shift 2
    local t=0
    until "$@" >/dev/null 2>&1; do
        sleep 2; t=$((t + 2))
        if [ "$t" -ge "$timeout" ]; then
            echo "TIMEOUT waiting for $desc" >&2
            # Dump the tails of every log we have BEFORE the EXIT trap removes
            # the containers, so a timeout is debuggable from the terminal.
            for c in hams_sim_robocasa hams_ros; do
                echo "--- docker logs $c (tail) ---" >&2
                docker logs --tail 25 "$c" >&2 2>&1 || true
            done
            echo "--- bringup logs in hams_ros (tail) ---" >&2
            docker exec hams_ros bash -c 'tail -n 40 /tmp/bringup_*.log' >&2 2>&1 || true
            return 1
        fi
    done
    echo "  ready: $desc (${t}s)"
}

# ---------------------------------------------------------------- ros container
# One persistent workspace container for the whole run (colcon build once);
# bringup is (re)launched inside it per episode. launch_ros.sh ends in `exec
# bash`, which stays alive because the compose service allocates a TTY.
echo "== starting hams_ros workspace container"
cleanup
$COMPOSE --profile ros run -d --rm --remove-orphans --name hams_ros ros \
    /home/code/h12_sim_scripts/launch_ros.sh
# Readiness = a LATE package is installed, not just install/setup.bash: a
# partial install (e.g. a packages-select run) satisfies launch_ros.sh's
# "up to date" heuristic while missing most of the workspace.
wait_for 1800 "core_ws colcon build (model_server installed)" \
    docker exec hams_ros test -d /home/code/core_ws/install/model_server
# Force-build h12_skills so the grasp_benchmark entry point exists even when
# launch_ros.sh skipped the build on an up-to-date install/.
docker exec hams_ros bash -lc \
    "source /opt/ros/humble/setup.bash && cd /home/code/core_ws && \
     colcon build --symlink-install --packages-select h12_skills"

IN_ROS="docker exec hams_ros bash -lc"
SRC="source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash && export ROS_DOMAIN_ID=$ROS_DOMAIN_ID"

for seed in $SEEDS; do
    for method in $METHODS; do
        stamp="${TASK}_${method}_seed${seed}"
        out_json="/home/code/core_ws/benchmark_results/${stamp}.json"
        echo
        echo "== episode: $stamp"

        # --- fresh sim -------------------------------------------------------
        docker rm -f hams_sim_robocasa >/dev/null 2>&1 || true
        # No --remove-orphans here: with only the robocasa profile active, compose
        # treats the ros-profile hams_ros as an orphan and deletes it mid-run.
        $COMPOSE --profile robocasa run -d --rm \
            --name hams_sim_robocasa robocasa \
            /home/code/h12_sim_scripts/launch_robocasa.sh --headless \
            --task "$TASK" --layout "$LAYOUT" --style "$STYLE" --seed "$seed" \
            --record-video "/home/code/h1_robocasa/benchmark_videos/${stamp}.mp4"
        wait_for 300 "sim ROS bridges" \
            bash -c "docker logs hams_sim_robocasa 2>&1 | grep -q 'ROS bridges up'"

        # --- fresh bringup (nodes must restart with the sim clock) ------------
        $IN_ROS "pkill -INT -f h1_sim_bringup" || true
        sleep 5
        # Headless bringup: rviz + the slider GUI would land on the host X
        # display and contend for the GPU with the sim/SAM3/GraspGenX (a full
        # default bringup has frozen the desktop); nav (FAST-LIO) and the MJPC
        # lower-body estimator+controller aren't needed since the robot is held
        # by the elastic-band tether and the benchmark drives the upper-body
        # frame_task IK in the pelvis frame directly — dropping MJPC also frees
        # RAM/GPU/CPU for SAM3 + GraspGenX.
        docker exec -d hams_ros bash -lc \
            "$SRC && ros2 launch h1_bringup h1_sim_bringup.launch.py \
             use_rviz:=false use_sliders:=false use_nav:=false use_mjpc:=false \
             model_visualization:=false \
             > /tmp/bringup_${stamp}.log 2>&1"
        wait_for 300 "graspgen service" \
            $IN_ROS "$SRC && ros2 service list | grep -q graspgen"
        wait_for 300 "sam_segment service" \
            $IN_ROS "$SRC && ros2 service list | grep -q sam_segment"
        wait_for 120 "named_config action" \
            $IN_ROS "$SRC && ros2 action list | grep -q named_config"
        sleep 5   # settle: model warmup, first camera frames, TF tree

        # --- the episode -------------------------------------------------------
        if ! $IN_ROS "$SRC && ros2 run h12_skills grasp_benchmark \
                --method $method --object '$OBJECT' --gt-name $GT_NAME \
                --arm $ARM --out $out_json"; then
            echo "episode $stamp FAILED (see container logs)" >&2
        fi
        $IN_ROS "pkill -INT -f h1_sim_bringup" || true
        docker rm -f hams_sim_robocasa >/dev/null 2>&1 || true
    done
done

echo
echo "== summary"
python3 benchmarks/grasp_synthesis/summarize.py "$RESULTS_DIR"
