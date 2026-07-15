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
# Perception source: 'gemini' (detector) or 'gt' (crop the cloud around the
# ground-truth centroid, no detector, zero API calls). Default 'gt' — it's
# reproducible and keeps a matrix off the gemini free-tier quota; pass -b gemini
# for an end-to-end run that includes detection.
BOX_SOURCE="gt"

while getopts "m:s:t:o:g:a:l:y:b:" opt; do
    case "$opt" in
        m) METHODS="$OPTARG" ;;
        s) SEEDS="$OPTARG" ;;
        t) TASK="$OPTARG" ;;
        o) OBJECT="$OPTARG" ;;
        g) GT_NAME="$OPTARG" ;;
        a) ARM="$OPTARG" ;;
        l) LAYOUT="$OPTARG" ;;
        y) STYLE="$OPTARG" ;;
        b) BOX_SOURCE="$OPTARG" ;;
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
# Cache gemini_server responses for the whole matrix (see docker-compose.yml).
# A seeded episode renders the same head-cam frame every run, so every method
# gets a BYTE-IDENTICAL box — the fairness property the comparison depends on —
# and a re-run costs no API calls. Without this a matrix exhausts the free tier's
# 20-requests/day quota partway through and the 429s look exactly like
# "perception found nothing". Set GEMINI_CACHE_DIR= to force live calls.
export GEMINI_CACHE_DIR="${GEMINI_CACHE_DIR-/home/code/core_ws/benchmark_results/gemini_cache}"

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
# Force-build h12_skills AND its message deps, so the grasp_benchmark entry point
# exists even when launch_ros.sh skipped the build on an up-to-date install/.
# --packages-up-to (not --packages-select) is load-bearing: launch_ros.sh's "up to
# date" heuristic looks only at install/, so after a custom_ros_messages bump that
# adds an action (e.g. SkillFrontierExplore) the generated python is missing while
# install/ still looks complete — the skills node then dies at IMPORT and every
# episode fails for a reason that has nothing to do with grasping. Building up-to
# h12_skills regenerates the messages it imports.
docker exec hams_ros bash -lc \
    "source /opt/ros/humble/setup.bash && cd /home/code/core_ws && \
     colcon build --symlink-install --packages-up-to h12_skills"

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
        # /skill/grasp specifically: named_config is served by h12_ros2_controller,
        # NOT the skills node, so it comes up even when h12_skills has died at
        # import (a stale custom_ros_messages install/ did exactly that). Without
        # this gate the episode runs anyway and fails for an unrelated-looking reason.
        wait_for 120 "skills node (/skill/grasp action)" \
            $IN_ROS "$SRC && ros2 action list | grep -q '/skill/grasp'"
        sleep 5   # settle: model warmup, first camera frames, TF tree

        # --- the episode -------------------------------------------------------
        if ! $IN_ROS "$SRC && ros2 run h12_skills grasp_benchmark \
                --method $method --object '$OBJECT' --gt-name $GT_NAME \
                --arm $ARM --box-source $BOX_SOURCE --out $out_json"; then
            echo "episode $stamp FAILED (see container logs)" >&2
        fi
        # Salvage the bringup log BEFORE the containers go away: it holds the
        # skill's own filter/tier/IK lines (how many candidates survived each
        # stage, which ones were tried, why each was rejected), which is the only
        # record of WHY an episode failed. The EXIT trap removes the containers,
        # so a log left inside one is lost exactly when it's most wanted.
        docker cp "hams_ros:/tmp/bringup_${stamp}.log" \
            "$RESULTS_DIR/${stamp}.bringup.log" >/dev/null 2>&1 || true
        $IN_ROS "pkill -INT -f h1_sim_bringup" || true
        docker rm -f hams_sim_robocasa >/dev/null 2>&1 || true
    done
done

echo
echo "== summary"
python3 benchmarks/grasp_synthesis/summarize.py "$RESULTS_DIR"
