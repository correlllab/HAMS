#!/bin/bash
# Usage: ./launch.sh [--reset-cache] [--headless] [task_name]
# Default task: Isaac-PickPlace-Cylinder-H12-27dof-Inspire-Joint
# Set HEADLESS=1 in the environment (or pass --headless) to run without a viewport.
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate humanoid_sim_env

# Leave RMW_IMPLEMENTATION unset so rclpy picks the default (FastRTPS). Using
# rmw_cyclonedds_cpp here collides with the PyPI cyclonedds wheel that
# unitree_sdk2py drags in: both would share one libddsc mapping, and the
# second `dds_create_domain(1)` fails Precondition Not Met. FastRTPS and
# CycloneDDS interoperate over RTPS on ROS_DOMAIN_ID=1, so the controller
# sees publishers from both paths. Do not set this unless Isaac Sim's C++
# ros2_bridge (OmniGraph ROS2 action graphs) is actually in use — pure
# rclpy in ros_bridge.py does not need it.
TASK="Isaac-Stack-RgyBlock-H12-27dof-Inspire-Joint"
HEADLESS_FLAG=""
[ "${HEADLESS:-0}" = "1" ] && HEADLESS_FLAG="--headless"
# Auto-headless if no display is reachable (SSH without X11, cloud VM, CI).
if [ -z "$HEADLESS_FLAG" ] && [ -z "${DISPLAY:-}" ]; then
    echo "[launch_isaac] no DISPLAY set — forcing --headless"
    HEADLESS_FLAG="--headless"
fi

for arg in "$@"; do
    case "$arg" in
        --reset-cache) [ -d "$HOME/.cache/ov/texturecache" ] && rm -rf "$HOME/.cache/ov/texturecache" ;;
        --headless)    HEADLESS_FLAG="--headless" ;;
        --*)           ;;  # ignore other flags
        *)             TASK="$arg" ;;
    esac
done

export PYTHONUNBUFFERED=1
python -u /home/code/CL_isaaclab_sim/sim_main.py \
    --device cuda \
    --enable_cameras \
    $HEADLESS_FLAG \
    --task "$TASK"
