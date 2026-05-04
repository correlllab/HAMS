#!/bin/bash
# Usage: ./launch_isaac.sh [--reset-cache] [--headless] [task_name]
# Default task: Isaac-Stack-RgyBlock-H12-27dof-Inspire-Joint
# Set HEADLESS=1 in the environment (or pass --headless) to run without a viewport.
#
# ROS publishing uses Isaac-Sim's bundled isaacsim.ros2.bridge extension
# (loaded by Kit at app startup); there is no /opt/ros/humble in this image
# and rclpy is not installed.
set -e

TASK="Isaac-PickPlace-Cylinder-H12-27dof-Inspire-Joint"
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
# Use the conda env's interpreter directly. Compose runs this script
# non-interactively, so ~/.bashrc (which `conda activate`s the env) is not
# sourced — bare `python` would resolve to the base conda Python without
# isaacsim/isaaclab installed.
python /home/code/h12_sim_scripts/dds_bridge.py < /dev/tty > /dev/tty 2>&1 & disown
exec /opt/conda/envs/unitree_sim_env/bin/python -u \
    /home/code/CL_isaaclab_sim/sim_main.py \
    --device cuda \
    --enable_cameras \
    $HEADLESS_FLAG \
    --task "$TASK"
#/bin/bash
if [ "$1" == "--reset-cache" ]; then 
  rm -rf ~/.cache/ov/texturecache
fi
python /home/code/h12_sim_scripts/dds_bridge.py < /dev/tty > /dev/tty 2>&1 & disown
python3  /home/code/CL_isaaclab_sim/sim_main.py \
  --device cuda \
  --task Isaac-PickPlace-Cylinder-H12-27dof-Inspire-Joint \
  --enable_inspire_dds \
  --enable_cameras \
  --robot_type h1_2  < /dev/tty > /dev/tty 2>&1
