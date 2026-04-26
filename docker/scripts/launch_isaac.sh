#!/bin/bash
# Usage: ./launch_isaac.sh [--reset-cache] [--headless] [task_name]
# Default task: Isaac-Stack-RgyBlock-H12-27dof-Inspire-Joint
# Set HEADLESS=1 in the environment (or pass --headless) to run without a viewport.
set -e

source /opt/ros/humble/setup.bash

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
