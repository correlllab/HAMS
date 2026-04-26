#!/bin/bash
# Usage: ./launch_mujoco.sh [--fixed] [--force link1 link2 ...] [--headless]
# Windowed MuJoCo sim by default; pass --headless to disable the passive viewer
# (auto-applied when no DISPLAY is reachable, e.g. SSH without X11 / CI).
# Sim publishes:
#   - rt/lowstate on CycloneDDS domain 1 (unitree_sdk2py)
#   - /head/color/image_raw, /head/depth/image_raw, /head/color/camera_info,
#     /lidar/points, /tf on ROS 2 (ROS_DOMAIN_ID=1 unless overridden)
set -e

source /opt/ros/humble/setup.bash

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"

# Default to the pelvis-fixed scene: the free-standing elastic-band scene has
# startup velocity transients on shoulder_pitch that trip h12_ros2_controller's
# estop before any control input runs. Pass args explicitly to override.
if [ "$#" -eq 0 ]; then
    set -- --fixed
fi

# Auto-add --headless if no display is reachable (SSH without X11, cloud VM, CI).
case " $* " in
    *" --headless "*) ;;
    *)
        if [ -z "${DISPLAY:-}" ]; then
            echo "[launch_mujoco] no DISPLAY set — forcing --headless"
            set -- "$@" --headless
        fi
        ;;
esac

# MUJOCO_GL: glfw needs an X display, egl is offscreen.
case " $* " in
    *" --headless "*) export MUJOCO_GL=egl ;;
    *) export MUJOCO_GL=glfw ;;
esac

cd /home/code/h1_mujoco
python h12_mujoco.py "$@"
