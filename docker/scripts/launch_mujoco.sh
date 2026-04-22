#!/bin/bash
# Usage: ./launch_mujoco.sh [--fixed] [--force link1 link2 ...] [--viewer]
# Headless MuJoCo sim. Visualize via RViz/Foxglove over ROS 2.
# Sim publishes:
#   - rt/lowstate on CycloneDDS domain 1 (unitree_sdk2py)
#   - /head/color/image_raw, /head/depth/image_raw, /head/color/camera_info,
#     /lidar/points, /tf on ROS 2 (ROS_DOMAIN_ID=1 unless overridden)
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate humanoid_sim_env

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
# When --viewer is requested, force GLFW (the passive viewer needs a window).
case " $* " in
    *" --viewer "*) export MUJOCO_GL=glfw ;;
    *) export MUJOCO_GL="${MUJOCO_GL:-egl}" ;;
esac

cd /home/code/h1_mujoco
# Default to the pelvis-fixed scene: the free-standing elastic-band scene has
# startup velocity transients on shoulder_pitch that trip h12_ros2_controller's
# estop before any control input runs. Pass args explicitly to override.
if [ "$#" -eq 0 ]; then
    set -- --fixed
fi
python h12_mujoco.py "$@"
