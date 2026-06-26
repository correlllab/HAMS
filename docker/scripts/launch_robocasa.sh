#!/bin/bash
# Usage: ./launch_robocasa.sh [--headless]
# Windowed MuJoCo sim by default; pass --headless to disable the passive viewer
# (auto-applied when no DISPLAY is reachable, e.g. SSH without X11 / CI).
# Sim publishes:
#   - rt/lowstate on CycloneDDS domain 1 (unitree_sdk2py)
#   - /head/color/image_raw, /head/depth/image_raw, /head/color/camera_info,
#     /lidar/points, /tf on ROS 2 (ROS_DOMAIN_ID=1 unless overridden)
set -e

source /opt/ros/humble/setup.bash
# livox_ros_driver2 (CustomMsg/CustomPoint) is baked into the robocasa image at
# /opt/livox_ws by RobocasaDockerfile. Source it so mujoco_ros_bridge.py can
# import livox_ros_driver2.msg.
source /opt/livox_ws/install/setup.bash

# Run colcon every launch — it's a fast no-op when nothing's changed.
# build/install/log are bind-mounted from the host at container_cache/msgs_ws/
# (see docker-compose.yml), so they persist across `docker compose run --rm`
# cycles. Wipe that host directory if you need a clean rebuild.
MSGS_WS=/home/code/msgs_ws
echo "[launch_robocasa] building $MSGS_WS"
(cd "$MSGS_WS" && colcon build --symlink-install \
    --packages-select magpie_msgs custom_ros_messages)
source "$MSGS_WS/install/setup.bash"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"

# Auto-add --headless if no display is reachable (SSH without X11, cloud VM, CI).
case " $* " in
    *" --headless "*) ;;
    *)
        if [ -z "${DISPLAY:-}" ]; then
            echo "[launch_robocasa] no DISPLAY set — forcing --headless"
            set -- "$@" --headless
        fi
        ;;
esac

# MUJOCO_GL: glfw needs an X display, egl is offscreen.
case " $* " in
    *" --headless "*) export MUJOCO_GL=egl ;;
    *) export MUJOCO_GL=glfw ;;
esac

cd /home/code/h1_robocasa
python h12_mujoco.py "$@"
