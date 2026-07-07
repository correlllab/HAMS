#!/bin/bash
# Apple-Silicon / CPU launcher for the MuJoCo (RoboCasa) sim.
# Same as launch_robocasa.sh but forces OSMesa software rendering and headless
# operation (no NVIDIA EGL, no X display under Colima). Extra args pass through
# to h12_mujoco.py.
set -e

source /opt/ros/humble/setup.bash
# livox_ros_driver2 (CustomMsg/CustomPoint) is baked into the robocasa image at
# /opt/livox_ws so mujoco_ros_bridge.py can import it.
source /opt/livox_ws/install/setup.bash

# Build the IDL-only msgs workspace (fast no-op when unchanged).
# build/install go to /opt paths backed by named docker volumes on the VM's
# native ext4: writing colcon's --symlink-install tree onto the virtiofs
# bind-mount stalls for minutes (hundreds of symlinks at ~0% CPU). Named volumes
# are both fast and persistent, so this is a real no-op on unchanged rebuilds.
MSGS_WS=/home/code/msgs_ws
MSGS_BUILD=/opt/msgs_ws/build
MSGS_INSTALL=/opt/msgs_ws/install
echo "[launch_robocasa_mac] building $MSGS_WS -> $MSGS_INSTALL"
(cd "$MSGS_WS" && colcon build --symlink-install \
    --build-base "$MSGS_BUILD" --install-base "$MSGS_INSTALL" \
    --packages-select magpie_msgs custom_ros_messages)
source "$MSGS_INSTALL/setup.bash"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"

# Pure-software offscreen rendering — the only backend available without a GPU.
export MUJOCO_GL=osmesa

# Force headless: there is no reachable X display inside the Colima VM.
case " $* " in
    *" --headless "*) ;;
    *) set -- "$@" --headless ;;
esac

cd /home/code/h1_robocasa
echo "[launch_robocasa_mac] MUJOCO_GL=$MUJOCO_GL ROS_DOMAIN_ID=$ROS_DOMAIN_ID args: $*"
python h12_mujoco.py "$@"
