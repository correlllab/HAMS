#!/bin/bash
# Apple-Silicon / CPU launcher for the SLIM ROS workspace.
# Builds only the core bringup package subset (skips the vision / grasp / nav2
# / FAST-LIO packages that need the heavy ML stack), then launches the minimal
# Mac bringup (robot_state_publisher, joint_state_publisher, Pink IK
# frame_task_server, safety_node) against the running MuJoCo sim.
#
# Pass `bash` as the first arg to drop to a shell instead of launching.
set -e

source /opt/ros/humble/setup.bash

WS=/home/code/core_ws
# build/install go to /opt paths backed by named docker volumes on the VM's
# native ext4: writing colcon's --symlink-install tree onto the virtiofs
# bind-mount stalls for minutes. Named volumes are fast AND persistent, so the
# 9-min C++ package (h12_ros2_model) incrementally caches across runs.
BUILD_BASE=/opt/core_ws/build
INSTALL_BASE=/opt/core_ws/install
cd "$WS"

# Only the packages the minimal Mac bringup needs. h1_bringup itself is NOT
# built: it declares exec-deps on the heavy fast_lio / model_server /
# livox_ros_driver2 packages (which need the ML stack), and the minimal launch
# file is run by absolute path so the package need not be installed. Everything
# else in core_ws (model_server, h12_skills, FAST_LIO, cl_realsense,
# magpie_control, nav2 ...) is intentionally skipped on this CPU-only image.
PKGS="custom_ros_messages magpie_msgs h12_ros2_model h12_ros2_controller h12_safety_layer"

echo "[launch_ros_mac] colcon build --packages-select $PKGS"
colcon build --symlink-install \
    --build-base "$BUILD_BASE" --install-base "$INSTALL_BASE" \
    --packages-select $PKGS

source "$INSTALL_BASE/setup.bash"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"

if [ "${1:-}" = "bash" ]; then
    echo "[launch_ros_mac] workspace built; dropping to shell (ROS_DOMAIN_ID=$ROS_DOMAIN_ID)"
    exec bash
fi

echo "[launch_ros_mac] launching minimal bringup (ROS_DOMAIN_ID=$ROS_DOMAIN_ID)"
exec ros2 launch "$WS/src/h1_bringup/launch/h1_sim_bringup_mac.launch.py"
