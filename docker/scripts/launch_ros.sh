#!/bin/bash
# Build (if needed) and source the core_ws workspace, then drop to bash.
# core_ws is bind-mounted from the host; build/install/log persist there too.
set -e

source /opt/ros/humble/setup.bash

WS=/home/code/core_ws
cd "$WS"

if [ ! -d src ] || [ -z "$(ls -A src 2>/dev/null)" ]; then
    echo "[launch_ros] $WS/src is empty — did you forget 'git submodule update --init --recursive'?"
fi

# Rebuild only if install/ is missing or any package.xml is newer than its install marker.
NEEDS_BUILD=0
if [ ! -d install ]; then
    NEEDS_BUILD=1
elif [ -n "$(find src -name package.xml -newer install/setup.bash 2>/dev/null | head -1)" ]; then
    NEEDS_BUILD=1
fi

if [ "$NEEDS_BUILD" = "1" ]; then
    # livox_ros_driver2 ships a build.sh that picks the ROS 2 file variants
    # (package_ROS2.xml → package.xml, launch_ROS2 → launch) and drives colcon
    # from the workspace root. Use it verbatim — reimplementing drifts.
    LIVOX_DIR="$WS/src/livox_ros_driver2"
    if [ -x "$LIVOX_DIR/build.sh" ]; then
        echo "[launch_ros] livox_ros_driver2/build.sh humble"
        (cd "$LIVOX_DIR" && ./build.sh humble)
    else
        echo "[launch_ros] colcon build"
        colcon build
    fi
else
    echo "[launch_ros] install/ is up to date — skipping build (run 'colcon build' to force)"
fi

# vision_pipeline's setup.py doesn't declare ModelWeights as package_data, so
# colcon doesn't copy the .pt files into install/. We can't patch the upstream
# submodule, so copy them after the build.
VP_SRC="$WS/src/vision_pipeline/vision_pipeline/core/ModelWeights"
VP_DST="$WS/install/vision_pipeline/lib/python3.10/site-packages/vision_pipeline/core/ModelWeights"
if [ -d "$VP_SRC" ] && [ -d "$WS/install/vision_pipeline" ]; then
    mkdir -p "$VP_DST"
    cp -u "$VP_SRC"/*.pt "$VP_DST/" 2>/dev/null || true
fi

source install/setup.bash 2>/dev/null || true

exec bash
