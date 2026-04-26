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
    # from the workspace root. Upstream invokes `colcon build --cmake-args ...`
    # without --symlink-install; inject the flag so vision_pipeline's
    # ModelWeights/*.pt are reachable via the install symlink instead of
    # needing a manual post-build copy. Idempotent — won't re-patch if already
    # present (e.g. after a previous run on the same bind-mounted clone).
    LIVOX_DIR="$WS/src/livox_ros_driver2"
    if [ -x "$LIVOX_DIR/build.sh" ]; then
        if ! grep -q -- '--symlink-install' "$LIVOX_DIR/build.sh"; then
            echo "[launch_ros] patching livox_ros_driver2/build.sh to add --symlink-install"
            sed -i 's|colcon build --cmake-args|colcon build --symlink-install --cmake-args|' \
                "$LIVOX_DIR/build.sh"
        fi
        echo "[launch_ros] livox_ros_driver2/build.sh humble"
        (cd "$LIVOX_DIR" && ./build.sh humble)
    else
        echo "[launch_ros] colcon build"
        colcon build --symlink-install
    fi
else
    echo "[launch_ros] install/ is up to date — skipping build (run 'colcon build --symlink-install' to force)"
fi

source install/setup.bash 2>/dev/null || true

exec bash
