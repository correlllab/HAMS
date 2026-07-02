#!/bin/bash
# Build (if needed) and source the core_ws workspace, then drop to bash.
# core_ws is bind-mounted from the host; build/install/log persist there too.
set -e

source /opt/ros/humble/setup.bash

# --- MuJoCo MPC (MJPC) build-cache hydrate (image seed -> persistent mount) ---
# ../container_cache/mjpc_build persists the CMake build tree at the in-tree path
# /home/code/mujoco_mpc/build. On first launch it is empty, so hydrate it from the
# baked seed and back-date the freshly-checked-out submodule source so the first
# in-container rebuild_mjpc.sh is warm (incremental) instead of a ~15-min cold
# rebuild. Guarded on the absence of CMakeCache.txt -> runs exactly once; later
# `run --rm` containers reuse the already-hydrated host dir. No-ops cleanly if the
# submodule isn't checked out (import mujoco_mpc still works from dist-packages).
MJPC_SRC=/home/code/mujoco_mpc
MJPC_BUILD=/home/code/mujoco_mpc/build
MJPC_SEED=/opt/mjpc-build-seed
if [ -f "$MJPC_SRC/CMakeLists.txt" ] && [ ! -e "$MJPC_BUILD/CMakeCache.txt" ] \
   && [ -d "$MJPC_SEED" ]; then
    echo "[launch_ros] hydrating MJPC build cache from seed ($MJPC_SEED -> $MJPC_BUILD)"
    mkdir -p "$MJPC_BUILD"
    cp -a "$MJPC_SEED/." "$MJPC_BUILD/"    # preserve mtimes/symlinks/_deps stamps
    # The host submodule was checked out AFTER the image build, so its source
    # mtimes are NEWER than the seeded objects -> Ninja would recompile everything
    # and CMake would reconfigure. Push SOURCE mtimes into the past (NOT the
    # multi-GB build tree, whose internal mtime ordering must stay intact). Prune
    # the build dir and .git. Fork source == seed source (same patched SHA), so no
    # patching is needed here.
    echo "[launch_ros] back-dating MJPC source mtimes so the seed stays warm"
    find "$MJPC_SRC" \( -path "$MJPC_BUILD" -o -name .git \) -prune -o \
         -exec touch -h -d '2000-01-01T00:00:00' {} +
fi

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
