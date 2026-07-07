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
    # the build dir and .git.
    echo "[launch_ros] back-dating MJPC source mtimes so the seed stays warm"
    find "$MJPC_SRC" \( -path "$MJPC_BUILD" -o -name .git \) -prune -o \
         -exec touch -h -d '2000-01-01T00:00:00' {} +
    # The blanket back-date assumes source SHA == seed SHA. When the submodule is
    # AHEAD of the image's MJPC_REF (normal between image rebakes), that would
    # hydrate a silently STALE build (Ninja sees everything up to date). Fix: re-touch
    # exactly the files that differ from the seed ref so Ninja recompiles just the
    # delta. Seed ref comes from the baked .mjpc_ref stamp (newer images) or the
    # fallback constant (= the MJPC_REF the current image was built with). Needs the
    # ../.git/modules/mujoco_mpc ro mount for git to work on the submodule source.
    SEED_REF=$(cat "$MJPC_SEED/.mjpc_ref" 2>/dev/null \
               || echo 9f3cb6488aa82efc67893261cee6a1af560e4bc1)
    git config --global --get-all safe.directory 2>/dev/null | grep -qx "$MJPC_SRC" \
        || git config --global --add safe.directory "$MJPC_SRC"
    if HEAD_REF=$(git -C "$MJPC_SRC" rev-parse HEAD 2>/dev/null); then
        if [ "$HEAD_REF" != "$SEED_REF" ]; then
            echo "[launch_ros] source ($HEAD_REF) != seed ($SEED_REF) — re-touching the changed files"
            git -C "$MJPC_SRC" diff --name-only "$SEED_REF" HEAD 2>/dev/null \
                | (cd "$MJPC_SRC" && xargs -r -d '\n' touch -c -h) \
                || { echo "[launch_ros] WARNING: seed-delta diff failed — re-touching ALL sources (cold-ish rebuild, but correct)"; \
                     find "$MJPC_SRC" \( -path "$MJPC_BUILD" -o -name .git \) -prune -o -exec touch -h {} + ; }
        fi
    else
        echo "[launch_ros] WARNING: git unusable on $MJPC_SRC (missing .git/modules mount?) — re-touching ALL sources (cold-ish rebuild, but correct)"
        find "$MJPC_SRC" \( -path "$MJPC_BUILD" -o -name .git \) -prune -o -exec touch -h {} +
    fi
fi

# --- unitree_sdk2 C++ install hydrate (for the h12_deploy_mjpc controller shim) ---
# deploy_common.cc (compiled by colcon from the submodule) links unitree_sdk2's
# C++ ChannelFactory/Publisher/Subscriber. The repo ships PREBUILT
# libunitree_sdk2.a + CycloneDDS .so per arch, so this is a clone + header/lib
# copy — seconds, no compilation. /opt/unitree_install is a persistent host
# mount (container_cache/unitree_install), so this runs exactly once per host.
# SHA pinned to the host's ~/unitree_sdk2 build (63c6f53) so fork and shim link
# the SAME SDK.
UNITREE_PREFIX=/opt/unitree_install
UNITREE_SDK_REF=63c6f53103e2f28e23b807d7399e92338c9d07d3
if [ ! -f "$UNITREE_PREFIX/lib/libunitree_sdk2.a" ]; then
    echo "[launch_ros] hydrating unitree_sdk2 C++ install -> $UNITREE_PREFIX (@$UNITREE_SDK_REF)"
    rm -rf /tmp/unitree_sdk2
    if git clone https://github.com/unitreerobotics/unitree_sdk2.git /tmp/unitree_sdk2 \
       && git -C /tmp/unitree_sdk2 checkout --quiet "$UNITREE_SDK_REF" \
       && cmake -S /tmp/unitree_sdk2 -B /tmp/unitree_sdk2/build \
                -DCMAKE_INSTALL_PREFIX="$UNITREE_PREFIX" >/dev/null \
       && cmake --install /tmp/unitree_sdk2/build >/dev/null; then
        echo "[launch_ros] unitree_sdk2 installed"
    else
        echo "[launch_ros] WARNING: unitree_sdk2 hydrate FAILED — h12_deploy_mjpc's controller will be skipped by its CMake guard"
    fi
    rm -rf /tmp/unitree_sdk2
fi

# --- MJPC incremental rebuild (ninja no-op scan when already warm) ---
# Brings libmjpc/threadpool + the staged task assets + the dist-packages
# agent_server current with the mounted submodule BEFORE colcon links
# h12_deploy_mjpc against the build tree (after a hydrate the tree is at the
# image's MJPC_REF; the delta-touch above marked exactly what must recompile).
if [ -f "$MJPC_SRC/CMakeLists.txt" ] && [ -e "$MJPC_BUILD/CMakeCache.txt" ]; then
    /home/code/h12_sim_scripts/rebuild_mjpc.sh
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
