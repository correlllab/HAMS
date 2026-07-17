#!/bin/bash
# Thin in-container MJPC (re)build for the hams_ros container.
#
#   docker exec -it hams_ros /home/code/h12_sim_scripts/rebuild_mjpc.sh [target ...]
#
# Source = /home/code/mujoco_mpc        (bind-mounted submodule, host-editable)
# Build  = /home/code/mujoco_mpc/build  (bind-mounted from container_cache/
#                                        mjpc_build; persists on the host)
#
# Configuration mirrors the only Linux config upstream CI tests
# (ubuntu-22.04 + clang-13 + Ninja + Release + gRPC service ON). Upstream's
# default IPO/LTO stays on, so the static libs are clang bitcode — consumers
# (h12_deploy_mjpc) must compile/link with clang-13 + ld.lld-13 too.
#
# Configure runs only when the build tree is cold; warm rebuilds are a single
# ninja invocation (ninja re-runs cmake itself when CMakeLists change) — edit a
# task .cc and rerun this in seconds. Task XML/mesh edits are re-staged into
# build/mjpc/tasks by the same invocation (copy_resources always runs).
#
# Default targets: the GUI (mjpc), the headless eval runner (testspeed), and
# the gRPC server (agent_server — kept building so switching h12_deploy_mjpc to
# the python/gRPC client later stays cheap). Pass explicit targets to override.
# A cold build (empty container_cache/mjpc_build) fetches + compiles MuJoCo,
# abseil and gRPC: expect ~15-25 min, once per host.
set -euo pipefail

SRC=/home/code/mujoco_mpc
BUILD=$SRC/build

if [ ! -f "$SRC/CMakeLists.txt" ]; then
    echo "[rebuild_mjpc] $SRC is empty — is the mujoco_mpc submodule mounted (and initialized on the host)?" >&2
    exit 1
fi

# Configure only when the tree has never SUCCESSFULLY configured (build.ninja
# is written at the end of a good configure — keying on it, not on
# CMakeCache.txt, means an interrupted cold configure just reconfigures on the
# next run instead of wedging the cache). Warm trees skip straight to ninja,
# which re-runs cmake itself when CMakeLists change.
#
# Legacy note: a cache tree inherited from the old seed machinery has
# FETCHCONTENT_FULLY_DISCONNECTED=ON baked in (its _deps/.git dirs were
# trimmed); that still works, but dep/tag bumps in mjpc's CMakeLists then
# require wiping container_cache/mjpc_build. Freshly configured trees keep
# FetchContent online and handle dep bumps normally.
if [ ! -f "$BUILD/build.ninja" ]; then
    echo "[rebuild_mjpc] cold build tree — configuring (first build fetches MuJoCo/abseil/gRPC: ~15-25 min)"
    cmake -S "$SRC" -B "$BUILD" -G Ninja \
        -DCMAKE_C_COMPILER=clang-13 -DCMAKE_CXX_COMPILER=clang++-13 \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
        -DMJPC_BUILD_GRPC_SERVICE:BOOL=ON \
        -DMJPC_BUILD_TESTS:BOOL=OFF -DMJPC_GRPC_BUILD_TESTS:BOOL=OFF \
        -DPYMJPC_BUILD_TESTS:BOOL=OFF
fi

if [ $# -gt 0 ]; then TARGETS=("$@"); else TARGETS=(mjpc agent_server testspeed); fi
exec cmake --build "$BUILD" --target "${TARGETS[@]}"
