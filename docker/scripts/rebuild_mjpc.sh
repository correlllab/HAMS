#!/bin/bash
# Incrementally rebuild the MJPC C++ agent_server from the bind-mounted submodule
# source using the persistent (seeded) CMake build tree, then refresh the binary
# that `from mujoco_mpc import agent` auto-spawns. Run INSIDE the ros container:
#
#   docker exec -it hams_ros /home/code/h12_sim_scripts/rebuild_mjpc.sh            # C++ edit
#   docker exec -it hams_ros /home/code/h12_sim_scripts/rebuild_mjpc.sh --install  # + assets/proto/libmujoco
#
# The default path is fast (seconds for a single-file edit) because the build tree
# was seeded warm at image build and hydrated by launch_ros.sh. --install is the
# heavier path for when task XML/mesh assets or the python API changed.
set -euo pipefail

MJPC_SRC=/home/code/mujoco_mpc
MJPC_BUILD=/home/code/mujoco_mpc/build      # in-tree; persisted via container_cache/mjpc_build
PKG=/usr/local/lib/python3.10/dist-packages/mujoco_mpc

[ -f "$MJPC_SRC/CMakeLists.txt" ] || {
    echo "[rebuild_mjpc] ERROR: no source at $MJPC_SRC — is the ../mujoco_mpc submodule mounted/checked out?" >&2
    exit 1
}

# 1. Configure only if the build dir is cold (normally launch_ros.sh seeded it, so
#    this is skipped). Flags/generator IDENTICAL to the RosDockerfile seed build so
#    setup.py's build_ext never triggers a reconfigure.
if [ ! -f "$MJPC_BUILD/CMakeCache.txt" ]; then
    echo "[rebuild_mjpc] cold build dir — full configure (first build is ~15 min: gRPC/MuJoCo/abseil)"
    FC_FLAG=()
    # If _deps is already populated (partial tree), forbid a network re-fetch.
    [ -d "$MJPC_BUILD/_deps" ] && FC_FLAG=(-DFETCHCONTENT_FULLY_DISCONNECTED=ON)
    cmake -S "$MJPC_SRC" -B "$MJPC_BUILD" -G Ninja \
        -DCMAKE_C_COMPILER=clang-13 -DCMAKE_CXX_COMPILER=clang++-13 \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
        -DMJPC_BUILD_GRPC_SERVICE=ON \
        -DMJPC_BUILD_TESTS=OFF -DMJPC_GRPC_BUILD_TESTS=OFF -DPYMJPC_BUILD_TESTS=OFF \
        "${FC_FLAG[@]}"
fi

# 2. Incremental build of just the C++ server (identical env to the seed build).
echo "[rebuild_mjpc] cmake --build (incremental) --target agent_server"
CC=clang-13 CXX=clang++-13 CMAKE_GENERATOR=Ninja \
    cmake --build "$MJPC_BUILD" --target agent_server -j"$(nproc)"

# Locate the freshly built binary (setup.py builds it to build/bin/agent_server).
BIN="$MJPC_BUILD/bin/agent_server"
[ -x "$BIN" ] || BIN="$(find "$MJPC_BUILD" -name agent_server -type f -perm -u+x -print -quit)"
[ -n "$BIN" ] && [ -x "$BIN" ] || { echo "[rebuild_mjpc] ERROR: agent_server binary not found after build" >&2; exit 1; }

# 3. ALWAYS refresh the binary the bridge spawns: `from mujoco_mpc import agent`
#    auto-runs <pkg>/mjpc/agent_server, so a C++ edit only takes effect once this
#    exe is copied over. (Without this, a default rebuild is a no-op for the ROS
#    bridge / h12_deploy_mjpc.)
cp -a "$BIN" "$PKG/mjpc/agent_server"
echo "[rebuild_mjpc] refreshed $PKG/mjpc/agent_server  (from $BIN)"

# 4. Optional full reinstall: refreshes task XML/mesh assets + regenerated proto
#    (needed when assets or the python API changed — NOT for a pure C++ edit).
if [ "${1:-}" = "--install" ]; then
    echo "[rebuild_mjpc] full setup.py install (assets + proto)"
    ( cd "$MJPC_SRC/python" \
      && CC=clang-13 CXX=clang++-13 CMAKE_GENERATOR=Ninja \
         python setup.py install --single-version-externally-managed \
                                 --record /tmp/mjpc-install-files.txt \
      && rm -f /tmp/mjpc-install-files.txt )
    # Re-stage the build's OWN libmujoco (ABI must match the C++ server, esp. after
    # a MuJoCo-tag bump) — mirrors RosDockerfile step 4.
    cp -a "$MJPC_BUILD"/lib/libmujoco.so* /usr/local/lib/ && ldconfig
    # Only re-clamp setuptools if something actually bumped it off 59.x (a legacy
    # setup.py install normally does not), to avoid a spurious network fetch.
    if ! python -c "import setuptools,sys; sys.exit(0 if setuptools.__version__.startswith('59.') else 1)"; then
        echo "[rebuild_mjpc] re-clamping setuptools/wheel to the colcon-safe pins"
        pip install --no-cache-dir --no-deps "numpy<2" "setuptools==59.6.0" "wheel<0.44"
        rm -f /usr/local/lib/python3.10/dist-packages/distutils-precedence.pth
    fi
    echo "[rebuild_mjpc] dist-packages mujoco_mpc fully refreshed"
fi

echo "[rebuild_mjpc] done — restart your MJPC node so it re-spawns the updated agent_server"
