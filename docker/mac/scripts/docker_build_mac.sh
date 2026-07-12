#!/usr/bin/env bash
# Build the Apple-Silicon (arm64 / CPU-only) Docker images.
# The Mac counterpart of docker/scripts/docker_build.sh.
#
# Usage: docker_build_mac.sh [service ...]
#   Services: robocasa, ros
#   With no args, both are built.
#   Examples:
#     docker_build_mac.sh                  # base + robocasa + ros
#     docker_build_mac.sh robocasa         # base + robocasa
#     docker_build_mac.sh ros              # base + ros
#
# Both arm64 images are `FROM hams_base:latest`, so the base is always built
# first -- from docker/mac/BaseDockerfile.arm64 (ubuntu:22.04, CPU-only), NOT the
# x86 docker/BaseDockerfile (nvidia/cuda, won't build without the NVIDIA runtime).
#
# There is no `isaac` here: Isaac Sim needs an NVIDIA GPU, so the arm64 port omits
# it. And unlike the x86 ros build there is no MJPC_REF -- the slim arm64 ros image
# ships no MuJoCo-MPC, so the mujoco_mpc submodule pin is irrelevant to this build.
set -euo pipefail
cd "$(dirname "$0")/../../.."

COMPOSE=docker/mac/docker-compose.yml
VALID_SERVICES=(robocasa ros)

if [ "$#" -eq 0 ]; then
    SERVICES=("${VALID_SERVICES[@]}")
else
    SERVICES=("$@")
    for s in "${SERVICES[@]}"; do
        ok=0
        for v in "${VALID_SERVICES[@]}"; do
            if [ "$s" = "$v" ]; then ok=1; break; fi
        done
        if [ "$ok" -ne 1 ]; then
            echo "error: unknown service '$s' (valid: ${VALID_SERVICES[*]})" >&2
            echo "       note: isaac is NVIDIA-only and has no arm64 build." >&2
            exit 2
        fi
    done
fi

# hams_base for arm64. Context is the repo root (matches the compose build
# contexts, which are ../.. relative to docker/mac/).
echo "Building base (docker/mac/BaseDockerfile.arm64)..."
docker build -t hams_base:latest -f docker/mac/BaseDockerfile.arm64 .

echo "Building services: ${SERVICES[*]}"
docker compose -f "$COMPOSE" build "${SERVICES[@]}"
