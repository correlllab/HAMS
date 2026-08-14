#!/usr/bin/env bash
# Build Docker images.
#
# Usage: docker_build.sh [profile ...]
#   Profiles: isaac, robocasa, ros
#   With no args, all three are built.
#   Examples:
#     docker_build.sh                  # build all
#     docker_build.sh isaac            # build isaac only
#     docker_build.sh robocasa ros     # build robocasa + ros
#
# RoboCasa and Ros inherit from golem_base; the base image is built first
# whenever either of those profiles is selected. Isaac is self-contained
# (Sim 5.1 / Lab v2.3.2 / Python 3.11) and does not use golem_base.
set -euo pipefail
cd "$(dirname "$0")/../.."

# Lock the ros image's MuJoCo-MPC build to the mujoco_mpc submodule's committed
# SHA. RosDockerfile clones mujoco_mpc fresh over HTTPS at build time (no SSH
# keys / no git context inside the build), so it can't read the gitlink itself
# -- we read it here on the host and pass it in as the MJPC_REF build arg (see
# docker-compose.yml ros.build.args). Without this a `git submodule` bump would
# silently ship a stale MJPC. Empty (not a git checkout) -> Dockerfile default.
MJPC_REF="$(git rev-parse HEAD:mujoco_mpc 2>/dev/null || true)"
if [ -n "${MJPC_REF}" ]; then
    export MJPC_REF
    echo "MJPC_REF (from mujoco_mpc submodule pin): ${MJPC_REF}"
fi

VALID_PROFILES=(isaac robocasa ros)

if [ "$#" -eq 0 ]; then
    PROFILES=("${VALID_PROFILES[@]}")
else
    PROFILES=("$@")
    for p in "${PROFILES[@]}"; do
        ok=0
        for v in "${VALID_PROFILES[@]}"; do
            if [ "$p" = "$v" ]; then ok=1; break; fi
        done
        if [ "$ok" -ne 1 ]; then
            echo "error: unknown profile '$p' (valid: ${VALID_PROFILES[*]})" >&2
            exit 2
        fi
    done
fi

needs_base=0
for p in "${PROFILES[@]}"; do
    if [ "$p" = "robocasa" ] || [ "$p" = "ros" ]; then
        needs_base=1
        break
    fi
done

if [ "$needs_base" -eq 1 ]; then
    echo "Building base..."
    docker build -t golem_base:latest -f docker/BaseDockerfile .
fi

PROFILE_ARGS=()
for p in "${PROFILES[@]}"; do
    PROFILE_ARGS+=(--profile "$p")
done

echo "Building profiles: ${PROFILES[*]}"
docker compose -f docker/docker-compose.yml "${PROFILE_ARGS[@]}" build
