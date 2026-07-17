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
# RoboCasa and Ros inherit from hams_base; the base image is built first
# whenever either of those profiles is selected. Isaac is self-contained
# (Sim 5.1 / Lab v2.3.2 / Python 3.11) and does not use hams_base.
set -euo pipefail
cd "$(dirname "$0")/../.."

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
    docker build -t hams_base:latest -f docker/BaseDockerfile .
fi

PROFILE_ARGS=()
for p in "${PROFILES[@]}"; do
    PROFILE_ARGS+=(--profile "$p")
done

echo "Building profiles: ${PROFILES[*]}"
docker compose -f docker/docker-compose.yml "${PROFILE_ARGS[@]}" build
