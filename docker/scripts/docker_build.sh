#!/usr/bin/env bash
# Build Docker images. The base image must be built first (Isaac/MuJoCo inherit from it).
# Uncomment/comment profiles below to select which images to build.
set -euo pipefail
cd "$(dirname "$0")/../.."

# Profiles to build. Add/remove as needed: isaac, mujoco, ros
# PROFILES=(isaac)
PROFILES=(isaac mujoco ros)   # ← uncomment to build all

echo "Building base..."
docker build -t humanoid_sim_base:latest -f docker/BaseDockerfile .

PROFILE_ARGS=()
for p in "${PROFILES[@]}"; do
    PROFILE_ARGS+=(--profile "$p")
done

echo "Building profiles: ${PROFILES[*]}"
docker compose -f docker/docker-compose.yml "${PROFILE_ARGS[@]}" build
