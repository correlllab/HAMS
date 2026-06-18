#!/bin/bash
# Usage:
#   ./docker_run.sh [isaac|mujoco|ros]                -> auto-starts the simulation/workspace
#   ./docker_run.sh [isaac|mujoco|ros] bash           -> drops to a shell instead
#   ./docker_run.sh [isaac|mujoco|ros] <exec> <args>  -> runs any command inside the container
#
# To change the Isaac task or MuJoCo flags, either:
#   - Drop to bash and invoke launch_isaac.sh / launch_mujoco.sh with your args, or
#   - Invoke the launcher directly: ./docker_run.sh isaac /home/code/h12_sim_scripts/launch_isaac.sh TASKNAME

SIM=${1:?Usage: ./docker_run.sh [isaac|mujoco|ros] [override command/args...]}
shift
cd "$(dirname "$0")/../.."

# Load local secrets / overrides (GEMINI_API_KEY, ROS_DOMAIN_ID, ...). compose
# also auto-loads docker/.env, but sourcing here makes the values available to
# this script too (e.g. the ROS_DOMAIN_ID normalization below). set -a exports
# them so the values reach `docker compose` as host-env vars. Note: this
# overwrites any same-named vars already exported in your shell.
if [ -f docker/.env ]; then
    set -a
    source docker/.env
    set +a
fi

# Pre-create host-side bind sources so dockerd doesn't materialise them
# root-owned on first run. The cache holds msgs_ws build/install/log between
# container restarts (see docker-compose.yml).
mkdir -p container_cache/msgs_ws

# ROS_DOMAIN_ID handling. Domain 0 is the real robot's DDS command bus.
#   - Sims (isaac/mujoco) must never run on it: reject an explicit 0 loudly.
#   - The ros profile may use it (real robot), but only after confirming the
#     interactive prompt below.
#   - An unset/empty value defaults to 1, the simulation domain.
if [ "${ROS_DOMAIN_ID:-}" = "0" ]; then
    if [ "$SIM" != "ros" ]; then
        echo "ERROR: ROS_DOMAIN_ID=0 is the real robot's DDS domain; the '$SIM' sim may not run on it." >&2
        echo "       Set ROS_DOMAIN_ID to a positive value (e.g. 1) in docker/.env, or unset it to default to 1." >&2
        exit 1
    fi
    echo "WARNING: ROS_DOMAIN_ID=0 -> DDS domain 0 is the REAL ROBOT command bus." >&2
    echo "         Nodes will publish/subscribe on the live robot." >&2
    read -r -p "Proceed on DDS domain 0 (real robot)? [y/N] " reply
    case "$reply" in
        y|Y|yes|YES|Yes) ;;
        *) echo "Aborted: refused to run on DDS domain 0." >&2; exit 1 ;;
    esac
fi
if [ -z "${ROS_DOMAIN_ID:-}" ]; then
    export ROS_DOMAIN_ID=1
fi

xhost +local:docker 2>/dev/null || true

# If the first arg is a flag (e.g. --viewer), forward it to the default launcher
# instead of letting docker treat it as the command name.
if [ $# -gt 0 ] && [ "${1#-}" != "$1" ]; then
    set -- "/home/code/h12_sim_scripts/launch_${SIM}.sh" "$@"
fi

# Stable container name per sim profile so `docker exec`/`docker logs` work
# without copy-pasting a generated UUID. --rm cleans it up on exit; if a
# previous run was killed without cleanup, force-remove the stale name first.
NAME="humanoid_sim_${SIM}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker compose -f docker/docker-compose.yml --profile "$SIM" run --rm --remove-orphans --name "$NAME" "$SIM" "$@"
