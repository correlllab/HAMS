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

# Normalize ROS_DOMAIN_ID: treat 0/unset/empty as 1, otherwise pass through.
# Sims publish on domain 1; domain 0 is reserved for the real robot.
if [ -z "${ROS_DOMAIN_ID:-}" ] || [ "${ROS_DOMAIN_ID}" = "0" ]; then
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
