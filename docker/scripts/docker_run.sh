#!/bin/bash
# Usage:
#   ./docker_run.sh [isaac|mujoco|ros]                -> auto-starts the simulation/workspace
#   ./docker_run.sh [isaac|mujoco|ros] bash           -> drops to a shell instead
#   ./docker_run.sh [isaac|mujoco|ros] <exec> <args>  -> runs any command inside the container
#
# To change the Isaac task or MuJoCo flags, either:
#   - Drop to bash and invoke launch_isaac.sh / launch_mujoco.sh with your args, or
#   - Invoke the launcher directly: ./docker_run.sh isaac /home/code/h12_sim_scripts/launch_isaac.sh TASKNAME

SIM=${1:?Usage: ./docker_run.sh [isaac|mujoco] [override command/args...]}
shift
cd "$(dirname "$0")/../.."

# Host preflight: warn rather than fail, so CI and headless nodes still work.
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[docker_run] warning: nvidia-smi not found on host — GPU passthrough will fail" >&2
elif ! nvidia-smi >/dev/null 2>&1; then
    echo "[docker_run] warning: nvidia-smi present but returned error — check driver" >&2
fi
if ! docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia'; then
    echo "[docker_run] warning: nvidia container runtime not registered with Docker" >&2
fi

# Ensure $XAUTHORITY points at an existing file so the bind mount doesn't fail
# on hosts where the env var is unset or the file is absent (headless, CI, SSH).
if [ -z "${XAUTHORITY:-}" ] || [ ! -f "${XAUTHORITY:-}" ]; then
    if [ -f "$HOME/.Xauthority" ]; then
        export XAUTHORITY="$HOME/.Xauthority"
    else
        XAUTH_STUB=$(mktemp -t humanoid_sim_xauth.XXXXXX)
        export XAUTHORITY="$XAUTH_STUB"
        trap 'rm -f "$XAUTH_STUB"' EXIT
    fi
fi

xhost +local:docker 2>/dev/null || true

# If the first arg is a flag (e.g. --viewer), forward it to the default launcher
# instead of letting docker treat it as the command name.
if [ $# -gt 0 ] && [ "${1#-}" != "$1" ]; then
    set -- "/home/code/h12_sim_scripts/launch_${SIM}.sh" "$@"
fi

docker compose -f docker/docker-compose.yml --profile "$SIM" run --rm --remove-orphans "$SIM" "$@"
