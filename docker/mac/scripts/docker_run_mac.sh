#!/bin/bash
# Apple-Silicon (arm64 / CPU-only) run helper -- the Mac counterpart of
# docker/scripts/docker_run.sh.
#
# Usage:
#   ./docker_run_mac.sh [robocasa|ros]                 -> start the sim / workspace (compose default command)
#   ./docker_run_mac.sh [robocasa|ros] bash            -> drop to a raw shell instead (no auto colcon build)
#   ./docker_run_mac.sh [robocasa|ros] <exec> <args>   -> run any command inside the container
#
# Display/feature toggles are env vars read by docker/mac/docker-compose.yml --
# export them ahead of the service name and compose picks them up:
#   GOLEM_DISPLAY=vnc ./docker_run_mac.sh robocasa   # MuJoCo viewer -> noVNC :6080
#   GOLEM_RVIZ=vnc    ./docker_run_mac.sh ros        # RViz         -> noVNC :6081
# then open docker/mac/scripts/mac_vnc_tunnel.sh from the Mac (Colima does not
# forward host-network container ports; a tunnel is required).
#
# Differences from the x86 helper: no isaac (NVIDIA-only), no MJPC cache, and no
# `xhost` (macOS has no host X server -- GUIs stream over VNC, not XQuartz).
#
# NOTE: this starts ONE service. For the paired sim you still need both robocasa
# and ros up -- run this in two terminals and start `ros` promptly, or use
# `docker compose -f docker/mac/docker-compose.yml up` to start both at once.
# RoboCasa left running alone for more than a few seconds releases the robot's
# motors ("Command timeout") and the H1 collapses (see README, macOS section).

SIM=${1:?Usage: ./docker_run_mac.sh [robocasa|ros] [override command/args...]}
shift
cd "$(dirname "$0")/../../.."

# HAMS -> GOLEM rename guard. The toggles above all default to off, so a stale
# HAMS_* export would be silently dropped by compose interpolation. Fail loudly.
# Note: this only fires via this script -- `docker compose ... up` bypasses it.
legacy=$(env | sed -n 's/^\(HAMS_[A-Z0-9_]*\)=.*/\1/p' | tr '\n' ' ')
if [ -n "$legacy" ]; then
    echo "ERROR: legacy HAMS_* env vars set: $legacy" >&2
    echo "       The project is now GOLEM; these were renamed to GOLEM_*." >&2
    echo "       Update your shell exports / scripts and re-run." >&2
    exit 1
fi

COMPOSE=docker/mac/docker-compose.yml

case "$SIM" in
    robocasa|ros) ;;
    *)
        echo "error: unknown service '$SIM' (valid: robocasa, ros)" >&2
        echo "       note: isaac is NVIDIA-only and has no arm64 build." >&2
        exit 2
        ;;
esac

# The Mac compose needs no docker/.env (it only reads ROS_DOMAIN_ID, defaulted in
# the compose file). Source it anyway if present so any shell overrides reach
# `docker compose` as host-env vars; set -a exports them for interpolation.
if [ -f docker/.env ]; then
    set -a
    source docker/.env
    set +a
fi

# Pre-create the one host bind source so dockerd doesn't materialise it
# root-owned on first run: the robocasa container's msgs_ws colcon cache.
# Everything else (build/install trees) lives in named volumes on the Colima VM,
# and there is no MJPC cache here (the slim arm64 ros image ships no MuJoCo-MPC).
mkdir -p container_cache/msgs_ws

# ROS_DOMAIN_ID handling. Domain 0 is the real robot's DDS command bus; the CPU
# sim must never publish onto it. Unset/empty defaults to 1 (the sim domain).
# Unlike the x86 helper there is no "confirm to run on 0" path: there is no
# real-robot workflow from a Mac, so 0 is always rejected.
if [ "${ROS_DOMAIN_ID:-}" = "0" ]; then
    echo "ERROR: ROS_DOMAIN_ID=0 is the real robot's DDS domain; the arm64 CPU sim may not run on it." >&2
    echo "       Set ROS_DOMAIN_ID to a positive value (e.g. 1) in docker/.env, or unset it to default to 1." >&2
    exit 1
fi
if [ -z "${ROS_DOMAIN_ID:-}" ]; then
    export ROS_DOMAIN_ID=1
fi

# If the first arg is a flag (e.g. --headless), forward it to the default mac
# launcher instead of letting docker treat it as the command name.
if [ $# -gt 0 ] && [ "${1#-}" != "$1" ]; then
    set -- "/home/code/h12_sim_scripts/launch_${SIM}_mac.sh" "$@"
fi

# Stable container name per service (matches container_name in the compose file)
# so `docker exec golem_ros ...` works without a generated name. --rm cleans up on
# exit; force-remove a stale one left by a previously killed run.
if [ "$SIM" = "ros" ]; then NAME="golem_ros"; else NAME="golem_sim_${SIM}"; fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker compose -f "$COMPOSE" run --rm --remove-orphans --name "$NAME" "$SIM" "$@"
