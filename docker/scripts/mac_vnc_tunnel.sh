#!/bin/bash
# Mac-side helper: SSH-tunnel the in-container noVNC/VNC ports to the Mac's
# localhost. Colima does NOT forward host-network container ports to the Mac
# (and this Colima has no reachable VM IP), so a tunnel over Colima's own SSH is
# the way in. Pairs with HAMS_DISPLAY=vnc (MuJoCo viewer) and HAMS_RVIZ=vnc (RViz).
#
# Forwards both viewers' ports (harmless if only one is running):
#   MuJoCo viewer : http://localhost:6080/vnc.html   (VNC localhost:5900)
#   RViz          : http://localhost:6081/vnc.html   (VNC localhost:5901)
#
# Usage:
#   ./mac_vnc_tunnel.sh          open the tunnel, print the URLs
#   ./mac_vnc_tunnel.sh --open   ... and open both noVNC pages in the browser
#   ./mac_vnc_tunnel.sh --stop   close the tunnel
#
# Env: COLIMA_PROFILE (default: default).
set -euo pipefail

PROFILE="${COLIMA_PROFILE:-default}"
# Ports to forward: noVNC (browser) + raw VNC, for the MuJoCo viewer and RViz.
PORTS=(6080 5900 6081 5901)
MUJOCO_URL="http://localhost:6080/vnc.html?autoconnect=1&resize=scale"
RVIZ_URL="http://localhost:6081/vnc.html?autoconnect=1&resize=scale"

# Kill whatever ssh is holding any of our forwarded ports. We match by listening
# port (via lsof), NOT by command line: `ssh -f` daemonizes and truncates its
# argv, so pgrep/pkill on "-L <port>" would miss it.
stop_tunnel() {
    local killed=0 pid port
    for port in "${PORTS[@]}"; do
        for pid in $(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null); do
            if ps -p "$pid" -o comm= 2>/dev/null | grep -q '^ssh'; then
                kill "$pid" 2>/dev/null && killed=1
            fi
        done
    done
    [ "$killed" = 1 ] && echo "tunnel closed" || echo "no tunnel running"
}

case "${1:-}" in
    --stop) stop_tunnel; exit 0 ;;
esac

command -v colima >/dev/null 2>&1 || { echo "ERROR: colima not found on PATH"; exit 1; }

# Ask colima for the SSH config to its VM and extract the Host alias from it.
CFG="$(mktemp -t colima-ssh.XXXXXX)"
if ! colima ssh-config --profile "$PROFILE" >"$CFG" 2>/dev/null; then
    echo "ERROR: 'colima ssh-config' failed — is colima running? (colima start)"; rm -f "$CFG"; exit 1
fi
HOST_ALIAS="$(awk '/^Host /{print $2; exit}' "$CFG")"
[ -n "$HOST_ALIAS" ] || { echo "ERROR: could not read Host alias from colima ssh-config"; rm -f "$CFG"; exit 1; }

# Replace any existing tunnel on these ports (else ExitOnForwardFailure trips).
stop_tunnel >/dev/null
sleep 1

# Build -L args for every port (forwarding a port whose remote side isn't
# listening yet is fine — only a LOCAL bind clash would fail, and we just cleared those).
FWD_ARGS=()
for port in "${PORTS[@]}"; do
    FWD_ARGS+=(-L "${port}:127.0.0.1:${port}")
done

ssh -F "$CFG" -f -N -o ExitOnForwardFailure=yes "${FWD_ARGS[@]}" "$HOST_ALIAS"

echo "Tunnel up (Mac -> Colima VM):"
echo "  MuJoCo viewer : $MUJOCO_URL"
echo "  RViz          : $RVIZ_URL"
echo "  (raw VNC: localhost:5900 = MuJoCo, localhost:5901 = RViz)"
echo "Close it with: $0 --stop"

if [ "${1:-}" = "--open" ]; then
    open "$MUJOCO_URL"
    open "$RVIZ_URL"
fi
