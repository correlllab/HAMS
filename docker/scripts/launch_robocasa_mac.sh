#!/bin/bash
# Apple-Silicon / CPU launcher for the MuJoCo (RoboCasa) sim.
# Same as launch_robocasa.sh but CPU/software rendering, no NVIDIA EGL.
#
# Two display modes, selected by the HAMS_DISPLAY env var (default: headless):
#   HAMS_DISPLAY=headless  OSMesa offscreen, no viewer window (original behaviour)
#   HAMS_DISPLAY=vnc       render the interactive MuJoCo viewer with software GL
#                          (llvmpipe) into an in-container Xvfb and stream it out
#                          over VNC/noVNC. MuJoCo's GL window cannot be forwarded
#                          to macOS XQuartz on Apple Silicon (broken indirect
#                          GLX), so we ship pixels instead. View from the Mac by
#                          running docker/scripts/mac_vnc_tunnel.sh, then opening
#                          http://localhost:6080/vnc.html.
#
# Extra args pass through to h12_mujoco.py.
set -e

HAMS_DISPLAY="${HAMS_DISPLAY:-headless}"

# VNC/noVNC ports (only used when HAMS_DISPLAY=vnc). Bound to localhost inside the
# Colima VM; reachable from the Mac only through the SSH tunnel (Colima does not
# forward host-network container ports, so a tunnel is required — see
# mac_vnc_tunnel.sh).
VNC_DISPLAY=:99
VNC_PORT=5900
NOVNC_PORT=6080
VNC_GEOMETRY="${VNC_GEOMETRY:-1280x800x24}"

# Bring up Xvfb + fluxbox + x11vnc + noVNC and point DISPLAY at the virtual
# framebuffer. All children are killed on exit via the trap below.
start_vnc_stack() {
    echo "[launch_robocasa_mac] starting VNC stack on $VNC_DISPLAY ($VNC_GEOMETRY)"
    Xvfb "$VNC_DISPLAY" -screen 0 "$VNC_GEOMETRY" +extension GLX +render -noreset \
        >/tmp/xvfb.log 2>&1 &
    export DISPLAY="$VNC_DISPLAY"
    # wait for the X server to accept connections
    for _ in $(seq 1 30); do
        xdpyinfo -display "$VNC_DISPLAY" >/dev/null 2>&1 && break
        sleep 0.5
    done
    xdpyinfo -display "$VNC_DISPLAY" >/dev/null 2>&1 \
        || { echo "[launch_robocasa_mac] Xvfb failed to start"; cat /tmp/xvfb.log; exit 1; }

    fluxbox >/tmp/fluxbox.log 2>&1 &   # bg set by the fbsetbg shim (Dockerfile)
    sleep 1

    # -localhost: only accept VNC connections from within the VM (the SSH tunnel
    # terminates on the VM's loopback). -nopw is safe because the port is not
    # otherwise reachable from the Mac.
    x11vnc -display "$VNC_DISPLAY" -rfbport "$VNC_PORT" -localhost \
        -forever -shared -nopw -quiet -bg >/tmp/x11vnc.log 2>&1
    sleep 1

    websockify --web /usr/share/novnc "127.0.0.1:$NOVNC_PORT" "localhost:$VNC_PORT" \
        >/tmp/websockify.log 2>&1 &
    sleep 1

    echo "[launch_robocasa_mac] noVNC ready. From the Mac run mac_vnc_tunnel.sh, then open:"
    echo "[launch_robocasa_mac]   http://localhost:${NOVNC_PORT}/vnc.html?autoconnect=1&resize=scale"
}

cleanup_vnc_stack() {
    pkill -f "websockify .*${NOVNC_PORT}" 2>/dev/null || true
    pkill -x x11vnc 2>/dev/null || true
    pkill -x fluxbox 2>/dev/null || true
    pkill -f "Xvfb ${VNC_DISPLAY}" 2>/dev/null || true
}

source /opt/ros/humble/setup.bash
# livox_ros_driver2 (CustomMsg/CustomPoint) is baked into the robocasa image at
# /opt/livox_ws so mujoco_ros_bridge.py can import it.
source /opt/livox_ws/install/setup.bash

# Build the IDL-only msgs workspace (fast no-op when unchanged).
# build/install go to /opt paths backed by named docker volumes on the VM's
# native ext4: writing colcon's --symlink-install tree onto the virtiofs
# bind-mount stalls for minutes (hundreds of symlinks at ~0% CPU). Named volumes
# are both fast and persistent, so this is a real no-op on unchanged rebuilds.
MSGS_WS=/home/code/msgs_ws
MSGS_BUILD=/opt/msgs_ws/build
MSGS_INSTALL=/opt/msgs_ws/install
echo "[launch_robocasa_mac] building $MSGS_WS -> $MSGS_INSTALL"
(cd "$MSGS_WS" && colcon build --symlink-install \
    --build-base "$MSGS_BUILD" --install-base "$MSGS_INSTALL" \
    --packages-select magpie_msgs custom_ros_messages)
source "$MSGS_INSTALL/setup.bash"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"

if [ "$HAMS_DISPLAY" = "vnc" ]; then
    # Interactive viewer via software GL into Xvfb, streamed over VNC/noVNC.
    trap cleanup_vnc_stack EXIT
    start_vnc_stack
    export LIBGL_ALWAYS_SOFTWARE=1   # force llvmpipe; there is no GPU under Colima
    export MUJOCO_GL=glfw            # windowed viewer (renders onto Xvfb :99)
    # Do NOT inject --headless here: the whole point of this mode is the window.
else
    # Pure-software offscreen rendering — the only backend available without a GPU.
    export MUJOCO_GL=osmesa
    # Force headless: there is no reachable X display inside the Colima VM.
    case " $* " in
        *" --headless "*) ;;
        *) set -- "$@" --headless ;;
    esac
fi

cd /home/code/h1_robocasa
echo "[launch_robocasa_mac] HAMS_DISPLAY=$HAMS_DISPLAY MUJOCO_GL=$MUJOCO_GL ROS_DOMAIN_ID=$ROS_DOMAIN_ID args: $*"
python h12_mujoco.py "$@"
