#!/bin/bash
# Apple-Silicon / CPU launcher for the SLIM ROS workspace.
# Builds only the core bringup package subset (skips the vision / grasp / nav2
# / FAST-LIO packages that need the heavy ML stack), then launches the minimal
# Mac bringup (robot_state_publisher, joint_state_publisher, Pink IK
# frame_task_server, safety_node) against the running MuJoCo sim.
#
# Pass `bash` as the first arg to drop to a shell instead of launching.
#
# Set HAMS_RVIZ=vnc to also run RViz2 rendered with software GL (llvmpipe) into
# an in-container Xvfb and streamed over VNC/noVNC on port 6081 (separate from
# the MuJoCo viewer's 6080). RViz's GL window can't forward to XQuartz on Apple
# Silicon, so we ship pixels. View from the Mac: docker/scripts/mac_vnc_tunnel.sh
# then open http://localhost:6081/vnc.html.
set -e

HAMS_RVIZ="${HAMS_RVIZ:-0}"

# VNC/noVNC for RViz (only used when HAMS_RVIZ=vnc). Localhost-bound in the VM;
# reachable from the Mac only via the SSH tunnel (mac_vnc_tunnel.sh). Ports are
# offset from the robocasa MuJoCo viewer's (5900/6080) so both run together.
#
# DISPLAY MUST differ from robocasa's :99. Both containers run network_mode:host,
# so they share ONE network namespace — and X11's abstract socket
# (@/tmp/.X11-unix/X<n>) plus TCP 60<nn> are namespace-scoped. Two Xvfb on :99
# would collide: one wins the abstract socket and BOTH viewers land on that single
# display (both noVNC ports then show the same mixed screen). :100 keeps RViz on
# its own X server.
RVIZ_DISPLAY=:100
RVIZ_VNC_PORT=5901
RVIZ_NOVNC_PORT=6081
RVIZ_GEOMETRY="${RVIZ_GEOMETRY:-1600x900x24}"
RVIZ_CONFIG="${RVIZ_CONFIG:-/home/code/h12_sim_scripts/h1_sim.rviz}"

start_rviz_stack() {
    echo "[launch_ros_mac] starting RViz VNC stack on $RVIZ_DISPLAY ($RVIZ_GEOMETRY)"
    Xvfb "$RVIZ_DISPLAY" -screen 0 "$RVIZ_GEOMETRY" +extension GLX +render -noreset \
        >/tmp/xvfb_rviz.log 2>&1 &
    export DISPLAY="$RVIZ_DISPLAY"
    for _ in $(seq 1 30); do
        xdpyinfo -display "$RVIZ_DISPLAY" >/dev/null 2>&1 && break
        sleep 0.5
    done
    xdpyinfo -display "$RVIZ_DISPLAY" >/dev/null 2>&1 \
        || { echo "[launch_ros_mac] Xvfb failed to start"; cat /tmp/xvfb_rviz.log; return 1; }

    fluxbox >/tmp/fluxbox_rviz.log 2>&1 &
    sleep 1
    x11vnc -display "$RVIZ_DISPLAY" -rfbport "$RVIZ_VNC_PORT" -localhost \
        -forever -shared -nopw -quiet -bg >/tmp/x11vnc_rviz.log 2>&1
    sleep 1
    websockify --web /usr/share/novnc "127.0.0.1:$RVIZ_NOVNC_PORT" "localhost:$RVIZ_VNC_PORT" \
        >/tmp/websockify_rviz.log 2>&1 &
    sleep 1

    export LIBGL_ALWAYS_SOFTWARE=1   # force llvmpipe; no GPU under Colima
    local cfg_arg=()
    [ -f "$RVIZ_CONFIG" ] && cfg_arg=(-d "$RVIZ_CONFIG")
    rviz2 "${cfg_arg[@]}" >/tmp/rviz2.log 2>&1 &
    echo "[launch_ros_mac] RViz launched. From the Mac run mac_vnc_tunnel.sh, then open:"
    echo "[launch_ros_mac]   http://localhost:${RVIZ_NOVNC_PORT}/vnc.html?autoconnect=1&resize=scale"
}

source /opt/ros/humble/setup.bash

WS=/home/code/core_ws
# build/install go to /opt paths backed by named docker volumes on the VM's
# native ext4: writing colcon's --symlink-install tree onto the virtiofs
# bind-mount stalls for minutes. Named volumes are fast AND persistent, so the
# 9-min C++ package (h12_ros2_model) incrementally caches across runs.
BUILD_BASE=/opt/core_ws/build
INSTALL_BASE=/opt/core_ws/install
cd "$WS"

# Only the packages the minimal Mac bringup needs. h1_bringup itself is NOT
# built: it declares exec-deps on the heavy fast_lio / model_server /
# livox_ros_driver2 packages (which need the ML stack), and the minimal launch
# file is run by absolute path so the package need not be installed. Everything
# else in core_ws (model_server, h12_skills, FAST_LIO, cl_realsense,
# magpie_control, nav2 ...) is intentionally skipped on this CPU-only image.
PKGS="custom_ros_messages magpie_msgs h12_ros2_model h12_ros2_controller h12_safety_layer"

echo "[launch_ros_mac] colcon build --packages-select $PKGS"
colcon build --symlink-install \
    --build-base "$BUILD_BASE" --install-base "$INSTALL_BASE" \
    --packages-select $PKGS

source "$INSTALL_BASE/setup.bash"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"

# Optionally bring up RViz (rendered to Xvfb, streamed over noVNC). Started
# before the bringup so it's up whether we launch or drop to a shell; RViz
# tolerates topics/TF arriving after it starts.
if [ "$HAMS_RVIZ" = "vnc" ] || [ "$HAMS_RVIZ" = "1" ]; then
    start_rviz_stack || echo "[launch_ros_mac] RViz stack failed to start (continuing without it)"
fi

if [ "${1:-}" = "bash" ]; then
    echo "[launch_ros_mac] workspace built; dropping to shell (ROS_DOMAIN_ID=$ROS_DOMAIN_ID)"
    exec bash
fi

echo "[launch_ros_mac] launching minimal bringup (ROS_DOMAIN_ID=$ROS_DOMAIN_ID)"
exec ros2 launch "$WS/src/h1_bringup/launch/h1_sim_bringup_mac.launch.py"
