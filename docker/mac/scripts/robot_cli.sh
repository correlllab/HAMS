#!/bin/bash
# Convenience helpers for driving the H1 from inside the hams_ros container.
# This dir is live-mounted (docker/mac/docker-compose.yml: ./scripts -> /home/code/h12_sim_scripts),
# so edits on the Mac appear here immediately — no rebuild.
#
# Usage (inside hams_ros):
#   source /home/code/h12_sim_scripts/robot_cli.sh
#   rob_poses                       # list named postures
#   rob_pose t_pose                 # move to a named posture (default 3s)
#   rob_pose arms_overhead 4        # ... over 4 seconds
#   rob_grip right close            # close/open a gripper
#   rob_gripset left 0.5            # set gripper position (0=open .. 1=closed)
#   rob_eepose right                # print current right-hand pose (pelvis frame)
#   rob_home                        # arms-down baseline
# (rob_reach / Cartesian reaching is disabled — the FrameTask action crashes the
#  controller node; drive the robot in joint space with rob_pose for now.)
#
# Watch it live in your browser (from the Mac): http://localhost:6080/vnc.html

# --- ROS environment (idempotent) ---
source /opt/ros/humble/setup.bash 2>/dev/null
source /opt/core_ws/install/setup.bash 2>/dev/null
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Named postures available to rob_pose (from utility/named_config.py):
#   home t_pose arms_front arms_front_elbow t_pose_elbow arms_overhead
#   arms_asym arms_front_yaw elbow_only
rob_poses() {
    echo "home t_pose arms_front arms_front_elbow t_pose_elbow arms_overhead arms_asym arms_front_yaw elbow_only"
}

# rob_pose <config_name> [seconds]
rob_pose() {
    local cfg="${1:?usage: rob_pose <name> [secs] — see rob_poses}" secs="${2:-3}"
    echo "[rob_pose] -> $cfg over ${secs}s"
    ros2 action send_goal /named_config custom_ros_messages/action/NamedConfig \
        "{config_name: $cfg, duration: {sec: $secs, nanosec: 0}}"
}

rob_home() { rob_pose home "${1:-3}"; }

# rob_grip <left|right> <open|close>
rob_grip() {
    local side="${1:?usage: rob_grip <left|right> <open|close>}" act="${2:?open|close}"
    ros2 service call "/${side}/gripper/${act}" std_srvs/srv/Trigger
}

# rob_gripset <left|right> <0.0..1.0>   (0 = open, 1 = closed)
rob_gripset() {
    local side="${1:?usage: rob_gripset <left|right> <0..1>}" pos="${2:?0..1}"
    ros2 service call "/${side}/gripper/set_position" magpie_msgs/srv/SetGripperPosition \
        "{position: $pos}"
}

# rob_eepose <left|right>   — current hand pose in the pelvis frame
# (timeout-guarded: the CLI echo can hang on a QoS mismatch; if it prints
# nothing, that's the known ros2-CLI QoS artifact, not a missing topic.)
rob_eepose() {
    local side="${1:-right}"
    timeout 8 ros2 topic echo "/${side}_ee_pose" --once \
        || echo "[rob_eepose] no sample in 8s (QoS-match CLI artifact; the topic is live)"
}

# rob_reach — DISABLED. The /frame_task (FrameTask) action currently CRASHES the
# frame_task_server: after a goal sets self.frame_names, the node's periodic
# /frame_poses publisher calls get_frame_transformation() on that frame in the
# FULL model, where a wrist-frame id is out of range -> IndexError kills the node
# (frame_task_server.py:124 -> robot_model.py:426). The arm does move first, but
# the controller then dies and all further commands hang on "waiting for action
# server". Until that's fixed, drive the robot in joint space (rob_pose) instead.
rob_reach() {
    echo "rob_reach is disabled: the FrameTask action crashes frame_task_server"
    echo "(get_frame_transformation IndexError on the wrist frame). Use rob_pose <name>."
    return 1
}

# --- Lower-body locomotion (needs HAMS_LOWERBODY=switch) ------------------------
# The switchable controller starts in FAME (stand) and auto-switches stand<->walk
# from ||/cmd_vel||: any velocity command makes it walk; zero makes it stand. There
# are no start services — rob_go drives /cmd_vel, rob_stop zeroes it (back to FAME).

# rob_stand — command zero velocity; the controller auto-switches back to FAME.
rob_stand() { ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}" >/dev/null 2>&1; }
# rob_walk — no-op: walk mode is entered automatically by commanding a velocity.
rob_walk()  { echo "[rob_walk] send a velocity (rob_go ...) — walk engages automatically"; }

# rob_go <vx> [vy] [wz] [secs]  — drive the walk policy (m/s, m/s, rad/s).
# Publishes /cmd_vel at 20 Hz for `secs` (default 6). vx>0 forward, wz>0 turn left.
rob_go() {
    local vx="${1:-0.4}" vy="${2:-0.0}" wz="${3:-0.0}" secs="${4:-6}"
    echo "[rob_go] vx=$vx vy=$vy wz=$wz for ${secs}s"
    timeout "$secs" ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist \
        "{linear: {x: $vx, y: $vy}, angular: {z: $wz}}"
}

# rob_stop — zero velocity; the controller auto-switches back to FAME (stand still).
rob_stop() {
    ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}" >/dev/null 2>&1
}

# rob_odom — current base position in the odom (world) frame.
rob_odom() { timeout 5 ros2 topic echo /odom --field pose.pose.position --once; }

# rob_explore — autonomous frontier-based exploration (needs HAMS_SLAM=1 HAMS_NAV2=1).
# Sends the /skill/frontier_explore action, which repeatedly navigates to the nearest
# unexplored frontier on the SLAM map until it's fully mapped. The controller
# auto-switches to walk as soon as nav2 publishes /cmd_vel (no manual walk step) and
# stands (FAME) between goals. Ctrl-C to stop (cancels the goal). Watch RViz (6081).
rob_explore() {
    ros2 action send_goal /skill/frontier_explore \
        custom_ros_messages/action/SkillFrontierExplore \
        "{min_frontier_cells: 0, blacklist_radius: 0.0, min_goal_distance: 0.0, goal_timeout: 0.0, timeout: {sec: 1800, nanosec: 0}}" \
        --feedback
}

echo "robot_cli loaded."
echo "  postures : rob_pose t_pose | rob_grip right close"
echo "  locomote : rob_go 0.4 0 0.3 (fwd+turn) -> rob_stop   (auto stand<->walk; needs HAMS_LOWERBODY=switch)"
echo "  explore  : rob_explore   (autonomous frontier mapping; needs HAMS_SLAM=1 HAMS_NAV2=1)"
