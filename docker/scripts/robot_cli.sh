#!/bin/bash
# Convenience helpers for driving the H1 from inside the hams_ros container.
# This dir is live-mounted (docker-compose.mac.yml: ./scripts -> /home/code/h12_sim_scripts),
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

echo "robot_cli loaded. Try: rob_poses  (lists names) | rob_pose t_pose  (MOVES) | rob_grip right close"
