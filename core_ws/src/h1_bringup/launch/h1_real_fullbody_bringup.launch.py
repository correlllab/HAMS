"""Full-body MJPC stand on the REAL H1-2 (desktop container).

Real-robot analog of h1_sim_fullbody_bench.launch.py: runs the WHOLE-BODY
controller (mjpc_fullbody_core = h12_control_node, nu=27, Lean task strat 6 --
the validated real-robot stand) with safety in FULL mode. The full node owns
BOTH arms and legs and publishes the whole rt/safety/lowcmd_in channel, so --
unlike the lower-body bringup -- there is NO frame_task_server and NO split
safety. This is the pure whole-body balance/reach mode (Lean), NOT the
loco-manip pipeline (that needs the lower-body controller + frame_task IK).

TOPOLOGY (differs from the lower-body real setup -- read before running):
  * ROBOT PC: run DRIVERS + ESTOP ONLY --
        ros2 launch h1_bringup h1_real_drivers.launch.py
        ros2 run estop estop_node
    Do NOT launch h1_real_robot_bringup.launch.py: it starts safety in SPLIT
    mode + frame_task_server, which would (a) run a SECOND safety node and
    (b) publish arms on lowcmd_upper_in -- both fight this full-body chain.
  * DESKTOP (this file): safety FULL + RW-EKF estimator + full-body MPC core.

  # export ROS_DOMAIN_ID first (the DDS + rclpy halves must share a domain).
  ros2 launch h1_bringup h1_real_fullbody_bringup.launch.py \
      start_position_verified:=true

Switch to the LOWER-BODY loco-manip controller instead with
h1_real_desktop_bringup.launch.py lowerbody:=mjpc (split safety + frame_task).

IMPORTANT: ROS_DOMAIN_ID must be exported in the launching shell (the safety
layer's unitree_sdk2py DDS and the rclpy launcher must agree on the domain).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_share = get_package_share_directory('h1_bringup')
    # Real robot: wall time (no /clock). Matches h1_real_desktop_bringup.
    sim_time_param = {'use_sim_time': False}
    default_cfg = os.path.join(bringup_share, 'config', 'mjpc_fullbody_real.yaml')

    return LaunchDescription([
        # Engage the full-body policy only once the operator has verified a safe
        # start position (start_position_verified:=true). Defaults to false so a
        # bare launch brings up safety + estimator but never commands the robot.
        DeclareLaunchArgument('start_position_verified', default_value='false'),
        # Param file for the MJPC estimator + full-body controller pair.
        DeclareLaunchArgument('mjpc_config', default_value=default_cfg),

        # Full-body safety relay (FULL mode: one whole-body rt/safety/lowcmd_in
        # channel). This is the REAL full-safety config -- the sim bench uses
        # sim_safety_full.yaml instead. Runs unconditionally: with no controller
        # commands it simply holds/clamps.
        Node(
            package='h12_safety_layer',
            executable='safety_node',
            name='safety_node',
            arguments=['--config', 'default_safety_full.yaml'],
            parameters=[sim_time_param],
            output='screen',
        ),

        # RW-EKF base estimator -> rt/sportmodestate (owns it in debug mode;
        # identical role to the lower-body real bringup). Runs unconditionally
        # so the state feed is live before the operator arms the controller.
        Node(
            package='h12_deploy_mjpc',
            executable='estimator_node',
            name='h12_deploy_mjpc_estimator',
            parameters=[sim_time_param, LaunchConfiguration('mjpc_config')],
            output='screen',
        ),

        # Whole-body MJPC controller: the rclpy launcher execs mjpc_fullbody_core
        # (selected via controller_exe in mjpc_fullbody_real.yaml). Same
        # MJPC_TASKS_DIR env as the lower-body node so the Lean task model
        # resolves against the runtime-hydrated build tree that matches the
        # linked mjpc libs. Gated on start_position_verified so the legs+arms
        # are never auto-commanded on a bare launch.
        Node(
            package='h12_deploy_mjpc',
            executable='mjpc_deploy_lowerbody_controller',
            name='mjpc_deploy_lowerbody_controller',
            parameters=[sim_time_param, LaunchConfiguration('mjpc_config')],
            additional_env={'MJPC_TASKS_DIR': '/home/code/mujoco_mpc/build/mjpc/tasks'},
            output='screen',
            condition=IfCondition(LaunchConfiguration('start_position_verified')),
        ),
    ])
