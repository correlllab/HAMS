"""Full-body MJPC stand BENCH on RoboCasa.

Runs the WHOLE-BODY controller (mjpc_fullbody_core = h12_control_node, nu=27,
Lean task strat 6 -- the validated 281 s real-robot stand) through the SAME
in-container DDS/estimator path as the lower-body bringup, so it's an
apples-to-apples reference for confirming the sim2sim2sim contact-parity fix.

Differs from h1_sim_bringup.launch.py: safety runs in FULL mode (the full node
publishes the whole rt/safety/lowcmd_in), there is NO frame_task_server (the full
node owns the arms), and no vision/nav/rviz. Pair with the RoboCasa sim launched
with --twin-contact (balance-only contact parity).

  ros2 launch h1_bringup h1_sim_fullbody_bench.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_share = get_package_share_directory('h1_bringup')
    sim_time_param = {'use_sim_time': False}
    default_cfg = os.path.join(bringup_share, 'config', 'mjpc_fullbody.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('mjpc_config', default_value=default_cfg),

        # Full-body safety relay (full_body_mode; one whole-body channel).
        Node(
            package='h12_safety_layer',
            executable='safety_node',
            name='safety_node',
            arguments=['--config', 'sim_safety_full.yaml'],
            parameters=[sim_time_param],
            output='screen',
        ),

        # RW-EKF base estimator (identical to the lower-body bringup).
        Node(
            package='h12_deploy_mjpc',
            executable='estimator_node',
            name='h12_deploy_mjpc_estimator',
            parameters=[sim_time_param, LaunchConfiguration('mjpc_config')],
            output='screen',
        ),

        # Whole-body MJPC controller: the rclpy launcher execs mjpc_fullbody_core
        # (selected via controller_exe in mjpc_fullbody.yaml). Same MJPC_TASKS_DIR
        # env as the lower-body node so the Lean task model resolves.
        Node(
            package='h12_deploy_mjpc',
            executable='mjpc_deploy_lowerbody_controller',
            name='mjpc_deploy_lowerbody_controller',
            parameters=[sim_time_param, LaunchConfiguration('mjpc_config')],
            additional_env={'MJPC_TASKS_DIR': '/home/code/mujoco_mpc/build/mjpc/tasks'},
            output='screen',
        ),
    ])
