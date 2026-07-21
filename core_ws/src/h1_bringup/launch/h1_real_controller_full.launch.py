"""Robot-side controller substrate for the WHOLE-BODY MPC grasp stack.

The full-body sibling of h1_real_controller.launch.py. That one wires the SPLIT
architecture: it starts the safety layer in SPLIT mode (relax_safety_split.yaml
-> reads rt/safety/lowcmd_lower_in + lowcmd_upper_in) and launches
frame_task_server (the arm controller). Our unified whole-body MPC publishes a
SINGLE full-body channel (rt/safety/lowcmd_in), so the split-mode safety layer
would ignore it entirely and the robot would not move.

This launch fixes both differences:
  * safety_node runs in FULL-BODY mode (relax_safety_full.yaml -> reads
    rt/safety/lowcmd_in, the channel mjpc_fullbody_core drives).
  * frame_task_server is DROPPED -- the whole-body MPC owns the arms AND legs;
    there is no separate arm controller.

Everything else (estop, joint_state_publisher, robot_state_publisher) is kept
IDENTICAL to the split controller launch. The split files are untouched, so
`h1_real_robot_bringup.launch.py` (RL/ALMI + FrameTask) still works as before.

⚠ ROS_DOMAIN_ID must be exported in the launching shell (0 = real robot); the
safety layer's DDS half and the rclpy half must share the same domain.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


ASSETS_DIR = '/home/unitree/HAMS/CL_Assets'


def generate_launch_description():
    with open(os.path.join(ASSETS_DIR, 'ros_assets', 'h1_2_magpie_ros.urdf'), 'r') as urdf_file:
        robot_description = urdf_file.read()

    sim_time_param = {'use_sim_time': False}

    return LaunchDescription([
        # Full-body safety config. relax_safety_full.yaml = full_body_mode reading
        # rt/safety/lowcmd_in (parity clamps with the relax_split the ALMI path
        # uses). Swap to default_safety_full / tight_safety_full to tighten limits.
        DeclareLaunchArgument('safety_config', default_value='relax_safety_full.yaml'),

        Node(
            package='estop',
            executable='estop_node',
            name='estop_node',
            parameters=[sim_time_param],
            output='screen',
        ),
        Node(
            package='h12_ros2_controller',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[sim_time_param],
            output='screen',
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}, sim_time_param],
            output='screen',
        ),
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='h12_safety_layer',
                    executable='safety_node',
                    name='safety_node',
                    parameters=[sim_time_param],
                    arguments=['--config', LaunchConfiguration('safety_config')],
                    output='screen',
                ),
            ],
        ),
        # NOTE: frame_task_server is intentionally NOT launched here. The
        # whole-body MPC commands all 27 joints (arms included); a live
        # frame_task_server would only publish on the split upper channel, which
        # the full_body_mode safety layer ignores anyway.
    ])
