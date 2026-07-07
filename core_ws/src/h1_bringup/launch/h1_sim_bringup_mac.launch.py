"""Minimal, CPU-only (Apple Silicon) bringup for the H1 MuJoCo sim.

A trimmed variant of h1_sim_bringup.launch.py that runs ONLY the core robot
nodes — no vision foundation models (gemini/sam/graspgen), no h12_skills, no
Nav2/SLAM, no rviz/sliders. Those need the heavy x86-tuned ML stack that the
slim arm64 ROS image intentionally omits.

Nodes started:
  * static camera_link -> camera_color_optical_frame TF
  * joint_state_publisher      (h12_ros2_controller)
  * robot_state_publisher
  * frame_task_server          (h12_ros2_controller, Pink IK)
  * safety_node                (h12_safety_layer)

Launch by path from the slim ROS container:
  ros2 launch /home/code/core_ws/src/h1_bringup/launch/h1_sim_bringup_mac.launch.py
"""
import os

from launch import LaunchDescription
from launch_ros.actions import Node

ASSETS_DIR = '/home/code/CL_Assets'


def generate_launch_description():
    with open(os.path.join(ASSETS_DIR, 'ros_assets', 'h1_2_magpie_ros.urdf'), 'r') as urdf_file:
        robot_description = urdf_file.read()

    # MuJoCo publishes /clock with sim time; keep all nodes on it.
    sim_time_param = {'use_sim_time': True}

    return LaunchDescription([
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_optical_frame_broadcaster',
            arguments=['0', '0', '0',
                       '-1.5707963267948966', '0', '-1.5707963267948966',
                       'camera_link', 'camera_color_optical_frame'],
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
        Node(
            package='h12_ros2_controller',
            executable='frame_task_server',
            name='frame_task_server',
            arguments=['--config', 'sim_safety_split.yaml'],
            parameters=[sim_time_param],
            output='screen',
        ),
        Node(
            package='h12_safety_layer',
            executable='safety_node',
            name='safety_node',
            arguments=['--config', 'sim_safety_split.yaml'],
            parameters=[sim_time_param],
            output='screen',
        ),
    ])
