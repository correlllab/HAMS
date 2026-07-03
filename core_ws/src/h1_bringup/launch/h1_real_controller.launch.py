import os

from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


# IMPORTANT: ROS_DOMAIN_ID must be exported in the launching shell. The safety
# layer uses unitree_sdk2py's DDS ChannelSubscriber (which honours
# $ROS_DOMAIN_ID and falls back to the YAML's network.domain_id only if the env
# is unset), while the controller-side rclpy nodes pick up the env directly.
# If the variable is missing, the DDS half and the rclpy half can end up on
# different domains and the safety layer will see no commands.
ASSETS_DIR = '/home/unitree/HAMS/CL_Assets'


def generate_launch_description():
    with open(os.path.join(ASSETS_DIR, 'ros_assets', 'h1_2_magpie_ros.urdf'), 'r') as urdf_file:
        robot_description = urdf_file.read()

    sim_time_param = {'use_sim_time': False}

    return LaunchDescription([
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
                    arguments=['--config', 'default_safety_split.yaml'],
                    output='screen',
                ),
            ],
        ),
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='h12_ros2_controller',
                    executable='frame_task_server',
                    name='frame_task_server',
                    arguments=['--config', 'safety_split.yaml'],
                    parameters=[sim_time_param],
                    output='screen',
                ),
            ],
        ),
    ])
