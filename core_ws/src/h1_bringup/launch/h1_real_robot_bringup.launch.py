import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

# IMPORTANT: ROS_DOMAIN_ID must be exported in the launching shell. The safety
# layer uses unitree_sdk2py's DDS ChannelSubscriber (which honours
# $ROS_DOMAIN_ID and falls back to the YAML's network.domain_id only if the env
# is unset), while walking_node / frame_task_server / mujoco_ros_bridge are
# rclpy nodes that pick up the env directly. If the variable is missing, the
# DDS half and the rclpy half can end up on different domains and the safety
# layer will see no commands.
ASSETS_DIR = '/home/unitree/Humanoid_Simulation/CL_Assets'

def generate_launch_description():
    # The included h12_ros2_controller/full_launch.py and vision_pipeline/vp.launch.py
    # both start their own rviz2. We can't patch those upstream packages, so this
    # bringup inlines their non-rviz nodes and runs a single rviz with sim.rviz.
    bringup_share = get_package_share_directory('h1_bringup')

    with open(os.path.join(ASSETS_DIR, 'ros_assets', 'h1_2_magpie_ros.urdf'), 'r') as urdf_file:
        robot_description = urdf_file.read()

    # MuJoCo publishes /clock with sim time. All nodes should use it so
    # TF lookups and sensor timestamps are coherent with the simulation.
    sim_time_param = {'use_sim_time': False}

    nav_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'h1_navigation.launch.py')
        ),
        launch_arguments={'use_sim_time': 'false'}.items(),
    )
    drivers_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'h1_drivers.launch.py')
        ),
        launch_arguments={'use_sim_time': 'false'}.items(),
    )

    return LaunchDescription([

        drivers_launch,

        nav_launch,

        Node(
            package='estop',
            executable='estop_node',
            name='estop_node',
            parameters=[sim_time_param],
            output='screen',
        ),

        Node(
            package="h12_ros2_controller",
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[sim_time_param],
            output='screen'
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
                    parameters=[sim_time_param, ],
                    arguments=['--config', "default_safety_split.yaml"],
                    output='screen',
                ),
            ]
        ),
        
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='h12_ros2_controller',
                    executable='frame_task_server',
                    name='frame_task_server',
                    arguments=['--config', 'tight_safety_split.yaml'],
                    parameters=[sim_time_param],
                    output='screen',
                ),
            ]
        ),
        
    ])
