import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ASSETS_DIR = '/home/code/assets'


def generate_launch_description():
    # The included h12_ros2_controller/full_launch.py and vision_pipeline/vp.launch.py
    # both start their own rviz2. We can't patch those upstream packages, so this
    # bringup inlines their non-rviz nodes and runs a single rviz with sim.rviz.
    bringup_share = get_package_share_directory('h1_bringup')
    default_rviz = os.path.join(bringup_share, 'rviz', 'sim.rviz')
    default_config = os.path.join(bringup_share, 'config', 'sim_network.yaml')

    with open(os.path.join(ASSETS_DIR, 'ros_assets', 'h1_2_handless_ros.urdf'), 'r') as urdf_file:
        robot_description = urdf_file.read()

    config = LaunchConfiguration('config')

    # MuJoCo publishes /clock with sim time. All nodes should use it so
    # TF lookups and sensor timestamps are coherent with the simulation.
    sim_time_param = {'use_sim_time': True}

    return LaunchDescription([
        SetEnvironmentVariable('ROS_DOMAIN_ID', '1'),

        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz),
        DeclareLaunchArgument('config', default_value=default_config),

        # h12_ros2_controller (from full_launch.py, minus rviz)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}, sim_time_param],
            output='screen',
        ),
        Node(
            package='h1_bringup',
            executable='sim_joint_state_publisher',
            name='joint_state_publisher',
            parameters=[sim_time_param],
            output='screen',
        ),
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='h12_ros2_controller',
                    executable='frame_task_server',
                    name='frame_task_server',
                    arguments=['--config', config],
                    parameters=[sim_time_param],
                    output='screen',
                ),
            ],
        ),

        # vision_pipeline (from vp.launch.py, minus rviz)
        Node(
            package='vision_pipeline',
            executable='vp',
            name='vp_node',
            parameters=[sim_time_param],
            output='screen',
        ),

        Node(
            package='h12_safety_layer',
            executable='safety_node',
            name='safety_node',
            parameters=[sim_time_param],
            output='screen',
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2_sim',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            parameters=[sim_time_param],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        ),

        # cv2 bundles its own Qt plugins that can clobber rviz2's Qt when both
        # load in the same launch. Delay the slider GUI so rviz2 has already
        # initialised Qt from the system libs before cv2 imports.
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='h1_bringup',
                    executable='wrist_slider_gui',
                    name='wrist_slider_gui',
                    parameters=[sim_time_param],
                    output='screen',
                ),
            ],
        ),
    ])
