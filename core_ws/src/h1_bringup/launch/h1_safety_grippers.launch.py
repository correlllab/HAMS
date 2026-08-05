import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Minimal real-robot actuation bringup: estop + safety layer + both grippers.
# No lidar, no cameras, no state publishers, no arm IK -- just the pieces that
# have to be alive for a command to reach a motor safely. Use it when working on
# gripper/low-level actuation without paying for the full driver stack; the full
# onboard bringup (h1_real_robot_bringup.launch.py) is still the normal path.
#
# IMPORTANT: ROS_DOMAIN_ID must be exported in the launching shell (0 on the real
# robot). The safety layer uses unitree_sdk2py's DDS ChannelSubscriber (which
# honours $ROS_DOMAIN_ID and falls back to the YAML's network.domain_id only if
# the env is unset), while estop's rclpy half picks up the env directly. If the
# variable is missing the two halves can land on different domains and the safety
# layer will see no commands.


def generate_launch_description():
    bringup_share = get_package_share_directory('h1_bringup')

    # Same config the full real bringup runs with (h1_real_controller.launch.py);
    # overridable so a session can tighten/relax limits without editing launch.
    safety_config_arg = DeclareLaunchArgument(
        'safety_config',
        default_value='relax_safety_split.yaml',
        description='h12_safety_layer config name or path (resolved against its config/ dir)',
    )

    sim_time_param = {'use_sim_time': False}

    estop = Node(
        package='estop',
        executable='estop_node',
        name='estop_node',
        parameters=[sim_time_param],
        output='screen',
    )

    # Staggered behind estop: the safety layer latches on the estop status it
    # reads at startup, so give the estop node time to open its serial port and
    # publish real state first (mirrors h1_real_controller.launch.py).
    safety = TimerAction(
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
    )

    grippers = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'h1_grippers.launch.py')
        ),
    )

    return LaunchDescription([
        safety_config_arg,
        estop,
        safety,
        grippers,
    ])
