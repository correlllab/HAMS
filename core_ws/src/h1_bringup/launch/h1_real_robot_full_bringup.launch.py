"""Robot-side bringup for the WHOLE-BODY MPC grasp stack.

The full-body sibling of h1_real_robot_bringup.launch.py. Identical drivers +
navigation, but the controller substrate is h1_real_controller_FULL (safety layer
in full_body_mode reading rt/safety/lowcmd_in, and NO frame_task_server) instead
of the split controller (split-mode safety + FrameTask arms).

Run this on the ROBOT, paired with the full desktop bringup on the companion box:
    # on the robot:
    ros2 launch h1_bringup h1_real_robot_full_bringup.launch.py
    # on your device (inside hams_ros, same ROS_DOMAIN_ID):
    ros2 launch h1_bringup h1_real_desktop_full_bringup.launch.py start_position_verified:=true

The split bringup (h1_real_robot_bringup.launch.py) is left untouched -- the
RL/ALMI + FrameTask path still works exactly as before.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    bringup_share = get_package_share_directory('h1_bringup')

    nav_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'h1_navigation.launch.py')
        ),
        launch_arguments={'use_sim_time': 'false'}.items(),
    )
    drivers_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'h1_real_drivers.launch.py')
        ),
        launch_arguments={'use_sim_time': 'false'}.items(),
    )

    # FULL-BODY controller substrate: full_body_mode safety, no FrameTask.
    controller_full_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'h1_real_controller_full.launch.py')
        ),
        launch_arguments={
            'safety_config': LaunchConfiguration('safety_config'),
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('safety_config', default_value='relax_safety_full.yaml'),
        drivers_launch,
        nav_launch,
        controller_full_launch,
    ])
