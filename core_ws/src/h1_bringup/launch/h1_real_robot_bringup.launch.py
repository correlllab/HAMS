import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


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

    controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'h1_real_controller.launch.py')
        ),
    )

    return LaunchDescription([
        drivers_launch,
        nav_launch,
        controller_launch,
    ])
