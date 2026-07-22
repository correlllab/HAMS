"""Launch the optional H12 spatial-memory skill."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Declare portable sensor/backend parameters and start the memory node."""
    arguments = [
        DeclareLaunchArgument(
            'embodied_agent_root',
            default_value=EnvironmentVariable(
                'EMBODIED_AGENT_ROOT', default_value='/opt/EmbodiedAgent')),
        DeclareLaunchArgument('data_dir', default_value='/data/spatial_memory'),
        DeclareLaunchArgument(
            'camera_topic',
            default_value='/realsense/head/color/image_raw/compressed'),
        DeclareLaunchArgument('world_frame', default_value='odom'),
        DeclareLaunchArgument('robot_frame', default_value='pelvis'),
        DeclareLaunchArgument('capture_interval_sec', default_value='2.0'),
        DeclareLaunchArgument('model', default_value='siglip_base'),
        DeclareLaunchArgument('device', default_value='auto'),
        DeclareLaunchArgument('recall_k', default_value='12'),
        DeclareLaunchArgument('vlm_model', default_value='gemini-3.5-flash'),
    ]
    memory_node = Node(
        package='h12_skills',
        executable='spatial_memory',
        name='h12_spatial_memory',
        output='screen',
        parameters=[{
            'use_sim_time': ParameterValue(
                LaunchConfiguration('use_sim_time'), value_type=bool),
            'embodied_agent_root': LaunchConfiguration('embodied_agent_root'),
            'data_dir': LaunchConfiguration('data_dir'),
            'camera_topic': LaunchConfiguration('camera_topic'),
            'world_frame': LaunchConfiguration('world_frame'),
            'robot_frame': LaunchConfiguration('robot_frame'),
            'capture_interval_sec': ParameterValue(
                LaunchConfiguration('capture_interval_sec'), value_type=float),
            'model': LaunchConfiguration('model'),
            'device': LaunchConfiguration('device'),
            'recall_k': ParameterValue(
                LaunchConfiguration('recall_k'), value_type=int),
            'vlm_model': LaunchConfiguration('vlm_model'),
        }],
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value=os.environ.get('HAMS_SIM_TIME', 'true')),
        *arguments,
        memory_node,
    ])
