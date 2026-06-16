import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# IMPORTANT: ROS_DOMAIN_ID must be exported in the launching shell. The safety
# layer uses unitree_sdk2py's DDS ChannelSubscriber (which honours
# $ROS_DOMAIN_ID and falls back to the YAML's network.domain_id only if the env
# is unset), while walking_node / frame_task_server / mujoco_ros_bridge are
# rclpy nodes that pick up the env directly. If the variable is missing, the
# DDS half and the rclpy half can end up on different domains and the safety
# layer will see no commands.
ASSETS_DIR = '/home/code/CL_Assets'


def generate_launch_description():
    # The included h12_ros2_controller/full_launch.py and vision_pipeline/vp.launch.py
    # both start their own rviz2. We can't patch those upstream packages, so this
    # bringup inlines their non-rviz nodes and runs a single rviz with sim.rviz.
    bringup_share = get_package_share_directory('h1_bringup')
    default_rviz = os.path.join(bringup_share, 'rviz', 'sim.rviz')

    # MuJoCo publishes /clock with sim time. All nodes should use it so
    # TF lookups and sensor timestamps are coherent with the simulation.
    sim_time_param = {'use_sim_time': False}

    return LaunchDescription([
        
        # vision_pipeline (from vp.launch.py, minus rviz)
        Node(
            package='vision_pipeline',
            executable='vp',
            name='vp_node',
            parameters=[sim_time_param],
            output='screen',
        ),

        # Switchable lower-body RL controller (walk / FAME stand-squat).
        # Auto-engages the FAME standing policy; switch via /lowerbody/start_walk
        # or /lowerbody/set_policy (waits for a safe handover before committing).
        Node(
            package='h12_lowerbody_controller',
            executable='lowerbody_controller_node',
            name='lowerbody_controller_node',
            parameters=[sim_time_param, {'active_policy': 'fame'}],
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

        # slider_debugger waits up to 5s on /left_ee_pose & /right_ee_pose,
        # which frame_task_server publishes only after its IK solver finishes
        # initialising (URDF load + 150-step torso init — empirically ~7s).
        # 10s leaves headroom so the sliders seed from the live pose.
        #
        # Intentionally NOT using sim_time: the GUI's wait_for_initial_poses
        # measures wall-clock; with use_sim_time=True a fast sim that's
        # already past 5s makes get_clock().now() jump and trip the timeout
        # immediately, falling back to all-zero targets that drive the IK
        # toward unreachable poses inside the body.
        TimerAction(
            period=1.0,
            actions=[
                Node(
                    package='h1_bringup',
                    executable='slider_debugger.py',
                    name='slider_debugger',
                    output='screen',
                    condition=IfCondition(LaunchConfiguration('use_sliders')),
                ),
            ],
        ),
    ])
