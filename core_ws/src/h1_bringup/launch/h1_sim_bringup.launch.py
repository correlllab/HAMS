import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
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

    with open(os.path.join(ASSETS_DIR, 'ros_assets', 'h1_2_magpie_ros.urdf'), 'r') as urdf_file:
        robot_description = urdf_file.read()

    # MuJoCo publishes /clock with sim time. All nodes should use it so
    # TF lookups and sensor timestamps are coherent with the simulation.
    sim_time_param = {'use_sim_time': True}

    nav_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'h1_navigation.launch.py')
        ),
        launch_arguments={'use_sim_time': 'true'}.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('use_sliders', default_value='true'),
        # 'none' = robot starts held by the elastic band, idle, until a policy is
        # started via /lowerbody/start_<walk|fame> (or set active_policy to
        # auto-engage one at launch).
        # Which lower-body node to run. Default 'fame_node' = the FAME-only node
        # used by the band-held open_fridge demo (unchanged). Set to
        # 'lowerbody_controller_node' for the switchable walk<->FAME controller
        # (adds /lowerbody/start_walk|start_fame, /cmd_vel walking) — used by the
        # walk-to-fridge navigation task.
        DeclareLaunchArgument('rviz_config', default_value=default_rviz),

        nav_launch,

        # The MuJoCo bridge back-projects depth into 3D using REP-103 optical
        # convention (+z = forward, +x = right, +y = down) but stamps the
        # resulting camera_info / image / depth messages with the optical
        # frame name below. The URDF defines only camera_link (a ROS link
        # frame, +x = forward, +z = up); without this static TF the vision
        # pipeline would treat optical-convention points as if they were in
        # camera_link, rotating every detection ~90 deg out of place.
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
            package="h12_ros2_controller",
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[sim_time_param],
            output='screen'
        ),

        # h12_ros2_controller (from full_launch.py, minus rviz)
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
            arguments=['--config', 'safety_split.yaml'],
            parameters=[sim_time_param],
            output='screen',
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
            parameters=[sim_time_param, ],
            arguments=['--config', "default_safety_split.yaml"],
            output='screen',
        ),

        # Lower-body controller with switchable RL policies (walk / FAME stand).
        # Defaults to the FAME standing policy so the base holds still for
        # manipulation; switch to 'walk' for navigation by publishing the policy
        # name on /lowerbody/set_policy (the controller waits for a safe handover
        # — standing still, arms home — before committing the switch).
        Node(
            package='h12_lowerbody_controller',
            executable="lowerbody_controller_node",
            name="lowerbody_node",
            parameters=[sim_time_param, {'active_policy': "fame"}],
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
