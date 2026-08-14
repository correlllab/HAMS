"""Minimal, CPU-only (Apple Silicon) bringup for the H1 MuJoCo sim.

A trimmed variant of h1_sim_bringup.launch.py that runs ONLY the core robot
nodes — no vision foundation models (gemini/sam/graspgen), no h12_skills, no
Nav2/SLAM, no rviz/sliders. Those need the heavy x86-tuned ML stack that the
slim arm64 ROS image intentionally omits.

Nodes started:
  * static camera_link -> camera_color_optical_frame TF
  * static livox_link -> lidar_link TF (so RViz can place /livox/pointcloud)
  * joint_state_publisher      (h12_ros2_controller)
  * robot_state_publisher
  * frame_task_server          (h12_ros2_controller, Pink IK)
  * safety_node                (h12_safety_layer)

Launch by path from the slim ROS container:
  ros2 launch /home/code/core_ws/src/h1_bringup/launch/h1_sim_bringup_mac.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

ASSETS_DIR = '/home/code/CL_Assets'


def generate_launch_description():
    with open(os.path.join(ASSETS_DIR, 'ros_assets', 'h1_2_magpie_ros.urdf'), 'r') as urdf_file:
        robot_description = urdf_file.read()

    # MuJoCo publishes /clock with sim time; keep all nodes on it.
    sim_time_param = {'use_sim_time': True}

    nodes = [
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
        # The MuJoCo lidar bridge stamps /livox/{lidar,pointcloud} with frame_id
        # 'lidar_link', which is co-located with the URDF's 'livox_link' mount but
        # is not itself a URDF link. Publish the identity bridge so RViz (and any
        # lidar consumer) can transform the cloud into the robot/TF tree.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='lidar_frame_broadcaster',
            arguments=['0', '0', '0', '0', '0', '0', 'livox_link', 'lidar_link'],
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
    ]

    # Optional lower-body locomotion controller (GOLEM_LOWERBODY). Off by default:
    # the elastic-band tether then holds the robot upright. The node self-sequences
    # its band release (waits for the IK to finish) and publishes leg setpoints on
    # /safety/lowcmd_lower_in for the safety_node to merge.
    #   GOLEM_LOWERBODY=fame   RMA standing/squatting policy — balances the robot
    #                         standing UNSUPPORTED (verified). Does not locomote.
    #   GOLEM_LOWERBODY=walk   TorchScript walk policy on its own (marches in place;
    #                         topples in the cluttered kitchen).
    #   GOLEM_LOWERBODY=switch switchable controller: starts in FAME (stand) and
    #                         auto-switches stand<->walk from ||/cmd_vel|| (no start
    #                         services). This is the one to use for stand<->walk and
    #                         the nav demo — nav2's /cmd_vel makes it walk; it stands
    #                         again when the command drops to ~0.
    lowerbody = os.environ.get('GOLEM_LOWERBODY', '').strip().lower()
    _lowerbody_exec = {'fame': 'fame_node', 'walk': 'walking_node',
                       'switch': 'lowerbody_controller_node'}.get(lowerbody)
    if _lowerbody_exec:
        _lb_params = [sim_time_param]
        if _lowerbody_exec == 'lowerbody_controller_node':
            # Start in FAME; the node then auto-switches to walk on /cmd_vel.
            _lb_params.append({'active_policy': 'fame'})
        nodes.append(Node(
            package='h12_lowerbody_rl',
            executable=_lowerbody_exec,
            name=_lowerbody_exec,
            parameters=_lb_params,
            output='screen',
        ))

    # Optional 2D SLAM (GOLEM_SLAM=1): pointcloud_to_laserscan flattens the Livox
    # hemisphere cloud into a horizontal scan, and slam_toolbox builds an occupancy
    # /map from it against the sim's ground-truth /odom (odom -> pelvis). No
    # FAST-LIO needed. REQUIRES the robocasa/sim container started with
    # GOLEM_SIM_ODOM=1 (off by default) to publish that /odom; without it slam_toolbox
    # has no odom and the map stays empty. p2l params match h1_navigation.launch.py
    # but are fed from the raw /livox/pointcloud instead of FAST-LIO's registered cloud.
    if os.environ.get('GOLEM_SLAM', '').strip().lower() in ('1', 'true', 'on'):
        nodes.append(Node(
            package='pointcloud_to_laserscan',
            executable='pointcloud_to_laserscan_node',
            name='pointcloud_to_laserscan',
            parameters=[{
                'target_frame': 'pelvis',
                # Relaxed vs h1_navigation's FAST-LIO tuning for the raw sim lidar:
                # a taller band (counter tops / cabinets). range_min 0.65 drops the
                # robot's OWN legs/feet — the torso lidar looks down and returns them
                # at ~0.4 m; mapping those puts a phantom obstacle under the robot and
                # nav2 rejects the start as "lethal space". 0.65 clears the body while
                # still catching the counter (the robot spawns ~1 m off it via
                # GOLEM_SPAWN_BACKOFF). Lower it only if you also see real obstacles
                # missed at close range.
                'min_height': -0.85, 'max_height': 1.2,
                'angle_min': -3.14159, 'angle_max': 3.14159,
                'angle_increment': 0.0087,
                'range_min': 0.65, 'range_max': 8.0,
                'use_inf': True, 'scan_time': 0.0333,
                'transform_tolerance': 1.0, 'queue_size': 20,
            }, sim_time_param],
            remappings=[('cloud_in', '/livox/pointcloud'), ('scan', '/converted_scan')],
            output='screen',
        ))
        nodes.append(Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            parameters=[
                os.path.join(get_package_share_directory('h1_bringup'),
                             'config', 'slam_toolbox_h1.yaml'),
                {'odom_frame': 'odom', 'scan_topic': '/converted_scan'},
                sim_time_param,
            ],
            output='screen',
        ))

    # Optional autonomous navigation (GOLEM_NAV2=1, implies SLAM for the map): the
    # nav2 stack plans a collision-free path on the SLAM map + lidar costmap and
    # drives it via /cmd_vel, which the walk policy consumes (nav2's
    # velocity_smoother publishes /cmd_vel). Send goals with the RViz "2D Nav Goal"
    # tool or the /navigate_to_pose action. With GOLEM_LOWERBODY=switch the controller
    # auto-switches to walk as soon as nav2 publishes /cmd_vel (no manual step). The
    # robot must be standing in open floor — nav2 rejects a goal if the robot's start
    # cell is occupied (e.g. pressed against a counter).
    # nav2_config_mac.yaml is nav2_config.yaml with the odom frame swapped from
    # FAST-LIO's camera_init to the sim's odom.
    if os.environ.get('GOLEM_NAV2', '').strip().lower() in ('1', 'true', 'on'):
        nav2_launch = os.path.join(
            get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py')
        nav2_cfg = os.path.join(
            get_package_share_directory('h1_bringup'), 'config', 'nav2_config_mac.yaml')
        nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch),
            launch_arguments={'params_file': nav2_cfg, 'use_sim_time': 'true'}.items(),
        ))

    return LaunchDescription(nodes)
