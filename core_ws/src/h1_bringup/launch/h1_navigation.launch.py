import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    bringup_share = get_package_share_directory('h1_bringup')
    config_dir = os.path.join(bringup_share, 'config')

    use_sim_time = LaunchConfiguration('use_sim_time')
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation (MuJoCo) clock if true'
    )

    fast_lio_config_file = LaunchConfiguration('fast_lio_config_file')
    declare_fast_lio_config_file = DeclareLaunchArgument(
        'fast_lio_config_file', default_value='mid360.yaml',
        description='Config file for fast_lio (resolved against h1_bringup/config)'
    )

    nav2_params_file = LaunchConfiguration('nav2_params_file')
    declare_nav2_params_file = DeclareLaunchArgument(
        'nav2_params_file',
        default_value=os.path.join(config_dir, 'nav2_config.yaml'),
        description='Full path to the Nav2 parameters file to use'
    )

    slam_params_file = LaunchConfiguration('slam_params_file')
    declare_slam_params_file = DeclareLaunchArgument(
        'slam_params_file',
        default_value=os.path.join(config_dir, 'slam_toolbox_h1.yaml'),
        description='Full path to the slam_toolbox parameters file'
    )

    # FAST-LIO. Subscribes to /livox/lidar and /livox/imu per mid360.yaml.
    # Assumes the Livox driver (or its sim equivalent) is already publishing them.
    fast_lio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        parameters=[PathJoinSubstitution([config_dir, fast_lio_config_file]),
                    {'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Attach FAST-LIO's 'body' frame to H1's 'pelvis' frame.
    #
    # FAST-LIO publishes camera_init -> body where 'body' is the IMU frame
    # (livox_imu_site in MJCF, which sits at the origin of livox_link).
    # So body == livox_link at startup. The URDF places livox_link on torso_link at
    #   xyz=(0.04874, 0, 0.67980) rpy=(0, 0.24015730507441985, 0)
    # (see CL_Assets/ros_assets/h1_2_magpie_ros.urdf, livox_joint), and torso_joint
    # sits at the pelvis origin with identity rotation.
    #
    # Inverting that transform (body -> pelvis):
    #   rotation: RotY(-0.24015730507441985)
    #   translation: -RotY(-0.24015730507441985) * (0.04874, 0, 0.67980)
    #              ≈ (0.11433, 0.0, -0.67188)
    static_tf_broadcaster = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='body_to_pelvis_broadcaster',
        arguments=['0.11433', '0.0', '-0.67188',
                   '0.0', '-0.24015730507441985', '0.0',
                   'body', 'pelvis'],
        output='screen'
    )

    pointcloud_to_laserscan_node = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='cloud_registered_to_laserscan',
        output='screen',
        parameters=[
            {'target_frame': 'pelvis'},
            {'use_sim_time': use_sim_time},
            # Height band in pelvis frame. Tight enough to exclude the
            # robot's own head/neck (above) and feet (well below); wide
            # enough to catch knee-to-chest height obstacles at distance.
            # Pelvis sits ~1.03 m above the floor, so -0.90 collapses returns
            # from ~0.13 m up (was -0.55 ≈ 0.48 m / knee height) to catch low
            # shelves and table legs/aprons. Lower bound is a floor-clearance
            # margin: drop further only until the floor/own-feet start marking.
            {'min_height': -0.90},
            {'max_height': 0.55},
            {'angle_min': -3.14159},
            {'angle_max': 3.14159},
            {'angle_increment': 0.0087},
            # range_min is the 2D horizontal distance from the pelvis origin
            # after transforming the cloud to target_frame. The H1 body
            # envelope is ≤ 0.35 m horizontal from pelvis, so 0.6 m drops
            # every self-return (legs in mid-stride, arms hanging, torso)
            # while leaving close-but-not-on-robot obstacles visible.
            {'range_min': 0.6},
            {'range_max': 6.0},
            {'use_inf': True},
            {'scan_time': 0.0333},
            {'transform_tolerance': 0.05},
            {'queue_size': 20},
        ],
        remappings=[
            ('cloud_in', '/cloud_registered_body'),
            ('scan', '/converted_scan'),
        ]
    )

    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[
            slam_params_file,
            {
                'base_frame': 'pelvis',
                'odom_frame': 'camera_init',
                'scan_topic': '/converted_scan',
                'odom_topic': '/Odometry',
                'use_sim_time': use_sim_time,
            }
        ]
    )

    # Nav2: bring up the navigation half only. slam_toolbox above already provides
    # /map and the map -> camera_init transform, so we don't need map_server or
    # AMCL. navigation_launch.py runs controller_server, planner_server,
    # behavior_server, bt_navigator, smoother_server, waypoint_follower, and
    # velocity_smoother under lifecycle_manager_navigation.
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': nav2_params_file,
            'autostart': 'true',
            'namespace': '',
            'default_bt_xml_filename': os.path.join(
                get_package_share_directory('nav2_bt_navigator'),
                'behavior_trees',
                'navigate_to_pose_w_replanning_and_recovery.xml'),
        }.items(),
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_fast_lio_config_file,
        declare_nav2_params_file,
        declare_slam_params_file,
        fast_lio_node,
        static_tf_broadcaster,
        pointcloud_to_laserscan_node,
        slam_toolbox_node,
        nav2_launch,
    ])
