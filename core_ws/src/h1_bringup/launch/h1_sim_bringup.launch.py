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
from launch_ros.parameter_descriptions import ParameterValue

# IMPORTANT: ROS_DOMAIN_ID must be exported in the launching shell. The safety
# layer uses unitree_sdk2py's DDS ChannelSubscriber (which honours
# $ROS_DOMAIN_ID and falls back to the YAML's network.domain_id only if the env
# is unset), while walking_node / frame_task_server / mujoco_ros_bridge are
# rclpy nodes that pick up the env directly. If the variable is missing, the
# DDS half and the rclpy half can end up on different domains and the safety
# layer will see no commands.
ASSETS_DIR = '/home/code/CL_Assets'


def generate_launch_description():
    # The included h12_ros2_controller/full_launch.py starts its own rviz2. We
    # can't patch that upstream package, so this bringup inlines its non-rviz
    # nodes (plus the model_server vision services) and runs a single rviz with sim.rviz.
    bringup_share = get_package_share_directory('h1_bringup')
    default_rviz = os.path.join(bringup_share, 'rviz', 'sim.rviz')

    with open(os.path.join(ASSETS_DIR, 'ros_assets', 'h1_2_magpie_ros.urdf'), 'r') as urdf_file:
        robot_description = urdf_file.read()

    # MuJoCo publishes /clock with sim time. All nodes should use it so
    # TF lookups and sensor timestamps are coherent with the simulation.
    sim_time_param = {'use_sim_time': True}

    # Per-model debug logging + visualization toggles, shared by the graspgen,
    # gemini, sam, and skills nodes. Both on by default; disable with
    # `model_logging:=false model_visualization:=false` at launch. clear_logs (on
    # by default) wipes each model's dir on startup so every run begins fresh.
    # Output lands in each package's logs/<model>/ (bind-mounted, persists on host).
    model_log_params = {
        'enable_logging': ParameterValue(
            LaunchConfiguration('model_logging'), value_type=bool),
        'enable_visualization': ParameterValue(
            LaunchConfiguration('model_visualization'), value_type=bool),
        'clear_logs': ParameterValue(
            LaunchConfiguration('model_clear_logs'), value_type=bool),
    }

    nav_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'h1_navigation.launch.py')
        ),
        launch_arguments={'use_sim_time': 'true'}.items(),
        condition=IfCondition(LaunchConfiguration('use_nav')),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('use_sliders', default_value='true'),
        DeclareLaunchArgument('use_nav', default_value='true'),
        DeclareLaunchArgument('use_skills', default_value='true'),
        # MJPC lower-body estimator + balance controller. On by default for a
        # normal bringup; the grasp benchmark passes use_mjpc:=false — the robot
        # is held by the elastic-band tether and the upper-body frame_task IK is
        # independent of the lower body, so skipping MJPC frees RAM/GPU/CPU.
        DeclareLaunchArgument('use_mjpc', default_value='true'),
        DeclareLaunchArgument('model_logging', default_value='true'),
        DeclareLaunchArgument('model_visualization', default_value='true'),
        DeclareLaunchArgument('model_clear_logs', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz),

        nav_launch,

        # NOTE: the head camera's optical frame (pelvis -> camera_color_optical_frame)
        # is published by the MuJoCo sim bridge directly from the MJCF head_cam
        # pose (exact by construction, since it's the same camera that renders
        # the depth we back-project). A static transform here (previously
        # head_camera_link/camera_link -> camera_color_optical_frame) used the
        # URDF head-camera pose, which does NOT match the MJCF camera and mis-
        # placed every head-cam cloud by ~0.8 m; it is intentionally removed so
        # the sim-published frame is the single source of truth. (A real-robot
        # bringup, where the RealSense driver owns this frame, is unaffected.)

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
            arguments=['--config', 'sim_safety_split.yaml'],
            parameters=[sim_time_param],
            output='screen',
        ),

        # vision foundation-model services (gemini + sam, served by model_server)
        Node(
            package='model_server',
            executable='gemini_server',
            name='gemini_server',
            # GT-perception sweeps never call this server; skipping saves the
            # model load + steady-state CPU/GPU (HAMS_SKIP_VLM_SERVERS=1).
            condition=IfCondition('0' if os.environ.get('HAMS_SKIP_VLM_SERVERS', '0') == '1' else '1'),
            parameters=[sim_time_param, model_log_params],
            output='screen',
        ),
        Node(
            package='model_server',
            executable='sam_server',
            name='sam_server',
            # GT-perception sweeps never call this server; skipping saves the
            # model load + steady-state CPU/GPU (HAMS_SKIP_VLM_SERVERS=1).
            condition=IfCondition('0' if os.environ.get('HAMS_SKIP_VLM_SERVERS', '0') == '1' else '1'),
            parameters=[sim_time_param, model_log_params],
            output='screen',
        ),

        Node(
            package='h12_safety_layer',
            executable='safety_node',
            name='safety_node',
            parameters=[sim_time_param, ],
            arguments=['--config', "sim_safety_split.yaml"],
            output='screen',
        ),

        # Switchable lower-body RL controller (walk / FAME stand-squat).
        # Auto-engages the FAME standing policy; switch via /lowerbody/start_walk
        # or /lowerbody/set_policy (waits for a safe handover before committing).
        # Node(
        #     package='h12_lowerbody_rl',
        #     executable='lowerbody_controller_node',
        #     name='lowerbody_controller_node',
        #     parameters=[sim_time_param, {'active_policy': 'fame'}],
        #     output='screen',
        # ),

        # --- MJPC lower-body base estimator (RW-EKF) -> rt/sportmodestate_est ---
        Node(
            package='h12_deploy_mjpc',
            executable='estimator_node',
            name='h12_deploy_mjpc_estimator',   # MUST match the yaml key
            parameters=[
                sim_time_param,
                os.path.join(get_package_share_directory('h1_bringup'),
                            'config', 'mjpc_sim.yaml'),
            ],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_mjpc')),
        ),

        # --- MJPC lower-body balance controller (spawns mjpc_lowerbody_core) ---
        Node(
            package='h12_deploy_mjpc',
            executable='mjpc_deploy_lowerbody_controller',
            name='mjpc_deploy_lowerbody_controller',   # MUST match the yaml key
            parameters=[
                sim_time_param,
                os.path.join(get_package_share_directory('h1_bringup'),
                            'config', 'mjpc_sim.yaml'),
            ],
            # mjpc resolves task model XMLs relative to MJPC_TASKS_DIR; without it
            # the model is null -> mj_makeData segfault on startup.
            additional_env={'MJPC_TASKS_DIR': '/home/code/mujoco_mpc/build/mjpc/tasks'},
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_mjpc')),
        ),


        # graspgen_server: GraspGenX 6-DOF grasp-planning service (`graspgen`).
        # Heavy GPU model (checkpoints + magpie gripper description), so it's
        # gated with the skills that use it; the grasp skill chains
        # gemini -> sam -> graspgen -> frame_task.
        Node(
            package='model_server',
            executable='graspgen_server',
            name='graspgen_server',
            parameters=[sim_time_param, model_log_params],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_skills')),
        ),

        # h12_skills: serves the /skill/* atomic-skill actions (open_door,
        # grasp, pick_place, ...). On startup it waits ~10s each on the vision
        # pipeline + graspgen services, the grippers, and the frame_task
        # action server (all started above), then idles ready for goals.
        Node(
            package='h12_skills',
            executable='skills',
            name='h12_skills',
            parameters=[sim_time_param, model_log_params],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_skills')),
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
        # TimerAction(
        #     period=1.0,
        #     actions=[
        #         Node(
        #             package='h1_bringup',
        #             executable='slider_debugger.py',
        #             name='slider_debugger',
        #             output='screen',
        #             condition=IfCondition(LaunchConfiguration('use_sliders')),
        #         ),
        #     ],
        # ),
    ])
