import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, LaunchConfigurationEquals, LaunchConfigurationNotEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import AndSubstitution, LaunchConfiguration, PythonExpression
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
        # Debug-only: renders the measured robot + a blue ghost of the MJPC plan to an
        # mp4 (written on shutdown). Reads rt/lowstate + the estimator + rt/mjpc/plan;
        # commands nothing. use_mjpc_viz:=false for a clean bringup.
        DeclareLaunchArgument('use_mjpc_viz', default_value='true'),
        # Which controller drives the legs. 'almi' (default) = the switchable RL
        # controller (h12_lowerbody_rl lowerbody_controller_node): ALMI stands,
        # hands off to the walk policy on /cmd_vel, and follows nav2. 'fame' /
        # 'walk' pin that same RL controller to those policies instead. The RL
        # controller launches for any value != 'mjpc'. NOTE: the MJPC nodes
        # below are currently COMMENTED OUT, so 'mjpc' launches NO leg
        # controller (legs uncontrolled) — the RL controller is the only working
        # sim option right now, which is why the default is 'almi'. Only the SIM
        # bringup defaults to almi; the real-robot bringup still defaults to mjpc
        # behind the start_position_verified interlock.
        DeclareLaunchArgument('model_logging', default_value='true'),
        DeclareLaunchArgument('model_visualization', default_value='true'),
        DeclareLaunchArgument('model_clear_logs', default_value='true'),
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
            arguments=['--config', 'sim_safety_split.yaml'],
            parameters=[sim_time_param],
            output='screen',
        ),

        # vision foundation-model services (gemini + sam, served by model_server)
        Node(
            package='model_server',
            executable='gemini_server',
            name='gemini_server',
            parameters=[sim_time_param, model_log_params],
            output='screen',
        ),
        Node(
            package='model_server',
            executable='sam_server',
            name='sam_server',
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

        # Switchable lower-body RL controller (almi / fame / walk) — launched
        # only when lowerbody:= names an RL policy; the MJPC controller below
        # is gated off in that case. The chosen policy is pinned as
        # active_policy (fame/walk still auto-switch between themselves on
        Node(
            package='h12_lowerbody_rl',
            executable='lowerbody_controller_node',
            name='lowerbody_controller_node',
            parameters=[sim_time_param,
                        {'active_policy':"almi",  # almi/fame/walk
                         # Sim auto-engages after the pre-pose settles (unattended
                         # runs); the operator-confirm arming gate is a REAL-robot
                         # step (see h1_real_desktop_bringup.launch.py).
                         'engage_wait_for_confirm': False,
                         # Sim's IMU has no mounting bias to correct (the live
                         # trim sliders still work for experiments).
                         'imu_offset_roll_deg': 0.0,
                         'imu_offset_pitch_deg': 0.0,
                         'imu_offset_yaw_deg': 0.0}],
            output='screen',
        ),

        # --- MJPC lower-body base estimator (RW-EKF) -> rt/sportmodestate_est ---
        # Node(
        #     package='h12_deploy_mjpc',
        #     executable='estimator_node',
        #     name='h12_deploy_mjpc_estimator',   # MUST match the yaml key
        #     parameters=[
        #         sim_time_param,
        #         os.path.join(get_package_share_directory('h1_bringup'),
        #                     'config', 'mjpc_sim.yaml'),
        #     ],
        #     output='screen',
        # ),

        # # --- MJPC lower-body balance controller (spawns mjpc_lowerbody_core) ---
        # Node(
        #     package='h12_deploy_mjpc',
        #     executable='mjpc_deploy_lowerbody_controller',
        #     name='mjpc_deploy_lowerbody_controller',   # MUST match the yaml key
        #     parameters=[
        #         sim_time_param,
        #         os.path.join(get_package_share_directory('h1_bringup'),
        #                     'config', 'mjpc_sim.yaml'),
        #     ],
        #     # mjpc resolves task model XMLs relative to MJPC_TASKS_DIR; without it
        #     # the model is null -> mj_makeData segfault on startup.
        #     additional_env={'MJPC_TASKS_DIR': '/home/code/mujoco_mpc/build/mjpc/tasks'},
        #     output='screen',
        #     condition=LaunchConfigurationEquals('lowerbody', 'mjpc'),
        # ),

        # --- MJPC plan debug visualizer (measured robot + blue ghost of the plan) ---
        # Subscribes only: rt/lowstate + rt/sportmodestate_est (the estimator above) +
        # rt/mjpc/plan (published by the controller's planner thread). Writes an mp4 on
        # shutdown. Nothing in the control path depends on it.
        # Node(
        #     package='h12_deploy_mjpc',
        #     executable='mjpc_debug_visualizer',
        #     name='mjpc_debug_visualizer',   # MUST match the yaml key
        #     parameters=[
        #         sim_time_param,
        #         os.path.join(get_package_share_directory('h1_bringup'),
        #                     'config', 'mjpc_sim.yaml'),
        #     ],
        #     # MJPC_TASKS_DIR: the viewer loads the SAME staged task XML as the
        #     # controller, so the plan's qpos rows line up with its model.
        #     # MUJOCO_GL=osmesa: offscreen software GL -- needs no GPU, and unlike egl
        #     # it works under both compose and `docker run --gpus all`.
        #     additional_env={'MJPC_TASKS_DIR': '/home/code/mujoco_mpc/build/mjpc/tasks',
        #                     'MUJOCO_GL': 'osmesa'},
        #     output='screen',
        #     # Viz reads rt/mjpc/plan, which only exists when the MJPC
        #     # controller is the active lower body.
        #     condition=IfCondition(AndSubstitution(
        #         LaunchConfiguration('use_mjpc_viz'),
        #         PythonExpression(
        #             ["'", LaunchConfiguration('lowerbody'), "' == 'mjpc'"]),
        #     )),
        # ),


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

