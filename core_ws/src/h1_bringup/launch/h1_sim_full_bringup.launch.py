"""RoboCasa-sim bringup for the WHOLE-BODY MPC stack.

The sim sibling of h1_real_desktop_full_bringup.launch.py, and the full-body
sibling of h1_sim_bringup.launch.py. That one runs the SPLIT architecture (the
h12_lowerbody_rl controller for the legs + frame_task_server for the arms,
safety layer in sim_safety_split.yaml). Our unified whole-body MPC
(mjpc_fullbody_core, nu=27) owns the legs AND the arms and publishes a SINGLE
full-body channel rt/safety/lowcmd_in, so the split-mode safety layer would
ignore it entirely and the robot would not move.

Differences vs h1_sim_bringup.launch.py -- everything else is kept identical:
  * safety_node runs in FULL-BODY mode (sim_safety_full.yaml -> reads
    rt/safety/lowcmd_in, the channel mjpc_fullbody_core drives).
  * frame_task_server is DROPPED -- the whole-body MPC owns the arms.
  * the h12_lowerbody_rl controller is DROPPED -- it owns the legs, and only
    one writer may feed the safety layer.
  * ADDED: estimator_node (rt/sportmodestate_est) + the MPC core.

The split sim bringup is untouched.

    # terminal 1: the RoboCasa plant (USER-RUN -- see docker/ docs)
    # terminal 2:
    ros2 launch h1_bringup h1_sim_full_bringup.launch.py

⚠ ROS_DOMAIN_ID must be exported in the launching shell; the safety layer's DDS
half (unitree_sdk2py ChannelSubscriber) and the rclpy half must share a domain.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ASSETS_DIR = '/home/code/CL_Assets'
# mjpc resolves task model XMLs relative to MJPC_TASKS_DIR; without it the model
# is null -> mj_makeData segfault on startup.
MJPC_TASKS_DIR = '/home/code/mujoco_mpc/build/mjpc/tasks'


def generate_launch_description():
    bringup_share = get_package_share_directory('h1_bringup')
    default_rviz = os.path.join(bringup_share, 'rviz', 'sim.rviz')
    mjpc_sim_yaml = os.path.join(bringup_share, 'config', 'mjpc_sim.yaml')

    with open(os.path.join(ASSETS_DIR, 'ros_assets', 'h1_2_magpie_ros.urdf'), 'r') as urdf_file:
        robot_description = urdf_file.read()

    # MuJoCo publishes /clock with sim time. All nodes should use it so TF
    # lookups and sensor timestamps are coherent with the simulation.
    sim_time_param = {'use_sim_time': True}

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
        DeclareLaunchArgument('use_nav', default_value='true'),
        # DEFAULT FALSE (differs from the split sim bringup, which defaults true):
        # h12_skills chains gemini -> sam -> graspgen -> frame_task, and waits on
        # the frame_task ACTION SERVER on startup. This bringup does not launch
        # frame_task_server (the whole-body MPC owns the arms), so the skills node
        # would wait forever. The real full bringup solves this with
        # graspgen_pose_adapter + grasp_orchestrator (gRPC straight to the MPC);
        # that seam is not wired for sim yet, so skills stay off here.
        DeclareLaunchArgument('use_skills', default_value='false'),
        # DEFAULT FALSE (differs from the split sim bringup): debug_visualizer_node
        # HARDCODES the Stabilize task XML (debug_visualizer_node.py:43,
        # humanoid_bench/stabilize/Stabilize_H12_Magpie.xml). That model is the
        # legs-only nu=12 build, so against this nu=27 full-body core the plan's
        # qpos rows do not line up and the blue ghost is posed WRONG. Only set
        # this true once the viewer takes the task as a parameter.
        DeclareLaunchArgument('use_mjpc_viz', default_value='false'),
        DeclareLaunchArgument('model_logging', default_value='true'),
        DeclareLaunchArgument('model_visualization', default_value='true'),
        DeclareLaunchArgument('model_clear_logs', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz),
        # Whole-body MPC task knobs. 'Lean H12 Magpie' is the nu=27 full-body
        # task (strategy 6 = h12_simple_stand); 'Grasp H12 Magpie' is the grasp
        # sibling the real full bringup defaults to. The binary is hardwired
        # nu=27, so only full-body tasks are valid here.
        DeclareLaunchArgument('task', default_value='Lean H12 Magpie'),
        DeclareLaunchArgument('strategy', default_value='6'),

        nav_launch,

        # The MuJoCo bridge back-projects depth into 3D using REP-103 optical
        # convention but stamps the messages with the optical frame name below.
        # The URDF defines only camera_link, so without this static TF the vision
        # pipeline rotates every detection ~90 deg out of place.
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

        # NOTE: frame_task_server is intentionally NOT launched. The whole-body
        # MPC commands all 27 joints (arms included); a live frame_task_server
        # would only publish on the split upper channel, which the full_body_mode
        # safety layer ignores anyway -- and two writers is the bug we avoid.

        # vision foundation-model services (identical to h1_sim_bringup)
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

        # FULL-BODY safety config: full_body_mode reading rt/safety/lowcmd_in
        # (the split sim bringup uses sim_safety_split.yaml here).
        Node(
            package='h12_safety_layer',
            executable='safety_node',
            name='safety_node',
            parameters=[sim_time_param],
            arguments=['--config', 'sim_safety_full.yaml'],
            output='screen',
        ),

        # NOTE: h12_lowerbody_rl/lowerbody_controller_node is intentionally NOT
        # launched -- the whole-body MPC owns the legs. Only one controller may
        # feed the safety layer.

        # --- MJPC base estimator (RW-EKF) -> rt/sportmodestate_est -------------
        # Plant-clocked (tick_dt 0.005) in mjpc_sim.yaml: RoboCasa runs far below
        # realtime, and the wall-clocked default over-integrates odometry by 1/RTF.
        Node(
            package='h12_deploy_mjpc',
            executable='estimator_node',
            name='h12_deploy_mjpc_estimator',   # MUST match the yaml key
            parameters=[sim_time_param, mjpc_sim_yaml],
            output='screen',
        ),

        # --- WHOLE-BODY MPC controller (spawns mjpc_fullbody_core, nu=27) ------
        # Run through controller_launcher (the rclpy shim), NOT a bare
        # ExecuteProcess as the real full bringup uses. The shim gives us the
        # ROS-param interface, SIGINT forwarding to the core's damping safe-hold,
        # and the LD_LIBRARY_PATH fix that keeps libddsc/libddscxx resolving from
        # unitree_sdk2's bundled CycloneDDS (mixed ROS/SDK builds corrupt the heap
        # at ChannelFactoryInitialize).
        #
        # The node NAME is deliberately the lower-body one so this inherits every
        # RoboCasa-tuned value already benched in mjpc_sim.yaml's
        # mjpc_deploy_lowerbody_controller block -- twin_dt 0.005 (MUST equal
        # RoboCasa's model.opt.timestep), gravity_ff 1.0 (sim; real keeps 0.85),
        # sportstate_topic rt/sportmodestate_est, latency_rtf, stale_sec 1.5, and
        # drop_band false (the plant owns band release via --band-auto-release;
        # the launcher's timer release is a stateful TOGGLE and would re-arm it).
        # Only the full-body-specific knobs are overridden below -- parameters
        # later in the list win.
        Node(
            package='h12_deploy_mjpc',
            executable='mjpc_deploy_lowerbody_controller',
            name='mjpc_deploy_lowerbody_controller',   # MUST match the yaml key
            parameters=[
                sim_time_param,
                mjpc_sim_yaml,
                {
                    # which compiled core to exec (default is mjpc_lowerbody_core)
                    'controller_exe': 'mjpc_fullbody_core',
                    # --arm_aware is a LOWER-BODY-only flag; the full-body core
                    # owns the arms and does not define it, so passing it would
                    # abort on an unknown flag. This suppresses it.
                    'has_arm_aware': False,
                    'task': ParameterValue(
                        LaunchConfiguration('task'), value_type=str),
                    'strategy': ParameterValue(
                        LaunchConfiguration('strategy'), value_type=int),
                },
            ],
            additional_env={'MJPC_TASKS_DIR': MJPC_TASKS_DIR},
            output='screen',
        ),

        # --- MJPC plan debug visualizer (measured robot + blue ghost) ----------
        # OFF by default -- see the use_mjpc_viz declaration above: the viewer
        # hardcodes the nu=12 Stabilize XML and cannot pose an nu=27 plan.
        Node(
            package='h12_deploy_mjpc',
            executable='mjpc_debug_visualizer',
            name='mjpc_debug_visualizer',   # MUST match the yaml key
            parameters=[sim_time_param, mjpc_sim_yaml],
            # MUJOCO_GL=osmesa: offscreen software GL -- needs no GPU, and unlike
            # egl it works under both compose and `docker run --gpus all`.
            additional_env={'MJPC_TASKS_DIR': MJPC_TASKS_DIR,
                            'MUJOCO_GL': 'osmesa'},
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_mjpc_viz')),
        ),

        # graspgen_server: GraspGenX 6-DOF grasp-planning service. Heavy GPU
        # model, so it is gated with the skills that use it.
        Node(
            package='model_server',
            executable='graspgen_server',
            name='graspgen_server',
            parameters=[sim_time_param, model_log_params],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_skills')),
        ),
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
    ])
