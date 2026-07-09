import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
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

    # MuJoCo publishes /clock with sim time. All nodes should use it so
    # TF lookups and sensor timestamps are coherent with the simulation.
    sim_time_param = {'use_sim_time': False}

    # Per-model debug logging + visualization toggles, shared by the graspgen,
    # gemini, sam, and skills nodes. Both on by default; disable with
    # `model_logging:=false model_visualization:=false` at launch. clear_logs (on
    # by default) wipes each model's dir on startup so every run begins fresh.
    # Output lands in each package's logs/<model>/ (bind-mounted to host).
    model_log_params = {
        'enable_logging': ParameterValue(
            LaunchConfiguration('model_logging'), value_type=bool),
        'enable_visualization': ParameterValue(
            LaunchConfiguration('model_visualization'), value_type=bool),
        'clear_logs': ParameterValue(
            LaunchConfiguration('model_clear_logs'), value_type=bool),
    }

    return LaunchDescription([
        
        # Engage the FAME lower-body standing policy only once the operator has
        # verified the robot is in a safe start position
        # (start_position_verified:=true). Defaults to false so a bare launch
        # never auto-commands the legs on the real robot.
        DeclareLaunchArgument('start_position_verified', default_value='false'),
        DeclareLaunchArgument('use_skills', default_value='true'),
        # Perception + rviz share the machine with the real-time MPC control
        # loop. For a clean stand/balance test (matching the bare-controller
        # runs), set use_vision:=false use_rviz:=false so the GPU vision models
        # (gemini/sam/graspgen) and rviz don't steal cycles from the planner.
        DeclareLaunchArgument('use_vision', default_value='true'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('model_logging', default_value='true'),
        DeclareLaunchArgument('model_visualization', default_value='true'),
        DeclareLaunchArgument('model_clear_logs', default_value='true'),

        # Which lower-body controller drives the legs (both write
        # rt/safety/lowcmd_lower_in, so exactly ONE may run):
        #   mjpc = the MJPC legs-only Stabilize controller (h12_deploy_mjpc),
        #   fame = the FAME RL stand-squat policy (h12_lowerbody_rl).
        # Either way the arms are owned by frame_task_server (split safety),
        # started on the robot-PC bringup. Both are AND-gated with
        # start_position_verified so a bare launch never commands the legs.
        DeclareLaunchArgument('lowerbody', default_value='mjpc'),
        # Param file for the MJPC estimator + controller pair (real values by
        # default; used only when lowerbody:=mjpc).
        DeclareLaunchArgument(
            'mjpc_config',
            default_value=os.path.join(bringup_share, 'config', 'mjpc_real.yaml')),

        # vision foundation-model services (gemini + sam, served by model_server).
        # Gated on use_vision so a stand-only test can drop them (they compete
        # with the control loop for CPU/GPU).
        Node(
            package='model_server',
            executable='gemini_server',
            name='gemini_server',
            parameters=[sim_time_param, model_log_params],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_vision')),
        ),
        Node(
            package='model_server',
            executable='sam_server',
            name='sam_server',
            parameters=[sim_time_param, model_log_params],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_vision')),
        ),

        # graspgen_server + h12_skills: the GraspGenX planning service and the
        # /skill/* action servers. The grasp skill chains gemini -> sam ->
        # graspgen -> frame_task. graspgen_server loads a heavy GPU model, so
        # both are gated on use_skills. The skills node waits ~10s each (non-
        # fatal) on gemini/sam/graspgen, the grippers, and frame_task — the
        # latter two arrive over DDS from the robot/driver bringup.
        Node(
            package='model_server',
            executable='graspgen_server',
            name='graspgen_server',
            parameters=[sim_time_param, model_log_params],
            output='screen',
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration('use_vision'),
                "' == 'true' and '", LaunchConfiguration('use_skills'),
                "' == 'true'"])),
        ),
        Node(
            package='h12_skills',
            executable='skills',
            name='h12_skills',
            parameters=[sim_time_param, model_log_params],
            output='screen',
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration('use_vision'),
                "' == 'true' and '", LaunchConfiguration('use_skills'),
                "' == 'true'"])),
        ),

        # ---- Lower-body controller (select with lowerbody:=mjpc|fame) --------
        # Both publish rt/safety/lowcmd_lower_in, so each is gated on its
        # selector AND start_position_verified -- only one ever commands legs.

        # FAME RL stand-squat policy (h12_lowerbody_rl). Switch behaviours via
        # /lowerbody/start_walk or /lowerbody/set_policy (safe-handover guarded).
        Node(
            package='h12_lowerbody_rl',
            executable='lowerbody_controller_node',
            name='lowerbody_controller_node',
            parameters=[sim_time_param, {'active_policy': 'fame'}],
            output='screen',
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration('start_position_verified'),
                "' == 'true' and '", LaunchConfiguration('lowerbody'),
                "' == 'fame'"])),
        ),

        # MJPC legs-only Stabilize controller (h12_deploy_mjpc): the fork's
        # RW-EKF proprioceptive base estimator feeding the fork's shared deploy
        # core (embedded mjpc::Agent, raw unitree_sdk2 DDS) which drives the 12
        # leg joints; the upper body stays with frame_task IK (split safety).
        # Params in mjpc_config (default mjpc_real.yaml). Same node pair as the
        # sim bringup's use_mjpc_lowerbody group. The estimator OWNS
        # rt/sportmodestate in debug mode -- do NOT also run the laptop
        # base_estimator_node (one-writer rule, see mjpc_real.yaml).
        # Estimator gated on lowerbody ONLY (not start_position_verified): it is
        # READ-ONLY (reads rt/lowstate, publishes rt/sportmodestate) and never
        # commands the robot, so it runs during pre-flight too. Launch with
        # start_position_verified:=false to bring up the state feed and verify it
        # (dds_topic_check -> rt/sportmodestate @200Hz) BEFORE arming the legs.
        Node(
            package='h12_deploy_mjpc',
            executable='estimator_node',
            name='h12_deploy_mjpc_estimator',
            parameters=[sim_time_param, LaunchConfiguration('mjpc_config')],
            output='screen',
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration('lowerbody'), "' == 'mjpc'"])),
        ),
        Node(
            package='h12_deploy_mjpc',
            executable='mjpc_deploy_lowerbody_controller',
            name='mjpc_deploy_lowerbody_controller',
            parameters=[sim_time_param, LaunchConfiguration('mjpc_config')],
            # mjpc resolves task model XMLs relative to MJPC_TASKS_DIR (else
            # <exe>/../mjpc/tasks, which doesn't exist for this ROS binary -> a
            # null model -> mj_makeData segfault). Point it at the runtime-
            # hydrated build tree that matches the linked mjpc libs.
            additional_env={'MJPC_TASKS_DIR': '/home/code/mujoco_mpc/build/mjpc/tasks'},
            output='screen',
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration('start_position_verified'),
                "' == 'true' and '", LaunchConfiguration('lowerbody'),
                "' == 'mjpc'"])),
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2_sim',
            arguments=['-d',default_rviz],
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
                    condition=IfCondition(LaunchConfiguration('use_rviz')),
                ),
            ],
        ),
    ])
