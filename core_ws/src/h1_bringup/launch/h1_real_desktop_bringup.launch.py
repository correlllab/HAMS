import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
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
    # Companion-desktop bringup: runs the model_server vision services, the
    # /skill/* servers, and the MJPC lower-body controller/estimator, then a
    # single rviz with sim.rviz. The state publishers and frame_task_server run
    # on the onboard PC (h1_real_controller.launch.py), not here.
    bringup_share = get_package_share_directory('h1_bringup')
    default_rviz = os.path.join(bringup_share, 'rviz', 'sim.rviz')

    # Real robot runs on wall-clock time (no MuJoCo /clock), so use_sim_time is
    # False; all nodes below share this parameter.
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

    # h12_skills-only: per-run base-sway telemetry (run_telemetry.py). Each
    # grasp / pick_place run writes its own logs/runs/<skill>/<stamp>/ with a
    # 10 Hz telemetry.csv + result.json (pelvis-from-odometry sway, mirroring
    # the fridge study). On by default; disable with `record_runs:=false`.
    skills_params = {
        'record_runs': ParameterValue(
            LaunchConfiguration('record_runs'), value_type=bool),
    }

    return LaunchDescription([
        
        # Engage the FAME lower-body standing policy only once the operator has
        # verified the robot is in a safe start position
        # (start_position_verified:=true). Defaults to false so a bare launch
        # never auto-commands the legs on the real robot.
        DeclareLaunchArgument('start_position_verified', default_value='false'),
        # Which controller drives the legs once start_position_verified:=true.
        # 'almi' (default) = the switchable RL controller pinned to the ALMI
        # policy; 'fame' | 'walk' pin the other RL policies; 'mjpc' restores
        # the MJPC balance controller and gates the RL node off — only one
        # lower-body controller may feed /safety/lowcmd_lower_in. Every leg
        # path stays behind the start_position_verified interlock, and the RL
        # node additionally holds at the pre-pose crouch until the operator
        # confirms:  ros2 service call /lowerbody/confirm_engage std_srvs/srv/Trigger
        DeclareLaunchArgument('lowerbody', default_value='almi'),
        DeclareLaunchArgument('use_skills', default_value='true'),
        DeclareLaunchArgument('model_logging', default_value='true'),
        DeclareLaunchArgument('model_visualization', default_value='true'),
        DeclareLaunchArgument('model_clear_logs', default_value='true'),
        DeclareLaunchArgument('record_runs', default_value='true'),

        # Both hand/wrist RealSense cameras and their camera->link static TFs.
        # These live on the companion desktop (the wrists' USB runs here), while
        # the head camera comes up with the onboard driver bringup
        # (h1_real_drivers.launch.py). Equivalent of
        # `ros2 launch cl_realsense h12_hand_cameras.launch.py`.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('cl_realsense'),
                    'launch', 'h12_hand_cameras.launch.py')
            )
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

        # yolo_server: YOLO-World open-vocabulary detection publisher. Subscribes
        # to the head + both hand color cameras (its DEFAULT_IMAGE_TOPICS) and
        # publishes a DetectionBundle on <image_topic>/detections at 5 Hz per cam,
        # using the fine-tuned battery weights (weights/yolo_world_battery_best.pt).
        # Detects the battery-workcell classes by default (DEFAULT_QUERIES: Bolt,
        # BusBar, InteriorScrew, Nut, OrageCove, Screw, ScrewHole). The vocabulary
        # is read live from the `queries` param — retune at runtime, e.g.:
        #   ros2 param set /yolo_server queries "['Bolt','Nut','Screw']"
        Node(
            package='model_server',
            executable='yolo_server',
            name='yolo_server',
            parameters=[sim_time_param, model_log_params],
            output='screen',
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
            condition=IfCondition(LaunchConfiguration('use_skills')),
        ),
        Node(
            package='h12_skills',
            executable='skills',
            name='h12_skills',
            parameters=[sim_time_param, model_log_params, skills_params],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_skills')),
        ),

        # Switchable lower-body RL controller (almi / fame / walk) — launched
        # only when lowerbody:= names an RL policy AND the operator has verified
        # the start position; the MJPC controller below is gated off in that
        # case. The chosen policy is pinned as active_policy (fame/walk still
        # auto-switch between themselves on /cmd_vel; almi never auto-switches —
        # it stands AND walks itself).
        Node(
            package='h12_lowerbody_rl',
            executable='lowerbody_controller_node',
            name='lowerbody_controller_node',
            parameters=[sim_time_param,
                        {'active_policy': LaunchConfiguration('lowerbody'),
                         # The elastic band is a SIM fixture; on real there is
                         # no /elastic_band/toggle service, so the release path
                         # only costs a 1 s blocking wait_for_service inside the
                         # engage tick (first policy command then computed from
                         # 1 s-stale state) plus a redundant second LSTM reset.
                         # The operator lowers the gantry instead.
                         'disable_elastic_band': False,
                         # Startup seed for the LIVE IMU trim (adjust during a
                         # run with rqt_reconfigure sliders or ros2 param set).
                         # Negative pitch = anti-lean bias (policy believes it
                         # pitches further forward and fights harder); the
                         # hanging-plumb static calibration would be +0.264.
                         'imu_offset_roll_deg': 0.0,
                         'imu_offset_pitch_deg': -2.5,
                         'imu_offset_yaw_deg': 0.0}],
            output='screen',
            condition=IfCondition(AndSubstitution(
                LaunchConfiguration('start_position_verified'),
                PythonExpression(
                    ["'", LaunchConfiguration('lowerbody'), "' != 'mjpc'"]),
            )),
        ),


        Node(
            package='h12_deploy_mjpc',
            executable='estimator_node',
            name='h12_deploy_mjpc_estimator',   # MUST match the yaml key
            parameters=[
                sim_time_param,
                os.path.join(get_package_share_directory('h1_bringup'),
                            'config', 'mjpc_real.yaml'),
            ],
            # 'log' (not 'screen') mutes the node's periodic [est] stdout from the
            # console — it still goes to the ROS log files.
            output='log',
            condition=IfCondition(LaunchConfiguration('start_position_verified')),

        ),

        # # --- MJPC lower-body balance controller (spawns mjpc_lowerbody_core) ---
        # Node(
        #     package='h12_deploy_mjpc',
        #     executable='mjpc_deploy_lowerbody_controller',
        #     name='mjpc_deploy_lowerbody_controller',   # MUST match the yaml key
        #     parameters=[
        #         sim_time_param,
        #         os.path.join(get_package_share_directory('h1_bringup'),
        #                     'config', 'mjpc_real.yaml'),
        #     ],
        #     # mjpc resolves task model XMLs relative to MJPC_TASKS_DIR; without it
        #     # the model is null -> mj_makeData segfault on startup.
        #     additional_env={'MJPC_TASKS_DIR': '/home/code/mujoco_mpc/build/mjpc/tasks'},
        #     output='screen',
        #     condition=IfCondition(AndSubstitution(
        #         LaunchConfiguration('start_position_verified'),
        #         PythonExpression(
        #             ["'", LaunchConfiguration('lowerbody'), "' == 'mjpc'"]),
        #     )),
        # ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2_sim',
            arguments=['-d',default_rviz],
            parameters=[sim_time_param],
            output='screen',
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
        #         ),
        #     ],
        # ),
    ])
