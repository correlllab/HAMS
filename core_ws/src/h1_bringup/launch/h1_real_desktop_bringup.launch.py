import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    TimerAction,
)
from launch.conditions import IfCondition
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

    # --- CPU affinity (soft pin, i7-14700F hybrid: 0-15 P-cores, 16-27 E-cores) ---
    # Give MJPC's lowerbody controller (and the mjpc_lowerbody_core planner it
    # Popens, which inherits this mask) the 12 fast P-core threads 0-11 to itself,
    # sized to plan_threads: 12 in config/mjpc_real.yaml so the planner threadpool
    # fills exactly those cores and nothing else contends.
    #
    # The 200Hz base estimator gets its OWN dedicated P-core (12). It used to share
    # 0-11 with the planner, so when the 12 planner threads saturated those cores the
    # estimator's DDS receive thread starved -> rt/lowstate stopped draining -> frozen
    # velocity + dead-reckoning drift + the controller's state-stale safe-hold. A
    # dedicated core keeps the estimator's receive+filter loop scheduled.
    #
    # Core 13 is reserved for eno1's network-RX IRQ/softirq, which is pinned on the
    # HOST (see README "Network / DDS tuning" step 3 + scripts/pin_net_irq.sh). eno1
    # has a single RX queue, and irqbalance was parking its IRQ on core 11 -> every
    # /lowstate + tf packet was processed in softirq on a core the planner had
    # saturated, so the 500Hz robot state arrived at niraj at ~400Hz with multi-second
    # stalls. Keeping the NIC softirq on its own core, off the RT set (0-12), fixes it.
    #
    # Everything else (vision model servers, rviz) is pushed onto 14-27 so it can't
    # steal a P-core mid-balance-loop or the network core. SOFT reservation: taskset
    # pins these processes but host/kernel threads can still float onto 0-13 (add
    # isolcpus=0-13 to the kernel cmdline for a hard reservation).
    MJPC_CPUS = 'taskset -c 0-11'       # lowerbody controller + planner (plan_threads=12)
    ESTIMATOR_CPUS = 'taskset -c 12'    # dedicated core for the 200Hz base estimator
    OTHER_CPUS = 'taskset -c 14-27'     # vision servers, rviz, everything else (13 = NIC IRQ)

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
        # Which controller drives the legs once start_position_verified:=true.
        # 'mjpc' (default) = the MJPC balance controller below, unchanged
        # behavior. Any other value ('almi' | 'fame' | 'walk') swaps in the
        # switchable RL controller pinned to that policy and gates the MJPC
        # controller off — only one lower-body controller may feed
        # /safety/lowcmd_lower_in. Every leg path stays behind the
        # start_position_verified interlock.
        DeclareLaunchArgument('lowerbody', default_value='mjpc'),
        DeclareLaunchArgument('use_skills', default_value='true'),
        DeclareLaunchArgument('model_logging', default_value='true'),
        DeclareLaunchArgument('model_visualization', default_value='true'),
        DeclareLaunchArgument('model_clear_logs', default_value='true'),

        # vision foundation-model services (gemini + sam, served by model_server)
        Node(
            package='model_server',
            executable='gemini_server',
            name='gemini_server',
            prefix=OTHER_CPUS,
            parameters=[sim_time_param, model_log_params],
            output='screen',
        ),
        Node(
            package='model_server',
            executable='sam_server',
            name='sam_server',
            prefix=OTHER_CPUS,
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
            prefix=OTHER_CPUS,
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
            prefix=OTHER_CPUS,
            parameters=[sim_time_param, model_log_params],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_skills')),
        ),
        Node(
            package='h12_skills',
            executable='skills',
            name='h12_skills',
            prefix=OTHER_CPUS,
            parameters=[sim_time_param, model_log_params],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_skills')),
        ),

        # Switchable lower-body RL controller (almi / fame / walk) — launched
        # only when lowerbody:= names an RL policy AND the operator has verified
        # the start position; the MJPC controller below is gated off in that
        # case. The chosen policy is pinned as active_policy (fame/walk still
        # auto-switch between themselves on /cmd_vel; almi never auto-switches —
        # it stands AND walks itself). Runs on the MJPC P-core set, which is
        # free whenever this node is selected instead of MJPC.
        Node(
            package='h12_lowerbody_rl',
            executable='lowerbody_controller_node',
            name='lowerbody_controller_node',
            prefix=MJPC_CPUS,
            parameters=[sim_time_param,
                        {'active_policy': LaunchConfiguration('lowerbody')}],
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
            prefix=ESTIMATOR_CPUS,
            parameters=[
                sim_time_param,
                os.path.join(get_package_share_directory('h1_bringup'),
                            'config', 'mjpc_real.yaml'),
            ],
            output='screen',
            condition=IfCondition(LaunchConfiguration('start_position_verified')),

        ),

        # --- MJPC lower-body balance controller (spawns mjpc_lowerbody_core) ---
        Node(
            package='h12_deploy_mjpc',
            executable='mjpc_deploy_lowerbody_controller',
            name='mjpc_deploy_lowerbody_controller',   # MUST match the yaml key
            # taskset pins THIS launcher; the mjpc_lowerbody_core it Popens inherits
            # the 0-11 affinity mask (see controller_launcher.py). plan_threads (yaml)
            # caps its planner threadpool to match.
            prefix=MJPC_CPUS,
            parameters=[
                sim_time_param,
                os.path.join(get_package_share_directory('h1_bringup'),
                            'config', 'mjpc_real.yaml'),
            ],
            # mjpc resolves task model XMLs relative to MJPC_TASKS_DIR; without it
            # the model is null -> mj_makeData segfault on startup.
            additional_env={'MJPC_TASKS_DIR': '/home/code/mujoco_mpc/build/mjpc/tasks'},
            output='screen',
            condition=IfCondition(AndSubstitution(
                LaunchConfiguration('start_position_verified'),
                PythonExpression(
                    ["'", LaunchConfiguration('lowerbody'), "' == 'mjpc'"]),
            )),
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2_sim',
            prefix=OTHER_CPUS,
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
