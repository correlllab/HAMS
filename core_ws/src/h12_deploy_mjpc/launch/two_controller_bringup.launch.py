#!/usr/bin/env python3
"""Two-controller handoff bringup (piece 1/3: the launch that ties it together).

Brings up ONE safety layer (split_mode, UNTOUCHED code) fed by the SELECTOR mux,
with BOTH control stacks warm from the start so the live swap is a pointer flip:

  producers                              selector            safety(split, two_controller yaml)
  ---------                              --------            -----------------------------------
  fullbody mjpc_fullbody_core  -> lowcmd_in ==[A]==\\
     (Lean, drive strategy; drives the walk)         \\
  lowerbody mjpc_lowerbody_core-> lowcmd_lower_in =[B]=> selector => lowcmd_lower_sel -> low_cmd_lower_in
     (Stabilize stand-6, WARM, NOT selected yet)      /                lowcmd_upper_sel -> low_cmd_upper_in
  frame_task_server           -> lowcmd_upper_in =[B]=/
     (arms IK, WARM, inits at measured arm pose)

Flow: gantry-assisted power-on takeover on the FULLBODY only; remove gantry;
drive with the whole-body controller (teleop is launched separately, see
enable_teleop hook); park in a stand; then the operator runs the ORCHESTRATOR
(`ros2 run h12_deploy_mjpc orchestrator_node`) which flips arms A->B, then legs
A->B, then releases the fullbody. One-way.

Everything here is HAMS-side: the mujoco_mpc deploy C++ is UNMODIFIED (the
fullbody publishes its normal full lowcmd_in; the selector splits it), and the
safety layer code is UNMODIFIED (only its two split INPUT topics point at the
selector via two_controller_safety_split.yaml, which also carries the
locomotion-permissive knee velocity limit).

SAFETY: real-robot runs only. Set network_interface to the robot NIC. Both MJPC
planners run concurrently during the whole drive+handoff, so plan_threads is
capped per core; the fullbody is released right after the leg flip to free it.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from h12_ros2_controller.utility.path_definition_ros import URDF_MAGPIE_ROS_PATH

CTRL_PKG = 'h12_ros2_controller'
DEPLOY_PKG = 'h12_deploy_mjpc'


def generate_launch_description():
    iface = LaunchConfiguration('network_interface')
    domain = LaunchConfiguration('domain_id')
    sportstate = LaunchConfiguration('sportstate_topic')
    gravity_ff = LaunchConfiguration('gravity_ff')

    safety_config = LaunchConfiguration('safety_config')
    frametask_config = LaunchConfiguration('frametask_config')

    fb_task = LaunchConfiguration('fullbody_task')
    fb_strategy = LaunchConfiguration('fullbody_strategy')
    fb_threads = LaunchConfiguration('fullbody_plan_threads')
    lb_task = LaunchConfiguration('lowerbody_task')
    lb_strategy = LaunchConfiguration('lowerbody_strategy')
    lb_threads = LaunchConfiguration('lowerbody_plan_threads')

    init_lower = LaunchConfiguration('init_lower')
    init_upper = LaunchConfiguration('init_upper')
    selector_hz = LaunchConfiguration('selector_hz')
    selector_stale = LaunchConfiguration('selector_stale_sec')

    with open(URDF_MAGPIE_ROS_PATH, 'r') as f:
        robot_description = f.read()

    args = [
        # --- shared plumbing ---
        DeclareLaunchArgument('network_interface', default_value='',
                              description='Robot NIC for DDS (set for real hardware; '
                                          "'' = unitree_sdk2 auto-pin)"),
        DeclareLaunchArgument('domain_id', default_value='0'),
        DeclareLaunchArgument('sportstate_topic', default_value='rt/sportmodestate_est',
                              description='Base-state estimator output; BOTH cores read it'),
        DeclareLaunchArgument('gravity_ff', default_value='0.85'),
        # --- safety + arms configs (suffixes MUST both be _split) ---
        DeclareLaunchArgument('safety_config', default_value='two_controller_safety_split',
                              description='split safety yaml: topics->selector, knee vel 0.72'),
        DeclareLaunchArgument('frametask_config', default_value='safety_split',
                              description='frame_task_server controller config (_split)'),
        # --- fullbody (drive) ---
        DeclareLaunchArgument('fullbody_task', default_value='Lean H12 Magpie'),
        DeclareLaunchArgument('fullbody_strategy', default_value='24',
                              description='24 = WSS drive (stand<->trot on cmd_vel); 23 = trot'),
        DeclareLaunchArgument('fullbody_plan_threads', default_value='8',
                              description='cap: two MJPC planners overlap during handoff'),
        # --- lowerbody (stand-6, warm) ---
        DeclareLaunchArgument('lowerbody_task', default_value='Stabilize H12 Magpie'),
        DeclareLaunchArgument('lowerbody_strategy', default_value='6'),
        DeclareLaunchArgument('lowerbody_plan_threads', default_value='6'),
        # --- selector ---
        DeclareLaunchArgument('init_lower', default_value='A'),
        DeclareLaunchArgument('init_upper', default_value='A'),
        DeclareLaunchArgument('selector_hz', default_value='500.0'),
        DeclareLaunchArgument('selector_stale_sec', default_value='0.1'),
        # --- optional drive teleop hook (launch your WASD/cmd_vel bridge here) ---
        DeclareLaunchArgument('enable_teleop', default_value='false',
                              description='placeholder for the WASD drive teleop; wire your '
                                          'existing cmd_vel bridge into the block below'),
    ]

    # ---- t=0: state publishers + estop + base estimator ----
    base = [
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             name='robot_state_publisher',
             parameters=[{'robot_description': robot_description}], output='screen'),
        Node(package=CTRL_PKG, executable='joint_state_publisher',
             name='joint_state_publisher', output='screen'),
        Node(package='estop', executable='estop_node', name='estop_node', output='screen'),
        Node(package=DEPLOY_PKG, executable='estimator_node', name='base_estimator',
             parameters=[{'domain_id': domain, 'iface': iface}], output='screen'),
    ]

    # ---- t=5: safety layer (split_mode) reading the selector's *_sel outputs ----
    safety = TimerAction(period=5.0, actions=[
        Node(package='h12_safety_layer', executable='safety_node', name='safety_node',
             arguments=['--config', safety_config], output='screen'),
    ])

    # ---- t=5: frame_task_server -> lowcmd_upper_in  (B upper, WARM at measured pose) ----
    frametask = TimerAction(period=5.0, actions=[
        Node(package=CTRL_PKG, executable='frame_task_server', name='frame_task_server',
             arguments=['--config', frametask_config], output='screen'),
    ])

    # ---- t=6: selector mux (both channels init on A = fullbody) ----
    selector = TimerAction(period=6.0, actions=[
        Node(package=DEPLOY_PKG, executable='selector_node', name='two_controller_selector',
             arguments=[
                 '--domain', domain, '--iface', iface,
                 '--publish-hz', selector_hz, '--stale-sec', selector_stale,
                 '--init-lower', init_lower, '--init-upper', init_upper,
             ], output='screen'),
    ])

    # ---- t=8: fullbody core (A) -> lowcmd_in  (drives; gantry-assisted cold takeover) ----
    fullbody = TimerAction(period=8.0, actions=[
        Node(package=DEPLOY_PKG, executable='mjpc_deploy_lowerbody_controller',
             name='mjpc_fullbody', parameters=[{
                 'controller_exe': 'mjpc_fullbody_core',
                 'task': fb_task, 'strategy': fb_strategy,
                 'gravity_ff': gravity_ff, 'sportstate_topic': sportstate,
                 'network_interface': iface, 'domain_id': domain,
                 'plan_threads': fb_threads,
                 'has_arm_aware': False,   # fullbody owns the arms; no arm_aware flag
                 'drop_band': False,       # real robot: no sim elastic band
             }], output='screen'),
    ])

    # ---- t=8: lowerbody core (B lower) -> lowcmd_lower_in  (Stabilize stand-6, WARM) ----
    # NO align_start: the robot is already standing on the fullbody, so this core
    # just warm-plans stand-6 from the MEASURED pose; when the leg flip lands its
    # plan is already tracking reality -> smooth ramp, not a step.
    lowerbody = TimerAction(period=8.0, actions=[
        Node(package=DEPLOY_PKG, executable='mjpc_deploy_lowerbody_controller',
             name='mjpc_lowerbody', parameters=[{
                 'controller_exe': 'mjpc_lowerbody_core',
                 'task': lb_task, 'strategy': lb_strategy,
                 'gravity_ff': gravity_ff, 'sportstate_topic': sportstate,
                 'network_interface': iface, 'domain_id': domain,
                 'plan_threads': lb_threads,
                 'has_arm_aware': True, 'arm_aware': True,
                 'drop_band': False,
             }], output='screen'),
    ])

    # ---- optional: WASD / cmd_vel drive teleop for the fullbody (bring your own) ----
    # Left as a hook: enable_teleop:=true and add your teleop+bridge Node here.
    teleop = TimerAction(period=9.0, condition=IfCondition(LaunchConfiguration('enable_teleop')),
                         actions=[
        # Node(package='<your_teleop_pkg>', executable='<wasd_cmd_vel_bridge>', ...),
    ])

    return LaunchDescription(
        args + base + [safety, frametask, selector, fullbody, lowerbody, teleop]
    )
