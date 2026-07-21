"""Companion-desktop bringup for the WHOLE-BODY MPC grasp stack.

The full-body sibling of h1_real_desktop_bringup.launch.py. That one runs the
SPLIT control architecture (an RL/ALMI lower-body controller for the legs +
frame_task_server for the arms, chained by the h12_skills action servers). This
one runs the UNIFIED whole-body MPC: `mjpc_fullbody_core` (nu=27) owns the legs
AND the arms, so there is no separate leg controller and no FrameTask.

Kept IDENTICAL to the split bringup: the perception services
(gemini/sam/yolo/graspgen from model_server) + rviz.
Dropped: h12_skills node + lowerbody RL controller + sim2real monitor.
Added:   estimator_node + mjpc_fullbody_core + graspgen_pose_adapter +
         grasp_orchestrator (the grasp seam that reuses h12_skills' perception
         helpers in-process and streams the target to the MPC over gRPC).

Run alongside the ROBOT-side FULL bringup (h1_real_robot_full_bringup.launch.py),
which runs the safety layer in full_body_mode reading rt/safety/lowcmd_in -- the
channel this whole-body node feeds -- and does NOT launch frame_task_server. Do
NOT pair with the plain h1_real_robot_bringup: that starts the safety layer in
split_mode (reads lowcmd_lower/upper_in) and would ignore our full-body channel.
    # on the robot:
    ros2 launch h1_bringup h1_real_robot_full_bringup.launch.py
    # on your device:
    ros2 launch h1_bringup h1_real_desktop_full_bringup.launch.py start_position_verified:=true

The split bringup is untouched -- lowerbody:=almi still works exactly as before.
⚠ Requires ROS_DOMAIN_ID exported (0 = real robot); the grasp code must be in the
container's mujoco_mpc (git submodule) and colcon-built (mjpc_fullbody_core).
"""
import os

from ament_index_python.packages import (
    get_package_prefix,
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    bringup_share = get_package_share_directory('h1_bringup')
    default_rviz = os.path.join(bringup_share, 'rviz', 'sim.rviz')
    mjpc_real_yaml = os.path.join(bringup_share, 'config', 'mjpc_real.yaml')
    sim_time_param = {'use_sim_time': False}

    model_log_params = {
        'enable_logging': ParameterValue(
            LaunchConfiguration('model_logging'), value_type=bool),
        'enable_visualization': ParameterValue(
            LaunchConfiguration('model_visualization'), value_type=bool),
        'clear_logs': ParameterValue(
            LaunchConfiguration('model_clear_logs'), value_type=bool),
    }

    # The full-body MPC is a DIRECT gflags binary (deliberately NOT an rclcpp
    # target), installed to lib/h12_deploy_mjpc. Run it with ExecuteProcess, NOT
    # `ros2 run` -- the ros2 launcher path injects imu_pitch_offset_deg 0.0 and
    # skips mjpc_real.yaml, which zeros the IMU calibration (DRIVE24 forensics).
    fullbody_bin = os.path.join(
        get_package_prefix('h12_deploy_mjpc'), 'lib', 'h12_deploy_mjpc',
        'mjpc_fullbody_core')

    return LaunchDescription([
        # SAFETY INTERLOCK: the whole-body MPC (which commands all 27 joints) only
        # launches when the operator confirms the robot is in a safe start pose:
        #   ros2 launch ... h1_real_desktop_full_bringup.launch.py start_position_verified:=true
        # A bare launch (default false) brings up ONLY perception + the grasp seam
        # (no robot commands); the MPC + estimator stay down until verified. Mirrors
        # the split bringup's start_position_verified gate on the leg controller.
        DeclareLaunchArgument('start_position_verified', default_value='false'),
        DeclareLaunchArgument('model_logging', default_value='true'),
        DeclareLaunchArgument('model_visualization', default_value='true'),
        DeclareLaunchArgument('model_clear_logs', default_value='true'),
        # Grasp task knobs (override at launch):
        DeclareLaunchArgument('task', default_value='Grasp H12 Magpie'),  # e.g. 'Lean H12 Magpie'
        DeclareLaunchArgument('strategy', default_value='6'),      # 6=stand; live-switch to 21
        DeclareLaunchArgument('object_name', default_value='Screw'),  # what to grasp
        DeclareLaunchArgument('place_object', default_value=''),    # '' = grasp only (v1); e.g. 'plate'

        # ---- perception services (IDENTICAL to h1_real_desktop_bringup) --------
        Node(package='model_server', executable='gemini_server', name='gemini_server',
             parameters=[sim_time_param, model_log_params], output='screen'),
        Node(package='model_server', executable='sam_server', name='sam_server',
             parameters=[sim_time_param, model_log_params], output='screen'),
        Node(package='model_server', executable='yolo_server', name='yolo_server',
             parameters=[sim_time_param, model_log_params], output='screen'),
        Node(package='model_server', executable='graspgen_server', name='graspgen_server',
             parameters=[sim_time_param, model_log_params], output='screen'),

        # ---- WHOLE-BODY MPC controller (replaces RL legs + FrameTask arms) -----
        # base estimator: synthesises rt/sportmodestate from proprioception.
        Node(package='h12_deploy_mjpc', executable='estimator_node',
             name='h12_deploy_mjpc_estimator',        # MUST match the yaml key
             parameters=[sim_time_param, mjpc_real_yaml], output='screen',
             condition=IfCondition(LaunchConfiguration('start_position_verified'))),
        # the full-body grasp MPC (nu=27). DRIVE24-validated flags; grasp task.
        ExecuteProcess(
            cmd=[fullbody_bin,
                 '--task', LaunchConfiguration('task'),
                 '--strategy', LaunchConfiguration('strategy'),
                 '--gravity_ff', '0.85', '--twin_dt', '0.001',
                 '--imu_pitch_offset_deg', '0.0', '--imu_roll_offset_deg', '1.3'],
            additional_env={'MJPC_TASKS_DIR': '/home/code/mujoco_mpc/build/mjpc/tasks'},
            output='screen',
            condition=IfCondition(LaunchConfiguration('start_position_verified'))),

        # ---- perception -> MPC grasp seam (reuses h12_skills helpers) ----------
        Node(package='h12_deploy_mjpc', executable='graspgen_pose_adapter',
             name='graspgen_pose_adapter',
             parameters=[sim_time_param,
                         {'object_name': LaunchConfiguration('object_name'),
                          'place_object': LaunchConfiguration('place_object')}],
             output='screen'),
        Node(package='h12_deploy_mjpc', executable='grasp_orchestrator',
             name='grasp_orchestrator',
             parameters=[sim_time_param], output='screen'),

        # ---- rviz (IDENTICAL to h1_real_desktop_bringup) ----------------------
        Node(package='rviz2', executable='rviz2', name='rviz2_sim',
             arguments=['-d', default_rviz], parameters=[sim_time_param], output='screen'),
    ])
