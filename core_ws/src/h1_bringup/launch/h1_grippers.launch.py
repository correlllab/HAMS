from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Magpie grippers, one node per side. Factored out of
    # h1_real_drivers.launch.py so the calibration constants live in exactly one
    # place: h1_real_drivers (full onboard driver set) and
    # h1_safety_grippers (actuation-only bringup) both include this file.
    #
    # The gripper_node has no built-in left/right concept (node name,
    # gripper/state topic and services are all relative), so each side is
    # distinguished by namespace; topics/services land under /left/... and
    # /right/.... auto_detect_port MUST be False with two grippers attached,
    # otherwise both race for the first /dev/ttyUSB*|ACM*. Ports are the stable
    # /dev/serial/by-id/ paths (follow the board's USB serial, so they survive
    # reboots/replugs and never swap left<->right).
    left_gripper = Node(
        package='magpie_control',
        executable='gripper_node',
        name='gripper_node',
        namespace='left',
        output='screen',
        parameters=[{
            'auto_detect_port': False,
            # OpenRB-150 serial ...FF101F14 (was ttyACM0)
            'port': '/dev/serial/by-id/usb-ROBOTIS_OpenRB-150_B0935A515157375037202020FF101F14-if00',
            'use_eflesh': False,
            # Finger angle limits (degrees), per-gripper calibration.
            # Recalibrated 2026-07-21 after gripper repair. min/max are the
            # measured servo angles at the open/closed stops; theta_90
            # (parallel-jaw ref) scaled from the prior parallel fraction.
            'finger1theta_min': 87.98,
            'finger1theta_max': 162.4,
            'finger1theta_90': 135.71,
            'finger2theta_min': 130.6,
            'finger2theta_max': 209.09,
            'finger2theta_90': 157.89,
        }],
    )

    right_gripper = Node(
        package='magpie_control',
        executable='gripper_node',
        name='gripper_node',
        namespace='right',
        output='screen',
        parameters=[{
            'auto_detect_port': False,
            # OpenRB-150 serial ...FF0F0E1C (replacement board, was ...FF122F35)
            'port': '/dev/serial/by-id/usb-ROBOTIS_OpenRB-150_3EC94E825157375037202020FF0F0E1C-if00',
            'use_eflesh': False,
            # Finger angle limits (degrees), per-gripper calibration.
            'finger1theta_min': 86.51,
            'finger1theta_max': 168.33,
            'finger1theta_90': 141.83,
            'finger2theta_min': 132.55,
            'finger2theta_max': 211.44,
            'finger2theta_90': 159.05,
        }],
    )

    return LaunchDescription([
        left_gripper,
        right_gripper,
    ])
