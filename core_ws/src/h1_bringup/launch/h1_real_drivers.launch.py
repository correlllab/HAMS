import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # Equivalent of `ros2 launch livox_ros_driver2 msg_MID360_launch.py`,
    # inlined so the MID360 driver comes up as part of this bringup. Params
    # mirror msg_MID360_launch.py; user_config_path points at the
    # humanoid-validated MID360_config.json stored under h1_bringup/config
    # (Unitree onboard net 192.168.123.164 host / 192.168.123.120 lidar,
    # lidar mounted inverted -> extrinsic roll 180).
    bringup_share = get_package_share_directory('h1_bringup')
    mid360_config = os.path.join(bringup_share, 'config', 'MID360_config.json')

    livox_mid360_params = [
        {'xfer_format': 1},      # 0-Pointcloud2(PointXYZRTL), 1-customized pointcloud format
        {'multi_topic': 0},      # 0-All LiDARs share the same topic, 1-One LiDAR one topic
        {'data_src': 0},         # 0-lidar, others-Invalid data src
        {'publish_freq': 10.0},  # publish frequency (Hz)
        {'output_data_type': 0},
        {'frame_id': 'livox_frame'},
        {'lvx_file_path': '/home/livox/livox_test.lvx'},
        {'user_config_path': mid360_config},
        {'cmdline_input_bd_code': 'livox0000000001'},
    ]

    livox_mid360 = Node(
        package='livox_ros_driver2',
        executable='livox_ros_driver2_node',
        name='livox_lidar_publisher',
        output='screen',
        parameters=livox_mid360_params,
    )

    # Equivalent of `ros2 launch cl_realsense h12_rs_cams.launch.py` — brings up
    # the head and left-hand RealSense cameras and their camera->link static TFs.
    realsense_share = get_package_share_directory('cl_realsense')
    realsense_cams = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(realsense_share, 'launch', 'h12_rs_cams.launch.py')
        )
    )

    # Magpie grippers, one node per side. The gripper_node has no built-in
    # left/right concept (node name, gripper/state topic and services are all
    # relative), so each side is distinguished by namespace; topics/services
    # land under /left/... and /right/.... auto_detect_port MUST be False with
    # two grippers attached, otherwise both race for the first /dev/ttyUSB*|ACM*.
    # Ports are the stable /dev/serial/by-id/ paths (follow the board's USB
    # serial, so they survive reboots/replugs and never swap left<->right).
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
            'finger1theta_min': 86.8,
            'finger1theta_max': 164.52,
            'finger1theta_90': 138.02,
            'finger2theta_min': 130.5,
            'finger2theta_max': 210.56,
            'finger2theta_90': 157.0,
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
            # OpenRB-150 serial ...FF122F35 (was ttyACM1)
            'port': '/dev/serial/by-id/usb-ROBOTIS_OpenRB-150_9F2640E15157375037202020FF122F35-if00',
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
        livox_mid360,
        realsense_cams,
        left_gripper,
        right_gripper,
    ])
