#!/usr/bin/env python3
"""Live slider GUI for the H1 sim's frame_task_server and magpie grippers.

Twelve sliders (XYZ + RPY for each wrist) drive the `/frame_task` action
targeting `left_wrist_yaw_link` and `right_wrist_yaw_link`. Two more sliders
drive the left/right magpie `gripper/set_position` services (0 = closed,
1 = fully open, mapped to millimetres via the `gripper_max_mm` param).

Usage:
    ros2 run h1_bringup slider_debugger
    # or
    python3 core_ws/src/h1_bringup/scripts/slider_debugger.py
"""

import math
import tkinter as tk

import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from builtin_interfaces.msg import Duration

from custom_ros_messages.action import FrameTask
from custom_ros_messages.srv import SetGripperPosition


WRIST_FRAMES = ('left_wrist_yaw_link', 'right_wrist_yaw_link')
SPIN_PERIOD_MS = 100
SEND_PERIOD_MS = 200
# Long horizon: frame_task_server breaks out of its control loop on
# convergence and we cancel the goal as soon as the slider changes. Setting
# duration to match SEND_PERIOD_MS made the server preempt itself before any
# visible motion happened — and added a 50-step settling delay (~500 ms)
# after every "successful" 200 ms goal.
GOAL_DURATION_SEC = 30


class SliderDebugger(Node):
    def __init__(self):
        super().__init__('slider_debugger')

        self.declare_parameter('left_gripper_service', '/gripper/left/set_position')
        self.declare_parameter('right_gripper_service', '/gripper/right/set_position')
        self.declare_parameter('gripper_max_mm', 85.0)
        self.left_gripper_srv_name = self.get_parameter('left_gripper_service').value
        self.right_gripper_srv_name = self.get_parameter('right_gripper_service').value
        self.gripper_max_mm = float(self.get_parameter('gripper_max_mm').value)

        self.frame_task_cli = ActionClient(self, FrameTask, '/frame_task')
        self.left_gripper_cli = self.create_client(SetGripperPosition, self.left_gripper_srv_name)
        self.right_gripper_cli = self.create_client(SetGripperPosition, self.right_gripper_srv_name)

        self._initial_left = None
        self._initial_right = None
        self._left_sub = self.create_subscription(
            PoseStamped, '/left_ee_pose', self._on_left_pose, 1)
        self._right_sub = self.create_subscription(
            PoseStamped, '/right_ee_pose', self._on_right_pose, 1)

        self._goal_handle = None
        self._last_wrist_values = None
        self._last_left_grip = None
        self._last_right_grip = None
        self._warned_left_grip = False
        self._warned_right_grip = False

    def _on_left_pose(self, msg: PoseStamped):
        if self._initial_left is None:
            self._initial_left = self._pose_to_xyzrpy(msg.pose)

    def _on_right_pose(self, msg: PoseStamped):
        if self._initial_right is None:
            self._initial_right = self._pose_to_xyzrpy(msg.pose)

    @staticmethod
    def _pose_to_xyzrpy(pose: Pose):
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        rpy = Rotation.from_quat(q).as_euler('xyz')
        return (pose.position.x, pose.position.y, pose.position.z,
                float(rpy[0]), float(rpy[1]), float(rpy[2]))

    def wait_for_initial_poses(self, timeout_sec=5.0):
        # Use time.monotonic, NOT self.get_clock().now() — when use_sim_time
        # is True the first /clock arrival can jump past `timeout_sec` in one
        # step and trip an immediate "timeout" before any pose has arrived.
        import time as _time
        end = _time.monotonic() + timeout_sec
        while rclpy.ok() and (self._initial_left is None or self._initial_right is None):
            rclpy.spin_once(self, timeout_sec=0.05)
            if _time.monotonic() > end:
                break
        if self._initial_left is None:
            self.get_logger().warn(
                'Timed out waiting for /left_ee_pose; initialising left sliders to zero')
            self._initial_left = (0.0,) * 6
        if self._initial_right is None:
            self.get_logger().warn(
                'Timed out waiting for /right_ee_pose; initialising right sliders to zero')
            self._initial_right = (0.0,) * 6

    def _build_pose(self, x, y, z, r, p, yw) -> Pose:
        qx, qy, qz, qw = Rotation.from_euler('xyz', [r, p, yw]).as_quat()
        return Pose(
            position=Point(x=float(x), y=float(y), z=float(z)),
            orientation=Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw)),
        )

    def send_wrist_targets(self, left_xyzrpy, right_xyzrpy):
        if not self.frame_task_cli.server_is_ready():
            return
        if self._goal_handle is not None:
            try:
                self._goal_handle.cancel_goal_async()
            except Exception:
                pass
            self._goal_handle = None

        goal = FrameTask.Goal()
        goal.frame_names = list(WRIST_FRAMES)
        goal.frame_targets = [
            self._build_pose(*left_xyzrpy),
            self._build_pose(*right_xyzrpy),
        ]
        goal.duration = Duration(sec=GOAL_DURATION_SEC, nanosec=0)

        send_future = self.frame_task_cli.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().warn(f'frame_task send failed: {exc}')
            return
        if handle is not None and handle.accepted:
            self._goal_handle = handle

    def send_gripper(self, side: str, slider_val: float):
        cli = self.left_gripper_cli if side == 'left' else self.right_gripper_cli
        warned_attr = '_warned_left_grip' if side == 'left' else '_warned_right_grip'
        srv_name = self.left_gripper_srv_name if side == 'left' else self.right_gripper_srv_name
        if not cli.service_is_ready():
            if not getattr(self, warned_attr):
                self.get_logger().warn(
                    f'{srv_name} not available; gripper commands will be skipped')
                setattr(self, warned_attr, True)
            return
        req = SetGripperPosition.Request()
        req.position = float(slider_val) * self.gripper_max_mm
        req.speed = 1.0
        cli.call_async(req)

    def cancel_active_goal(self):
        if self._goal_handle is not None:
            try:
                self._goal_handle.cancel_goal_async()
            except Exception:
                pass
            self._goal_handle = None


def _build_wrist_sliders(parent, label_prefix: str, init_xyzrpy):
    sliders = {}
    for name, lo, hi in (('X', -1.0, 1.0), ('Y', -1.0, 1.0), ('Z', -1.0, 1.0)):
        s = tk.Scale(parent, label=f'{label_prefix} {name}',
                     from_=lo, to=hi, resolution=0.01,
                     orient=tk.HORIZONTAL, length=260)
        s.pack(pady=2)
        sliders[name] = s
    for name in ('Roll', 'Pitch', 'Yaw'):
        s = tk.Scale(parent, label=f'{label_prefix} {name}',
                     from_=-math.pi, to=math.pi, resolution=0.01,
                     orient=tk.HORIZONTAL, length=260)
        s.pack(pady=2)
        sliders[name] = s
    sliders['X'].set(init_xyzrpy[0])
    sliders['Y'].set(init_xyzrpy[1])
    sliders['Z'].set(init_xyzrpy[2])
    sliders['Roll'].set(init_xyzrpy[3])
    sliders['Pitch'].set(init_xyzrpy[4])
    sliders['Yaw'].set(init_xyzrpy[5])
    return sliders


def _read_wrist(sliders):
    return (sliders['X'].get(), sliders['Y'].get(), sliders['Z'].get(),
            sliders['Roll'].get(), sliders['Pitch'].get(), sliders['Yaw'].get())


def main():
    rclpy.init()
    node = SliderDebugger()
    node.get_logger().info('Waiting for /left_ee_pose and /right_ee_pose to seed sliders...')
    node.wait_for_initial_poses(timeout_sec=5.0)

    root = tk.Tk()
    root.title('H1 Slider Debugger')

    left_frame = tk.LabelFrame(root, text='Left wrist + gripper', padx=8, pady=8)
    right_frame = tk.LabelFrame(root, text='Right wrist + gripper', padx=8, pady=8)
    left_frame.pack(side=tk.LEFT, padx=10, pady=10, anchor='n')
    right_frame.pack(side=tk.RIGHT, padx=10, pady=10, anchor='n')

    left_sliders = _build_wrist_sliders(left_frame, 'Left', node._initial_left)
    right_sliders = _build_wrist_sliders(right_frame, 'Right', node._initial_right)

    left_grip = tk.Scale(left_frame, label='Left Gripper (0=closed, 1=open)',
                         from_=0.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=260)
    left_grip.set(0.5)
    left_grip.pack(pady=2)
    right_grip = tk.Scale(right_frame, label='Right Gripper (0=closed, 1=open)',
                          from_=0.0, to=1.0, resolution=0.01,
                          orient=tk.HORIZONTAL, length=260)
    right_grip.set(0.5)
    right_grip.pack(pady=2)

    def on_close():
        node.cancel_active_goal()
        try:
            root.destroy()
        except Exception:
            pass

    root.protocol('WM_DELETE_WINDOW', on_close)

    def spin_tick():
        if rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            root.after(SPIN_PERIOD_MS, spin_tick)

    def send_tick():
        if not rclpy.ok():
            return
        left_xyzrpy = _read_wrist(left_sliders)
        right_xyzrpy = _read_wrist(right_sliders)
        wrist_values = left_xyzrpy + right_xyzrpy
        if wrist_values != node._last_wrist_values:
            node.send_wrist_targets(left_xyzrpy, right_xyzrpy)
            node._last_wrist_values = wrist_values

        lg = left_grip.get()
        if node._last_left_grip is None or lg != node._last_left_grip:
            if node._last_left_grip is not None:
                node.send_gripper('left', lg)
            node._last_left_grip = lg
        rg = right_grip.get()
        if node._last_right_grip is None or rg != node._last_right_grip:
            if node._last_right_grip is not None:
                node.send_gripper('right', rg)
            node._last_right_grip = rg

        root.after(SEND_PERIOD_MS, send_tick)

    root.after(SPIN_PERIOD_MS, spin_tick)
    root.after(SEND_PERIOD_MS, send_tick)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        node.cancel_active_goal()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
