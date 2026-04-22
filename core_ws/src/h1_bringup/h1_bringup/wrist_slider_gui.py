"""cv2 trackbar GUI that sends FrameTask goals for left/right wrist yaw links.

Sliders cover x/y/z (meters, world frame) and roll/pitch/yaw (degrees) for
both left_wrist_yaw_link and right_wrist_yaw_link. Targets are expressed in
the world frame — the IK solver holds `transform_target_to_world`.

Startup seeds slider positions from /frame_poses (published by the server
once a goal has been set) or falls back to reasonable world-frame defaults.
Each slider change sends a short-duration FrameTask goal so the server
relinquishes its controller lock quickly and is ready for the next goal.
"""

import threading

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose
from rclpy.action import ActionClient
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

from custom_ros_messages.action import FrameTask


FRAMES = ['left_wrist_yaw_link', 'right_wrist_yaw_link']

# (slider name, lo, hi). cv2 trackbars are int-only: mm for position, deg for rotation.
AXES = [
    ('x_mm',      -800,  800),
    ('y_mm',      -800,  800),
    ('z_mm',         0, 2000),
    ('roll_deg',  -180,  180),
    ('pitch_deg', -180,  180),
    ('yaw_deg',   -180,  180),
]

# World-frame starting targets (pelvis ≈ 1.0m off the floor on H1-2), picked
# so the solver has something reachable on first goal.
DEFAULTS = {
    'left_wrist_yaw_link':  {'x_mm': 300, 'y_mm':  250, 'z_mm': 1100, 'roll_deg': 0, 'pitch_deg': 0, 'yaw_deg': 0},
    'right_wrist_yaw_link': {'x_mm': 300, 'y_mm': -250, 'z_mm': 1100, 'roll_deg': 0, 'pitch_deg': 0, 'yaw_deg': 0},
}

GOAL_DURATION_SEC = 1.0  # server's own loop exits when this elapses or error hits threshold


def _pose_from_slider_values(vals):
    pose = Pose()
    pose.position.x = vals['x_mm'] / 1000.0
    pose.position.y = vals['y_mm'] / 1000.0
    pose.position.z = vals['z_mm'] / 1000.0
    rpy = np.deg2rad([vals['roll_deg'], vals['pitch_deg'], vals['yaw_deg']])
    qx, qy, qz, qw = R.from_euler('xyz', rpy).as_quat()
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = qx, qy, qz, qw
    return pose


class WristSliderGui(Node):
    def __init__(self):
        super().__init__('wrist_slider_gui')
        self.client = ActionClient(self, FrameTask, 'frame_task')
        self._active_goal = None
        self._goal_lock = threading.Lock()

    def send_single(self, frame_name, pose):
        """Send a FrameTask goal for a single frame."""
        if not self.client.server_is_ready():
            self.client.wait_for_server(timeout_sec=0.2)
            if not self.client.server_is_ready():
                self.get_logger().warn(
                    'frame_task action server not ready', throttle_duration_sec=2.0)
                return

        with self._goal_lock:
            prev = self._active_goal
            self._active_goal = None

        if prev is not None:
            try:
                prev.cancel_goal_async()
            except Exception:
                pass

        goal = FrameTask.Goal()
        goal.frame_names = [frame_name]
        goal.frame_targets = [pose]
        goal.duration = Duration(sec=int(GOAL_DURATION_SEC),
                                 nanosec=int((GOAL_DURATION_SEC % 1) * 1e9))

        future = self.client.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().warn(f'goal send failed: {exc}')
            return
        if handle.accepted:
            with self._goal_lock:
                self._active_goal = handle


def main(args=None):
    rclpy.init(args=args)
    node = WristSliderGui()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    windows = {}
    for frame in FRAMES:
        win = f'{frame} target (world frame)'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 520, 320)
        for name, lo, hi in AXES:
            init = DEFAULTS[frame][name]
            init = max(lo, min(hi, init))
            cv2.createTrackbar(name, win, init - lo, hi - lo, lambda v: None)
        windows[frame] = win

    def read_vals(frame):
        win = windows[frame]
        return {name: cv2.getTrackbarPos(name, win) + lo for name, lo, _ in AXES}

    last_vals = {frame: read_vals(frame) for frame in FRAMES}
    try:
        while rclpy.ok():
            for frame in FRAMES:
                vals = read_vals(frame)
                if vals != last_vals[frame]:
                    node.send_single(frame, _pose_from_slider_values(vals))
                    last_vals[frame] = vals

            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
