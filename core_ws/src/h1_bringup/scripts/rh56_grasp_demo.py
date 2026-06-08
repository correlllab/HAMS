#!/usr/bin/env python3
"""Width-space RH56 grasp demo for the H1-2 ROS/MuJoCo simulation.

The node sends short /frame_task wrist goals while streaming the matching hand
command for each width waypoint. It uses a precomputed RH56 FK cache, so the
ROS container does not need to import MuJoCo just to run the planner.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Pose, Quaternion
from std_msgs.msg import Float64MultiArray

from custom_ros_messages.action import FrameTask, NamedConfig

try:
    from magpie_msgs.srv import SetGripperPosition
except ModuleNotFoundError:
    SetGripperPosition = None


NON_THUMB_FINGERS = ["index", "middle", "ring", "pinky"]
ACTUATOR_ORDER = ["pinky", "ring", "middle", "index", "thumb_proximal", "thumb_yaw"]
GRASP_FINGER_SETS = {
    2: ["index"],
    3: ["index", "middle"],
    4: ["index", "middle", "ring"],
    5: ["index", "middle", "ring", "pinky"],
}
THUMB_YAW_LINE = 1.16
PRACTICAL_MIN_WIDTH_M = 0.005

# Planner world: +X forward, +Y left, +Z up.
# H12 pelvis: +Y forward, +X right, +Z up.
R_PELVIS_TO_WORLD = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
R_WORLD_TO_PELVIS = R_PELVIS_TO_WORLD.T

# H1-2 wrist_yaw -> RH56 hand-base attachment transform, copied from the RH56
# planner. These constants define where the hand base sits relative to the H12
# wrist frame; /frame_task still targets the wrist_yaw frame.
R_WRIST_TO_HAND_RIGHT = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
T_WRIST_TO_HAND_RIGHT = np.array([0.054, 0.0, 0.0])
R_HAND_TO_WRIST_RIGHT = R_WRIST_TO_HAND_RIGHT.T
T_HAND_TO_WRIST_RIGHT = -R_HAND_TO_WRIST_RIGHT @ T_WRIST_TO_HAND_RIGHT

R_WRIST_TO_HAND_LEFT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
T_WRIST_TO_HAND_LEFT = np.array([0.054, 0.0, 0.0])
R_HAND_TO_WRIST_LEFT = R_WRIST_TO_HAND_LEFT.T
T_HAND_TO_WRIST_LEFT = -R_HAND_TO_WRIST_LEFT @ T_WRIST_TO_HAND_LEFT


def _duration(seconds: float) -> Duration:
    sec = int(seconds)
    return Duration(sec=sec, nanosec=int((seconds - sec) * 1e9))


def _plane_rot(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rx_m @ ry_m @ rz_m


def _tilted_base_rot(tilt_y: float) -> np.ndarray:
    cy, sy = np.cos(tilt_y), np.sin(tilt_y)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx_pi = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    return ry_m @ rx_pi


def _matrix_to_pose(matrix: np.ndarray) -> Pose:
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return Pose(
        position=Point(
            x=float(matrix[0, 3]),
            y=float(matrix[1, 3]),
            z=float(matrix[2, 3]),
        ),
        orientation=Quaternion(
            x=float(quat[0]),
            y=float(quat[1]),
            z=float(quat[2]),
            w=float(quat[3]),
        ),
    )


class CachedInspireFK:
    """MuJoCo-free RH56 fingertip FK backed by a precomputed npz table."""

    def __init__(self, cache_path: Path):
        data = np.load(str(cache_path))
        self._finger_ctrl = {name: data[f"fc_{name}"] for name in NON_THUMB_FINGERS}
        self._finger_tips = {name: data[f"ft_{name}"] for name in NON_THUMB_FINGERS}
        self._thumb_pitch_vals = data["thumb_pitch"]
        self._thumb_yaw_vals = data["thumb_yaw"]
        self._thumb_tips = data["thumb_tips"]

        self.ctrl_min = {name: float(vals[0]) for name, vals in self._finger_ctrl.items()}
        self.ctrl_max = {name: float(vals[-1]) for name, vals in self._finger_ctrl.items()}
        self.ctrl_min["thumb_proximal"] = float(self._thumb_pitch_vals[0])
        self.ctrl_max["thumb_proximal"] = float(self._thumb_pitch_vals[-1])
        self.ctrl_min["thumb_yaw"] = float(self._thumb_yaw_vals[0])
        self.ctrl_max["thumb_yaw"] = float(self._thumb_yaw_vals[-1])
        self.ctrl_min["thumb_yaw_line"] = THUMB_YAW_LINE
        self.ctrl_max["thumb_yaw_line"] = THUMB_YAW_LINE

        self._finger_interp = {
            name: interp1d(
                self._finger_ctrl[name],
                self._finger_tips[name],
                axis=0,
                kind="linear",
                bounds_error=False,
                fill_value=(self._finger_tips[name][0], self._finger_tips[name][-1]),
            )
            for name in NON_THUMB_FINGERS
        }
        self._thumb_interp = RegularGridInterpolator(
            (self._thumb_pitch_vals, self._thumb_yaw_vals),
            self._thumb_tips,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def finger_tip(self, name: str, ctrl: float) -> np.ndarray:
        ctrl = float(np.clip(ctrl, self.ctrl_min[name], self.ctrl_max[name]))
        return np.asarray(self._finger_interp[name](ctrl), dtype=float)

    def thumb_tip(self, ctrl_pitch: float, ctrl_yaw: float) -> np.ndarray:
        cp = float(np.clip(ctrl_pitch, self.ctrl_min["thumb_proximal"], self.ctrl_max["thumb_proximal"]))
        cy = float(np.clip(ctrl_yaw, self.ctrl_min["thumb_yaw"], self.ctrl_max["thumb_yaw"]))
        return np.asarray(self._thumb_interp([[cp, cy]])[0], dtype=float)

    def z_range_for_finger(self, name: str) -> tuple[float, float]:
        zvals = self._finger_tips[name][:, 2]
        return float(zvals.min()), float(zvals.max())

    def coplanar_ctrls(self, target_z_base: float, fingers: Iterable[str]) -> Dict[str, Optional[float]]:
        result: Dict[str, Optional[float]] = {}
        for name in fingers:
            zvals = self._finger_tips[name][:, 2]
            z_min, z_max = float(zvals.min()), float(zvals.max())
            if not (z_min <= target_z_base <= z_max):
                result[name] = None
                continue

            def z_error(ctrl: float, target: float = target_z_base, finger: str = name) -> float:
                return float(self._finger_interp[finger](ctrl)[2]) - target

            try:
                result[name] = float(brentq(z_error, self.ctrl_min[name], self.ctrl_max[name], xtol=1e-6))
            except ValueError:
                ctrl_vals = self._finger_ctrl[name]
                result[name] = float(ctrl_vals[int(np.argmin(np.abs(zvals - target_z_base)))])
        return result

    def thumb_tip_at_z(self, target_z_base: float, ctrl_yaw: float) -> Optional[float]:
        cy = float(np.clip(ctrl_yaw, self.ctrl_min["thumb_yaw"], self.ctrl_max["thumb_yaw"]))
        pitches = self._thumb_pitch_vals
        zvals = self._thumb_interp(np.column_stack([pitches, np.full_like(pitches, cy)]))[:, 2]
        z_min, z_max = float(zvals.min()), float(zvals.max())
        if not (z_min <= target_z_base <= z_max):
            return None

        def z_error(pitch: float) -> float:
            return float(self._thumb_interp([[pitch, cy]])[0, 2]) - target_z_base

        try:
            return float(brentq(z_error, self.ctrl_min["thumb_proximal"], self.ctrl_max["thumb_proximal"], xtol=1e-6))
        except ValueError:
            return None


@dataclass
class ClosureResult:
    mode: str
    midpoint: np.ndarray
    width: float
    finger_span: float
    tip_positions: Dict[str, np.ndarray]
    ctrl_values: Dict[str, float]
    base_tilt_y: float

    def world_base(self, world_grasp_z: float, plane_rx: float, plane_ry: float, plane_rz: float) -> np.ndarray:
        rot = _plane_rot(plane_rx, plane_ry, plane_rz) @ _tilted_base_rot(self.base_tilt_y)
        mid_w = rot @ self.midpoint
        return np.array([-mid_w[0], -mid_w[1], world_grasp_z - mid_w[2]])


class ClosureGeometry:
    """Line and plane RH56 closure geometry using cached fingertip FK."""

    def __init__(self, fk: CachedInspireFK):
        self.fk = fk

    def _joint_dist(self, ref_finger: str, s: float, ctrl_yaw: Optional[float] = None) -> float:
        if ctrl_yaw is None:
            ctrl_yaw = self.fk.ctrl_max["thumb_yaw"]
        mn, cm = self.fk.ctrl_min, self.fk.ctrl_max
        ctrl_pitch = mn["thumb_proximal"] + s * (cm["thumb_proximal"] - mn["thumb_proximal"])
        ctrl_ref = mn[ref_finger] + s * (cm[ref_finger] - mn[ref_finger])
        delta = self.fk.thumb_tip(ctrl_pitch, ctrl_yaw) - self.fk.finger_tip(ref_finger, ctrl_ref)
        return float(np.hypot(delta[0], delta[2]))

    def _joint_closure_range(self, ref_finger: str, ctrl_yaw: Optional[float] = None) -> tuple[float, float, float]:
        s_vals = np.linspace(0.0, 1.0, 120)
        d_vals = np.array([self._joint_dist(ref_finger, float(s), ctrl_yaw) for s in s_vals])
        idx_min = int(np.argmin(d_vals))
        return float(s_vals[idx_min]), float(d_vals[idx_min]), float(d_vals[0])

    def _solve_joint_closure(self, ref_finger: str, target_width: float, ctrl_yaw: Optional[float] = None) -> tuple[float, float, float]:
        s_min, d_min, d_open = self._joint_closure_range(ref_finger, ctrl_yaw)
        if target_width >= d_open:
            s = 0.0
        elif target_width <= d_min:
            s = s_min
        else:
            try:
                s = float(brentq(lambda value: self._joint_dist(ref_finger, value, ctrl_yaw) - target_width, 0.0, s_min, xtol=1e-5))
            except ValueError:
                s = s_min
        mn, cm = self.fk.ctrl_min, self.fk.ctrl_max
        ctrl_pitch = mn["thumb_proximal"] + s * (cm["thumb_proximal"] - mn["thumb_proximal"])
        ctrl_ref = mn[ref_finger] + s * (cm[ref_finger] - mn[ref_finger])
        return s, ctrl_pitch, ctrl_ref

    def width_range(self, mode: str, n_fingers: int) -> tuple[float, float]:
        ref = "index" if mode == "line" or n_fingers == 2 else "middle"
        _, d_min, d_open = self._joint_closure_range(ref, THUMB_YAW_LINE if mode == "line" else None)
        return max(d_min, PRACTICAL_MIN_WIDTH_M), d_open

    def solve(self, mode: str, target_width: float, n_fingers: int) -> ClosureResult:
        if mode == "line":
            return self.line(target_width)
        return self.plane(target_width, n_fingers=n_fingers)

    def line(self, target_width: float) -> ClosureResult:
        yaw = self.fk.ctrl_max["thumb_yaw_line"]
        _, ctrl_pitch, c_idx = self._solve_joint_closure("index", target_width, ctrl_yaw=yaw)
        index_tip = self.fk.finger_tip("index", c_idx)
        thumb_tip = self.fk.thumb_tip(ctrl_pitch, yaw)
        delta = thumb_tip - index_tip
        tilt_y = float(np.clip(np.arctan2(-delta[2], delta[0]), -np.pi / 2.0, np.pi / 2.0))
        ctrl = {name: self.fk.ctrl_min[name] for name in NON_THUMB_FINGERS}
        ctrl["index"] = float(c_idx)
        ctrl["thumb_proximal"] = float(ctrl_pitch)
        ctrl["thumb_yaw"] = float(yaw)
        return ClosureResult(
            mode="2-finger line",
            midpoint=(thumb_tip + index_tip) / 2.0,
            width=float(np.hypot(delta[0], delta[2])),
            finger_span=0.0,
            tip_positions={"thumb": thumb_tip, "index": index_tip},
            ctrl_values=ctrl,
            base_tilt_y=tilt_y,
        )

    def plane(self, target_width: float, n_fingers: int) -> ClosureResult:
        fingers = GRASP_FINGER_SETS[n_fingers]
        ref_finger = "middle" if "middle" in fingers else fingers[0]
        _, ctrl_pitch, c_ref = self._solve_joint_closure(ref_finger, target_width)
        ref_tip = self.fk.finger_tip(ref_finger, c_ref)
        thumb_tip = self.fk.thumb_tip(ctrl_pitch, self.fk.ctrl_max["thumb_yaw"])
        target_z = float(ref_tip[2])
        coplanar = self.fk.coplanar_ctrls(target_z, fingers=fingers)
        for finger in fingers:
            if coplanar.get(finger) is None:
                _, z_max = self.fk.z_range_for_finger(finger)
                coplanar[finger] = self.fk.ctrl_min[finger] if target_z >= z_max else self.fk.ctrl_max[finger]
        tips = {finger: self.fk.finger_tip(finger, float(coplanar[finger])) for finger in fingers}
        tips["thumb"] = thumb_tip
        nonthumb = np.array([tips[finger] for finger in fingers])
        centroid = nonthumb.mean(axis=0)
        delta = thumb_tip - centroid
        tilt_y = float(np.clip(np.arctan2(-delta[2], delta[0]), -np.pi / 2.0, np.pi / 2.0))
        ctrl = {name: self.fk.ctrl_min[name] for name in NON_THUMB_FINGERS}
        for finger in fingers:
            ctrl[finger] = float(coplanar[finger])
        ctrl["thumb_proximal"] = float(ctrl_pitch)
        ctrl["thumb_yaw"] = float(self.fk.ctrl_max["thumb_yaw"])
        return ClosureResult(
            mode=f"{n_fingers}-finger plane",
            midpoint=np.vstack([nonthumb, thumb_tip]).mean(axis=0),
            width=float(np.hypot(delta[0], delta[2])),
            finger_span=float(nonthumb[:, 1].max() - nonthumb[:, 1].min()) if len(fingers) > 1 else 0.0,
            tip_positions=tips,
            ctrl_values=ctrl,
            base_tilt_y=tilt_y,
        )


class RH56GraspDemo(Node):
    def __init__(self, args: argparse.Namespace, fk: CachedInspireFK, closure: ClosureGeometry):
        super().__init__("rh56_grasp_demo")
        self.args = args
        self.fk = fk
        self.closure = closure
        self.frame = f"{args.side}_wrist_yaw_link"
        self.frame_task_client = ActionClient(self, FrameTask, "/frame_task")
        self.named_config_client = ActionClient(self, NamedConfig, "/named_config")
        self.hand_pub = None
        if args.hand_backend in ("inspire-topic", "both"):
            self.hand_pub = self.create_publisher(Float64MultiArray, f"/{args.side}_hand_cmd", 10)
        self.gripper_client = None
        if args.hand_backend in ("magpie", "both") and SetGripperPosition is not None:
            self.gripper_client = self.create_client(SetGripperPosition, f"/gripper/{args.side}/set_position")

    def wait_ready(self) -> bool:
        if not self.frame_task_client.wait_for_server(timeout_sec=self.args.wait_timeout):
            self.get_logger().error("/frame_task action server is not available")
            return False
        if self.gripper_client is not None and not self.gripper_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Magpie gripper service is not available; wrist motion will still run")
        return True

    def send_named_home(self) -> bool:
        if not self.named_config_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("/named_config action server is not available; skipping home")
            return False
        goal = NamedConfig.Goal()
        goal.config_name = "home"
        goal.duration = _duration(self.args.home_duration)
        future = self.named_config_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().warn("home goal rejected")
            return False
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=self.args.home_duration + 5.0)
        result = result_future.result()
        return bool(result and result.status == GoalStatus.STATUS_SUCCEEDED and result.result.success)

    def hand_open_values(self, result: ClosureResult) -> np.ndarray:
        ctrl = np.array([result.ctrl_values.get(name, self.fk.ctrl_min[name]) for name in ACTUATOR_ORDER], dtype=float)
        ctrl_min = np.array([self.fk.ctrl_min[name] for name in ACTUATOR_ORDER], dtype=float)
        ctrl_max = np.array([self.fk.ctrl_max[name] for name in ACTUATOR_ORDER], dtype=float)
        span = np.where(ctrl_max > ctrl_min, ctrl_max - ctrl_min, 1.0)
        return 1.0 - np.clip((ctrl - ctrl_min) / span, 0.0, 1.0)

    def wrist_target_pelvis(self, result: ClosureResult) -> np.ndarray:
        plane_rpy = np.deg2rad(np.asarray(self.args.plane_rpy_deg, dtype=float))
        grasp_xyz = np.array([self.args.grasp_x, self.args.grasp_y, self.args.grasp_z], dtype=float)
        world_t_hand = np.eye(4)
        world_t_hand[:3, :3] = _plane_rot(*plane_rpy) @ _tilted_base_rot(result.base_tilt_y)
        world_t_hand[:3, 3] = result.world_base(grasp_xyz[2], *plane_rpy) + np.array([grasp_xyz[0], grasp_xyz[1], 0.0])

        if self.args.side == "right":
            hand_to_wrist_r = R_HAND_TO_WRIST_RIGHT
            hand_to_wrist_t = T_HAND_TO_WRIST_RIGHT
        else:
            hand_to_wrist_r = R_HAND_TO_WRIST_LEFT
            hand_to_wrist_t = T_HAND_TO_WRIST_LEFT

        world_t_wrist = np.eye(4)
        world_t_wrist[:3, :3] = world_t_hand[:3, :3] @ hand_to_wrist_r
        world_t_wrist[:3, 3] = world_t_hand[:3, 3] + world_t_hand[:3, :3] @ hand_to_wrist_t

        pelvis_t_wrist = np.eye(4)
        pelvis_t_wrist[:3, :3] = R_WORLD_TO_PELVIS @ world_t_wrist[:3, :3]
        pelvis_t_wrist[:3, 3] = R_WORLD_TO_PELVIS @ world_t_wrist[:3, 3]
        return pelvis_t_wrist

    def publish_hand(self, open_values: np.ndarray, width_m: float, include_magpie: bool) -> None:
        if self.hand_pub is not None:
            msg = Float64MultiArray()
            msg.data = [float(v) for v in np.clip(open_values, 0.0, 1.0)]
            self.hand_pub.publish(msg)
        if include_magpie and self.gripper_client is not None and self.gripper_client.service_is_ready():
            req = SetGripperPosition.Request()
            req.position = float(np.clip(width_m * 1000.0, 0.0, self.args.magpie_max_mm))
            req.speed = float(self.args.magpie_speed)
            self.gripper_client.call_async(req)

    def send_segment(self, target: np.ndarray, open_values: np.ndarray, width_m: float, duration: float) -> bool:
        goal = FrameTask.Goal()
        goal.frame_names = [self.frame]
        goal.frame_targets = [_matrix_to_pose(target)]
        goal.duration = _duration(duration)

        send_future = self.frame_task_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error("frame_task goal rejected")
            return False

        self.publish_hand(open_values, width_m, include_magpie=True)
        result_future = handle.get_result_async()
        deadline = time.monotonic() + duration + self.args.segment_timeout_margin
        next_pub = 0.0
        while rclpy.ok() and not result_future.done():
            now = time.monotonic()
            if now >= deadline:
                handle.cancel_goal_async()
                self.get_logger().error("frame_task segment timed out")
                return False
            if now >= next_pub:
                self.publish_hand(open_values, width_m, include_magpie=False)
                next_pub = now + 1.0 / max(self.args.hand_publish_hz, 1.0)
            rclpy.spin_once(self, timeout_sec=0.02)

        result = result_future.result()
        return bool(result and result.status == GoalStatus.STATUS_SUCCEEDED and result.result.success)

    def run_plan(self) -> bool:
        mode = self.args.mode
        n_fingers = 2 if mode == "line" else self.args.fingers
        lo, hi = self.closure.width_range(mode, n_fingers)
        widths = np.linspace(self.args.start_width_mm / 1000.0, self.args.target_width_mm / 1000.0, self.args.steps)
        self.get_logger().info(
            "mode=%s fingers=%d range=[%.1f, %.1f] mm backend=%s frame=%s",
            mode,
            n_fingers,
            lo * 1000.0,
            hi * 1000.0,
            self.args.hand_backend,
            self.frame,
        )

        for i, raw_width in enumerate(widths, start=1):
            width = float(np.clip(raw_width, lo, hi))
            result = self.closure.solve(mode, width, n_fingers)
            target = self.wrist_target_pelvis(result)
            open_values = self.hand_open_values(result)
            pos = target[:3, 3]
            self.get_logger().info(
                "step %02d/%02d width=%.1f mm wrist_pelvis=[%.3f, %.3f, %.3f] hand=[%s]",
                i,
                len(widths),
                result.width * 1000.0,
                pos[0],
                pos[1],
                pos[2],
                ", ".join(f"{v:.2f}" for v in open_values),
            )
            if self.args.dry_run:
                continue
            if not self.send_segment(target, open_values, result.width, self.args.segment_duration):
                return False
        return True


def default_cache_path() -> Path:
    env_path = os.environ.get("RH56_FK_CACHE")
    if env_path:
        return Path(env_path)
    try:
        from ament_index_python.packages import get_package_share_directory
        share_path = Path(get_package_share_directory("h1_bringup")) / "data" / "rh56_fk_cache.npz"
        if share_path.exists():
            return share_path
    except Exception:
        pass
    source_path = Path(__file__).resolve().parents[1] / "data" / "rh56_fk_cache.npz"
    return source_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a width-space RH56 grasp on the H1-2 sim.")
    parser.add_argument("--mode", choices=("line", "plane"), default="plane")
    parser.add_argument("--fingers", type=int, choices=(3, 4, 5), default=4)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--start-width-mm", type=float, default=110.0)
    parser.add_argument("--target-width-mm", type=float, default=40.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--segment-duration", type=float, default=0.45)
    parser.add_argument("--segment-timeout-margin", type=float, default=5.0)
    parser.add_argument("--hand-publish-hz", type=float, default=20.0)
    parser.add_argument("--hand-backend", choices=("magpie", "inspire-topic", "both", "none"), default="magpie")
    parser.add_argument("--magpie-max-mm", type=float, default=110.0)
    parser.add_argument("--magpie-speed", type=float, default=1.0)
    parser.add_argument("--grasp-x", type=float, default=0.35, help="Planner-world grasp center X, forward, metres")
    parser.add_argument("--grasp-y", type=float, default=-0.25, help="Planner-world grasp center Y, left positive, metres")
    parser.add_argument("--grasp-z", type=float, default=0.10, help="Planner-world grasp center Z, up, metres")
    parser.add_argument("--plane-rpy-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--fk-cache", type=Path, default=default_cache_path())
    parser.add_argument("--wait-timeout", type=float, default=10.0)
    parser.add_argument("--home-first", action="store_true")
    parser.add_argument("--home-duration", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if args.hand_backend in ("magpie", "both") and SetGripperPosition is None:
        raise SystemExit("magpie_msgs is not importable; use --hand-backend inspire-topic or none")
    if not args.fk_cache.exists():
        raise SystemExit(f"FK cache not found: {args.fk_cache}. Set RH56_FK_CACHE or pass --fk-cache.")

    fk = CachedInspireFK(args.fk_cache)
    closure = ClosureGeometry(fk)

    rclpy.init()
    node = RH56GraspDemo(args, fk, closure)
    try:
        if not args.dry_run and not node.wait_ready():
            return 2
        if args.home_first and not args.dry_run:
            node.get_logger().info("sending named_config home")
            node.send_named_home()
        return 0 if node.run_plan() else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
