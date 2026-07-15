#!/usr/bin/env python3
"""grasp_benchmark — compare parallel-jaw grasp-synthesis methods on a RoboCasa task.

Runs ONE grasp episode (perceive -> synthesize -> reach -> close -> lift) against
the live RoboCasa sim + bringup stack and writes a JSON record with success and
timing. Designed for the CheesyBread task (single right arm, robot held by the
elastic-band tether) but works on any task with a graspable cfg object.

Methods (--method):
  centroid           head-cam SAM cloud -> centroid, fixed top-down grasp from
                     the home pose. The naive baseline.
  topdown_antipodal  head-cam finds the object, the wrist is parked above it,
                     the WRIST-cam SAM cloud's top layer is PCA'd and the
                     fingers close across the minor (narrowest) axis. Top-down
                     antipodal baseline.
  graspgenx          gemini box -> SAM mask -> cloud -> GraspGenX 6-DOF grasps,
                     executed best-first exactly as ranked (no re-ranking).
  vlm_judge          candidates from BOTH GraspGenX (top-2) and top-layer PCA
                     (short_side + long_side), rendered as one labelled contact
                     sheet; Gemini picks the grasp (magpie pickup-pipeline
                     style VLM-as-judge).
  skill              the DEPLOYED /skill/grasp action (skills/grasp.py) as
                     pick_place invokes it: box -> SAM -> GraspGenX -> priority
                     tier + diversity re-rank -> walk the ranked candidates with
                     a multi-pass servo. Not a reimplementation, so this is the
                     reference the methods above are worth comparing against —
                     and the only one that retries after an unreachable
                     candidate. It does not lift, so the harness lifts for it.

Success is ground truth, not vision: the sim's MeasurementBridge publishes
/robocasa/object_poses (JSON, MuJoCo world frame) and the episode succeeds when
the target object's z rises by --success-dz and stays there through the hold.

Example (inside the hams_ros container, bringup already running):
  ros2 run h12_skills grasp_benchmark -- --method graspgenx \
      --object "wedge of cheese" --gt-name cheese \
      --out /home/code/core_ws/benchmark_results/graspgenx_seed42.json
"""

import argparse
import json
import threading
import time

import numpy as np

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from geometry_msgs.msg import Pose
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Bool, String
from tf2_ros import TransformException

from custom_ros_messages.action import NamedConfig, SkillGrasp

from .base import SkillsBase, GRASP_FRAMES, MIN_GRASP_POINTS
from .perception_utils import (
    decode_compressed_depth_image, deproject_mask, transform_points,
    transform_to_matrix, mat_to_quat, pose_to_matrix,
)


# GraspGenX gripper-base frame: origin at the magpie base, +Z = approach into the
# object, +X = finger-closing axis; the fingertip contact point sits at +Z
# TCP_DEPTH_M (magpie config fingertip ~[0, 0.0022, 0.1146]). Analytic methods
# (centroid / antipodal / PCA) build their poses in this same convention so ALL
# methods execute through the identical drive-the-graspgenx-frame path.
TCP_DEPTH_M = 0.1146
HEAD_OPTICAL_FRAME = 'camera_color_optical_frame'

APPROACH_DIST_M = 0.10     # pre-grasp standoff along the grasp -Z (back off)
APPROACH_SEC = 8.0         # IK budget for the (long) home -> pre-grasp motion
# The contact move is short in DISTANCE (one standoff) but it is the one that has
# to land accurately: the fingers close wherever it stops. At 4.0s it converged
# to only ~9.5mm (vs 0.2mm for the pre-grasp, which gets 8s) and the gripper
# shoved the object off the counter instead of straddling it. Give it the same
# budget as the approach so it actually settles before the fingers close.
CONTACT_SEC = 8.0          # pre-grasp -> grasp
SETTLE_SEC = 0.5           # let the arm come to rest before closing the fingers
LIFT_SEC = 4.0
HOLD_SEC = 2.0             # post-lift hold before judging success
LIFT_DIST_M = 0.15
SCAN_HEIGHT_M = 0.35       # wrist-camera standoff above the object for the scan
TOP_LAYER_M = 0.025        # top-layer slab thickness for antipodal/PCA analysis
FINGER_SINK_M = 0.02       # fingertip depth below the object's top surface
WIDTH_MARGIN_M = 0.02      # pre-open margin beyond the measured object width
MAX_ATTEMPTS = 5           # ranked candidates to try before giving up
# Approach tilts off straight-down, tried in this order (see _tilted_pose).
# Measured pre-grasp convergence on this arm: 20deg -> 3mm (reaches cleanly),
# 30deg -> 17-44mm, topdown -> 41mm, 45deg -> 128mm. Unsurprising: the wrist
# pitch is clamped to ±0.4625 rad (±26.5°), so the arm simply cannot point the
# gripper far off its natural attitude. Lead with 20deg.
GRASP_TILTS = ((np.pi / 9, '20deg'), (np.pi / 6, '30deg'),
               (0.0, 'topdown'), (np.pi / 4, '45deg'))
# The pre-grasp is only a 10 cm standoff via-point: a couple of cm of slop there
# is irrelevant, and holding it to the same 1 cm tolerance as the contact pose
# rejected otherwise-fine grasps (the 30° candidate missed by 17 mm). The CONTACT
# move keeps the strict default, since that's where the fingers actually close.
PREGRASP_LIN_TOL = 0.035   # m
PREGRASP_ANG_TOL = 0.12    # rad
# Budget for the whole deployed /skill/grasp call (--method skill). It is far
# larger than any single benchmark method's because the skill can walk up to
# MAX_GRASP_ATTEMPTS ranked candidates, each with its own multi-pass servo
# (skills/grasp.py: SERVO_DURATION_SEC x SERVO_MAX_ITER), before giving up.
SKILL_TIMEOUT_SEC = 300.0
# --box-source gt crop radius: keep head/wrist-cloud points within this distance
# of the object's ground-truth centroid (pelvis frame) as "the object". 12 cm
# comfortably contains a tabletop object like the CheesyBread wedge (~8 cm) plus
# a little support surface — matching what the gemini-box cloud picks up — while
# excluding neighbouring clutter. Widen for a larger target.
GT_CROP_R = 0.12


def _tilted_pose(fingertip_xyz, yaw, tilt=0.0):
    """graspgenx-frame Pose for a grasp `tilt` radians off straight-down, leaning
    outward along the robot->object direction so the gripper comes in from
    above-AND-behind. Fingertips land on `fingertip_xyz` (pelvis); the fingers
    close along the pelvis-XY direction `yaw`; the base sits TCP_DEPTH_M back
    along the approach.

    Why the tilt exists: the two extremes both fail on this robot. A straight-down
    grasp (tilt=0) is IK-unreachable — the H1-2 wrist pitch is clamped to ±0.4625
    rad (±26.5°), so the arm cannot point the gripper at the floor. A horizontal
    grasp (tilt=90°) is reachable but just sweeps an object off the counter. A
    30-45° tilt keeps enough downward component to lift while staying inside the
    wrist's range.
    """
    tip = np.asarray(fingertip_xyz, dtype=float)
    # Horizontal robot->object direction: the grasp leans out along it.
    u = np.array([tip[0], tip[1], 0.0])
    n = np.linalg.norm(u)
    u = u / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])

    z_axis = np.array([np.sin(tilt) * u[0], np.sin(tilt) * u[1], -np.cos(tilt)])
    z_axis /= np.linalg.norm(z_axis)

    # Closing axis: the requested yaw, projected perpendicular to the approach.
    c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    x_axis = c - np.dot(c, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:                 # yaw parallel to approach
        c = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        x_axis = c - np.dot(c, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    qx, qy, qz, qw = mat_to_quat(np.column_stack([x_axis, y_axis, z_axis]))
    base = tip - TCP_DEPTH_M * z_axis
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in base)
    p.orientation.x, p.orientation.y = float(qx), float(qy)
    p.orientation.z, p.orientation.w = float(qz), float(qw)
    return p


def _topdown_pose(fingertip_xyz, yaw):
    """Straight-down grasp (the tilt=0 special case)."""
    return _tilted_pose(fingertip_xyz, yaw, 0.0)


def _offset_along_z(pose, dist):
    """Translate `pose` by `dist` along its OWN +Z (approach) axis."""
    T = pose_to_matrix(pose)
    out = Pose()
    out.position.x = pose.position.x + float(T[0, 2] * dist)
    out.position.y = pose.position.y + float(T[1, 2] * dist)
    out.position.z = pose.position.z + float(T[2, 2] * dist)
    out.orientation = pose.orientation
    return out


def _robust_top_z(cloud):
    """Height of the object's top surface, robust to depth outliers.

    cloud[:, 2].max() is NOT safe here: mask-edge pixels straddle the object
    boundary and pick up background depth, which back-projects into stray points
    well above the object. On the cheese that pushed z.max() to 0.151 m when the
    real top was ~0.107 m — a max()-based grasp closed on thin air 2.4 cm ABOVE
    the object and never touched it. The visible cloud is essentially the top
    face, so a high percentile is both robust and the right estimate."""
    return float(np.percentile(cloud[:, 2], 75))


def _top_layer_pca(cloud):
    """PCA of the top slab of `cloud` (N,3 pelvis frame). Returns dict with the
    slab centroid, top z, XY principal axes (major first), extents along them,
    and the closing yaw that grips across each axis."""
    z = cloud[:, 2]
    top_z = _robust_top_z(cloud)          # NOT z.max() — see _robust_top_z
    top = cloud[z > top_z - TOP_LAYER_M]
    if len(top) < 3:
        top = cloud
    xy = top[:, :2]
    mu = xy.mean(axis=0)
    cov = np.cov((xy - mu).T) if len(xy) > 2 else np.eye(2)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]                     # columns: major, minor
    local = (xy - mu) @ vecs
    extents = local.max(axis=0) - local.min(axis=0)
    major_yaw = float(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return dict(
        centroid=np.array([mu[0], mu[1], float(top[:, 2].mean())]),
        top_z=top_z,
        major_yaw=major_yaw,
        minor_yaw=major_yaw + np.pi / 2.0,
        major_extent=float(extents[0]),
        minor_extent=float(extents[1]),
        n_points=int(len(top)),
    )


class GraspBenchmark(SkillsBase):
    """SkillsBase gives us the frame_task IK client, per-arm grippers, the
    gemini/sam/graspgen services, TF, and the head-camera caches. This adds the
    wrist camera, the ground-truth object-pose feed, and the /named_config
    client, then runs one benchmark episode from the main thread."""

    def __init__(self, arm, gt_name='', box_source='gemini'):
        super().__init__(node_name='grasp_benchmark')
        self.arm = arm
        self._gt_name = gt_name
        self._box_source = box_source
        # Whether the benchmark's own arm moves go through the OMPL planner
        # (do_plan=True) or drive the frame_task IK directly (False). The deployed
        # skill plans its pre-grasp, so True matches it; False isolates whether the
        # planner (vs. raw IK reachability) is what rejects a grasp.
        self._do_plan = True
        # Metres to drive each grasp deeper along its approach axis before executing
        # (mirrors skills/grasp.py GRASP_OFFSET) — a grasp-quality knob for the
        # ~1cm-short contact that leaves small objects insecure. Set via --grasp-offset.
        self._grasp_offset = 0.0
        self.named_config_cli = ActionClient(
            self, NamedConfig, '/named_config', callback_group=self._cb_group)

        wrist_ns = f'/realsense/{arm}_hand'
        self._wrist_color = None
        self._wrist_depth = None
        self._wrist_info = None
        self.create_subscription(
            CompressedImage, f'{wrist_ns}/color/image_raw/compressed',
            lambda m: setattr(self, '_wrist_color', m), qos_profile_sensor_data)
        self.create_subscription(
            CompressedImage,
            f'{wrist_ns}/aligned_depth_to_color/image_raw/compressedDepth',
            lambda m: setattr(self, '_wrist_depth', m), qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, f'{wrist_ns}/color/camera_info',
            lambda m: setattr(self, '_wrist_info', m), qos_profile_sensor_data)

        self._obj_poses = {}
        self.create_subscription(
            String, '/robocasa/object_poses', self._on_obj_poses, 10)
        self._task_success = False
        self.create_subscription(
            Bool, '/robocasa/success',
            lambda m: setattr(self, '_task_success', bool(m.data)), 10)

    def _on_obj_poses(self, msg):
        try:
            self._obj_poses = json.loads(msg.data)
        except ValueError:
            pass

    # ------------------------------------------------------------- ground truth
    def gt_pos(self, name):
        """Target object's ground-truth MuJoCo-world position, or None."""
        v = self._obj_poses.get(name)
        return np.array(v[:3], dtype=float) if v else None

    def gt_pos_pelvis(self, name):
        """Ground-truth object position expressed in the PELVIS frame — the frame
        perception and the controller actually work in. Uses the '__pelvis__'
        entry the sim's MeasurementBridge publishes alongside the objects. This is
        the only way to check a perceived cloud against truth."""
        obj = self.gt_pos(name)
        pel = self._obj_poses.get('__pelvis__')
        if obj is None or not pel:
            return None
        p = np.array(pel[:3], dtype=float)
        qw, qx, qy, qz = (float(v) for v in pel[3:7])
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])
        return R.T @ (obj - p)

    # ------------------------------------------------------------------- homing
    def go_home(self, duration_sec=8.0):
        goal = NamedConfig.Goal()
        goal.config_name = 'home'
        goal.duration.sec = int(duration_sec)
        resp = self._send_action(self.named_config_cli, goal,
                                 result_timeout=duration_sec + 15.0)
        ok = resp is not None and resp.result.success
        if not ok:
            self.get_logger().warn('named_config home did not report success')
        return ok

    # ------------------------------------------------------ wrist-camera cloud
    def wrist_cloud(self, text, target_frame='pelvis'):
        """SAM-segment the wrist camera's latest frame with `text` and
        back-project through its aligned depth into (N,3) points in
        `target_frame`. TF for the wrist optical frame is broadcast by the sim
        bridge (pelvis -> <arm>_hand_camera_color_optical_frame)."""
        if self._wrist_color is None or self._wrist_depth is None or self._wrist_info is None:
            self.get_logger().error('wrist_cloud: no wrist camera data yet')
            return None
        if self._box_source == 'gt':
            # No detector: keep every wrist-depth pixel and crop the resulting
            # cloud around the GT centroid below (mask = all valid pixels).
            mask = None
        else:
            # Box the WRIST frame too (not just the head): text-only SAM finds
            # nothing, so without this the scan silently returns None and the
            # method falls back to the head cloud — i.e. it stops being the
            # wrist-camera method at all.
            box = self._gemini_box(text, image=self._wrist_color,
                                   width=self._wrist_info.width,
                                   height=self._wrist_info.height)
            mask_msg = self.segment(text=text, positive_boxes=box,
                                    image=self._wrist_color)
            if mask_msg is None:
                return None
            mask = (np.frombuffer(bytes(mask_msg.data), dtype=np.uint8)
                    .reshape(mask_msg.height, mask_msg.width) > 127)
        try:
            depth = decode_compressed_depth_image(self._wrist_depth).astype(np.float32) / 1000.0
        except (ValueError, TypeError) as e:
            self.get_logger().error(f'wrist_cloud: depth decode failed: {e}')
            return None
        if mask is None:
            mask = np.ones(depth.shape, dtype=bool)
        elif depth.shape != mask.shape:
            self.get_logger().error(
                f'wrist_cloud: mask {mask.shape} != depth {depth.shape}')
            return None
        info = self._wrist_info
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        pts_cam = deproject_mask(mask, depth, fx, fy, cx, cy, 0.05, 2.0)
        if len(pts_cam) < 20:
            self.get_logger().warn(f'wrist_cloud: only {len(pts_cam)} points')
            return None
        cam_frame = self._wrist_depth.header.frame_id or info.header.frame_id
        from rclpy.time import Time
        from rclpy.duration import Duration as RclpyDuration
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame, cam_frame, Time(), timeout=RclpyDuration(seconds=1.0))
        except Exception as e:
            self.get_logger().error(f'wrist_cloud: TF {cam_frame} failed: {e}')
            return None
        pts = transform_points(pts_cam, transform_to_matrix(tf.transform)).astype(np.float32)
        if self._box_source == 'gt' and target_frame == 'pelvis':
            c = self.gt_pos_pelvis(self._gt_name)
            if c is not None:
                pts = pts[np.linalg.norm(pts - c[None, :], axis=1) <= GT_CROP_R]
                if len(pts) < 20:
                    self.get_logger().warn(
                        f'wrist_cloud[gt]: only {len(pts)} points within '
                        f'{GT_CROP_R * 100:.0f}cm of GT')
                    return None
        return pts

    # ------------------------------------------------------ debug snapshots
    def _debug_wrist_snapshot(self, tag):
        """Dump the eye-in-hand view at a moment of interest (e.g. just before
        the fingers close). If the target isn't sitting between the fingers here,
        the grasp pose / gripper-frame convention is off — not the gripper."""
        import os
        import cv2
        out_dir = '/home/code/core_ws/benchmark_results/debug'
        for name, msg in (('wrist', self._wrist_color), ('head', self.latest_image())):
            if msg is None:
                continue
            img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                continue
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f'{tag}_{name}.png')
            cv2.imwrite(path, img)
            self.get_logger().info(f'debug snapshot -> {path}')

    # -------------------------------------------------------- head-camera cloud
    def head_cloud(self, text):
        """SAM (text prompt) on the head camera -> object cloud in pelvis."""
        mask = self.segment(text=text)
        if mask is None:
            return None
        return self.mask_to_cloud(mask, target_frame='pelvis')

    # ------------------------------------------------------- grasp execution
    def execute(self, candidates, width_m=None):
        """Walk `candidates` [(Pose in pelvis, label), ...] best-first: pre-open,
        drive the graspgenx frame to the standoff then to the grasp, close.
        Returns (executed_ok, chosen_index, label)."""
        frame = GRASP_FRAMES[self.arm]
        open_mm = min((width_m * 1000.0 + WIDTH_MARGIN_M * 1000.0) if width_m
                      else 106.0, 106.0)
        if not self.set_gripper(self.arm, open_mm):
            return False, -1, 'gripper pre-open failed'
        for i, (raw_pose, label) in enumerate(candidates[:MAX_ATTEMPTS]):
            # Drive the grasp _grasp_offset m deeper along its own approach axis
            # (mirrors the skill's GRASP_OFFSET): the contact servo tends to stop
            # ~1 cm short, so a small positive offset closes the fingers ONTO a
            # small object instead of in front of it. The pre-grasp standoff is
            # measured from the shifted grasp, so both move together.
            pose = _offset_along_z(raw_pose, self._grasp_offset)
            approach = _offset_along_z(pose, -APPROACH_DIST_M)
            self.get_logger().info(f'candidate {i} ({label}): trying pre-grasp')
            if not self.move_frame_to(frame, approach, duration_sec=APPROACH_SEC,
                                      lin_tol=PREGRASP_LIN_TOL,
                                      ang_tol=PREGRASP_ANG_TOL,
                                      do_plan=self._do_plan):
                self.get_logger().warn(f'candidate {i} pre-grasp unreachable')
                continue
            # Contact move is best-effort: we already committed to a reachable
            # standoff, so close even if the last few mm don't converge. Log the
            # residual — a large one means the fingers close off-target and tend
            # to knock the object rather than grip it.
            contact_ok = self.move_frame_to(frame, pose, duration_sec=CONTACT_SEC,
                                            do_plan=self._do_plan)
            if not contact_ok:
                self.get_logger().warn(
                    f'candidate {i}: contact move did not fully converge; '
                    'closing anyway (grasp may be off-target)')
            time.sleep(SETTLE_SEC)
            self._debug_wrist_snapshot(f'contact_cand{i}')
            if not self.close_gripper(self.arm, 1.0):
                return False, i, 'gripper close failed'
            self._last_grasp_pose = pose
            return True, i, label
        return False, -1, 'no reachable candidate'

    def lift(self):
        pose = Pose()
        pose.position.x = self._last_grasp_pose.position.x
        pose.position.y = self._last_grasp_pose.position.y
        pose.position.z = self._last_grasp_pose.position.z + LIFT_DIST_M
        pose.orientation = self._last_grasp_pose.orientation
        return self.move_frame_to(GRASP_FRAMES[self.arm], pose,
                                  duration_sec=LIFT_SEC, do_plan=self._do_plan)

    def frame_pose_pelvis(self, frame):
        """Live pose of `frame` in the pelvis frame, straight off TF."""
        try:
            tf = self.tf_buffer.lookup_transform(
                'pelvis', frame, Time(), timeout=RclpyDuration(seconds=2.0))
        except TransformException as e:
            self.get_logger().error(f'TF pelvis <- {frame!r} failed: {e}')
            return None
        pose = Pose()
        pose.position.x = tf.transform.translation.x
        pose.position.y = tf.transform.translation.y
        pose.position.z = tf.transform.translation.z
        pose.orientation = tf.transform.rotation
        return pose

    def lift_from_current(self):
        """Lift straight up from wherever the gripper ACTUALLY ended up.

        The benchmark's own methods lift from the pose they commanded
        (self._last_grasp_pose), but the deployed skill grasps in ANOTHER process
        and its chosen pose never crosses the action boundary, so read the driven
        frame's live pose off TF instead."""
        pose = self.frame_pose_pelvis(GRASP_FRAMES[self.arm])
        if pose is None:
            return False
        pose.position.z += LIFT_DIST_M
        return self.move_frame_to(GRASP_FRAMES[self.arm], pose,
                                  duration_sec=LIFT_SEC, do_plan=self._do_plan)

    # ------------------------------------------------------------- the methods
    def run_skill(self, obj_text):
        """Dispatch the DEPLOYED /skill/grasp action and report whether it grasped.

        Unlike the method_* functions below, this measures skills/grasp.py exactly
        as pick_place invokes it — gemini/yolo box -> SAM -> GraspGenX -> priority
        tier + diversity re-rank -> walk-the-ranked-candidates servo — rather than a
        benchmark reimplementation of it. That makes it the reference the other
        methods are worth comparing against, and the only one that exercises the
        skill's retry loop.

        It plans AND executes internally, so it does not fit the plan -> execute
        split the other methods use, and it deliberately does not lift (see
        SkillGrasp.action); the caller lifts with lift_from_current().
        Returns (grasped, message)."""
        goal = SkillGrasp.Goal()
        goal.target_object = obj_text
        goal.arm = self.arm
        goal.timeout.sec = int(SKILL_TIMEOUT_SEC)
        resp = self._send_action(self.grasp_skill_cli, goal,
                                 accept_timeout=30.0,
                                 result_timeout=SKILL_TIMEOUT_SEC + 30.0)
        if resp is None:
            return False, 'no result from /skill/grasp (server up? timed out?)'
        return bool(resp.result.success), str(resp.result.message)

    def method_centroid(self, obj_text):
        """Naive baseline: object-cloud centroid, fixed top-down grasp, fingers
        closing across the pelvis-Y axis (whatever the object's orientation)."""
        cloud = self._boxed_cloud(obj_text)
        if cloud is None:
            return None
        # median, not mean: mask-edge outliers drag the mean off the object.
        c = np.median(cloud, axis=0)
        tip = np.array([c[0], c[1],
                        max(_robust_top_z(cloud) - FINGER_SINK_M,
                            float(np.percentile(cloud[:, 2], 5)))])
        # Walk the tilts (45 -> 30 -> straight-down); the last entry keeps a second
        # yaw so an IK-blocked primary still has a fallback.
        cands = [(_tilted_pose(tip, np.pi / 2, t), f'centroid-y-{name}')
                 for t, name in GRASP_TILTS]
        cands.append((_tilted_pose(tip, 0.0, np.pi / 4), 'centroid-x-45deg'))
        return dict(candidates=cands, width_m=None,
                    meta=dict(n_points=int(len(cloud)), centroid=c.tolist()))

    def method_topdown_antipodal(self, obj_text):
        """Wrist-camera top-down antipodal: park the wrist above the head-cam
        centroid, re-segment from the wrist camera, PCA the top layer, close
        across the minor (narrowest) axis at the slab centroid."""
        rough = self._boxed_cloud(obj_text)
        if rough is None:
            return None
        c = rough.mean(axis=0)
        scan_tip = np.array([c[0], c[1], rough[:, 2].max() + SCAN_HEIGHT_M - TCP_DEPTH_M])
        scan = _topdown_pose(scan_tip, np.pi / 2)
        if not self.move_frame_to(GRASP_FRAMES[self.arm], scan,
                                  duration_sec=APPROACH_SEC, do_plan=self._do_plan):
            self.get_logger().warn('antipodal: scan pose unreachable, using head cloud')
            cloud = rough
        else:
            time.sleep(1.0)                       # let a fresh wrist frame land
            cloud = self.wrist_cloud(obj_text)
            if cloud is None:
                self.get_logger().warn('antipodal: wrist segmentation failed, using head cloud')
                cloud = rough
        pca = _top_layer_pca(cloud)
        tip = np.array([pca['centroid'][0], pca['centroid'][1],
                        max(pca['top_z'] - FINGER_SINK_M, cloud[:, 2].min())])
        # The method's idea is the CLOSING axis (PCA minor = narrowest span); the
        # approach tilt is orthogonal to that, so walk the tilts on the minor axis
        # first and keep the major axis as a fallback.
        cands = [(_tilted_pose(tip, pca['minor_yaw'], t), f'antipodal-minor-{name}')
                 for t, name in GRASP_TILTS]
        cands.append((_tilted_pose(tip, pca['major_yaw'], np.pi / 4),
                      'antipodal-major-45deg'))
        return dict(candidates=cands, width_m=pca['minor_extent'],
                    meta={k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in pca.items()})

    def method_graspgenx(self, obj_text):
        """GraspGenX as ranked: gemini box -> SAM -> cloud -> planner, executed
        strictly best-first by the model's own scores."""
        cloud = self._boxed_cloud(obj_text)
        if cloud is None:
            return None
        scene = self.scene_to_cloud(target_frame='pelvis')
        resp = self.plan_grasp(cloud, gripper_name='magpie', frame='pelvis',
                               scene_cloud=scene, arm=self.arm)
        if resp is None:
            return None
        cands = [(g.pose, f'ggx-{i}(s={resp.scores[i]:.2f})')
                 for i, g in enumerate(resp.grasps)]
        return dict(candidates=cands, width_m=float(resp.gripper_width),
                    meta=dict(n_grasps=len(cands),
                              scores=[float(s) for s in resp.scores[:5]]))

    def method_vlm_judge(self, obj_text):
        """VLM-as-judge over GraspGenX + PCA candidates (magpie pickup-pipeline
        style): render each candidate's closing axis on its own head-camera tile,
        ask Gemini to pick, execute the choice first with the rest as fallback."""
        cloud = self._boxed_cloud(obj_text)
        if cloud is None:
            return None
        scene = self.scene_to_cloud(target_frame='pelvis')
        pca = _top_layer_pca(cloud)
        tip = np.array([pca['centroid'][0], pca['centroid'][1],
                        max(pca['top_z'] - FINGER_SINK_M, cloud[:, 2].min())])
        # 45° tilt: a straight-down PCA candidate is IK-unreachable on this arm.
        cands = [(_tilted_pose(tip, pca['minor_yaw'], np.pi / 4), 'pca-short_side'),
                 (_tilted_pose(tip, pca['major_yaw'], np.pi / 4), 'pca-long_side')]
        resp = self.plan_grasp(cloud, gripper_name='magpie', frame='pelvis',
                               scene_cloud=scene, arm=self.arm)
        width = pca['minor_extent']
        if resp is not None:
            for i, g in enumerate(resp.grasps[:2]):
                cands.append((g.pose, f'ggx-{i}(s={resp.scores[i]:.2f})'))
            width = float(resp.gripper_width)
        choice, reason = self._judge(cands, tip, obj_text)
        order = [choice] + [i for i in range(len(cands)) if i != choice]
        return dict(candidates=[cands[i] for i in order], width_m=width,
                    meta=dict(judged=cands[choice][1], reason=reason,
                              labels=[l for _, l in cands]))

    def _gemini_box(self, obj_text, image=None, width=None, height=None):
        """gemini -> pixel-xyxy box around `obj_text` in `image` (default: the head
        frame), or None. EVERY method routes its SAM call through this: text-only
        SAM cannot reliably ground these RoboCasa assets (it returns no mask at all
        for 'cheese'), so the box is what makes segmentation work. Keeping the
        detection identical across methods is also what makes the benchmark a fair
        comparison of grasp SYNTHESIS rather than of who got lucky with SAM."""
        from .skills.grasp import GEMINI_GRASP_PROMPT
        from .perception_utils import extract_json
        if width is None or height is None:
            info = self.latest_caminfo()
            if not info or not info.width:
                return None
            width, height = info.width, info.height
        txt = self.query_gemini(GEMINI_GRASP_PROMPT.format(obj=obj_text),
                                image=image, timeout_sec=120.0)
        data = extract_json(txt)
        entry = data[0] if isinstance(data, list) and data else (
            data if isinstance(data, dict) else None)
        if not (isinstance(entry, dict) and 'box_2d' in entry):
            self.get_logger().warn(f'gemini returned no box for {obj_text!r}')
            return None
        try:
            y1, x1, y2, x2 = (float(v) for v in entry['box_2d'])
            px1, px2 = sorted((x1 / 1000.0 * width, x2 / 1000.0 * width))
            py1, py2 = sorted((y1 / 1000.0 * height, y2 / 1000.0 * height))
            return [px1, py1, px2, py2]
        except (ValueError, TypeError):
            return None

    def _gt_cloud(self):
        """Object cloud straight from ground truth: the whole head-camera cloud
        cropped to a ball of radius GT_CROP_R around the target's GT centroid
        (pelvis frame). Skips gemini AND SAM entirely, so a seeded episode yields
        a FIXED, reproducible object cloud and the benchmark measures grasp
        SYNTHESIS with detection removed as a variable. Requires --gt-name.

        The crop keeps a little of the support surface around the object, exactly
        as the gemini box path does (box_to_cloud picks up whatever shows inside
        the box), so the two cloud sources stay comparable rather than one being
        pristine and the other not."""
        c = self.gt_pos_pelvis(self._gt_name)
        if c is None:
            self.get_logger().error(
                f'_gt_cloud: no ground truth for {self._gt_name!r} — is the sim '
                'publishing /robocasa/object_poses and is --gt-name correct?')
            return None
        scene = self.scene_to_cloud(target_frame='pelvis')
        if scene is None:
            return None
        obj = scene[np.linalg.norm(scene - c[None, :], axis=1) <= GT_CROP_R]
        self.get_logger().info(
            f'[gt-cloud] {len(obj)}/{len(scene)} head-cloud pts within '
            f'{GT_CROP_R * 100:.0f}cm of GT {self._gt_name!r} '
            f'(pelvis {c[0]:.3f},{c[1]:.3f},{c[2]:.3f})')
        if len(obj) < MIN_GRASP_POINTS:
            self.get_logger().warn(
                f'_gt_cloud: only {len(obj)} points (< {MIN_GRASP_POINTS}); '
                'the object may be occluded or outside the head-camera view')
            return None
        return obj.astype(np.float32)

    def _boxed_cloud(self, obj_text):
        """Object cloud for a method. --box-source gt uses the ground-truth crop
        (_gt_cloud); the default gemini path is: gemini box -> SAM (text+box) ->
        pelvis cloud. Shared by ALL FOUR methods so they differ only in how they
        synthesize a grasp, not in how the object is perceived."""
        if self._box_source == 'gt':
            return self._gt_cloud()
        box = self._gemini_box(obj_text)
        mask = self.segment(text=obj_text, positive_boxes=box)
        if mask is None:
            return None
        self._debug_mask_overlay(mask, box, 'perceive')
        cloud = self.mask_to_cloud(mask, target_frame='pelvis')
        if cloud is not None:
            c = cloud.mean(axis=0)
            self.get_logger().info(
                f'[debug] perceived centroid(pelvis)=({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})'
                f'  n={len(cloud)}  z-range=[{cloud[:, 2].min():.3f}, {cloud[:, 2].max():.3f}]')
            gt = self.gt_pos_pelvis(self._gt_name)
            if gt is not None:
                err = c - gt
                self.get_logger().info(
                    f'[debug] GROUND TRUTH(pelvis)=({gt[0]:.3f}, {gt[1]:.3f}, {gt[2]:.3f})'
                    f'  ERROR=({err[0]:+.3f}, {err[1]:+.3f}, {err[2]:+.3f}) '
                    f'|err|={np.linalg.norm(err):.3f} m')
        return cloud

    def _debug_mask_overlay(self, mask_msg, box, tag):
        """Draw the gemini box + SAM mask on the head frame so we can see WHAT
        got segmented (the target, or the surface it sits on)."""
        import os
        import cv2
        img_msg = self.latest_image()
        if img_msg is None or mask_msg is None:
            return
        bgr = cv2.imdecode(np.frombuffer(bytes(img_msg.data), np.uint8),
                           cv2.IMREAD_COLOR)
        if bgr is None:
            return
        m = (np.frombuffer(bytes(mask_msg.data), np.uint8)
             .reshape(mask_msg.height, mask_msg.width) > 127)
        if m.shape[:2] == bgr.shape[:2]:
            overlay = bgr.copy()
            overlay[m] = (0, 0, 255)                      # mask in red
            bgr = cv2.addWeighted(overlay, 0.5, bgr, 0.5, 0)
        if box:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)   # gemini box
        out_dir = '/home/code/core_ws/benchmark_results/debug'
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f'{tag}_maskoverlay.png')
        cv2.imwrite(path, bgr)
        self.get_logger().info(
            f'debug mask overlay -> {path} (mask px={int(m.sum())}, box={box})')

    # ----------------------------------------------------------- the VLM judge
    def _judge(self, cands, tip, obj_text):
        """Tile one annotated head-camera image per candidate (its finger-closing
        axis drawn through the grasp point) and have Gemini pick. Returns
        (index, reason); falls back to index 0 on any failure."""
        import cv2
        img_msg = self.latest_image()
        if img_msg is None:
            return 0, 'no image; defaulted'
        bgr = cv2.imdecode(np.frombuffer(bytes(img_msg.data), np.uint8),
                           cv2.IMREAD_COLOR)
        if bgr is None:
            return 0, 'decode failed; defaulted'
        try:
            from rclpy.time import Time
            from rclpy.duration import Duration as RclpyDuration
            tf = self.tf_buffer.lookup_transform(
                HEAD_OPTICAL_FRAME, 'pelvis', Time(),
                timeout=RclpyDuration(seconds=1.0))
        except Exception as e:
            self.get_logger().warn(f'judge: TF failed ({e}); defaulted')
            return 0, 'tf failed; defaulted'
        T = transform_to_matrix(tf.transform)
        info = self.latest_caminfo()
        K = np.array(info.k).reshape(3, 3)

        def project(p3):
            pc = T[:3, :3] @ np.asarray(p3, float) + T[:3, 3]
            if pc[2] <= 0.01:
                return None
            uv = K @ (pc / pc[2])
            return int(round(uv[0])), int(round(uv[1]))

        letters = 'ABCD'
        tiles = []
        half = 0.06
        for i, (pose, label) in enumerate(cands[:4]):
            Tp = pose_to_matrix(pose)
            close_axis = Tp[:3, 0]                       # +X = closing axis
            contact = Tp[:3, 3] + Tp[:3, 2] * TCP_DEPTH_M  # fingertip point
            a = project(contact - close_axis * half)
            b = project(contact + close_axis * half)
            tile = bgr.copy()
            if a and b:
                cv2.line(tile, a, b, (0, 255, 0), 2)
                for e in (a, b):
                    cv2.circle(tile, e, 4, (0, 165, 255), -1)
            cv2.putText(tile, letters[i], (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 255), 2, cv2.LINE_AA)
            tiles.append(tile)
        while len(tiles) % 2:
            tiles.append(np.zeros_like(bgr))
        rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
        sheet = np.vstack(rows)
        ok, buf = cv2.imencode('.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return 0, 'encode failed; defaulted'
        sheet_msg = CompressedImage()
        sheet_msg.format = 'jpeg'
        sheet_msg.data = buf.tobytes()

        n = min(len(cands), 4)
        prompt = (
            f'The image is a contact sheet of {n} candidate grasps for the '
            f'"{obj_text}", one per tile, labelled {"/".join(letters[:n])} in the '
            f'top-left corner (tiles read left-to-right, top-to-bottom). In each '
            f'tile the GREEN line is the two-finger gripper CLOSING axis and the '
            f'ORANGE dots are where the fingers press. A GOOD grasp closes across '
            f'the NARROWEST span onto two FLAT opposing faces; a BAD grasp runs '
            f'diagonally into corners or across the widest span.\n'
            f'Reply EXACTLY:\nCHOICE: <{"|".join(letters[:n])}>\n'
            f'REASON: <one sentence>')
        txt = self.query_gemini(prompt, image=sheet_msg, timeout_sec=90.0)
        if not txt:
            return 0, 'gemini failed; defaulted'
        choice, reason = 0, ''
        for line in txt.splitlines():
            up = line.strip().upper()
            if up.startswith('CHOICE:'):
                c = up.split(':', 1)[1].strip()
                for k in range(n):
                    if c.startswith(letters[k]):
                        choice = k
                        break
            elif up.startswith('REASON:'):
                reason = line.strip().split(':', 1)[1].strip()
        return choice, reason


METHODS = ('centroid', 'topdown_antipodal', 'graspgenx', 'vlm_judge', 'skill')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--method', required=True, choices=METHODS)
    ap.add_argument('--object', default='wedge of cheese',
                    help='text prompt for gemini/SAM')
    ap.add_argument('--gt-name', default='cheese',
                    help='task cfg object name in /robocasa/object_poses')
    ap.add_argument('--arm', default='right', choices=('left', 'right'))
    ap.add_argument('--out', default='', help='result JSON path')
    ap.add_argument('--success-dz', type=float, default=0.08,
                    help='ground-truth lift height that counts as success [m]')
    ap.add_argument('--grasp-offset', type=float, default=0.0,
                    help='drive each grasp this many metres deeper along its '
                         'approach axis before executing (grasp-quality knob for '
                         'the ~1cm-short contact; positive = deeper into object)')
    ap.add_argument('--no-plan', action='store_true',
                    help='drive the benchmark arm moves via frame_task IK directly '
                         'instead of the OMPL planner (do_plan=False) — isolates '
                         'whether the planner rejects grasps that raw IK can reach')
    ap.add_argument('--box-source', default='gemini', choices=('gemini', 'gt'),
                    help="how methods perceive the object: 'gemini' (gemini box "
                         "-> SAM) or 'gt' (crop the head/wrist cloud around the "
                         "ground-truth centroid, NO detector — removes detection "
                         "as a variable and costs zero API calls). NOTE: 'vlm_judge' "
                         "still calls gemini for its JUDGE step, and 'skill' runs "
                         "the deployed skill which detects with gemini internally, "
                         "so --box-source gt does not make either detector-free.")
    args = ap.parse_args()
    if args.box_source == 'gt' and args.method in ('vlm_judge', 'skill'):
        print(f"[grasp_benchmark] WARNING: --box-source gt does not remove gemini "
              f"from method {args.method!r} (it still calls gemini internally).")

    rclpy.init()
    node = GraspBenchmark(args.arm, gt_name=args.gt_name,
                          box_source=args.box_source)
    node._do_plan = not args.no_plan
    node._grasp_offset = args.grasp_offset
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    rec = dict(method=args.method, object=args.object, gt_name=args.gt_name,
               arm=args.arm, success=False, error='')
    try:
        # sim + ground-truth feed up?
        for _ in range(100):
            if node.latest_image() is not None and node.gt_pos(args.gt_name) is not None:
                break
            time.sleep(0.2)
        base = node.gt_pos(args.gt_name)
        if base is None:
            raise RuntimeError(
                f'no ground truth for {args.gt_name!r} on /robocasa/object_poses '
                '(is the sim running with the updated measurement_bridge?)')
        if node.latest_image() is None:
            raise RuntimeError('no head-camera image (is the sim publishing?)')

        node.go_home()
        rec['gt_start'] = base.tolist()

        t0 = time.monotonic()
        if args.method == 'skill':
            # The deployed skill plans AND executes behind one action call, so
            # there is no plan/execute split to time separately; charge it all to
            # exec_time_s and lift from the pose it actually reached.
            ok, msg = node.run_skill(args.object)
            rec.update(plan_time_s=0.0, meta={}, executed=ok,
                       chosen_index=-1, chosen_label=msg)
            if ok:
                node.lift_from_current()
                time.sleep(HOLD_SEC)
            rec['exec_time_s'] = round(time.monotonic() - t0, 2)
        else:
            plan = getattr(node, f'method_{args.method}')(args.object)
            rec['plan_time_s'] = round(time.monotonic() - t0, 2)
            if plan is None:
                raise RuntimeError('grasp synthesis produced no candidates')
            rec['meta'] = plan['meta']

            t1 = time.monotonic()
            ok, idx, label = node.execute(plan['candidates'], plan['width_m'])
            rec.update(executed=ok, chosen_index=idx, chosen_label=label)
            if ok:
                node.lift()
                time.sleep(HOLD_SEC)
            rec['exec_time_s'] = round(time.monotonic() - t1, 2)

        now = node.gt_pos(args.gt_name)
        dz = float(now[2] - base[2]) if now is not None else 0.0
        rec.update(gt_end=(now.tolist() if now is not None else None),
                   lift_dz_m=round(dz, 4),
                   success=bool(ok and dz >= args.success_dz),
                   task_success=node._task_success)
    except Exception as e:                                    # noqa: BLE001
        rec['error'] = str(e)
        node.get_logger().error(f'benchmark episode failed: {e}')
    finally:
        line = json.dumps(rec, indent=2)
        print(line)
        if args.out:
            import os
            os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
            with open(args.out, 'w') as f:
                f.write(line + '\n')
        rclpy.shutdown()
        spin.join(timeout=2.0)


if __name__ == '__main__':
    main()
