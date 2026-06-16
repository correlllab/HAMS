#!/usr/bin/env python3
"""Shared infrastructure for the h12 skills node.

`SkillsBase` holds everything every skill needs: the service/action clients
(vision pipeline, grippers, frame_task, nav2), the TF buffer, the head-camera
color cache, the optional Gemini Robotics client, the per-skill action clients
(self.skill_clients, used for composition like pick_place -> grasp), and the
helper layers built on top of them (detection, motion primitives, future
plumbing). Each skill lives in its own module under skills/ as a mixin class
whose _exec_* method operates on `self`; SkillsNode (node.py) multiply-inherits
from SkillsBase plus every skill mixin, so the skill bodies resolve
self.detect_grasp/self.move_frame_to/... against the combined node at runtime.

Motions are first-pass and deliberately naive, in the spirit of open_fridge.py:
straight-line gripper targets in the pelvis frame, fixed approach offsets along
+x (handle assumed roughly facing the robot), and simple arcs/twists where a
hinge or rotation axis is involved.

SkillGrasp plans an ANTIPODAL grasp from the detection point cloud (PCA of the
footprint perpendicular to the approach axis: fingers close along the minor
axis, wrist rolled to match, gripper pre-opened to the measured width). Every
skill that must hold something composes SkillGrasp through this node's own
client (self.skill_clients['grasp']) rather than closing the gripper itself.

Unlike open_fridge.py (a run-to-completion script that owns the spin loop with
rclpy.spin_until_future_complete), this node executes skills *inside* action
server callbacks while a MultiThreadedExecutor spins, so all inner service and
action calls wait on futures with an event instead of spinning.
"""

import math
import os
import struct
import threading
import time
from functools import partial

import numpy as np
import rclpy
from rclpy.action import ActionClient, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration as RclpyDuration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PointStamped, Pose, PoseStamped, Quaternion
from sensor_msgs.msg import CompressedImage

from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_point

# Gemini Robotics (google-genai). Optional: the node still serves skills without
# it; only Gemini-backed reasoning is disabled if the SDK or key is missing.
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

from custom_ros_messages.srv import UpdateTrackedObject, UpdateBeliefs, Query
from custom_ros_messages.action import FrameTask
from custom_ros_messages.action import (
    SkillCloseDoor, SkillOpenDoor, SkillCloseLid, SkillOpenLid,
    SkillNavigate, SkillGrasp, SkillPickPlace, SkillPress, SkillSlideRack,
    SkillTurnLever, SkillTwistKnob,
)
from nav2_msgs.action import NavigateToPose
from magpie_msgs.srv import SetGripperPosition


CAMERA_NS = '/realsense/head'
# Color image the mujoco bridge / realsense publishes (JPEG, sensor QoS); the
# same topic the vision pipeline subscribes to. JPEG bytes feed Gemini directly.
COLOR_IMAGE_TOPIC = f'{CAMERA_NS}/color/image_raw/compressed'
# Gemini Robotics ER model (matches vision_pipeline/config.py "gemini_model").
GEMINI_MODEL = 'gemini-robotics-er-1.6-preview'
# Grip-site frames at the gripper closure point: fixed children of the wrist
# yaw links with the same orientation, pushed forward to the fingertips, so
# frame_task targets are the grasp point itself (no reach offset needed).
ARM_FRAMES = {'left': 'left_grip_site', 'right': 'right_grip_site'}
GRIPPER_SRVS = {'left': '/gripper/left/set_position',
                'right': '/gripper/right/set_position'}

# Gripper travel (mm), matching slider_debugger / the magpie hands.
GRIPPER_OPEN_MM = 85.0
GRIPPER_CLOSED_MM = 0.0

# Antipodal grasping. The parallel-jaw fingers are assumed to close along the
# wrist's +y axis at identity orientation; the wrist rolls about +x (the
# approach axis) so they close along the minor axis of the detection cloud's
# y-z footprint, landing both contacts on opposing surfaces.
MIN_GRASP_POINTS = 20          # below this, fall back to a centered grasp
GRASP_WIDTH_MARGIN_MM = 25.0   # pre-open this much wider than the object
GRASP_PREOPEN_MIN_MM = 20.0    # never pre-open narrower than this

# Approach geometry along pelvis +x: frame_task targets the grip-site frames
# (the gripper closure point), so "contact" places the gripper at the detected
# point itself and "approach" backs that off by APPROACH_BACKOFF.
APPROACH_BACKOFF = 0.25

# open_door pull when goal.pull_distance == 0 [m]. open_fridge.py pulls the
# wrist all the way back to the pelvis plane (x=0, typically 0.5-0.8 m of
# travel) despite its "25 cm" comment; 0.5 m approximates a full opening.
DEFAULT_PULL_DISTANCE = 0.5
PUSH_DEPTH = 0.10              # close_door push past the handle plane [m]
LID_HINGE_RADIUS = 0.30        # assumed handle-to-hinge distance for lids [m]
DEFAULT_OPEN_ANGLE = math.pi / 2   # lid opening when goal.open_angle == 0 [rad]
DEFAULT_TURN_ANGLE = math.pi / 2   # lever/knob turn when goal.angle == 0 [rad]
PRESS_DEPTH = 0.05             # press past the detected button plane [m]
SLIDE_DISTANCE = 0.30          # rack travel for slide_rack [m]
PLACE_HOVER = 0.15             # hover height above a place target [m]
NAV_STANDOFF = 0.75            # stop this far in front of a navigate target [m]

DETECT_SETTLE_SEC = 2.0        # wait after update_beliefs (open_fridge used 2 s)
GRIP_SETTLE_SEC = 2.0          # wait after closing the gripper on something
DEFAULT_SKILL_TIMEOUT = 120.0  # used when goal.timeout is zero [s]

# Map-frame presets for SkillNavigate: name -> (x, y, yaw). Targets not listed
# here are detected via the vision pipeline and approached to NAV_STANDOFF.
NAMED_LOCATIONS = {}

# Skill action clients: name -> (action type, action server name). The same
# table drives the action servers SkillsNode provides.
SKILL_ACTIONS = {
    'close_door':  (SkillCloseDoor,  '/skill/close_door'),
    'open_door':   (SkillOpenDoor,   '/skill/open_door'),
    'close_lid':   (SkillCloseLid,   '/skill/close_lid'),
    'open_lid':    (SkillOpenLid,    '/skill/open_lid'),
    'navigate':    (SkillNavigate,   '/skill/navigate'),
    'grasp':       (SkillGrasp,      '/skill/grasp'),
    'pick_place':  (SkillPickPlace,  '/skill/pick_place'),
    'press':       (SkillPress,      '/skill/press'),
    'slide_rack':  (SkillSlideRack,  '/skill/slide_rack'),
    'turn_lever':  (SkillTurnLever,  '/skill/turn_lever'),
    'twist_knob':  (SkillTwistKnob,  '/skill/twist_knob'),
}


def _centroid_from_cloud(cloud):
    """Mean XYZ of a PointCloud2 (ported verbatim from open_fridge.py)."""
    n = cloud.width * cloud.height
    if n == 0:
        return None
    data = bytes(cloud.data)
    step = cloud.point_step
    xs, ys, zs = [], [], []
    for i in range(n):
        off = i * step
        x = struct.unpack_from('f', data, off + 0)[0]
        y = struct.unpack_from('f', data, off + 4)[0]
        z = struct.unpack_from('f', data, off + 8)[0]
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
            xs.append(x); ys.append(y); zs.append(z)
    if not xs:
        return None
    return (float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs)))


def _cloud_to_xyz(cloud):
    """All finite XYZ points of a PointCloud2 as an (N, 3) array (float32
    x/y/z at offsets 0/4/8, the same layout _centroid_from_cloud assumes)."""
    n = cloud.width * cloud.height
    step = cloud.point_step
    if n == 0 or step < 12:
        return None
    buf = np.frombuffer(bytes(cloud.data), dtype=np.uint8)
    if buf.size < n * step:
        return None
    xyz = buf[:n * step].reshape(n, step)[:, :12].copy().view(np.float32)
    pts = xyz.reshape(n, 3).astype(np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts if len(pts) else None


def _transform_to_rt(tfm):
    """Rotation matrix + translation vector of a TransformStamped."""
    q = tfm.transform.rotation
    t = tfm.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])
    return rot, np.array([t.x, t.y, t.z])


def _antipodal_in_yz(points):
    """Antipodal grasp plan in the plane perpendicular to the +x approach axis.

    PCA of the cloud's y-z footprint: closing the fingers along the minor
    (smallest-extent) principal axis puts the two contacts on opposing
    surfaces with roughly opposing normals — the antipodal condition for a
    parallel-jaw gripper. Returns (wrist roll about +x [rad, folded into
    [-pi/2, pi/2]], grasp width [mm]); falls back to (0.0, GRIPPER_OPEN_MM)
    for sparse or degenerate clouds.
    """
    if points is None or len(points) < MIN_GRASP_POINTS:
        return 0.0, GRIPPER_OPEN_MM
    yz = points[:, 1:3] - points[:, 1:3].mean(axis=0)
    cov = np.cov(yz.T)
    if not np.all(np.isfinite(cov)):
        return 0.0, GRIPPER_OPEN_MM
    evals, evecs = np.linalg.eigh(cov)           # eigenvalues ascending
    if evals[1] <= 1e-12:                        # degenerate footprint
        return 0.0, GRIPPER_OPEN_MM
    minor = evecs[:, 0]                          # (y, z) closing direction
    roll = math.atan2(minor[1], minor[0])        # angle from +y toward +z
    # The closing axis is a line, not a direction: fold into [-pi/2, pi/2]
    # so the wrist takes the smaller of the two equivalent rolls.
    if roll > math.pi / 2:
        roll -= math.pi
    elif roll < -math.pi / 2:
        roll += math.pi
    proj = yz @ minor
    width_mm = float(np.percentile(proj, 97.5) - np.percentile(proj, 2.5)) * 1000.0
    return roll, width_mm


def _roll_quat(angle):
    """Quaternion (x, y, z, w) for a roll of `angle` about the +x (approach) axis."""
    half = 0.5 * float(angle)
    return (math.sin(half), 0.0, 0.0, math.cos(half))


class _Run:
    """Per-goal execution context: feedback/result plumbing + cancel/deadline checks.

    Usage in an execute callback:
        run = _Run(self, goal_handle, SkillOpenDoor, 'open_door')
        if not run.phase('detect', 0.0):
            return run.result            # canceled or timed out
        ...
        return run.succeed('door opened')   # or run.abort('reason')
    """

    def __init__(self, node, goal_handle, action_type, label):
        self._node = node
        self._gh = goal_handle
        self._label = label
        self.feedback = action_type.Feedback()
        self.result = action_type.Result()
        self.deadline = time.monotonic() + node._timeout_sec(goal_handle.request)

    def remaining(self):
        return max(0.0, self.deadline - time.monotonic())

    def _finish_canceled(self):
        self.result.success = False
        self.result.message = 'canceled'
        self._gh.canceled()
        self._node.get_logger().info(f'[{self._label}] canceled')
        return self.result

    def phase(self, name, progress):
        """Publish a feedback phase. Returns False (with the goal terminated and
        self.result filled in) if the goal was canceled or the deadline passed."""
        if self._gh.is_cancel_requested:
            self._finish_canceled()
            return False
        if time.monotonic() > self.deadline:
            self.result.success = False
            self.result.message = 'skill timeout'
            self._gh.abort()
            self._node.get_logger().error(f'[{self._label}] timed out')
            return False
        self.feedback.phase = name
        self.feedback.progress = float(progress)
        self._gh.publish_feedback(self.feedback)
        self._node.get_logger().info(f'[{self._label}] phase={name}')
        return True

    def abort(self, message):
        # A step that "failed" because the goal was canceled mid-motion should
        # finish as canceled, not as a misleading failure.
        if self._gh.is_cancel_requested:
            return self._finish_canceled()
        self.result.success = False
        self.result.message = message
        self._node.get_logger().error(f'[{self._label}] {message}')
        self._gh.abort()
        return self.result

    def succeed(self, message='ok'):
        # Don't report success for a goal whose cancel was already accepted.
        if self._gh.is_cancel_requested:
            return self._finish_canceled()
        self.feedback.phase = 'done'
        self.feedback.progress = 1.0
        self._gh.publish_feedback(self.feedback)
        self.result.success = True
        self.result.message = message
        self._node.get_logger().info(f'[{self._label}] done: {message}')
        self._gh.succeed()
        return self.result


class SkillsBase(Node):
    """Node holding the clients, perception layer, and motion primitives shared
    by every skill mixin. SkillsNode (node.py) adds the skill action servers."""

    def __init__(self, node_name='h12_skills'):
        super().__init__(node_name)

        # Everything shares one reentrant group so skill execute callbacks can
        # block on inner service/action futures while the executor keeps
        # spinning (and so pick_place can call the grasp server in-process).
        self._cb_group = ReentrantCallbackGroup()

        # --- service clients (vision pipeline + grippers) ---------------------
        self.track_cli = self.create_client(
            UpdateTrackedObject, '/vp_update_tracked_object', callback_group=self._cb_group)
        self.beliefs_cli = self.create_client(
            UpdateBeliefs, '/vp_update_beliefs', callback_group=self._cb_group)
        self.query_cli = self.create_client(
            Query, '/vp_query_tracked_objects', callback_group=self._cb_group)
        self.gripper_clis = {
            arm: self.create_client(SetGripperPosition, srv, callback_group=self._cb_group)
            for arm, srv in GRIPPER_SRVS.items()
        }

        # --- action clients (arm IK + nav2) -----------------------------------
        self.frame_task_cli = ActionClient(
            self, FrameTask, '/frame_task', callback_group=self._cb_group)
        self.nav_cli = ActionClient(
            self, NavigateToPose, '/navigate_to_pose', callback_group=self._cb_group)

        # --- subscriber (TF) ---------------------------------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- color image subscription ------------------------------------------
        # Latest head-camera color frame, kept for Gemini Robotics reasoning.
        self._cv_bridge = CvBridge()
        self._latest_color = None        # most recent RGB frame (np.ndarray) or None
        self._latest_color_jpeg = None   # raw JPEG bytes (for Gemini) or None
        self._latest_color_stamp = None  # builtin_interfaces/Time of that frame
        self.create_subscription(
            CompressedImage, COLOR_IMAGE_TOPIC, self._on_color_image,
            qos_profile_sensor_data, callback_group=self._cb_group)

        # --- Gemini Robotics client --------------------------------------------
        self.gemini_client = self._init_gemini()

        # --- skill action clients ----------------------------------------------
        # One client per skill; used for composition (pick_place -> grasp) and
        # as the reference for external callers.
        self.skill_clients = {
            name: ActionClient(self, action_type, server, callback_group=self._cb_group)
            for name, (action_type, server) in SKILL_ACTIONS.items()
        }

        # Wait for the underlying endpoints (non-fatal, mirrors open_fridge.py).
        self.get_logger().info('Waiting for VP services, grippers, and action servers...')
        self.track_cli.wait_for_service(timeout_sec=10.0)
        self.beliefs_cli.wait_for_service(timeout_sec=10.0)
        self.query_cli.wait_for_service(timeout_sec=10.0)
        for cli in self.gripper_clis.values():
            cli.wait_for_service(timeout_sec=10.0)
        self.frame_task_cli.wait_for_server(timeout_sec=10.0)
        self.nav_cli.wait_for_server(timeout_sec=10.0)

    # ----------------------------------------------------- gemini + perception
    def _init_gemini(self):
        """Create the Gemini Robotics client. Non-fatal: returns None (and the
        node serves skills without Gemini) if the SDK or API key is missing.
        The key is read from $GEMINI_API_KEY / $GOOGLE_API_KEY (the google-genai
        SDK also picks these up itself when no key is passed)."""
        if genai is None:
            self.get_logger().warn(
                'google-genai not installed — Gemini Robotics client disabled')
            return None
        api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        try:
            client = genai.Client(api_key=api_key) if api_key else genai.Client()
        except Exception as e:  # missing key, bad creds, etc.
            self.get_logger().warn(
                f'Gemini Robotics client init failed ({e}) — disabled')
            return None
        self.get_logger().info(f'Gemini Robotics client ready (model {GEMINI_MODEL})')
        return client

    def _on_color_image(self, msg: CompressedImage):
        """Cache the latest head-camera color frame (decoded RGB + raw JPEG)."""
        self._latest_color_jpeg = bytes(msg.data)
        self._latest_color_stamp = msg.header.stamp
        try:
            self._latest_color = self._cv_bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().warn(
                f'color image decode failed: {e}', throttle_duration_sec=5.0)

    def latest_color_image(self):
        """Most recent head-camera frame as RGB np.ndarray (H, W, 3), or None."""
        return self._latest_color

    def latest_color_jpeg(self):
        """Most recent head-camera frame as JPEG bytes (ready for Gemini), or None."""
        return self._latest_color_jpeg

    # ------------------------------------------------------------------ utils
    def _on_skill_cancel(self, _cancel_request):
        return CancelResponse.ACCEPT

    def _timeout_sec(self, goal, default=DEFAULT_SKILL_TIMEOUT):
        t = float(goal.timeout.sec) + float(goal.timeout.nanosec) * 1e-9
        return t if t > 0.0 else default

    def _wait_future(self, future, timeout_sec, outer_gh=None):
        """Block this (executor worker) thread until the future resolves.
        Safe under the MultiThreadedExecutor: other threads keep spinning.
        Polls in small increments so a cancel of the outer skill goal
        (outer_gh) is noticed promptly. Returns None on timeout or cancel."""
        event = threading.Event()
        future.add_done_callback(lambda _f: event.set())
        deadline = time.monotonic() + max(0.0, timeout_sec)
        while not event.is_set():
            if outer_gh is not None and outer_gh.is_cancel_requested:
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None
            event.wait(timeout=min(0.1, remaining))
        return future.result()

    def _call_service(self, client, request, name, timeout_sec=30.0):
        if not client.service_is_ready():
            self.get_logger().error(f'{name} service not available')
            return None
        result = self._wait_future(client.call_async(request), timeout_sec)
        if result is None:
            self.get_logger().error(f'{name} call failed or timed out')
        return result

    def _send_action(self, client, goal, feedback_cb=None,
                     accept_timeout=10.0, result_timeout=120.0, outer_gh=None):
        """send_goal -> wait for acceptance -> wait for result.
        Returns the result response (with .status and .result), or None. If the
        result wait fails (timeout or outer-skill cancel), the in-flight goal
        is canceled ON THE SERVER — Future.cancel() would only drop the local
        future and leave the robot executing an orphaned goal."""
        if result_timeout <= 0.05:
            self.get_logger().error('no time left to send action goal')
            return None
        handle = self._wait_future(
            client.send_goal_async(goal, feedback_callback=feedback_cb), accept_timeout)
        if handle is None or not handle.accepted:
            self.get_logger().error('action goal rejected or send timed out')
            return None
        response = self._wait_future(handle.get_result_async(), result_timeout, outer_gh)
        if response is None:
            self.get_logger().warn('canceling in-flight inner action goal')
            self._wait_future(handle.cancel_goal_async(), 2.0)
        return response

    # --------------------------------------------------- vision pipeline layer
    def track_object(self, query, action='add'):
        req = UpdateTrackedObject.Request()
        req.object = query
        req.action = action
        return self._call_service(self.track_cli, req, 'UpdateTrackedObject')

    def update_beliefs(self, camera=CAMERA_NS):
        req = UpdateBeliefs.Request()
        req.camera_name_space = camera
        return self._call_service(self.beliefs_cli, req, 'UpdateBeliefs')

    def query_centroid(self, query, target_frame='pelvis'):
        """Centroid of the best detection of `query`, in `target_frame`.
        Among multiple detections, prefers the one nearest the robot
        (open_fridge.py picked the cloud closest to a point near the pelvis)."""
        req = Query.Request()
        req.query = query
        req.confidence_threshold = 0.3
        req.pc_name = ''
        result = self._call_service(self.query_cli, req, 'Query')
        if not result or not result.success or not result.clouds:
            return None
        ref = self._frame_origin_in(target_frame, 'pelvis') or (0.0, 0.0, 0.1)
        best, best_dist = None, None
        for cloud in result.clouds:
            centroid = _centroid_from_cloud(cloud)
            if centroid is None:
                continue
            transformed = self._transform_point(
                centroid, cloud.header.frame_id, cloud.header.stamp, target_frame)
            if transformed is None:
                continue
            dist = sum((a - b) ** 2 for a, b in zip(transformed, ref))
            if best_dist is None or dist < best_dist:
                best_dist, best = dist, transformed
        return best

    def detect_object(self, query, target_frame='pelvis'):
        """Full open_fridge detection recipe: track -> update beliefs from the
        head camera -> settle -> query the centroid."""
        if self.track_object(query) is None:
            return None
        if self.update_beliefs(CAMERA_NS) is None:
            return None
        self.get_clock().sleep_for(RclpyDuration(seconds=DETECT_SETTLE_SEC))
        centroid = self.query_centroid(query, target_frame)
        if centroid is not None:
            cx, cy, cz = centroid
            self.get_logger().info(
                f'{query!r} centroid ({target_frame}): ({cx:.3f}, {cy:.3f}, {cz:.3f})')
        return centroid

    def query_cloud_points(self, query, target_frame='pelvis'):
        """Points of the best detection of `query` as an (N, 3) array in
        `target_frame` (best = centroid nearest the robot, like query_centroid)."""
        req = Query.Request()
        req.query = query
        req.confidence_threshold = 0.3
        req.pc_name = ''
        result = self._call_service(self.query_cli, req, 'Query')
        if not result or not result.success or not result.clouds:
            return None
        ref = np.asarray(
            self._frame_origin_in(target_frame, 'pelvis') or (0.0, 0.0, 0.1))
        best_pts, best_dist = None, None
        for cloud in result.clouds:
            pts = _cloud_to_xyz(cloud)
            if pts is None:
                continue
            try:
                tfm = self.tf_buffer.lookup_transform(
                    target_frame, cloud.header.frame_id, cloud.header.stamp,
                    timeout=RclpyDuration(seconds=1.0))
            except TransformException as e:
                self.get_logger().error(
                    f'TF lookup {cloud.header.frame_id!r} -> {target_frame!r} '
                    f'failed: {e}')
                continue
            rot, trans = _transform_to_rt(tfm)
            pts = pts @ rot.T + trans
            dist = float(np.sum((pts.mean(axis=0) - ref) ** 2))
            if best_dist is None or dist < best_dist:
                best_dist, best_pts = dist, pts
        return best_pts

    def detect_grasp(self, query, target_frame='pelvis'):
        """detect_object's recipe, but planning an antipodal grasp from the
        full detection cloud. Returns ((x, y, z) centroid, wrist roll [rad],
        grasp width [mm]) or None."""
        if self.track_object(query) is None:
            return None
        if self.update_beliefs(CAMERA_NS) is None:
            return None
        self.get_clock().sleep_for(RclpyDuration(seconds=DETECT_SETTLE_SEC))
        pts = self.query_cloud_points(query, target_frame)
        if pts is None:
            return None
        centroid = tuple(float(v) for v in pts.mean(axis=0))
        roll, width_mm = _antipodal_in_yz(pts)
        self.get_logger().info(
            f'{query!r} grasp plan ({target_frame}): centroid '
            f'({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}), '
            f'roll {math.degrees(roll):.0f} deg, width {width_mm:.0f} mm '
            f'({len(pts)} pts)')
        return centroid, roll, width_mm

    def _transform_point(self, xyz, source_frame, stamp, target_frame):
        pt = PointStamped()
        pt.header.frame_id = source_frame
        pt.header.stamp = stamp
        pt.point.x, pt.point.y, pt.point.z = xyz
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, stamp,
                timeout=RclpyDuration(seconds=1.0))
        except TransformException as e:
            self.get_logger().error(
                f'TF lookup {source_frame!r} -> {target_frame!r} failed: {e}')
            return None
        transformed = do_transform_point(pt, transform)
        return (float(transformed.point.x), float(transformed.point.y),
                float(transformed.point.z))

    def _frame_origin_in(self, target_frame, source_frame):
        """Origin of `source_frame` expressed in `target_frame`, or None."""
        try:
            tfm = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time(),
                timeout=RclpyDuration(seconds=1.0))
        except TransformException:
            return None
        t = tfm.transform.translation
        return (float(t.x), float(t.y), float(t.z))

    # ------------------------------------------------------- motion primitives
    def move_frame_to(self, arm, x, y, z, duration_sec=3, quat=(0.0, 0.0, 0.0, 1.0),
                      outer_gh=None):
        """Send the arm's grip-site frame to (x, y, z) in the pelvis frame via
        /frame_task (open_fridge.py's move_frame_to, retargeted from the wrist
        to the gripper closure point, orientation added).
        Pass the skill's goal handle as outer_gh so a skill cancel promptly
        cancels the in-flight frame_task goal too."""
        goal = FrameTask.Goal()
        goal.frame_names = [ARM_FRAMES[arm]]
        pose = Pose()
        pose.position = Point(x=float(x), y=float(y), z=float(z))
        pose.orientation = Quaternion(
            x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))
        goal.frame_targets = [pose]
        goal.duration = Duration(sec=int(duration_sec), nanosec=0)

        self.get_logger().info(
            f'frame_task: {ARM_FRAMES[arm]} -> ({x:.3f}, {y:.3f}, {z:.3f}) '
            f'in {duration_sec}s')
        response = self._send_action(
            self.frame_task_cli, goal, result_timeout=float(duration_sec) + 10.0,
            outer_gh=outer_gh)
        return response is not None and response.status == GoalStatus.STATUS_SUCCEEDED

    def set_gripper(self, arm, position_mm, speed=1.0):
        req = SetGripperPosition.Request()
        req.position = float(position_mm)
        req.speed = float(speed)
        result = self._call_service(
            self.gripper_clis[arm], req, f'SetGripperPosition({arm})')
        if result is None:
            return False
        self.get_logger().info(
            f'gripper {arm}: success={result.success} '
            f'actual={result.actual_position:.2f} mm — {result.message}')
        return result.success

    def open_gripper(self, arm):
        return self.set_gripper(arm, GRIPPER_OPEN_MM)

    def close_gripper(self, arm):
        return self.set_gripper(arm, GRIPPER_CLOSED_MM)

    def navigate_to(self, x, y, yaw=0.0, frame='map', timeout_sec=120.0, outer_gh=None):
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position = Point(x=float(x), y=float(y), z=0.0)
        half = 0.5 * float(yaw)
        goal.pose.pose.orientation = Quaternion(
            x=0.0, y=0.0, z=float(math.sin(half)), w=float(math.cos(half)))

        self.get_logger().info(
            f'navigate_to_pose -> ({x:.3f}, {y:.3f}, yaw={yaw:.3f}) in {frame!r}')
        response = self._send_action(self.nav_cli, goal, result_timeout=timeout_sec,
                                     outer_gh=outer_gh)
        return response is not None and response.status == GoalStatus.STATUS_SUCCEEDED

    def _validated_arm(self, run, goal):
        arm = (goal.arm or 'right').strip().lower()
        if arm not in ARM_FRAMES:
            return None
        return arm

    # ---------------------------------------------------------- skill clients
    # Client-side callbacks for the skill actions (used when composing skills
    # or invoking them from this node; bind the skill name with partial()).
    def skill_feedback_cb(self, skill, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(
            f'[{skill}] feedback: phase={fb.phase} progress={fb.progress:.2f}')

    def skill_goal_response_cb(self, skill, future):
        handle = future.result()
        accepted = handle is not None and handle.accepted
        self.get_logger().info(f'[{skill}] goal {"accepted" if accepted else "REJECTED"}')

    def skill_result_cb(self, skill, future):
        response = future.result()
        if response is None:
            self.get_logger().error(f'[{skill}] result unavailable')
            return
        r = response.result
        self.get_logger().info(f'[{skill}] result: success={r.success} ({r.message})')

    def _call_grasp_skill(self, gh, run, target_object, arm):
        """Compose: run the SkillGrasp action (detect + antipodal grasp) through
        this node's own client, budgeted to the caller's remaining time.
        Returns (ok, message)."""
        rem = run.remaining()
        grasp_goal = SkillGrasp.Goal()
        grasp_goal.target_object = target_object
        grasp_goal.arm = arm
        grasp_goal.timeout = Duration(sec=int(rem), nanosec=int((rem % 1.0) * 1e9))
        response = self._send_action(
            self.skill_clients['grasp'], grasp_goal,
            feedback_cb=partial(self.skill_feedback_cb, 'grasp'),
            result_timeout=rem, outer_gh=gh)
        if response is None:
            return False, ('no response' if run.remaining() > 0 else 'skill timeout')
        if response.status != GoalStatus.STATUS_SUCCEEDED or not response.result.success:
            return False, response.result.message or 'grasp action did not succeed'
        return True, response.result.message
