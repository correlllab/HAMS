#!/usr/bin/env python3
"""Shared infrastructure for the h12 skills node.

`SkillsBase` is the ROS node every skill mixin plugs into. It owns the external
clients the skills depend on — the arm IK action (frame_task), the per-arm
gripper services, the Gemini query service, the SAM segmentation service, and the
GraspGen planning service — plus the head-camera color/depth/intrinsics caches
(which feed the perception services and the mask→cloud back-projection), the TF
listener, the executor-safe service/action call plumbing (_call_service /
_send_action / _wait_future), and the `_Run` per-goal execution context.

Perception pipeline helpers (`query_gemini`, `segment`, `mask_to_cloud`,
`detect_object_cloud`, `plan_grasp`) and the motion primitives (`move_frame_to`,
gripper ops) are implemented here; each skill under skills/ composes them in its
`_exec_*` mixin.
SkillsNode (node.py) multiply-inherits from SkillsBase plus every skill mixin.
"""

import threading
import time

import numpy as np

from rclpy.action import ActionClient, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration as RclpyDuration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3Stamped
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Header

from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformBroadcaster, TransformListener, TransformException

from custom_ros_messages.msg import DetectionBundle
from custom_ros_messages.srv import GeminiQuery, SamSegment, GraspGen
from custom_ros_messages.action import FrameTask, NamedConfig
from custom_ros_messages.action import (
    SkillCloseDoor, SkillOpenDoor, SkillCloseLid, SkillOpenLid,
    SkillNavigate, SkillGrasp, SkillPickPlace, SkillPress, SkillSlideRack,
    SkillTurnLever, SkillTwistKnob, SkillFrontierExplore,
)
from magpie_msgs.srv import SetGripperPosition, SetGripperForce
from magpie_msgs.msg import GripperState
from std_srvs.srv import Trigger

from .perception_utils import (
    decode_compressed_depth_image, deproject_mask, transform_points,
    transform_to_matrix, pose_to_matrix, matrix_to_pose, quat_geodesic,
    extract_json,
)


CAMERA_NS = '/realsense/head'
# Color (JPEG), aligned depth (compressedDepth, uint16 mm), and intrinsics.
COLOR_IMAGE_TOPIC = f'{CAMERA_NS}/color/image_raw/compressed'
DEPTH_IMAGE_TOPIC = f'{CAMERA_NS}/aligned_depth_to_color/image_raw/compressedDepth'
CAMERA_INFO_TOPIC = f'{CAMERA_NS}/color/camera_info'
# yolo_server's per-camera DetectionBundle output for the head color frame (it
# derives '<image_topic>/detections'). Bbox pixels are in this color frame's grid,
# so they line up directly with the aligned depth cache for back-projection. Feeds
# the battery-object top-down grasp (skills/grasp.py) which skips gemini/sam/graspgen.
DETECTION_TOPIC = f'{COLOR_IMAGE_TOPIC}/detections'
# Per-arm HAND-camera DetectionBundle topics (same yolo_server, one channel per
# camera; like the head, the hand cameras publish color on '.../image_raw' since
# cl_realsense's h12_hand_cameras.launch.py split). The battery grasp reads the
# relevant arm's hand detections once the GraspGenX frame is parked above the
# object.
ARM_DETECTION_TOPICS = {
    'left':  '/realsense/left_hand/color/image_raw/compressed/detections',
    'right': '/realsense/right_hand/color/image_raw/compressed/detections',
}
# The raw per-arm hand-camera color streams those bundles are derived from —
# cached directly so a hand frame is available even when yolo_server's hand
# channels are down (feeds the grasp skill's approach snapshot).
ARM_COLOR_IMAGE_TOPICS = {
    'left':  '/realsense/left_hand/color/image_raw/compressed',
    'right': '/realsense/right_hand/color/image_raw/compressed',
}

# GraspGenX gripper-BASE frames (URDF frames whose axes match the GraspGenX
# planning convention: +Z approach, +X finger-close, origin at the magpie gripper
# base). A raw GraspGenX grasp pose is sent straight to frame_task targeting these
# — no axis-permutation / TCP-depth correction needed. See skills/grasp.py.
GRASP_FRAMES = {'left': 'left_graspgenx_frame', 'right': 'right_graspgenx_frame'}
# Per-arm gripper service namespace; the magpie driver/sim expose
# <ns>/set_position, <ns>/set_force, <ns>/open and <ns>/close under each.
GRIPPER_NS = {'left': '/left/gripper', 'right': '/right/gripper'}

# Grip-force limit (N) applied via set_force before closing on an object.
# Halved from 30 N: 30 N on a small part is more than the grasp needs and is what
# drives the Dynamixels into the overload the grip-hold loop then has to keep
# clearing. This is the default for both set_gripper_force and close_gripper, so
# every close in the package moves together.
GRIP_FORCE_N = 15.0
# --- grip-hold loop (see close_gripper / _grip_hold_loop) --------------------
# The magpie fingers are Dynamixels that trip into OVERLOAD while holding
# against an object and then go limp, dropping it. reset_overload clears that
# without opening the fingers, so a background thread per arm re-asserts the
# grip at this rate: reset_overload, then close, every tick.
#
# LIFETIME: the loop deliberately outlives the skill that started it — the
# object is still held after /skill/grasp returns — and stops only when the
# gripper is commanded to an aperture again (open_gripper / set_gripper). It is
# a daemon thread, so it also dies with the node.
GRIP_HOLD_RATE_HZ = 5.0
# Per-call timeout inside the loop [s]. Well under the 1/rate period so a
# stalled gripper service can't drag the loop off cadence; a tick that times out
# is simply retried on the next one.
GRIP_HOLD_CALL_TIMEOUT_SEC = 0.15
# The loop calls two services at GRIP_HOLD_RATE_HZ, so a persistently dead
# gripper service would emit ~10 log lines/s. Warn only once per this many
# consecutive failed ticks instead (failures are logged, never fatal — a flaky
# gripper service must not turn a good grasp into a failed goal).
GRIP_HOLD_WARN_EVERY = 25
# ROS-time wait (get_clock().sleep_for — respects sim time) after an open/close so
# the fingers finish moving before the arm starts its next motion. The /close
# Trigger returns in ~5 ms and the position wait under-estimates finger travel, so
# without this the arm moves while the gripper is still opening/closing. Raise if
# the fingers are slower than this.
GRIPPER_SETTLE_SEC = 1.0
# Full-open gripper aperture [mm]. The magpie driver clamps set_position to
# 0-106 mm (gripper.set_goal_aperture), so 106 mm is "fully open" and 0 mm is
# "fully closed" — the endpoints the open/close percentage helpers map to.
GRIPPER_MAX_WIDTH_MM = 106.0
# Used when goal.timeout is zero [s]. INFINITE: a goal that names no timeout is
# taken as "run to completion" — the skill then ends only by finishing, failing,
# or being canceled, never on a clock. Deliberate: the long-running skills
# (frontier exploration, a visual-servo grasp) have no meaningful upper bound,
# and the old 300 s ceiling silently aborted them mid-work. Send a NON-ZERO
# goal.timeout whenever a bound is actually wanted.
#
# Consequences of the infinity, all checked: _Run.deadline becomes inf so
# remaining() is inf and phase() never trips; callers that spend remaining() do
# so through min(finite, ...) and stay finite; the one caller that converts it to
# an integer Duration (pick_place's inner grasp goal) special-cases it. Cancel
# remains the operator's escape hatch and is polled throughout.
DEFAULT_SKILL_TIMEOUT = float('inf')
# Gemini query-service call timeout [s]. The gemini-robotics-er model used for
# grasp detection can take ~3 min to answer — far beyond _call_service's 30s
# default — so give GeminiQuery its own generous ceiling.
GEMINI_TIMEOUT_SEC = 240.0

# Ask Gemini for a box that fully ENCOMPASSES the whole object — this box is a
# prompt for a segmentation model (SAM), which needs a box around the entire
# target so its mask captures the full object cloud the downstream consumer
# (GraspGenX planning, place-target localization) works on. Same
# [y1,x1,y2,x2]-normalized-to-1000 convention the vision pipeline uses.
GEMINI_GRASP_PROMPT = (
    'Locate the {obj} so a segmentation model can be prompted with a bounding box. '
    'Return a JSON array with exactly one box that FULLY ENCOMPASSES the entire '
    '{obj} — every visible part of it (including any handle, spout, lid, or other '
    'protrusion) must fall inside the box, with the edges snug to the object\'s '
    'outermost extent. Do not box just one part and do not crop the object: '
    '[{{"box_2d": [y1, x1, y2, x2], "label": "{obj}", "score": 0.9}}], integer '
    'coordinates normalized to 0-1000 with y first. Always return a 4-number '
    '"box_2d" bounding box — never a single point. '
    'If the {obj} is not visible, return [].'
)
# Depth back-projection range [m] and the floor on usable object points.
# Keep MIN_GRASP_POINTS in sync with graspgen_server.MIN_OBJECT_POINTS so a cloud
# the server would reject is dropped client-side with an accurate message.
DEPTH_MIN_M = 0.1
DEPTH_MAX_M = 3.0
MIN_GRASP_POINTS = 30

# Debug/visualization TF frames (e.g. the grasp target) are re-broadcast at this
# period so they stay alive in RViz instead of expiring after a single send.
TF_REPUBLISH_PERIOD_SEC = 0.1

# World-frame grasp servo (servo_frame_to_world): the grasp is anchored in a
# world-fixed frame and re-resolved into the live pelvis frame each iteration so
# pelvis drift during the arm motion is cancelled. WORLD_FRAME is FAST-LIO's
# odometry origin (camera_init -> body -> pelvis); switch to 'map' for the
# slam_toolbox frame (globally consistent but jumps on loop closure).
WORLD_FRAME = 'camera_init'   # FAST-LIO odometry world frame; switch to 'map' for slam_toolbox
SERVO_ITER = 3                # max world-frame servo refinement iterations
SERVO_LIN_TOL = 0.005         # 5 mm world-position convergence tolerance
SERVO_ANG_TOL = 0.02          # ~1.15 deg world-orientation convergence tolerance
# Multiply a move's duration BUDGET by this when it is sent with slow_mode=True.
# The frame_task server's SLOW_MODE_SCALE (0.25) scales joint VELOCITY, while a
# FrameTask duration is a TIMEOUT — so at 1/4 speed the same budget covers 1/4
# the distance and the move would time out short of the target. Keep this at
# 1 / SLOW_MODE_SCALE so a slow move gets the same reach as a full-speed one.
SLOW_MODE_TIME_SCALE = 4.0

# Frame-task early-stop (see _send_frame_move's stable_stop_msgs): a feedback is
# counted as "unchanged" from the running reference when BOTH the linear and
# angular error stay within these bands. Sized above the arm's steady-state
# jitter (~0.1 mm / <1 mrad) so a settled hold is recognized, but well below any
# real motion so an actually-converging move never trips it.
FRAME_TASK_STABLE_EPS_LIN = 0.0005    # 0.5 mm
FRAME_TASK_STABLE_EPS_ANG = 0.002     # 2 mrad

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
    'frontier_explore': (SkillFrontierExplore, '/skill/frontier_explore'),
}


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
    """Node holding the clients, camera caches, call plumbing, perception
    pipeline, and motion primitives shared by every skill mixin. SkillsNode
    (node.py) adds the skill action servers."""

    def __init__(self, node_name='h12_skills'):
        super().__init__(node_name)

        # Everything shares one reentrant group so skill execute callbacks can
        # block on inner service/action futures while the executor keeps
        # spinning (and so pick_place can call the grasp server in-process).
        self._cb_group = ReentrantCallbackGroup()

        # --- gripper service clients (per arm) --------------------------------
        self.gripper_clis = {
            arm: self.create_client(SetGripperPosition, f'{ns}/set_position',
                                    callback_group=self._cb_group)
            for arm, ns in GRIPPER_NS.items()
        }
        self.gripper_open_clis = {
            arm: self.create_client(Trigger, f'{ns}/open', callback_group=self._cb_group)
            for arm, ns in GRIPPER_NS.items()
        }
        self.gripper_close_clis = {
            arm: self.create_client(Trigger, f'{ns}/close', callback_group=self._cb_group)
            for arm, ns in GRIPPER_NS.items()
        }
        self.gripper_force_clis = {
            arm: self.create_client(SetGripperForce, f'{ns}/set_force',
                                    callback_group=self._cb_group)
            for arm, ns in GRIPPER_NS.items()
        }
        # Clears a Dynamixel overload WITHOUT opening the fingers
        # (magpie_control gripper.py:reset_overload) — the grip-hold loop's
        # other half. See _grip_hold_loop.
        self.gripper_reset_clis = {
            arm: self.create_client(Trigger, f'{ns}/reset_overload',
                                    callback_group=self._cb_group)
            for arm, ns in GRIPPER_NS.items()
        }
        # arm -> (thread, stop Event) for the running grip-hold loops.
        self._grip_hold = {}
        # Live gripper aperture [mm] from the driver's published state (10 Hz),
        # cached per arm. Unlike _gripper_actual_mm (only refreshed by a position
        # command), this tracks the aperture after ANY close — including the
        # force-based /close close_gripper now uses — so the held-object check
        # stays valid. Read via gripper_aperture().
        self._gripper_state_mm = {}
        self.gripper_state_subs = {
            arm: self.create_subscription(
                GripperState, f'{ns}/state',
                lambda msg, a=arm: self._on_gripper_state(a, msg),
                10, callback_group=self._cb_group)
            for arm, ns in GRIPPER_NS.items()
        }
        # Aperture [mm] the driver reported after the last set_gripper per arm —
        # lets callers verify a close actually stopped ON an object (~0 mm =
        # fingers met each other, nothing held; see pick_place's held check).
        self._gripper_actual_mm = {}

        # --- perception service clients ---------------------------------------
        # gemini_query: image (+/- text) -> Gemini text; sam_segment: image +
        # box/text -> mono8 mask; graspgen: object cloud -> ranked 6-DOF grasps.
        self.gemini_cli = self.create_client(
            GeminiQuery, 'gemini_query', callback_group=self._cb_group)
        self.sam_cli = self.create_client(
            SamSegment, 'sam_segment', callback_group=self._cb_group)
        self.graspgen_cli = self.create_client(
            GraspGen, 'graspgen', callback_group=self._cb_group)

        # --- arm IK action client ---------------------------------------------
        self.frame_task_cli = ActionClient(
            self, FrameTask, '/frame_task', callback_group=self._cb_group)
        # Named arm configurations (t_pose/home/...), served by the same
        # frame_task server.
        self.named_config_cli = ActionClient(
            self, NamedConfig, '/named_config', callback_group=self._cb_group)

        # --- in-process skill composition ---------------------------------------
        # pick_place drives the grasp through this node's OWN /skill/grasp action
        # server (the ReentrantCallbackGroup + MultiThreadedExecutor make the
        # nested call safe). No wait_for_server at startup: the server comes up
        # with this same node. Because caller and server share the process, rich
        # RESULTS travel on the instance instead of the action result message —
        # see skills/grasp.py (GraspOutcome) for the contract.
        self.grasp_skill_cli = ActionClient(
            self, SkillGrasp, SKILL_ACTIONS['grasp'][1],
            callback_group=self._cb_group)
        self._last_grasp_outcome = None   # GraspOutcome of the last generic grasp

        # --- TF listener + broadcaster ----------------------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Broadcast planned grasp targets (graspgenx_target_frame) for RViz. These
        # are re-sent on a timer (publish_tf / _republish_tfs) so the frames stay
        # alive in RViz instead of expiring after a single one-shot send.
        self.tf_broadcaster = TransformBroadcaster(self)
        self._persistent_tfs = {}            # child_frame_id -> TransformStamped
        self._tf_lock = threading.Lock()
        self.create_timer(TF_REPUBLISH_PERIOD_SEC, self._republish_tfs,
                          callback_group=self._cb_group)
        # Visual-servo residual (dx/dy/dz in the hand-camera optical frame),
        # one message per servo iteration — for live plotting (PlotJuggler).
        self.servo_error_pub = self.create_publisher(
            Vector3Stamped, '/skill/grasp/servo_error', 10)
        # Visual-servo EXECUTED move (TF pose after the move minus before, in
        # the pelvis frame) — compare against servo_error to see how much of
        # each commanded correction the arm actually delivered.
        self.servo_move_pub = self.create_publisher(
            Vector3Stamped, '/skill/grasp/servo_move', 10)
        # Executed move minus the commanded correction, elementwise (== pose
        # after the move minus the commanded goal): the move's tracking
        # shortfall, in the pelvis frame. Zero = the arm did exactly as asked.
        self.servo_move_residual_pub = self.create_publisher(
            Vector3Stamped, '/skill/grasp/servo_move_residual', 10)
        # Annotated hand-camera frame for the visual servo: the measured object
        # (red dot), the setpoint with its X/Y tolerance (green ellipse), and the
        # commanded correction (arrow), projected into the image. One per servo
        # iteration. RAW sensor_msgs/Image, matching yolo_server's '/annotated'
        # convention, so an RViz Image display on this topic just works — a
        # compressed-only topic never appears in RViz's topic list at all,
        # because nothing publishes the base name it looks for. Depth 1 and only
        # while a servo runs (~0.4 Hz), so the bandwidth is bounded and brief.
        self.servo_image_pub = self.create_publisher(
            Image, '/skill/grasp/servo_image', 1)

        # --- head-camera caches (color for the services, depth+info for lifting
        #     a 2-D mask to a 3-D cloud) -----------------------------------------
        self._latest_image = None      # color CompressedImage (for gemini/sam)
        self._latest_depth = None      # aligned depth CompressedImage (mm)
        self._latest_caminfo = None    # color CameraInfo (intrinsics + frame)
        self._latest_detections = None  # head-camera YOLO DetectionBundle
        self._latest_arm_detections = {arm: None for arm in ARM_DETECTION_TOPICS}
        self._latest_arm_images = {arm: None for arm in ARM_COLOR_IMAGE_TOPICS}
        self.create_subscription(
            CompressedImage, COLOR_IMAGE_TOPIC, self._on_color_image,
            qos_profile_sensor_data, callback_group=self._cb_group)
        self.create_subscription(
            CompressedImage, DEPTH_IMAGE_TOPIC, self._on_depth_image,
            qos_profile_sensor_data, callback_group=self._cb_group)
        self.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC, self._on_caminfo,
            qos_profile_sensor_data, callback_group=self._cb_group)
        self.create_subscription(
            DetectionBundle, DETECTION_TOPIC, self._on_detections,
            qos_profile_sensor_data, callback_group=self._cb_group)
        for arm, topic in ARM_DETECTION_TOPICS.items():
            self.create_subscription(
                DetectionBundle, topic,
                lambda msg, a=arm: self._on_arm_detections(a, msg),
                qos_profile_sensor_data, callback_group=self._cb_group)
        for arm, topic in ARM_COLOR_IMAGE_TOPICS.items():
            self.create_subscription(
                CompressedImage, topic,
                lambda msg, a=arm: self._on_arm_image(a, msg),
                qos_profile_sensor_data, callback_group=self._cb_group)

        # Wait for the underlying endpoints (non-fatal).
        self.get_logger().info('Waiting for gemini/sam/graspgen, grippers, frame_task...')
        self.gemini_cli.wait_for_service(timeout_sec=10.0)
        self.sam_cli.wait_for_service(timeout_sec=10.0)
        self.graspgen_cli.wait_for_service(timeout_sec=10.0)
        for clis in (self.gripper_clis, self.gripper_open_clis,
                     self.gripper_close_clis, self.gripper_force_clis):
            for cli in clis.values():
                cli.wait_for_service(timeout_sec=10.0)
        self.frame_task_cli.wait_for_server(timeout_sec=10.0)
        self.named_config_cli.wait_for_server(timeout_sec=10.0)

    # ----------------------------------------------------------- camera caches
    def _on_color_image(self, msg: CompressedImage):
        """Cache the latest head-camera color frame (for gemini/sam requests)."""
        self._latest_image = msg

    def _on_depth_image(self, msg: CompressedImage):
        """Cache the latest aligned-depth frame (for mask→cloud back-projection)."""
        self._latest_depth = msg

    def _on_caminfo(self, msg: CameraInfo):
        """Cache the latest color CameraInfo (intrinsics + optical frame id)."""
        self._latest_caminfo = msg

    def _on_detections(self, msg: DetectionBundle):
        """Cache the latest head-camera YOLO DetectionBundle (battery grasp input)."""
        self._latest_detections = msg

    def _on_arm_detections(self, arm, msg: DetectionBundle):
        """Cache the latest hand-camera YOLO DetectionBundle for `arm`."""
        self._latest_arm_detections[arm] = msg

    def _on_arm_image(self, arm, msg: CompressedImage):
        """Cache the latest raw hand-camera color frame for `arm`."""
        self._latest_arm_images[arm] = msg

    def latest_image(self):
        """Most recent head-camera color frame (sensor_msgs/CompressedImage), or None."""
        return self._latest_image

    def latest_caminfo(self):
        """Most recent color CameraInfo, or None."""
        return self._latest_caminfo

    def latest_detections(self):
        """Most recent head-camera YOLO DetectionBundle, or None."""
        return self._latest_detections

    def latest_arm_detections(self, arm):
        """Most recent hand-camera YOLO DetectionBundle for `arm`, or None."""
        return self._latest_arm_detections.get(arm)

    def latest_arm_image(self, arm):
        """Most recent raw hand-camera color frame for `arm`
        (sensor_msgs/CompressedImage), or None."""
        return self._latest_arm_images.get(arm)

    # --------------------------------------------------------------- debug TF
    def publish_tf(self, transform):
        """Register a TransformStamped to be CONTINUOUSLY re-broadcast (re-stamped
        on a timer) so the frame stays alive in RViz instead of expiring after a
        single send. Keyed by child_frame_id: call again with the same child to
        move it, or drop_tf(child) to stop publishing it. Sends once immediately
        so the frame appears without waiting for the next tick."""
        with self._tf_lock:
            self._persistent_tfs[transform.child_frame_id] = transform
        self.tf_broadcaster.sendTransform(transform)

    def drop_tf(self, child_frame_id):
        """Stop continuously broadcasting a frame previously registered via publish_tf."""
        with self._tf_lock:
            self._persistent_tfs.pop(child_frame_id, None)

    def _republish_tfs(self):
        """Timer: re-stamp every registered debug frame to 'now' and re-broadcast,
        so RViz keeps showing them instead of dropping them as stale."""
        with self._tf_lock:
            tfs = list(self._persistent_tfs.values())
        if not tfs:
            return
        stamp = self.get_clock().now().to_msg()
        for t in tfs:
            t.header.stamp = stamp
        self.tf_broadcaster.sendTransform(tfs)

    # ------------------------------------------------------------------ utils
    def _on_skill_cancel(self, _cancel_request):
        return CancelResponse.ACCEPT

    def _timeout_sec(self, goal, default=DEFAULT_SKILL_TIMEOUT):
        t = float(goal.timeout.sec) + float(goal.timeout.nanosec) * 1e-9
        return t if t > 0.0 else default

    def _wait_future(self, future, timeout_sec, outer_gh=None, stop_event=None):
        """Block this (executor worker) thread until the future resolves.
        Safe under the MultiThreadedExecutor: other threads keep spinning.
        Polls in small increments so a cancel of the outer skill goal
        (outer_gh) is noticed promptly. Returns None on timeout or cancel, or
        when `stop_event` (an optional threading.Event) is set — the caller uses
        that to bail out early (e.g. frame_task feedback gone stable)."""
        event = threading.Event()
        future.add_done_callback(lambda _f: event.set())
        deadline = time.monotonic() + max(0.0, timeout_sec)
        while not event.is_set():
            if outer_gh is not None and outer_gh.is_cancel_requested:
                return None
            if stop_event is not None and stop_event.is_set():
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return None
            event.wait(timeout=min(0.1, remaining))
        return future.result()

    def _call_service(self, client, request, name, timeout_sec=30.0, outer_gh=None):
        """Call `client` and block (executor-safe) for the result. Pass the
        skill's goal handle as `outer_gh` so a cancel of that goal aborts the
        wait promptly instead of blocking on the in-flight call."""
        if not client.service_is_ready():
            self.get_logger().error(f'{name} service not available')
            return None
        result = self._wait_future(
            client.call_async(request), timeout_sec, outer_gh)
        if result is None:
            self.get_logger().error(f'{name} call failed or timed out')
        return result

    def _send_action(self, client, goal, feedback_cb=None,
                     accept_timeout=10.0, result_timeout=120.0, outer_gh=None,
                     stop_event=None):
        """send_goal -> wait for acceptance -> wait for result.
        Returns the result response (with .status and .result), or None. If the
        result wait fails (timeout, outer-skill cancel, or `stop_event` set),
        the in-flight goal is canceled ON THE SERVER — Future.cancel() would
        only drop the local future and leave the robot executing an orphaned
        goal. `stop_event` lets a feedback callback end the wait early (e.g.
        frame_task feedback gone stable)."""
        if result_timeout <= 0.05:
            self.get_logger().error('no time left to send action goal')
            return None
        handle = self._wait_future(
            client.send_goal_async(goal, feedback_callback=feedback_cb), accept_timeout)
        if handle is None or not handle.accepted:
            self.get_logger().error('action goal rejected or send timed out')
            return None
        response = self._wait_future(handle.get_result_async(), result_timeout,
                                     outer_gh, stop_event=stop_event)
        if response is None:
            self.get_logger().warn('canceling in-flight inner action goal')
            self._wait_future(handle.cancel_goal_async(), 2.0)
        return response

    # ------------------------------------------------- perception (gemini/sam)
    def query_gemini(self, prompt, image=None, timeout_sec=GEMINI_TIMEOUT_SEC,
                     outer_gh=None):
        """Ask the gemini_query service `prompt` about `image` (defaults to the
        latest head-camera frame). Returns Gemini's text response, or None.
        `timeout_sec` defaults high (GEMINI_TIMEOUT_SEC) because the grasp model
        is slow; pass a smaller value (e.g. the skill's remaining budget) for
        latency-sensitive callers. Pass the skill goal handle as `outer_gh` so a
        cancel aborts the (possibly minutes-long) call promptly."""
        img = image if image is not None else self.latest_image()
        if img is None:
            self.get_logger().error('query_gemini: no head-camera image yet')
            return None
        req = GeminiQuery.Request()
        req.image = img
        req.prompt = prompt
        resp = self._call_service(self.gemini_cli, req, 'GeminiQuery',
                                  timeout_sec=timeout_sec, outer_gh=outer_gh)
        if resp is None or not resp.success:
            return None
        return resp.response

    def segment(self, text='', positive_boxes=None, negative_boxes=None,
                image=None, outer_gh=None):
        """Run the sam_segment service on `image` (defaults to the latest
        head-camera frame) with a text prompt and/or flattened pixel-xyxy box
        exemplars. Returns the mono8 mask (sensor_msgs/Image), or None. Pass the
        skill goal handle as `outer_gh` so a cancel aborts the wait promptly."""
        img = image if image is not None else self.latest_image()
        if img is None:
            self.get_logger().error('segment: no head-camera image yet')
            return None
        req = SamSegment.Request()
        req.image = img
        req.text = text or ''
        req.positive_boxes = [float(v) for v in (positive_boxes or [])]
        req.negative_boxes = [float(v) for v in (negative_boxes or [])]
        resp = self._call_service(self.sam_cli, req, 'SamSegment',
                                  outer_gh=outer_gh)
        if resp is None or not resp.success:
            return None
        return resp.mask

    def _gemini_box(self, obj, run, gh):
        """Query Gemini for one bounding box of `obj`; return pixel [x1,y1,x2,y2]
        or None (SAM then uses the text prompt alone). The call is capped
        at the skill's remaining time budget (never longer than GEMINI_TIMEOUT_SEC)
        and `gh` is threaded through so a goal cancel aborts it promptly."""
        timeout = min(GEMINI_TIMEOUT_SEC, run.remaining())
        txt = self.query_gemini(GEMINI_GRASP_PROMPT.format(obj=obj),
                                timeout_sec=timeout, outer_gh=gh)
        data = extract_json(txt)
        entry = None
        if isinstance(data, list) and data:
            entry = data[0]
        elif isinstance(data, dict):
            entry = data
        if not isinstance(entry, dict) or 'box_2d' not in entry:
            return None
        info = self.latest_caminfo()
        if info is None or not info.width or not info.height:
            return None
        try:
            y1, x1, y2, x2 = (float(v) for v in entry['box_2d'])
        except (ValueError, TypeError):
            return None
        w, h = info.width, info.height
        px1, px2 = sorted((x1 / 1000.0 * w, x2 / 1000.0 * w))
        py1, py2 = sorted((y1 / 1000.0 * h, y2 / 1000.0 * h))
        return [px1, py1, px2, py2]

    def detect_object_cloud(self, obj, run, gh, box_provider=None, use_sam=True):
        """Locate `obj` in the head camera: bounding box -> object cloud, plus the
        whole-frame scene cloud as obstacle context. Returns (obj_cloud,
        scene_cloud, None) on success or (None, None, reason). Shared by the grasp
        skill (pick object) and pick_place (place target).

        `box_provider(obj, run, gh) -> pixel-xyxy box or None` supplies the box
        (default self._gemini_box). The grasp skill runs its own split
        detect_object/segment_object pipeline instead; pick_place uses this for
        the place target.

        `use_sam` selects how the box becomes an object cloud:
          * True  (default) — SAM segments the box region and only the masked pixels
            are back-projected. Best for larger/irregular objects.
          * False — the box RECTANGLE is back-projected directly (box_to_cloud), no
            SAM. Used for small components (screws/nuts/bolts) where SAM
            segmentation is unreliable at that scale; the box then MUST be present.

        With SAM: always prompts it with the object name; a whole-object box is
        passed too as a positive exemplar so text + box together pin down the right
        instance. The box is optional there (text alone is the fallback). Gemini
        latency is highly variable and None is a legitimate "no box", so
        cancel/timeout is disambiguated here before spending the SAM call."""
        box = (box_provider or self._gemini_box)(obj, run, gh)  # box (pixel xyxy) or None
        if gh.is_cancel_requested or run.remaining() <= 0.0:
            return None, None, 'detection canceled or timed out'
        if use_sam:
            mask = self.segment(text=obj, positive_boxes=box, outer_gh=gh)
            if mask is None:
                return None, None, f'no mask for {obj!r}'
            obj_cloud = self.mask_to_cloud(mask, target_frame='pelvis')
            reason = f'{obj!r} mask produced no usable cloud'
        else:
            if box is None:
                return None, None, f'no bounding box for {obj!r} (SAM disabled)'
            obj_cloud = self.box_to_cloud(box, target_frame='pelvis')
            reason = f'{obj!r} bounding box produced no usable cloud'
        if obj_cloud is None:
            return None, None, reason
        # Whole-frame cloud as obstacle context so graspgen can collision-filter
        # grasps against the surroundings (optional — None just skips filtering).
        scene = self.scene_to_cloud(target_frame='pelvis')
        return obj_cloud, scene, None

    def _depth_to_cloud(self, mask, target_frame):
        """Back-project `mask` (bool HxW over the color grid, or None for every
        valid pixel) with the latest aligned depth/intrinsics into (N, 3) points
        in `target_frame`, or None. Shared by mask_to_cloud (object) and
        scene_to_cloud (whole frame)."""
        depth_msg, info = self._latest_depth, self._latest_caminfo
        if depth_msg is None or info is None:
            self.get_logger().error('_depth_to_cloud: missing depth or caminfo')
            return None
        try:
            depth = decode_compressed_depth_image(depth_msg).astype(np.float32) / 1000.0
        except (ValueError, TypeError) as e:
            self.get_logger().error(f'_depth_to_cloud: depth decode failed: {e}')
            return None
        if mask is None:
            mask = np.ones(depth.shape, dtype=bool)
        elif depth.shape != mask.shape:
            self.get_logger().error(
                f'_depth_to_cloud: mask {mask.shape} != depth {depth.shape}')
            return None
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        pts_cam = deproject_mask(mask, depth, fx, fy, cx, cy, DEPTH_MIN_M, DEPTH_MAX_M)
        cam_frame = depth_msg.header.frame_id or info.header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame, cam_frame, Time(), timeout=RclpyDuration(seconds=1.0))
        except TransformException as e:
            self.get_logger().error(
                f'_depth_to_cloud: TF {cam_frame!r} -> {target_frame!r} failed: {e}')
            return None
        return transform_points(pts_cam, transform_to_matrix(tf.transform)).astype(np.float32)

    def mask_to_cloud(self, mask_msg, target_frame='pelvis'):
        """Back-project a mono8 mask + the latest aligned depth/intrinsics into an
        (N, 3) object point cloud in `target_frame`, or None. Depth and mask must
        share the color pixel grid (the realsense publishes aligned depth)."""
        if mask_msg is None:
            self.get_logger().error('mask_to_cloud: missing mask')
            return None
        mask = (np.frombuffer(bytes(mask_msg.data), dtype=np.uint8)
                .reshape(mask_msg.height, mask_msg.width) > 127)
        pts = self._depth_to_cloud(mask, target_frame)
        if pts is None:
            return None
        if len(pts) < MIN_GRASP_POINTS:
            self.get_logger().warn(f'mask_to_cloud: only {len(pts)} valid points')
            return None
        return pts

    def box_to_cloud(self, box, target_frame='pelvis'):
        """Back-project the pixel-xyxy `box` RECTANGLE (every valid pixel inside it,
        no SAM mask) with the latest aligned depth/intrinsics into an (N, 3) object
        cloud in `target_frame`, or None. For small components (screws/nuts/bolts)
        where SAM segmentation is unreliable — the detector box + depth IS the
        object cloud. The box is clamped to the image; note it also picks up
        whatever surface shows between/around the part inside the box."""
        info = self.latest_caminfo()
        if box is None or info is None or not info.width or not info.height:
            self.get_logger().error('box_to_cloud: missing box or caminfo')
            return None
        w, h = int(info.width), int(info.height)
        x1, y1, x2, y2 = box
        u0, u1 = max(0, int(round(min(x1, x2)))), min(w, int(round(max(x1, x2))))
        v0, v1 = max(0, int(round(min(y1, y2)))), min(h, int(round(max(y1, y2))))
        if u1 <= u0 or v1 <= v0:
            self.get_logger().warn(f'box_to_cloud: degenerate box {box}')
            return None
        mask = np.zeros((h, w), dtype=bool)
        mask[v0:v1, u0:u1] = True
        pts = self._depth_to_cloud(mask, target_frame)
        if pts is None:
            return None
        if len(pts) < MIN_GRASP_POINTS:
            self.get_logger().warn(f'box_to_cloud: only {len(pts)} valid points')
            return None
        return pts

    def scene_to_cloud(self, target_frame='pelvis'):
        """Back-project the whole latest depth frame (every valid pixel, no mask)
        into an (N, 3) scene cloud in `target_frame` — obstacle context for grasp
        collision filtering. Returns None if depth/caminfo/TF are unavailable."""
        return self._depth_to_cloud(None, target_frame)

    @staticmethod
    def bundle_camera_frame(bundle):
        """Optical frame id the bundle's depth (and so bundle_detection_point's
        output) is expressed in, or None. The depth frame is authoritative;
        camera_info's is the fallback."""
        if bundle is None:
            return None
        return (bundle.depth_image.header.frame_id
                or bundle.camera_info.header.frame_id or None)

    def bundle_detection_point(self, bundle, det, min_depth=DEPTH_MIN_M):
        """Back-project detection `det` of DetectionBundle `bundle` to a single
        (x, y, z) point IN THE BUNDLE CAMERA'S OWN OPTICAL FRAME, or None.

        Uses only what the bundle carries — yolo_server attaches each camera's
        aligned depth + color intrinsics — so this works for ANY camera the
        detector runs on without the skills node subscribing to it. That is what
        makes the hand-camera visual servo possible (the node caches head depth
        only). Alignment is the precondition: `depth_image` must be depth
        resampled onto the color grid the boxes were measured in.

        Deprojects the CENTRAL QUARTER of the box and takes the median, matching
        _detection_point — box edges pick up background depth, and on a small
        part that background is the support surface, which would bias the range.
        Pair with bundle_camera_frame for the frame the point is expressed in.

        `min_depth` overrides the DEPTH_MIN_M floor on valid depth. The default
        (0.1 m) is tuned for the head camera and would clip a hand camera that is
        deliberately CLOSER than that to its target, so a close-range caller must
        lower it — see grasp.py's visual servo."""
        if bundle is None or det is None:
            return None
        depth_msg, info = bundle.depth_image, bundle.camera_info
        if not depth_msg.data or not info.width or not info.height:
            self.get_logger().warn(
                'bundle_detection_point: bundle carries no depth/caminfo — is '
                'yolo_server new enough to attach them?')
            return None
        try:
            depth = decode_compressed_depth_image(depth_msg).astype(np.float32) / 1000.0
        except (ValueError, TypeError) as e:
            self.get_logger().warn(f'bundle_detection_point: depth decode failed: {e}')
            return None
        h, w = depth.shape
        if (int(info.height), int(info.width)) != (h, w):
            self.get_logger().warn(
                f'bundle_detection_point: depth {h}x{w} != caminfo '
                f'{info.height}x{info.width} — depth is not aligned to the color '
                'grid the boxes were measured in')
            return None
        cx_box = (det.bbox_min.x + det.bbox_max.x) / 2.0
        cy_box = (det.bbox_min.y + det.bbox_max.y) / 2.0
        half_w = max(1.0, abs(det.bbox_max.x - det.bbox_min.x) * 0.25)
        half_h = max(1.0, abs(det.bbox_max.y - det.bbox_min.y) * 0.25)
        u0, u1 = max(0, int(round(cx_box - half_w))), min(w, int(round(cx_box + half_w)))
        v0, v1 = max(0, int(round(cy_box - half_h))), min(h, int(round(cy_box + half_h)))
        if u1 <= u0 or v1 <= v0:
            return None
        mask = np.zeros((h, w), dtype=bool)
        mask[v0:v1, u0:u1] = True
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        pts = deproject_mask(mask, depth, fx, fy, cx, cy, min_depth, DEPTH_MAX_M)
        if pts is None or len(pts) == 0:
            return None
        return np.median(pts, axis=0)

    def plan_grasp(self, cloud, frame, gripper_name, scene_cloud=None, arm=''):
        """Send an (N, 3) object cloud to the graspgen service. Pass an optional
        (M, 3) `scene_cloud` (same frame) to have the server collision-filter
        grasps against surrounding obstacles. Pass `arm` ('left'/'right') so the
        server keeps only that arm's approach-direction sector (right: forward..+y,
        left: forward..-y). Returns the GraspGen response (ranked `grasps`
        PoseStamped[], `scores`, `gripper_width`) in `frame`, or None."""
        if cloud is None or len(cloud) < MIN_GRASP_POINTS:
            self.get_logger().error('plan_grasp: empty/too-small cloud')
            return None
        header = Header()
        header.frame_id = frame
        header.stamp = self.get_clock().now().to_msg()
        cloud_msg = point_cloud2.create_cloud_xyz32(
            header, np.asarray(cloud, dtype=np.float32))
        req = GraspGen.Request()
        req.object_cloud = cloud_msg
        if scene_cloud is not None and len(scene_cloud):
            req.scene_cloud = point_cloud2.create_cloud_xyz32(
                header, np.asarray(scene_cloud, dtype=np.float32))
        req.gripper_name = gripper_name
        req.arm = arm
        resp = self._call_service(self.graspgen_cli, req, 'GraspGen', timeout_sec=60.0)
        if resp is None or not resp.success or not resp.grasps:
            if resp is not None and resp.message:
                self.get_logger().error(f'plan_grasp: {resp.message}')
            return None
        return resp

    # ------------------------------------------------------- motion primitives
    def move_frame_to(self, frame, pose,
                      outer_gh=None, duration_sec=3, do_plan=True,
                      slow_mode=False, label='', stable_stop_msgs=None):
        """Send a frame to `pose` (position + orientation, in the pelvis frame) via
        /frame_task as a single combined move to the full target pose (target
        position + target orientation reached together).

        `frame` is the URDF frame to drive (GRASP_FRAMES[arm], the GraspGenX
        gripper-base frame, for grasping). Pass the skill's goal handle as outer_gh
        so a skill cancel promptly cancels the in-flight frame_task goal too. Returns
        whether the move reached the target.

        `slow_mode` asks the frame_task server to run this goal at
        SLOW_MODE_SCALE (1/4) arm speed — for delicate moves near an object or a
        held payload. NOTE it does not stretch `duration_sec`, which is a
        TIMEOUT: a slow move covers ~1/4 the distance in the same budget, so
        callers must lengthen the duration themselves (see SLOW_MODE_TIME_SCALE
        at the skill call sites).

        `label` names WHAT this move is for (e.g. 'contact descent'); it is
        carried into every frame_task log line so a stalled or runaway move can
        be attributed to the pipeline step that commanded it.

        `stable_stop_msgs` (int, or None to disable): stop the move early and
        return once the server's streamed feedback repeats UNCHANGED for this
        many messages — the arm has settled and the server is only holding
        steady state. See _send_frame_move."""
        x, y, z = float(pose.position.x), float(pose.position.y), float(pose.position.z)
        quat = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        target = Pose()
        target.position = Point(x=x, y=y, z=z)
        target.orientation = Quaternion(
            x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))
        return self._send_frame_move(frame, target, duration_sec, outer_gh, do_plan,
                                     slow_mode=slow_mode, label=label,
                                     stable_stop_msgs=stable_stop_msgs)

    def _send_frame_move(self, frame, pose, duration_sec, outer_gh, do_plan,
                         slow_mode=False, label='', stable_stop_msgs=None):
        """Command ONE /frame_task goal: drive `frame` to `pose` (a Pose already in
        the pelvis frame) over `duration_sec`, returning whether it reached the
        target (gated on the server's streamed IK convergence). The building block
        for move_frame_to.

        `stable_stop_msgs`: if set, cancel the goal and return as soon as the
        server's streamed (lin, ang) feedback repeats UNCHANGED for that many
        messages — the arm has settled and the server is only holding steady
        state (which otherwise runs to its hold timeout). The reached/not-reached
        verdict still comes from the same loose final-error gate below."""
        goal = FrameTask.Goal()
        goal.frame_names = [frame]
        goal.frame_targets = [pose]
        goal.plan = do_plan
        goal.slow_mode = bool(slow_mode)
        # Every log line for this move carries the caller's label so the
        # purpose of the motion is visible next to its convergence stream.
        tag = f'{frame} | {label}' if label else frame
        x, y, z = pose.position.x, pose.position.y, pose.position.z
        # Preserve fractional seconds (int() truncation silently shortened the
        # motion budget, e.g. 1.8s -> 1s).
        whole = int(duration_sec)
        goal.duration = Duration(
            sec=whole, nanosec=int(round((float(duration_sec) - whole) * 1e9)))

        self.get_logger().info(
            f'frame_task: {tag} -> ({x:.3f}, {y:.3f}, {z:.3f}) in {duration_sec}s'
            + (' [SLOW]' if slow_mode else '') + (' [planned]' if do_plan else ''))

        # Log the frame_task server's streamed IK convergence (errors_linear /
        # errors_angular, one entry per driven frame) so a skill's approach/contact
        # motion shows whether it actually reached the target instead of just a
        # final pass/fail. Throttled — the server publishes every control step.
        # `last` retains the most recent values for a one-line summary on resolve.
        last = {}
        # Early-stop when the feedback stops changing (stable_stop_msgs): count
        # consecutive identical (rounded) feedbacks and set the event at the
        # threshold so the result wait bails and the goal is canceled.
        stable_event = threading.Event() if stable_stop_msgs else None
        stable = {'ref': None, 'n': 0}

        def _log_feedback(feedback_msg):
            fb = feedback_msg.feedback
            last['lin'] = max(fb.errors_linear) if fb.errors_linear else 0.0
            last['ang'] = max(fb.errors_angular) if fb.errors_angular else 0.0
            self.get_logger().info(
                f'frame_task[{tag}] converging: '
                f'lin={last["lin"] * 1000:.1f}mm ang={last["ang"]:.3f}rad',
                throttle_duration_sec=0.5)
            if stable_event is not None and not stable_event.is_set():
                # Stable = within a small band of the running reference (NOT
                # exact equality — steady-state feedback jitters ~0.1 mm, which
                # would reset an exact match every message). The reference is
                # only re-anchored when a reading leaves the band, so real motion
                # keeps resetting the count while a settled hold accrues it.
                ref = stable['ref']
                if (ref is not None
                        and abs(last['lin'] - ref[0]) <= FRAME_TASK_STABLE_EPS_LIN
                        and abs(last['ang'] - ref[1]) <= FRAME_TASK_STABLE_EPS_ANG):
                    stable['n'] += 1
                else:
                    stable['ref'], stable['n'] = (last['lin'], last['ang']), 1
                if stable['n'] >= stable_stop_msgs:
                    self.get_logger().info(
                        f'frame_task[{tag}] feedback stable (within '
                        f'{FRAME_TASK_STABLE_EPS_LIN * 1000:.1f}mm/'
                        f'{FRAME_TASK_STABLE_EPS_ANG:.3f}rad) for {stable_stop_msgs} '
                        f'msgs (lin={last["lin"] * 1000:.1f}mm '
                        f'ang={last["ang"]:.3f}rad) — stopping early')
                    stable_event.set()

        response = self._send_action(
            self.frame_task_cli, goal, feedback_cb=_log_feedback,
            result_timeout=float(duration_sec) + 10.0, outer_gh=outer_gh,
            stop_event=stable_event)
        stable_stopped = stable_event is not None and stable_event.is_set()
        ok = stable_stopped or (
            response is not None and response.status == GoalStatus.STATUS_SUCCEEDED)
        # The frame_task server reports success even when it times out before
        # converging, so its status alone can't tell a reached target from an
        # unreachable one. Gate on the last streamed pelvis-frame error instead
        # (loose vs the server's 1mm/2mrad to absorb feedback-vs-break jitter).
        if ok and last:
            ok = last['lin'] < 0.01 and last['ang'] < 0.05
        if last:
            self.get_logger().info(
                f'frame_task[{tag}] done: lin={last["lin"] * 1000:.1f}mm '
                f'ang={last["ang"]:.3f}rad success={ok}')
        return ok

    def _transform_pose(self, pose, source_frame, target_frame):
        """Express `pose` (a geometry_msgs/Pose given in `source_frame`) in
        `target_frame` via live TF. Returns a new Pose, or None if the transform
        is unavailable. Uses Time() (latest available) — like _depth_to_cloud — to
        avoid extrapolation-into-the-future errors on a lagging SLAM/odom link."""
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame, source_frame, Time(), timeout=RclpyDuration(seconds=1.0))
        except TransformException as e:
            self.get_logger().warn(
                f'_transform_pose: TF {source_frame!r} -> {target_frame!r} failed: {e}')
            return None
        # transform_to_matrix(lookup_transform(target, source)) == T^target_source,
        # mapping source coords -> target coords (same convention as _depth_to_cloud).
        return matrix_to_pose(transform_to_matrix(tf.transform) @ pose_to_matrix(pose))

    def _world_frame_error(self, frame, target_pos, target_quat):
        """Measured (lin_m, ang_rad) between `frame`'s ACTUAL pose in WORLD_FRAME
        and a world target (np.array xyz, (x,y,z,w) quat), or (None, None) on TF
        failure. lookup_transform(WORLD_FRAME, frame) IS the frame's world pose."""
        try:
            tf = self.tf_buffer.lookup_transform(
                WORLD_FRAME, frame, Time(), timeout=RclpyDuration(seconds=0.5))
        except TransformException as e:
            self.get_logger().warn(f'servo: lookup {WORLD_FRAME} -> {frame!r}: {e}')
            return None, None
        t, q = tf.transform.translation, tf.transform.rotation
        lin = float(np.linalg.norm(np.array([t.x, t.y, t.z]) - target_pos))
        ang = quat_geodesic((q.x, q.y, q.z, q.w), target_quat)
        return lin, ang

    def goto_named_config(self, name, duration_sec=0.0, plan=False,
                          slow_mode=False, result_timeout=60.0, outer_gh=None):
        """Drive the arms to the frame_task server's named configuration `name`
        (e.g. 't_pose', 'home') via the /named_config action. `duration_sec`
        0 = the server's default move timeout; `plan`=True requests the planned
        move instead of the direct IK descent. `slow_mode` runs the move at the
        server's reduced (SLOW_MODE_SCALE) speed — pass a larger `duration_sec`
        (and result_timeout) so the slow move has budget to arrive. Returns True
        when the server reports success. Pass the outer skill goal handle as
        `outer_gh` so canceling the skill also cancels the in-flight arm move."""
        goal = NamedConfig.Goal()
        goal.plan = bool(plan)
        goal.config_name = name
        goal.slow_mode = bool(slow_mode)
        goal.duration.sec = int(duration_sec)
        goal.duration.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
        response = self._send_action(self.named_config_cli, goal,
                                     result_timeout=result_timeout,
                                     outer_gh=outer_gh)
        ok = (response is not None
              and response.status == GoalStatus.STATUS_SUCCEEDED
              and response.result.success)
        self.get_logger().info(
            f'named_config {name!r}: {"reached" if ok else "FAILED"}')
        return ok

    def servo_frame_to_world(self, frame, world_pose, fallback_pose, outer_gh=None,
                             duration_sec=10.0, max_iter=SERVO_ITER,
                             lin_tol=SERVO_LIN_TOL, ang_tol=SERVO_ANG_TOL,
                             do_plan=True, slow_mode=False, label='',
                             stable_stop_msgs=None):
        """Drive `frame` to `world_pose` (a Pose in WORLD_FRAME), compensating for
        pelvis drift during the motion. Each iteration re-resolves the fixed world
        target into the LIVE pelvis frame, commands frame_task there, then MEASURES
        the frame's actual world pose and stops once within (lin_tol m, ang_tol rad).

        Returns True on world convergence (or best-effort after max_iter on a
        reachable target); False if the first move leaves the frame far from the
        world target (unreachable -> the caller tries the next grasp candidate).

        If `world_pose` is None (no world anchor — navigation/odom down) or the
        world TF drops mid-servo, falls back to ONE direct pelvis move to
        `fallback_pose` so the skill still works without drift compensation.

        `do_plan` is forwarded to every frame_task move: pass False for a short,
        committed straight-in move (e.g. pre-grasp -> contact) where collision-aware
        planning would fight the intended approach into the object.

        `slow_mode` is likewise forwarded to every move of the servo, including
        the drift-fix iterations and the no-world-anchor fallback, so the whole
        servo runs at one speed. The CALLER scales `duration_sec` by
        SLOW_MODE_TIME_SCALE; the short per-iteration drift-fix budget is scaled
        here, since callers never see it.

        `label` names what this servo is for; it is forwarded into every
        frame_task move (with the iteration appended) and used in the servo's
        own log lines."""
        tag = f'{frame} | {label}' if label else frame
        if world_pose is None:
            self.get_logger().warn(
                f'servo[{tag}]: no {WORLD_FRAME} anchor; direct pelvis move '
                '(pelvis-drift compensation OFF)')
            return self.move_frame_to(frame, fallback_pose, outer_gh=outer_gh,
                                      duration_sec=duration_sec, do_plan=do_plan,
                                      slow_mode=slow_mode, label=label,
                                      stable_stop_msgs=stable_stop_msgs)
        tp = np.array([world_pose.position.x, world_pose.position.y,
                       world_pose.position.z])
        tq = (world_pose.orientation.x, world_pose.orientation.y,
              world_pose.orientation.z, world_pose.orientation.w)
        ok, lin, ang = False, None, None
        for i in range(int(max_iter)):
            if outer_gh is not None and outer_gh.is_cancel_requested:
                return False
            pelvis_pose = self._transform_pose(world_pose, WORLD_FRAME, 'pelvis')
            if pelvis_pose is None:            # lost world TF mid-servo -> fallback
                self.get_logger().warn(
                    f'servo[{tag}]: lost {WORLD_FRAME} TF; pelvis fallback')
                return self.move_frame_to(frame, fallback_pose, outer_gh=outer_gh,
                                          duration_sec=duration_sec, do_plan=do_plan,
                                          slow_mode=slow_mode, label=label,
                                          stable_stop_msgs=stable_stop_msgs)
            # iter 0 = full approach (caller already scaled it for slow mode);
            # later = short drift fixes, scaled here so they too reach at 1/4 speed.
            dur = duration_sec if i == 0 else (
                3.0 * (SLOW_MODE_TIME_SCALE if slow_mode else 1.0))
            iter_label = (f'{label} (drift fix {i})' if label and i else label)
            ok = self.move_frame_to(frame, pelvis_pose, outer_gh=outer_gh,
                                    duration_sec=dur, do_plan=do_plan,
                                    slow_mode=slow_mode, label=iter_label,
                                    stable_stop_msgs=stable_stop_msgs)
            lin, ang = self._world_frame_error(frame, tp, tq)
            if lin is None:                    # could not measure -> pelvis-frame result
                return ok
            self.get_logger().info(
                f'servo[{tag}] iter {i}: {WORLD_FRAME} err lin={lin * 1000:.1f}mm '
                f'ang={ang:.3f}rad (tol {lin_tol * 1000:.0f}mm/{ang_tol:.3f}rad, '
                f'reached={ok})')
            if lin <= lin_tol and ang <= ang_tol:
                return True
            # First move both failed to converge in pelvis AND left a large world
            # error => genuinely unreachable; bail so the caller tries the next grasp.
            if i == 0 and not ok and (lin > 0.05 or ang > 0.20):
                self.get_logger().warn(
                    f'servo[{tag}]: unreachable; caller tries next grasp')
                return False
        # Ran the full iteration budget without hitting tolerance. This is NOT a
        # failure: return best-effort success so the caller proceeds with the
        # near-converged pose rather than aborting on non-convergence. Genuine
        # unreachability is already caught by the iter-0 fast-fail above.
        self.get_logger().warn(
            f'servo[{tag}]: tol not met in {max_iter} iters; proceeding '
            f'best-effort (last lin={lin * 1000:.1f}mm ang={ang:.3f}rad)')
        return True

    def _on_gripper_state(self, arm, msg):
        """Cache the driver's published gripper aperture [mm] for `arm`."""
        self._gripper_state_mm[arm] = float(msg.position)

    def gripper_aperture(self, arm):
        """Latest gripper opening width [mm] for `arm`, or None. Prefers the
        driver's published state (which tracks the aperture after ANY close,
        including the force-based /close); falls back to the last position-command
        result when no state has been received yet."""
        val = self._gripper_state_mm.get(arm)
        return val if val is not None else self._gripper_actual_mm.get(arm)

    def set_gripper(self, arm, position_mm, speed=1.0):
        """Direct position command (mm); used to pre-open to a measured grasp
        width. Full open/close go through the dedicated services below.

        Cancels any grip-hold loop on this arm FIRST: this is the one funnel
        every aperture command goes through (open_gripper included), so an
        explicit position always wins over a hold that would otherwise keep
        re-closing on top of it."""
        self._stop_grip_hold(arm)
        req = SetGripperPosition.Request()
        req.position = float(position_mm)
        req.speed = float(speed)
        result = self._call_service(
            self.gripper_clis[arm], req, f'SetGripperPosition({arm})')
        if result is None:
            return False
        self._gripper_actual_mm[arm] = float(result.actual_position)
        self.get_logger().info(
            f'gripper {arm}: success={result.success} '
            f'actual={result.actual_position:.2f} mm — {result.message}')
        return result.success

    def set_gripper_force(self, arm, max_force_n=GRIP_FORCE_N):
        """Bound the grip force (N) before closing on an object."""
        req = SetGripperForce.Request()
        req.max_force = float(max_force_n)
        result = self._call_service(
            self.gripper_force_clis[arm], req, f'SetGripperForce({arm})')
        if result is None:
            return False
        self.get_logger().info(
            f'gripper {arm} force limit {max_force_n:.1f} N: '
            f'success={result.success} — {result.message}')
        return result.success

    def _trigger_gripper(self, client, name):
        result = self._call_service(client, Trigger.Request(), name)
        if result is None:
            return False
        self.get_logger().info(f'{name}: success={result.success} — {result.message}')
        return result.success

    def open_gripper(self, arm, amount=1.0):
        """Open the gripper by `amount`, a fraction of its full range where 0 is
        fully closed and 1 is fully open (default 1 = fully open). Commanded as a
        position so intermediate openings are honored. Waits GRIPPER_SETTLE_SEC
        after commanding so the fingers finish moving before the arm does.

        This is what ENDS a grip-hold loop (via set_gripper): the hold started by
        close_gripper runs until the gripper is opened, outliving the skill that
        grasped."""
        amount = min(max(float(amount), 0.0), 1.0)
        if not self.set_gripper(arm, amount * GRIPPER_MAX_WIDTH_MM):
            return False
        self.get_clock().sleep_for(RclpyDuration(seconds=GRIPPER_SETTLE_SEC))   # let the fingers finish before the arm moves
        return True

    def close_gripper(self, arm, force_n=GRIP_FORCE_N):
        """Close the gripper with a SET FORCE, using the gripper node's own
        services: bound the grip force (torque limit) via set_force, then trigger
        the node's /close, which drives the fingers shut and holds against the
        object at that force. A force-based close rather than a position command —
        the fingers seek and grip the object at `force_n` (N) whatever its width.
        Waits GRIPPER_SETTLE_SEC after triggering so the fingers finish closing
        before the arm moves (the /close service returns almost immediately).

        Then starts the GRIP-HOLD LOOP (_grip_hold_loop): a background thread
        that re-asserts the grip at GRIP_HOLD_RATE_HZ so a Dynamixel overload
        can't quietly drop the object. It keeps running after this call and
        after the skill returns — the object is still held — until the gripper
        is opened. Every caller of close_gripper gets this."""
        if not self.set_gripper_force(arm, force_n):
            return False
        if not self._trigger_gripper(self.gripper_close_clis[arm], f'close({arm})'):
            return False
        self.get_clock().sleep_for(RclpyDuration(seconds=GRIPPER_SETTLE_SEC))   # let the fingers finish closing before the arm moves
        # Only now, so the loop's first tick can't race the initial close.
        self._start_grip_hold(arm)
        return True

    # ------------------------------------------------------------- grip hold
    def _start_grip_hold(self, arm):
        """Start (or restart) the background grip-hold loop for `arm`. Safe to
        call when one is already running — the old thread is stopped first, so
        there is never more than one loop per arm."""
        self._stop_grip_hold(arm)
        stop = threading.Event()
        thread = threading.Thread(target=self._grip_hold_loop, args=(arm, stop),
                                  name=f'grip_hold_{arm}', daemon=True)
        self._grip_hold[arm] = (thread, stop)
        thread.start()
        self.get_logger().info(
            f'gripper {arm}: grip-hold loop started at {GRIP_HOLD_RATE_HZ:.0f} Hz '
            '(reset_overload + close); runs until the gripper is opened')

    def _stop_grip_hold(self, arm):
        """Stop the grip-hold loop for `arm` and WAIT for its thread to finish,
        so no tick is in flight when the caller then commands the gripper —
        otherwise a stray re-close would race the open it is about to send."""
        entry = self._grip_hold.pop(arm, None)
        if entry is None:
            return
        thread, stop = entry
        stop.set()
        thread.join(timeout=2.0)
        if thread.is_alive():
            self.get_logger().warn(
                f'gripper {arm}: grip-hold thread did not stop within 2s; '
                'it will exit on its next tick')
        else:
            self.get_logger().info(f'gripper {arm}: grip-hold loop stopped')

    def _grip_hold_loop(self, arm, stop):
        """Re-assert the grip on `arm` until `stop` is set: every tick, call
        reset_overload (clear any Dynamixel overload without releasing) and then
        close (re-drive the fingers shut at the force set by close_gripper).

        Runs on its own thread. That is safe here because the node spins a
        MultiThreadedExecutor over one ReentrantCallbackGroup, so blocking on a
        service future off the executor thread does not deadlock it. Calls go
        through _wait_future directly rather than _call_service so that failures
        can be throttled — at this rate _call_service's per-failure error log
        would be ~10 lines/s against a dead service.

        Failures are never fatal: a tick that fails is logged (throttled) and
        the loop simply tries again."""
        period = 1.0 / GRIP_HOLD_RATE_HZ
        calls = ((self.gripper_reset_clis[arm], 'reset_overload'),
                 (self.gripper_close_clis[arm], 'close'))
        fails = 0
        while not stop.is_set():
            ok, failed = True, ''
            for client, label in calls:
                if stop.is_set():
                    return
                result = None
                if client.service_is_ready():
                    result = self._wait_future(
                        client.call_async(Trigger.Request()),
                        GRIP_HOLD_CALL_TIMEOUT_SEC)
                if result is None or not result.success:
                    ok, failed = False, label
            if ok:
                fails = 0
            else:
                if fails % GRIP_HOLD_WARN_EVERY == 0:
                    self.get_logger().warn(
                        f'gripper {arm}: grip-hold {failed} failed '
                        f'({fails + 1} consecutive tick(s)); still retrying')
                fails += 1
            if stop.wait(period):
                break

    def _validated_arm(self, goal):
        arm = goal.arm.strip().lower()
        if arm not in GRASP_FRAMES:
            return None
        return arm
