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
`plan_grasp`) and the motion primitives (`move_frame_to`, gripper ops) are
implemented here; each skill under skills/ composes them in its `_exec_*` mixin.
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
from geometry_msgs.msg import Point, Pose, Quaternion
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Header

from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformBroadcaster, TransformListener, TransformException

from custom_ros_messages.srv import GeminiQuery, SamSegment, GraspGen
from custom_ros_messages.action import FrameTask
from custom_ros_messages.action import (
    SkillCloseDoor, SkillOpenDoor, SkillCloseLid, SkillOpenLid,
    SkillNavigate, SkillGrasp, SkillPickPlace, SkillPress, SkillSlideRack,
    SkillTurnLever, SkillTwistKnob,
)
from magpie_msgs.srv import SetGripperPosition, SetGripperForce
from std_srvs.srv import Trigger

from .perception_utils import (
    decode_compressed_depth_image, deproject_mask, transform_points,
    transform_to_matrix, pose_to_matrix, matrix_to_pose, quat_geodesic,
)


CAMERA_NS = '/realsense/head'
# Color (JPEG), aligned depth (compressedDepth, uint16 mm), and intrinsics.
COLOR_IMAGE_TOPIC = f'{CAMERA_NS}/color/image_raw/compressed'
DEPTH_IMAGE_TOPIC = f'{CAMERA_NS}/aligned_depth_to_color/image_raw/compressedDepth'
CAMERA_INFO_TOPIC = f'{CAMERA_NS}/color/camera_info'

# GraspGenX gripper-BASE frames (URDF frames whose axes match the GraspGenX
# planning convention: +Z approach, +X finger-close, origin at the magpie gripper
# base). A raw GraspGenX grasp pose is sent straight to frame_task targeting these
# — no axis-permutation / TCP-depth correction needed. See skills/grasp.py.
GRASP_FRAMES = {'left': 'left_graspgenx_frame', 'right': 'right_graspgenx_frame'}
# Per-arm gripper service namespace; the magpie driver/sim expose
# <ns>/set_position, <ns>/set_force, <ns>/open and <ns>/close under each.
GRIPPER_NS = {'left': '/left/gripper', 'right': '/right/gripper'}

# Grip-force limit (N) applied via set_force before closing on an object.
GRIP_FORCE_N = 30.0
# Used when goal.timeout is zero [s]. Generous because the grasp skill's Gemini
# detection alone (gemini-robotics-er) can take ~3 min; this deadline is a safety
# ceiling checked at phase boundaries, not a target — most skills finish far sooner.
DEFAULT_SKILL_TIMEOUT = 300.0
# Gemini query-service call timeout [s]. The gemini-robotics-er model used for
# grasp detection can take ~3 min to answer — far beyond _call_service's 30s
# default — so give GeminiQuery its own generous ceiling.
GEMINI_TIMEOUT_SEC = 240.0
# Depth back-projection range [m] and the floor on usable object points.
# Keep MIN_GRASP_POINTS in sync with graspgen_server.MIN_OBJECT_POINTS so a cloud
# the server would reject is dropped client-side with an accurate message.
DEPTH_MIN_M = 0.1
DEPTH_MAX_M = 3.0
MIN_GRASP_POINTS = 100

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

        # --- head-camera caches (color for the services, depth+info for lifting
        #     a 2-D mask to a 3-D cloud) -----------------------------------------
        self._latest_image = None      # color CompressedImage (for gemini/sam)
        self._latest_depth = None      # aligned depth CompressedImage (mm)
        self._latest_caminfo = None    # color CameraInfo (intrinsics + frame)
        self.create_subscription(
            CompressedImage, COLOR_IMAGE_TOPIC, self._on_color_image,
            qos_profile_sensor_data, callback_group=self._cb_group)
        self.create_subscription(
            CompressedImage, DEPTH_IMAGE_TOPIC, self._on_depth_image,
            qos_profile_sensor_data, callback_group=self._cb_group)
        self.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC, self._on_caminfo,
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

    def latest_image(self):
        """Most recent head-camera color frame (sensor_msgs/CompressedImage), or None."""
        return self._latest_image

    def latest_caminfo(self):
        """Most recent color CameraInfo, or None."""
        return self._latest_caminfo

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

    def scene_to_cloud(self, target_frame='pelvis'):
        """Back-project the whole latest depth frame (every valid pixel, no mask)
        into an (N, 3) scene cloud in `target_frame` — obstacle context for grasp
        collision filtering. Returns None if depth/caminfo/TF are unavailable."""
        return self._depth_to_cloud(None, target_frame)

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
                      outer_gh=None, duration_sec=3,):
        """Send a frame to (x, y, z) in the pelvis frame via /frame_task. Defaults
        to the arm's grip-site frame; pass `frame` to drive a different URDF frame
        (e.g. GRASP_FRAMES[arm], the GraspGenX gripper-base frame, for grasping).
        Pass the skill's goal handle as outer_gh so a skill cancel promptly cancels
        the in-flight frame_task goal too."""
        frame_name = frame
        goal = FrameTask.Goal()
        goal.frame_names = [frame_name]
        x, y, z = pose.position.x, pose.position.y, pose.position.z
        quat = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        pose = Pose()
        pose.position = Point(x=float(x), y=float(y), z=float(z))
        pose.orientation = Quaternion(
            x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))
        goal.frame_targets = [pose]
        # Preserve fractional seconds (int() truncation silently shortened the
        # motion budget, e.g. 1.8s -> 1s).
        whole = int(duration_sec)
        goal.duration = Duration(
            sec=whole, nanosec=int(round((float(duration_sec) - whole) * 1e9)))

        self.get_logger().info(
            f'frame_task: {frame_name} -> ({x:.3f}, {y:.3f}, {z:.3f}) '
            f'in {duration_sec}s')

        # Log the frame_task server's streamed IK convergence (errors_linear /
        # errors_angular, one entry per driven frame) so a skill's approach/contact
        # motion shows whether it actually reached the target instead of just a
        # final pass/fail. Throttled — the server publishes every control step.
        # `last` retains the most recent values for a one-line summary on resolve.
        last = {}

        def _log_feedback(feedback_msg):
            fb = feedback_msg.feedback
            last['lin'] = max(fb.errors_linear) if fb.errors_linear else 0.0
            last['ang'] = max(fb.errors_angular) if fb.errors_angular else 0.0
            self.get_logger().info(
                f'frame_task[{frame_name}] converging: '
                f'lin={last["lin"] * 1000:.1f}mm ang={last["ang"]:.3f}rad',
                throttle_duration_sec=0.5)

        response = self._send_action(
            self.frame_task_cli, goal, feedback_cb=_log_feedback,
            result_timeout=float(duration_sec) + 10.0, outer_gh=outer_gh)
        ok = response is not None and response.status == GoalStatus.STATUS_SUCCEEDED
        # The frame_task server reports success even when it times out before
        # converging, so its status alone can't tell a reached target from an
        # unreachable one. Gate on the last streamed pelvis-frame error instead
        # (loose vs the server's 1mm/2mrad to absorb feedback-vs-break jitter).
        if ok and last:
            ok = last['lin'] < 0.01 and last['ang'] < 0.05
        if last:
            self.get_logger().info(
                f'frame_task[{frame_name}] done: lin={last["lin"] * 1000:.1f}mm '
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

    def servo_frame_to_world(self, frame, world_pose, fallback_pose, outer_gh=None,
                             duration_sec=10.0, max_iter=SERVO_ITER,
                             lin_tol=SERVO_LIN_TOL, ang_tol=SERVO_ANG_TOL):
        """Drive `frame` to `world_pose` (a Pose in WORLD_FRAME), compensating for
        pelvis drift during the motion. Each iteration re-resolves the fixed world
        target into the LIVE pelvis frame, commands frame_task there, then MEASURES
        the frame's actual world pose and stops once within (lin_tol m, ang_tol rad).

        Returns True on world convergence (or best-effort after max_iter on a
        reachable target); False if the first move leaves the frame far from the
        world target (unreachable -> the caller tries the next grasp candidate).

        If `world_pose` is None (no world anchor — navigation/odom down) or the
        world TF drops mid-servo, falls back to ONE direct pelvis move to
        `fallback_pose` so the skill still works without drift compensation."""
        if world_pose is None:
            self.get_logger().warn(
                f'servo[{frame}]: no {WORLD_FRAME} anchor; direct pelvis move '
                '(pelvis-drift compensation OFF)')
            return self.move_frame_to(frame, fallback_pose, outer_gh=outer_gh,
                                      duration_sec=duration_sec)
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
                    f'servo[{frame}]: lost {WORLD_FRAME} TF; pelvis fallback')
                return self.move_frame_to(frame, fallback_pose, outer_gh=outer_gh,
                                          duration_sec=duration_sec)
            dur = duration_sec if i == 0 else 3.0   # iter 0 = full approach; later = drift fixes
            ok = self.move_frame_to(frame, pelvis_pose, outer_gh=outer_gh,
                                    duration_sec=dur)
            lin, ang = self._world_frame_error(frame, tp, tq)
            if lin is None:                    # could not measure -> pelvis-frame result
                return ok
            self.get_logger().info(
                f'servo[{frame}] iter {i}: {WORLD_FRAME} err lin={lin * 1000:.1f}mm '
                f'ang={ang:.3f}rad (tol {lin_tol * 1000:.0f}mm/{ang_tol:.3f}rad, '
                f'reached={ok})')
            if lin <= lin_tol and ang <= ang_tol:
                return True
            # First move both failed to converge in pelvis AND left a large world
            # error => genuinely unreachable; bail so the caller tries the next grasp.
            if i == 0 and not ok and (lin > 0.05 or ang > 0.20):
                self.get_logger().warn(
                    f'servo[{frame}]: unreachable; caller tries next grasp')
                return False
        # Ran the full iteration budget without hitting tolerance. This is NOT a
        # failure: return best-effort success so the caller proceeds with the
        # near-converged pose rather than aborting on non-convergence. Genuine
        # unreachability is already caught by the iter-0 fast-fail above.
        self.get_logger().warn(
            f'servo[{frame}]: tol not met in {max_iter} iters; proceeding '
            f'best-effort (last lin={lin * 1000:.1f}mm ang={ang:.3f}rad)')
        return True

    def set_gripper(self, arm, position_mm, speed=1.0):
        """Direct position command (mm); used to pre-open to a measured grasp
        width. Full open/close go through the dedicated services below."""
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

    def open_gripper(self, arm):
        """Open fully via the dedicated open service."""
        return self._trigger_gripper(self.gripper_open_clis[arm], f'gripper/open({arm})')

    def close_gripper(self, arm):
        """Bound the grip force, then close fully on the object."""
        if not self.set_gripper_force(arm):
            return False
        return self._trigger_gripper(self.gripper_close_clis[arm], f'gripper/close({arm})')

    def _validated_arm(self, goal):
        arm = goal.arm.strip().lower()
        if arm not in GRASP_FRAMES:
            return None
        return arm
