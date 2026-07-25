"""SkillPickPlace: prep -> detect the place target -> grasp the pick object
(in-process /skill/grasp) -> lift it clear -> park the graspgenx frame
top-down above the place target -> open the gripper.

The place is deliberately simple: no orientation fan or rest-height
descent. The place pose reuses the grasp skill's top-down
orientation (_top_down_pose — forward azimuth, level wrist, pitched
TOP_DOWN_PITCH_DEG down) with the finger CONTACT point PLACE_HEIGHT above the
place target's top face, and the object is dropped from there.

The place point is detected BEFORE the (long) grasp and anchored in
WORLD_FRAME so pelvis drift during the grasp does not move the goal; at place
time it is re-resolved into the live pelvis frame and the commanded pose is
driven with servo_frame_to_world's drift compensation. Without a world TF the
skill falls back to the detect-time pelvis point, uncompensated — the same
degradation as the grasp skill without odometry.

The post-grasp lift is world-anchored the same way (see the LIFT_* constants):
its rungs are absolute offsets from the executed grasp pose along world +Z,
each re-servoed against the measured world pose, and they drive the PINCH-POINT
frame rather than the GraspGenX gripper-base frame — so the pull stays vertical
and straight where the object actually sits, not 11.5 cm behind it.
"""

import copy

import numpy as np

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time
from tf2_ros import TransformException

from custom_ros_messages.action import SkillGrasp, SkillPickPlace

from ..base import _Run, GRASP_FRAMES, WORLD_FRAME
from ..perception_utils import (matrix_to_pose, pose_to_matrix,
                                transform_to_matrix)
from .grasp import (_approach_target_tf, _pose_at, _top_down_pose,
                    SERVO_MAX_ITER, SERVO_LIN_TOL, SERVO_ANG_TOL,
                    APPROACH_DURATION_SEC, FRAME_TASK_STABLE_STOP_MSGS)


# [m] height of the released object's CONTACT point above the place target's
# top face — the drop distance.
PLACE_HEIGHT = 0.10
# z-band below the place cloud's top percentile counted as its top face when
# averaging the drop point (excludes side-face points that would skew it).
TOP_FACE_BAND = 0.015
# Portion of the skill budget reserved for the place: the inner grasp goal's
# timeout is the remaining budget minus this, so a slow Gemini detect can't
# starve the placement half of the skill.
PLACE_RESERVE_SEC = 60.0
LIFT_HEIGHT = 0.10       # [m] total straight-up lift after the grasp closes
# [m] per-waypoint increment of that lift: the rise is commanded as a ladder of
# LIFT_STEP moves instead of one LIFT_HEIGHT move, so the arm clears the pick
# site in short committed hops that stay interruptible. The line is anchored
# ONCE at the executed grasp (outcome.pose) and each waypoint is an ABSOLUTE
# offset from that anchor, so per-rung error cannot accumulate into the next —
# and every rung also pulls the hand back onto the grasp's own vertical instead
# of inheriting whatever residual and sag the close left behind.
#
# The anchor is in WORLD_FRAME whenever the grasp had a world TF, so "+Z" is
# GRAVITY up and each rung is re-servoed against the measured world pose
# (_servo_pose). Commanding pelvis +Z instead makes the pull follow the torso:
# under a balancing lower body the pelvis pitches and sways during a rung, and
# an uncompensated pelvis-frame command drags the held object sideways.
LIFT_STEP = 0.05
# [s] commanded motion time per lift increment.
LIFT_STEP_DURATION_SEC = 4.0
# The lift drives the PINCH-POINT frame, not the GraspGenX gripper-base frame
# the grasp targets are computed for: the two sit GRIPPER_BASE_TO_CONTACT_M
# (0.1146 m) apart along the approach, so nulling the error at the base lets a
# wrist-orientation residual swing the fingertips through ~0.1146 x ang_tol
# without the move ever noticing. On a small part held in the fingertips that
# arc is the difference between lifting it and rolling it out of the jaws.
PINCH_FRAMES = {'left': 'left_grasp_frame', 'right': 'right_grasp_frame'}
# Debug TF (RViz): the detected place point / commanded place pose.
PLACE_TARGET_FRAME = 'pick_place_target_frame'


class PickPlaceSkill:
    def _exec_pick_place(self, gh):
        """prep -> detect (place target) -> grasp -> place above -> release.

        Optional goal fields (see SkillPickPlace.action; zero/empty = off):
          arm — "" or "none" leaves the pick arm to the grasp skill (nearest
            arm); the resolved arm comes back on the outcome.
          top_down, visual_servo — forwarded to /skill/grasp."""
        goal = gh.request
        run = _Run(self, gh, SkillPickPlace, 'pick_place')
        arm = self._validated_arm(goal)
        if arm is None and goal.arm.strip().lower() not in ('', 'none'):
            return run.abort(f'invalid arm {goal.arm!r}')
        obj, place = goal.target_object, goal.place_target
        if not place.strip():
            return run.abort('empty place_target')

        def fail(message):
            """Abort, lifting the hand clear first (mid-carry / over the
            stack). `arm` is read at call time — it is None until the inner
            grasp reports which hand it actually used."""
            self._retreat_after_failure(arm, gh)
            return run.abort(message)

        # --- detect: place-target top face, world-anchored --------------------
        if not run.phase('detect_place_target', 0.0):
            return run.result
        # Arms clear of the head camera before the detection image is grabbed
        # (the grasp skill preps again before its own detection — a no-op by
        # then). Fires before this skill has moved the arm itself: no retreat.
        if not self.goto_named_config('prep', plan=True, slow_mode=True,
                                      duration_sec=30.0, result_timeout=90.0,
                                      outer_gh=gh):
            return run.abort("move to 'prep' before detection failed")
        place_cloud, _, err = self.detect_object_cloud(place, run, gh)
        if err:
            return fail(f'place target: {err}')
        top_p = _top_face_point(place_cloud)          # (x, y, z_top) in pelvis
        top_w = self._transform_pose(_pose_at(top_p), 'pelvis', WORLD_FRAME)
        self.publish_tf(_approach_target_tf(
            WORLD_FRAME if top_w is not None else 'pelvis',
            top_w if top_w is not None else _pose_at(top_p),
            self.get_clock().now().to_msg(), child=PLACE_TARGET_FRAME))
        self.get_logger().info(
            f'pick_place: {place!r} top face at pelvis '
            f'({top_p[0]:.3f}, {top_p[1]:.3f}, {top_p[2]:.3f}), '
            f'world anchor {"ok" if top_w is not None else "UNAVAILABLE"}')

        # --- grasp: in-process /skill/grasp -----------------------------------
        if not run.phase('grasp_object', 0.2):
            return run.result
        outcome, err = self._grasp_via_skill(goal, obj, arm, run, gh)
        if err:
            return fail(err)
        arm = outcome.arm    # the hand that actually grasped drives the place

        # --- lift: rise LIFT_HEIGHT straight up as a ladder of LIFT_STEP hops,
        # anchored ONCE at the grasp that was just executed (outcome.pose — in
        # WORLD_FRAME when the grasp had a world TF, so the rungs climb GRAVITY
        # +Z and each is drift-compensated). Targets are computed for the
        # GraspGenX frame the grasp pose is expressed in, then re-expressed for
        # the PINCH-POINT frame that actually holds the object (to_pinch). See
        # the LIFT_* constants -------------------------------------------------
        lift_frame = PINCH_FRAMES[arm]
        T_off = self._pinch_frame_offset(arm)
        if T_off is None:
            return fail(f'TF {GRASP_FRAMES[arm]} -> {lift_frame} unavailable, '
                        'so the post-grasp lift target is unknown')

        def to_pinch(pose_gx):
            """A target computed FOR the GraspGenX gripper-base frame, restated
            as the equivalent target for the pinch-point frame (a body-fixed
            offset, so this holds in the world and pelvis frames alike)."""
            return matrix_to_pose(pose_to_matrix(pose_gx) @ T_off)

        world_ok = outcome.frame == WORLD_FRAME
        n_rungs = max(1, int(round(LIFT_HEIGHT / LIFT_STEP)))
        self.get_logger().info(
            f'pick_place: lifting {lift_frame} {LIFT_HEIGHT * 100:.0f}cm up '
            f'({"gravity" if world_ok else "pelvis"} +Z) in {n_rungs} x '
            f'{LIFT_STEP * 100:.0f}cm rungs from the executed grasp pose'
            + ('' if world_ok else ' — NO world anchor, drift compensation OFF'))
        for k in range(1, n_rungs + 1):
            if gh.is_cancel_requested or run.remaining() <= 0.0:
                return fail('canceled or timed out during post-grasp lift')
            dz = min(k * LIFT_STEP, LIFT_HEIGHT)
            rung = copy.deepcopy(outcome.pose)
            rung.position.z += dz
            if not self._servo_pose(
                    lift_frame, to_pinch(rung), world_ok, gh, do_plan=False,
                    duration_sec=LIFT_STEP_DURATION_SEC,
                    stable_stop_msgs=FRAME_TASK_STABLE_STOP_MSGS,
                    label=f'lift rung {k}/{n_rungs} (+{dz * 100:.0f}cm Z)'):
                return fail(f'post-grasp lift failed at {dz * 100:.0f}cm of '
                            f'{LIFT_HEIGHT * 100:.0f}cm')

        # --- place: park the graspgenx frame top-down above the target --------
        if not run.phase('place', 0.7):
            return run.result
        # Re-resolve the world-anchored place point into the LIVE pelvis frame
        # (the grasp drifted the detect-time one), then build the top-down
        # place pose: contact point PLACE_HEIGHT above the top face.
        top_now = top_p
        if top_w is not None:
            p = self._transform_pose(top_w, WORLD_FRAME, 'pelvis')
            if p is None:
                return fail(f'{WORLD_FRAME} -> pelvis TF lost before the place')
            top_now = np.array([p.position.x, p.position.y, p.position.z])
        else:
            self.get_logger().warn(
                'pick_place: no world anchor; placing in the pelvis frame '
                'with NO drift compensation')
        place_p = _top_down_pose(top_now + [0.0, 0.0, PLACE_HEIGHT])
        place_w = (self._transform_pose(place_p, 'pelvis', WORLD_FRAME)
                   if top_w is not None else None)
        self.publish_tf(_approach_target_tf(
            WORLD_FRAME if place_w is not None else 'pelvis',
            place_w if place_w is not None else place_p,
            self.get_clock().now().to_msg(), child=PLACE_TARGET_FRAME))
        self.get_logger().info(
            f'pick_place: {GRASP_FRAMES[arm]} to the top-down release pose '
            f'{PLACE_HEIGHT * 100:.0f}cm above {place!r}')
        # Planned transit (the hand carries the object across the workspace).
        if not self.servo_frame_to_world(
                GRASP_FRAMES[arm], place_w, place_p, outer_gh=gh,
                duration_sec=APPROACH_DURATION_SEC, max_iter=SERVO_MAX_ITER,
                lin_tol=SERVO_LIN_TOL, ang_tol=SERVO_ANG_TOL,
                label='carry to release pose above place target'):
            return fail(f'release pose above {place!r} unreachable')
        if gh.is_cancel_requested or run.remaining() <= 0.0:
            return fail('canceled or timed out during place')

        # --- release ----------------------------------------------------------
        if not run.phase('release', 0.9):
            return run.result
        if not self.open_gripper(arm):
            return fail('gripper open failed')
        return run.succeed(
            f'placed {obj!r} on {place!r} (released {PLACE_HEIGHT * 100:.0f}cm '
            f'above, grasp score {outcome.score:.2f})')

    def _grasp_via_skill(self, goal, obj, arm, run, gh):
        """Send an in-process /skill/grasp goal for `obj` and return
        (GraspOutcome, None) or (None, reason). The caller's top_down and
        visual_servo are forwarded explicitly — an unset field silently
        defaults to false, which is how visual_servo once got dropped here.
        `arm` None = auto (the grasp skill picks; outcome.arm reports back).
        Inner feedback is proxied into the 'grasp_object' phase; canceling the
        outer goal cancels the in-flight inner goal (_send_action's outer_gh
        plumbing)."""
        inner_timeout = max(10.0, run.remaining() - PLACE_RESERVE_SEC)
        inner = SkillGrasp.Goal()
        inner.target_object = obj
        inner.arm = arm or ''
        inner.top_down = goal.top_down
        inner.visual_servo = goal.visual_servo
        if inner_timeout == float('inf'):
            # No deadline on this goal (timeout 0 = run to completion) — a
            # zero inner Duration means exactly the same thing.
            inner.timeout = Duration(sec=0, nanosec=0)
        else:
            whole = int(inner_timeout)
            inner.timeout = Duration(
                sec=whole, nanosec=int(round((inner_timeout - whole) * 1e9)))

        def _proxy(feedback_msg):
            run.feedback.phase = 'grasp_object'
            run.feedback.progress = (
                0.2 + 0.5 * float(feedback_msg.feedback.progress))
            gh.publish_feedback(run.feedback)

        resp = self._send_action(self.grasp_skill_cli, inner, feedback_cb=_proxy,
                                 result_timeout=inner_timeout + 15.0, outer_gh=gh)
        if (resp is None or resp.status != GoalStatus.STATUS_SUCCEEDED
                or not resp.result.success):
            msg = (resp.result.message if resp is not None
                   else 'goal failed or timed out')
            return None, f'grasp failed: {msg}'
        outcome = getattr(self, '_last_grasp_outcome', None)
        if outcome is None:
            return None, f'grasp of {obj!r} returned no outcome for placement'
        return outcome, None

    def _pinch_frame_offset(self, arm):
        """Fixed rigid transform (4x4) of the arm's pinch-point frame
        (PINCH_FRAMES) expressed in its GraspGenX gripper-base frame — both are
        fixed URDF children of the wrist yaw link, so the offset is constant;
        read once per goal from TF. A target pose computed FOR the graspgenx
        frame becomes the equivalent pinch-frame target as T_target @ offset.
        None when TF can't resolve it (robot description not up)."""
        try:
            tf = self.tf_buffer.lookup_transform(
                GRASP_FRAMES[arm], PINCH_FRAMES[arm], Time(),
                timeout=RclpyDuration(seconds=1.0))
        except TransformException:
            return None
        return transform_to_matrix(tf.transform)

    def _servo_pose(self, frame, pose, world_ok, gh, do_plan,
                    duration_sec=LIFT_STEP_DURATION_SEC, slow_mode=False,
                    label='', stable_stop_msgs=None):
        """Drive URDF frame `frame` to `pose` — a Pose in WORLD_FRAME when
        `world_ok` (servo_frame_to_world's drift compensation: the fixed world
        target is re-resolved into the LIVE pelvis frame every pass and the
        frame's actual world pose is measured back), else a pelvis-frame Pose
        driven directly with no compensation. Same relaxed tolerances as the
        grasp skill's servo."""
        if world_ok:
            world_pose = pose
            fallback = self._transform_pose(pose, WORLD_FRAME, 'pelvis')
            if fallback is None:
                self.get_logger().warn(
                    f'pick_place: {WORLD_FRAME} -> pelvis TF unavailable')
                return False
        else:
            world_pose, fallback = None, pose
        return self.servo_frame_to_world(
            frame, world_pose, fallback, outer_gh=gh,
            duration_sec=duration_sec, max_iter=SERVO_MAX_ITER,
            lin_tol=SERVO_LIN_TOL, ang_tol=SERVO_ANG_TOL, do_plan=do_plan,
            slow_mode=slow_mode, label=label,
            stable_stop_msgs=stable_stop_msgs)


def _top_face_point(cloud):
    """(x, y, z_top) of `cloud`'s top face, in the cloud's own (z-up) frame:
    z_top is the 95th-percentile z (robust to depth speckle) and x, y average
    only the points within TOP_FACE_BAND below it — the visible top surface —
    so side-face points don't skew the drop point."""
    z = cloud[:, 2]
    z_top = float(np.percentile(z, 95))
    top = cloud[z > z_top - TOP_FACE_BAND]
    if len(top) == 0:                 # degenerate band -> whole-cloud centroid
        top = cloud
    return np.array([float(np.mean(top[:, 0])), float(np.mean(top[:, 1])), z_top])
