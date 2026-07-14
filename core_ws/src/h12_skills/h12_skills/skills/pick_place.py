"""SkillPickPlace: block stacking — detect the place target, grasp the pick
object via the node's own /skill/grasp action, then carry, place, and release.

Flow (one _Run budget across every phase):
  detect  — gemini box -> sam mask -> place-target cloud; its top-face point is
            anchored in WORLD_FRAME so pelvis drift during the (long) grasp
            does not move the goal.
  grasp   — in-process /skill/grasp goal; the executed grasp pose and
            held-object geometry come back through _last_grasp_outcome (see
            skills/grasp.py — the action messages are unchanged), and the
            reached gripper aperture confirms an object is actually held.
  carry   — lift straight up, then transit to a standoff directly ABOVE the
            commanded release pose.
  place   — descend vertically until the held object's centroid sits
            rest_height above the place target's top face. The descent is
            grasp-agnostic — the OBJECT always arrives from above, whatever the
            grasp orientation — so no downward-pitch bias is imposed on the
            grasp anymore (the old approach-axis drive-in needed pitched-down
            grasps and starved the candidate pool). A stall NEAR the release
            pose still releases (early touchdown); a distant stall aborts.
  release — open the gripper, back the fingers out along the reverse grasp
            approach so they can't catch the placed object, then rise clear.

Placement math: the gripper->object offset o_g = R^T (c - p), measured at grasp
time from the SAME pelvis snapshot as the executed grasp, is constant while the
object is held. Keeping the executed grasp's world orientation R through
carry/place keeps the object level (it lands in its pick-time orientation), so
only position is commanded: frame target = desired_centroid - R @ o_g. The
object's rest height (centroid above its own bottom, measured while it sat on
the table) is exactly how high its centroid must sit above the place target's
top face when it comes to rest there.

Commanded frame: every post-grasp motion drives the arm's pinch-point
{left,right}_grasp_frame through /frame_task (PLACE_FRAMES — the frame the
frame_task planner's frame_names list and z-floor safety track), NOT the
graspgenx gripper-base frame the math above is written in. Targets are computed
in the GraspGenX convention, then re-expressed through the fixed URDF
graspgenx->grasp-frame offset (_place_frame_offset) before being sent.
"""

import copy
from dataclasses import replace

import numpy as np

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time
from tf2_ros import TransformException

from custom_ros_messages.action import SkillGrasp, SkillPickPlace

from ..base import _Run, GRASP_FRAMES, WORLD_FRAME
from ..perception_utils import pose_to_matrix, matrix_to_pose, transform_to_matrix
from .grasp import (get_approach_pose, _approach_target_tf, _pose_at,
                    SERVO_DURATION_SEC, SERVO_MAX_ITER, SERVO_LIN_TOL,
                    SERVO_ANG_TOL)


# URDF frame the post-grasp (carry/place/release) motions COMMAND via
# /frame_task: the pinch-point "grasp frame", a fixed wrist child ~15 cm past
# the graspgenx gripper-base frame along the tool axis, and the frame the
# frame_task planner's frame_names list and z-floor safety actually track.
# Placement targets are still COMPUTED in the GraspGenX convention and
# re-expressed through the fixed URDF offset (_place_frame_offset).
PLACE_FRAMES = {'left': 'left_grasp_frame', 'right': 'right_grasp_frame'}

LIFT_HEIGHT = 0.12       # [m] straight-up lift after the grasp closes
PRE_PLACE_DIST = 0.10    # [m] pre-place standoff straight above the release pose
# [m] release the object this far above its rest pose. Keep it >= the place
# servo's convergence tolerance (SERVO_LIN_TOL, 25 mm): a tolerance larger than
# the clearance can eat it and press the object into the stack before release.
PLACE_CLEARANCE = 0.03
RETRACT_DIST = 0.15      # [m] post-release finger back-off along the reverse grasp approach
# Descent stalls within this distance of the release pose still release — early
# touchdown on the stack (rest_height is a single-view estimate); stalls farther
# away abort with the object still held rather than dropping it off-target.
PLACE_STALL_MAX_M = 0.05
# Reached gripper aperture at or below this after the grasp's close means the
# fingers met each other — nothing is held, so there is nothing to place.
MIN_HELD_APERTURE_MM = 5.0
# z-band below the place cloud's top percentile counted as its top face when
# averaging the drop point (excludes side-face points that would skew it).
TOP_FACE_BAND = 0.015
# Held-object rest height fallback when the grasp outcome measured none —
# half of a nominal 5 cm block.
FALLBACK_REST_HEIGHT = 0.025
# Portion of the skill budget reserved for carry/place/release: the inner grasp
# goal's timeout is the remaining budget minus this, so a slow Gemini detect
# can't starve the placement half of the skill.
PLACE_RESERVE_SEC = 60.0
# Debug TF (RViz): the detected place point, later moved to the commanded place
# pose for the grasp frame.
PLACE_TARGET_FRAME = 'pick_place_target_frame'


class PickPlaceSkill:
    def _exec_pick_place(self, gh):
        """detect (place target) -> grasp (in-process /skill/grasp) -> carry ->
        place -> release."""
        goal = gh.request
        run = _Run(self, gh, SkillPickPlace, 'pick_place')
        arm = self._validated_arm(goal)
        if arm is None:
            return run.abort(f'invalid arm {goal.arm!r}')
        obj, place = goal.target_object, goal.place_target
        if not place.strip():
            return run.abort('empty place_target')

        # --- detect: place-target top face, world-anchored ---------------------
        if not run.phase('detect', 0.0):
            return run.result
        place_cloud, _, err = self.detect_object_cloud(place, run, gh)
        if err:
            return run.abort(f'place target: {err}')
        top_p = _top_face_point(place_cloud)          # (x, y, z_top) in pelvis
        # Anchor the place point in the world frame NOW, while the pelvis is
        # still at its detect-time pose — the grasp motions will drift it.
        top_w_pose = self._transform_pose(_pose_at(top_p), 'pelvis', WORLD_FRAME)
        top_w = None if top_w_pose is None else np.array(
            [top_w_pose.position.x, top_w_pose.position.y, top_w_pose.position.z])
        self.publish_tf(_approach_target_tf(
            WORLD_FRAME if top_w is not None else 'pelvis',
            _pose_at(top_w if top_w is not None else top_p),
            self.get_clock().now().to_msg(), child=PLACE_TARGET_FRAME))
        self.get_logger().info(
            f'pick_place: {place!r} top face at pelvis '
            f'({top_p[0]:.3f}, {top_p[1]:.3f}, {top_p[2]:.3f}), '
            f'world anchor {"ok" if top_w is not None else "UNAVAILABLE"}')

        # --- grasp: in-process /skill/grasp -------------------------------------
        if not run.phase('grasp', 0.2):
            return run.result
        outcome, err = self._grasp_via_skill(obj, arm, run, gh)
        if err:
            return run.abort(err)
        # Verify the close actually stopped on something: the magpie driver
        # reports the aperture it reached; ~0 mm means the fingers met each
        # other and the "grasped" object isn't in the hand.
        aperture = self.gripper_aperture(arm)
        if aperture is not None and aperture <= MIN_HELD_APERTURE_MM:
            return run.abort(
                f'grasp of {obj!r} closed on nothing (gripper at {aperture:.1f}mm)')

        # --- placement geometry -------------------------------------------------
        # Work in one consistent frame: world only when BOTH the place point and
        # the grasp outcome are world-anchored, else fall back to pelvis (drift
        # uncompensated, same degradation as the grasp skill without odometry).
        world_ok = outcome.frame == WORLD_FRAME and top_w is not None
        if outcome.frame == WORLD_FRAME and not world_ok:
            # The world TF appeared mid-skill: the grasp outcome is world-anchored
            # but the place point never was. Re-express the outcome in the LIVE
            # pelvis frame so everything below shares the place anchor's frame —
            # mixing them would drive world-frame poses as pelvis poses.
            pose_p = self._transform_pose(outcome.pose, WORLD_FRAME, 'pelvis')
            cent_p = self._transform_pose(
                _pose_at(outcome.centroid), WORLD_FRAME, 'pelvis')
            if pose_p is None or cent_p is None:
                return run.abort(
                    'grasp outcome is world-anchored but the world TF is '
                    'unavailable to bring it back to the pelvis frame')
            outcome = replace(outcome, pose=pose_p, frame='pelvis',
                              centroid=np.array([cent_p.position.x,
                                                 cent_p.position.y,
                                                 cent_p.position.z]))
        if not world_ok:
            self.get_logger().warn(
                'pick_place: no consistent world anchor; placing in the pelvis '
                'frame with NO drift compensation')
        top = top_w if world_ok else top_p
        T_g = pose_to_matrix(outcome.pose)
        R, p = T_g[:3, :3], T_g[:3, 3]
        o_g = R.T @ (outcome.centroid - p)   # gripper->object, constant while held
        rest_h = (outcome.rest_height if outcome.rest_height > 0.005
                  else FALLBACK_REST_HEIGHT)
        c_release = top + [0.0, 0.0, rest_h + PLACE_CLEARANCE]
        T_place = np.eye(4)
        T_place[:3, :3] = R
        T_place[:3, 3] = c_release - R @ o_g
        place_pose = matrix_to_pose(T_place)
        # Standoff straight ABOVE the release pose (same orientation): the place
        # descent is vertical, so any grasp orientation delivers the object from
        # above instead of driving it sideways into the stack.
        pre_place = copy.deepcopy(place_pose)
        pre_place.position.z += PRE_PLACE_DIST
        self.publish_tf(_approach_target_tf(
            WORLD_FRAME if world_ok else 'pelvis', place_pose,
            self.get_clock().now().to_msg(), child=PLACE_TARGET_FRAME))
        # From here on, motions COMMAND the pinch-point grasp frame, not the
        # graspgenx frame the targets above are computed for: fetch the fixed
        # URDF offset between the two once and re-express every target with it.
        place_frame = PLACE_FRAMES[arm]
        T_off = self._place_frame_offset(arm)
        if T_off is None:
            return run.abort(
                f'TF {GRASP_FRAMES[arm]} -> {place_frame} unavailable')

        def to_place(pose_gx):
            return matrix_to_pose(pose_to_matrix(pose_gx) @ T_off)

        # --- carry: lift, then transit to the pre-place standoff ----------------
        if not run.phase('carry', 0.55):
            return run.result
        lift = copy.deepcopy(outcome.pose)
        lift.position.z += LIFT_HEIGHT
        if not self._servo_pose(place_frame, to_place(lift), world_ok, gh,
                                do_plan=False):
            return run.abort('lift after grasp failed')
        if gh.is_cancel_requested or run.remaining() <= 0.0:
            return run.abort('canceled or timed out during carry')
        if not self._servo_pose(place_frame, to_place(pre_place), world_ok, gh,
                                do_plan=True):
            return run.abort('pre-place pose unreachable')

        # --- place: descend vertically onto the stack ----------------------------
        if not run.phase('place', 0.75):
            return run.result
        place_cmd = to_place(place_pose)
        if not self._servo_pose(place_frame, place_cmd, world_ok, gh,
                                do_plan=False):
            if gh.is_cancel_requested or run.remaining() <= 0.0:
                return run.abort('canceled or timed out during place')
            # A stall NEAR the release pose is early touchdown (rest_height is a
            # single-view estimate) — release there rather than aborting while
            # pressing the object into the stack. A distant stall means the
            # descent genuinely failed; abort with the object still held.
            lin_err = self._lin_error_to(place_frame, place_cmd,
                                         WORLD_FRAME if world_ok else 'pelvis')
            if lin_err is None or lin_err > PLACE_STALL_MAX_M:
                return run.abort(
                    'place descent stalled far from the release pose'
                    + ('' if lin_err is None else f' ({lin_err * 100:.1f}cm short)'))
            self.get_logger().warn(
                f'pick_place: descent stalled {lin_err * 100:.1f}cm from the '
                'release pose (early touchdown?); releasing here')

        # --- release + retreat ----------------------------------------------------
        if not run.phase('release', 0.9):
            return run.result
        if not self.open_gripper(arm):
            return run.abort('gripper open failed')
        # Two-step retreat: back the open fingers out along the reverse grasp
        # approach so they can't catch the just-placed object, THEN rise clear of
        # the stack. Best-effort — the object is already placed.
        retract = get_approach_pose(place_pose, -RETRACT_DIST)
        ok_clear = self._servo_pose(place_frame, to_place(retract), world_ok, gh,
                                    do_plan=False)
        rise = copy.deepcopy(retract)
        rise.position.z += PRE_PLACE_DIST
        ok_rise = self._servo_pose(place_frame, to_place(rise), world_ok, gh,
                                   do_plan=False)
        if not (ok_clear and ok_rise):
            self.get_logger().warn('pick_place: post-release retreat incomplete')
        return run.succeed(
            f'placed {obj!r} on {place!r} '
            f'(rest height {rest_h * 100:.1f}cm, grasp score {outcome.score:.2f})')

    def _grasp_via_skill(self, obj, arm, run, gh):
        """Send an in-process /skill/grasp goal for `obj` and return
        (GraspOutcome, None) or (None, reason). No orientation bias is imposed —
        the vertical place descent works from any executed grasp. Inner feedback
        is proxied into this skill's 'grasp' phase; a cancel of the outer goal
        cancels the in-flight inner goal (via _send_action's outer_gh
        plumbing)."""
        inner_timeout = max(10.0, run.remaining() - PLACE_RESERVE_SEC)
        inner = SkillGrasp.Goal()
        inner.target_object = obj
        inner.arm = arm
        whole = int(inner_timeout)
        inner.timeout = Duration(
            sec=whole, nanosec=int(round((inner_timeout - whole) * 1e9)))

        def _proxy(feedback_msg):
            run.feedback.phase = 'grasp'
            run.feedback.progress = (
                0.2 + 0.35 * float(feedback_msg.feedback.progress))
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
            # e.g. the battery path served the goal — it never fills an outcome.
            return None, f'grasp of {obj!r} returned no outcome for placement'
        return outcome, None

    def _servo_pose(self, frame, pose, world_ok, gh, do_plan,
                    duration_sec=SERVO_DURATION_SEC):
        """Drive URDF frame `frame` to `pose` — a Pose in WORLD_FRAME when
        `world_ok` (drift-compensated servo; the pelvis fallback is re-resolved
        fresh at call time), else a pelvis-frame Pose driven directly. Same
        relaxed tolerances as the grasp skill's servo."""
        if world_ok:
            world_pose, fallback = pose, self._transform_pose(
                pose, WORLD_FRAME, 'pelvis')
            if fallback is None:
                self.get_logger().warn(
                    f'pick_place: {WORLD_FRAME} -> pelvis TF unavailable')
                return False
        else:
            world_pose, fallback = None, pose
        return self.servo_frame_to_world(
            frame, world_pose, fallback, outer_gh=gh,
            duration_sec=duration_sec, max_iter=SERVO_MAX_ITER,
            lin_tol=SERVO_LIN_TOL, ang_tol=SERVO_ANG_TOL, do_plan=do_plan)

    def _place_frame_offset(self, arm):
        """Fixed rigid transform (4x4) of the arm's pinch-point grasp frame
        (PLACE_FRAMES) expressed in its graspgenx frame — both are fixed URDF
        children of the wrist yaw link, so the offset is constant; read it once
        per goal from TF. A target pose computed FOR the graspgenx frame becomes
        the equivalent grasp-frame target as T_target @ offset. None when TF
        can't resolve it (robot description not up)."""
        try:
            tf = self.tf_buffer.lookup_transform(
                GRASP_FRAMES[arm], PLACE_FRAMES[arm], Time(),
                timeout=RclpyDuration(seconds=1.0))
        except TransformException:
            return None
        return transform_to_matrix(tf.transform)

    def _lin_error_to(self, frame, pose, frame_id):
        """Measured distance [m] between URDF frame `frame` and `pose` (a Pose
        in `frame_id`), or None when TF can't resolve it. Used to tell an
        early-touchdown place stall from a genuinely failed descent."""
        try:
            tf = self.tf_buffer.lookup_transform(
                frame_id, frame, Time(),
                timeout=RclpyDuration(seconds=0.5))
        except TransformException:
            return None
        t = tf.transform.translation
        return float(np.linalg.norm([t.x - pose.position.x,
                                     t.y - pose.position.y,
                                     t.z - pose.position.z]))


def _top_face_point(cloud):
    """(x, y, z_top) of `cloud`'s top face, in the cloud's own (z-up) frame:
    z_top is the 95th-percentile z (robust to depth speckle above the object)
    and x, y average only the points within TOP_FACE_BAND below it — the
    visible top surface — so side-face points don't skew the drop point."""
    z = cloud[:, 2]
    z_top = float(np.percentile(z, 95))
    top = cloud[z > z_top - TOP_FACE_BAND]
    if len(top) == 0:                 # degenerate band -> whole-cloud centroid
        top = cloud
    return np.array([float(np.mean(top[:, 0])), float(np.mean(top[:, 1])), z_top])
