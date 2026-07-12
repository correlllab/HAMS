"""SkillPickPlace: block stacking — detect the place target, grasp the pick
object via the node's own /skill/grasp action, then carry, place, and release.

Flow (one _Run budget across every phase):
  detect  — gemini box -> sam mask -> place-target cloud; its top-face point is
            anchored in WORLD_FRAME so pelvis drift during the (long) grasp
            does not move the goal.
  grasp   — in-process /skill/grasp goal, biased to downward-pitched approaches
            via the _grasp_overrides side channel; the executed grasp pose and
            held-object geometry come back through _last_grasp_outcome (see
            skills/grasp.py — the action messages are unchanged).
  carry   — lift straight up, then transit to a pre-place standoff backed off
            along the executed grasp's own approach axis.
  place   — drive in along that approach axis until the held object's centroid
            sits rest_height above the place target's top face.
  release — open the gripper, then retract back out along the reverse approach.

Placement math: the gripper->object offset o_g = R^T (c - p), measured at grasp
time from the SAME pelvis snapshot as the executed grasp, is constant while the
object is held. Keeping the executed grasp's world orientation R through
carry/place keeps the object level (it lands in its pick-time orientation), so
only position is commanded: frame target = desired_centroid - R @ o_g. The
object's rest height (centroid above its own bottom, measured while it sat on
the table) is exactly how high its centroid must sit above the place target's
top face when it comes to rest there.
"""

import copy

import numpy as np

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration

from custom_ros_messages.action import SkillGrasp, SkillPickPlace

from ..base import _Run, GRASP_FRAMES, WORLD_FRAME
from ..perception_utils import pose_to_matrix, matrix_to_pose
from .grasp import (get_approach_pose, _approach_target_tf, _pose_at,
                    SERVO_DURATION_SEC, SERVO_MAX_ITER, SERVO_LIN_TOL,
                    SERVO_ANG_TOL)


# Grasp-candidate constraint passed to /skill/grasp through the in-process
# override side channel: keep only approaches pitched at least this far below
# horizontal. The H1 arm cannot reach straight down (~45 deg below horizontal is
# its practical steepest), so 30 deg admits those best oblique grasps while
# rejecting near-horizontal ones that would ram the gripper into the stack when
# placing.
MIN_DOWNWARD_PITCH_DEG = 30.0

LIFT_HEIGHT = 0.12       # [m] straight-up lift after the grasp closes
PRE_PLACE_DIST = 0.10    # [m] pre-place standoff along the grasp approach axis
PLACE_CLEARANCE = 0.01   # [m] release the object this far above its rest pose
RETRACT_DIST = 0.15      # [m] post-release back-off along the reverse approach
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

        # --- grasp: in-process /skill/grasp, biased downward --------------------
        if not run.phase('grasp', 0.2):
            return run.result
        outcome, err = self._grasp_via_skill(obj, arm, run, gh)
        if err:
            return run.abort(err)

        # --- placement geometry -------------------------------------------------
        # Work in one consistent frame: world only when BOTH the place point and
        # the grasp outcome are world-anchored, else fall back to pelvis (drift
        # uncompensated, same degradation as the grasp skill without odometry).
        world_ok = outcome.frame == WORLD_FRAME and top_w is not None
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
        pre_place = get_approach_pose(place_pose, -PRE_PLACE_DIST)
        self.publish_tf(_approach_target_tf(
            WORLD_FRAME if world_ok else 'pelvis', place_pose,
            self.get_clock().now().to_msg(), child=PLACE_TARGET_FRAME))

        # --- carry: lift, then transit to the pre-place standoff ----------------
        if not run.phase('carry', 0.55):
            return run.result
        lift = copy.deepcopy(outcome.pose)
        lift.position.z += LIFT_HEIGHT
        if not self._servo_pose(arm, lift, world_ok, gh, do_plan=False):
            return run.abort('lift after grasp failed')
        if gh.is_cancel_requested or run.remaining() <= 0.0:
            return run.abort('canceled or timed out during carry')
        if not self._servo_pose(arm, pre_place, world_ok, gh, do_plan=True):
            return run.abort('pre-place pose unreachable')

        # --- place: drive in along the grasp approach ---------------------------
        if not run.phase('place', 0.75):
            return run.result
        if not self._servo_pose(arm, place_pose, world_ok, gh, do_plan=False):
            return run.abort('place pose unreachable')

        # --- release + retract ---------------------------------------------------
        if not run.phase('release', 0.9):
            return run.result
        if not self.open_gripper(arm):
            return run.abort('gripper open failed')
        # Back out the way we came in (reverse approach) so the open fingers
        # can't catch the just-placed object or the stack.
        retract = get_approach_pose(place_pose, -RETRACT_DIST)
        if not self._servo_pose(arm, retract, world_ok, gh, do_plan=False):
            self.get_logger().warn('pick_place: post-release retract incomplete')
        return run.succeed(
            f'placed {obj!r} on {place!r} '
            f'(rest height {rest_h * 100:.1f}cm, grasp score {outcome.score:.2f})')

    def _grasp_via_skill(self, obj, arm, run, gh):
        """Send an in-process /skill/grasp goal for `obj`, biased to
        downward-pitched approaches via the _grasp_overrides side channel, and
        return (GraspOutcome, None) or (None, reason). Inner feedback is proxied
        into this skill's 'grasp' phase; a cancel of the outer goal cancels the
        in-flight inner goal (via _send_action's outer_gh plumbing)."""
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

        self._grasp_overrides = {'min_downward_pitch_deg': MIN_DOWNWARD_PITCH_DEG}
        resp = self._send_action(self.grasp_skill_cli, inner, feedback_cb=_proxy,
                                 result_timeout=inner_timeout + 15.0, outer_gh=gh)
        # If the goal was rejected / never sent, the server never consumed the
        # override — clear it so it can't leak into a later standalone grasp.
        self._grasp_overrides = None
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

    def _servo_pose(self, arm, pose, world_ok, gh, do_plan,
                    duration_sec=SERVO_DURATION_SEC):
        """Drive the arm's GraspGenX frame to `pose` — a Pose in WORLD_FRAME when
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
            GRASP_FRAMES[arm], world_pose, fallback, outer_gh=gh,
            duration_sec=duration_sec, max_iter=SERVO_MAX_ITER,
            lin_tol=SERVO_LIN_TOL, ang_tol=SERVO_ANG_TOL, do_plan=do_plan)


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
