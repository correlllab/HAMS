"""SkillPickPlace: block stacking — detect the place target, grasp the pick
object via the node's own /skill/grasp action, then carry, place, and release.

Flow (one _Run budget across every phase):
  detect  — gemini box -> sam mask -> place-target cloud; its top-face point is
            anchored in WORLD_FRAME so pelvis drift during the (long) grasp
            does not move the goal.
  grasp   — in-process /skill/grasp goal; the executed grasp pose, the arm that
            ran it and held-object geometry come back through
            _last_grasp_outcome (see skills/grasp.py — the grasp RESULT message
            is unchanged). goal.arm may be "" / "none": the grasp skill then
            picks the arm nearest the object and everything after the grasp
            follows outcome.arm.
  carry   — lift straight up in LIFT_STEP increments, then transit to a
            standoff directly ABOVE the release pose, walking a short list of
            candidate place ORIENTATIONS (executed grasp orientation first,
            then the tier-style heuristic fan — see below) until one standoff
            is IK-reachable, the same fallback shape as the grasp skill's
            tier-major candidate walk.
  place   — descend vertically until the held object's centroid sits
            rest_height above the place target's top face. The descent is
            grasp-agnostic — the OBJECT always arrives from above, whatever the
            grasp orientation — so no downward-pitch bias is imposed on the
            grasp anymore (the old approach-axis drive-in needed pitched-down
            grasps and starved the candidate pool); goal.top_down still forwards
            an explicit above-the-object pick when the caller asks for one. A
            stall NEAR the release pose still releases (early touchdown); a
            distant stall aborts.
  release — open the gripper, back the fingers out along the reverse of the
            committed place orientation's approach so they can't catch the
            placed object, then rise clear.

Placement math: the gripper->object offset o_g = R^T (c - p), measured at grasp
time from the SAME pelvis snapshot as the executed grasp, is constant while the
object is held — it lives in the GRIPPER frame. Only position is therefore
solved, per candidate orientation R': frame target = desired_centroid - R' @ o_g,
which holds for ANY R', so the object does not have to land in its pick-time
orientation. The executed grasp orientation is still tried FIRST (the object
stays level and lands as picked, making the rest-height math exact); the
heuristic fallback orientations (_heuristic_place_orientations — the grasp
skill's forward/diagonal/center azimuth tiers at a level wrist, pitched into
their preferred 20-45 deg band) reorient the held object, leaving rest_height
approximate, which the slow, stall-tolerant descent absorbs. The object's rest
height (centroid above its own bottom, measured while it sat on the table) is
exactly how high its centroid must sit above the place target's top face when
it comes to rest there in its pick-time orientation.

Commanded frame: every post-grasp motion drives the arm's pinch-point
{left,right}_grasp_frame through /frame_task (PLACE_FRAMES — the frame the
frame_task planner's frame_names list and z-floor safety track), NOT the
graspgenx gripper-base frame the math above is written in. Targets are computed
in the GraspGenX convention, then re-expressed through the fixed URDF
graspgenx->grasp-frame offset (_place_frame_offset) before being sent.
"""

import copy
import math
from dataclasses import replace

import numpy as np

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time
from tf2_ros import TransformException

from custom_ros_messages.action import SkillGrasp, SkillPickPlace

from ..base import _Run, GRASP_FRAMES, SLOW_MODE_TIME_SCALE, WORLD_FRAME
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

LIFT_HEIGHT = 0.10       # [m] total straight-up lift after the grasp closes
# [m] per-waypoint increment of that lift: the rise is commanded as a ladder of
# LIFT_STEP moves instead of one LIFT_HEIGHT move, so the arm clears the pick
# site in short committed hops that stay interruptible. Each waypoint is an
# ABSOLUTE offset from the executed grasp pose, so servo error at one rung
# cannot accumulate into the next.
LIFT_STEP = 0.05
# [s] commanded motion time per lift increment (a 5 cm hop does not need the
# full SERVO_DURATION_SEC the 12 cm one-shot lift used).
LIFT_STEP_DURATION_SEC = 4.0
PRE_PLACE_DIST = 0.10    # [m] pre-place standoff straight above the release pose
# Synthetic place-orientation fan — the fallback candidates tried behind the
# executed grasp orientation when its pre-place standoff is IK-unreachable.
# Approach azimuths in "toward center" degrees, mirrored per hand exactly like
# the grasp skill's _grasp_priority_tier classes (forward, diagonal, deeper
# center fan), each pitched PLACE_HEURISTIC_PITCH_DEG below horizontal (inside
# the grasp tiers' preferred TIER_PITCH_MIN..MAX band) with a level wrist.
# Ordered like the tiers: forward first.
PLACE_HEURISTIC_AZ_DEG = (0.0, 45.0, 70.0, 90.0)
PLACE_HEURISTIC_PITCH_DEG = 30.0
# Heuristic candidates closer (geodesic) than this to the executed grasp
# orientation are dropped: they would fail IK the same way the grasp
# orientation just did and waste a full planned servo attempt.
PLACE_ORIENT_DUP_DEG = 10.0
# [m] Height of the GRASPGENX FRAME ITSELF above the place target's top face for
# a top_down place (goal.top_down mirrors the pick: released from above rather
# than lowered onto the stack). Purely geometric — it deliberately ignores
# rest_height and the gripper->object offset the normal place solves for, so the
# object is DROPPED the remaining distance rather than set down.
#
# NB this is the frame height, not the object's: the fingers sit
# GRIPPER_BASE_TO_CONTACT_M (0.1146 m) down the approach from the graspgenx
# origin, which at the 80 deg top-down approach is ~0.113 m of that height. So
# at 0.25 the held object is released ~0.137 m above the target.
TOP_DOWN_PLACE_HEIGHT = 0.25
# [m] release the object this far above its rest pose. Keep it >= the place
# servo's convergence tolerance (SERVO_LIN_TOL, 25 mm): a tolerance larger than
# the clearance can eat it and press the object into the stack before release.
PLACE_CLEARANCE = 0.03
RETRACT_DIST = 0.15      # [m] post-release finger back-off along the reverse place approach
# Descent stalls within this distance of the release pose still release — early
# touchdown on the stack (rest_height is a single-view estimate); stalls farther
# away abort with the object still held rather than dropping it off-target.
PLACE_STALL_MAX_M = 0.05
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
        place -> release.

        Optional goal fields (see SkillPickPlace.action; zero/empty = off):
          arm — "" or "none" leaves the pick arm to the grasp skill, which takes
            the one nearest the object; the resolved arm comes back on the
            outcome and drives carry/place/release.
          top_down — forwarded to /skill/grasp: pick the object from above
            instead of from the graspgen candidate pool."""
        goal = gh.request
        run = _Run(self, gh, SkillPickPlace, 'pick_place')
        arm = self._validated_arm(goal)
        if arm is None and goal.arm.strip().lower() not in ('', 'none'):
            return run.abort(f'invalid arm {goal.arm!r}')
        obj, place = goal.target_object, goal.place_target
        if not place.strip():
            return run.abort('empty place_target')

        def fail(message):
            """Abort, lifting the hand clear first. Anything past the goal
            validation above can fail with the arm mid-carry or holding the
            object over the stack, so every one of those paths retreats before
            reporting. `arm` is read at call time, which matters: it is None
            until the inner grasp reports which hand it actually used."""
            self._retreat_after_failure(arm, gh)
            return run.abort(message)

        # --- detect: place-target top face, world-anchored ---------------------
        if not run.phase('detect_place_target', 0.0):
            return run.result
        place_cloud, _, err = self.detect_object_cloud(place, run, gh)
        if err:
            return fail(f'place target: {err}')
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
        if not run.phase('grasp_object', 0.2):
            return run.result
        outcome, err = self._grasp_via_skill(goal, obj, arm, run, gh)
        if err:
            return fail(err)
        # The grasp skill resolves an auto ("" / "none") arm against the object's
        # own centroid, which pick_place hasn't detected — so take the arm it
        # actually used. Every arm-indexed step below (place frame, release)
        # must address that same hand.
        arm = outcome.arm

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
                return fail(
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
        # Candidate place ORIENTATIONS, best-first. o_g is fixed in the GRIPPER
        # frame while the object is held, so the position solve below
        # (target = c_release - R' @ o_g) holds for ANY orientation R' — the
        # object does not have to land in its pick-time orientation. The
        # executed grasp orientation still goes first (object stays level,
        # rest-height math exact); behind it comes the tier-style heuristic fan
        # (PLACE_HEURISTIC_AZ_DEG), so an IK-unreachable pre-place falls back
        # through reorientations instead of failing the skill. A reoriented
        # candidate tilts/yaws the held object and turns rest_height into an
        # approximation — the slow descent and the early-touchdown stall
        # release below absorb that error.
        cand_orients = [('grasp orientation', R)]
        for lbl, R_p in _heuristic_place_orientations(arm):
            R_c = self._rot_to_working_frame(R_p, world_ok)
            if R_c is None or _rot_angle_deg(R, R_c) < PLACE_ORIENT_DUP_DEG:
                continue
            cand_orients.append((lbl, R_c))
        c_release = top + [0.0, 0.0, rest_h + PLACE_CLEARANCE]
        if goal.top_down:
            self.get_logger().info(
                f'pick_place: top-down place — {GRASP_FRAMES[arm]} to '
                f'{TOP_DOWN_PLACE_HEIGHT * 100:.0f}cm above {place!r}, then release')

        def _place_pose_for(R_c):
            """Release Pose (working frame) putting the held object at its rest
            spot with the gripper at orientation R_c. A top_down goal keeps its
            purely geometric fixed-height release — park the GRASPGENX frame
            TOP_DOWN_PLACE_HEIGHT above the place point and open there, no
            rest_height / gripper->object math (the object is dropped from
            above, not lowered onto the surface)."""
            T_place = np.eye(4)
            T_place[:3, :3] = R_c
            T_place[:3, 3] = (top + [0.0, 0.0, TOP_DOWN_PLACE_HEIGHT]
                              if goal.top_down else c_release - R_c @ o_g)
            return matrix_to_pose(T_place)

        # From here on, motions COMMAND the pinch-point grasp frame, not the
        # graspgenx frame the targets above are computed for: fetch the fixed
        # URDF offset between the two once and re-express every target with it.
        place_frame = PLACE_FRAMES[arm]
        T_off = self._place_frame_offset(arm)
        if T_off is None:
            return fail(
                f'TF {GRASP_FRAMES[arm]} -> {place_frame} unavailable')

        def to_place(pose_gx):
            return matrix_to_pose(pose_to_matrix(pose_gx) @ T_off)

        # --- carry: lift, then transit to the pre-place standoff ----------------
        if not run.phase('carry', 0.55):
            return run.result
        # Rise to LIFT_HEIGHT as a ladder of LIFT_STEP hops. Every rung is an
        # absolute offset from the executed grasp pose (no error accumulation),
        # and the budget check between rungs keeps the long lift interruptible.
        n_rungs = max(1, int(round(LIFT_HEIGHT / LIFT_STEP)))
        for k in range(1, n_rungs + 1):
            dz = min(k * LIFT_STEP, LIFT_HEIGHT)
            lift = copy.deepcopy(outcome.pose)
            lift.position.z += dz
            if not self._servo_pose(place_frame, to_place(lift), world_ok, gh,
                                    do_plan=False,
                                    duration_sec=LIFT_STEP_DURATION_SEC):
                return fail(
                    f'lift after grasp failed at {dz * 100:.0f}cm of '
                    f'{LIFT_HEIGHT * 100:.0f}cm')
            if gh.is_cancel_requested or run.remaining() <= 0.0:
                return fail('canceled or timed out during carry')
        self.get_logger().info(
            f'pick_place: lifted {LIFT_HEIGHT * 100:.0f}cm in {n_rungs} x '
            f'{LIFT_STEP * 100:.0f}cm increments')
        # Walk the candidate orientations at the pre-place standoff (straight
        # ABOVE each candidate's release pose — the descent is vertical, so
        # every candidate delivers the object from above) until one is
        # reachable, mirroring the grasp skill's tier-major IK fallback walk.
        place_pose = orient_lbl = None
        for lbl, R_c in cand_orients:
            cand_pose = _place_pose_for(R_c)
            pre_place = copy.deepcopy(cand_pose)
            pre_place.position.z += PRE_PLACE_DIST
            self.publish_tf(_approach_target_tf(
                WORLD_FRAME if world_ok else 'pelvis', cand_pose,
                self.get_clock().now().to_msg(), child=PLACE_TARGET_FRAME))
            if self._servo_pose(place_frame, to_place(pre_place), world_ok, gh,
                                do_plan=True):
                place_pose, orient_lbl = cand_pose, lbl
                if lbl != cand_orients[0][0]:
                    self.get_logger().info(
                        f'pick_place: placing with the {lbl} instead of the '
                        'executed grasp orientation')
                break
            if gh.is_cancel_requested or run.remaining() <= 0.0:
                return fail('canceled or timed out during carry')
            self.get_logger().warn(
                f'pick_place: pre-place unreachable with the {lbl}; trying '
                'the next candidate orientation')
        if place_pose is None:
            return fail('pre-place pose unreachable with every candidate '
                        f'orientation ({len(cand_orients)} tried)')

        # --- place: descend vertically onto the stack ----------------------------
        if not run.phase('place', 0.75):
            return run.result
        place_cmd = to_place(place_pose)
        # Slow: this descent lowers a held object onto the stack, and the stall
        # detection below only distinguishes early touchdown from a failed
        # descent AFTER the fact — at full speed a misjudged rest_height presses
        # the object into the stack before that check ever runs.
        if not self._servo_pose(place_frame, place_cmd, world_ok, gh,
                                do_plan=False, slow_mode=True):
            if gh.is_cancel_requested or run.remaining() <= 0.0:
                return fail('canceled or timed out during place')
            # A stall NEAR the release pose is early touchdown (rest_height is a
            # single-view estimate) — release there rather than aborting while
            # pressing the object into the stack. A distant stall means the
            # descent genuinely failed; abort with the object still held.
            lin_err = self._lin_error_to(place_frame, place_cmd,
                                         WORLD_FRAME if world_ok else 'pelvis')
            if lin_err is None or lin_err > PLACE_STALL_MAX_M:
                return fail(
                    'place descent stalled far from the release pose'
                    + ('' if lin_err is None else f' ({lin_err * 100:.1f}cm short)'))
            self.get_logger().warn(
                f'pick_place: descent stalled {lin_err * 100:.1f}cm from the '
                'release pose (early touchdown?); releasing here')

        # --- release + retreat ----------------------------------------------------
        if not run.phase('release', 0.9):
            return run.result
        if not self.open_gripper(arm):
            return fail('gripper open failed')
        # Two-step retreat: back the open fingers out along the reverse of the
        # COMMITTED place orientation's approach axis so they can't catch the
        # just-placed object, THEN rise clear of the stack. Best-effort — the
        # object is already placed.
        # Both retreat legs run slow: the open fingers are still inside the
        # just-placed object's footprint, so a fast withdrawal is what knocks it
        # off the stack.
        retract = get_approach_pose(place_pose, -RETRACT_DIST)
        ok_clear = self._servo_pose(place_frame, to_place(retract), world_ok, gh,
                                    do_plan=False, slow_mode=True)
        rise = copy.deepcopy(retract)
        rise.position.z += PRE_PLACE_DIST
        ok_rise = self._servo_pose(place_frame, to_place(rise), world_ok, gh,
                                   do_plan=False, slow_mode=True)
        if not (ok_clear and ok_rise):
            self.get_logger().warn('pick_place: post-release retreat incomplete')
        return run.succeed(
            f'placed {obj!r} on {place!r} ('
            + (f'top-down release {TOP_DOWN_PLACE_HEIGHT * 100:.0f}cm up'
               if goal.top_down else f'rest height {rest_h * 100:.1f}cm')
            + f', {orient_lbl}, grasp score {outcome.score:.2f})')

    def _grasp_via_skill(self, goal, obj, arm, run, gh):
        """Send an in-process /skill/grasp goal for `obj` and return
        (GraspOutcome, None) or (None, reason). No orientation bias is imposed of
        our own — the vertical place descent works from any executed grasp — but
        the caller's optional top_down and visual_servo are forwarded, so a
        pick_place grasp behaves exactly like the same direct /skill/grasp call.
        `arm` is None for auto: the inner goal
        carries "" and the grasp skill picks, reporting back in outcome.arm.
        Inner feedback is proxied into this skill's 'grasp_object' phase; a
        cancel of
        the outer goal cancels the in-flight inner goal (via _send_action's
        outer_gh plumbing)."""
        inner_timeout = max(10.0, run.remaining() - PLACE_RESERVE_SEC)
        inner = SkillGrasp.Goal()
        inner.target_object = obj
        inner.arm = arm or ''
        # Forward BOTH grasp modifiers. Every optional field the grasp skill
        # understands has to be relayed explicitly — an unset field silently
        # defaults to false, which is how visual_servo used to be dropped here
        # and made a pick_place grasp behave differently from the identical
        # direct /skill/grasp call.
        inner.top_down = goal.top_down
        inner.visual_servo = goal.visual_servo
        if inner_timeout == float('inf'):
            # This goal carries no deadline (timeout 0 = run to completion, see
            # DEFAULT_SKILL_TIMEOUT), so remaining() is infinite and can't be put
            # in an integer Duration. Hand the inner grasp a zero Duration, which
            # means exactly the same thing — the semantic propagates unchanged.
            inner.timeout = Duration(sec=0, nanosec=0)
        else:
            whole = int(inner_timeout)
            inner.timeout = Duration(
                sec=whole, nanosec=int(round((inner_timeout - whole) * 1e9)))

        def _proxy(feedback_msg):
            run.feedback.phase = 'grasp_object'
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
                    duration_sec=SERVO_DURATION_SEC, slow_mode=False):
        """Drive URDF frame `frame` to `pose` — a Pose in WORLD_FRAME when
        `world_ok` (drift-compensated servo; the pelvis fallback is re-resolved
        fresh at call time), else a pelvis-frame Pose driven directly. Same
        relaxed tolerances as the grasp skill's servo.

        `slow_mode` runs the move at the frame_task server's quarter speed AND
        scales `duration_sec` by SLOW_MODE_TIME_SCALE to match, since the
        duration is a timeout rather than a trajectory time."""
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
            duration_sec=duration_sec * (SLOW_MODE_TIME_SCALE if slow_mode else 1.0),
            max_iter=SERVO_MAX_ITER,
            lin_tol=SERVO_LIN_TOL, ang_tol=SERVO_ANG_TOL, do_plan=do_plan,
            slow_mode=slow_mode)

    def _rot_to_working_frame(self, R_p, world_ok):
        """Rotation R_p (3x3, LIVE pelvis axes) re-expressed in the working
        frame the placement math runs in: world when `world_ok`, else the
        pelvis frame unchanged. The heuristic place orientations are defined
        against the pelvis (forward / toward-center azimuths), so in world
        mode they must be rotated once through the current pelvis->world
        attitude — the robot stands still through carry/place, so a single
        conversion at candidate-build time holds. None when the TF is
        unavailable (that candidate is skipped)."""
        if not world_ok:
            return R_p
        T = np.eye(4)
        T[:3, :3] = R_p
        pose_w = self._transform_pose(matrix_to_pose(T), 'pelvis', WORLD_FRAME)
        if pose_w is None:
            return None
        return pose_to_matrix(pose_w)[:3, :3]

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


def _heuristic_place_orientations(arm):
    """[(label, R)] of the synthetic fallback place orientations for `arm`, in
    LIVE pelvis axes (z-up, +X forward, +Y left), best-first: the grasp skill's
    tier-style azimuth fan (PLACE_HEURISTIC_AZ_DEG, "toward center" degrees
    mirrored per hand exactly like _grasp_priority_tier), each pitched
    PLACE_HEURISTIC_PITCH_DEG below horizontal with a level wrist (+Y as close
    to up as the pitch allows — the same wrist _top_down_pose builds). These
    are GRASPGENX-convention orientations (+Z approach, +X finger closing),
    interchangeable with the executed grasp orientation they stand in for."""
    pitch = math.radians(PLACE_HEURISTIC_PITCH_DEG)
    out = []
    for az_center in PLACE_HEURISTIC_AZ_DEG:
        azim = math.radians(az_center if arm == 'right' else -az_center)
        # +Z (approach): pitched down, swung `azim` from pelvis +X toward the
        # body's center line.
        z_axis = np.array([math.cos(pitch) * math.cos(azim),
                           math.cos(pitch) * math.sin(azim),
                           -math.sin(pitch)])
        # +X (fingers): horizontal and perpendicular to the approach, signed so
        # +Y = Z x X comes out with a positive up-component (level wrist). At
        # zero azimuth this is exactly _top_down_pose's pelvis +Y choice.
        x_axis = np.cross([0.0, 0.0, 1.0], z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        out.append((f'heuristic orientation (az {az_center:.0f}deg)',
                    np.column_stack((x_axis, y_axis, z_axis))))
    return out


def _rot_angle_deg(Ra, Rb):
    """Geodesic angle [deg] between rotations Ra and Rb (3x3):
    arccos((trace(Ra^T Rb) - 1) / 2). Used to drop heuristic place candidates
    that near-duplicate the executed grasp orientation."""
    cos = (float(np.trace(Ra.T @ Rb)) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))


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
