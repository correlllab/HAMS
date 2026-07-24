"""SkillPickPlace: prep -> detect the place target -> grasp the pick object
(in-process /skill/grasp) -> park the graspgenx frame top-down above the place
target -> open the gripper.

The place is deliberately simple: no carry ladder, orientation fan, or
rest-height descent. The place pose reuses the grasp skill's top-down
orientation (_top_down_pose — forward azimuth, level wrist, pitched
TOP_DOWN_PITCH_DEG down) with the finger CONTACT point PLACE_HEIGHT above the
place target's top face, and the object is dropped from there.

The place point is detected BEFORE the (long) grasp and anchored in
WORLD_FRAME so pelvis drift during the grasp does not move the goal; at place
time it is re-resolved into the live pelvis frame and the commanded pose is
driven with servo_frame_to_world's drift compensation. Without a world TF the
skill falls back to the detect-time pelvis point, uncompensated — the same
degradation as the grasp skill without odometry.
"""

import copy

import numpy as np

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration

from custom_ros_messages.action import SkillGrasp, SkillPickPlace

from ..base import _Run, GRASP_FRAMES, WORLD_FRAME
from .grasp import (_approach_target_tf, _pose_at, _top_down_pose,
                    SERVO_MAX_ITER, SERVO_LIN_TOL, SERVO_ANG_TOL,
                    APPROACH_DURATION_SEC)


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
# Straight-up retract along pelvis +Z right after the grasp, to lift the object
# clear of the surface before the carry. Planned (collision-aware) and run in
# SLOW mode so it doesn't jerk the just-grasped object; the duration is the
# slow-mode-scaled budget (~4x, so a slow move still reaches over RETRACT_HEIGHT).
RETRACT_HEIGHT = 0.15
RETRACT_SEC = 20.0
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

        # --- retract: lift the grasped object STRAIGHT UP along pelvis +Z,
        # planned, to clear it off the surface before the carry ---------------
        frame = GRASP_FRAMES[arm]
        cur = self._frame_pose_in_pelvis(frame)
        if cur is None:
            return fail('TF for the post-grasp retract unavailable')
        up = copy.deepcopy(cur)
        up.position.z += RETRACT_HEIGHT
        self.get_logger().info(
            f'pick_place: retracting {frame} {RETRACT_HEIGHT * 100:.0f}cm up '
            '(pelvis +Z), planned, after the grasp')
        if not self.move_frame_to(frame, up, outer_gh=gh,
                                  duration_sec=RETRACT_SEC, do_plan=True,
                                  slow_mode=True,
                                  label='retract up after grasp'):
            return fail('post-grasp retract (pelvis +Z) failed')

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
