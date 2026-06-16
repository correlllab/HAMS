"""Shared lever/knob executor for SkillTurnLever and SkillTwistKnob.

node.py binds the concrete action type / label / motion-phase per skill with
partial(), so this module needs no Skill* action-type imports of its own."""

from ..base import _Run, _roll_quat, DEFAULT_TURN_ANGLE


class TurnSkill:
    def _exec_turn(self, action_type, label, motion_phase, gh):
        """Shared lever/knob executor: detect -> grasp (composed via SkillGrasp)
        -> rotate the wrist about the approach (+x) axis by the requested
        signed angle, relative to the antipodal grasp roll."""
        goal = gh.request
        run = _Run(self, gh, action_type, label)
        arm = self._validated_arm(run, goal)
        if arm is None:
            return run.abort(f'invalid arm {goal.arm!r}')
        direction = goal.direction.strip().lower()
        if direction not in ('cw', 'ccw'):
            return run.abort(f'invalid direction {goal.direction!r} (use "cw"|"ccw")')
        angle = goal.angle if goal.angle > 0.0 else DEFAULT_TURN_ANGLE
        # cw/ccw are as seen by the robot FACING the mechanism: the rotation
        # axis (+x, the approach direction) points away from the viewer, so a
        # positive right-hand roll about +x (left edge up: +y -> +z) appears
        # CLOCKWISE to the viewer.
        signed = angle if direction == 'cw' else -angle

        if not run.phase('detect', 0.0):
            return run.result
        plan = self.detect_grasp(goal.target_object)
        if plan is None:
            return run.abort(f'no {goal.target_object!r} detected')
        (hx, hy, hz), roll, _width = plan

        if not run.phase('grasp', 0.4):
            return run.result
        ok, why = self._call_grasp_skill(gh, run, goal.target_object, arm)
        if not ok:
            return run.abort(f'grasp failed: {why}')

        if not run.phase(motion_phase, 0.75):
            return run.result
        # Twist RELATIVE to the antipodal grasp roll the wrist already holds —
        # an absolute target of `signed` alone would first untwist the grasp.
        if not self.move_frame_to(arm, hx, hy, hz, duration_sec=3,
                                  quat=_roll_quat(roll + signed), outer_gh=gh):
            return run.abort(f'{motion_phase} motion failed')
        self.open_gripper(arm)   # release

        return run.succeed(
            f'{motion_phase}ed {goal.target_object!r} {direction} by {angle:.2f} rad')
