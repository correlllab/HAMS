"""SkillOpenLid: detect -> grasp (composed) -> open along a hinge arc."""

import math

from custom_ros_messages.action import SkillOpenLid

from ..base import _Run, _roll_quat, DEFAULT_OPEN_ANGLE, LID_HINGE_RADIUS


class OpenLidSkill:
    def _exec_open_lid(self, gh):
        """detect -> grasp (composed via SkillGrasp) -> open along an arc about
        the hinge. Hinge is assumed at the far edge (+x), LID_HINGE_RADIUS
        behind the grasped point, axis horizontal: the grasp point rises and
        moves toward the hinge as the lid rotates up."""
        goal = gh.request
        run = _Run(self, gh, SkillOpenLid, 'open_lid')
        arm = self._validated_arm(run, goal)
        if arm is None:
            return run.abort(f'invalid arm {goal.arm!r}')
        angle = goal.open_angle if goal.open_angle > 0.0 else DEFAULT_OPEN_ANGLE

        if not run.phase('detect', 0.0):
            return run.result
        plan = self.detect_grasp(goal.target_object)
        if plan is None:
            return run.abort(f'no {goal.target_object!r} detected')
        (hx, hy, hz), roll, _width = plan
        grasp_quat = _roll_quat(roll)

        if not run.phase('grasp', 0.4):
            return run.result
        ok, why = self._call_grasp_skill(gh, run, goal.target_object, arm)
        if not ok:
            return run.abort(f'grasp failed: {why}')

        if not run.phase('open', 0.75):
            return run.result
        r = LID_HINGE_RADIUS
        for a in (0.5 * angle, angle):    # two waypoints along the arc
            wx = hx + r * (1.0 - math.cos(a))
            wz = hz + r * math.sin(a)
            if not self.move_frame_to(arm, wx, hy, wz, duration_sec=3,
                                      quat=grasp_quat, outer_gh=gh):
                return run.abort('lid arc motion failed')
        self.open_gripper(arm)            # release the open lid

        return run.succeed(f'opened {goal.target_object!r} to {angle:.2f} rad')
