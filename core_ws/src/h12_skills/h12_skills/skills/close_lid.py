"""SkillCloseLid: detect -> approach above -> push the hinged lid shut."""

import math

from custom_ros_messages.action import SkillCloseLid

from ..base import _Run, APPROACH_BACKOFF, LID_HINGE_RADIUS


class CloseLidSkill:
    def _exec_close_lid(self, gh):
        """detect -> approach above -> push the hinged lid down shut."""
        goal = gh.request
        run = _Run(self, gh, SkillCloseLid, 'close_lid')
        arm = self._validated_arm(run, goal)
        if arm is None:
            return run.abort(f'invalid arm {goal.arm!r}')

        if not run.phase('detect', 0.0):
            return run.result
        c = self.detect_object(goal.target_object)
        if c is None:
            return run.abort(f'no {goal.target_object!r} detected')
        hx, hy, hz = c

        # Assume the lid is fully open (90 deg) per open_lid's hinge model: the
        # detected handle then sits directly ABOVE the hinge, so the hinge is at
        # (hx, hz - r) and the closed handle position is (hx - r, hz - r). Push
        # the handle back down along the reverse arc — a straight push down at
        # the detected x would have zero closing torque arm.
        r = LID_HINGE_RADIUS
        closed_x, closed_z = hx - r, hz - r

        if not run.phase('approach', 0.3):
            return run.result
        if not self.close_gripper(arm):   # push with a closed hand
            return run.abort('gripper close failed')
        if not self.move_frame_to(arm, hx, hy, hz + 0.1,
                                  duration_sec=4, outer_gh=gh):
            return run.abort('approach motion failed')

        if not run.phase('close', 0.6):
            return run.result
        for a in (math.pi / 4, 0.0):      # reverse arc waypoints (45 deg, closed)
            wx = closed_x + r * (1.0 - math.cos(a))
            wz = closed_z + r * math.sin(a)
            if a == 0.0:
                wz -= 0.05                # overtravel past closed to seat the lid
            if not self.move_frame_to(arm, wx, hy, wz, duration_sec=3, outer_gh=gh):
                return run.abort('close-arc motion failed')
        self.move_frame_to(arm, closed_x - APPROACH_BACKOFF, hy,
                           closed_z + 0.15, duration_sec=2, outer_gh=gh)  # retreat

        return run.succeed(f'closed {goal.target_object!r}')
