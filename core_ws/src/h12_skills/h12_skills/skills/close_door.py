"""SkillCloseDoor: detect -> approach -> push."""

from custom_ros_messages.action import SkillCloseDoor

from ..base import _Run, APPROACH_BACKOFF, PUSH_DEPTH


class CloseDoorSkill:
    def _exec_close_door(self, gh):
        """detect -> approach -> push (closed-hand push past the handle plane)."""
        goal = gh.request
        run = _Run(self, gh, SkillCloseDoor, 'close_door')
        arm = self._validated_arm(run, goal)
        if arm is None:
            return run.abort(f'invalid arm {goal.arm!r}')

        if not run.phase('detect', 0.0):
            return run.result
        c = self.detect_object(goal.target_object)
        if c is None:
            return run.abort(f'no {goal.target_object!r} detected')
        hx, hy, hz = c

        if not run.phase('approach', 0.3):
            return run.result
        if not self.close_gripper(arm):   # push with a closed hand
            return run.abort('gripper close failed')
        approach_x = hx - APPROACH_BACKOFF
        if not self.move_frame_to(arm, approach_x, hy, hz, duration_sec=4, outer_gh=gh):
            return run.abort('approach motion failed')

        if not run.phase('push', 0.6):
            return run.result
        if not self.move_frame_to(arm, hx + PUSH_DEPTH, hy, hz,
                                  duration_sec=3, outer_gh=gh):
            return run.abort('push motion failed')
        self.move_frame_to(arm, approach_x, hy, hz, duration_sec=2, outer_gh=gh)  # retreat

        return run.succeed(f'closed {goal.target_object!r}')
