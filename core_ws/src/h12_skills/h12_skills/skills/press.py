"""SkillPress: detect -> approach -> press past the button plane."""

from custom_ros_messages.action import SkillPress

from ..base import _Run, APPROACH_BACKOFF, PRESS_DEPTH


class PressSkill:
    def _exec_press(self, gh):
        """detect -> approach -> press past the button plane with a closed hand."""
        goal = gh.request
        run = _Run(self, gh, SkillPress, 'press')
        arm = self._validated_arm(run, goal)
        if arm is None:
            return run.abort(f'invalid arm {goal.arm!r}')

        if not run.phase('detect', 0.0):
            return run.result
        c = self.detect_object(goal.target_object)
        if c is None:
            return run.abort(f'no {goal.target_object!r} detected')
        bx, by, bz = c
        press_x = bx + PRESS_DEPTH
        approach_x = press_x - APPROACH_BACKOFF

        if not run.phase('approach', 0.3):
            return run.result
        if not self.close_gripper(arm):   # press with the knuckles
            return run.abort('gripper close failed')
        if not self.move_frame_to(arm, approach_x, by, bz, duration_sec=4, outer_gh=gh):
            return run.abort('approach motion failed')

        if not run.phase('press', 0.6):
            return run.result
        if not self.move_frame_to(arm, press_x, by, bz, duration_sec=2, outer_gh=gh):
            return run.abort('press motion failed')
        self.move_frame_to(arm, approach_x, by, bz, duration_sec=2, outer_gh=gh)  # retreat

        return run.succeed(f'pressed {goal.target_object!r}')
