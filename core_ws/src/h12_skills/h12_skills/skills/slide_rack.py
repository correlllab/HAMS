"""SkillSlideRack: detect -> grasp (composed) -> slide in/out along x."""

from custom_ros_messages.action import SkillSlideRack

from ..base import _Run, _roll_quat, SLIDE_DISTANCE


class SlideRackSkill:
    def _exec_slide_rack(self, gh):
        """detect -> grasp (composed via SkillGrasp) -> slide in/out along x."""
        goal = gh.request
        run = _Run(self, gh, SkillSlideRack, 'slide_rack')
        arm = self._validated_arm(run, goal)
        if arm is None:
            return run.abort(f'invalid arm {goal.arm!r}')
        direction = goal.direction.strip().lower()
        if direction not in ('in', 'out'):
            return run.abort(f'invalid direction {goal.direction!r} (use "in"|"out")')

        if not run.phase('detect', 0.0):
            return run.result
        plan = self.detect_grasp(goal.target_object)
        if plan is None:
            return run.abort(f'no {goal.target_object!r} detected')
        (rx, ry, rz), roll, _width = plan
        grasp_quat = _roll_quat(roll)

        if not run.phase('grasp', 0.4):
            return run.result
        ok, why = self._call_grasp_skill(gh, run, goal.target_object, arm)
        if not ok:
            return run.abort(f'grasp failed: {why}')

        if not run.phase('slide', 0.75):
            return run.result
        slide = -SLIDE_DISTANCE if direction == 'out' else SLIDE_DISTANCE
        if not self.move_frame_to(arm, rx + slide, ry, rz, duration_sec=4,
                                  quat=grasp_quat, outer_gh=gh):
            return run.abort('slide motion failed')
        self.open_gripper(arm)   # release the rack

        return run.succeed(f'slid {goal.target_object!r} {direction}')
