"""SkillPickPlace: detect place target -> grasp (composed) -> carry -> place."""

from custom_ros_messages.action import SkillPickPlace

from ..base import _Run, APPROACH_BACKOFF, PLACE_HOVER


class PickPlaceSkill:
    def _exec_pick_place(self, gh):
        """detect (place target) -> grasp (via the SkillGrasp client) ->
        carry -> place -> release."""
        goal = gh.request
        run = _Run(self, gh, SkillPickPlace, 'pick_place')
        arm = self._validated_arm(run, goal)
        if arm is None:
            return run.abort(f'invalid arm {goal.arm!r}')

        if not run.phase('detect', 0.0):
            return run.result
        place = self.detect_object(goal.place_target)
        if place is None:
            return run.abort(f'no place target {goal.place_target!r} detected')
        px, py, pz = place

        if not run.phase('grasp', 0.2):
            return run.result
        ok, why = self._call_grasp_skill(gh, run, goal.target_object, arm)
        if not ok:
            return run.abort(f'grasp failed: {why}')

        if not run.phase('carry', 0.5):
            return run.result
        hover_x = px - APPROACH_BACKOFF
        if not self.move_frame_to(arm, hover_x, py, pz + PLACE_HOVER, duration_sec=4,
                                  outer_gh=gh):
            return run.abort('carry motion failed')

        if not run.phase('place', 0.7):
            return run.result
        if not self.move_frame_to(arm, px, py, pz + 0.05,
                                  duration_sec=3, outer_gh=gh):
            return run.abort('place motion failed')

        if not run.phase('release', 0.9):
            return run.result
        if not self.open_gripper(arm):
            return run.abort('gripper release failed')
        self.move_frame_to(arm, hover_x, py, pz + PLACE_HOVER, duration_sec=2,
                           outer_gh=gh)  # retreat

        return run.succeed(
            f'placed {goal.target_object!r} at {goal.place_target!r}')
