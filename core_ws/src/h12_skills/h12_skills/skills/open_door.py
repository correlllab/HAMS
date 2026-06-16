"""SkillOpenDoor: detect -> grasp (composed) -> pull."""

from custom_ros_messages.action import SkillOpenDoor

from ..base import _Run, _roll_quat, DEFAULT_PULL_DISTANCE


class OpenDoorSkill:
    def _exec_open_door(self, gh):
        """detect -> grasp (composed via the SkillGrasp action) -> pull.

        Detects locally as well: the pull endpoint and the grasp-roll estimate
        are needed here, and SkillGrasp's Result (success/message only) cannot
        return them. SkillGrasp re-detects internally; both detections see the
        same object, so the antipodal roll estimates agree."""
        goal = gh.request
        run = _Run(self, gh, SkillOpenDoor, 'open_door')
        arm = self._validated_arm(run, goal)
        if arm is None:
            return run.abort(f'invalid arm {goal.arm!r}')
        pull = goal.pull_distance if goal.pull_distance > 0.0 else DEFAULT_PULL_DISTANCE

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

        if not run.phase('pull', 0.75):
            return run.result
        # Hold the grasp roll during the pull so the wrist does not untwist
        # out of the antipodal grip.
        if not self.move_frame_to(arm, hx - pull, hy, hz,
                                  duration_sec=4, quat=grasp_quat, outer_gh=gh):
            return run.abort('pull motion failed')
        self.open_gripper(arm)   # release the handle, leave the door open

        return run.succeed(f'opened {goal.target_object!r} by {pull:.2f} m')
