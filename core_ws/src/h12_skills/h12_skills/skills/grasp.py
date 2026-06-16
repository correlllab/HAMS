"""SkillGrasp: detect (antipodal plan) -> approach -> grasp."""

import math

from rclpy.duration import Duration as RclpyDuration

from custom_ros_messages.action import SkillGrasp

from ..base import (_Run, _roll_quat, APPROACH_BACKOFF, GRIPPER_OPEN_MM,
                    GRASP_WIDTH_MARGIN_MM, GRASP_PREOPEN_MIN_MM, GRIP_SETTLE_SEC)


class GraspSkill:
    def _exec_grasp(self, gh):
        """detect (antipodal plan) -> approach -> grasp. No lift (by design).

        The detection cloud's y-z footprint is PCA'd; the wrist rolls so the
        fingers close along the minor axis (antipodal contacts on opposing
        surfaces) and the gripper pre-opens to the measured width before the
        approach, so the fingers straddle the object on arrival."""
        goal = gh.request
        run = _Run(self, gh, SkillGrasp, 'grasp')
        arm = self._validated_arm(run, goal)
        if arm is None:
            return run.abort(f'invalid arm {goal.arm!r}')

        if not run.phase('detect', 0.0):
            return run.result
        plan = self.detect_grasp(goal.target_object)
        if plan is None:
            return run.abort(f'no {goal.target_object!r} detected')
        (hx, hy, hz), roll, width_mm = plan
        grasp_quat = _roll_quat(roll)
        preopen = min(GRIPPER_OPEN_MM,
                      max(width_mm + GRASP_WIDTH_MARGIN_MM, GRASP_PREOPEN_MIN_MM))
        if width_mm + GRASP_WIDTH_MARGIN_MM > GRIPPER_OPEN_MM:
            self.get_logger().warn(
                f'{goal.target_object!r} measured width {width_mm:.0f} mm may '
                f'exceed the gripper opening ({GRIPPER_OPEN_MM:.0f} mm)')

        if not run.phase('approach', 0.35):
            return run.result
        if not self.set_gripper(arm, preopen):
            return run.abort('gripper pre-open failed')
        if not self.move_frame_to(arm, hx - APPROACH_BACKOFF, hy, hz,
                                  duration_sec=4, quat=grasp_quat, outer_gh=gh):
            return run.abort('approach motion failed')

        if not run.phase('grasp', 0.7):
            return run.result
        if not self.move_frame_to(arm, hx, hy, hz, duration_sec=2,
                                  quat=grasp_quat, outer_gh=gh):
            return run.abort('contact motion failed')
        if not self.close_gripper(arm):
            return run.abort('gripper close failed')
        self.get_clock().sleep_for(RclpyDuration(seconds=GRIP_SETTLE_SEC))

        return run.succeed(
            f'grasped {goal.target_object!r} (antipodal: roll '
            f'{math.degrees(roll):.0f} deg, width {width_mm:.0f} mm)')
