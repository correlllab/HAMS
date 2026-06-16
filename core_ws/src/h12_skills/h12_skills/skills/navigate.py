"""SkillNavigate: planning -> navigating (preset or detected target)."""

import math

from custom_ros_messages.action import SkillNavigate

from ..base import _Run, NAMED_LOCATIONS, NAV_STANDOFF


class NavigateSkill:
    def _exec_navigate(self, gh):
        """planning -> navigating. Resolves the named target from
        NAMED_LOCATIONS, else detects it in the map frame and stops
        NAV_STANDOFF in front of it, facing it."""
        goal = gh.request
        run = _Run(self, gh, SkillNavigate, 'navigate')
        name = goal.target_object.strip()

        if not run.phase('planning', 0.0):
            return run.result
        if name in NAMED_LOCATIONS:
            gx, gy, yaw = NAMED_LOCATIONS[name]
        else:
            c = self.detect_object(name, target_frame='map')
            if c is None:
                return run.abort(f'no preset or detection for {name!r}')
            origin = self._frame_origin_in('map', 'pelvis')
            if origin is None:
                return run.abort('robot pose in map unavailable')
            dx, dy = c[0] - origin[0], c[1] - origin[1]
            dist = math.hypot(dx, dy)
            if dist < 1e-3:
                return run.abort('target coincides with the robot')
            ux, uy = dx / dist, dy / dist
            gx, gy = c[0] - NAV_STANDOFF * ux, c[1] - NAV_STANDOFF * uy
            yaw = math.atan2(dy, dx)

        if not run.phase('navigating', 0.3):
            return run.result
        if not self.navigate_to(gx, gy, yaw, timeout_sec=run.remaining(), outer_gh=gh):
            return run.abort(
                'navigation failed' if run.remaining() > 0 else 'skill timeout')

        return run.succeed(f'arrived at {name!r}')
