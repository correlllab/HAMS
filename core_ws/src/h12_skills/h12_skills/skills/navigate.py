"""SkillNavigate: planning -> navigating (preset or detected target)."""


class NavigateSkill:
    def _exec_navigate(self, gh):
        """planning -> navigating. Resolves the named target from
        NAMED_LOCATIONS, else detects it in the map frame and stops
        NAV_STANDOFF in front of it, facing it."""
        raise NotImplementedError('navigate skill not implemented')
