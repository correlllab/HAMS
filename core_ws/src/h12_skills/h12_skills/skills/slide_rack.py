"""SkillSlideRack: detect -> grasp (composed) -> slide in/out along x."""


class SlideRackSkill:
    def _exec_slide_rack(self, gh):
        """detect -> grasp (composed via SkillGrasp) -> slide in/out along x."""
        raise NotImplementedError('slide_rack skill not implemented')
