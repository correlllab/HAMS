"""SkillOpenLid: detect -> grasp (composed) -> open along a hinge arc."""


class OpenLidSkill:
    def _exec_open_lid(self, gh):
        """detect -> grasp (composed via SkillGrasp) -> open along an arc about
        the hinge. Hinge is assumed at the far edge (+x), LID_HINGE_RADIUS
        behind the grasped point, axis horizontal: the grasp point rises and
        moves toward the hinge as the lid rotates up."""
        raise NotImplementedError('open_lid skill not implemented')
