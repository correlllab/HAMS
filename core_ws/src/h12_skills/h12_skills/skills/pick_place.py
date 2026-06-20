"""SkillPickPlace: detect place target -> grasp (composed) -> carry -> place."""


class PickPlaceSkill:
    def _exec_pick_place(self, gh):
        """detect (place target) -> grasp (via the SkillGrasp client) ->
        carry -> place -> release."""
        raise NotImplementedError('pick_place skill not implemented')
