"""SkillOpenDoor: detect -> grasp (composed) -> pull."""


class OpenDoorSkill:
    def _exec_open_door(self, gh):
        """detect -> grasp (composed via the SkillGrasp action) -> pull.

        Detects locally as well: the pull endpoint and the grasp-roll estimate
        are needed here, and SkillGrasp's Result (success/message only) cannot
        return them. SkillGrasp re-detects internally; both detections see the
        same object, so the antipodal roll estimates agree."""
        raise NotImplementedError('open_door skill not implemented')
