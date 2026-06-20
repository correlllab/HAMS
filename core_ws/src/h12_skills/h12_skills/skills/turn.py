"""Shared lever/knob executor for SkillTurnLever and SkillTwistKnob.

node.py binds the concrete action type / label / motion-phase per skill with
partial(), so this module needs no Skill* action-type imports of its own."""


class TurnSkill:
    def _exec_turn(self, action_type, label, motion_phase, gh):
        """Shared lever/knob executor: detect -> grasp (composed via SkillGrasp)
        -> rotate the wrist about the approach (+x) axis by the requested
        signed angle, relative to the antipodal grasp roll."""
        raise NotImplementedError('turn skill not implemented')
