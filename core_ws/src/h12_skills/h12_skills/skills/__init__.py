"""Per-skill mixin classes for SkillsNode.

Each module defines one mixin whose _exec_* method operates on `self`; SkillsNode
(h12_skills/node.py) multiply-inherits from SkillsBase plus every mixin here, so
the bodies resolve self.detect_grasp/self.move_frame_to/... at runtime."""

from .open_door import OpenDoorSkill
from .close_door import CloseDoorSkill
from .open_lid import OpenLidSkill
from .close_lid import CloseLidSkill
from .navigate import NavigateSkill
from .grasp import GraspSkill
from .pick_place import PickPlaceSkill
from .press import PressSkill
from .slide_rack import SlideRackSkill
from .turn import TurnSkill

__all__ = [
    'OpenDoorSkill', 'CloseDoorSkill', 'OpenLidSkill', 'CloseLidSkill',
    'NavigateSkill', 'GraspSkill', 'PickPlaceSkill', 'PressSkill',
    'SlideRackSkill', 'TurnSkill',
]
