#!/usr/bin/env python3
"""h12 skills node: action servers for the RoboCasa-style atomic skills.

Serves the 11 Skill* actions from custom_ros_messages on /skill/<name>. The
shared machinery (vision pipeline, frame_task, grippers, nav2, the per-skill
action clients used for composition) lives in SkillsBase (base.py); each skill's
execute callback is a mixin under skills/. SkillsNode multiply-inherits from
SkillsBase plus every skill mixin and wires the action servers here.

Skills compose through this node's own clients (SkillPickPlace calls SkillGrasp
via self.skill_clients['grasp']); those clients also double as a reference for
invoking the skills externally. Skills execute *inside* action server callbacks
while a MultiThreadedExecutor spins, so inner service/action calls wait on
futures with an event instead of spinning.
"""

from functools import partial

import rclpy
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor

from .base import SkillsBase, SKILL_ACTIONS
from .skills import (
    OpenDoorSkill, CloseDoorSkill, OpenLidSkill, CloseLidSkill, NavigateSkill,
    GraspSkill, PickPlaceSkill, PressSkill, SlideRackSkill, TurnSkill,
)


class SkillsNode(SkillsBase, OpenDoorSkill, CloseDoorSkill, OpenLidSkill,
                 CloseLidSkill, NavigateSkill, GraspSkill, PickPlaceSkill,
                 PressSkill, SlideRackSkill, TurnSkill):
    def __init__(self):
        super().__init__()   # SkillsBase: clients, perception, motion primitives

        # --- skill action servers ----------------------------------------------
        # turn_lever/twist_knob share _exec_turn; bind their action type (from
        # SKILL_ACTIONS), label, and motion phase with partial().
        executors = {
            'open_door':  self._exec_open_door,
            'close_door': self._exec_close_door,
            'open_lid':   self._exec_open_lid,
            'close_lid':  self._exec_close_lid,
            'navigate':   self._exec_navigate,
            'grasp':      self._exec_grasp,
            'pick_place': self._exec_pick_place,
            'press':      self._exec_press,
            'slide_rack': self._exec_slide_rack,
            'turn_lever': partial(self._exec_turn, SKILL_ACTIONS['turn_lever'][0],
                                  'turn_lever', 'turn'),
            'twist_knob': partial(self._exec_turn, SKILL_ACTIONS['twist_knob'][0],
                                  'twist_knob', 'twist'),
        }
        self.skill_servers = {
            name: ActionServer(
                self, SKILL_ACTIONS[name][0], SKILL_ACTIONS[name][1],
                execute_callback=executors[name],
                cancel_callback=self._on_skill_cancel,
                callback_group=self._cb_group,
            )
            for name in SKILL_ACTIONS
        }

        self.get_logger().info(
            f'h12_skills ready: serving {sorted(self.skill_servers)} on /skill/<name>')


def main():
    rclpy.init()
    node = SkillsNode()
    # Skills block inside their execute callbacks while waiting on inner
    # service/action futures (and pick_place calls the grasp server hosted by
    # this same node), so a multithreaded executor is required.
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
