#!/usr/bin/env python3
"""h12 skills node: action servers for the RoboCasa-style atomic skills.

Serves the 11 Skill* actions from custom_ros_messages on /skill/<name>. The
shared machinery — the frame_task arm IK action, the per-arm gripper services,
the gemini_query and sam_segment perception services, and the head-camera image
cache — lives in SkillsBase (base.py); each skill's execute callback is a mixin
under skills/. SkillsNode multiply-inherits from SkillsBase plus every skill
mixin and wires the action servers here.

Skills execute *inside* action server callbacks while a MultiThreadedExecutor
spins, so inner service/action calls wait on futures with an event instead of
spinning.
"""

from functools import partial

import rclpy
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor

from .base import SkillsBase, SKILL_ACTIONS
from .model_logging import ModelLogger, declare_logging_params
from .skills import (
    OpenDoorSkill, CloseDoorSkill, OpenLidSkill, CloseLidSkill, NavigateSkill,
    GraspSkill, PickPlaceSkill, PressSkill, SlideRackSkill, TurnSkill,
)


def _describe_msg(msg):
    """Flatten an action goal's primitive fields into a JSON-friendly dict.
    Lists/nested messages are summarized (length / type) rather than dumped."""
    out = {}
    try:
        for fname in msg.get_fields_and_field_types():
            v = getattr(msg, fname)
            if isinstance(v, (bool, int, float, str)):
                out[fname] = v
            elif isinstance(v, (list, tuple, bytes)):
                out[fname] = f'[{len(v)} items]'
            else:
                out[fname] = type(v).__name__
    except Exception as e:
        out['_error'] = str(e)
    return out


class SkillsNode(SkillsBase, OpenDoorSkill, CloseDoorSkill, OpenLidSkill,
                 CloseLidSkill, NavigateSkill, GraspSkill, PickPlaceSkill,
                 PressSkill, SlideRackSkill, TurnSkill):
    def __init__(self):
        super().__init__()   # SkillsBase: clients, perception, motion primitives

        log, viz, clear = declare_logging_params(self)
        self.logger = ModelLogger(self, 'skills', 'h12_skills', __file__,
                                  log=log, visualize=viz, clear=clear)

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
                execute_callback=self._safe_exec(
                    name, SKILL_ACTIONS[name][0], executors[name]),
                cancel_callback=self._on_skill_cancel,
                callback_group=self._cb_group,
            )
            for name in SKILL_ACTIONS
        }

        self.get_logger().info(
            f'h12_skills ready: serving {sorted(self.skill_servers)} on /skill/<name>')

    def _safe_exec(self, label, action_type, fn):
        """Wrap a skill execute callback so any escaping exception aborts the goal
        with a Result, instead of leaving it stuck in EXECUTING (also catches the
        NotImplementedError from the not-yet-implemented skill stubs)."""
        def _cb(goal_handle):
            rec = self.logger.start()
            rec.set(skill=label, request=_describe_msg(goal_handle.request))
            self._log_head_frame(rec, f'{label}_start')
            try:
                result = fn(goal_handle)
                rec.finish(success=bool(getattr(result, 'success', False)),
                           message=getattr(result, 'message', ''))
                return result
            except Exception as e:
                self.get_logger().error(f'[{label}] internal error: {e}')
                result = action_type.Result()
                result.success = False
                result.message = f'internal error: {e}'
                goal_handle.abort()
                rec.finish(success=False, message=result.message)
                return result
        return _cb

    def _log_head_frame(self, rec, tag):
        """Save the latest head-camera frame as a visualization artifact, captioned
        with the skill tag. No-op unless visualization is on and a frame is cached."""
        if not self.logger.visualize:
            return
        img = self.latest_image()
        if img is None:
            return
        try:
            import cv2
            import numpy as np
            buf = np.frombuffer(bytes(img.data), dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                return
            cv2.putText(bgr, tag, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)
            rec.save_overlay(tag, bgr)
        except Exception as e:
            self.get_logger().warn(f'head-frame viz failed: {e}')


def main():
    rclpy.init()
    node = SkillsNode()
    # Skills block inside their execute callbacks while waiting on inner
    # service/action futures, so a multithreaded executor is required.
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
