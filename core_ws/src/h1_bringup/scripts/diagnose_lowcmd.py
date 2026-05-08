#!/usr/bin/env python3
"""Diagnose whether rt/lowcmd is reaching MuJoCo with changing joint targets.

Run this from inside the ROS container *while* the launch is up and you are
dragging arm sliders. Prints the rate and the q values for the wrist + shoulder
joints once per second, plus the rate of rt/lowstate (sim → controller).

    python3 core_ws/src/h1_bringup/scripts/diagnose_lowcmd.py

Decision tree (matches the plan at .claude/plans/...md):
- lowcmd Hz ≈ 500 and q values change as you drag → controller is publishing
  fine; investigate the MuJoCo bridge applying it.
- lowcmd Hz ≈ 500 but q values are CONSTANT → IK output isn't reaching the
  publisher; investigate apply_joint_position.
- lowcmd Hz ≈ 0 → publisher daemon wedged; check stdout for estop / DDS errors.
- lowstate Hz ≈ 0 → the controller has no robot state, so IK has no error to
  drive; investigate the MuJoCo lowstate publisher.
"""

import os
import time

os.environ.setdefault('ROS_DOMAIN_ID', '1')

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_


# Joint indices per
# core_ws/src/h12_ros2_controller/h12_ros2_controller/utility/joint_limits.py.
WATCH = [
    (14, 'L_shldr_pitch'),
    (20, 'L_wrist_yaw'),
    (21, 'R_shldr_pitch'),
    (27, 'R_wrist_yaw'),
]


def main():
    ChannelFactoryInitialize(id=int(os.environ['ROS_DOMAIN_ID']))

    cmd_count = [0]
    state_count = [0]
    last_cmd_q = [None]
    last_state_q = [None]

    def cmd_cb(msg):
        cmd_count[0] += 1
        last_cmd_q[0] = [(name, round(msg.motor_cmd[i].q, 4)) for i, name in WATCH]

    def state_cb(msg):
        state_count[0] += 1
        last_state_q[0] = [(name, round(msg.motor_state[i].q, 4)) for i, name in WATCH]

    cmd_sub = ChannelSubscriber('rt/lowcmd', LowCmd_)
    cmd_sub.Init(cmd_cb, 10)
    state_sub = ChannelSubscriber('rt/lowstate', LowState_)
    state_sub.Init(state_cb, 10)

    print(f'listening on ROS_DOMAIN_ID={os.environ["ROS_DOMAIN_ID"]}; drag sliders now')
    for sec in range(20):
        time.sleep(1.0)
        print(
            f'[{sec+1:2d}s] '
            f'lowcmd Hz≈{cmd_count[0]:4d}  q={last_cmd_q[0]}\n'
            f'      lowstate Hz≈{state_count[0]:4d}  q={last_state_q[0]}'
        )
        cmd_count[0] = 0
        state_count[0] = 0


if __name__ == '__main__':
    main()
