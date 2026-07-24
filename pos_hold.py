#!/usr/bin/env python3
"""Position the gripper centered+closed on the box screw, then HOLD (sleep) so an
external head-cam grab can capture the contact. Prints the grip + screw z."""
import sys, time, threading
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from h12_skills.grasp_benchmark import GRASP_FRAMES
import battery_bench as bb

SCREW = 'screw_27'
FLOOR = float(sys.argv[1]) if len(sys.argv) > 1 else -0.052
HOLD = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
rclpy.init()
from h12_skills.grasp_benchmark import GraspBenchmark
bench = GraspBenchmark('right', gt_name='', box_source='gt')
ex = MultiThreadedExecutor(num_threads=6); ex.add_node(bench)
threading.Thread(target=ex.spin, daemon=True).start()
for _ in range(100):
    if bench._obj_poses.get('__pelvis__') is not None: break
    time.sleep(0.2)
frame = GRASP_FRAMES['right']

bb.reset_env(bench, 'right'); time.sleep(0.8)
nail = bench.gt_pos_pelvis(SCREW)
_, tip = bb._screw_grasp_pose(bench, SCREW)
over = np.asarray(tip) + np.array([0.0, 0.0, bb.HOVER_M])
bench.move_frame_to(frame, bb._top_down_pose(over), duration_sec=bb.OVER_SEG_SEC, do_plan=bb.DO_PLAN)
cmd = np.array([nail[0]+bb.AIM_DX, nail[1]+bb.AIM_DY, FLOOR])
for it in range(4):
    bench.move_frame_to(frame, bb._top_down_pose(cmd), duration_sec=bb.SERVO_ITER_SEC, do_plan=False)
    m = bb._finger_mid_xy(bench, 'right')
    if m is not None: cmd[:2] = cmd[:2] - (m - nail[:2])
c = bb._contact_point(bench, frame)
print(f'positioned: TCPz={c[2]*1000:.0f}mm  fingerMid={bb._finger_mid_xy(bench,"right")}  nail={nail[:2]}', flush=True)
bench.close_gripper('right', force_n=6.0); time.sleep(2.5)
g = bench._grip_last.get('right')
print(f'CLOSED grip={g[0] if g else None}mm force={g[1] if g else None}N  screw_z={bench.gt_pos(SCREW)[2]:.4f}', flush=True)
print(f'HOLDING {HOLD}s for capture...', flush=True)
time.sleep(HOLD)
print('done', flush=True)
rclpy.shutdown()
