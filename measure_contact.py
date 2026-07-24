#!/usr/bin/env python3
"""Reset, run the default 80-deg descent, and report EXACTLY where the gripper's
CONTACT (pinch) point bottoms out vs the screw — so we can move the nail to the
80-deg landing point instead of the (impossible) 90-deg straight-down point."""
import sys, time, threading
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from h12_skills.grasp_benchmark import GRASP_FRAMES
import battery_bench as bb

SCREW = sys.argv[1] if len(sys.argv) > 1 else 'screw_27'
rclpy.init()
from h12_skills.grasp_benchmark import GraspBenchmark
bench = GraspBenchmark('right', gt_name='', box_source='gt')
ex = MultiThreadedExecutor(num_threads=6); ex.add_node(bench)
threading.Thread(target=ex.spin, daemon=True).start()
for _ in range(100):
    if bench._obj_poses.get('__pelvis__') is not None: break
    time.sleep(0.2)
frame = GRASP_FRAMES['right']

bb.reset_env(bench, 'right')
_, tip = bb._screw_grasp_pose(bench, SCREW)
over = np.asarray(tip) + np.array([0,0,bb.HOVER_M])
bench.move_frame_to(frame, bb._top_down_pose(over), duration_sec=bb.OVER_SEG_SEC, do_plan=True)
best_c, best_z = None, 1e9
for k in range(1, 9):
    _, t = bb._screw_grasp_pose(bench, SCREW); t=np.asarray(t)
    h = bb.HOVER_M*(1.0-k/8.0)
    bench.move_frame_to(frame, bb._top_down_pose(t+np.array([0,0,h])), duration_sec=bb.SERVO_ITER_SEC, do_plan=False)
    c = bb._contact_point(bench, frame)           # where the fingers actually close
    if c is not None and c[2] < best_z:
        best_z, best_c = c[2], c
scr = bench.gt_pos_pelvis(SCREW)
print('=== 80-deg landing (pinch/contact) vs screw, pelvis ===')
print(f'screw origin        : {np.round(scr,4)}')
print(f'deepest CONTACT pt  : {np.round(best_c,4)}')
print(f'move nail by (dx,dy,dz to put it UNDER the contact): '
      f'{np.round(best_c-scr,4)}   (dx={1000*(best_c[0]-scr[0]):+.1f} dy={1000*(best_c[1]-scr[1]):+.1f} dz={1000*(best_c[2]-scr[2]):+.1f} mm)')
# also report in WORLD so we can move the screw body pos directly
pw = bench.gt_pos(SCREW)
print(f'screw WORLD origin  : {np.round(pw,4)}')
rclpy.shutdown()
