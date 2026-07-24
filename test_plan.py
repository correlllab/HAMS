#!/usr/bin/env python3
"""Test: can OMPL plan the gripper DIRECTLY onto the raised screw (top-down),
reaching a deeper arm config than diff-IK stepping? Reset, plan to grasp pose,
report reached frame z + contact err, close, lift, grade."""
import sys, time, threading
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from h12_skills.grasp_benchmark import GRASP_FRAMES
import battery_bench as bb

SCREW = sys.argv[1] if len(sys.argv) > 1 else 'screw_27'
AIM_BELOW = float(sys.argv[2]) if len(sys.argv) > 2 else 0.006   # aim this far BELOW screw origin

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
p = bench.gt_pos_pelvis(SCREW)
aim = np.array([p[0], p[1], p[2] - AIM_BELOW])          # contact target: at/below screw origin
grasp_pose = bb._top_down_pose(aim)

# 1) hover directly above (planned), 2) PLAN straight to the grasp pose
over = bb._top_down_pose(aim + np.array([0,0,bb.HOVER_M]))
bench.move_frame_to(frame, over, duration_sec=bb.OVER_SEG_SEC, do_plan=True)
time.sleep(0.4)
bench.move_frame_to(frame, grasp_pose, duration_sec=bb.CONTACT_SEC, do_plan=True)   # PLANNED descent
time.sleep(0.8)

fp = bench.frame_pose_pelvis(frame)
err = bb._frame_err_to(bench, frame, aim)
print('=== planned descent result ===')
print(f'screw origin z (pelvis): {p[2]:.4f}  aim z: {aim[2]:.4f}')
print(f'reached frame z: {fp.position.z:.4f}  (frame xyz={fp.position.x:.3f},{fp.position.y:.3f},{fp.position.z:.3f})')
print(f'contact err to aim: {err*1000:.1f} mm')

z0 = bench.gt_pos(SCREW)
bench.close_gripper('right'); time.sleep(1.2)
g = bench._grip_last.get('right')
lifted = bench.frame_pose_pelvis(frame)
if lifted is not None:
    lifted.position.z += bb.LIFT_DIST_M
    bench.move_frame_to(frame, lifted, duration_sec=4.0, do_plan=True)
    time.sleep(1.2)
z1 = bench.gt_pos(SCREW)
dz = float(z1[2]-z0[2]) if (z0 is not None and z1 is not None) else 0.0
print(f'grip_final_mm: {g[0] if g else None}   screw_lift_dz: {dz:.4f}   SUCCESS={dz>=0.03}')
rclpy.shutdown()
