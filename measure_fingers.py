#!/usr/bin/env python3
"""Drive to the bottomed top-down grasp pose over a screw, then report exactly
where the gripper FINGERTIPS sit vs the screw (pelvis frame). No close/lift."""
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
pose, tip = bb._screw_grasp_pose(bench, SCREW)

# hover then descend straight down to the bottomed pose (no close)
over = np.asarray(tip) + np.array([0,0,bb.HOVER_M])
bench.move_frame_to(frame, bb._top_down_pose(over), duration_sec=bb.OVER_SEG_SEC, do_plan=bb.DO_PLAN)
prev_z=None
for k in range(1, 7):
    _, t = bb._screw_grasp_pose(bench, SCREW); t=np.asarray(t)
    h = bb.HOVER_M*(1.0-k/6.0)
    bench.move_frame_to(frame, bb._top_down_pose(t+np.array([0,0,h])), duration_sec=bb.SERVO_ITER_SEC, do_plan=False)
    fp = bench.frame_pose_pelvis(frame); az = fp.position.z if fp else None
    if az is not None and prev_z is not None and az > prev_z-0.003: break
    prev_z=az
time.sleep(0.8)

def P(name):
    p = bench.frame_pose_pelvis(name)
    return None if p is None else np.array([p.position.x, p.position.y, p.position.z])

scr = bench.gt_pos_pelvis(SCREW)
lf, rf = P('rg_left_finger'), P('rg_right_finger')
gf = P('right_grasp_frame')
print('=== bottomed top-down pose: fingertips vs screw (pelvis) ===')
print(f'screw_{SCREW[-2:]} origin : {np.round(scr,4) if scr is not None else None}  (head top ~ z+0.011)')
print(f'right_grasp_frame  : {np.round(gf,4) if gf is not None else None}')
print(f'rg_left_finger     : {np.round(lf,4) if lf is not None else None}')
print(f'rg_right_finger    : {np.round(rf,4) if rf is not None else None}')
if lf is not None and rf is not None and scr is not None:
    mid = 0.5*(lf+rf)
    print(f'finger MIDPOINT    : {np.round(mid,4)}')
    print(f'finger gap (|L-R|) : {np.linalg.norm(lf-rf)*1000:.1f} mm')
    print(f'midpoint - screw   : dx={1000*(mid[0]-scr[0]):+.1f} dy={1000*(mid[1]-scr[1]):+.1f} dz={1000*(mid[2]-scr[2]):+.1f} mm')
    print(f'lowest fingertip z : {min(lf[2],rf[2]):.4f}  vs screw head top z ~ {scr[2]+0.011:.4f}')
    print(f'  -> fingertip is {1000*(min(lf[2],rf[2])-(scr[2]+0.011)):+.1f} mm above screw head top (neg = below/around it)')
rclpy.shutdown()
