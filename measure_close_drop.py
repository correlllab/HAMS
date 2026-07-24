#!/usr/bin/env python3
"""Measure the offset between where battery_bench AIMS (contact point) and where
the fingers ACTUALLY close (fingertip midpoint after close). Accounts for the
tilt (tip lands fwd/back) and the close-drop (fingers arc down)."""
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

def P(name):
    p = bench.frame_pose_pelvis(name)
    return None if p is None else np.array([p.position.x, p.position.y, p.position.z])
def fingermid():
    lf, rf = P('rg_left_finger'), P('rg_right_finger')
    return None if (lf is None or rf is None) else 0.5*(lf+rf)

bb.reset_env(bench, 'right')
p = bench.gt_pos_pelvis(SCREW)
nail = np.array([p[0], p[1], p[2]])
print('nail (pelvis):', np.round(nail,4).tolist())

# drive the gripper so its CONTACT lands on the nail (simple: hover then one move)
pose, tip = bb._screw_grasp_pose(bench, SCREW)
over = np.asarray(tip) + np.array([0,0,0.12])
bench.move_frame_to(frame, bb._top_down_pose(over), duration_sec=6.0, do_plan=bb.DO_PLAN)
bench.move_frame_to(frame, bb._top_down_pose(np.asarray(tip)), duration_sec=10.0, do_plan=False)
time.sleep(0.8)

c_open = bb._contact_point(bench, frame)
fm_open = fingermid()
print('=== OPEN ===')
print('  bench contact point :', np.round(c_open,4).tolist() if c_open is not None else None)
print('  finger midpoint     :', np.round(fm_open,4).tolist() if fm_open is not None else None)

bench.close_gripper('right'); time.sleep(2.5)
c_close = bb._contact_point(bench, frame)
fm_close = fingermid()
print('=== CLOSED ===')
print('  bench contact point :', np.round(c_close,4).tolist() if c_close is not None else None)
print('  finger midpoint     :', np.round(fm_close,4).tolist() if fm_close is not None else None)

if fm_close is not None:
    off = nail - fm_close
    print('=== OFFSET: nail - closed_finger_midpoint (add this to the AIM) ===')
    print(f'  dx={1000*off[0]:+.0f}mm dy={1000*off[1]:+.0f}mm dz={1000*off[2]:+.0f}mm')
if fm_open is not None and fm_close is not None:
    drop = fm_close - fm_open
    print(f'  close-drop (closed-open finger mid): dx={1000*drop[0]:+.0f} dy={1000*drop[1]:+.0f} dz={1000*drop[2]:+.0f}mm')
rclpy.shutdown()
