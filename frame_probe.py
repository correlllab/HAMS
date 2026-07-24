#!/usr/bin/env python3
"""Where is EVERY relevant right-hand frame right now (pelvis), and where do the
FINGERTIPS actually close? Compare to the point battery_bench drives/aims so we
find the offset between the driven frame and the true grasp point."""
import time, threading
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from h12_skills.grasp_benchmark import GRASP_FRAMES, TCP_DEPTH_M
import battery_bench as bb

rclpy.init()
from h12_skills.grasp_benchmark import GraspBenchmark
bench = GraspBenchmark('right', gt_name='', box_source='gt')
ex = MultiThreadedExecutor(num_threads=4); ex.add_node(bench)
threading.Thread(target=ex.spin, daemon=True).start()
for _ in range(100):
    if bench._obj_poses.get('__pelvis__') is not None: break
    time.sleep(0.2)

def P(name):
    p = bench.frame_pose_pelvis(name)
    return None if p is None else np.array([p.position.x, p.position.y, p.position.z])

frames = ['right_graspgenx_frame', 'right_grasp_frame', 'right_wrist_yaw_link',
          'rg_left_finger', 'rg_right_finger', 'rg_right_crank', 'rg_right_rocker',
          'right_hand_camera_color_optical_frame']
print('=== right-hand frames (pelvis) ===')
for f in frames:
    v = P(f)
    print(f'  {f:42s}: {np.round(v,4).tolist() if v is not None else None}')

driven = GRASP_FRAMES['right']
fp = bench.frame_pose_pelvis(driven)
contact = bb._contact_point(bench, driven)
print(f'--- driven frame = {driven} (TCP_DEPTH={TCP_DEPTH_M}) ---')
print(f'  driven origin      : {np.round(P(driven),4).tolist() if P(driven) is not None else None}')
print(f'  bench _contact_pt  : {np.round(contact,4).tolist() if contact is not None else None}')
lf, rf = P('rg_left_finger'), P('rg_right_finger')
if lf is not None and rf is not None:
    mid = 0.5*(lf+rf)
    print(f'  finger MIDPOINT    : {np.round(mid,4).tolist()}  gap={np.linalg.norm(lf-rf)*1000:.0f}mm')
    if contact is not None:
        print(f'  contact - fingerMid: dz={1000*(contact[2]-mid[2]):+.0f}mm  (if contact is BELOW fingers, my aim puts fingers too high)')
rclpy.shutdown()
