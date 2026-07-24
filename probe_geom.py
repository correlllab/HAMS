#!/usr/bin/env python3
"""Read-only geometry probe for the battery grasp: screw GT positions in pelvis,
which screw 'auto' picks, current right_grasp_frame pose, and the top-down hover
pose the bench would command. No motion."""
import threading, time
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from h12_skills.grasp_benchmark import GRASP_FRAMES
import battery_bench as bb

rclpy.init()
from h12_skills.grasp_benchmark import GraspBenchmark
bench = GraspBenchmark('right', gt_name='', box_source='gt')
ex = MultiThreadedExecutor(num_threads=4)
ex.add_node(bench)
threading.Thread(target=ex.spin, daemon=True).start()

for _ in range(100):
    if bench._obj_poses.get('__pelvis__') is not None:
        break
    time.sleep(0.2)

frame = GRASP_FRAMES['right']
fp = bench.frame_pose_pelvis(frame)
print('=== current right_grasp_frame (pelvis) ===')
if fp is not None:
    print(f'  origin: [{fp.position.x:.3f}, {fp.position.y:.3f}, {fp.position.z:.3f}]')

print('=== all screws (pelvis frame), sorted by x ===')
rows = []
for name in sorted(k for k in bench._obj_poses if k.startswith('screw')):
    p = bench.gt_pos_pelvis(name)
    if p is not None:
        rows.append((name, p))
for name, p in sorted(rows, key=lambda r: r[1][0]):
    print(f'  {name:12s} x={p[0]:+.3f} y={p[1]:+.3f} z={p[2]:+.3f}')

auto = bb._pick_screw(bench, 'right')
print(f'=== auto-picked screw: {auto} ===')
if auto:
    pose, tip = bb._screw_grasp_pose(bench, auto)
    over = np.asarray(tip) + np.array([0.0, 0.0, bb.HOVER_M])
    hover_pose = bb._top_down_pose(over)
    print(f'  screw tip (contact target, pelvis): [{tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f}]')
    print(f'  HOVER frame origin: [{hover_pose.position.x:.3f}, {hover_pose.position.y:.3f}, {hover_pose.position.z:.3f}]')
    print(f'  TOP_DOWN_PITCH_DEG={bb.TOP_DOWN_PITCH_DEG}  HOVER_M={bb.HOVER_M}  DO_PLAN={bb.DO_PLAN}')
    # shoulder ~ z=+0.42 rel pelvis; report vertical drop from shoulder to screw
    print(f'  screw depth below pelvis: {-tip[2]:.3f} m')

rclpy.shutdown()
