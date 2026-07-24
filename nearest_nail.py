#!/usr/bin/env python3
"""Where is the right gripper's CONTACT point now, and which yellow nail is it
closest to? That nail is the reachable target."""
import time, threading
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from h12_skills.grasp_benchmark import GRASP_FRAMES
import battery_bench as bb

rclpy.init()
from h12_skills.grasp_benchmark import GraspBenchmark
bench = GraspBenchmark('right', gt_name='', box_source='gt')
ex = MultiThreadedExecutor(num_threads=4); ex.add_node(bench)
threading.Thread(target=ex.spin, daemon=True).start()
for _ in range(100):
    if bench._obj_poses.get('__pelvis__') is not None: break
    time.sleep(0.2)
frame = GRASP_FRAMES['right']
c = bb._contact_point(bench, frame)
fp = bench.frame_pose_pelvis(frame)
print('right gripper frame (pelvis):', None if fp is None else [round(fp.position.x,3),round(fp.position.y,3),round(fp.position.z,3)])
print('right gripper CONTACT (pelvis):', None if c is None else np.round(c,4).tolist())
best=None; bestd=1e9
for name in sorted(k for k in bench._obj_poses if k.startswith('screw')):
    p = bench.gt_pos_pelvis(name)
    if p is None: continue
    if p[0] > 0.6: continue     # far row unreachable
    d = float(np.linalg.norm(c - p)) if c is not None else 1e9
    dxy = float(np.hypot(c[0]-p[0], c[1]-p[1])) if c is not None else 1e9
    if d < bestd:
        bestd=d; best=(name, p, d, dxy)
print('--- 3 nearest near-row nails to the contact ---')
rows=[]
for name in sorted(k for k in bench._obj_poses if k.startswith('screw')):
    p = bench.gt_pos_pelvis(name)
    if p is None or p[0]>0.6: continue
    d = float(np.linalg.norm(c-p)) if c is not None else 1e9
    rows.append((d,name,p))
for d,name,p in sorted(rows)[:4]:
    print(f'  {name}: pelvis={np.round(p,3).tolist()} dist={d*1000:.0f}mm dxy={1000*np.hypot(c[0]-p[0],c[1]-p[1]):.0f}mm dz={1000*(c[2]-p[2]):.0f}mm')
rclpy.shutdown()
