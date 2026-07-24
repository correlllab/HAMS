#!/usr/bin/env python3
"""Minimal arm-control isolation: go_home (NO scene-reset burst), then ONE direct
diff-IK move to a reachable low pose over the nail, and report whether the frame
actually descended. No hover, no multi-step, no OMPL."""
import time, threading, math
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from h12_skills.grasp_benchmark import GRASP_FRAMES, TCP_DEPTH_M
from h12_skills.perception_utils import matrix_to_pose
import battery_bench as bb

rclpy.init()
from h12_skills.grasp_benchmark import GraspBenchmark
bench = GraspBenchmark('right', gt_name='', box_source='gt')
ex = MultiThreadedExecutor(num_threads=6); ex.add_node(bench)
threading.Thread(target=ex.spin, daemon=True).start()
for _ in range(100):
    if bench._obj_poses.get('__pelvis__') is not None: break
    time.sleep(0.2)
frame = GRASP_FRAMES['right']

def fz():
    p = bench.frame_pose_pelvis(frame)
    return None if p is None else round(p.position.z, 4)

bench.go_home(duration_sec=5.0); time.sleep(3.0)
print('after go_home, frame_z =', fz())

nail = bench.gt_pos_pelvis('screw_27')
# a MODERATE, clearly-reachable pose: 45deg tilt, contact 8cm ABOVE the nail
def tilted(contact, pitch_deg):
    p = math.radians(pitch_deg)
    z = np.array([math.cos(p), 0.0, -math.sin(p)]); x = np.array([0.0, 1.0, 0.0]); y = np.cross(z, x)
    T = np.eye(4); T[:3, :3] = np.column_stack((x, y, z)); T[:3, 3] = np.asarray(contact) - TCP_DEPTH_M*z
    return matrix_to_pose(T)

for tag, cz, pitch in [('high hover', nail[2]+0.15, 45), ('mid', nail[2]+0.06, 45)]:
    contact = np.array([nail[0], nail[1], cz])
    ok = bench.move_frame_to(frame, tilted(contact, pitch), duration_sec=12.0, do_plan=False)
    print(f'[{tag}] cmd contact_z={cz:.3f} -> reached={ok} frame_z={fz()}')
rclpy.shutdown()
