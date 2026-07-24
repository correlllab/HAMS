#!/usr/bin/env python3
"""Isolate OMPL: reset, report the driven frame at home, then plan (do_plan=True)
to a HIGH pre-grasp above screw_27 and to the top-down grasp frame. Report the
achieved error + whether it looked like a clean plan or a stalled diff-IK move."""
import time, threading
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Pose, Point, Quaternion
from h12_skills.grasp_benchmark import GRASP_FRAMES, TCP_DEPTH_M

rclpy.init()
from h12_skills.grasp_benchmark import GraspBenchmark
bench = GraspBenchmark('right', gt_name='screw_27', box_source='gt')
ex = MultiThreadedExecutor(num_threads=6); ex.add_node(bench)
threading.Thread(target=ex.spin, daemon=True).start()
for _ in range(100):
    if bench.gt_pos('screw_27') is not None: break
    time.sleep(0.2)
frame = GRASP_FRAMES['right']

def fp():
    p = bench.frame_pose_pelvis(frame)
    return None if p is None else np.array([p.position.x, p.position.y, p.position.z])

# reset to home
from std_msgs.msg import Empty
pub = bench.create_publisher(Empty, '/hams/reset_scene', 10)
for _ in range(6): pub.publish(Empty()); time.sleep(0.1)
time.sleep(1.5)
bench.go_home(duration_sec=5.0); time.sleep(3.0)
print('driven frame at HOME (pelvis):', np.round(fp(), 3))

scr = bench.gt_pos_pelvis('screw_27')      # [0.273,-0.150,-0.165]
print('screw_27 pelvis:', np.round(scr, 3))

def topdown_pose(contact_xyz, tilt=0.0):
    # straight-down z-axis, fingers along +Y; frame = contact - TCP_DEPTH*zaxis
    import math
    z = np.array([math.sin(tilt), 0.0, -math.cos(tilt)])
    x = np.array([0.0, 1.0, 0.0])
    y = np.cross(z, x)
    T = np.eye(4); T[:3, :3] = np.column_stack((x, y, z))
    T[:3, 3] = np.asarray(contact_xyz) - TCP_DEPTH_M * z
    from h12_skills.perception_utils import matrix_to_pose
    return matrix_to_pose(T)

# targets: (label, contact_xyz, tilt)
tip = np.array([scr[0], scr[1], scr[2] + 0.005])
targets = [
    ('PRE-GRASP topdown (10cm up)', tip + np.array([0, 0, 0.10]), 0.0),
    ('GRASP topdown',               tip,                          0.0),
    ('GRASP 30deg tilt',            tip,                          np.pi/6),
]
for label, contact, tilt in targets:
    pose = topdown_pose(contact, tilt)
    goal = np.array([pose.position.x, pose.position.y, pose.position.z])
    t0 = time.time()
    ok = bench.move_frame_to(frame, pose, duration_sec=20.0, do_plan=True,
                             lin_tol=0.035, ang_tol=0.12)
    dt = time.time() - t0
    cur = fp()
    err = np.linalg.norm(cur - goal) if cur is not None else None
    print(f'[{label}] do_plan=True reached={ok} in {dt:.1f}s  '
          f'frame_target={np.round(goal,3)} achieved={np.round(cur,3)} '
          f'lin_err={err*1000:.0f}mm')
rclpy.shutdown()
