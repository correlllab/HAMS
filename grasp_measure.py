#!/usr/bin/env python3
"""Grasp the nail, then measure WHERE the grasped nail sits in the wrist frame — that
is the exact point the gripper grasps the object, so the red marker goes there."""
import time, threading, numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from rclpy.duration import Duration
from h12_skills.grasp_benchmark import GRASP_FRAMES
import battery_bench as bb

SCREW = 'screw_27'
rclpy.init()
from h12_skills.grasp_benchmark import GraspBenchmark
b = GraspBenchmark('right', gt_name='', box_source='gt')
ex = MultiThreadedExecutor(num_threads=6); ex.add_node(b)
threading.Thread(target=ex.spin, daemon=True).start()
for _ in range(100):
    if b._obj_poses.get('__pelvis__') is not None: break
    time.sleep(0.2)
frame = GRASP_FRAMES['right']

bb.reset_env(b, 'right'); time.sleep(0.8)
nail = b.gt_pos_pelvis(SCREW)
_, tip = bb._screw_grasp_pose(b, SCREW)
over = np.asarray(tip) + np.array([0,0,0.10])
b.move_frame_to(frame, bb._top_down_pose(over), duration_sec=6.0, do_plan=bb.DO_PLAN)
cmd = np.array([nail[0], nail[1], -0.052])
for it in range(2):
    b.move_frame_to(frame, bb._top_down_pose(cmd), duration_sec=6.0, do_plan=False)
    m = bb._tip_mid_xy(b, "right")
    if m is not None: cmd[:2] = cmd[:2] - (m - nail[:2])
time.sleep(1.0)
b.close_gripper('right', force_n=6.0); time.sleep(3.0)

# nail position now, in the wrist frame
from h12_skills.perception_utils import pose_to_matrix
from geometry_msgs.msg import Pose
try:
    t = b.tf_buffer.lookup_transform('right_wrist_yaw_link', 'pelvis', Time(), timeout=Duration(seconds=2.0))
    tr = t.transform.translation; q = t.transform.rotation
    p = Pose(); p.position.x, p.position.y, p.position.z = tr.x, tr.y, tr.z; p.orientation = q
    M = pose_to_matrix(p)  # pelvis -> wrist
    nail_w = b.gt_pos_pelvis(SCREW)                 # nail in pelvis
    nw = (M @ np.array([nail_w[0], nail_w[1], nail_w[2], 1.0]))[:3]
    g = b._grip_last.get('right')
    print(f'grip={g[0] if g else None}mm force={g[1] if g else None}N screw_z={b.gt_pos(SCREW)[2]:.4f}')
    print(f'==> GRASPED NAIL in WRIST frame (put red marker here): {np.round(nw,4).tolist()}')
    print(f'    current red marker at x=0.255; measured tips at x=0.245')
except Exception as e:
    print('tf err', e)
rclpy.shutdown()
