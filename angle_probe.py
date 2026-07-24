#!/usr/bin/env python3
"""Drive the arm to its top-down pose over the nail, then read the gripper's
ACTUAL orientation in the world (approach axis + angle from vertical) and where
the fingers truly close, so we grasp based on the achieved angle, not commanded."""
import sys, time, threading, math
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from h12_skills.grasp_benchmark import GRASP_FRAMES, TCP_DEPTH_M
from h12_skills.perception_utils import pose_to_matrix
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
nail = bench.gt_pos_pelvis(SCREW)
print('nail (pelvis):', np.round(nail,4).tolist())

_, tip = bb._screw_grasp_pose(bench, SCREW)
over = np.asarray(tip) + np.array([0,0,0.12])
bench.move_frame_to(frame, bb._top_down_pose(over), duration_sec=6.0, do_plan=bb.DO_PLAN)
# multi-step descend (the motion that got close), tracking achieved frame z
lowest=None
for k in range(1,11):
    h = 0.12*(1.0-k/10.0)
    bench.move_frame_to(frame, bb._top_down_pose(np.asarray(tip)+np.array([0,0,h])),
                        duration_sec=5.0, do_plan=False)
    fp=bench.frame_pose_pelvis(frame)
    az=fp.position.z if fp else None
    if az is not None and (lowest is None or az<lowest): lowest=az
    if az is not None and az>(lowest+0.02): break     # stop if it starts rising a lot

fp = bench.frame_pose_pelvis(frame)
if fp is None:
    print('NO TF for gripper frame — arm/TF broken'); rclpy.shutdown(); sys.exit()
T = pose_to_matrix(fp)
zax = T[:3,2]                       # gripper approach axis (points toward fingers) in pelvis
origin = np.array([fp.position.x, fp.position.y, fp.position.z])
contact = origin + TCP_DEPTH_M*zax
angle_from_vertical = math.degrees(math.acos(max(-1,min(1,-zax[2]))))  # 0 = straight down
print('=== ACHIEVED gripper pose (pelvis) ===')
print(f'  frame origin      : {np.round(origin,4).tolist()}')
print(f'  approach axis z   : {np.round(zax,3).tolist()}')
print(f'  ANGLE from vertical: {angle_from_vertical:.1f} deg  (0=straight down)')
print(f'  contact (fingertip): {np.round(contact,4).tolist()}')
off = nail - contact
print(f'=== contact -> nail offset (drive the gripper this much more): dx={1000*off[0]:+.0f} dy={1000*off[1]:+.0f} dz={1000*off[2]:+.0f} mm ===')
# where would the fingers land if we followed the ACHIEVED axis down to the nail's z?
if abs(zax[2])>1e-3:
    s = (nail[2]-origin[2])/zax[2]
    landing = origin + s*zax
    print(f'  following achieved axis to nail-z, fingers land at xy=({landing[0]:.3f},{landing[1]:.3f}) vs nail xy=({nail[0]:.3f},{nail[1]:.3f})')
    print(f'  => to hit the nail, shift the aim by dx={1000*(nail[0]-landing[0]):+.0f} dy={1000*(nail[1]-landing[1]):+.0f} mm')
rclpy.shutdown()
