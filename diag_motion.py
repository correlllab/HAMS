#!/usr/bin/env python3
"""Instrument the grasp motion to find WHICH phase knocks the stud off, and to
trace the executed (OMPL) path. After reset, step through hover->center->descend->
close->lift; after each phase record the screw's world z (has it fallen?), the
gripper frame pose (the executed path), the finger midpoint, and save a head-cam
image. Prints a timeline so we can see exactly when screw z drops from 0.91."""
import sys, time, threading
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from h12_skills.grasp_benchmark import GRASP_FRAMES
import battery_bench as bb

SCREW = sys.argv[1] if len(sys.argv) > 1 else 'screw_27'
FLOOR = float(sys.argv[2]) if len(sys.argv) > 2 else -0.050
rclpy.init()
from h12_skills.grasp_benchmark import GraspBenchmark
bench = GraspBenchmark('right', gt_name='', box_source='gt')
ex = MultiThreadedExecutor(num_threads=6); ex.add_node(bench)
threading.Thread(target=ex.spin, daemon=True).start()
for _ in range(100):
    if bench._obj_poses.get('__pelvis__') is not None: break
    time.sleep(0.2)
frame = GRASP_FRAMES['right']

def sz():
    p = bench.gt_pos(SCREW)
    return round(float(p[2]), 4) if p is not None else None
def gp():
    p = bench.frame_pose_pelvis(frame)
    return None if p is None else [round(p.position.x,3), round(p.position.y,3), round(p.position.z,3)]
def fm():
    m = bb._finger_mid_xy(bench, 'right')
    return None if m is None else [round(float(m[0]),3), round(float(m[1]),3)]
def gap():
    lf = bench.frame_pose_pelvis('rg_left_finger'); rf = bench.frame_pose_pelvis('rg_right_finger')
    if lf is None or rf is None: return None
    d = ((lf.position.x-rf.position.x)**2+(lf.position.y-rf.position.y)**2+(lf.position.z-rf.position.z)**2)**0.5
    return round(d*1000,1)
def fingertip_z():
    c = bb._contact_point(bench, frame)
    return None if c is None else round(float(c[2])*1000,1)
def img(tag):
    try:
        import grab_img  # noqa
    except Exception:
        pass
try:
    from sensor_msgs.msg import Image  # for optional capture
except Exception:
    Image = None

log = []
def mark(tag):
    log.append((tag, sz(), gp(), fm()))
    print(f'[{tag:10s}] screw_z={sz()}  fingertip_z={fingertip_z()}mm  fingerMid={fm()}  finger_gap={gap()}mm', flush=True)

bb.reset_env(bench, 'right')
time.sleep(1.0)
mark('reset')

nail = bench.gt_pos_pelvis(SCREW)
_, tip = bb._screw_grasp_pose(bench, SCREW)

# 1) OMPL hover (the planned route) — sample the frame DURING the move to trace path
over = np.asarray(tip) + np.array([0.0, 0.0, bb.HOVER_M])
th = threading.Thread(target=lambda: bench.move_frame_to(frame, bb._top_down_pose(over),
                                                         duration_sec=bb.OVER_SEG_SEC, do_plan=bb.DO_PLAN))
th.start()
path = []
while th.is_alive():
    path.append((gp(), sz()))
    time.sleep(0.3)
th.join()
mark('hover')
print('  OMPL executed path (gripper xyz, screw_z):')
for pp in path:
    print('    ', pp)

# 2) center at safe height
cmd = np.array([nail[0]+bb.AIM_DX, nail[1]+bb.AIM_DY, FLOOR])
for it in range(3):
    bench.move_frame_to(frame, bb._top_down_pose(cmd), duration_sec=bb.SERVO_ITER_SEC, do_plan=False)
    m = bb._finger_mid_xy(bench, 'right')
    if m is not None:
        cmd[:2] = cmd[:2] - (m - nail[:2])
    mark(f'center{it}')

# 3) close
bench.close_gripper('right'); time.sleep(2.5)
g = bench._grip_last.get('right')
print('  grip_final_mm =', g[0] if g else None, ' driver-reported')
mark('close')

# 3b) small VERTICAL nudge up (2cm) — if held, the stud rises with the gripper
lifted = bench.frame_pose_pelvis(frame)
if lifted is not None:
    lifted.position.z += 0.02
    bench.move_frame_to(frame, lifted, duration_sec=3.0, do_plan=False)
    time.sleep(1.0)
mark('nudge+2cm')

# 4) full lift (straight up, slow, NO plan so it stays vertical)
lifted = bench.frame_pose_pelvis(frame)
if lifted is not None:
    lifted.position.z += bb.LIFT_DIST_M
    bench.move_frame_to(frame, lifted, duration_sec=6.0, do_plan=False)
    time.sleep(1.2)
mark('lift')

print('\n=== TIMELINE (screw z; 0.91=standing, <0.9=knocked) ===')
for tag, z, g_, m_ in log:
    flag = '' if (z is not None and z > 0.88) else '   <-- FELL'
    print(f'  {tag:10s} screw_z={z}{flag}')
rclpy.shutdown()
