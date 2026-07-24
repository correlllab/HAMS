#!/usr/bin/env python3
"""Reset, hover the gripper at a safe height over the nail (closed-loop centered),
then CLOSE and measure where the finger tips actually end up — so we set the hover
height once so the CLOSED fingers land on the 5mm nail head. Never go below floor."""
import sys, time, threading
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from h12_skills.grasp_benchmark import GRASP_FRAMES
import battery_bench as bb

SCREW = sys.argv[1] if len(sys.argv) > 1 else 'screw_27'
FLOOR = float(sys.argv[2]) if len(sys.argv) > 2 else -0.085
rclpy.init()
from h12_skills.grasp_benchmark import GraspBenchmark
bench = GraspBenchmark('right', gt_name='', box_source='gt')
ex = MultiThreadedExecutor(num_threads=6); ex.add_node(bench)
threading.Thread(target=ex.spin, daemon=True).start()
for _ in range(100):
    if bench._obj_poses.get('__pelvis__') is not None: break
    time.sleep(0.2)
frame = GRASP_FRAMES['right']

def P(n):
    p = bench.frame_pose_pelvis(n)
    return None if p is None else np.array([p.position.x, p.position.y, p.position.z])
def tips():
    lf, rf = P('rg_left_finger'), P('rg_right_finger')
    return lf, rf

bb.reset_env(bench, 'right')
nail = bench.gt_pos_pelvis(SCREW)
print(f'nail: {np.round(nail,4).tolist()}  (head top ~ {nail[2]+0.005:.3f})')

# closed-loop to (nail_xy, FLOOR)
over = np.array([nail[0]+bb.AIM_DX, nail[1]+bb.AIM_DY, nail[2]+0.12])
bench.move_frame_to(frame, bb._top_down_pose(over), duration_sec=6.0, do_plan=bb.DO_PLAN)
cmd = np.array([nail[0]+bb.AIM_DX, nail[1]+bb.AIM_DY, FLOOR])
for it in range(8):
    bench.move_frame_to(frame, bb._top_down_pose(cmd), duration_sec=5.0, do_plan=False)
    c = bb._contact_point(bench, frame); n = bench.gt_pos_pelvis(SCREW)
    e = np.array([c[0]-n[0], c[1]-n[1], c[2]-FLOOR])
    if np.linalg.norm(e[:2]) < 0.006 and abs(e[2]) < 0.006: break
    cmd = cmd - e; cmd[2] = min(cmd[2], FLOOR+0.06)

c = bb._contact_point(bench, frame); lf, rf = tips()
print('=== OPEN (settled) ===')
print(f'  contact(TCP)  : {np.round(c,4).tolist()}')
print(f'  rg_left_finger: {np.round(lf,4).tolist() if lf is not None else None}')
print(f'  rg_right_finger:{np.round(rf,4).tolist() if rf is not None else None}')
print(f'  lowest finger Z: {min(lf[2],rf[2]):.4f}  vs nail head top {nail[2]+0.005:.4f}')

bench.close_gripper('right'); time.sleep(2.0)
c2 = bb._contact_point(bench, frame); lf2, rf2 = tips()
print('=== CLOSED ===')
print(f'  contact(TCP)  : {np.round(c2,4).tolist()}')
print(f'  rg_left_finger: {np.round(lf2,4).tolist() if lf2 is not None else None}')
print(f'  rg_right_finger:{np.round(rf2,4).tolist() if rf2 is not None else None}')
print(f'  lowest finger Z: {min(lf2[2],rf2[2]):.4f}')
if lf is not None and lf2 is not None:
    print(f'  CLOSE-DROP lowest finger: {1000*(min(lf2[2],rf2[2])-min(lf[2],rf[2])):+.0f} mm')
    print(f'  finger gap open={np.linalg.norm(lf-rf)*1000:.0f}mm closed={np.linalg.norm(lf2-rf2)*1000:.0f}mm')
z1 = bench.gt_pos(SCREW)
print(f'  screw world z now: {z1[2]:.4f}')
rclpy.shutdown()
