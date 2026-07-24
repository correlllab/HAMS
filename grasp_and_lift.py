#!/usr/bin/env python3
"""Adam's recipe: position ONCE (center), close, WAIT for the grasp to complete,
then a clean straight lift with NO more servoing. Reports whether the screw rose."""
import sys, time, threading
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from h12_skills.grasp_benchmark import GRASP_FRAMES
import battery_bench as bb

SCREW = 'screw_27'
FLOOR = float(sys.argv[1]) if len(sys.argv) > 1 else -0.052
WAIT  = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
LIFT  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
rclpy.init()
from h12_skills.grasp_benchmark import GraspBenchmark
bench = GraspBenchmark('right', gt_name='', box_source='gt')
ex = MultiThreadedExecutor(num_threads=6); ex.add_node(bench)
threading.Thread(target=ex.spin, daemon=True).start()
for _ in range(100):
    if bench._obj_poses.get('__pelvis__') is not None: break
    time.sleep(0.2)
frame = GRASP_FRAMES['right']
def sz(): p = bench.gt_pos(SCREW); return float(p[2]) if p is not None else None

bb.reset_env(bench, 'right'); time.sleep(0.8)
nail = bench.gt_pos_pelvis(SCREW)
_, tip = bb._screw_grasp_pose(bench, SCREW)
over = np.asarray(tip) + np.array([0.0, 0.0, bb.HOVER_M])
bench.move_frame_to(frame, bb._top_down_pose(over), duration_sec=bb.OVER_SEG_SEC, do_plan=bb.DO_PLAN)
# position: converge the finger-mid onto the screw (this is the POSITIONING servo,
# before the grasp — not the post-grasp jostle Adam warned about)
cmd = np.array([nail[0]+bb.AIM_DX, nail[1]+bb.AIM_DY, FLOOR])
for it in range(4):
    bench.move_frame_to(frame, bb._top_down_pose(cmd), duration_sec=bb.SERVO_ITER_SEC, do_plan=False)
    m = bb._finger_mid_xy(bench, 'right')
    if m is not None: cmd[:2] = cmd[:2] - (m - nail[:2])
c = bb._contact_point(bench, frame)
print(f'positioned: TCPz={c[2]*1000:.0f}mm fingerMid={np.round(bb._finger_mid_xy(bench,"right"),4).tolist()} nail={np.round(nail[:2],4).tolist()}', flush=True)

# GRASP — close and WAIT for it to complete (no servoing after this)
bench.close_gripper('right')
print(f'closing, waiting {WAIT}s for grasp to complete...', flush=True)
time.sleep(WAIT)
g = bench._grip_last.get('right')
z_grasped = sz()
print(f'GRASPED grip={g[0] if g else None}mm force={g[1] if g else None}N screw_z={z_grasped:.4f}', flush=True)

# LIFT — one clean straight-up move, no plan, no re-servo
lp = bench.frame_pose_pelvis(frame)
lp.position.z += LIFT
bench.move_frame_to(frame, lp, duration_sec=6.0, do_plan=False)
time.sleep(2.0)
z_lift = sz()
dz = z_lift - z_grasped
print(f'LIFTED screw_z={z_lift:.4f}  dz={dz:+.4f}  HELD={"YES" if dz>0.03 else ("partial" if dz>0.005 else "NO")}', flush=True)
time.sleep(6.0)   # hold so a head-cam grab can capture the lifted state
rclpy.shutdown()
