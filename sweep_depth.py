#!/usr/bin/env python3
"""Sweep the grip DEPTH to find where the pads actually clamp the screw. For each
floor: reset, hover, center the finger-mid on the screw at that floor, close, read
the driver-reported grip, then raise the gripper 4cm and check whether the screw
ROSE with it (screw_dz>0 => really held). Prints a table so we pick the depth that
grips + lifts. The TF finger frames are static mounts (useless for gap), so 'held'
is judged by the screw following the gripper."""
import sys, time, threading
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from h12_skills.grasp_benchmark import GRASP_FRAMES
import battery_bench as bb

SCREW = 'screw_27'
FLOORS = [float(x) for x in (sys.argv[1:] or ['-0.045', '-0.052', '-0.060', '-0.068'])]
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
    p = bench.gt_pos(SCREW); return float(p[2]) if p is not None else None
def tcpz():
    c = bb._contact_point(bench, frame); return round(float(c[2])*1000,1) if c is not None else None

rows = []
for FLOOR in FLOORS:
    bb.reset_env(bench, 'right'); time.sleep(0.8)
    nail = bench.gt_pos_pelvis(SCREW)
    _, tip = bb._screw_grasp_pose(bench, SCREW)
    over = np.asarray(tip) + np.array([0.0, 0.0, bb.HOVER_M])
    bench.move_frame_to(frame, bb._top_down_pose(over), duration_sec=bb.OVER_SEG_SEC, do_plan=bb.DO_PLAN)
    cmd = np.array([nail[0]+bb.AIM_DX, nail[1]+bb.AIM_DY, FLOOR])
    for it in range(4):
        bench.move_frame_to(frame, bb._top_down_pose(cmd), duration_sec=bb.SERVO_ITER_SEC, do_plan=False)
        m = bb._finger_mid_xy(bench, 'right')
        if m is not None: cmd[:2] = cmd[:2] - (m - nail[:2])
    tcp_at_grip = tcpz()
    z_before = sz()
    bench.close_gripper('right'); time.sleep(2.2)
    g = bench._grip_last.get('right'); grip = g[0] if g else None
    z_closed = sz()
    # raise 4cm straight up
    lp = bench.frame_pose_pelvis(frame)
    if lp is not None:
        lp.position.z += 0.04
        bench.move_frame_to(frame, lp, duration_sec=4.0, do_plan=False); time.sleep(1.0)
    z_lift = sz()
    dz = (z_lift - z_before) if (z_lift is not None and z_before is not None) else None
    held = (dz is not None and dz > 0.01)
    rows.append((FLOOR, tcp_at_grip, grip, round(z_before,3) if z_before else None,
                 round(z_closed,3) if z_closed else None, round(z_lift,3) if z_lift else None,
                 round(dz,3) if dz is not None else None, held))
    print(f'FLOOR={FLOOR:+.3f}  TCPz={tcp_at_grip}mm  grip={grip}mm  '
          f'z: {z_before:.3f}->{z_closed:.3f}->{z_lift:.3f}  dz={dz:+.3f}  HELD={held}', flush=True)

print('\n=== SWEEP SUMMARY ===')
print(f'{"floor":>7} {"TCPz":>7} {"grip":>6} {"z0":>6} {"zclose":>7} {"zlift":>6} {"dz":>7} held')
for r in rows:
    print(f'{r[0]:>7.3f} {str(r[1]):>7} {str(r[2]):>6} {str(r[3]):>6} {str(r[4]):>7} {str(r[5]):>6} {str(r[6]):>7} {r[7]}')
winners = [r for r in rows if r[7]]
print('\nWINNING depths (screw rose >1cm):', [r[0] for r in winners] or 'NONE')
rclpy.shutdown()
