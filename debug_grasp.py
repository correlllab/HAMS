#!/usr/bin/env python3
"""Debug the battery grasp WITH head-cam captures at every phase, so we can SEE
where it hits the table / misses. Mirrors battery_bench.run's vertical-descent
motion but saves a labelled PNG after each step. Servo (GT re-read) by default."""
import sys, os, time, threading, json
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
import cv2

from h12_skills.grasp_benchmark import GRASP_FRAMES
import battery_bench as bb

SCREW = sys.argv[1] if len(sys.argv) > 1 else 'screw_27'
METHOD = sys.argv[2] if len(sys.argv) > 2 else 'servo'
OUTDIR = '/tmp/gcap'
os.makedirs(OUTDIR, exist_ok=True)

rclpy.init()
from h12_skills.grasp_benchmark import GraspBenchmark
bench = GraspBenchmark('right', gt_name='', box_source='gt')

# head-cam subscriber on the same node
_img = {'f': None}
def _on_img(msg):
    arr = np.frombuffer(msg.data, np.uint8)
    _img['f'] = cv2.imdecode(arr, cv2.IMREAD_COLOR)
bench.create_subscription(CompressedImage,
    '/realsense/head/color/image_raw/compressed', _on_img, qos_profile_sensor_data)

ex = MultiThreadedExecutor(num_threads=6)
ex.add_node(bench)
threading.Thread(target=ex.spin, daemon=True).start()

for _ in range(100):
    if bench._obj_poses.get('__pelvis__') is not None:
        break
    time.sleep(0.2)

frame = GRASP_FRAMES['right']
log = []
def snap(tag):
    time.sleep(0.6)
    f = _img['f']
    err = bb._frame_err_to(bench, frame, tip)
    fp = bench.frame_pose_pelvis(frame)
    o = [round(fp.position.x,3), round(fp.position.y,3), round(fp.position.z,3)] if fp else None
    log.append(dict(tag=tag, err_mm=round(err*1000,1) if err else None, frame_xyz=o))
    if f is not None:
        cv2.imwrite(f'{OUTDIR}/{tag}.png', f)
    print(f'[{tag}] err={err*1000 if err else -1:.1f}mm frame={o}', flush=True)

# --- reset to original ---
bb.reset_env(bench, 'right')
pose, tip = bb._screw_grasp_pose(bench, SCREW)
snap('00_reset')

# --- 1. OVER (tuck via OMPL) ---
over = np.asarray(tip) + np.array([0.0, 0.0, bb.HOVER_M])
bench.move_frame_to(frame, bb._top_down_pose(over), duration_sec=bb.OVER_SEG_SEC, do_plan=bb.DO_PLAN)
snap('01_hover')

# --- 2. pure-vertical descent, capture each step ---
N = 6
reread = (METHOD == 'servo')
base = None
for k in range(1, N+1):
    if reread:
        _, t = bb._screw_grasp_pose(bench, SCREW)
    else:
        t = base if base is not None else tip
    t = np.asarray(t)
    if not reread and base is None:
        base = t
    h = bb.HOVER_M * (1.0 - k/float(N))
    desc = bb._top_down_pose(t + np.array([0.0, 0.0, h]))
    bench.move_frame_to(frame, desc, duration_sec=bb.SERVO_ITER_SEC, do_plan=False)
    snap(f'02_desc{k}')

# --- 3. close + lift ---
snap('03_precontact')
bench.close_gripper('right')
time.sleep(1.2)
snap('04_closed')
lifted = bench.frame_pose_pelvis(frame)
if lifted is not None:
    lifted.position.z += bb.LIFT_DIST_M
    bench.move_frame_to(frame, lifted, duration_sec=4.0, do_plan=bb.DO_PLAN)
    time.sleep(1.2)
snap('05_lifted')

z1 = bench.gt_pos(SCREW)
print('=== summary ===', flush=True)
print(json.dumps(log, indent=1), flush=True)
print('screw z after:', round(float(z1[2]),4) if z1 is not None else None, flush=True)
rclpy.shutdown()
