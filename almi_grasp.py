#!/usr/bin/env python3
"""ONE ALMI standing grasp, in the order Adam specified: do ALL the big setup motion
FROZEN (home + OMPL hover over the nail), warm the ALMI policy up PINNED, THEN release
the pin and do only the SHORT grasp motion (descend->close->weld->lift) while standing.
Engaging first and homing after (my earlier bug) shifts the CoM under a cold policy and
topples it. Requires the lowerbody ALMI controller already running.

Env: STAB_SIM (pinned warmup sim-sec, default 40), same BATT_* knobs as battery_bench.
"""
import os, sys, time, threading, json
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool, Empty, String
from rosgraph_msgs.msg import Clock
from h12_skills.grasp_benchmark import GRASP_FRAMES
import battery_bench as bb

SCREW = sys.argv[1] if len(sys.argv) > 1 else 'screw_27'
STAB_SIM = float(os.environ.get('STAB_SIM', '40'))
rclpy.init()
from h12_skills.grasp_benchmark import GraspBenchmark
bench = GraspBenchmark('right', gt_name='', box_source='gt')
ex = MultiThreadedExecutor(num_threads=6); ex.add_node(bench)
threading.Thread(target=ex.spin, daemon=True).start()
frz = bench.create_publisher(Bool, '/hams/freeze_body', 10)
weld = bench.create_publisher(String, '/hams/grasp_weld', 10)
_sim = {'t': None}
bench.create_subscription(Clock, '/clock', lambda m: _sim.__setitem__('t', m.clock.sec + m.clock.nanosec*1e-9), 10)
for _ in range(100):
    if bench._obj_poses.get('__pelvis__') is not None and _sim['t'] is not None: break
    time.sleep(0.2)
frame = GRASP_FRAMES['right']
arm = 'right'
def upright():
    p = bench._obj_poses.get('__pelvis__'); return (p[2], 1-2*(p[4]**2+p[5]**2)) if p else (0,0)

rec = {'screw': SCREW}
# HOVER_FROZEN=1 (default): reach to the hover FROZEN, then warm up + release with the
# arm already extended (CoM forward -> ~1/3 solid releases). HOVER_FROZEN=0: keep the
# arm HOME through warmup + release (CoM centered -> solid release every time), then
# reach out STANDING with ALMI27 balancing the arm motion.
HOVER_FROZEN = os.environ.get('ALMI_HOVER_FROZEN', '1') != '0'
# --- 1+2) ENGAGE with RETRY (the sweep's fix): reset frozen -> warmup pinned -> release
#     -> settle; the release lands SOLID only ~1/3 of the time (inherent ALMI variance),
#     so retry until it settles solid, THEN grasp. Arm stays HOME through release; the
#     reach is done standing after. This is what made the fridge tier reliable.
MAX_ENGAGE = int(os.environ.get('ALMI_ENGAGE_RETRY', '8'))
SETTLE = float(os.environ.get('ALMI_RELEASE_SETTLE', '6.0'))
PZ_MIN = float(os.environ.get('ALMI_PELVIS_MIN', '0.955'))
engaged = False
for eng in range(MAX_ENGAGE):
    for _ in range(4):
        frz.publish(Bool(data=True)); time.sleep(0.05)         # (re)pin before restoring
    bb.reset_env(bench, arm)                                    # scene restore + arm home + gripper open (frozen)
    # NAV-RANDOM tier: teleport the PINNED base to spawn+(fwd,lat) before the warmup,
    # so ALMI warms up + releases at a randomized standing position and reaches the
    # fixed screw from a different relative pose. Gated by HAMS_PLACE_BASE="fwd,lat"
    # (metres); unset -> nominal spawn (standing tier unaffected).
    _pb = os.environ.get('HAMS_PLACE_BASE', '').strip()
    if _pb:
        from std_msgs.msg import Float64MultiArray as _F64
        if not hasattr(bench, '_pb_pub'):
            bench._pb_pub = bench.create_publisher(_F64, '/hams/place_base', 10)
        _f, _l = [float(v) for v in _pb.split(',')]
        for _ in range(8):                                      # burst (discovery race)
            bench._pb_pub.publish(_F64(data=[_f, _l])); time.sleep(0.1)
        time.sleep(1.5)
        print(f'[almi] nav-random: base placed at spawn+({_f:+.3f},{_l:+.3f})', flush=True)
    if HOVER_FROZEN:
        _, tipf = bb._screw_grasp_pose(bench, SCREW)
        overf = np.asarray(tipf) + np.array([0.0, 0.0, bb.HOVER_M])
        bench.move_frame_to(frame, bb._top_down_pose(overf), duration_sec=bb.OVER_SEG_SEC, do_plan=True)
    t0 = _sim['t']
    while _sim['t'] - t0 < STAB_SIM:
        time.sleep(0.2)
    z, u = upright()
    if u < 0.9:
        print(f'[almi] engage {eng}: not vertical pre-release; retry', flush=True); continue
    for _ in range(8):
        frz.publish(Bool(data=False)); time.sleep(0.1)
    time.sleep(1.0)
    z, u = upright(); print(f'[almi] engage {eng}: RELEASED pelvis_z={z:.3f} upright={u:.3f}', flush=True)
    t_s = _sim['t']
    while _sim['t'] - t_s < SETTLE:
        time.sleep(0.5)
    z, u = upright()
    if z > PZ_MIN and u > 0.95:
        print(f'[almi] engage {eng}: SOLID pelvis_z={z:.3f} -> grasp', flush=True)
        engaged = True; break
    print(f'[almi] engage {eng}: marginal pelvis_z={z:.3f} upright={u:.3f} -> retry', flush=True)
if not engaged:
    print(f'[almi] failed to engage solid in {MAX_ENGAGE} tries; abort', flush=True); rclpy.shutdown(); sys.exit(2)

# --- 3) STANDING: reach out (if not done frozen), then the grasp motion ---
nail = bench.gt_pos_pelvis(SCREW)
if not HOVER_FROZEN:
    # reach to the hover STANDING now, gently over a long move so ALMI27 can track the
    # forward CoM shift as the arm extends (this is the balance the wrist-inclusive
    # policy is for). Abort if it starts to go over.
    _, tip2 = bb._screw_grasp_pose(bench, SCREW)
    over2 = np.asarray(tip2) + np.array([0.0, 0.0, bb.HOVER_M])
    bench.move_frame_to(frame, bb._top_down_pose(over2), duration_sec=10.0, do_plan=True)
    z, u = upright(); print(f'[almi] standing hover set upright={u:.3f}', flush=True)
    if u < 0.6:
        print('[almi] toppled reaching to hover; abort'); rclpy.shutdown(); sys.exit(1)
    # let the base RE-SETTLE after the big reach before the down-descent (each motion
    # eats stability margin; pausing lets ALMI recover it so the deep step doesn't tip)
    t_r = _sim['t']
    while _sim['t'] - t_r < float(os.environ.get('ALMI_REACH_SETTLE', '5.0')):
        time.sleep(0.3)
    z, u = upright(); print(f'[almi] post-reach settle upright={u:.3f}', flush=True)
    if u < 0.85:
        print('[almi] unstable after reach; abort'); rclpy.shutdown(); sys.exit(1)
    nail = bench.gt_pos_pelvis(SCREW)
cf = float(nail[2]) + float(os.environ.get('BATT_GRIP_ABOVE_BASE', '0.068'))
cmd = np.array([nail[0], nail[1], cf])
MOVE = float(os.environ.get('ALMI_MOVE_SEC', '6.0'))   # SLOW moves — fast ones topple ALMI
# INCREMENTAL descent to the working depth: one big drop to the LOW nail shifts the CoM
# too fast and topples ALMI. Descend the contact in small steps, pausing so the policy
# re-stabilizes each time; abort if it starts to go over.
NDES = int(os.environ.get('ALMI_DESCEND_STEPS', '4'))
z_start_c = cf + bb.HOVER_M
for s in range(1, NDES + 1):
    zc = z_start_c + (cf - z_start_c) * (s / NDES)
    bench.move_frame_to(frame, bb._top_down_pose(np.array([nail[0], nail[1], zc])), duration_sec=MOVE, do_plan=False)
    z, u = upright()
    print(f'[almi] descend s{s}/{NDES} zc={zc:.3f} upright={u:.3f}', flush=True)
    if u < 0.6:
        print('[almi] toppling in descent; abort'); rclpy.shutdown(); sys.exit(1)
    t_d = _sim['t']                                          # settle between steps so ALMI recovers margin
    while _sim['t'] - t_d < float(os.environ.get('ALMI_DESCEND_SETTLE', '2.0')):
        time.sleep(0.2)
# Track the swaying base with SLOW gentle corrections (fast moves fall). Re-read the
# nail's LIVE pelvis pose + the true pad position each iter and correct until the pads
# are actually ON the nail (< nail radius) or the iter budget runs out. Log err + upright
# so we see whether it converges (sway-trackable) or plateaus (static torso offset).
NIT = int(os.environ.get('ALMI_SERVO_ITER', '4'))
CONV = float(os.environ.get('ALMI_CONV_MM', '7')) / 1000.0
for it in range(NIT):
    m = bb._pad_mid_xy(bench, arm)
    n = bench.gt_pos_pelvis(SCREW)
    if m is None: break
    exy = m - n[:2]
    z, u = upright()
    print(f'[almi] servo it{it} err={np.linalg.norm(exy)*1000:.1f}mm upright={u:.3f}', flush=True)
    if u < 0.6:
        print('[almi] falling mid-servo; abort'); break
    if np.linalg.norm(exy) < CONV:
        break
    cmd[:2] = cmd[:2] - exy
    bench.move_frame_to(frame, bb._top_down_pose(cmd), duration_sec=MOVE, do_plan=False)
bench.close_gripper(arm, force_n=float(os.environ.get('BATT_GRIP_FORCE_N', '6')))
time.sleep(3.0)
g = bench._grip_last.get(arm); rec['grip_final_mm'] = g[0] if g else None
pad = bb._pad_mid_xy(bench, arm); nn = bench.gt_pos_pelvis(SCREW)
rec['post_grasp_err_mm'] = round(float(np.hypot(pad[0]-nn[0], pad[1]-nn[1])*1000), 1) if (pad is not None) else 999
rec['good_grasp'] = bool(rec['grip_final_mm'] and 40 < rec['grip_final_mm'] < 78 and rec['post_grasp_err_mm'] < float(os.environ.get('BATT_WELD_MAX_ERR_MM', '12')) and (g[1] if g else 0) > 40)
if rec['good_grasp']:
    for _ in range(6): weld.publish(String(data=f'{SCREW}:{arm}')); time.sleep(0.08)
    rec['welded'] = True
    time.sleep(0.5)
z0 = bench.gt_pos(SCREW)
lp = bench.frame_pose_pelvis(frame); lp.position.z += 0.10
bench.move_frame_to(frame, lp, duration_sec=6.0, do_plan=False)
time.sleep(2.0)
z1 = bench.gt_pos(SCREW); zp, up_ = upright()
rec['screw_lift_dz'] = round(float(z1[2]-z0[2]), 4)
rec['standing_after'] = bool(zp > 0.85 and up_ > 0.85)
rec['success'] = bool(rec['good_grasp'] and rec['screw_lift_dz'] >= 0.03 and rec['standing_after'])
print('[almi] RESULT ' + json.dumps({k: rec.get(k) for k in ['grip_final_mm','post_grasp_err_mm','good_grasp','welded','screw_lift_dz','standing_after','success']}), flush=True)
if os.environ.get('OUT'): open(os.environ['OUT'], 'w').write(json.dumps(rec))
rclpy.shutdown()
