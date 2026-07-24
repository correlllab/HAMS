#!/usr/bin/env python3
"""Battery-workcell screw pickup: SERVO vs NO-SERVO, ground-truth targeted.

The paper's battery comparison. In SIM, YOLO/cameras are the real-robot path;
here we CHEAT with the screw's ground-truth site pose (GraspBenchmark.
gt_pos_pelvis reads the sim's /robocasa/object_poses), so perception is exact
and out of scope. The two methods differ ONLY in how the arm is driven to that
GT target:

  noservo  open loop: resolve the screw's pelvis-frame pose ONCE, command the
           approach + contact, close. If the base moves after the command, the
           arm goes to a now-stale pose.
  servo    closed loop, UPDATED AT AN INTERVAL: re-read the screw's live GT
           pose in the CURRENT pelvis frame every iteration, re-command, and
           stop once the driven gripper frame is within tol of the screw.
           On a swaying/offset base this tracks the true relative target.

This is plain position servoing to a GT target (not visual servoing), so it is
meaningful exactly where the base is not static: frozen ~ equal, standing /
randomized is where the interval re-read compensates base drift and open-loop
misses. Both drive the arm DIRECTLY via frame_task IK (no deployed-skill
detection), so it does not depend on YOLO, SAM, or the skill's prep step.

Usage (inside hams_ros, sim running BatteryWorkcell):
  python3 battery_bench.py --method servo   --arm right --screw auto --out t.json
  python3 battery_bench.py --method noservo --arm right --screw screw_08 --out t.json
Grading: --success-mode lift (default) = the targeted screw's GT z rises
success-dz after the grasp+lift; contact = gripper settles in the hold band.
"""
import argparse
import json
import os
import threading
import time

import math

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Pose

# Reuse the benchmark node (GT feed, gripper tracking, frame driving, helpers).
from h12_skills.grasp_benchmark import (
    GraspBenchmark, GRASP_FRAMES, _offset_along_z, APPROACH_DIST_M, TCP_DEPTH_M,
)
from h12_skills.perception_utils import matrix_to_pose

# DEPLOYED top-down strategy (the real robot's battery grasp): approach pitched
# TOP_DOWN_PITCH_DEG below horizontal (NOT vertical — main's value keeps the wrist
# reachable), fingers along pelvis +Y, base one contact-length back so the finger
# CONTACT lands on the target. Verbatim from main skills/grasp.py::_top_down_pose.
TOP_DOWN_PITCH_DEG = float(os.environ.get('BATT_TOPDOWN_PITCH', '80.0'))
GRIPPER_BASE_TO_CONTACT_M = TCP_DEPTH_M   # 0.1146


def _top_down_pose(contact_xyz):
    pitch = math.radians(TOP_DOWN_PITCH_DEG)
    z_axis = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
    x_axis = np.array([0.0, 1.0, 0.0])
    y_axis = np.cross(z_axis, x_axis)
    T = np.eye(4)
    T[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    T[:3, 3] = np.asarray(contact_xyz, float) - GRIPPER_BASE_TO_CONTACT_M * z_axis
    return matrix_to_pose(T)

# Deployed servo/approach config, matched to main skills/grasp.py exactly:
# SERVO_MAX_ITER=6, SERVO_LIN_TOL=0.025 (25 mm best-effort — the 80deg top-down
# is steeper than the arm's ~45deg practical reach, so it converges best-effort,
# NOT tightly), long approach budget so a planned transit is never cut off. The
# contact move stops CONTACT_OFFSET short of the grasp, then closes.
SERVO_MAX_ITER = int(os.environ.get('BATT_SERVO_ITER', '6'))
SERVO_LIN_TOL = float(os.environ.get('BATT_SERVO_LIN_TOL', '0.025'))   # m (deployed value)
APPROACH_SEC = float(os.environ.get('BATT_APPROACH_SEC', '20.0'))
CONTACT_SEC = float(os.environ.get('BATT_CONTACT_SEC', '12.0'))
SERVO_ITER_SEC = float(os.environ.get('BATT_SERVO_ITER_SEC', '6.0'))   # per-iter translate budget
FINGER_SINK_M = float(os.environ.get('BATT_FINGER_SINK', '-0.030'))  # aim the OPEN pinch at the peg TOP (~30mm above base). The fingertips sit BELOW the pinch, so at the peg top they stay ABOVE the table; closing arcs them DOWN onto the standing peg. Aiming lower drove the fingertips into the table
GRASP_OFFSET_M = float(os.environ.get('BATT_GRASP_OFFSET', '0.0'))
CONTACT_OFFSET_M = float(os.environ.get('BATT_CONTACT_OFFSET', '0.025'))  # stop short (deployed 0.25*APPROACH)
OPEN_MM = float(os.environ.get('BATT_OPEN_MM', '106.0'))               # pre-open wide
# OMPL planning (the real robot's path): the planner finds reachable collision-
# free configs that the local diff-IK misses. ON by default; BATT_PLAN=0 for
# raw-IK ablation. Both methods use it, so it is not the servo/noservo variable.
DO_PLAN = os.environ.get('BATT_PLAN', '1').strip().lower() not in ('0', 'off', 'false', 'no')
LIFT_DIST_M = 0.12
CONTACT_MIN_MM, CONTACT_MAX_MM = 3.0, 95.0   # a real screw hold sits between
CLOSE_GATE_MM = float(os.environ.get('BATT_CLOSE_GATE_MM', '45.0'))  # only close if within this of the screw
HOVER_M = float(os.environ.get('BATT_HOVER_M', '0.10'))        # hover height above the screw
LIFT_SEG_SEC = float(os.environ.get('BATT_LIFT_SEC', '4.0'))   # lift segment
OVER_SEG_SEC = float(os.environ.get('BATT_OVER_SEC', '6.0'))   # traverse segment
BIAS_GAIN = float(os.environ.get('BATT_BIAS_GAIN', '0.9'))     # integral bias toward the nail


# The arm can't hold a perfect vertical (wrist clamp) — it achieves ~15deg off
# vertical, so the tilted fingertip lands FORWARD+SIDE of straight-down. Shift the
# aim by the measured sin/cos offset so the tilted tip lands ON the screw.
AIM_DX = float(os.environ.get('BATT_AIM_DX', '0.0'))
AIM_DY = float(os.environ.get('BATT_AIM_DY', '0.0'))


def _screw_grasp_pose(bench, screw):
    """DEPLOYED top-down grasp Pose (graspgenx frame, PELVIS) at the screw's live
    GT, shifted by the achieved-angle offset so the tilted fingertip closes ON the
    screw. Re-reads gt_pos_pelvis each call so the servo loop tracks the CURRENT
    relative target."""
    p = bench.gt_pos_pelvis(screw)
    if p is None:
        return None, None
    tip = np.array([p[0] + AIM_DX, p[1] + AIM_DY, p[2] - FINGER_SINK_M])
    return _offset_along_z(_top_down_pose(tip), GRASP_OFFSET_M), tip


def _approach_axis(pose):
    """+Z (approach) axis of a Pose, in pelvis."""
    from h12_skills.perception_utils import pose_to_matrix
    return pose_to_matrix(pose)[:3, 2]


def _translate_contact_to(bench, frame, target_contact):
    """PURE-TRANSLATION pose that keeps the gripper's CURRENT (wrist-achievable)
    orientation but shifts it so its CONTACT point lands on `target_contact`.
    This is what the eye-in-hand camera does in the deployed grasp (aligns in
    pure translation), realised here from GT — decoupling the alignment from the
    steep top-down orientation the sim wrist can't fully reach."""
    from geometry_msgs.msg import Pose
    from h12_skills.perception_utils import pose_to_matrix
    fp = bench.frame_pose_pelvis(frame)
    if fp is None:
        return None
    zax = pose_to_matrix(fp)[:3, 2]
    new = np.asarray(target_contact) - zax * TCP_DEPTH_M     # base so contact = target
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(new[0]), float(new[1]), float(new[2]))
    p.orientation = fp.orientation                            # keep achievable orientation
    return p


def _finger_mid_xy(bench, arm='right'):
    """Midpoint of the two finger links in pelvis XY — the ACTUAL point the fingers
    close around (NOT the TCP/graspgenx marker, which sits forward of it at a tilt).
    This is what must be centred on the nail, per Adam's 'tops of the fingers'."""
    pre = 'rg' if arm == 'right' else 'lg'
    lf = bench.frame_pose_pelvis(f'{pre}_left_finger')
    rf = bench.frame_pose_pelvis(f'{pre}_right_finger')
    if lf is None or rf is None:
        return None
    return np.array([0.5 * (lf.position.x + rf.position.x),
                     0.5 * (lf.position.y + rf.position.y)])


def _contact_point(bench, frame):
    """The driven gripper CONTACT point in pelvis (frame origin + TCP_DEPTH along
    +Z). We know this EXACTLY from TF — it is where the fingers will close."""
    from h12_skills.perception_utils import pose_to_matrix
    fp = bench.frame_pose_pelvis(frame)
    if fp is None:
        return None
    T = pose_to_matrix(fp)
    return np.array([fp.position.x, fp.position.y, fp.position.z]) + T[:3, 2] * TCP_DEPTH_M


def _frame_err_to(bench, frame, tip_pelvis):
    """Linear error [m] between the driven gripper CONTACT point and the target."""
    c = _contact_point(bench, frame)
    return None if c is None else float(np.linalg.norm(c - np.asarray(tip_pelvis)))


def _pick_screw(bench, arm):
    """Closest reachable screw to the arm's shoulder side (pelvis frame). Prefer
    screws on the arm's own side (right arm -> pelvis -y) and within a reach ball."""
    best, best_d = None, 1e9
    side = -1.0 if arm == 'right' else 1.0     # right arm favours pelvis -Y
    for name in sorted(k for k in bench._obj_poses if k.startswith('screw')):
        p = bench.gt_pos_pelvis(name)
        if p is None:
            continue
        # reach heuristic: in front (x>0.15), not too far, same side, near shoulder h
        if p[0] < 0.15 or p[0] > 0.62 or (side * p[1]) < -0.10:
            continue
        d = float(np.hypot(p[0] - 0.35, p[1] - side * 0.20)) + 0.5 * abs(p[2] + 0.15)
        if d < best_d:
            best, best_d = name, d
    return best


def run(bench, method, screw, arm, success_dz):
    frame = GRASP_FRAMES[arm]
    rec = dict(method=method, screw=screw, arm=arm, success=False, executed=False)
    bench.set_gripper(arm, OPEN_MM)
    time.sleep(0.5)
    z0 = bench.gt_pos(screw)
    rec['screw_z0'] = round(float(z0[2]), 4) if z0 is not None else None

    pose, tip = _screw_grasp_pose(bench, screw)
    if pose is None:
        rec['error'] = 'no GT for screw'
        return rec

    # --- MOTION: STOP GOING LOWER (Adam's rule). The gripper FINGERS extend below
    #     the pinch and arc DOWN as they close, so descending onto the nail drives
    #     the fingertips into the TABLE. Instead: 1) OMPL hover directly above the
    #     nail (arm routes over, never sweeps in low). 2) ONE move that positions
    #     the OPEN pinch at the peg TOP — high enough that the fingertips stay
    #     ABOVE the table. 3) close: the fingers drop onto the standing peg and
    #     grab it by the edge. No stepping, no descent, no forward drift. ---
    # 1) LIFT + route over the nail (planned hover). 2) straight-down keep-best
    #    descent (direct diff-IK). RECORD the full trajectory so we can see exactly
    #    where it stalls / when the contact dips to the hard table (world z=0.887 =>
    #    pelvis ~ -0.143). Every step logs err, gripper-frame z, and contact xyz.
    rec['traj'] = []
    # CONTACT_FLOOR_Z = target fingertip HEIGHT (above the screw head). The close
    # then lengthens the reach and drops onto the nail from here. NEVER go below it.
    over_tip = np.asarray(tip) + np.array([0.0, 0.0, HOVER_M])
    bench.move_frame_to(frame, _top_down_pose(over_tip),
                        duration_sec=OVER_SEG_SEC, do_plan=DO_PLAN)   # lift + route over (planner)
    # POSITION -> GRASP -> LIFT (Adam's recipe, the motion that actually holds):
    #   1) position ONCE at the working depth. servo: converge the finger-MIDPOINT onto
    #      the screw's LIVE GT (closed loop — this is what corrects base drift). noservo:
    #      command the pose resolved ONCE at the start (open loop, no correction).
    #   2) close GENTLY and WAIT for the grasp to complete.
    #   3) lift STRAIGHT UP with NO further servoing — re-nudging after the grasp, or a
    #      too-hard squeeze, forces the light part back out.
    # Two corrections, not four: enough to center to ~1-2mm (each step cuts the error
    # to ~a third), which grips reliably, without the visible "go there, re-nudge 4x"
    # hesitation. One was too few when the first move landed far off. noservo=0 (open).
    CORR_ITERS = int(os.environ.get('BATT_SERVO_ITER', '2')) if method == 'servo' else 0
    WAIT_GRASP = float(os.environ.get('BATT_WAIT_GRASP', '6.0'))
    SETTLE_SEC = float(os.environ.get('BATT_SETTLE_SEC', '2.0'))    # arm rest before close
    # DON'T close all the way. The sim's /close drives the jaw to FULLY closed, so the
    # grip force is kp*(large error) ~ 580 N and ejects a light part. Instead command
    # the jaw to a target aperture just PAST the object width: it presses with a bounded
    # force kp*(small error), a controlled grip that holds without flinging. A well-
    # centered grasp sticks; a miss reaches the commanded aperture on air (no object,
    # ~0 force) and drops on the lift — exactly the clean pass/fail the experiment wants.
    GRIP_N = float(os.environ.get('BATT_GRIP_FORCE_N', '6.0'))
    GRASP_TARGET_MM = float(os.environ.get('BATT_GRASP_TARGET_MM', '38.0'))  # jaw closes TO here
    GRIP_ABOVE_BASE_M = float(os.environ.get('BATT_GRIP_ABOVE_BASE', '0.068'))
    nail0 = bench.gt_pos_pelvis(screw)                            # the pose noservo commits to
    # DECIDE how low to go from the screw's OWN GT height (generalises across screw
    # heights, and to real screws of different lengths): command the fingertip toward
    # the screw top (GT base z + offset); the TCP->command offset then lands the pads
    # on the screw body. BATT_CONTACT_FLOOR overrides with an absolute pelvis z.
    _cf = os.environ.get('BATT_CONTACT_FLOOR', 'auto')
    CONTACT_FLOOR_Z = (float(nail0[2]) + GRIP_ABOVE_BASE_M) if _cf == 'auto' else float(_cf)
    rec['contact_floor_z'] = round(CONTACT_FLOOR_Z, 4)
    cmd = np.array([nail0[0] + AIM_DX, nail0[1] + AIM_DY, CONTACT_FLOOR_Z])
    bench.move_frame_to(frame, _top_down_pose(cmd), duration_sec=SERVO_ITER_SEC, do_plan=False)
    for it in range(CORR_ITERS):
        nail_live = bench.gt_pos_pelvis(screw)                   # servo re-reads live GT (tracks drift)
        m = _finger_mid_xy(bench, arm)
        if m is None:
            break
        exy = m - nail_live[:2]
        cmd[:2] = cmd[:2] - exy                                  # correct the command by the measured miss
        bench.move_frame_to(frame, _top_down_pose(cmd), duration_sec=SERVO_ITER_SEC, do_plan=False)
        rec['traj'].append(dict(it=it, errXY_mm=round(float(np.linalg.norm(exy)) * 1000, 1)))
        bench.get_logger().info(f'[batt {method}] it{it}: errXY={np.linalg.norm(exy)*1000:.1f}mm')
    _fm = _finger_mid_xy(bench, arm)
    _n = bench.gt_pos_pelvis(screw)
    rec['final_err_mm'] = round(float(np.hypot(_fm[0] - _n[0], _fm[1] - _n[1])) * 1000, 1) \
        if (_fm is not None and _n is not None) else 999.0
    rec['approach_ok'] = rec['final_err_mm'] < CLOSE_GATE_MM
    rec['servo_iters'] = CORR_ITERS
    # SETTLE: let the arm come fully to rest at the grasp pose, then DO NOT command it
    # again until the lift — so it holds dead still while the gripper closes (no
    # fidgeting during the grasp). The lift below is the only post-position arm motion.
    time.sleep(SETTLE_SEC)
    # GRASP: the sim jaw is a position drive — a partial target stalls in mid-air before
    # it reaches the object, so it MUST be commanded fully closed; the force is bounded
    # by kp (set from GRIP_N). 6 N -> ~240 N at the pads: reliably holds, well under the
    # 580 N (15 N) that ejects. WAIT for the grasp to complete. NO servoing after this.
    rec['executed'] = True
    bench.close_gripper(arm, force_n=GRIP_N)
    time.sleep(WAIT_GRASP)
    g = bench._grip_last.get(arm)
    rec['grip_final_mm'] = g[0] if g else None
    rec['grip_force_n'] = round(float(g[1]), 1) if g else None
    # CLOSED-ON-OBJECT tally: the jaw came to rest in the OBJECT-CONTACT band — wider
    # than a closed-on-air miss (~35 mm) but clearly closed (not stalled near-open).
    # A miss closes to ~35 mm; a stalled/weak close sits near-open (~90 mm); a real
    # clamp on the ~36 mm box lands ~50-65 mm. Tracked separately from lift success.
    OBJ_MIN_MM = float(os.environ.get('BATT_OBJ_MIN_MM', '40.0'))
    OBJ_MAX_MM = float(os.environ.get('BATT_OBJ_MAX_MM', '78.0'))
    rec['closed_on_object'] = bool(rec['grip_final_mm'] is not None
                                   and OBJ_MIN_MM < rec['grip_final_mm'] < OBJ_MAX_MM)
    z_grasped = bench.gt_pos(screw)
    rec['screw_z_grasped'] = round(float(z_grasped[2]), 4) if z_grasped is not None else None
    # LIFT: one clean straight-up move, no plan, no re-servo.
    lp = bench.frame_pose_pelvis(frame)
    if lp is not None:
        lp.position.z += LIFT_DIST_M
        bench.move_frame_to(frame, lp, duration_sec=max(LIFT_SEG_SEC, 6.0), do_plan=False)
        time.sleep(1.5)
    z1 = bench.gt_pos(screw)
    rec['screw_z1'] = round(float(z1[2]), 4) if z1 is not None else None
    dz = (float(z1[2]) - float(z0[2])) if (z0 is not None and z1 is not None) else 0.0
    rec['screw_lift_dz'] = round(dz, 4)
    in_band = (rec['grip_final_mm'] is not None
               and CONTACT_MIN_MM < rec['grip_final_mm'] < CONTACT_MAX_MM)
    rec['success'] = bool(rec['executed'] and dz >= success_dz and in_band)
    return rec


def reset_env(bench, arm):
    """Clean start EVERY trial: restore the scene (screws + robot back to spawn
    under the freeze pin) and home the arm, so a test never begins from where the
    previous grasp left the arm stuck or the screws displaced."""
    from std_msgs.msg import Empty
    scene_pub = bench.create_publisher(Empty, '/hams/reset_scene', 10)
    for _ in range(8):                        # burst (discovery race)
        scene_pub.publish(Empty()); time.sleep(0.1)
    time.sleep(2.0)
    bench.go_home(duration_sec=5.0)           # /hams/reset_arm + named_config home
    time.sleep(3.0)
    bench.set_gripper(arm, OPEN_MM)
    time.sleep(1.0)
    bench.get_logger().info('[batt] env reset: scene restored + arm homed + gripper open')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--method', required=True, choices=('servo', 'noservo'))
    ap.add_argument('--arm', default='right', choices=('left', 'right'))
    ap.add_argument('--screw', default='auto', help="screw_NN or 'auto' (nearest reachable)")
    ap.add_argument('--success-dz', type=float, default=0.03, help='GT lift [m] = success')
    ap.add_argument('--no-reset', action='store_true', help='skip the pre-trial reset')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    rclpy.init()
    bench = GraspBenchmark(args.arm, gt_name='', box_source='gt')
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(bench)
    threading.Thread(target=ex.spin, daemon=True).start()

    rec = dict(method=args.method, success=False, error='')
    try:
        for _ in range(100):                              # wait for the GT feed
            if bench._obj_poses.get('__pelvis__') is not None:
                break
            time.sleep(0.2)
        if not args.no_reset:
            reset_env(bench, args.arm)                    # <-- clean start every trial
        screw = _pick_screw(bench, args.arm) if args.screw == 'auto' else args.screw
        if screw is None:
            raise RuntimeError('no reachable screw found from this stance')
        bench.get_logger().info(f'[batt] method={args.method} target={screw}')
        bench.reset_grip_track(args.arm)
        rec = run(bench, args.method, screw, args.arm, args.success_dz)
    except Exception as e:                                # noqa: BLE001
        rec['error'] = str(e)
    finally:
        line = json.dumps(rec, indent=2)
        print(line)
        if args.out:
            os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
            open(args.out, 'w').write(line + '\n')
        rclpy.shutdown()


if __name__ == '__main__':
    main()
