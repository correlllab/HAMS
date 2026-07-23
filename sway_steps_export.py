"""Export a consolidated step-level (10 Hz) sway dataset — one row per timestep
per episode — for the non-frozen tiers (standing + hanging, the bases that move).

Reads each trial's telemetry CSV + its result JSON, and emits a single tidy
long-format CSV with the raw base/gripper/door pose per step plus derived
task-frame sway columns (AP = toward handle, ML = lateral, recentered to the
episode's first sample) and instantaneous planar base speed.

Output: core_ws/benchmark_results/figures/paper_bank/sway_steps.csv
"""
import json, glob, csv, math, os

R = '/home/guest/Downloads/HAMS-test-grasping/core_ws/benchmark_results'
OUT = f'{R}/figures/paper_bank/sway_steps.csv'
HANDLE = (3.537, -2.889)                      # world XY of the handle bar
TIERS = [('standing', 'sweep_almi'), ('hanging', 'sweep_unfrozen')]
FS = 10.0

def outcome(d):
    if d.get('success'): return 'success'
    if d.get('robot_fell_during_trial'): return 'fall'
    if d.get('base_wandered'): return 'wander'
    g = d.get('grip_final_mm')
    if isinstance(g, (int, float)) and 20 < g < 85: return 'contact_unstable'
    return 'miss'

def fnum(v):
    try:
        x = float(v)
        return x if x == x else None            # drop NaN
    except (TypeError, ValueError):
        return None

cols = ['tier', 'method', 'trial', 'outcome', 'success', 'step', 't_rel_s',
        'pelvis_x', 'pelvis_y', 'pelvis_z', 'pelvis_yaw_deg',
        'grip_x', 'grip_y', 'grip_z', 'door_x', 'door_y', 'door_z',
        'ap_m', 'ml_m', 'planar_speed_mms']

n_rows = n_ep = 0
with open(OUT, 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(cols)
    for tier, dirn in TIERS:
        for tf in sorted(glob.glob(f'{R}/{dirn}/*/trial_*_telemetry.csv')):
            method = tf.split(f'{dirn}/')[1].split('/')[0]
            trial = os.path.basename(tf).split('_telemetry')[0]      # trial_XX
            jf = tf.replace('_telemetry.csv', '.json')
            try:
                d = json.load(open(jf))
            except Exception:
                d = {}
            oc = outcome(d); succ = int(bool(d.get('success')))
            rows = list(csv.DictReader(open(tf)))
            if not rows:
                continue
            # task frame from the first valid pose
            t0 = fnum(rows[0]['t_wall'])
            x0 = y0 = None
            for r in rows:
                x0, y0 = fnum(r['pelvis_x']), fnum(r['pelvis_y'])
                if x0 is not None and y0 is not None:
                    break
            if x0 is None:
                continue
            vx, vy = HANDLE[0] - x0, HANDLE[1] - y0
            vn = math.hypot(vx, vy) or 1.0
            ux, uy = vx / vn, vy / vn            # unit AP axis (toward handle)
            prev = None
            for i, r in enumerate(rows):
                t = fnum(r['t_wall']); px = fnum(r['pelvis_x']); py = fnum(r['pelvis_y'])
                ap = ml = spd = None
                if px is not None and py is not None:
                    ap = (px - x0) * ux + (py - y0) * uy
                    ml = -(px - x0) * uy + (py - y0) * ux
                    if prev is not None and t is not None and prev[2] is not None:
                        dt = t - prev[2]
                        if dt > 1e-6:
                            spd = math.hypot(px - prev[0], py - prev[1]) / dt * 1000.0
                    prev = (px, py, t)
                w.writerow([
                    tier, method, trial, oc, succ, i,
                    round(t - t0, 3) if (t is not None and t0 is not None) else '',
                    r['pelvis_x'], r['pelvis_y'], r['pelvis_z'], r['pelvis_yaw_deg'],
                    r['grip_x'], r['grip_y'], r['grip_z'], r['door_x'], r['door_y'], r['door_z'],
                    round(ap, 5) if ap is not None else '',
                    round(ml, 5) if ml is not None else '',
                    round(spd, 2) if spd is not None else '',
                ])
                n_rows += 1
            n_ep += 1
print(f'sway_steps.csv: {n_rows} rows across {n_ep} episodes -> {OUT}')
