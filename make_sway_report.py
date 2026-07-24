#!/usr/bin/env python3
"""Full sway report for the tethered battery sweep:
  1) sway_report/per_trial.csv   -- ONE ROW PER TRIAL, every sway + grasp metric
  2) sway_report/per_method.csv  -- per-method mean/std of every metric
  3) sway_report/SWAY_METRICS.md -- documents each metric + the summary tables
Posturography math is copied verbatim from sway_analysis.trial_metrics so the
numbers match the fridge study."""
import csv, glob, json, math, os, sys
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else 'core_ws/benchmark_results/sweep_battery_tethered'
HANDLE = tuple(float(v) for v in (sys.argv[2] if len(sys.argv) > 2 else '0.45,-0.05').split(','))
OUT = 'sway_report'; os.makedirs(OUT, exist_ok=True)

def load_tel(path):
    try: raw = np.genfromtxt(path, delimiter=',', names=True, dtype=float)
    except Exception: return None
    if raw.dtype.names is None or raw.size == 0: return None
    cols = {n: np.atleast_1d(raw[n]) for n in raw.dtype.names}
    if 'pelvis_x' not in cols: return None
    good = np.isfinite(cols['pelvis_x']) & np.isfinite(cols['pelvis_y'])
    if good.sum() < 10: return None
    return {k: v[good] for k, v in cols.items()}

def ellipse(x, y):
    if len(x) < 3: return 0.0, (0.0, 0.0)
    cov = np.cov(np.vstack([x, y])); k = 5.991
    ev = np.clip(np.linalg.eigh(cov)[0], 0, None); semi = np.sqrt(k * ev)
    return math.pi * k * math.sqrt(max(np.linalg.det(cov), 0.0)), (float(semi[-1]), float(semi[0]))

def mpf(sig, fs=10.0):
    s = sig - np.mean(sig)
    if len(s) < 8 or not np.any(s): return float('nan')
    f = np.fft.rfftfreq(len(s), d=1/fs); p = np.abs(np.fft.rfft(s))**2
    return float((f*p).sum()/p.sum()) if p.sum() > 0 else float('nan')

def sway(tel, h):
    x = tel['pelvis_x']*1000; y = tel['pelvis_y']*1000
    z = tel.get('pelvis_z', np.zeros_like(x))*1000; yaw = tel.get('pelvis_yaw_deg', np.zeros_like(x))
    t = tel.get('t_wall', np.arange(len(x))*0.1); dur = float(t[-1]-t[0]) if len(t) > 1 else 1.0
    mx, my = float(np.mean(x)), float(np.mean(y)); dx, dy = x-mx, y-my
    ax, ay = h[0]*1000-mx, h[1]*1000-my; an = math.hypot(ax, ay) or 1
    ux, uy = ax/an, ay/an; longi = dx*ux+dy*uy; lat = dx*(-uy)+dy*ux
    path = float(np.hypot(np.diff(x), np.diff(y)).sum()); radial = np.hypot(dx, dy)
    area, semi = ellipse(x, y)
    return dict(dur_s=round(dur, 1), drift_mm=round(math.hypot(0, 0), 1),
        max_disp_mm=round(float(np.max(np.hypot(x-mx, y-my))), 1), path_len_mm=round(path, 1),
        mean_speed_mms=round(path/dur if dur else 0, 2), rms_radial_mm=round(float(np.sqrt(np.mean(radial**2))), 2),
        sway_area_mm2=round(area, 1), rms_long_mm=round(float(np.sqrt(np.mean(longi**2))), 2),
        rms_lat_mm=round(float(np.sqrt(np.mean(lat**2))), 2), ellipse_ratio=round(semi[1]/semi[0] if semi[0] else 0, 2),
        yaw_rms_deg=round(float(np.std(yaw)), 2), yaw_range_deg=round(float(np.ptp(yaw)), 2),
        z_rms_mm=round(float(np.std(z)), 2), mpf_hz=round(mpf(radial), 3))

SWAY_KEYS = ['dur_s','max_disp_mm','path_len_mm','mean_speed_mms','rms_radial_mm','sway_area_mm2',
             'rms_long_mm','rms_lat_mm','ellipse_ratio','yaw_rms_deg','yaw_range_deg','z_rms_mm','mpf_hz']
GRASP_KEYS = ['success','good_grasp','closed','errored','final_err_mm','post_grasp_err_mm','lift_dz_m']

methods = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)) and not d.startswith('sway'))
rows = []
for m in methods:
    d = os.path.join(ROOT, m)
    for jf in sorted(glob.glob(d+'/trial_*.json')):
        tn = os.path.basename(jf).replace('trial_', '').replace('.json', '')
        try: r = json.load(open(jf))
        except Exception: r = {}
        row = dict(method=m, trial=tn,
            success=int(bool(r.get('success'))),
            good_grasp=int(bool(r.get('good_grasp') or r.get('holding'))),
            closed=int(bool(r.get('closed_on_object') or r.get('grip_contact_flag'))),
            errored=int(bool(r.get('error'))),
            final_err_mm=r.get('final_err_mm'), post_grasp_err_mm=r.get('post_grasp_err_mm'),
            lift_dz_m=r.get('screw_lift_dz') if isinstance(r.get('screw_lift_dz'), (int, float)) else r.get('lift_dz_m'))
        tel = load_tel(d+f'/trial_{tn}_telemetry.csv')
        row.update(sway(tel, HANDLE) if tel else {k: '' for k in SWAY_KEYS})
        rows.append(row)

cols = ['method','trial'] + GRASP_KEYS + SWAY_KEYS
with open(f'{OUT}/per_trial.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows: w.writerow({k: r.get(k, '') for k in cols})

# per-method aggregate
def num(xs):
    xs = [v for v in xs if isinstance(v, (int, float)) and math.isfinite(v)]
    return (np.mean(xs), np.std(xs)) if xs else (float('nan'), float('nan'))
agg = {}
for m in methods:
    mr = [r for r in rows if r['method'] == m]; n = len(mr)
    a = dict(n=n, success_rate=round(100*sum(r['success'] for r in mr)/n, 1) if n else 0,
             good_rate=round(100*sum(r['good_grasp'] for r in mr)/n, 1) if n else 0,
             closed_rate=round(100*sum(r['closed'] for r in mr)/n, 1) if n else 0,
             errored=sum(r['errored'] for r in mr))
    for k in GRASP_KEYS[4:] + SWAY_KEYS:
        mu, sd = num([r.get(k) for r in mr]); a[k+'_mean'] = round(mu, 3) if math.isfinite(mu) else ''
        a[k+'_std'] = round(sd, 3) if math.isfinite(sd) else ''
    agg[m] = a
with open(f'{OUT}/per_method.csv', 'w', newline='') as f:
    keys = ['method','n','success_rate','good_rate','closed_rate','errored'] + \
           [k+'_mean' for k in GRASP_KEYS[4:]+SWAY_KEYS]
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for m in methods: w.writerow({'method': m, **{k: agg[m].get(k, '') for k in keys if k != 'method'}})

DOC = {'dur_s':'trial telemetry duration (s)','max_disp_mm':'max pelvis displacement from mean station (mm)',
 'path_len_mm':'total pelvis path length in XY (mm) — cumulative wander','mean_speed_mms':'path_len/dur (mm/s)',
 'rms_radial_mm':'RMS radial sway about the held station (mm) — overall sway magnitude',
 'sway_area_mm2':'95% confidence-ellipse area of pelvis XY scatter (mm²) — sway spread',
 'rms_long_mm':'RMS sway along the approach axis to the target (mm)',
 'rms_lat_mm':'RMS sway perpendicular to the approach axis (mm)',
 'ellipse_ratio':'minor/major axis ratio of the sway ellipse (1=isotropic)',
 'yaw_rms_deg':'RMS pelvis yaw (deg) — twisting','yaw_range_deg':'peak-to-peak pelvis yaw (deg)',
 'z_rms_mm':'RMS pelvis vertical motion (mm) — bob','mpf_hz':'mean power frequency of radial sway (Hz)'}
with open(f'{OUT}/SWAY_METRICS.md', 'w') as f:
    f.write('# Tethered battery sweep — sway metrics\n\n')
    f.write(f'Source: `{ROOT}`  | approach-axis anchor (screw XY): `{HANDLE}`\n')
    f.write('Sway is measured on the pelvis (`__pelvis__`) trace at ~10 Hz, about the trajectory MEAN '
            '(the station the tether held). Posturography math identical to the fridge study.\n\n')
    f.write('## Files\n- `per_trial.csv` — one row per trial, every metric (the "full CSV, each step")\n'
            '- `per_method.csv` — per-method mean of every metric\n\n')
    f.write('## Metric definitions\n\n| metric | meaning |\n|---|---|\n')
    for k, v in DOC.items(): f.write(f'| `{k}` | {v} |\n')
    f.write('\n## Per-method summary\n\n')
    f.write('| method | n | success% | good% | closed% | err | rms_radial | sway_area | rms_long | rms_lat | yaw_rms | z_rms |\n')
    f.write('|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n')
    for m in methods:
        a = agg[m]
        f.write(f"| {m} | {a['n']} | {a['success_rate']} | {a['good_rate']} | {a['closed_rate']} | {a['errored']} | "
                f"{a.get('rms_radial_mm_mean','')} | {a.get('sway_area_mm2_mean','')} | {a.get('rms_long_mm_mean','')} | "
                f"{a.get('rms_lat_mm_mean','')} | {a.get('yaw_rms_deg_mean','')} | {a.get('z_rms_mm_mean','')} |\n")

print('wrote:', OUT+'/per_trial.csv', OUT+'/per_method.csv', OUT+'/SWAY_METRICS.md')
print('trials:', len(rows), '| methods:', methods)
