#!/usr/bin/env python3
"""Aggregate the tethered battery sweep into two tables: (A) grasp metrics from
trial_*.json, (B) sway metrics from trial_*_telemetry.csv (posturography math
copied verbatim from sway_analysis.trial_metrics so numbers match the fridge study).
Handles ALL method dir names (unlike sway_analysis's fridge-only PRETTY map)."""
import glob, json, math, os, sys
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else 'core_ws/benchmark_results/sweep_battery_tethered'
HANDLE = tuple(float(v) for v in (sys.argv[2] if len(sys.argv) > 2 else '0.45,-0.05').split(','))

def _load_tel(path):
    try:
        raw = np.genfromtxt(path, delimiter=',', names=True, dtype=float)
    except Exception:
        return None
    if raw.dtype.names is None or raw.size == 0: return None
    cols = {n: np.atleast_1d(raw[n]) for n in raw.dtype.names}
    if 'pelvis_x' not in cols: return None
    good = np.isfinite(cols['pelvis_x']) & np.isfinite(cols['pelvis_y'])
    if good.sum() < 10: return None
    return {k: v[good] for k, v in cols.items()}

def _ellipse_area(x, y):
    if len(x) < 3: return 0.0
    cov = np.cov(np.vstack([x, y])); k = 5.991
    return math.pi * k * math.sqrt(max(np.linalg.det(cov), 0.0))

def sway(tel, handle):
    x = tel['pelvis_x']*1000.0; y = tel['pelvis_y']*1000.0
    z = tel.get('pelvis_z', np.zeros_like(x))*1000.0
    yaw = tel.get('pelvis_yaw_deg', np.zeros_like(x))
    t = tel.get('t_wall', np.arange(len(x))*0.1)
    dur = float(t[-1]-t[0]) if len(t) > 1 else 1.0
    mx, my = float(np.mean(x)), float(np.mean(y)); dx, dy = x-mx, y-my
    ax, ay = handle[0]*1000.0-mx, handle[1]*1000.0-my; an = math.hypot(ax, ay) or 1.0
    ux, uy = ax/an, ay/an; lx, ly = -uy, ux
    longi = dx*ux+dy*uy; lat = dx*lx+dy*ly
    radial = np.hypot(dx, dy); path_len = float(np.hypot(np.diff(x), np.diff(y)).sum())
    return dict(dur_s=dur, path_len_mm=path_len, mean_speed_mms=path_len/dur if dur else 0,
        rms_radial_mm=float(np.sqrt(np.mean(radial**2))), sway_area_mm2=_ellipse_area(x, y),
        rms_long_mm=float(np.sqrt(np.mean(longi**2))), rms_lat_mm=float(np.sqrt(np.mean(lat**2))),
        yaw_rms_deg=float(np.std(yaw)), yaw_range_deg=float(np.ptp(yaw)), z_rms_mm=float(np.std(z)))

def fnum(xs):
    xs = [v for v in xs if isinstance(v, (int, float)) and math.isfinite(v)]
    return (np.mean(xs), np.std(xs)) if xs else (float('nan'), float('nan'))

methods = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)) and not d.startswith('sway'))
grasp_rows, sway_rows = [], []
for m in methods:
    d = os.path.join(ROOT, m)
    js = sorted(glob.glob(d+'/trial_*.json'))
    n=succ=good=closed=err=0; ferr=[]; perr=[]; dz=[]
    for f in js:
        try: r = json.load(open(f))
        except: err+=1; continue
        n+=1
        if r.get('error'): err+=1
        if r.get('success'): succ+=1
        if r.get('good_grasp') or r.get('holding'): good+=1
        if r.get('closed_on_object') or r.get('grip_contact_flag'): closed+=1
        if isinstance(r.get('final_err_mm'), (int, float)): ferr.append(r['final_err_mm'])
        if isinstance(r.get('post_grasp_err_mm'), (int, float)): perr.append(r['post_grasp_err_mm'])
        v = r.get('screw_lift_dz'); v = v if isinstance(v, (int, float)) else r.get('lift_dz_m')
        if isinstance(v, (int, float)): dz.append(v)
    fm = fnum(ferr); pm = fnum(perr); dm = fnum(dz)
    grasp_rows.append((m, n, succ, good, closed, err, fm, pm, dm))
    S = [sway(t, HANDLE) for t in (_load_tel(f) for f in sorted(glob.glob(d+'/trial_*_telemetry.csv'))) if t]
    if S:
        agg = {k: float(np.mean([s[k] for s in S])) for k in S[0]}
        sway_rows.append((m, len(S), agg))
    else:
        sway_rows.append((m, 0, None))

pct = lambda a, b: f'{100.0*a/b:.0f}%' if b else '-'
print('\n### TABLE A — grasp metrics (tethered battery, screw_27, N per method)\n')
print('| method | n | success | good-grasp/hold | closed/contact | errored | final_err mm (μ±σ) | post_err mm | lift_dz m |')
print('|---|--:|--:|--:|--:|--:|--:|--:|--:|')
for m, n, sc, gd, cl, er, fm, pm, dm in grasp_rows:
    fs = f'{fm[0]:.1f}±{fm[1]:.1f}' if math.isfinite(fm[0]) else '-'
    ps = f'{pm[0]:.1f}±{pm[1]:.1f}' if math.isfinite(pm[0]) else '-'
    ds = f'{dm[0]:.3f}±{dm[1]:.3f}' if math.isfinite(dm[0]) else '-'
    print(f'| {m} | {n} | {sc} ({pct(sc,n)}) | {gd} ({pct(gd,n)}) | {cl} ({pct(cl,n)}) | {er} | {fs} | {ps} | {ds} |')

print('\n### TABLE B — pelvis sway / posturography (mean over trials)\n')
print('| method | trials | dur s | path_len mm | speed mm/s | rms_radial mm | sway_area mm² | rms_long mm | rms_lat mm | yaw_rms° | z_rms mm |')
print('|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|')
for m, k, a in sway_rows:
    if a is None: print(f'| {m} | 0 | - | - | - | - | - | - | - | - | - |'); continue
    print(f"| {m} | {k} | {a['dur_s']:.0f} | {a['path_len_mm']:.0f} | {a['mean_speed_mms']:.1f} | "
          f"{a['rms_radial_mm']:.2f} | {a['sway_area_mm2']:.1f} | {a['rms_long_mm']:.2f} | "
          f"{a['rms_lat_mm']:.2f} | {a['yaw_rms_deg']:.2f} | {a['z_rms_mm']:.2f} |")
