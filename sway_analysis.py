#!/usr/bin/env python3
"""Base-sway / postural-stability analysis for the grasp sweeps.

Runs HOST-SIDE on recorded files ONLY (per-trial *_telemetry.csv + trial JSON,
plus *_spawn.json on the randomized tier). It never touches the sim, the
containers, or the DDS bus — safe to run at any time, and it is scheduled to
run LAST, after all collection, so it can't interfere with a live sweep.

Why these metrics (not invented): base sway is the robotics analogue of human
POSTUROGRAPHY — the pelvis is our center-of-mass proxy (we have no force plate,
so no true center-of-pressure). The metric set is the canonical open-access COP
variable list (Quijoux et al. 2021, Physiological Reports; algorithms +
open code), adapted to a CoM signal, plus the loco-manipulation end-effector-
stability view (base motion disturbing the reach). See README refs at bottom.

THE TASK-DEPENDENT PART (Adam's point — "depending on the task what's needed
for sway is different"): posturography splits sway into ANTEROPOSTERIOR (AP)
and MEDIOLATERAL (ML) because balance fails differently in each axis. Here we
decompose the base's planar sway into LONGITUDINAL (along the robot->target
approach axis) and LATERAL (perpendicular). A horizontal fridge-handle grasp
loads the approach axis (reach forward = pitch/AP disturbance); a top-down
screw pickup loads lateral+vertical. Reporting the two axes separately is what
lets the paper say which stability dimension each task actually stresses.

Per trial we compute, from the pelvis (x,y,z,yaw) trace:
  DISPLACEMENT (station-keeping error, vs the settled reference / trial mean):
    drift_mm            net |mean_pos - reference| — did the base walk off?
    max_disp_mm         peak radial displacement from reference
  SWAY (fluctuation about the trajectory — instability, not drift):
    path_len_mm         total planar excursion (sum of step distances)
    mean_speed_mms      path_len / duration (mean CoM velocity — a top COP
                        fall-risk discriminant)
    rms_radial_mm       RMS radial distance about the MEAN position
    sway_area_mm2       95% confidence-ellipse area of the xy scatter
    rms_long_mm/rms_lat_mm   RMS sway resolved on the approach/lateral axes
    ellipse_ratio       lateral/longitudinal ellipse semi-axis (shape of sway)
    yaw_rms_deg/yaw_range_deg   rotational sway
    z_rms_mm            vertical bob
    mpf_hz              mean power frequency of the radial sway (control-effort
                        proxy; higher = busier corrections). Nyquist ~5 Hz at
                        the 10 Hz telemetry rate — captures postural sway,
                        which is <2 Hz.
Then per method/cell: mean+/-sd of each, and sway-vs-outcome separation
(success vs fail means + a simple AUC), and plots.

Usage (host):
  python3 sway_analysis.py <sweep_dir> [<sweep_dir> ...] [--out DIR] \
      [--handle X,Y] [--label-map skill=new,graspgenx=ggx]
Each <sweep_dir> is a dir of <method>/trial_*.json (+ *_telemetry.csv), e.g.
  .../benchmark_results/sweep_almi
  .../benchmark_results/sweep_paper/hanging   (etc.)
Writes <out>/sway_summary.json, <out>/sway_table.md, and PNGs in <out>/.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Fridge-handle bar centre (world xy) — the default approach-axis anchor. For
# another task (e.g. the battery screw row) pass --handle X,Y.
DEFAULT_HANDLE = (3.5365, -2.889)
PRETTY = {'centroid': 'centroid', 'topdown_antipodal': 'pca',
          'topdown_irl': 'topdown(irl)', 'graspgenx': 'graspgenx-normal',
          'skill': 'new(skill)'}
METHOD_ORDER = ['skill', 'graspgenx', 'topdown_irl', 'topdown_antipodal', 'centroid']


def _load_telemetry(path):
    """CSV -> dict of float arrays, NaN-cleaned to rows with a valid pelvis xy."""
    try:
        raw = np.genfromtxt(path, delimiter=',', names=True, dtype=float)
    except Exception:
        return None
    if raw.dtype.names is None or raw.size == 0:
        return None
    cols = {n: np.atleast_1d(raw[n]) for n in raw.dtype.names}
    if 'pelvis_x' not in cols:
        return None
    good = np.isfinite(cols['pelvis_x']) & np.isfinite(cols['pelvis_y'])
    if good.sum() < 10:                       # too short to say anything
        return None
    return {k: v[good] for k, v in cols.items()}


def _conf_ellipse_area(x, y):
    """95% confidence-ellipse area of a 2D scatter (prediction ellipse, chi2 with
    2 dof: k = -2 ln(1-0.95) = 5.991). Area = pi * k * sqrt(det(cov))."""
    if len(x) < 3:
        return 0.0, (0.0, 0.0), 0.0
    cov = np.cov(np.vstack([x, y]))
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 0.0, None)
    k = 5.991
    semi = np.sqrt(k * evals)                 # semi-axis lengths (major last)
    area = math.pi * k * math.sqrt(max(np.linalg.det(cov), 0.0))
    # angle of the major axis (largest eigenvalue = last)
    ang = math.atan2(evecs[1, -1], evecs[0, -1])
    return area, (float(semi[-1]), float(semi[0])), ang


def _mpf(sig, fs=10.0):
    """Mean power frequency [Hz] of a (detrended) signal via its PSD."""
    s = sig - np.mean(sig)
    if len(s) < 8 or not np.any(s):
        return float('nan')
    f = np.fft.rfftfreq(len(s), d=1.0 / fs)
    p = np.abs(np.fft.rfft(s)) ** 2
    if p.sum() <= 0:
        return float('nan')
    return float((f * p).sum() / p.sum())


def trial_metrics(tel, handle_xy):
    """Posturography metric set for one pelvis trace. Returns a flat dict (mm,
    mm^2, deg, Hz). Sway is measured about the trajectory MEAN (the station the
    policy actually held); drift is mean-vs-reference and end-vs-reference."""
    x = tel['pelvis_x'] * 1000.0             # -> mm
    y = tel['pelvis_y'] * 1000.0
    z = tel.get('pelvis_z', np.zeros_like(x)) * 1000.0
    yaw = tel.get('pelvis_yaw_deg', np.zeros_like(x))
    t = tel.get('t_wall', np.arange(len(x)) * 0.1)
    dur = float(t[-1] - t[0]) if len(t) > 1 else 1.0

    mx, my = float(np.mean(x)), float(np.mean(y))
    dx, dy = x - mx, y - my                   # sway about the held station

    # Approach axis from mean pelvis -> target (unit), lateral = 90deg CCW.
    ax, ay = handle_xy[0] * 1000.0 - mx, handle_xy[1] * 1000.0 - my
    an = math.hypot(ax, ay) or 1.0
    ux, uy = ax / an, ay / an                 # longitudinal (toward target)
    lx, ly = -uy, ux                          # lateral
    longi = dx * ux + dy * uy
    lat = dx * lx + dy * ly

    steps = np.hypot(np.diff(x), np.diff(y))
    path_len = float(steps.sum())
    radial = np.hypot(dx, dy)
    area, semi, _ = _conf_ellipse_area(x, y)

    # reference: settled ref if the sweep logged one, else the trajectory mean
    ref = tel.get('_ref_xy')
    if ref is not None:
        rx, ry = ref[0] * 1000.0, ref[1] * 1000.0
    else:
        rx, ry = mx, my
    drift = math.hypot(mx - rx, my - ry)
    max_disp = float(np.max(np.hypot(x - rx, y - ry)))

    return dict(
        n=int(len(x)), dur_s=round(dur, 1),
        drift_mm=round(drift, 1), max_disp_mm=round(max_disp, 1),
        path_len_mm=round(path_len, 1),
        mean_speed_mms=round(path_len / dur if dur else 0.0, 2),
        rms_radial_mm=round(float(np.sqrt(np.mean(radial ** 2))), 2),
        sway_area_mm2=round(area, 1),
        rms_long_mm=round(float(np.sqrt(np.mean(longi ** 2))), 2),
        rms_lat_mm=round(float(np.sqrt(np.mean(lat ** 2))), 2),
        ellipse_ratio=round((semi[1] / semi[0]) if semi[0] else 0.0, 2),
        yaw_rms_deg=round(float(np.std(yaw)), 2),
        yaw_range_deg=round(float(np.ptp(yaw)), 2),
        z_rms_mm=round(float(np.std(z)), 2),
        mpf_hz=round(_mpf(radial), 3),
    )


def _auc(pos, neg):
    """Mann-Whitney AUC: P(a random success metric < a random fail metric).
    >0.5 means higher metric associates with FAILURE (more sway -> fail)."""
    if not pos or not neg:
        return float('nan')
    wins = sum((n > p) + 0.5 * (n == p) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


METRIC_KEYS = ['drift_mm', 'max_disp_mm', 'path_len_mm', 'mean_speed_mms',
               'rms_radial_mm', 'sway_area_mm2', 'rms_long_mm', 'rms_lat_mm',
               'ellipse_ratio', 'yaw_rms_deg', 'yaw_range_deg', 'z_rms_mm', 'mpf_hz']


def analyze_dir(sweep, handle_xy):
    """One sweep dir -> {method: {per-trial rows, aggregates, success split}}."""
    out = {}
    for m in METHOD_ORDER:
        mdir = os.path.join(sweep, m)
        if not os.path.isdir(mdir):
            continue
        rows = []
        for jf in sorted(glob.glob(os.path.join(mdir, 'trial_*.json'))):
            stem = jf[:-5]
            tel = _load_telemetry(stem + '_telemetry.csv')
            if tel is None:
                continue
            try:
                tj = json.load(open(jf))
            except Exception:
                continue
            sp = stem + '_spawn.json'
            if os.path.isfile(sp):
                try:
                    ref = json.load(open(sp)).get('settled_ref_xy', '')
                    parts = [float(v) for v in str(ref).split()[:2]]
                    if len(parts) == 2:
                        tel['_ref_xy'] = parts
                except Exception:
                    pass
            met = trial_metrics(tel, handle_xy)
            met['trial'] = os.path.basename(jf)
            met['success'] = bool(tj.get('success'))
            met['wandered'] = bool(tj.get('base_wandered'))
            met['_stem'] = stem            # for the stabilogram re-read
            rows.append(met)
        if not rows:
            continue
        agg, split = {}, {}
        for k in METRIC_KEYS:
            vals = [r[k] for r in rows if np.isfinite(r[k])]
            if vals:
                agg[k] = dict(mean=round(float(np.mean(vals)), 2),
                              sd=round(float(np.std(vals)), 2),
                              median=round(float(np.median(vals)), 2))
            pos = [r[k] for r in rows if r['success'] and np.isfinite(r[k])]
            neg = [r[k] for r in rows if not r['success'] and np.isfinite(r[k])]
            split[k] = dict(succ_mean=round(float(np.mean(pos)), 2) if pos else None,
                            fail_mean=round(float(np.mean(neg)), 2) if neg else None,
                            auc_fail=_auc(pos, neg))
        out[m] = dict(n=len(rows), n_success=sum(r['success'] for r in rows),
                      trials=rows, aggregate=agg, success_split=split)
    return out


# --------------------------------------------------------------------- plots
def _stabilogram(results, handle_xy, path):
    """Per-method overlaid pelvis sway traces (mean-centred, mm), with the 95%
    ellipse. The classic posturography stabilogram, one panel per method."""
    tiers = list(results)
    methods = [m for m in METHOD_ORDER if any(m in results[t] for t in tiers)]
    if not methods:
        return
    fig, axes = plt.subplots(len(tiers), len(methods),
                             figsize=(3.1 * len(methods), 3.1 * len(tiers)),
                             squeeze=False)
    for ti, tier in enumerate(tiers):
        for mi, m in enumerate(methods):
            ax = axes[ti][mi]
            data = results[tier].get(m)
            if data:
                allx, ally = [], []
                for r in data['trials']:
                    tel = _load_telemetry(r['_stem'] + '_telemetry.csv')
                    if tel is None:
                        continue
                    x, y = tel['pelvis_x'] * 1000.0, tel['pelvis_y'] * 1000.0
                    dx, dy = x - x.mean(), y - y.mean()
                    # rotate into (longitudinal, lateral) about the approach axis
                    axv = handle_xy[0] * 1000.0 - x.mean()
                    ayv = handle_xy[1] * 1000.0 - y.mean()
                    an = math.hypot(axv, ayv) or 1.0
                    ux, uy = axv / an, ayv / an
                    lo, la = dx * ux + dy * uy, -dx * uy + dy * ux
                    ax.plot(lo, la, lw=.5, alpha=.55,
                            color=('#2a9d3a' if r['success'] else '#c0392b'))
                    allx.append(lo); ally.append(la)
                if allx:
                    ex, ey = np.concatenate(allx), np.concatenate(ally)
                    _, semi, ang = _conf_ellipse_area(ex, ey)
                    th = np.linspace(0, 2 * math.pi, 90)
                    exy = np.stack([semi[0] * np.cos(th), semi[1] * np.sin(th)])
                    R = np.array([[math.cos(ang), -math.sin(ang)],
                                  [math.sin(ang), math.cos(ang)]])
                    e = R @ exy
                    ax.plot(e[0], e[1], lw=1.0, c='k', ls='--', alpha=.8)
                ax.set_title(f'{os.path.basename(tier)} · {PRETTY.get(m, m)}\n'
                             f'{data["n_success"]}/{data["n"]}', fontsize=8)
            ax.set_aspect('equal'); ax.axhline(0, lw=.4, c='.7'); ax.axvline(0, lw=.4, c='.7')
            ax.tick_params(labelsize=6)
    axes[-1][0].set_xlabel('longitudinal sway (mm)', fontsize=7)
    axes[-1][0].set_ylabel('lateral sway (mm)', fontsize=7)
    fig.suptitle('Base sway stabilograms (pelvis, mean-centred)', fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _metric_bars(results, path):
    """Grouped bars of the headline sway metrics per method, faceted by tier."""
    show = ['rms_radial_mm', 'sway_area_mm2', 'path_len_mm', 'rms_long_mm',
            'rms_lat_mm', 'mean_speed_mms']
    tiers = list(results)
    methods = [m for m in METHOD_ORDER if any(m in results[t] for t in tiers)]
    if not methods:
        return
    fig, axes = plt.subplots(len(show), 1, figsize=(1.6 + 1.3 * len(methods) * len(tiers), 2.1 * len(show)), squeeze=False)
    for si, key in enumerate(show):
        ax = axes[si][0]
        labels, vals, errs, colors = [], [], [], []
        palette = plt.cm.tab10(np.linspace(0, 1, max(len(tiers), 1)))
        for ti, tier in enumerate(tiers):
            for m in methods:
                d = results[tier].get(m)
                if not d or key not in d['aggregate']:
                    continue
                labels.append(f'{PRETTY.get(m, m)}\n{os.path.basename(tier)}')
                vals.append(d['aggregate'][key]['mean'])
                errs.append(d['aggregate'][key]['sd'])
                colors.append(palette[ti])
        xs = np.arange(len(vals))
        ax.bar(xs, vals, yerr=errs, color=colors, capsize=2)
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6)
        ax.set_ylabel(key, fontsize=7); ax.tick_params(labelsize=6)
        ax.grid(axis='y', lw=.3, alpha=.5)
    fig.suptitle('Sway metrics by method / tier (mean +/- sd)', fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _sway_vs_outcome(results, path):
    """rms_radial per trial, success vs fail, per method — does sway predict the
    grasp outcome? (The loco-manipulation claim: base instability breaks reach.)"""
    tiers = list(results)
    methods = [m for m in METHOD_ORDER if any(m in results[t] for t in tiers)]
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(methods) * len(tiers), 4))
    xt, xl = [], []
    i = 0
    for tier in tiers:
        for m in methods:
            d = results[tier].get(m)
            if not d:
                continue
            for r in d['trials']:
                if not np.isfinite(r['rms_radial_mm']):
                    continue
                c = '#2a9d3a' if r['success'] else '#c0392b'
                ax.scatter(i + np.random.uniform(-.18, .18), r['rms_radial_mm'],
                           c=c, s=16, alpha=.75, edgecolors='none')
            xt.append(i); xl.append(f'{PRETTY.get(m, m)}\n{os.path.basename(tier)}')
            i += 1
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=6)
    ax.set_ylabel('per-trial RMS radial sway (mm)', fontsize=8)
    ax.set_title('Base sway vs grasp outcome (green=success, red=fail)', fontsize=9)
    ax.grid(axis='y', lw=.3, alpha=.5)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _md_table(results, path):
    lines = ['# Base-sway summary (posturography metrics)', '',
             'Pelvis = CoM proxy. Sway about the held station; drift = station-'
             'keeping error. Longitudinal = along the approach axis, lateral = '
             'perpendicular (the task-dependent split). One block per tier.', '']
    for tier, mres in results.items():
        lines += [f'## {os.path.basename(tier)}', '',
                  '| method | n | succ | drift_mm | rms_radial | sway_area_mm2 | '
                  'rms_long | rms_lat | path_len | speed_mms | yaw_rms | mpf_hz |',
                  '|---|--|--|--|--|--|--|--|--|--|--|--|']
        for m in METHOD_ORDER:
            d = mres.get(m)
            if not d:
                continue
            a = d['aggregate']
            g = lambda k: a.get(k, {}).get('mean', '-')
            lines.append(f'| {PRETTY.get(m, m)} | {d["n"]} | {d["n_success"]} | '
                         f'{g("drift_mm")} | {g("rms_radial_mm")} | {g("sway_area_mm2")} | '
                         f'{g("rms_long_mm")} | {g("rms_lat_mm")} | {g("path_len_mm")} | '
                         f'{g("mean_speed_mms")} | {g("yaw_rms_deg")} | {g("mpf_hz")} |')
        # success-separation line for the headline metric
        lines += ['', '_sway→failure separation (AUC>0.5 = more sway predicts '
                  'failure), rms_radial:_ ']
        for m in METHOD_ORDER:
            d = mres.get(m)
            if d:
                s = d['success_split'].get('rms_radial_mm', {})
                lines.append(f'- {PRETTY.get(m, m)}: succ_mean={s.get("succ_mean")} '
                             f'fail_mean={s.get("fail_mean")} AUC={s.get("auc_fail")}')
        lines.append('')
    open(path, 'w').write('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('sweeps', nargs='+', help='sweep dir(s) of <method>/trial_*.json')
    ap.add_argument('--out', default='', help='output dir (default: first sweep /sway_viz)')
    ap.add_argument('--handle', default='', help='approach-axis target "X,Y" world '
                    f'(default fridge handle {DEFAULT_HANDLE})')
    args = ap.parse_args()
    handle = DEFAULT_HANDLE
    if args.handle:
        handle = tuple(float(v) for v in args.handle.split(','))[:2]
    out = args.out or os.path.join(args.sweeps[0], 'sway_viz')
    os.makedirs(out, exist_ok=True)

    results = {}
    for sw in args.sweeps:
        r = analyze_dir(sw, handle)
        if r:
            results[sw] = r
    if not results:
        print('[sway] no telemetry found under:', *args.sweeps); return

    json.dump({os.path.basename(k): v for k, v in results.items()},
              open(os.path.join(out, 'sway_summary.json'), 'w'), indent=2)
    _md_table(results, os.path.join(out, 'sway_table.md'))
    _metric_bars(results, os.path.join(out, 'sway_metric_bars.png'))
    _sway_vs_outcome(results, os.path.join(out, 'sway_vs_outcome.png'))
    _stabilogram(results, handle, os.path.join(out, 'sway_stabilograms.png'))
    print(f'[sway] wrote sway_summary.json, sway_table.md, 3 PNGs -> {out}')
    for sw, mres in results.items():
        print(f'  {os.path.basename(sw)}:')
        for m, d in mres.items():
            a = d['aggregate']
            print(f'    {PRETTY.get(m, m):16s} n={d["n"]:2d} succ={d["n_success"]:2d} '
                  f'rms_radial={a.get("rms_radial_mm", {}).get("mean")}mm '
                  f'sway_area={a.get("sway_area_mm2", {}).get("mean")}mm2 '
                  f'long/lat={a.get("rms_long_mm", {}).get("mean")}/'
                  f'{a.get("rms_lat_mm", {}).get("mean")}')


if __name__ == '__main__':
    main()

# References (methodology grounding, not invented):
#  - Quijoux et al. 2021, Physiological Reports 9:e15067 — "A review of center
#    of pressure (COP) variables to quantify standing balance", algorithms +
#    open-access code (path length, 95% ellipse, RMS, MPF, AP/ML split).
#  - Prieto et al. 1996, IEEE TBME — classic COP time/frequency measures.
#  - SoFTA (Learning Gentle Humanoid Locomotion, 2025) — end-effector stability
#    under base motion, the loco-manipulation framing for sway->reach coupling.
