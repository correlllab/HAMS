"""Sway/stability analysis per the researched method spec (2026-07-22).

Computes the posturography + margin-of-stability battery from per-trial
10 Hz telemetry CSVs and renders fig7..fig11 into the paper bank. Imported
by make_paper_bank.py; standalone-runnable too. Same functions are meant to
be copied into the battery study for identical cross-task analysis.
"""
import json, glob, os, math, csv
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import kruskal, mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = '/home/guest/Downloads/HAMS-test-grasping/core_ws/benchmark_results'
METHODS = ['centroid', 'topdown_antipodal', 'graspgenx', 'skill']
MLABEL = {'centroid': 'centroid', 'topdown_antipodal': 'pca',
          'graspgenx': 'graspgenx', 'skill': 'skill'}
OUTC = {'succ': '#55A868', 'unstable': '#C9A227', 'wander': '#C44E52',
        'fall': '#8172B3', 'other': '#937860'}
FS = 10.0                     # telemetry Hz
CUT = 2.5                     # low-pass cutoff
G = 9.81
HANDLE = np.array([3.537, -2.889])
# H1-2 stance footprint (m, generous rectangle around both feet in stance)
BOS_AP = 0.24                 # +/- from center, along AP (foot length + margin)
BOS_ML = 0.16                 # +/- from center, along ML (stance width /2 + foot)

def _outcome(d):
    if d.get('success'): return 'succ'
    if d.get('robot_fell_during_trial'): return 'fall'
    if d.get('base_wandered'): return 'wander'
    g = d.get('grip_final_mm')
    if isinstance(g, (int, float)) and 20 < g < 85: return 'unstable'
    return 'other'

def load_trials(tier_dir='sweep_almi'):
    """-> list of dicts: t, ap, ml, yaw, gx,gy (gripper), outcome, method."""
    out = []
    b, a = butter(4, CUT / (FS / 2), btype='low')
    for m in METHODS:
        for f in sorted(glob.glob(f'{R}/{tier_dir}/{m}/trial_*_telemetry.csv')):
            j = f.replace('_telemetry.csv', '.json')
            try:
                d = json.load(open(j))
                rows = list(csv.DictReader(open(f)))
            except Exception:
                continue
            def col(k):
                return np.array([float(r[k]) if r[k] not in ('nan', '') else np.nan
                                 for r in rows])
            t = col('t_wall'); x = col('pelvis_x'); y = col('pelvis_y')
            gx = col('grip_x'); gy = col('grip_y'); gz = col('grip_z')
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < 30: continue
            t, x, y = t[ok], x[ok], y[ok]
            gx, gy, gz = gx[ok], gy[ok], gz[ok]
            # uniform grid + zero-phase filter
            tu = np.arange(t[0], t[-1], 1 / FS)
            if len(tu) < 30: continue
            xi = np.interp(tu, t, x); yi = np.interp(tu, t, y)
            if len(xi) > 15:
                xi = filtfilt(b, a, xi); yi = filtfilt(b, a, yi)
            # task frame: AP = toward handle from start pose, ML = perpendicular
            v = HANDLE - np.array([xi[0], yi[0]]); v = v / np.linalg.norm(v)
            ap = (xi - xi[0]) * v[0] + (yi - yi[0]) * v[1]
            ml = -(xi - xi[0]) * v[1] + (yi - yi[0]) * v[0]
            gxi = np.interp(tu, t, gx) if not np.isnan(gx).all() else None
            gyi = np.interp(tu, t, gy) if not np.isnan(gy).all() else None
            out.append(dict(m=m, outcome=_outcome(d), t=tu - tu[0],
                            ap=ap, ml=ml, x=xi, y=yi, gx=gxi, gy=gyi,
                            z=float(d.get('tip_to_handle_mm') or np.nan)))
    return out

def metrics(tr):
    ap, ml, t = tr['ap'], tr['ml'], tr['t']
    dur = t[-1] - t[0] if len(t) > 1 else 1
    step = np.sqrt(np.diff(ap) ** 2 + np.diff(ml) ** 2)
    L = step.sum()
    apc, mlc = ap - ap.mean(), ml - ml.mean()
    cov = np.cov(np.vstack([apc, mlc]))
    ev = np.clip(np.linalg.eigvalsh(cov), 0, None)
    # MoS via extrapolated CoM (pelvis height ~0.97 m)
    w0 = math.sqrt(G / 0.97)
    vx = np.gradient(ap, 1 / FS); vy = np.gradient(ml, 1 / FS)
    xap = ap + vx / w0; xml = ml + vy / w0
    mos = np.minimum(BOS_AP - np.abs(xap - ap.mean()), BOS_ML - np.abs(xml - ml.mean()))
    return dict(
        mvelo=L / dur * 1000,                       # mm/s
        rms_ap=float(np.sqrt((apc ** 2).mean())) * 1000,
        rms_ml=float(np.sqrt((mlc ** 2).mean())) * 1000,
        a95=float(5.991 * math.pi * math.sqrt(max(ev[0] * ev[1], 0))) * 1e6,  # mm^2
        peak_r=float(np.sqrt(apc ** 2 + mlc ** 2).max()) * 1000,
        drift=float(math.hypot(ap[-1] - ap[0], ml[-1] - ml[0])) * 1000,
        min_mos=float(mos.min()) * 1000,
    )

def _raincloud(ax, groups, labels, colors_by_outcome, unit):
    for i, (vals, outs) in enumerate(groups):
        if not vals: continue
        vals = np.asarray(vals)
        # half-violin
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(vals)
            yy = np.linspace(vals.min(), vals.max(), 80)
            dd = kde(yy); dd = dd / dd.max() * 0.32
            ax.fill_betweenx(yy, i - dd, i, alpha=.35, color='#88AACC', lw=0)
        except Exception:
            pass
        # box
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([i - .06, i + .06], [med, med], c='k', lw=2, zorder=5)
        ax.add_patch(plt.Rectangle((i - .05, q1), .1, q3 - q1, fill=False, lw=1))
        # rain
        jit = np.random.RandomState(0).uniform(.08, .3, len(vals))
        ax.scatter(i + jit, vals, s=14, alpha=.75,
                   c=[OUTC[o] for o in outs], zorder=4)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(unit); ax.spines[['top', 'right']].set_visible(False)

def render(out_dir):
    trials = load_trials()
    if len(trials) < 8:
        print('sway: not enough telemetry yet'); return
    per = {}
    for tr in trials:
        per.setdefault(tr['m'], []).append((metrics(tr), tr['outcome']))

    # fig7: rainclouds for the headline metrics
    keys = [('mvelo', 'mean sway velocity [mm/s]'), ('rms_ml', 'ML RMS [mm]'),
            ('a95', '95% sway ellipse [mm²]'), ('min_mos', 'min margin-of-stability [mm]')]
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.6))
    for ax, (k, unit) in zip(axs.flat, keys):
        groups = [([mm[0][k] for mm in per.get(m, [])],
                   [mm[1] for mm in per.get(m, [])]) for m in METHODS]
        _raincloud(ax, groups, [MLABEL[m] for m in METHODS], OUTC, unit)
        try:
            arrs = [np.array([g[0][i2] for i2 in range(len(g[0]))]) for g in
                    [( [mm[0][k] for mm in per.get(m, [])], ) for m in METHODS] ]
        except Exception:
            arrs = []
        try:
            samples = [[mm[0][k] for mm in per.get(m, [])] for m in METHODS]
            samples = [s for s in samples if len(s) >= 5]
            if len(samples) >= 2:
                h, p = kruskal(*samples)
                ax.set_title(f'{unit.split(" [")[0]}  (KW p={p:.3g})', fontsize=10)
        except Exception:
            pass
        if k == 'min_mos':
            ax.axhline(0, ls='--', lw=1, c='#C44E52')
            ax.text(.02, 4, 'step required below 0', fontsize=7, c='#C44E52',
                    transform=ax.get_yaxis_transform())
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=o)
               for o, c in OUTC.items()]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8, frameon=False)
    fig.suptitle('Standing tier: base-disturbance metrics by grasp method (posturography battery)')
    plt.tight_layout(rect=[0, .05, 1, .97])
    plt.savefig(f'{out_dir}/fig7_sway_rainclouds.png', dpi=160); plt.close()

    # fig8: bird's-eye small multiples with median track + endpoint ellipse
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.8), sharex=True, sharey=True)
    for ax, m in zip(axs, METHODS):
        trs = [tr for tr in trials if tr['m'] == m]
        ends = []
        for tr in trs:
            ax.plot(tr['ml'] * 100, tr['ap'] * 100, color=OUTC[tr['outcome']],
                    alpha=.3, lw=.8)
            ends.append((tr['ml'][-1], tr['ap'][-1]))
        # median track over normalized progress
        if trs:
            N = 60
            stack_ap = np.vstack([np.interp(np.linspace(0, 1, N),
                                            np.linspace(0, 1, len(tr['ap'])), tr['ap'])
                                  for tr in trs])
            stack_ml = np.vstack([np.interp(np.linspace(0, 1, N),
                                            np.linspace(0, 1, len(tr['ml'])), tr['ml'])
                                  for tr in trs])
            ax.plot(np.median(stack_ml, 0) * 100, np.median(stack_ap, 0) * 100,
                    c='k', lw=2, label='median')
        if len(ends) > 3:
            e = np.array(ends) * 100
            cov = np.cov(e.T); mu = e.mean(0)
            ev, evec = np.linalg.eigh(cov)
            ang = math.degrees(math.atan2(evec[1, -1], evec[0, -1]))
            from matplotlib.patches import Ellipse
            ax.add_patch(Ellipse(mu, 2 * math.sqrt(5.991 * max(ev[1], 0)),
                                 2 * math.sqrt(5.991 * max(ev[0], 0)), angle=ang,
                                 fill=False, ls='--', lw=1.2, ec='#333'))
        ax.plot(0, 0, 'k+', ms=8)
        ax.set_title(MLABEL[m], fontsize=10); ax.set_xlabel('ML [cm]')
        ax.set_aspect('equal')
    axs[0].set_ylabel('AP (toward handle) [cm]')
    fig.suptitle('Pelvis trajectories (task frame), 95% endpoint ellipse dashed')
    plt.tight_layout(); plt.savefig(f'{out_dir}/fig8_birdseye.png', dpi=160); plt.close()

    # fig9: event-aligned ensemble (contact proxy: gripper nearest handle)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for grp, col, lab in (({'succ'}, OUTC['succ'], 'success'),
                          ({'wander', 'fall', 'unstable', 'other'}, OUTC['wander'], 'failure')):
        curves = []
        for tr in trials:
            if tr['outcome'] not in grp or tr['gx'] is None: continue
            d2h = np.hypot(tr['gx'] - HANDLE[0], tr['gy'] - HANDLE[1])
            ic = int(np.argmin(d2h))
            sp = np.hypot(np.gradient(tr['ap'], 1 / FS), np.gradient(tr['ml'], 1 / FS))
            tau = tr['t'] - tr['t'][ic]
            grid = np.arange(-3, 3.01, .1)
            if tau[0] > -3 or tau[-1] < 3: continue
            curves.append(np.interp(grid, tau, sp))
        if len(curves) >= 5:
            cs = np.vstack(curves) * 1000
            mu = cs.mean(0); se = cs.std(0) / math.sqrt(len(cs))
            grid = np.arange(-3, 3.01, .1)
            ax.plot(grid, mu, c=col, lw=2, label=f'{lab} (n={len(cs)})')
            ax.fill_between(grid, mu - 1.96 * se, mu + 1.96 * se, color=col, alpha=.2)
    ax.axvline(0, ls='--', lw=1, c='#555'); ax.text(.05, .95, 'closest approach',
        transform=ax.transAxes, fontsize=8, c='#555')
    ax.set_xlabel('time relative to contact [s]'); ax.set_ylabel('pelvis speed [mm/s]')
    ax.set_title('Event-aligned base speed: successes vs failures')
    ax.legend(fontsize=9); ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout(); plt.savefig(f'{out_dir}/fig9_event_aligned.png', dpi=160); plt.close()

    # metrics CSV for the paper
    with open(f'{out_dir}/sway_metrics.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['method', 'outcome', 'mvelo_mms', 'rms_ap_mm', 'rms_ml_mm',
                    'a95_mm2', 'peak_r_mm', 'drift_mm', 'min_mos_mm'])
        for m in METHODS:
            for mm, o in per.get(m, []):
                w.writerow([m, o] + [round(mm[k], 2) for k in
                            ('mvelo', 'rms_ap', 'rms_ml', 'a95', 'peak_r', 'drift', 'min_mos')])
    print(f'sway figs rendered: {sum(len(v) for v in per.values())} trials analyzed')

if __name__ == '__main__':
    render(f'{R}/figures/paper_bank')
