"""Regenerate ALL paper figures + tables from whatever trial data exists.

Idempotent: reads the four result dirs (frozen sweep/, hanging sweep_unfrozen/,
standing sweep_almi/ [new executor, n->30], sweep_almi_ablation/ [old executor])
and rewrites figures/paper_bank/. Run it any time; it reflects the disk.
Appends one tally line per run to paper_bank/history.log.
"""
import json, glob, os, math, csv, datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R = '/home/guest/Downloads/HAMS-test-grasping/core_ws/benchmark_results'
OUT = f'{R}/figures/paper_bank'
os.makedirs(OUT, exist_ok=True)
METHODS = ['centroid', 'topdown_antipodal', 'graspgenx', 'skill']
MLABEL = {'centroid': 'centroid', 'topdown_antipodal': 'pca (top-down)',
          'graspgenx': 'graspgenx (raw)', 'skill': 'skill (ranked)'}
TIERS = [('frozen', 'sweep'), ('hanging', 'sweep_unfrozen'), ('standing', 'sweep_almi')]
C = {'frozen': '#4C72B0', 'hanging': '#DD8452', 'standing': '#55A868', 'abl': '#B0B0B0'}

def load(tier_dir, m):
    out = []
    for f in sorted(glob.glob(f'{R}/{tier_dir}/{m}/trial_*.json')):
        try:
            out.append(json.load(open(f)))
        except Exception:
            pass                     # trial mid-write; next regen catches it
    return out

def wilson(s, n):
    if n == 0: return (0.0, 0.0)
    p, z = s / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))

data = {t: {m: load(d, m) for m in METHODS} for t, d in TIERS}
abl = {m: load('sweep_almi_ablation', m) for m in METHODS}
succ = lambda ts: sum(1 for d in ts if d.get('success'))

def fisher(a_s, a_n, b_s, b_n):
    """Two-sided Fisher exact p for batch heterogeneity (no scipy dep)."""
    from math import comb
    if a_n == 0 or b_n == 0: return 1.0
    N, K, n = a_n + b_n, a_s + b_s, a_n
    def pmf(k): return comb(K, k) * comb(N - K, n - k) / comb(N, n)
    p_obs = pmf(a_s)
    return min(1.0, sum(pmf(k) for k in range(max(0, K - (N - n)), min(K, n) + 1)
                        if pmf(k) <= p_obs + 1e-12))

# ---- batch heterogeneity: trials 1-20 vs 21-30 (top-ups) ----------------
het_flags = {}
for t, dirn in TIERS:
    if t == 'standing':
        continue   # one continuous 30-trial run, one protocol — no batch boundary
    for m in METHODS:
        ts = data[t][m]
        if len(ts) <= 20: continue
        a, b = ts[:20], ts[20:]
        p = fisher(succ(a), len(a), succ(b), len(b))
        if p < 0.05:
            het_flags[(t, m)] = (f'{succ(a)}/{len(a)}', f'{succ(b)}/{len(b)}', p)
if het_flags:
    with open(f'{OUT}/BATCH_HETEROGENEITY_WARNING.txt', 'w') as fh:
        for (t, m), (a, b, p) in het_flags.items():
            fh.write(f'{t}/{m}: original {a} vs top-up {b} DISAGREE (Fisher p={p:.4f}) '
                     f'— DO NOT POOL; report separately and diagnose.\n')
elif os.path.exists(f'{OUT}/BATCH_HETEROGENEITY_WARNING.txt'):
    os.remove(f'{OUT}/BATCH_HETEROGENEITY_WARNING.txt')

# ---- table.csv + history line -------------------------------------------
rows = []
for m in METHODS:
    row = {'method': MLABEL[m]}
    for t, _ in TIERS:
        ts = data[t][m]; s, n = succ(ts), len(ts)
        lo, hi = wilson(s, n)
        row[t] = f'{s}/{n}' + (' SPLIT!' if (t, m) in het_flags else '')
        row[f'{t}_ci'] = f'{lo*100:.0f}-{hi*100:.0f}%' if n else ''
    s, n = succ(abl[m]), len(abl[m])
    row['standing_old_exec'] = f'{s}/{n}'
    rows.append(row)
with open(f'{OUT}/results_table.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

stamp = datetime.datetime.now().strftime('%m-%d %H:%M')
tally = ' | '.join(f"{m}:" + "/".join(str(len(data[t][m])) for t, _ in TIERS) for m in METHODS)
with open(f'{OUT}/history.log', 'a') as fh:
    fh.write(f'[{stamp}] counts(frozen/hang/stand) {tally}\n')

# ---- fig 1: 3-tier grouped bars (rates, since n varies) ------------------
fig, ax = plt.subplots(figsize=(7.5, 4.4))
x = np.arange(len(METHODS)); w = 0.26
for i, (t, _) in enumerate(TIERS):
    vals, errs_lo, errs_hi, labels = [], [], [], []
    for m in METHODS:
        ts = data[t][m]; s, n = succ(ts), len(ts)
        p = s / n * 100 if n else 0
        lo, hi = wilson(s, n)
        vals.append(p); errs_lo.append(p - lo * 100); errs_hi.append(hi * 100 - p)
        labels.append(f'{s}/{n}' if n else '—')
    b = ax.bar(x + (i - 1) * w, vals, w, label=t, color=C[t],
               yerr=[errs_lo, errs_hi], capsize=3, error_kw={'lw': 1, 'alpha': .6})
    for r, lab in zip(b, labels):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 2, lab,
                ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([MLABEL[m] for m in METHODS], fontsize=9)
ax.set_ylabel('success rate [%]'); ax.set_ylim(0, 112)
ax.set_title('Grasp success by method × base condition (Wilson 95% CI)')
ax.legend(fontsize=9); ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OUT}/fig1_three_tier.png', dpi=160); plt.close()

# ---- fig 2: executor ablation (standing old vs new) ----------------------
fig, ax = plt.subplots(figsize=(6.8, 4))
for i, (src, lab, col) in enumerate((
        (abl, 'old executor (pelvis-frame)', C['abl']),
        (data['standing'], 'world-anchored executor', C['standing']))):
    vals, labels = [], []
    for m in METHODS:
        ts = src[m]; s, n = succ(ts), len(ts)
        vals.append(s / n * 100 if n else 0); labels.append(f'{s}/{n}' if n else '—')
    b = ax.bar(x + (i - .5) * .35, vals, .35, label=lab, color=col)
    for r, lab2 in zip(b, labels):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 2, lab2,
                ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([MLABEL[m] for m in METHODS], fontsize=9)
ax.set_ylabel('success rate [%]'); ax.set_ylim(0, 112)
ax.set_title('Standing tier: execution-layer ablation')
ax.legend(fontsize=9); ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OUT}/fig2_executor_ablation.png', dpi=160); plt.close()

# ---- fig 3: precision (tip-to-bar of successes, standing new) ------------
fig, ax = plt.subplots(figsize=(6.4, 4))
for m in METHODS:
    tips = [d['tip_to_handle_mm'] for d in data['standing'][m]
            if d.get('success') and isinstance(d.get('tip_to_handle_mm'), (int, float))]
    if tips:
        ax.scatter([MLABEL[m]] * len(tips), tips, alpha=.6, s=28)
ax.axhline(60, ls='--', lw=1, c='#888'); ax.text(0.01, 61, 'on-handle gate', fontsize=8, c='#888')
ax.set_ylabel('fingertip-to-bar distance [mm]')
ax.set_title('Standing successes: grasp precision per method')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OUT}/fig3_precision.png', dpi=160); plt.close()

# ---- fig 4: failure decomposition (standing new) -------------------------
cats = ['success', 'contact-unstable', 'wander', 'fall', 'miss/other']
cc = {'success': '#55A868', 'contact-unstable': '#C9A227',
      'wander': '#C44E52', 'fall': '#8172B3', 'miss/other': '#937860'}
fig, ax = plt.subplots(figsize=(6.8, 4))
bot = np.zeros(len(METHODS))
counts = {c: [] for c in cats}
for m in METHODS:
    n = {c: 0 for c in cats}
    for d in data['standing'][m]:
        if d.get('success'): n['success'] += 1
        elif d.get('robot_fell_during_trial'): n['fall'] += 1
        elif d.get('base_wandered'): n['wander'] += 1
        elif isinstance(d.get('grip_final_mm'), (int, float)) and 20 < d['grip_final_mm'] < 85:
            n['contact-unstable'] += 1
        else: n['miss/other'] += 1
    for c in cats: counts[c].append(n[c])
for c in cats:
    ax.bar([MLABEL[m] for m in METHODS], counts[c], bottom=bot, label=c, color=cc[c])
    bot += np.array(counts[c])
ax.set_ylabel('trials'); ax.set_title('Standing tier (world-anchored executor): outcome decomposition')
ax.legend(fontsize=8, ncol=2); ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OUT}/fig4_decomposition.png', dpi=160); plt.close()

# ---- fig 5: pelvis traces by outcome (standing new) ----------------------
fig, ax = plt.subplots(figsize=(6.2, 5.2))
kc = {'succ': '#55A868', 'wander': '#C44E52', 'fall': '#8172B3', 'other': '#937860'}
kn = {k: 0 for k in kc}
for m in METHODS:
    for f in sorted(glob.glob(f'{R}/sweep_almi/{m}/trial_*_telemetry.csv')):
        j = f.replace('_telemetry.csv', '.json')
        try: d = json.load(open(j))
        except Exception: continue
        kind = ('succ' if d.get('success') else
                'fall' if d.get('robot_fell_during_trial') else
                'wander' if d.get('base_wandered') else 'other')
        try:
            rows_ = [l.split(',') for l in open(f).read().strip().split('\n')[1:]]
            xs = [float(r[1]) for r in rows_ if r[1] not in ('nan', '')]
            ys = [float(r[2]) for r in rows_ if r[2] not in ('nan', '')]
        except Exception: continue
        if len(xs) < 5: continue
        ax.plot(xs, ys, color=kc[kind], alpha=.3, lw=.9); kn[kind] += 1
ax.plot([3.537], [-2.889], 'k*', ms=13, label='handle')
for k, c in kc.items():
    if kn[k]: ax.plot([], [], color=c, lw=2, label=f'{k} (n={kn[k]})')
ax.set_xlabel('world x [m]'); ax.set_ylabel('world y [m]')
ax.set_title('Standing tier: pelvis trajectories by outcome')
ax.legend(fontsize=8); ax.axis('equal'); ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OUT}/fig5_pelvis_traces.png', dpi=160); plt.close()

print(f'paper_bank regenerated @ {stamp}: ' + tally)


# ---- fig 6: frozen-GT control (perception vs executor/stance decomposition) --
fgt = {m: load('sweep_frozen_gt', m) for m in ('centroid', 'topdown_antipodal')}
if any(fgt.values()):
    fig, ax = plt.subplots(figsize=(7, 4))
    conds = [('frozen + Gemini\n(original)', lambda m: data['frozen'][m][:20]),
             ('frozen + GT crop\n(control)', lambda m: fgt[m]),
             ('standing + GT + world-servo', lambda m: data['standing'][m])]
    x2 = np.arange(2)
    for i, (lab, get) in enumerate(conds):
        vals, labels = [], []
        for m in ('centroid', 'topdown_antipodal'):
            ts = get(m); s2, n2 = succ(ts), len(ts)
            vals.append(s2 / n2 * 100 if n2 else 0); labels.append(f'{s2}/{n2}' if n2 else '—')
        b = ax.bar(x2 + (i - 1) * .28, vals, .28, label=lab)
        for r, lab2 in zip(b, labels):
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 2, lab2, ha='center', fontsize=8)
    ax.set_xticks(x2); ax.set_xticklabels(['centroid', 'pca (top-down)'])
    ax.set_ylabel('success rate [%]'); ax.set_ylim(0, 112)
    ax.set_title('Baseline-method decomposition: perception vs execution vs base')
    ax.legend(fontsize=8); ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout(); plt.savefig(f'{OUT}/fig6_frozen_gt_control.png', dpi=160); plt.close()

# ---- sway battery (fig7-9 + sway_metrics.csv), per researched method spec ----
try:
    import sys
    sys.path.insert(0, '/tmp/claude-1001/-home-guest-Downloads-HAMS-test-grasping/9645ef6d-75d8-4b70-b59a-82ef7165e409/scratchpad')
    import sway_analysis
    sway_analysis.render(OUT)
except Exception as e:
    print(f'sway analysis skipped: {e}')
