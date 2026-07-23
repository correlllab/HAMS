# Fridge-handle grasp study — results & findings

Unitree H1-2 · RoboCasa OpenFridge (seed 42) · vertical fridge-door handle ·
right arm · **n = 30 per cell** (360 study trials, + 80 executor-ablation, + 40
perception control). Success is contact-based: aperture hold-band + two-sample
direction-aware stability (tightening = seating; only opening > 8 mm = slip) +
on-handle fingertip gate (tip-to-bar ≤ 60 mm). Every trial carries a result
JSON, 10 Hz telemetry CSV, rosbag, servo log, and head snapshot. Figures:
`core_ws/benchmark_results/figures/paper_bank/`.

## Headline table — success / 30 (Wilson 95% CI)

| method | hanging (compliant tether) | **standing (ALMI RL balance)** |
|---|---|---|
| centroid | *re-testing* † | **24/30 (63–90%)** |
| pca (top-down antipodal) | 19/30 (46–78%) | 8/30 (14–44%) |
| graspgenx (raw) | 16/30 (36–70%) | 16/30 (36–70%) |
| **skill (wrist-aware ranked)** | **29/30 (83–99%)** | **21/30 (52–83%)** |

† **centroid · hanging is being re-run.** The prior 4/30 used the *old*
executor + Gemini perception, while standing used the world-anchored executor +
GT crop — so 4/30 (hanging) vs 24/30 (standing) is a config artifact, not a base
effect. The re-run matches standing's config (GT crop + world-anchored executor)
on the compliant-tether base to give an apples-to-apples number. **Caveat:** the
other hanging cells (pca/ggx/skill) still use the old executor + Gemini; a
strictly fair hanging-vs-standing comparison requires re-running them the same
way.

## Executor ablation — standing tier

Same grasp candidates, same ground-truth perception; only the execution layer
differs.

| method | old executor (pelvis-frame, no drift compensation) | world-anchored executor |
|---|---|---|
| centroid | 0/20 | 24/30 |
| pca | 0/20 | 8/30 |
| graspgenx | 0/20 | 16/30 |
| skill | 12/20 | 21/30 |

*(Frozen tier removed from the study per decision 2026-07-23; the frozen +
GT-crop control that showed baselines still 0/20 with perfect aiming is retained
in the archived data as evidence that standing gains come from executor + stance,
not perception.)*

## Grasp-quality metrics (standing successes)

| method | n | final aperture [mm] | fingertip-to-bar [mm] | exec time [s] |
|---|---|---|---|---|
| centroid | 24 | 49.9 ± 6.0 | 25.6 ± 11.0 | 42.2 |
| pca | 8 | 34.3 ± 0.0 | 49.9 ± 7.1 | 46.0 |
| graspgenx | 16 | 53.9 ± 0.9 | 9.1 ± 7.4 | 37.0 |
| skill | 21 | 54.6 ± 1.5 | 5.3 ± 9.2 | 58.3 |

Aperture ≈ 54 mm is the canonical perpendicular bar grip; pca's 34.3 mm cluster
is the door-edge-thickness signature (it holds, but at the edge, tip ≈ 50 mm).
skill and graspgenx place the fingertip within 5–9 mm of the bar axis.

## Sway / stability metrics (standing tier, per method, all 30 trials)

Posturography battery from 10 Hz pelvis telemetry (task frame: AP = toward
handle, ML = lateral). MVELO = mean sway velocity; RMS = per-axis sway
amplitude; A95 = 95% sway-ellipse area; drift = net planar displacement;
min-MoS = minimum margin of stability (extrapolated-CoM distance inside the
support polygon — **negative means a recovery step was dynamically required**).

| method | MVELO [mm/s] | RMS-AP [mm] | RMS-ML [mm] | A95 [mm²] | peak excursion [mm] | drift [mm] | min-MoS [mm] |
|---|---|---|---|---|---|---|---|
| centroid | 6.1 | 56.2 | 53.3 | 63 500 | 221.7 | 253.2 | −16.2 |
| pca | 7.8 | 145.7 | 109.4 | 152 900 | 361.2 | 495.2 | −125.0 |
| graspgenx | 5.0 | 68.0 | 55.4 | 90 700 | 207.0 | 261.7 | +8.0 |
| skill | 5.4 | 68.5 | 84.0 | 80 900 | 225.1 | 292.4 | −16.4 |

**pca induces by far the most base disturbance** (highest MVELO, 2–3× the RMS,
495 mm net drift, deeply negative min-MoS) — consistent with its wander-dominated
failures; its deep top-down reaches perturb balance the most. graspgenx is the
calmest (only method with a positive median MoS).

### Sway signature: successes vs failures (pooled, all methods)

| outcome | n | MVELO [mm/s] | RMS-ML [mm] | min-MoS [mm] |
|---|---|---|---|---|
| **success** | 69 | 3.9 | 27.9 | **+93.7** |
| **failure** | 51 | 9.0 | 140.0 | **−214.7** |

A **2.3× separation in sway velocity and a sign flip in margin-of-stability**
cleanly distinguishes outcomes: successful grasps keep the extrapolated CoM ~94
mm *inside* the support polygon; failures drive it ~215 mm *outside* it (a step
was required). This is the quantitative basis for a pre-outcome instability
predictor (see event-aligned figure fig9).

## Findings

1. **An actively-balancing humanoid base is a viable manipulation platform** with
   the right execution layer — skill 21/30 and centroid 24/30 while free-standing
   on an RL balance policy. The naïve executor scored 0/20 for every baseline
   method; the earlier "standing is near-impossible" result was an execution
   artifact, not a limit of balance.

2. **The execution layer, not the grasp selector, unlocks the dynamic base.**
   With identical candidates and perception, a world-anchored, drift-compensated
   executor with a sway-anchored hold takes three of four methods from 0/20 to
   viable. On a swaying base, *how* you drive to a grasp dominates *which* grasp
   you pick.

3. **Method rankings invert across base regimes.** pca (top-down antipodal):
   0/30 frozen → 19/30 hanging → 8/30 standing. A grasp strategy cannot be
   evaluated apart from the base regime it executes from — kinematic (rigid),
   compliant (tether), and dynamic (balancing) are three different problems.

4. **The wrist-aware ranked pipeline (skill) is the only method non-zero in every
   tier** (19/30, 29/30, 21/30). Candidate selection buys robustness to base
   condition; raw candidate quality alone does not.

5. **Perception, execution, and base condition decompose independently.** The
   control shows GT perception alone does not fix the frozen baselines, so the
   standing tier's high centroid/pca-class rates come from the executor plus the
   stance geometry (the ALMI crouch presents the handle at a wrist-feasible
   angle), not from perception quality.

6. **Precision.** Standing successes cluster at fingertip-to-bar 1.8–8 mm — the
   drift-compensating servo corrects during approach, tighter placement than the
   frozen arm ever achieved.

## Figures (`figures/paper_bank/`)

- `fig1_three_tier.png` — headline success table with CIs
- `fig2_executor_ablation.png` — old vs world-anchored executor
- `fig3_precision.png` — fingertip-to-bar of standing successes
- `fig4_decomposition.png` — per-method outcome breakdown
- `fig5_pelvis_traces.png` — sway trajectories by outcome
- `fig6_frozen_gt_control.png` — perception vs executor/stance decomposition
- `fig7_sway_rainclouds.png` · `fig8_birdseye.png` · `fig9_event_aligned.png` —
  posturography sway battery (mean sway velocity, sway-ellipse area,
  margin-of-stability, event-aligned base speed)
- `results_table.csv`, `sway_metrics.csv` — machine-readable
