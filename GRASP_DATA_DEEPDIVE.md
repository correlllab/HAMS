# Grasp Study Data Deep-Dive — everything the paper can use

Companion to `GRASP_SWEEP_README.md` (§1–17: protocols & replication). This
file is the *analysis* reference: every number, mechanism, statistical note,
threat-to-validity, and ready-to-use narrative element, each traceable to
files on disk. Generated 2026-07-22 from the complete 240-trial dataset.

---

## 1. Study design

| | |
|---|---|
| Robot | Unitree H1-2 (27 DoF), Magpie parallel-jaw gripper, right arm |
| Sim | RoboCasa/MuJoCo `OpenFridge`, seed 42 (fixed scene), headless |
| Task | Grasp the vertical fridge-door handle (two vertical bars at world x≈3.537, y∈{−2.842, −2.936}, z∈[0.92, 1.55]) |
| Conditions (base) | **frozen** (rigid weld pin), **hanging** (passive elastic tether + leg PD hold), **standing** (ALMI LSTM locomotion policy, actively balancing) |
| Methods (grasp) | **centroid** (cloud centroid + fixed approach), **pca** (`topdown_antipodal`), **graspgenx** (raw GraspGenX candidates, unranked), **skill** (deployed pipeline: GraspGenX candidates → wrist-feasible Y-up re-roll → ranked by wrist error → staged approach → world-anchored drift-compensated servo) |
| Trials | 20 per method × condition = **240**; fresh simulator env per trial (or verified re-tether reset); scene reset on every failure |
| Success criterion | Contact-based, 3 gates: settled aperture ∈ (20, 85) mm **and** 2-sample hold stability (Δ < 8 mm) **and** fingertip-to-bar-axis ≤ 60 mm (world-frame geometric gate) |

## 2. Headline table

| method | frozen | hanging | standing |
|---|---|---|---|
| centroid | 0/20 | 4/20 | 0/20 |
| pca (topdown_antipodal) | 0/20 | 19/20 | 0/20 |
| graspgenx (raw) | 13/20 | 12/20 | 0/20 |
| **skill (deployed pipeline)** | **16/20** | **17/20** | **12/20** |

Hanging skill is post-audit (19 raw − 2 reclassified edge-grabs; §7). All 240
trials graded under the same final criteria (earlier tiers retroactively
re-audited; only those 2 trials flipped study-wide).

### Wilson 95% CIs (n=20 per cell)

| cell | rate | 95% CI |
|---|---|---|
| skill frozen 16/20 | 80% | 58–92% |
| skill hanging 17/20 | 85% | 64–95% |
| skill standing 12/20 | 60% | 39–78% |
| ggx frozen 13/20 | 65% | 43–82% |
| pca hanging 19/20 | 95% | 76–99% |
| any 0/20 | 0% | 0–16% |

**Statistically solid claims:** every 0/20-vs-≥12/20 contrast (Fisher exact
p < 0.001); pca's hanging-vs-standing collapse (19/20 → 0/20, p < 10⁻⁷);
ggx's frozen-vs-standing collapse (13/20 → 0/20, p < 10⁻⁴).
**Not resolved at n=20:** skill frozen 16/20 vs standing 12/20 (p ≈ 0.30) —
phrase as "retains substantial performance," don't quantify the drop as
precise. Skill vs ggx *within* the standing tier: 12/20 vs 0/20, p < 10⁻⁴.

## 3. Grasp-quality metrics (successes only)

| tier / method | settled aperture [mm] | tip-to-bar [mm] | exec time [s] |
|---|---|---|---|
| frozen / ggx | 72.6 ± 3.1 (diagonal holds) | — (pre-dates tip metric) | 46.7 |
| frozen / skill | 59.0 ± 9.2 (canonical) | — | 161.2 |
| hanging / centroid | 71.4 ± 2.4 | — | 49.7 |
| hanging / pca | 56.7 ± 11.4 | — | 40.1 |
| hanging / ggx | 71.9 ± 1.6 | — | 54.5 |
| hanging / skill | 48.4 ± 8.6 (deep grips) | — | 143.2 |
| **standing / skill** | **54.6 ± 0.7** | **31.1 ± 7.4** | **73.9** |

Notable:
- **Aperture fingerprints separate grasp styles cleanly:** ~54–59 mm =
  canonical perpendicular bar grip (bar width); ~72 mm = GraspGenX diagonal
  hold; **34.3 mm = door-edge clamp** (sheet-metal thickness — this signature
  identified every wrong-target grab in the study).
- Standing-tier successes are the *tightest* cluster in the study (±0.7 mm):
  when the deployed pipeline succeeds on the dynamic base, it succeeds in
  exactly one way — the canonical grip. Stable bases admit sloppier variety.
- Exec-time caveat: cross-tier timing is confounded (frozen/hanging used
  Gemini+SAM perception, ~10–40 s/detect with retries; standing used the GT
  crop, ~0.5 s). Compare within-tier only.

## 4. Standing-tier failure decomposition (bag-verified)

| method | success | contact-unstable | wander | wander+fall | miss/other |
|---|---|---|---|---|---|
| skill | 12 | 4 | 2 | 2 | 0 |
| graspgenx | 0 | 12 | 2 | 2 | 4 |
| centroid | 0 | 19 | 0 | 0 | 1 |
| pca | 0 | 0 | 20 | 0 | 0 |

Mechanisms, each with raw evidence:

1. **Contact-unstable (ggx's 12):** grasps *reached the bar and closed*
   (apertures 43–55 mm) but the hold churned — bag-recorded aperture spread
   30+ mm over the final 2 sim-s (threshold: 8 mm). Physical mechanism: after
   close, the benchmark executor holds a **pelvis-relative** arm target while
   the handle is world-fixed; base sway shears the grip. The skill's
   world-anchored hold shows ≤ 0.2 mm spread on the same base. Files:
   `sweep_almi/graspgenx/trial_{01,04,05,07,08,09}.json` + bags;
   regrade: `bag_regrade.py` output.
2. **Wander (pca's 20/20):** the base walks off-station before fingertip
   contact. pca commands the deepest targets in the study (pelvis-x cluster
   0.445 + 0.06 offset), i.e. the longest fully-extended reaches → maximum
   balance excitation → ALMI's recovery steps accumulate (it has **no
   station-keeping below its 0.1 m/s command threshold**). Watchdog kills at
   >0.32 m; every flagged trial cross-validated against telemetry (24 checked,
   0 contradictions).
3. **Wander+fall (2 skill, 2 ggx):** in-grasp destabilization on marginal
   candidates — servo traces show world-frame error exploding mid-iteration
   (e.g., 47 mm → 268 mm as the base tips; `trial_*_skillslog.txt`).
4. **Door-edge clamps (centroid's 19):** the 34.3 mm signature at 65–94 mm
   from the bar axis; centroid's fixed normal-based approach targets the face
   centroid region in every tier.

## 5. The three-regime result (the paper's core curve)

pca across tiers: **0/20 → 19/20 → 0/20.** Same method, same perception
budget, same handle. Three physical regimes:

- **Frozen = kinematic regime.** The rigid base makes the H1-2 wrist-pitch
  limit (±26.5°) binding: centroid/pca propose orientations that are
  IK-infeasible → uniform `"no reachable candidate"` (nothing executed).
- **Hanging = compliant regime.** Pulling the arm drags the torso; the
  effective workspace grows; orientation infeasibility dissolves (pca
  0→19/20, centroid 0→4/20). *Passive compliance transforms reachability.*
- **Standing = dynamic regime.** The base neither holds (no frozen precision)
  nor yields (no workspace gift) — it *reacts*: reaches excite recovery
  stepping; holds are sheared by sway. Methods tuned for either static regime
  collapse.

Skill is the only method whose row is non-zero in all three regimes
(16/17/12): candidate ranking neutralizes the kinematic regime, and
drift-compensated execution + anchored holds survive the dynamic one.

## 6. Standing base characterization (independent of grasping)

- **Policy:** ALMI LSTM (`policy_lstm_12800.pt`, TorchScript), 50 Hz, 12-DoF
  leg commands, elastic-band-free after engage.
- **Inference cost is negligible:** mean 0.367 ms, p50 0.08 ms, p95 0.15 ms,
  max 24 ms (one-off scheduler hit) vs the 20 ms budget at 50 Hz (CPU!). File:
  `sweep_almi/almi_inference.json`. The tier's failures are *dynamics*, not
  compute.
- **Quiet stance:** upright=1.000, pelvis z 0.97 m, drift < 4 cm indefinitely
  when undisturbed (free-stand diagnostic, 45 sim-s watch).
- **Under interaction:** no station-keeping below the 0.1 m/s command
  threshold → recovery steps accumulate directionally under sustained arm
  activity. Pelvis trajectories per trial in `trial_*_telemetry.csv` (10 Hz);
  aggregate figure `figures/almi_pelvis_traces.png`.
- **Drift-equilibrium under servoing:** with the base retreating during arm
  moves, the world-frame servo converges to a systematic ~65 mm shortfall
  along the approach axis (error ≈ drift-rate × move-duration; iteration
  cannot close a moving equilibrium). Measured from executed-frame tip
  positions vs bar geometry across v5/v6 validation trials; compensated by
  `HAMS_GRASP_OFFSET=0.06`, whose value re-centers commanded poses into the
  ±5 cm success band derived from the **83 stable-base successes**
  (commanded-pose clusters: frozen-skill x̄=0.358 ± 0.025 m,
  hang-skill 0.412 ± 0.029, hang-pca 0.445 ± 0.019 — the band's width is
  itself a usable result: the gripper–bar pair tolerates ~±5 cm along the
  approach).

## 7. Grading integrity (what a reviewer can check)

- **Criteria evolution is documented and closed:** aperture-band → +2-sample
  stability → +on-handle geometric gate. Earlier tiers re-audited under final
  criteria; exactly **2 flips in 240 trials** (hanging-skill edge-grabs,
  19→17). Frozen 0/20s are criteria-independent (nothing executed).
- **Sim-time regrade audit:** the in-run stability samples land at +1/+3 s
  *wall*-clock (≈0.05/0.15 sim-s on the slow standing tier). All 80 standing
  trials re-graded from bags at true sim-time (`bag_regrade.py`): **zero
  flips**. Not borderline: failed contacts churn 30+ mm vs the 8 mm
  threshold; successes spread ≤ 0.2 mm.
- **Wander flags:** 24 cross-validated vs telemetry, 0 contradictions.
- **Completeness:** 240/240 result JSONs; standing tier additionally 80/80
  telemetry CSVs + bags + servo logs; 52/80 post-trial head JPGs (the 28
  absent are exactly the watchdog-killed trials; head video persists in
  bags); 3/80 bags unreadable (teardown race).

## 8. Threats to validity (disclosed, with bounding controls)

1. **Systems vs algorithms** (§17.1 README): `skill` bundles selection AND a
   better executor; baselines share a simpler executor. The standing column
   compares *pipelines*. Bounding control: unranked ggx candidates through
   the skill executor, n=20 (~3.5 h).
2. **Uniform grasp offset:** +0.06 m was calibrated on the skill's measured
   shortfall and applied to all methods; it deepens pca's already-deepest
   targets and may amplify its wander rate. Bounding control: pca standing
   at offset 0, n=20.
3. **Perception asymmetry:** standing used the GT world-point crop (all
   methods); frozen/hanging used Gemini+SAM. Direction is conservative for
   the headline (standing got perception free and still collapsed), but
   forbids cross-tier perception-time comparisons.
4. **Scope:** one object/scene/seed, sim-only, n=20/cell; the H1-2 wrist
   limit drives the frozen regime and is platform-specific.
5. **Protocol hazards encountered and fixed** (candidates for a "lessons"
   subsection): wall-vs-sim-clock timing; GT origins inside bodies; qpos
   teleports catapulting free bases (incl. via PD-handoff at pin release);
   model-server load spikes starving the sim (≈control-latency-induced policy
   failure); orphaned-process container races. Each reproduced, diagnosed,
   and countered; details in README §17.

## 9. Ready-to-use narrative elements

- *One-sentence result:* *"Across 240 trials, an actively-balancing humanoid
  base eliminated three of four grasp-synthesis baselines entirely (0/20
  each) while the deployed wrist-aware ranked pipeline retained 60% success —
  and the method rankings inverted between compliant and dynamic base
  regimes."*
- *The regime curve:* *"The same top-down antipodal method scores 0/20,
  19/20, and 0/20 as the base changes from rigid to compliant to
  actively-balancing — grasp synthesis cannot be evaluated apart from the
  base regime it will be executed from."*
- *The hold-shear mechanism:* *"Raw candidates reached the handle 12/20 times
  from the standing base but never held it: a pelvis-referenced grip on a
  world-fixed handle is sheared by every sway cycle (30+ mm aperture churn),
  while world-anchored holds settle to ±0.2 mm."*
- *The station-keeping gap:* *"Locomotion policies expose no interface for
  'stand still under interaction': below the walking threshold, commands are
  ignored, and balance recovery steps accumulate directionally under
  sustained arm activity — walking the robot off its own workstation."*
- Figures ready: `figures/three_tier_success.png` (headline bars),
  `figures/almi_pelvis_traces.png` (wander-by-outcome). Overlay tooling for
  grasp-line renders: `aggregate.py`.

## 10. Data inventory (provenance map)

```
core_ws/benchmark_results/
├── sweep/                 frozen tier    (JSON + logs per trial)
├── sweep_unfrozen/        hanging tier   (JSON + logs per trial)
├── sweep_almi/            standing tier  (per trial: result JSON,
│     10 Hz telemetry CSV [pelvis/grip/door], rosbag [object poses, gripper
│     state, head camera, joint states], skills-node servo log, head JPG)
│   └── almi_inference.json   policy latency benchmark (n=1000)
├── figures/               three_tier_success.png, almi_pelvis_traces.png
GRASP_SWEEP_README.md      full protocol + replication (§1–17.2)
bag_regrade.py             sim-time hold-stability regrade (audit tool)
unfrozen_sweep_host.sh     the tier harness (freeze-hold protocol)
almi_engage.py             sim-time settle-and-verify release
```

Every number in this document regenerates from those files; no state lives
outside the repo except container images.
