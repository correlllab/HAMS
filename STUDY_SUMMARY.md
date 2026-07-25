# Battery workcell screw-grasp study — compiled results

Unitree H1-2 in RoboCasa/MuJoCo. Task: grasp a fixed battery screw (`screw_27`) with a
GT-targeted 80° top-down grasp, comparing an **open-loop** grasp (`centroid`) against a
**closed-loop visual-servo** grasp (`centroid_vs`) across **balance tiers** of increasing
difficulty. Pelvis (`__pelvis__`) sway recorded at 10 Hz; posturography identical to the
fridge study. Grasp gate = fingers closed within 12 mm of target; success = grasped **and**
lifted (≥ ~0.05 m).

Tiers:
- **Tethered** — pelvis held by an elastic band (near-rigid); isolates the arm/grasp.
- **Standing (ALMI)** — robot balances on the ALMI RL locomotion policy for the *entire*
  grasp, band off, pin released (genuinely standing, no cheat).
- **Nav-random** — 6 random standing base positions × 5 (running on the 2nd machine).

## Grasp performance

| Tier | Method | n | Grasp | Success | Post-grasp err (mean) | Lift |
|---|---|---|---|---|---|---|
| Tethered | centroid (open-loop) | 30 | 100% | 100% | 10.62 mm | 0.089 m |
| Tethered | centroid_vs (servo) | 30 | 100% | 100% | **5.32 mm** | 0.087 m |
| Tethered | pca | 30 | 0% | 0% | — (28 errored) | — |
| Tethered | skill | 30 | 0% | 0% | — | — |
| Standing | centroid (open-loop) | 24† | 95.8% | 91.7% | 8.55 mm (±1.03) | 0.058 m |
| Standing | centroid_vs (servo) | 30 | 100% | 100% | **2.46 mm (±0.76)** | 0.061 m |
| Nav-random | centroid / centroid_vs | — | *pending (2nd machine)* | | | |

† Standing centroid is n=24: trials 11–16 were lost to a watchdog restart-collision mid-run
and can be back-filled with one resumable re-run (`bash sweep_standing.sh`). All other trials
are clean. `pca`/`skill` used the fridge perception harness (head camera), which cannot see the
tiny screw — 0'd out by design, not a regression.

## Pelvis sway (posturography, about trajectory mean)

| Tier | Method | RMS radial | Sway area (95% ellipse) | Max disp | Path len | Yaw RMS | z RMS |
|---|---|---|---|---|---|---|---|
| Tethered | centroid | 0.82 mm | 4.9 mm² | 4.5 mm | 77.0 mm | 0.25° | 3.39 mm |
| Tethered | centroid_vs | 0.79 mm | 4.6 mm² | 4.6 mm | 79.3 mm | 0.24° | 3.30 mm |
| Standing | centroid | 13.76 mm | 1057.7 mm² | 21.1 mm | 169.0 mm | 2.96° | 3.54 mm |
| Standing | centroid_vs | 13.61 mm | 1049.6 mm² | 20.7 mm | 172.4 mm | 2.81° | 3.52 mm |

(pca/skill sway omitted — their runs errored/closed on air, so pelvis traces aren't meaningful.)

## Findings

1. **Closed-loop servo tightens placement, more so as balance gets harder.** centroid_vs roughly
   halves post-grasp error in the tethered tier (10.6 → 5.3 mm) and cuts it **~3.5×** in the
   standing tier (8.6 → 2.5 mm). Open-loop drifts while ALMI shifts the base to balance the
   reach; the servo re-reads and corrects each descent step.
2. **The tier exposes real sway.** Band-tethered pelvis is nearly pinned (~0.8 mm RMS, ~5 mm²
   ellipse); standing on ALMI shows honest humanoid sway (~13.7 mm RMS, ~1050 mm² — **~17×
   larger area**). Within the standing tier, the servo's extra corrective arm motion costs
   **no** additional sway (13.61 vs 13.76 mm RMS) — tighter grasps for the same balance load.

## Files

| what | path |
|---|---|
| this summary | `STUDY_SUMMARY.md` |
| standing tier writeup | `STANDING_RESULTS.md` |
| standing sway (per-trial + per-method + doc) | `sway_report_standing/` |
| tethered sway (per-trial + per-method + doc) | `sway_report/` |
| raw per-trial JSON + 10 Hz telemetry | `core_ws/benchmark_results/…` (gitignored) |

Raw benchmark data stays gitignored; the per-trial/per-method CSVs are the durable, shareable form.
