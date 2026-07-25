# Standing (ALMI) tier — battery screw-grasp results

**Live snapshot** (centroid_vs still running to 30). The robot balances on the ALMI RL
locomotion policy for the *entire* grasp — band off, pin released, genuinely standing (no
tether). Two methods on the fixed screw (`screw_27`), GT-targeted 80° top-down grasp:

- **centroid** — open-loop (`ALMI_SERVO_ITER=0`): targets the GT screw once, no correction.
- **centroid_vs** — closed-loop visual servo (`ALMI_SERVO_ITER=4`): re-reads GT and corrects
  each descent step.

Telemetry is the pelvis (`__pelvis__`) trace at ~10 Hz, identical method to the tethered/fridge
sweep. Grasp gate = fingers closed within 12 mm of target; success = grasped **and** lifted.

## Grasp rate

| method | trials | grasp secured | success (grab+lift) | mean post-grasp err | mean lift |
|---|---|---|---|---|---|
| centroid (open-loop) | 24 | 95.8 % | **91.7 %** | **8.55 mm** | 0.058 m |
| centroid_vs (servo) | 10* | 100 % | **100 %** | **2.43 mm** | 0.061 m |

\*centroid_vs still running (10/30 at snapshot). centroid is missing trials 11–16 (lost to a
watchdog restart-collision mid-run; being re-run resumably — see repo notes).

**Headline:** both grasp reliably in sim (GT targeting makes open-loop strong), but the
closed-loop servo drives placement error down **~3.5×** (8.55 → 2.43 mm) — the precision
advantage is the real contrast, not a success-rate cliff.

## Pelvis sway (posturography, about trajectory mean)

| metric | centroid | centroid_vs |
|---|---|---|
| duration (s) | 336.4 | 339.6 |
| max displacement (mm) | 21.1 | 20.6 |
| path length (mm) | 169.0 | 172.2 |
| mean speed (mm/s) | 0.50 | 0.51 |
| RMS radial (mm) | 13.76 | 13.53 |
| sway area — 95 % ellipse (mm²) | 1057.7 | 1042.9 |
| RMS longitudinal (mm) | 9.60 | 9.54 |
| RMS lateral (mm) | 9.84 | 9.59 |
| yaw RMS (deg) | 2.96 | 2.80 |
| yaw range (deg) | 7.53 | 7.12 |
| z RMS (mm) | 3.54 | 3.51 |
| mean/median power freq (Hz) | 0.026 | 0.022 |

**Sway is essentially identical between the two methods** (~13.7 mm RMS radial, ~1050 mm²
ellipse) — the servo's extra corrective arm motion does **not** cost extra pelvis sway while
standing on ALMI. So the servo buys 3.5× tighter grasps for the same balance load.

## Files (formatted data)

- `sway_report_standing/per_trial.csv` — one row per trial, every sway + grasp metric.
- `sway_report_standing/per_method.csv` — per-method means of every metric.
- `sway_report_standing/SWAY_METRICS.md` — metric definitions + tables.
- Tethered tier: `sway_report/` (committed earlier).

Raw per-trial JSON + 10 Hz telemetry live under `core_ws/benchmark_results/…` (gitignored;
these formatted CSVs are the durable, shareable form).
