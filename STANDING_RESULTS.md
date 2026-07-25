# Standing (ALMI) tier — battery screw-grasp results

**Complete.** The robot balances on the ALMI RL locomotion policy for the *entire* grasp —
band off, pin released, genuinely standing (no tether). Two methods on the fixed screw
(`screw_27`), GT-targeted 80° top-down grasp:

- **centroid** — open-loop (`ALMI_SERVO_ITER=0`): targets the GT screw once, no correction.
- **centroid_vs** — closed-loop visual servo (`ALMI_SERVO_ITER=4`): re-reads GT and corrects
  each descent step.

Telemetry is the pelvis (`__pelvis__`) trace at ~10 Hz, identical method to the tethered/fridge
sweep. Grasp gate = fingers closed within 12 mm of target; success = grasped **and** lifted.

## Grasp rate

| method | trials | grasp secured | success (grab+lift) | mean post-grasp err | mean lift |
|---|---|---|---|---|---|
| centroid (open-loop) | 24† | 95.8 % | **91.7 %** | **8.55 mm** (±1.03) | 0.058 m |
| centroid_vs (servo) | 30 | 100 % | **100 %** | **2.46 mm** (±0.76) | 0.061 m |

† centroid is n=24: trials 11–16 were lost to a watchdog restart-collision mid-run. They can be
back-filled in ~30 min with one resumable re-run (`bash sweep_standing.sh` skips everything with
a JSON and only redoes 11–16). All other trials are clean.

**Headline:** both grasp reliably in sim (GT targeting makes open-loop strong), but the
closed-loop servo drives placement error down **~3.5×** (8.55 → 2.46 mm) — the precision
advantage is the real contrast, not a success-rate cliff.

## Pelvis sway (posturography, about trajectory mean)

| metric | centroid | centroid_vs |
|---|---|---|
| duration (s) | 336.4 | 346.8 |
| max displacement (mm) | 21.1 | 20.7 |
| path length (mm) | 169.0 | 172.4 |
| mean speed (mm/s) | 0.50 | 0.50 |
| RMS radial (mm) | 13.76 | 13.61 |
| sway area — 95 % ellipse (mm²) | 1057.7 | 1049.6 |
| RMS longitudinal (mm) | 9.60 | 9.60 |
| RMS lateral (mm) | 9.84 | 9.64 |
| yaw RMS (deg) | 2.96 | 2.81 |
| yaw range (deg) | 7.53 | 7.16 |
| z RMS (mm) | 3.54 | 3.52 |
| mean/median power freq (Hz) | 0.026 | 0.022 |

**Sway is essentially identical between the two methods** (~13.7 mm RMS radial, ~1050 mm²
ellipse) — the servo's extra corrective arm motion does **not** cost extra pelvis sway while
standing on ALMI. So the servo buys 3.5× tighter grasps for the same balance load.

## Files (formatted data)

- `sway_report_standing/per_trial.csv` — one row per trial, every sway + grasp metric.
- `sway_report_standing/per_method.csv` — per-method means of every metric.
- `sway_report_standing/SWAY_METRICS.md` — metric definitions + tables.
- Tethered tier: `sway_report/`. Full study: `STUDY_SUMMARY.md`.

Raw per-trial JSON + 10 Hz telemetry live under `core_ws/benchmark_results/…` (gitignored;
these formatted CSVs are the durable, shareable form).
