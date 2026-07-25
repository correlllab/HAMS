# Tethered battery sweep — sway metrics

Source: `core_ws/benchmark_results/sweep_battery_standing`  | approach-axis anchor (screw XY): `(0.45, -0.05)`
Sway is measured on the pelvis (`__pelvis__`) trace at ~10 Hz, about the trajectory MEAN (the station the tether held). Posturography math identical to the fridge study.

## Files
- `per_trial.csv` — one row per trial, every metric (the "full CSV, each step")
- `per_method.csv` — per-method mean of every metric

## Metric definitions

| metric | meaning |
|---|---|
| `dur_s` | trial telemetry duration (s) |
| `max_disp_mm` | max pelvis displacement from mean station (mm) |
| `path_len_mm` | total pelvis path length in XY (mm) — cumulative wander |
| `mean_speed_mms` | path_len/dur (mm/s) |
| `rms_radial_mm` | RMS radial sway about the held station (mm) — overall sway magnitude |
| `sway_area_mm2` | 95% confidence-ellipse area of pelvis XY scatter (mm²) — sway spread |
| `rms_long_mm` | RMS sway along the approach axis to the target (mm) |
| `rms_lat_mm` | RMS sway perpendicular to the approach axis (mm) |
| `ellipse_ratio` | minor/major axis ratio of the sway ellipse (1=isotropic) |
| `yaw_rms_deg` | RMS pelvis yaw (deg) — twisting |
| `yaw_range_deg` | peak-to-peak pelvis yaw (deg) |
| `z_rms_mm` | RMS pelvis vertical motion (mm) — bob |
| `mpf_hz` | mean power frequency of radial sway (Hz) |

## Per-method summary

| method | n | success% | good% | closed% | err | rms_radial | sway_area | rms_long | rms_lat | yaw_rms | z_rms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| centroid | 24 | 91.7 | 95.8 | 0.0 | 0 | 13.756 | 1057.65 | 9.602 | 9.835 | 2.955 | 3.535 |
| centroid_vs | 30 | 100.0 | 100.0 | 0.0 | 0 | 13.609 | 1049.567 | 9.601 | 9.643 | 2.813 | 3.523 |

## CoM / margin-of-stability battery

Extrapolated-CoM (XCoM) margin-of-stability battery, exact-method reproduction of the fridge study. CoM reference = base/pelvis pose (same code runs in sim and on the real robot). Preprocessing: uniform 10 Hz grid → 2.5 Hz 4th-order zero-phase Butterworth → task frame toward the target `(0.45, -0.05)`, recentered to trial start, no detrend. Constants: L_COM=0.97 m, g=9.81, support rectangle ±0.24/±0.16 m (AP/ML).

| metric | meaning |
|---|---|
| `mvelo_mms` | mean sway velocity = task-frame path length / duration (mm/s) |
| `rms_ap_mm` | RMS anteroposterior (toward-target) sway about the mean (mm) |
| `rms_ml_mm` | RMS mediolateral sway about the mean (mm) |
| `a95_mm2` | 95% sway-ellipse area of the AP/ML scatter (mm²) |
| `peak_r_mm` | peak radial excursion from the mean station (mm) |
| `drift_mm` | net planar displacement, first→last sample in task frame (mm) |
| `min_mos_mm` | min margin of stability = extrapolated-CoM (XCoM = pos + vel/√(g/L)) distance inside the fixed support rectangle (mm); **negative = a recovery step was dynamically required** |

## Per-method CoM/MoS summary

| method | n | mvelo_mms | rms_ap | rms_ml | a95_mm2 | peak_r | drift | min_mos_mm |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| centroid | 23 | 0.51 | 9.662 | 9.597 | 1050.743 | 20.878 | 3.865 | 136.174 |
| centroid_vs | 30 | 0.507 | 9.785 | 9.456 | 1048.167 | 20.61 | 1.937 | 133.783 |
