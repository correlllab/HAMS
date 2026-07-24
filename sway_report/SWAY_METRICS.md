# Tethered battery sweep — sway metrics

Source: `core_ws/benchmark_results/sweep_battery_tethered`  | approach-axis anchor (screw XY): `(0.45, -0.05)`
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
| centroid | 30 | 100.0 | 100.0 | 100.0 | 0 | 0.816 | 4.91 | 0.694 | 0.427 | 0.251 | 3.389 |
| centroid_vs | 30 | 100.0 | 100.0 | 100.0 | 0 | 0.793 | 4.6 | 0.676 | 0.41 | 0.239 | 3.299 |
| pca | 30 | 0.0 | 0.0 | 6.7 | 28 | 9.456 | 968.047 | 5.903 | 7.172 | 16.942 | 2.508 |
| skill | 30 | 0.0 | 0.0 | 100.0 | 0 | 1.616 | 6.017 | 0.202 | 1.603 | 0.499 | 1.412 |
