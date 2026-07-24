# Nav-random tier — run on the second computer

Runs the **same** ALMI standing battery grasp as the standing tier, but each trial
teleports the robot to a **randomized standing position** (Gaussian around the
nominal spawn) so it reaches the fixed screw from a different pose. 30 trials each
for `centroid` (open-loop) + `centroid_vs` (servo). Same validated grasp config,
same 10 Hz pelvis-sway telemetry.

> ⚠️ This script is set up but **was not runnable-tested on this machine** (the main
> box was busy running the standing tier). **Validate 1–2 trials visually first.**

## 1. Prereqs on the second computer
- This repo checked out (same worktree), Docker images built (`hams_sim_robocasa`, `hams_ros`).
- `docker/.env` set to the standing-tier config:
  ```
  ROS_DOMAIN_ID=27
  HAMS_FREEZE_BODY=1
  HAMS_STANCE=almi
  HAMS_SPAWN_FORWARD=0.18
  HAMS_SPAWN_LATERAL=0.10
  HAMS_SPAWN_LOWER=0.07
  HAMS_CAMERAS=0
  ```

## 2. Bring up the sim + ALMI (one command)
```bash
bash restart_standing.sh      # recreates frozen BatteryWorkcell sim + ros bringup + ALMI controller
```
Confirm: `sudo docker ps | grep hams` (both up), and in `hams_ros`:
`ros2 action list | grep frame_task` (1) and `pgrep -f lowerbody_controller_node` (>0).

## 3. Validate ONE trial before the full run
```bash
sudo docker exec -d hams_ros bash -lc 'source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; cd /tmp && \
  HAMS_PLACE_BASE=0.08,-0.04 ALMI_HOVER_FROZEN=0 STAB_SIM=30 ALMI_DESCEND_STEPS=6 ALMI_MOVE_SEC=8 ALMI_SERVO_ITER=4 \
  BATT_WELD_MAX_ERR_MM=12 BATT_GRIP_FORCE_N=6 BATT_TOPDOWN_PITCH=80 BATT_HOVER_M=0.10 BATT_FINGER_SINK=-0.002 \
  OUT=/tmp/r_test.json python3 /tmp/almi_grasp.py screw_27 > /tmp/r_test.log 2>&1'
# watch: sudo docker exec hams_ros grep -E '\[almi\]|RESULT' /tmp/r_test.log
```
Want to see: `nav-random: base placed at spawn+(...)`, the base released SOLID, servo
converges, `good_grasp:true`, `success:true`. If the base placement topples it, reduce
the sigmas (step 4) so the offsets are smaller.

## 4. Run the full sweep (detached, ~6–7h)
```bash
SWEEP_N=30 nohup setsid bash sweep_random.sh >/dev/null 2>&1 &
tail -f sweep_random.log       # progress: per-trial spawn + success
# optional smaller spread: RAND_SIG_FWD=0.08 RAND_SIG_LAT=0.05 SWEEP_N=30 bash sweep_random.sh
```
Also start the watchdog: `nohup setsid bash sim_watchdog_standing.sh >/dev/null 2>&1 &`

## 5. Outputs (host-persistent, bind-mounted)
`core_ws/benchmark_results/sweep_battery_random/{centroid,centroid_vs}/`:
- `trial_NN.json` — grasp result
- `trial_NN_telemetry.csv` — 10 Hz pelvis + grip trajectory (sway measurement)
- `trial_NN_spawn.json` — the randomized (fwd,lat) offset for provenance
- `trial_NN.log`

## 6. Bring the results back
Copy `core_ws/benchmark_results/sweep_battery_random/` to the main machine, then it
folds into the final report:
```bash
python3 make_sway_report.py core_ws/benchmark_results/sweep_battery_random 0.45,-0.05
```

## How the randomization works
`sweep_random.sh` samples a deterministic Gaussian `(fwd, lat)` per (method,trial),
sets `HAMS_PLACE_BASE="fwd,lat"`, and `almi_grasp.py` publishes `/hams/place_base`
right after the frozen scene reset — teleporting the **pinned** base to spawn+(fwd,lat)
**before** the ALMI warmup, so the policy stands up and reaches from the offset pose.
`HAMS_PLACE_BASE` unset = nominal spawn (identical to the standing tier).
