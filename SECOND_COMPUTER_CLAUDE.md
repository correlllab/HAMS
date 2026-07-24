# Claude runbook — run the NAV-RANDOM tier on this (second) computer

You are Claude Code on a second machine, running in parallel with the main box. Your
job: run the **nav-random tier** of a battery-workcell screw-grasp study, then hand the
results back. The tethered + standing tiers are already done on the main machine; this
is the 3rd tier. Follow these steps; report a short status after each.

## Context (what this tier is)
The robot stands on the **ALMI** RL locomotion policy (genuinely balancing — band off,
no pin) and grasps a fixed screw (`screw_27`) with a ground-truth-targeted top-down
grasp. The nav-random tier repeats that grasp from **6 random standing positions × 5
repeats = 30 trials**, for two methods: `centroid` (open-loop) and `centroid_vs`
(closed-loop GT servo). 10 Hz pelvis-sway telemetry is recorded per trial. The 6
positions are picked once and are identical for both methods (fair comparison).

## Hard constraints
- sudo password is `Unitreeh12`.
- Work ONLY in this worktree (`/home/guest/HAMS-grasp-2method` or wherever this repo is).
  **NEVER touch `/home/guest/Downloads/HAMS-test-grasping`** (a live checkout).
- All the scripts + the tuned harness are already in this repo. Do not re-tune the grasp.

## Step 1 — set docker/.env (standing-tier config)
Ensure these lines in `docker/.env` (edit if needed):
```
ROS_DOMAIN_ID=27
HAMS_FREEZE_BODY=1
HAMS_STANCE=almi
HAMS_SPAWN_FORWARD=0.18
HAMS_SPAWN_LATERAL=0.10
HAMS_SPAWN_LOWER=0.07
HAMS_CAMERAS=0
```

## Step 2 — bring up sim + ros + ALMI (one command)
```bash
bash restart_standing.sh
```
Verify (all must pass):
```bash
echo Unitreeh12 | sudo -S docker ps --format '{{.Names}} {{.Status}}' | grep hams   # both up
echo Unitreeh12 | sudo -S docker exec hams_ros bash -lc 'source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; ros2 action list | grep -c frame_task; pgrep -c -f lib/h12_lowerbody_rl/lowerbody_controller_node'
# want frame_task>=1 and almi node>=1
```

## Step 3 — VALIDATE one trial before the full run (do NOT skip)
```bash
echo Unitreeh12 | sudo -S docker cp almi_grasp.py hams_ros:/tmp/almi_grasp.py
echo Unitreeh12 | sudo -S docker cp battery_bench.py hams_ros:/tmp/battery_bench.py
echo Unitreeh12 | sudo -S docker exec -d hams_ros bash -lc 'source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; cd /tmp && HAMS_PLACE_BASE=0.08,-0.04 ALMI_HOVER_FROZEN=0 ALMI_ENGAGE_RETRY=10 STAB_SIM=30 ALMI_RELEASE_SETTLE=6 ALMI_PELVIS_MIN=0.92 ALMI_REACH_SETTLE=6 ALMI_DESCEND_STEPS=6 ALMI_MOVE_SEC=8 ALMI_DESCEND_SETTLE=3 ALMI_SERVO_ITER=4 ALMI_CONV_MM=6 BATT_WELD_MAX_ERR_MM=12 BATT_GRIP_FORCE_N=6 BATT_TOPDOWN_PITCH=80 BATT_HOVER_M=0.10 BATT_FINGER_SINK=-0.002 BATT_HOLD_SEC=1.5 OUT=/tmp/r_test.json python3 almi_grasp.py screw_27 > /tmp/r_test.log 2>&1; echo DONE >> /tmp/r_test.log'
# poll until DONE, then:
echo Unitreeh12 | sudo -S docker exec hams_ros bash -lc "grep -E '\[almi\]|RESULT' /tmp/r_test.log"
```
Want to see: `nav-random: base placed at spawn+(...)`, released SOLID (pelvis>0.92,
upright~1.0), `[almi] servo it` lines converging to <~3mm, and RESULT with
`good_grasp:true, welded:true, success:true`. If it **topples** on the base placement,
lower the spread and re-test: `RAND_SIG_FWD=0.07 RAND_SIG_LAT=0.045`. If the servo bails
/ closes far off (post_grasp_err > ~50mm), STOP and report — do not run the full sweep.

## Step 4 — run the full sweep (detached, ~6–7h) + watchdog
```bash
SWEEP_N=30 nohup setsid bash sweep_random.sh >/dev/null 2>&1 &
nohup setsid bash sim_watchdog_standing.sh >/dev/null 2>&1 &
tail -f sweep_random.log     # per-trial: position index + spawn offset + success
```
`SWEEP_N=30` = 6 positions × 5. The 6 positions are written once to
`RANDOM_POSITIONS.json`. Success rate should be high for `centroid_vs`; `centroid`
(open-loop) may miss more at offset positions — that is the expected result, not a bug.

## Step 5 — check health periodically
Every ~30 min: `tail sweep_random.log` + `sudo docker ps | grep hams`. If the sim/ALMI
died and the watchdog didn't recover it, run `bash restart_standing.sh` then relaunch
`sweep_random.sh` (it is resumable — skips finished trials).

## Step 6 — hand results back
Outputs are host-persistent under
`core_ws/benchmark_results/sweep_battery_random/{centroid,centroid_vs}/`
(`trial_NN.json`, `trial_NN_telemetry.csv`, `trial_NN_spawn.json`). Copy that whole
folder to the main machine (or commit + push this branch). The main machine folds it in:
`python3 make_sway_report.py core_ws/benchmark_results/sweep_battery_random 0.45,-0.05`.

Report when the sweep is complete with the per-method success counts.
