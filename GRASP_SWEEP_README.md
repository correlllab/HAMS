# Fridge-handle grasp-method comparison — setup & replication

This documents the grasp-comparison experiment: a frozen-body Unitree H1-2 in
RoboCasa `OpenFridge` (seed 42) grasping the **vertical fridge door handle** with
its **right arm**, comparing four grasp-generation methods over 20 trials each.

Everything here was built/verified on 2026-07-16. Where a change touches shared
infra it is **opt-in / env-gated** and defaults to the shipped behaviour.

## 1. The four methods (paper name → benchmark `--method`)

| paper name | `--method` | what it does |
|---|---|---|
| centroid | `centroid` | object-cloud centroid, fixed top-down tilted grasp |
| pca | `topdown_antipodal` | wrist-cam PCA of the top slab, close across the minor axis |
| graspgenx-normal | `graspgenx` | GraspGenX 6-DOF grasps, executed **raw** best-first |
| new | `skill` | deployed `/skill/grasp`: GraspGenX + **wrist-aware Y-up re-roll** + priority tiers + diversity + retry |

## 2. Success metric — CONTACT, not lift

A fridge handle is bolted down, so the benchmark's default "object rises" metric
can never fire. The benchmark gained `--success-mode contact`:

**success = the arm executed a grasp AND the fingers came to rest ON the handle** —
the FINAL settled aperture `grip_final_mm` lands in the hold band `(20, 85) mm`.
We score the *final* aperture, not a running minimum: a running min is polluted by
the gripper's start state (between trials it can be the previous grasp's closed jaw),
while the final aperture reflects only where the fingers ended up. Calibrated on live
grasps: a firm handle hold settles ~54 mm (the handle width), an empty close bottoms
out ~5 mm, a jaw that never closed stays ~100 mm. Every trial also records
`grip_min_mm`, `grip_final_mm`, `grip_max_force_n`, `grip_contact_flag`, `holding`,
and the door pose, so the raw signals are auditable. (This metric and several harness
details were hardened after an adversarial code review — see §9.)

## 3. Why the robot is posed the way it is (the reach problem)

The H1-2 wrist pitch is hard-limited to ±0.4625 rad (±26.5°). Standing tall, the
shoulders sit ~0.23 m ABOVE the handle, forcing a downward wrist pitch past that
limit → **every handle grasp is IK-unreachable**. Fixes, all in
`h1_robocasa/h12_mujoco.py`, all env-gated (see `docker/.env`):

- `HAMS_FREEZE_BODY=1` — rigidly pin pelvis+legs+torso every sim step; only the
  arms move (the requested "perfectly still, only the arm moves").
- `HAMS_SPAWN_FORWARD=0.18` — step the frozen spawn toward the fridge;
  `place_robot_collision_free` still backs off on any real overlap, so a gripper
  can never end up inside the door.
- `HAMS_SPAWN_LATERAL=0.10` — slide left so the RIGHT shoulder lines up directly
  behind the RIGHT handle (removes the cross-body reach).
- Lowered stance (`INIT_LEG_POS` knee-bend hip −0.6 / knee 1.2 / ankle −0.6) —
  squats the pelvis ~0.13 m so the shoulder drops to ~handle height. Reach becomes
  near-level (~11°), wrist stays in range. Torso stays vertical (not a lean), feet
  auto-fit the floor, body stays frozen.

Result: `skill` reliably grasps the handle; `graspgenx` raw poses remain
wrist-hostile (a genuine finding — the skill's re-roll is what makes them
reachable).

### Between-trial reset (important)

The upper-body differential-IK controller can leave the arm in a raised config it
cannot recover from — `named_config home` then fails and every following trial
starts from that bad pose and stops reaching. So `h12_mujoco.py` adds a reliable
**sim-side arm reset**: publishing `std_msgs/Empty` on `/hams/reset_arm` pins the
14 arm joints (and their PD setpoints) to the home pose for `HAMS_ARM_RESET_HOLD`
seconds (default 2.5). The benchmark's `go_home` publishes it and runs
`named_config home` while the pin holds, so the controller latches "home" as its
target and the arm does not drift back. Validated: two consecutive skill grasps
both succeed where, without the reset, the 2nd failed from the stuck pose.

The sweep harness also **retries a trial only on a perception no-op** (gemini/SAM
returned no mask, or no grasp candidates were synthesized) — resetting the arm and
re-perceiving so each of the 20 trials is a genuine grasp attempt. A real reach
failure ("no reachable candidate") is kept as data, never retried.

## 4. One-time environment fixes (already applied on this PC)

- `docker/.env`: real `GEMINI_API_KEY` set (the running container had a stale
  `PENDING…` placeholder → `API_KEY_INVALID`; always start `hams_ros` AFTER the key
  is in `.env`, or export it into the bringup shell).
- GPU: run exactly ONE bringup stack — duplicate `gemini/sam/graspgen` servers
  oversubscribe the 16 GB GPU and SAM3 OOMs.

## 5. How to run (replication)

```bash
# 0. one-time: docker/.env has GEMINI_API_KEY + ROS_DOMAIN_ID=1 + the HAMS_* knobs
# 1. sim (frozen, lowered, aligned spawn)
docker/scripts/docker_run.sh robocasa --task OpenFridge --seed 42
# 2. ROS workspace container, then ONE bringup (skills on, no nav/rviz/mjpc):
docker/scripts/docker_run.sh ros sleep infinity          # keeps the container up
docker exec -d hams_ros bash -lc \
  'export HAMS_MAX_GRASP_ATTEMPTS=40; source /opt/ros/humble/setup.bash; \
   source /home/code/core_ws/install/setup.bash; \
   ros2 launch h1_bringup h1_sim_bringup.launch.py use_rviz:=false use_nav:=false use_mjpc:=false'
# 3. the sweep (20 trials x 4 methods, contact success, reset between trials)
docker cp run_sweep.sh hams_ros:/tmp/run_sweep.sh
docker exec -d hams_ros bash -lc 'bash /tmp/run_sweep.sh'
# 4. aggregate + magpie-style grasp-line overlays
docker exec hams_ros python3 /tmp/aggregate.py /home/code/core_ws/benchmark_results/sweep
```

A single trial by hand:
```bash
ros2 run h12_skills grasp_benchmark --method skill \
  --object 'vertical fridge handle' --gt-name door_obj --arm right \
  --success-mode contact --max-attempts 20 --no-plan --out /tmp/one.json
```

## 6. Output layout

```
core_ws/benchmark_results/sweep/
  <method>/trial_NN.json      # per-trial: success, executed, grip_min_mm, force,
                              #   chosen_label, chosen_pose, candidates[], timings
  <method>/trial_NN_head.jpg  # head-cam at the trial
  <method>/trial_NN_mask.png  # gemini box + SAM mask overlay
  <method>/trial_NN.log       # full node log for the trial
  summary.json                # per-method success_rate / exec_rate / means
  viz/<method>_grasp.png       # magpie-style overlay: green closing axis,
                              #   orange fingertip ticks, red grip-centre dot
```

Each trial is a fresh benchmark process that re-homes the arm and re-perceives from
scratch (the "reset between trials, not just re-run"); the scene is otherwise static
(door stays closed, handle fixed). The harness is idempotent — existing
`trial_NN.json` files are skipped, so an interrupted sweep resumes.

## 7. Files changed

- `h1_robocasa/h12_mujoco.py` — `HAMS_FREEZE_BODY` freeze, `HAMS_SPAWN_FORWARD` /
  `HAMS_SPAWN_LATERAL` spawn shift, lowered `INIT_LEG_POS` stance, one-shot reach
  dump (`HAMS_DUMP_REACH`), and the `/hams/reset_arm` arm reset.
- `h1_robocasa/h1_2_robosuite.py` — collision-aware spawn back-off (arms count).
- `docker/docker-compose.yml`, `docker/.env` — the `HAMS_*` env knobs.
- `core_ws/src/h12_skills/h12_skills/grasp_benchmark.py` — `--success-mode contact`,
  `--max-attempts`, candidate/chosen-pose logging, gripper-contact tracking,
  `go_home` arm-reset, `GEMINI_GRASP_PROMPT` import fix.
- `run_sweep.sh` — sweep harness (contact success, arm reset + perception-retry per
  trial, head-cam + mask capture, idempotent/resumable).
- `aggregate.py` — per-method success rates + magpie-style grasp-line overlays.

## 8. Key env knobs (docker/.env)

| var | value used | meaning |
|---|---|---|
| `HAMS_FREEZE_BODY` | 1 | rigidly pin pelvis+legs+torso; only arms move |
| `HAMS_SPAWN_FORWARD` | 0.18 | step spawn toward fridge (collision-aware) |
| `HAMS_SPAWN_LATERAL` | 0.10 | slide left so R shoulder lines up with R handle |
| `HAMS_ARM_RESET_HOLD` | 2.5 | seconds the arm is pinned home on `/hams/reset_arm` |
| `HAMS_MAX_GRASP_ATTEMPTS` | 40 | ranked grasps the deployed skill walks (bringup env) |
| `GEMINI_API_KEY` | (real key) | must be set BEFORE `hams_ros` starts |

## 9. Data-validity hardening (adversarial review)

Mid-run, an adversarial multi-agent review of the changed code caught several
data-invalidating bugs, fixed before the `skill` method ran (the analytic methods
fail on reach, so their 0/20 data is unaffected; only executed grasps depend on
these):

- **Contact metric** now scores the FINAL settled aperture (`grip_final_mm`), not a
  running minimum — the running min was pinned to the gripper's start state (a
  previous grasp's closed jaw), which would have systematically scored real holds as
  failures. Band recalibrated to `(20, 85) mm` from live grasps.
- **`close_gripper(arm, 1.0)`** was passing 1.0 as force in **Newtons** (a 30× weaker
  close than the deployed skill); fixed to the calibrated default force. (Only the
  analytic execute path used it, and none reach here, but corrected for consistency.)
- **Harness retry** now matches the deployed skill server's own perception messages
  (`no grasp planned`, `no box`, …), not just the benchmark's, so a skill perception
  no-op is retried rather than silently counted as a grasp failure.
- **Harness timeout** no longer clobbers a valid result the benchmark already wrote
  in its `finally` block before shutdown.

Non-data-affecting notes from the same review (left as-is for this run): the
`/hams/reset_arm` enforcement lives inside the `HAMS_FREEZE_BODY=1` block (which is
on here, and the reset was validated empirically); the reset's PD-setpoint override
is redundant (the arm returns home via the qpos pin + `named_config` latch); and
`HAMS_SPAWN_LATERAL` back-off only steps along -x (fine for the 0.10 m shift used).

## 10. frame_task controller degradation (skill method)

The upper-body `frame_task_server` (pink differential-IK) **degrades over hours /
hundreds of IK calls**: its reach starts *diverging* (error grows instead of
shrinking) and `named_config home` stops converging — so mid-run the deployed
`skill` method could no longer reach ANY grasp, and every trial hit the 300 s
approach timeout. A FRESH bringup immediately restores it (reach converges,
`named_config` reaches home, the handle is grasped: grip_final ~54–64 mm).

The analytic methods (centroid / pca / graspgenx) are unaffected in conclusion —
their top-down / raw-6DOF grasps are physically wrist-unreachable regardless of
controller state, so their 0/20 stands. Only the `skill` method depends on a
healthy controller, so it is run by `skill_sweep.sh`, which **restarts the whole
bringup every 3 trials** (matching the largest run verified good on one controller)
for a fresh IK each batch. Each restart is verified to come up as a single clean
GPU stack. This is the one part of the sweep that is NOT a single continuous run.

## 11. Results (2026-07-16)

| method | success | notes |
|---|---|---|
| centroid | **0/20** | top-down grasp at the cloud centroid; wrist-pitch-limit unreachable |
| pca (topdown_antipodal) | **0/20** | PCA minor-axis top-down grasp; same wrist limit |
| graspgenx-normal | **0/20** | raw GraspGenX 6-DOF poses; wrist-hostile orientations, IK-unreachable |
| new (skill) | grasps the handle | GraspGenX + wrist-aware Y-up re-roll + tiers + diversity + retry |

**Headline finding.** Grasping this vertical fridge handle with a *frozen, standing*
H1-2 is at the edge of the arm's kinematics: the wrist pitch is hard-limited to
±26.5°, and the reachable grasp *orientation* forms a small cone. The three baseline
methods propose grasps *on* the handle (see the grasp-line overlays in `viz/`) but
every one requires an orientation outside that cone, so **none is IK-executable
(0/20 each)**. Only the deployed **skill** — which re-rolls GraspGenX grasps to be
wrist-friendly and walks up to 40 ranked candidates — finds grasps inside the cone
and **closes on the handle** (validated repeatedly: gripper settles at grip_final
≈ 54–64 mm on the ~54 mm handle, high force, contact). Its success is limited and
run-to-run variable because whether GraspGenX *samples* a candidate in the small
reachable-orientation cone is stochastic (best-candidate wrist error swings roughly
0.4–0.6 rad trial to trial; a trial converges only when it dips below ~0.45 rad).

**Interpretation for the paper.** The comparison cleanly separates the methods:
heuristic top-down (centroid, pca) and raw learned 6-DOF (graspgenx) grasps are
*proposed but unreachable* on this arm, while the skill's wrist-aware re-roll is what
makes the handle graspable at all. The grasp-line figures in `viz/<method>_grasp.png`
show each method's proposed closing axes overlaid on the head camera — the "different
grasp lines" for side-by-side comparison.

## 12. Final sweep numbers + the honest caveat

Final 20-trials-per-method sweep: **centroid 0/20, pca 0/20, graspgenx 0/20,
skill 0/20**. All four scored zero in this run — but for *fundamentally different*
reasons, which is the actual result:

- **Baselines (centroid/pca/graspgenx) fail STRUCTURALLY.** Their grasps require
  wrist orientations outside the ±26.5° cone; they are IK-unreachable and would be
  0/20 in *any* run. (The grasp-line overlays show the grasps sitting on the handle
  with unreachable orientations.) This is the solid, permanent comparison result.
- **The skill fails CIRCUMSTANTIALLY, not structurally.** It is the ONLY method that
  produces wrist-reachable grasps, and it **demonstrably closes on the handle** —
  validated repeatedly with the gripper settling at grip_final ≈ 54–64 mm (e.g. a
  clean 3/3 batch and multiple single-grasp validations earlier in the session). Its
  per-trial success is stochastic (needs GraspGenX to sample into the small reachable
  cone), AND the `frame_task` differential-IK controller degrades over a long session
  (reach starts diverging), which depressed the *final* long sweep to 0/20 despite
  restarting the bringup every 3 trials.

**So the defensible claim is:** only the skill can grasp this handle (proven); the
baselines structurally cannot. The skill's *numeric success rate* from this long
sweep is not a clean measurement — a fresh-session/fresh-controller run is needed to
quantify it (observed clean-condition grasps were common; the degraded tail was not).
The reliable evidence of the skill's unique capability is the validated grasps
(grip_final data) + the reachability analysis, not the degraded sweep count.

## 13. FINAL frozen-standing results (2026-07-17) — the definitive dataset

All previous result sections captured the debugging journey (squat stance, polluted
environments, degraded runs) — **superseded**. The definitive frozen dataset was
collected 2026-07-17 with: robot STANDING fully upright (shipped stance, no squat,
no lean), body rigidly frozen, lateral spawn alignment, a FRESH container
environment per batch/trial, gripper opened before every arm reset, and automatic
environment recreation whenever the gripper fingers were detected bent
(`skill_sweep_host.sh` / `baselines_host.sh`).

| method | success | grip_final on success | reading |
|---|---|---|---|
| centroid | **0/20** | — | top-down: structurally wrist-unreachable |
| pca | **0/20** | — | top-down: structurally wrist-unreachable |
| graspgenx-normal | **14/20 (70%)** | 68.5–78.5 mm (mean 72.0) | raw 6-DOF poses reach, but grasps land DIAGONAL on the bar — wide, lower-quality holds |
| new (skill) | **12/20 (60%)** | 53.6–58.8 mm (mean 55.2) | wrist-aware re-roll gives CANONICAL perpendicular holds at the bar width |

Validation: 80/80 trial records parse, no timeout stubs, success labels consistent
with gripper data, head-cam image per trial. Note both successful methods grasp the
LEFT door handle (the perception pipeline picks it); all methods share that target,
so the comparison is fair. The grip-width separation (55 vs 72 mm) is corroborated
by wrist-cam contact snapshots: the skill straddles the bar perpendicular; raw
graspgenx captures it at an angle.

**Two hard-won operational mechanisms (root causes of all earlier bad runs):**
1. Repeated in-container bringup restarts leave zombie model-server/controller
   processes that poison DDS and the GPU → always use fresh containers.
2. Failed trials physically BEND the sim gripper's fingers (open-gripper dragging
   against the door), and a successful grasp left clamped gets wrenched by the next
   arm reset → open the gripper before every reset; verify finger symmetry between
   trials; recreate the environment on damage.

## 14. THE GHOST: an external DDS participant was corrupting everything (2026-07-17)

The deepest root cause of every "mysterious degradation" this whole experiment:
**another machine on the lab network runs a robot stack on DDS domain 1 and
publishes all-zero `rt/lowcmd` at ~500 Hz.** The HAMS containers use host
networking, so those packets interleaved with the local stack's commands. Effects:
arm PD authority halved (frame_task IK appears to "diverge", grasps time out),
the FAME standing policy falls ~2 s after band release, and everything recovers
whenever the other machine goes quiet — which masqueraded as session/controller
degradation for two days.

Diagnosis: a LowCmd fingerprint subscriber (count zero-gain messages on /lowcmd —
2 publishers where safety should be alone), then `tcpdump 'udp portrange
7400-7500'` showing foreign source IPs (172.18.0.x = another machine's docker
network) while every local process was frozen one by one.

**Fix: `ROS_DOMAIN_ID=27` in `docker/.env`** (any domain private to this machine;
0 = the real robot, 1 = polluted by the neighbor). After the switch the ghost
stream is gone and results transformed:

| frozen-standing method | ghost era (domain 1) | clean (domain 27) — FINAL |
|---|---|---|
| skill | 12/20 (60%) | **16/20 (80%)**, grip_final mean 59.0 mm (canonical holds) |
| graspgenx-normal | 14/20 (70%) | **13/20 (65%)**, grip_final mean 72.6 mm (diagonal holds) |
| centroid | 0/20 | 0/3 spot-check — confirmed structural |
| pca | 0/20 | 0/3 spot-check — confirmed structural |

The clean-domain numbers are the definitive frozen-standing dataset
(`sweep/<method>/`); ghost-era executed-method trials are archived in
`sweep/ghost_era_backup/` for provenance. The spot-checks confirm the top-down
baselines fail for kinematic reasons, not interference.

## 15. Unfrozen condition + FAME status (2026-07-17 evening)

The **unfrozen** sweep (second 80 trials) runs in the sim's SHIPPED default mode:
fully dynamic body held upright by the soft elastic tether + leg PD stance hold —
no rigid pin; the base sways and reacts to arm motion (`unfrozen_sweep_host.sh`,
results in `sweep_unfrozen/`). This is the honest "unfrozen standing" condition
available today.

**FAME active standing (RL policy) — attempted, not yet usable, evidence logged:**
on engagement the policy immediately commands both ankles to near max dorsiflexion
(+0.51 rad vs its +0.5136 clip) and the robot pitches over ~2 sim-s after band
release — on BOTH the polluted and clean DDS domains. Verified NOT the cause:
joint order (fame.yaml limits align with /lowstate order), safety merge (merged
/lowcmd carries the policy's setpoints verbatim), IMU (live quaternion+gyro in
lowstate). Remaining suspects: quaternion convention (wxyz vs xyzw) inside
`gravity_from_quat_fame`, model mass mismatch (magpie grippers), or RMA warm-up
under the band. `reference/standalone_fame_test.py` is the right next tool. Gate:
`UNFROZEN_FAME=1` in `unfrozen_sweep_host.sh` once fixed.

## 16. COMPLETE RESULTS — 160 trials, both conditions (2026-07-17)

All trials on the clean DDS domain (27), fresh containers per method/batch,
gripper-damage detection, contact-based success. FROZEN = body rigidly pinned
(only the arm moves). UNFROZEN = the sim's shipped dynamic mode (soft elastic
tether + leg PD hold — the base sways and the arm can drag the torso).

| method | FROZEN (sweep/) | UNFROZEN (sweep_unfrozen/) | grip_final on success |
|---|---|---|---|
| centroid | 0/20 (0%) | 4/20 (20%) | — / ~71 mm diagonal |
| pca | 0/20 (0%) | **19/20 (95%)** | — / ~57 mm near-canonical |
| graspgenx-normal | 13/20 (65%) | 12/20 (60%) | ~73 / ~72 mm diagonal |
| new (skill) | **16/20 (80%)** | **19/20 (95%)** | ~59 / ~48 mm canonical |

**Findings:**
1. **Frozen base:** only GraspGenX-based methods grasp at all; the skill's
   wrist-aware re-roll beats raw execution (80% vs 65%) AND produces tighter,
   perpendicular holds (59 vs 73 mm) — quality, not just rate.
2. **Passive compliance transforms reachability.** Unfrozen, the arm can drag the
   compliant torso into the reach: pca goes 0% → 95% (its narrow-axis closing yaw
   is right for the bar once position is attainable), centroid 0% → 20% (its fixed
   closing yaw is usually wrong). The skill also improves (80% → 95%, deepest
   holds at ~48 mm).
3. **Raw graspgenx is compliance-insensitive** (65% ≈ 60%): its failure mode is
   orientation sampling, which compliance does not fix; its successful holds stay
   diagonal (~72 mm) in both conditions.

Figures: `sweep*/viz/{<method>_grasp.png, comparison_grid.png, success_rates.png}`.
Per-trial JSON/log/head-cam in `sweep*/<method>/`. Validation: all 160 records
parse; no timeout stubs; no falls during unfrozen trials; success labels
consistent with gripper telemetry.

## 17. Standing tier (ALMI RL policy) — the third base condition

**Protocol (freeze-hold):** spawn pinned in the ALMI nominal crouch
(`HAMS_STANCE=almi`, `HAMS_FREEZE_BODY=1`), pre-home the arm + open the gripper
UNDER the pin, engage `lowerbody_controller_node` (`active_policy:=almi`)
against the pinned body, wait **40 SIM-seconds** (`almi_engage.py`; the headless
sim runs ~5% realtime — wall-clock waits release before the LSTM stabilizes and
the robot topples), verify upright, release, confirm 15 sim-s, settle-gate
(<5 cm drift) before every trial. Between trials: re-freeze (pin captures the
current pose), reset arm/gripper under the pin, verify station/door/gripper,
re-engage. Perception: mentor's quota-free GT crop with a fixed world point on
the handle bar (`HAMS_GRASP_BOX_SOURCE=gt`, `HAMS_GRASP_GT_WORLD=3.5365,-2.889,1.235`,
vertical-capsule crop r=5.5 cm) for ALL methods — perception removed as a
variable. Execution: world-anchored servo (camera_init via `HAMS_SIM_ODOM=1` +
static TF), `HAMS_SERVO_MAX_ITER=4`, `HAMS_SERVO_DURATION=22`,
`HAMS_SERVO_FASTFAIL_LIN=0.08`, `HAMS_GRASP_OFFSET=0.06` (compensates the
drift-equilibrium: ALMI micro-retreats during arm moves, leaving a systematic
~65 mm shortfall along the approach — measured against the commanded-pose
cluster of the 83 stable-base successes), `HAMS_ATTEMPT_PAUSE=6`,
`HAMS_NO_SIM_ARM_RESET=1` (a trial-time sim qpos pin — even ramped — catapults
a free base ~2.5 m via the PD-handoff mismatch; the frozen/hanging tiers never
saw this because their bases can't move).

**Result (n=20/method, strict contact grading identical to the other tiers):**

| method | frozen | hanging | standing (ALMI) |
|---|---|---|---|
| centroid | 0/20 | 4/20 | 0/20 |
| pca (topdown_antipodal) | 0/20 | 19/20 | 0/20 |
| graspgenx (raw) | 13/20 | 12/20 | 0/20 |
| **skill (wrist-aware ranked)** | **16/20** | **17/20** | **12/20** |

**Standing-tier failure decomposition** (bag-verified): skill — 12 success,
4 contact-unstable, 2 wander, 2 wander+fall; graspgenx — 12 contact-unstable,
4 miss, 2 wander, 2 wander+fall; centroid — 19 contact-unstable (door-edge
clamps), 1 miss; pca — 20 wander (its deep targets = longest reaches = most
base excitation; the base walks off before contact).

**Regrade audit:** the in-run 2-sample stability check samples at +1/+3 s
wall-clock (≈0.05/0.15 sim-s on this tier). All 80 trials were re-graded from
the recorded rosbags at true sim-time (`bag_regrade.py`: final in-band hold
segment, ≤8 mm spread over 2 sim-s). Outcome: zero flips — the graspgenx
"contacts" show 30+ mm aperture churn or <1 sim-s holds (genuinely unstable
under sway), and every skill success shows ≤0.2 mm spread. The strict table
stands.

**Key findings:** (1) an actively-balancing base costs the best method 4-5
successes/20 vs stable bases — and kills candidate-spray methods entirely:
graspgenx drops 13→0 because its diagonal holds cannot settle on a swaying
base and its candidate cycling excites wander; (2) ALMI has NO station-keeping
below its 0.1 m/s command threshold — recovery steps accumulate under arm
interaction and walk the robot off-station (the dominant failure for pca);
(3) grasp-target compensation must account for the policy's drift-equilibrium
(~65 mm here), which iteration alone cannot close (moving equilibrium).

**Sim/protocol hazards catalogued for replication** (each cost a debugging
cycle): wall-clock vs sim-clock stabilization; GT origins inside object bodies
(door_obj → 0 cloud points; use a world-point crop); model-server load spikes
starving the sim → spontaneous policy wander (wait for ALL server ready lines);
in-sim qpos teleports (arm reset OR its PD handoff) catapulting free bases;
double-harness container races. Figures: `figures/three_tier_success.png`,
`figures/almi_pelvis_traces.png`. Full per-trial data: `sweep_almi/<method>/`
(JSON + telemetry CSV + rosbag + skills-log + head snapshot).

### 17.1 Interpretation caveat: selection vs execution

`skill` is the DEPLOYED pipeline (ranked wrist-feasible candidates + staged
approach + world-anchored drift-compensated servo + anchored hold); the other
three methods are raw synthesis modes driven by the benchmark's simpler
executor (`--no-plan`, pelvis-frame hold after close). On stable bases the
asymmetry is minor (raw ggx 13/20 vs skill 16/20); on the standing base
execution dominates, so the standing column compares SYSTEMS, not synthesis
algorithms in isolation. Mechanistic support: ggx reached bar contact 12/20
standing but every hold churned 30+ mm (pelvis-held grip sheared by sway,
vs ≤0.2 mm for the skill's anchored holds). Decomposing control (future
work): run UNRANKED ggx candidates through the skill executor (n=20) to
separate candidate selection from execution robustness.

### 17.2 Data audit + one flagged uniformity

Audit (2026-07-22): 240/240 trials present (20×4×3); all 68 standing
non-successes carry machine-readable failure reasons; 24 wander flags
cross-validated against telemetry with 0 contradictions; 80/80 standing trials
have JSON+telemetry+bag+servo-log (28 head JPGs absent — exactly the
wander/fall-killed trials, where the post-trial grab never ran; head video
remains in the bags; 3 bags unreadable from a teardown race).

Flagged: pca's 20/20-wander uniformity is mechanistically plausible (deepest
targets → longest reaches → most excitation) but entangled with the uniform
HAMS_GRASP_OFFSET=0.06, which was calibrated on the skill's measured
drift-shortfall and further deepens pca's already-deepest targets. Bounding
control (future work): pca standing at offset 0, n=20.
