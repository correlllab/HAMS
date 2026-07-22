# Paper-graph grasp sweep — new(skill) vs graspgenx-normal vs topdown(irl)

Branch `grasp-2method-comparison` (worktree `/home/guest/HAMS-grasp-2method`).
Everything for the paper graph lives HERE; the live checkout at
`/home/guest/Downloads/HAMS-test-grasping` (branch `test/grasping`) is never
modified, switched, or rebuilt — this branch only *drives* its docker setup.

## The graph cells

| harness method | aggregate label | what it is |
|---|---|---|
| `skill` | `new(skill)` | the deployed `/skill/grasp` path: GraspGenX + heuristic matching (envelope → Y-up re-roll → priority tiers → diversity → IK fallback) |
| `graspgenx` | `graspgenx-normal` | the SAME GraspGenX server + sampling config, executed strictly best-first by raw model score (no heuristic selection) |
| `topdown_irl` | `topdown(irl)` | HAMS main's DEPLOYED `goal.top_down` grasp, verbatim: ONE synthetic candidate on the object-cloud MEAN, tier-1 orientation (approach in the pelvis XZ plane, fingers along pelvis +Y) pitched 80° below horizontal, gripper base 0.1146 m back along the approach so the finger contact lands exactly on the centroid. No GraspGenX call, no orientation gates, no fallback pool. |

skill/graspgenx share one candidate generator (same server, same post-fe8a252
sampling config: 2048 candidates, 72 yaws, disabled server envelope), so that
delta cleanly isolates the heuristic-matching layer. This is a *cleaner*
ablation than the old-PC plan of replaying the pre-fe8a252 client on the new
pool — no sampling-config caveat needed in the paper.

`topdown_irl` is the as-deployed real-robot baseline — the exact code path a
`goal.top_down` grasp runs on HAMS main, ported verbatim (constants, mean
centroid, single candidate) in `grasp_benchmark_topdown.py`. Fidelity note
from main's own comments: 80° is still far past the H1-2 wrist's ~26.5°
practical pitch, so main itself says to "expect best-effort convergence from
the servo" — a low success rate here is the honest deployed-baseline result,
not a porting bug. The wrapper registers the method on the INSTALLED
`h12_skills.grasp_benchmark` at runtime (copied to the container's /tmp by the
sweep script), so the live checkout still needs no rebuild.

`pca` (= `topdown_antipodal`) and `centroid` are deliberately NOT run for this
graph. If a reviewer wants them: N=20 data already exists in
`core_ws/benchmark_results/sweep_almi_ablation/` (standing) and
`sweep_unfrozen/` (hanging).

## The paper grid (2026-07-22, supersedes the single-tier plan)

Final figure = the 3 methods above x 3 conditions, N=30 per cell:

| condition | protocol | collected by |
|---|---|---|
| hanging | band mode (`UNFROZEN_POLICY=""`) | `paper_grid_host.sh` (all 3 methods) |
| standing | ALMI tier | skill+graspgenx REUSED from campaign `sweep_almi/` (identical protocol, N=30); only `topdown_irl` collected new |
| standing, randomized start | ALMI tier + fresh env per trial at a spawn sampled around the tuned stance (fwd 0.18 / lat 0.10): independent Gaussians, sigma_long 11.43 cm / sigma_lat 6.83 cm, clipped +/-3 sigma, position only, deterministic per-trial seeds (`rand-v2`), each trial's spawn logged as `trial_NN_spawn.json` | `paper_grid_host.sh` (all 3 methods, runs LAST — axis assignment still pending the raw endpoints; override with `UNFROZEN_RAND_SIG_LONG/LAT`) |

### Randomized-start: framing + limitations (paper text material)

Spec review against the reference paper (2026-07-22): the paper has **no
initialization-noise spec** — "noisy initialization" in the Fig. 3 caption is
the localization prior, not a spawn perturbation. What we sample from is the
paper's **measured endpoint scatter of a 6 m walk-to-goal** on this platform
(Sec. IV-A, n=19), used as a proxy for post-navigation stance uncertainty.
Decisions + caveats to carry into the paper:

- **Axis swap**: the figure annotation (sigma_x=6.83 longitudinal,
  sigma_y=11.43 lateral) contradicts the plotted spreads (~50 cm longitudinal,
  ~21 cm lateral). We follow the PLOTTED data: sigma_long 11.43, sigma_lat
  6.83. **TODO (Adam): get the raw 19 endpoints from the authors** (Correll is
  an author) — that settles the swap, enables a full sample-covariance
  sampler (drift correlates the axes; independent axes throw that away), and
  reveals whether endpoints were ground truth or self-estimate.
- **Clip +/-3 sigma**, not 2: 2-sigma truncation shrinks realized sigma ~12%.
- **Position only, no yaw** — hard tooling constraint (the sim has no
  spawn-yaw knob and is frozen for the study). Flag explicitly: heading error
  plausibly *generates* much of the walking scatter, and base yaw moves the
  handle in the arm's workspace more than translation does.
- Magnitude caveats: it is a 6 m approach (drift scales with distance);
  n=19 puts ~17% relative standard error on each sigma.
- This condition models the ~95% nominal-navigation case only; the paper's
  1/20 gross failure ("veered several meters off") is NOT represented.
- **Reporting**: success rate over ALL 30 trials is the primary number;
  divergent trials are *graded* failed by the 0.6 m / 60 deg watchdog (never
  dropped), but scatter statistics use non-divergent trials only — mirroring
  the paper's own 19/20 convention. The informative figure is the sampled
  start offsets colored by grasp outcome (the start distribution itself is
  known by construction); for dose-response, fit a logistic on offset
  magnitude rather than binning.

`paper_grid_host.sh` sequences the 7 collected cells (hanging x3 -> standing
topdown_irl -> randomized x3, ~23 h total; the randomized tier alone is ~17 h
because every trial rebuilds the env — `/hams/reset_scene` restores a fixed
snapshot, so a rebuild is the only way to move the spawn without sim-code
changes). Everything resumes per cell. The auto-start launcher runs this
driver when the campaign completes cleanly.

## How to run (AFTER the current campaign sweep finishes)

The currently-running orchestrator (`unfrozen_sweep_host.sh`, ALMI tier, N=30,
4 methods → `sweep_almi`) owns the containers. Do not stop it. The new script
refuses to start while it (or any `grasp_benchmark` trial) is alive.

```bash
# 1. check nothing is running (should print nothing):
pgrep -af 'sweep_host|baselines_host|grasp_benchmar[k]'

# 2. run the sweep (N trials x {skill, graspgenx, topdown_irl}, ALMI standing
#    tier; N defaults to 20 — the armed auto-start uses UNFROZEN_N=30 -> 90
#    trials). UNFROZEN_METHODS="skill graspgenx" drops the topdown cell.
bash /home/guest/HAMS-grasp-2method/two_method_sweep_host.sh
```

Results land on the host at
`Downloads/HAMS-test-grasping/core_ws/benchmark_results/sweep_2method/<method>/`
(trial JSON + log + telemetry CSV + rosbag + head snapshot + skills-log grep,
identical schema to the campaign's other sweeps). Trials are resumable: re-run
the script and existing `trial_NN.json` are skipped.

Knobs (same env names as the unfrozen script):

```bash
UNFROZEN_N=30 bash two_method_sweep_host.sh                # more trials/cell
UNFROZEN_POLICY="" bash two_method_sweep_host.sh           # hanging tier instead
UNFROZEN_OUT=/home/code/core_ws/benchmark_results/sweep_almi \
  bash two_method_sweep_host.sh                            # top-up sweep_almi instead
UNFROZEN_METHODS="skill" bash two_method_sweep_host.sh     # one method only
```

Note: today's in-flight run already produces `skill` and `graspgenx` at N=30
(ALMI tier) in `sweep_almi/` — if it completes cleanly, the graph may need no
new trials at all; aggregate that dir directly and this script becomes the
clean re-run / top-up tool.

## Aggregate → graph numbers

Use this branch's `aggregate_2method.py` — the live checkout's `aggregate.py`
iterates a fixed method list and silently IGNORES a `topdown_irl` dir:

```bash
sudo docker cp /home/guest/HAMS-grasp-2method/aggregate_2method.py hams_ros:/tmp/aggregate_2method.py
sudo docker exec hams_ros bash -lc \
  'source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash; \
   python3 /tmp/aggregate_2method.py /home/code/core_ws/benchmark_results/sweep_2method'
```

Writes `summary.json` (success/exec rates, grip mm, force, exec time per
method) + per-method grasp-line overlay PNGs into `sweep_2method/viz/`.
Missing method dirs are skipped automatically, so any subset is fine (it also
aggregates `sweep_almi`-style dirs).

## Safety rails baked into the script

- `flock` single-instance lock + refusal while other orchestrators/trials run
  (`--force` overrides, only if you're sure).
- Runs from this worktree; copies its OWN committed helpers
  (`almi_engage.py`, `trial_recorder.py`, `almi_inference_bench.py`,
  `grab_head.py`) into the container — no dependency on `/tmp` or on another
  session's scratchpad surviving a reboot.
- The live checkout needs no branch switch and no colcon rebuild: both methods
  are already in the installed `grasp_benchmark`; only host-side orchestration
  differs. (`docker/.env` is sed-tuned per tier at runtime, exactly like the
  campaign scripts do — it's gitignored runtime config, not tracked state.)

## Housekeeping

- A backup of the live checkout's uncommitted state as of 2026-07-22 is at
  `/home/guest/hams_experiments/live_uncommitted_2026-07-22.diff`.
- This branch is local-only. Before any PR upstream, coordinate per CLAUDE.md
  §5 and drop this file + the sweep script or move them to a personal fork.
