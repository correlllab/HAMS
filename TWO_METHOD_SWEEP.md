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
