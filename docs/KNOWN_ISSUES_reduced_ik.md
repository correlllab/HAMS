# Known issue: reduced-space IK fails to resolve grasp poses (OMPL grasp path)

**Owner:** h12_ros2_controller (Yutong's planner/IK)
**Status:** open — worked around in the grasp skill, not fixed at the source
**Impact:** with `do_plan=True`, every CheesyBread grasp is rejected as
unreachable, so the deployed grasp skill grasped 0/N until worked around.

## Symptom

Driving a grasp through the OMPL planner (`FrameTask` goal with `plan=True`)
fails. Across a 12-episode CheesyBread run (`--box-source gt`, a perfect
ground-truth object cloud, legs held, MJPC off), the planner-path failures were:

| reason | count | where it is raised |
| --- | --- | --- |
| `IK failed to resolve current targets` | 19 | `upper_controller.plan_to_ik_target` (before OMPL) |
| `QP solver did not find a solution to the differential IK problem` | 5 | pink differential-IK QP |
| `start moves left_grasp_frame below z=0.0000` | 3 | `reduced_joint_planner._workspace_constraint_failure` |

The same grasp poses, driven with `plan=False` (the direct frame_task servo,
`_direct_frame_task_callback`), reach the target and close the gripper — i.e. the
targets ARE kinematically reachable; the planner path is what fails.

## Root cause (dominant): reduced IK does not converge

`plan_to_ik_target` (core/controller/upper_controller.py) solves the reduced IK
*before* handing a joint goal to OMPL:

```python
ik_result = self.ik_solver.solve_ik_reduced(alpha=ik_alpha, timeout=ik_timeout)
if not ik_result.success:
    raise RuntimeError('IK failed to resolve current targets')
```

`solve_ik_reduced` (core/ik_solver.py) iterates pink differential IK until it hits
`linear_threshold=1e-3` (1 mm) AND `angular_threshold=1e-2` (0.57°) or the 5 s
timeout. For these grasp poses it never converges:

- Loosening the bar to 5 mm / 2.9° (via a temporary override) did **not** help —
  it still failed. So this is non-convergence, not a too-tight threshold.
- pink logs `RuntimeWarning: invalid value encountered in sqrt` from
  `pink/limits/acceleration_limit.py:162` (`dt * np.sqrt(2*a_max*Delta_q_max)`).
  `Delta_q_max` goes **negative** when a joint is already at/over its limit, so
  the sqrt is NaN, the commanded velocity is NaN, and the configuration never
  converges. The H1-2 wrist pitch is clamped to ±0.4625 rad and grasp poses drive
  it to that limit (this is also why the sim e-stop band had to be widened), so
  the accel-limit NaN is the most likely mechanism. A degenerate reduced-model
  task Jacobian at these configurations is the other candidate.

The direct servo path avoids this because it drives the real robot toward the
target incrementally over `duration` seconds and does not require a 1 mm one-shot
solve.

## Secondary: workspace floor + idle-arm veto (3/27)

`frame_z_min: 0.0` (config, e.g. `sim_safety_split.yaml`) is a z-floor applied in
the **pelvis frame** to BOTH `left_grasp_frame` and `right_grasp_frame`. Two
problems for tabletop grasps:

1. A tabletop object sits *below* the pelvis (CheesyBread cheese ≈ pelvis
   z −0.05 m), so a valid grasp frame there is below the z=0 floor and rejected.
   The floor probably wants to be world-frame ("don't hit the ground"), not
   pelvis-frame.
2. `_workspace_constraint_failure` iterates **all** `frame_names`, including the
   idle arm. The left hand hangs slightly below the pelvis at home, so its
   *start* state fails the floor and vetoes a right-arm plan
   (`start moves left_grasp_frame below z=0.0000`). Suggest: only apply the floor
   to frames whose supporting joints are active in this plan (the planner already
   has `active_mask`), and/or don't reject the *start* state (you can't fix where
   the robot already is).

## Workaround in place

The grasp skill's pre-grasp servo now passes `do_plan=False`
(`skills/grasp.py`, the `servo_frame_to_world` call in `_exec_generic_grasp`),
matching its contact move, so grasping works via the direct servo. Revert to
`do_plan=True` once the reduced IK converges on grasp poses.

## Repro

```bash
# fails (OMPL path): "no reachable grasp / IK failed to resolve current targets"
benchmarks/grasp_synthesis/run_benchmark.sh -b gt -m graspgenx -s 42
# reaches + grasps (direct path):
benchmarks/grasp_synthesis/run_benchmark.sh -b gt -m graspgenx -s 42 -x --no-plan
```
Per-episode planner logs are salvaged to
`core_ws/benchmark_results/<stamp>.bringup.log`.
