#!/usr/bin/env python3
"""Benchmark method `topdown_irl`: the REAL robot's top-down grasp, verbatim.

This is the deployed goal.top_down path from HAMS main
(core_ws/src/h12_skills/h12_skills/skills/grasp.py on origin/main:
_top_down_candidates + _top_down_pose + TOP_DOWN_PITCH_DEG), ported UNCHANGED
as a grasp_benchmark method — the point of the cell is fidelity to what the
real robot runs, not a benchmark reinterpretation of it:

  * ONE synthetic candidate — no GraspGenX call, no orientation gates, no
    fallback pool (main skips graspgen AND every _graspgen_candidates gate);
  * anchored on the object-cloud MEAN (main: centroid_p = obj_cloud.mean(axis=0)
    — mean, not the benchmark's usual median);
  * the TIER-1 orientation (forward azimuth: approach in the pelvis XZ plane;
    fingers closing along pelvis +Y) pitched TOP_DOWN_PITCH_DEG = 80 deg below
    horizontal. NOT 90: main's comment records that a literally vertical
    approach is unreachable for the H1-2 wrist, so 80 is the deployed value —
    and still far past the wrist's ~26.5 deg practical pitch, so main itself
    warns to "expect best-effort convergence from the servo";
  * gripper BASE placed GRIPPER_BASE_TO_CONTACT_M back along the approach so
    the finger CONTACT lands exactly on the centroid (0.1146 m — the same
    constant grasp_benchmark already carries as TCP_DEPTH_M);
  * width unknown (main reports gripper_width=0.0) -> width_m=None here, which
    the executor turns into the full 106 mm pre-open — matching the deployed
    skill's fixed full pre-open before the approach.

Execution then flows through the SAME shared benchmark executor as every
other method (pre-open -> standoff -> contact -> close; world-anchored servo
on the ALMI tier), so this cell differs from the others ONLY in grasp
synthesis — the study's controlled variable.

Runs exactly like grasp_benchmark (it defers to the INSTALLED
h12_skills.grasp_benchmark for everything, so no rebuild is needed):

    python3 grasp_benchmark_topdown.py --method topdown_irl \
        --object 'vertical fridge handle' --gt-name door_obj --arm right \
        --box-source gt --success-mode contact --out .../trial_NN.json

Every other --method value still works unchanged through this entry point.
"""
import math

import numpy as np

from h12_skills import grasp_benchmark as gb
from h12_skills.perception_utils import matrix_to_pose

# Deployed values, verbatim from origin/main skills/grasp.py.
TOP_DOWN_PITCH_DEG = 80.0
GRIPPER_BASE_TO_CONTACT_M = 0.1146


def _top_down_pose(centroid):
    """GraspGenX gripper-base Pose (pelvis frame) for the steep top-down grasp of
    the object at `centroid`. Body copied verbatim from origin/main
    skills/grasp.py::_top_down_pose (same matrix_to_pose helper, byte-identical
    across the branches)."""
    pitch = math.radians(TOP_DOWN_PITCH_DEG)
    # +Z (approach): pitched `pitch` below horizontal at zero azimuth, so it
    # stays in the pelvis XZ plane (straight ahead, angled down).
    z_axis = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
    # +X (fingers): pelvis +Y, the tier-1 finger preference. Perpendicular to
    # the approach for any pitch, since the approach has no +Y component.
    x_axis = np.array([0.0, 1.0, 0.0])
    # +Y completes the right-handed frame; it comes out at cos(pitch) from
    # pelvis up — exactly the level-wrist that _roll_to_yup would pick.
    y_axis = np.cross(z_axis, x_axis)
    T = np.eye(4)
    T[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    T[:3, 3] = np.asarray(centroid, dtype=float) - GRIPPER_BASE_TO_CONTACT_M * z_axis
    return matrix_to_pose(T)


def method_topdown_irl(self, obj_text):
    """main's _top_down_candidates as a benchmark method: one candidate on the
    object-cloud mean, deployed score/width semantics (single try, width
    unknown). Returns the same plan dict shape as the other method_* fns."""
    cloud = self._boxed_cloud(obj_text)
    if cloud is None:
        return None
    c = cloud.mean(axis=0)          # main: centroid_p = obj_cloud.mean(axis=0)
    pose = _top_down_pose(c)
    return dict(candidates=[(pose, f'topdown-irl-{TOP_DOWN_PITCH_DEG:.0f}deg')],
                width_m=None,       # unknown, as deployed (gripper_width=0.0)
                meta=dict(n_points=int(len(cloud)),
                          centroid=[round(float(v), 4) for v in c],
                          pitch_deg=TOP_DOWN_PITCH_DEG,
                          ported_from='origin/main skills/grasp.py goal.top_down'))


gb.GraspBenchmark.method_topdown_irl = method_topdown_irl
gb.METHODS = tuple(gb.METHODS) + ('topdown_irl',)

if __name__ == '__main__':
    gb.main()
