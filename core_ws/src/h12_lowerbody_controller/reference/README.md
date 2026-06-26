# reference/ — standalone MuJoCo deploy scripts (not built)

These are the original **standalone** policy deployment scripts, kept for
provenance and as the source of truth for the observation/action math now
implemented in the ROS node. They are **not** part of the colcon build and are
not maintained to run as-is (they load their own local MuJoCo model and, for
the RMA script, expect the training-tree `RMA` package + an XML that is not in
this repo).

- `deploy_mujoco.py` — original walking-policy deploy → now `scripts/walking_node.py`.
- `mujoco_deploy_h12_rma.py` — original FAME/RMA deploy → now `scripts/fame_node.py`.

The live integration is two self-contained ROS nodes under
`h12_lowerbody_controller/scripts/`: `walking_node.py` (walking policy) and
`fame_node.py` (FAME/RMA standing-squat policy), each inlining the
observation/action math from its reference script above. `fame_node` also uses
the trimmed inference-only encoder under `h12_lowerbody_controller/rma/`.
