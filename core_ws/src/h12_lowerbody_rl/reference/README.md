# reference/ — standalone MuJoCo deploy scripts (not built)

These are the original **standalone** policy deployment scripts, kept for
provenance and as the source of truth for the observation/action math now
implemented in the ROS node. They are **not** part of the colcon build and are
not maintained to run as-is (they load their own local MuJoCo model and, for
the RMA script, expect the training-tree `RMA` package + an XML that is not in
this repo).

- `deploy_mujoco.py` — original walking-policy deploy → now `WalkPolicy`.
- `mujoco_deploy_h12_rma.py` — original FAME/RMA deploy → now `FamePolicy`.

The live integration is in `h12_lowerbody_rl/`:
`policy.py` (WalkPolicy, FamePolicy), `policy_manager.py` (safe-handover
switching), `scripts/lowerbody_controller_node.py` (the ROS node), and the
trimmed inference-only encoder under `h12_lowerbody_rl/rma/`.
