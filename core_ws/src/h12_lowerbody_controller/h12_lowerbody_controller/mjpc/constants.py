"""H1-2 control constants for the MJPC deploy node.

Ported verbatim from the fork's ``mjpc/deploy/h12_control_node.cc`` (the ``KP[]``,
``KV[]``, ``TAU_ESTOP[]``, ``FRC_LIMIT[]``, ``IMU_OFFSET`` arrays). 27 actuated
joints on the handless H1-2, in ``/lowstate`` (Unitree motor) order::

    [0:6]   left leg   hip_yaw hip_pitch hip_roll knee ankle_pitch ankle_roll
    [6:12]  right leg  (same)
    [12]    torso (yaw)
    [13:20] left arm   shoulder_p shoulder_r shoulder_y elbow wrist_r wrist_p wrist_y
    [20:27] right arm  (same)

``TAU_LIMIT`` is reused from ``h12_safety_layer.core.joint_limits.URDF_TORQUE_LIMITS``
(the single source of truth — identical values), with a local fallback so this
module imports standalone (e.g. for the unit test) when that package is not on the
path. KEEP THE FALLBACK IN SYNC with the safety layer.
"""

import numpy as np

KNU = 27  # actuated joints on the handless H1-2

# Lower-body split = the 12 legs (indices 0..11). The h12_safety_layer split merge
# (_build_split_cmd_locked, SPLIT_UPPER_START=12) takes motor_cmd[0:12] from
# rt/safety/lowcmd_lower_in and [12:27] (torso + arms) from the upper split, so a
# lower-body controller publishes ONLY these joints to the lower topic.
NUM_LOWER = 12
LOWER_JOINTS = tuple(range(NUM_LOWER))

# Per-joint PD gains == h1_2_modified actuator classes == the real LowCmd kp/kd
# (must match the planner model's <position> actuators and the twin's PD law).
KP = np.array(
    [150, 200, 200, 200, 80, 80,  150, 200, 200, 200, 80, 80,  200,
     30, 30, 20, 20, 15, 15, 15,   30, 30, 20, 20, 15, 15, 15],
    dtype=float,
)
KV = np.array(
    [5, 5, 5, 5, 4, 4,  5, 5, 5, 5, 4, 4,  5,
     10, 10, 10, 10, 2, 2, 2,  10, 10, 10, 10, 2, 2, 2],
    dtype=float,
)

# Safety-layer tau-ESTOP thresholds (estop torque_ratio x URDF torque limit, from
# default_safety_full.yaml). Used by the torque-aware target clamp so a commanded
# position can never demand estop-level torque.
TAU_ESTOP = np.array(
    [60, 130, 200, 300, 54, 36,  60, 130, 200, 300, 54, 36,  40,
     32, 32, 14.4, 14.4, 9.5, 9.5, 9.5,
     32, 32, 14.4, 14.4, 9.5, 9.5, 9.5],
    dtype=float,
)

# Operational H1-2 joint torque limits (Nm) == Unitree URDF actuatorfrcrange ==
# h12_safety_layer URDF_TORQUE_LIMITS. NOT the motor-peak from specs. Used for the
# B0 torque-headroom report and (as FRC_LIMIT) when patching a planner model.
try:
    from h12_safety_layer.core.joint_limits import URDF_TORQUE_LIMITS as _TAU_LIMIT
except Exception:  # pragma: no cover - standalone fallback; keep in sync with the safety layer
    _TAU_LIMIT = [
        200, 200, 200, 300, 60, 40,  200, 200, 200, 300, 60, 40,  200,
        40, 40, 18, 18, 19, 19, 19,   40, 40, 18, 18, 19, 19, 19,
    ]
TAU_LIMIT = np.asarray(_TAU_LIMIT, dtype=float)
FRC_LIMIT = TAU_LIMIT  # arm/leg operational force limit (identical table in the C++ node)

# IMU site position in the pelvis (free-joint) frame, from h1_2_handless.xml.
# Confirmed against CL_Assets/mujoco_assets/h1_2_magpie.xml (the twin's model).
IMU_OFFSET = np.array([-0.04452, -0.01891, 0.27756], dtype=float)

# Short joint names (qpos[7..33] order) for the status line / B0 report.
SHORT_NAMES = (
    "LhipY", "LhipP", "LhipR", "Lknee", "LankP", "LankR",
    "RhipY", "RhipP", "RhipR", "Rknee", "RankP", "RankR", "torso",
    "LshP", "LshR", "LshY", "Lelb", "LwrR", "LwrP", "LwrY",
    "RshP", "RshR", "RshY", "Relb", "RwrR", "RwrP", "RwrY",
)

# Default nominal standing stance (legs slightly bent), /lowstate order, used as the
# bring-up ramp destination and the nominal hold when no model keyframe is given.
# Legs from the H1-2 nominal stance; torso + arms at zero.
DEFAULT_STANCE = np.zeros(KNU, dtype=float)
DEFAULT_STANCE[:12] = [0.0, -0.16, 0.0, 0.36, -0.2, 0.0,
                       0.0, -0.16, 0.0, 0.36, -0.2, 0.0]

# Default nominal pelvis height (m) when rt/sportmodestate is unavailable (twin /
# debug mode); H1-2 stands ~1.03 m at the pelvis.
DEFAULT_BASE_HEIGHT = 1.03

assert KP.shape == KV.shape == TAU_ESTOP.shape == TAU_LIMIT.shape == (KNU,)
assert len(SHORT_NAMES) == KNU
