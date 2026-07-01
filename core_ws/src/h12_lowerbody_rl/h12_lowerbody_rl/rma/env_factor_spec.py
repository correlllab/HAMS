"""Environment factor (e_t) specification for H12 RMA with Isaac Gym (hand-only, spherical).

e_t = 15 upper-body joint positions (read from dof_pos) + 6 force components (sampled).
No torso forces. Hand forces are sampled in 3D via spherical sampling (magnitude then direction)
and applied to each wrist link in world frame.

- Hands: magnitude U(0, HAND_FORCE_MAGNITUDE_MAX), direction uniform on unit sphere.
  Stored in e_t as (Fx, Fy, Fz) per hand; applied as-is to left_wrist_roll_link and right_wrist_roll_link.
"""

from __future__ import annotations

from dataclasses import dataclass


# Hand force: spherical sampling. Magnitude in [0, max] (N); direction uniform on unit sphere.
# Resulting 3D force (Fx, Fy, Fz) is stored in e_t and applied to sim; component range for decoder is ±max.
HAND_FORCE_MAGNITUDE_RANGE: tuple[float, float] = (0.0, 30.0)  # N
# Per-axis range for decoder/output scaling (covers any direction with magnitude up to max)
HAND_FORCE_COMPONENT_RANGE: tuple[float, float] = (-30.0, 30.0)

# Per-step resample probability (RMA paper: 0.004).
RMA_RESAMPLE_PROB: float = 0.004


@dataclass(frozen=True)
class RmaEtSpec:
    """e_t for H12 RMA (hand-only, 3D spherical): 15 upper-body + left_hand_xyz(3) + right_hand_xyz(3) = 21."""

    upper_body_dof_dim: int = 15
    torso_force_dim: int = 0   # no torso
    hand_force_dim: int = 3   # 3D per hand
    num_hands: int = 2

    @property
    def force_total_dim(self) -> int:
        return self.torso_force_dim + self.num_hands * self.hand_force_dim

    @property
    def dim(self) -> int:
        return self.upper_body_dof_dim + self.force_total_dim

    @property
    def upper_body_slice(self) -> slice:
        return slice(0, self.upper_body_dof_dim)

    @property
    def torso_force_slice(self) -> slice:
        return slice(self.upper_body_dof_dim, self.upper_body_dof_dim + self.torso_force_dim)

    @property
    def left_wrist_force_slice(self) -> slice:
        return slice(
            self.upper_body_dof_dim + self.torso_force_dim,
            self.upper_body_dof_dim + self.torso_force_dim + self.hand_force_dim,
        )

    @property
    def right_wrist_force_slice(self) -> slice:
        return slice(
            self.upper_body_dof_dim + self.torso_force_dim + self.hand_force_dim,
            self.upper_body_dof_dim + self.force_total_dim,
        )


DEFAULT_ET_SPEC = RmaEtSpec()

UPPER_BODY_JOINT_NAMES: tuple[str, ...] = (
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Only wrist links (no torso).
RMA_FORCE_BODY_NAMES: tuple[str, ...] = (
    "left_wrist_roll_link",
    "right_wrist_roll_link",
)
