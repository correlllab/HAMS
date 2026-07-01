"""Lower-body policy plugins.

A ``Policy`` turns the current robot state into a 12-joint lower-body PD setpoint
(``LegCommand``). Each policy is fully self-describing — it owns its observation
layout, scales, PD gains, and nominal pose, all loaded from its own YAML — so the
controller node can run any of them through the same loop and switch between them.

Two concrete policies are provided:

* ``WalkPolicy``  — the original TorchScript walking policy (47-d obs, gait clock).
* ``FamePolicy``  — the RMA standing/squatting policy: an env-factor encoder
  (e_t -> z_t) plus a base policy over a history of proprio + z_t (252-d actor obs).
  It reads all 27 joints (legs + torso + arms) but only commands the 12 legs; the
  arms are driven externally by the upper-body IK and merely *observed* here.

Joint indexing matches ``/lowstate`` exactly (verified against
``CL_Assets/mujoco_assets/h1_2_magpie.xml``): index 0-11 legs, 12 torso,
13-19 left arm, 20-26 right arm.
"""

from __future__ import annotations

import collections
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import torch
import yaml

NUM_POLICY_JOINTS = 27          # legs(12) + torso(1) + left arm(7) + right arm(7)
NUM_LEG_JOINTS = 12
NUM_UPPER_JOINTS = 15           # torso + arms, indices 12:27


# --------------------------------------------------------------------------- #
# Shared data records
# --------------------------------------------------------------------------- #
@dataclass
class RobotState:
    """Everything a policy needs, extracted from /lowstate + external commands."""

    q: np.ndarray            # (27,) joint positions, /lowstate order
    dq: np.ndarray           # (27,) joint velocities
    quat: np.ndarray         # (4,) base orientation [w, x, y, z]
    gyro: np.ndarray         # (3,) base angular velocity
    cmd: np.ndarray          # (3,) velocity command [vx, vy, wz]
    height_cmd: float        # squat / base-height command (FAME only)
    t: float                 # sim time [s] (gait clock for the walker)


@dataclass
class LegCommand:
    """12-joint lower-body PD setpoint published to /safety/lowcmd_lower_in."""

    target_q: np.ndarray     # (12,)
    kp: np.ndarray           # (12,)
    kd: np.ndarray           # (12,)


# --------------------------------------------------------------------------- #
# Gravity helpers — each policy uses the convention it was trained with.
# --------------------------------------------------------------------------- #
def gravity_from_quat_walk(quat: np.ndarray) -> np.ndarray:
    """Projected gravity as used by the walking policy (matches deploy_mujoco.py)."""
    qw, qx, qy, qz = quat
    return np.array(
        [
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by the inverse of quaternion q ([w, x, y, z]) — FAME/RMA convention."""
    w, x, y, z = q
    qc = np.array([w, -x, -y, -z], dtype=np.float32)
    return np.array(
        [
            v[0] * (qc[0] ** 2 + qc[1] ** 2 - qc[2] ** 2 - qc[3] ** 2)
            + v[1] * 2 * (qc[1] * qc[2] - qc[0] * qc[3])
            + v[2] * 2 * (qc[1] * qc[3] + qc[0] * qc[2]),
            v[0] * 2 * (qc[1] * qc[2] + qc[0] * qc[3])
            + v[1] * (qc[0] ** 2 - qc[1] ** 2 + qc[2] ** 2 - qc[3] ** 2)
            + v[2] * 2 * (qc[2] * qc[3] - qc[0] * qc[1]),
            v[0] * 2 * (qc[1] * qc[3] - qc[0] * qc[2])
            + v[1] * 2 * (qc[2] * qc[3] + qc[0] * qc[1])
            + v[2] * (qc[0] ** 2 - qc[1] ** 2 - qc[2] ** 2 + qc[3] ** 2),
        ],
        dtype=np.float32,
    )


def gravity_from_quat_fame(quat: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))


def _load_yaml(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _load_leg_position_limits(cfg: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    lower = cfg.get("legs_motor_pos_lower_limit_list")
    upper = cfg.get("legs_motor_pos_upper_limit_list")
    if lower is None or upper is None:
        return None, None

    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    if lower.shape != (NUM_LEG_JOINTS,) or upper.shape != (NUM_LEG_JOINTS,):
        raise ValueError(
            "legs_motor_pos_*_limit_list must each contain "
            f"{NUM_LEG_JOINTS} entries"
        )
    return lower, upper


# --------------------------------------------------------------------------- #
# Policy interface
# --------------------------------------------------------------------------- #
class Policy(ABC):
    """A swappable lower-body controller. Stateful (keeps history); reset on switch."""

    name: str

    #: 12 leg default angles, in /lowstate order — the nominal pose used by the
    #: handover gate to decide a switch is safe.
    nominal_lower: np.ndarray

    @abstractmethod
    def reset(self, state: RobotState) -> None:
        """Clear internal history and re-seed from the current state (bumpless)."""

    @abstractmethod
    def compute(self, state: RobotState) -> LegCommand:
        """Run one inference step and return a 12-joint leg setpoint."""


# --------------------------------------------------------------------------- #
# Walking policy (original)
# --------------------------------------------------------------------------- #
class WalkPolicy(Policy):
    name = "walk"
    GAIT_PERIOD = 0.8

    def __init__(self, config_path: str):
        cfg = _load_yaml(config_path)
        cfg_dir = os.path.dirname(os.path.abspath(config_path))
        policy_path = cfg.get("policy_path", "walkingPolicy.pt")
        if not os.path.isabs(policy_path):
            # walk.yaml's policy_path points at a training tree; fall back to the
            # weight shipped alongside the config.
            local = os.path.join(cfg_dir, "walkingPolicy.pt")
            policy_path = local if os.path.isfile(local) else os.path.join(cfg_dir, policy_path)

        self._kps = np.asarray(cfg["kps"], dtype=np.float32)
        self._kds = np.asarray(cfg["kds"], dtype=np.float32)
        self._default_angles = np.asarray(cfg["default_angles"], dtype=np.float32)
        self._ang_vel_scale = float(cfg["ang_vel_scale"])
        self._dof_pos_scale = float(cfg["dof_pos_scale"])
        self._dof_vel_scale = float(cfg["dof_vel_scale"])
        self._action_scale = float(cfg["action_scale"])
        self._cmd_scale = np.asarray(cfg["cmd_scale"], dtype=np.float32)
        self._num_actions = int(cfg["num_actions"])
        self._num_obs = int(cfg["num_obs"])
        self._lower_limit, self._upper_limit = _load_leg_position_limits(cfg)

        self._policy = torch.jit.load(policy_path)
        self._policy.eval()

        self.nominal_lower = self._default_angles.copy()
        self._action = np.zeros(self._num_actions, dtype=np.float32)
        self._obs = np.zeros(self._num_obs, dtype=np.float32)
        self._t_start: float | None = None

    def reset(self, state: RobotState) -> None:
        self._action[:] = 0.0
        self._t_start = state.t

    def compute(self, state: RobotState) -> LegCommand:
        if self._t_start is None:
            self._t_start = state.t

        n = self._num_actions
        q = state.q[:NUM_LEG_JOINTS]
        dq = state.dq[:NUM_LEG_JOINTS]
        qj = (q - self._default_angles) * self._dof_pos_scale
        dqj = dq * self._dof_vel_scale
        gravity = gravity_from_quat_walk(state.quat)
        omega = state.gyro * self._ang_vel_scale

        phase = ((state.t - self._t_start) % self.GAIT_PERIOD) / self.GAIT_PERIOD
        sin_phase, cos_phase = np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)

        self._obs[0:3] = omega
        self._obs[3:6] = gravity
        self._obs[6:9] = state.cmd * self._cmd_scale
        self._obs[9 : 9 + n] = qj
        self._obs[9 + n : 9 + 2 * n] = dqj
        self._obs[9 + 2 * n : 9 + 3 * n] = self._action
        self._obs[9 + 3 * n : 9 + 3 * n + 2] = (sin_phase, cos_phase)

        with torch.no_grad():
            self._action = self._policy(torch.from_numpy(self._obs).unsqueeze(0)).numpy().squeeze()
        target = self._action * self._action_scale + self._default_angles
        if self._lower_limit is not None:
            target = np.clip(target, self._lower_limit, self._upper_limit)
        return LegCommand(target_q=target, kp=self._kps, kd=self._kds)


# --------------------------------------------------------------------------- #
# FAME standing / squatting policy (RMA)
# --------------------------------------------------------------------------- #
class FamePolicy(Policy):
    """RMA loco-manipulation policy: env-factor encoder + history-conditioned actor.

    Controls the 12 legs; reads all 27 joints. ``e_t`` is built from the 15
    upper-body joint positions plus left/right wrist forces. There are no wrist
    force sensors in the sim, so the forces default to the YAML values (zeros) —
    the encoder still adapts the legs to the upper-body configuration the IK
    produces. Wire real wrist F/T into ``hand_force_provider`` to use them.
    """

    name = "fame"
    LATENT_DIM = 8
    Z_HISTORY = 3
    ET_DIM = 21  # 15 upper-body dof + left_xyz(3) + right_xyz(3)

    def __init__(self, config_path: str, hand_force_provider=None):
        cfg = _load_yaml(config_path)
        cfg_dir = os.path.dirname(os.path.abspath(config_path))

        def _resolve(key, default):
            p = cfg.get(key, default)
            return p if os.path.isabs(p) else os.path.join(cfg_dir, p)

        self._kps = np.asarray(cfg["kps"], dtype=np.float32)            # 12 legs
        self._kds = np.asarray(cfg["kds"], dtype=np.float32)
        self._default_angles = np.asarray(cfg["default_angles"], dtype=np.float32)        # 12
        self._default_angles_arms = np.asarray(cfg["default_angles_arms"], dtype=np.float32)  # 15
        self._ang_vel_scale = float(cfg["ang_vel_scale"])
        self._dof_pos_scale = float(cfg["dof_pos_scale"])
        self._dof_vel_scale = float(cfg["dof_vel_scale"])
        self._action_scale = float(cfg["action_scale"])
        self._cmd_scale = np.asarray(cfg["cmd_scale"], dtype=np.float32)
        self._num_actions = int(cfg["num_actions"])                     # 12
        self._num_obs = int(cfg["num_obs"])                            # 252
        self._obs_history_len = int(cfg["obs_history_len"])            # 3
        self._lower_limit, self._upper_limit = _load_leg_position_limits(cfg)
        self._left_force = np.asarray(cfg.get("left_hand_force", [0, 0, 0]), dtype=np.float32)
        self._right_force = np.asarray(cfg.get("right_hand_force", [0, 0, 0]), dtype=np.float32)
        self._hand_force_provider = hand_force_provider

        # Padded 27-joint default pose: legs + (torso, arms).
        self._padded_defaults = np.zeros(NUM_POLICY_JOINTS, dtype=np.float32)
        self._padded_defaults[:NUM_LEG_JOINTS] = self._default_angles
        self._padded_defaults[NUM_LEG_JOINTS:] = self._default_angles_arms

        self._policy = torch.jit.load(_resolve("policy_path", "policy.pt"))
        self._policy.eval()

        self._encoder = None
        enc_path = _resolve("encoder_path", "encoder_3800.pt")
        if os.path.isfile(enc_path):
            from h12_lowerbody_rl.rma import EnvFactorEncoder, EnvFactorEncoderCfg

            self._encoder = EnvFactorEncoder(EnvFactorEncoderCfg(in_dim=self.ET_DIM, latent_dim=self.LATENT_DIM))
            self._encoder.load_state_dict(torch.load(enc_path, map_location="cpu", weights_only=True))
            self._encoder.eval()

        self.nominal_lower = self._default_angles.copy()
        self._single_obs_dim = 3 + 1 + 3 + 3 + NUM_POLICY_JOINTS + NUM_POLICY_JOINTS + NUM_LEG_JOINTS  # 76
        self._action = np.zeros(self._num_actions, dtype=np.float32)
        self._obs_history: collections.deque = collections.deque(maxlen=self._obs_history_len)
        self._z_history = np.zeros((self.Z_HISTORY, self.LATENT_DIM), dtype=np.float32)

    @property
    def has_encoder(self) -> bool:
        return self._encoder is not None

    def _single_obs(self, state: RobotState) -> np.ndarray:
        qj = (state.q - self._padded_defaults) * self._dof_pos_scale
        dqj = state.dq * self._dof_vel_scale
        gravity = gravity_from_quat_fame(state.quat)
        omega = state.gyro * self._ang_vel_scale

        obs = np.zeros(self._single_obs_dim, dtype=np.float32)
        obs[0:3] = state.cmd * self._cmd_scale
        obs[3:4] = state.height_cmd
        obs[4:7] = omega
        obs[7:10] = gravity
        obs[10 : 10 + NUM_POLICY_JOINTS] = qj
        obs[10 + NUM_POLICY_JOINTS : 10 + 2 * NUM_POLICY_JOINTS] = dqj
        obs[10 + 2 * NUM_POLICY_JOINTS : 10 + 2 * NUM_POLICY_JOINTS + NUM_LEG_JOINTS] = self._action
        return obs

    def _encode(self, state: RobotState) -> np.ndarray:
        if self._encoder is None:
            return np.zeros(self.LATENT_DIM, dtype=np.float32)
        lf, rf = self._left_force, self._right_force
        if self._hand_force_provider is not None:
            lf, rf = self._hand_force_provider()
        upper = state.q[NUM_LEG_JOINTS:NUM_POLICY_JOINTS]  # 15, raw positions
        e_t = np.concatenate([upper, lf, rf]).astype(np.float32)
        with torch.no_grad():
            return self._encoder(torch.from_numpy(e_t).unsqueeze(0).float()).numpy().squeeze()

    def reset(self, state: RobotState) -> None:
        # Match the reference deploy exactly: start with ZERO-filled history and
        # let it warm up over the first few control ticks. Seeding with the live
        # takeover pose (a verified bug) feeds the actor a 252-d input far from
        # its training distribution on the first inference, producing a large
        # leg command that thrashes the robot / trips the safety e-stop. The
        # zero warm-up is what keeps the initial commands small and gentle.
        del state  # state-independent init, by design (kept for the Policy API)
        self._action[:] = 0.0
        self._obs_history.clear()
        for _ in range(self._obs_history_len):
            self._obs_history.append(np.zeros(self._single_obs_dim, dtype=np.float32))
        self._z_history[:] = 0.0

    def compute(self, state: RobotState) -> LegCommand:
        if len(self._obs_history) < self._obs_history_len:
            self.reset(state)

        single_obs = self._single_obs(state)
        self._obs_history.append(single_obs)

        z_t = self._encode(state)
        self._z_history[1:, :] = self._z_history[:-1, :]
        self._z_history[0, :] = z_t
        z_flat = np.flip(self._z_history, axis=0).flatten().astype(np.float32)

        proprio = np.concatenate(list(self._obs_history), axis=0)
        actor_obs = np.concatenate([proprio, z_flat], axis=0).astype(np.float32)
        if actor_obs.shape[0] != self._num_obs:
            raise ValueError(f"actor_obs dim {actor_obs.shape[0]} != num_obs {self._num_obs}")

        with torch.no_grad():
            self._action = self._policy(torch.from_numpy(actor_obs).unsqueeze(0)).numpy().squeeze()
        target = self._action * self._action_scale + self._default_angles
        if self._lower_limit is not None:
            target = np.clip(target, self._lower_limit, self._upper_limit)
        return LegCommand(target_q=target, kp=self._kps, kd=self._kds)
