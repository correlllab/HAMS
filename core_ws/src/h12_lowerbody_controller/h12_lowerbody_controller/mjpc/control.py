"""Control-law helpers for the MJPC deploy node.

The Python analog of the C++ node's gravity feed-forward (``g_model`` /
``qfrc_bias``), bring-up ramp, torque-aware target clamp, and whole-body LowCmd
assembly. Kept free of DDS plumbing so each piece is testable on its own.
"""

from __future__ import annotations

import numpy as np

from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

from .constants import KNU, KP, KV, TAU_ESTOP


class GravityFeedforward:
    """``tau = gff * qfrc_bias(qpos, qvel=0)[6 : 6+27]`` on a LOCAL mujoco model.

    Separate from the gRPC planner: it loads its own ``mujoco.MjModel`` of the H1-2
    (pip mujoco — self-contained, so the struct-layout mismatch with agent_server
    is irrelevant). Degrades gracefully to zero feed-forward if the model can't be
    loaded or ``gff == 0`` (the skeleton stays runnable without a perfect model).

    The model's joint order MUST match the /lowstate (27-motor) order for the
    ``qfrc_bias[6:]`` slice to line up; pass a bare H1-2 free-joint model.
    """

    def __init__(self, model_xml_path: str | None, gff: float = 0.85):
        self._gff = float(gff)
        self._model = None
        self._data = None
        self._warned = False
        if model_xml_path and self._gff != 0.0:
            try:
                import mujoco  # local import: only needed when gravity FF is on

                self._model = mujoco.MjModel.from_xml_path(model_xml_path)
                self._data = mujoco.MjData(self._model)
                self._mj = mujoco
            except Exception as exc:  # pragma: no cover - depends on the host XML
                print(f"[mjpc] GravityFeedforward: could not load '{model_xml_path}' "
                      f"({exc}); gravity feed-forward DISABLED (tau=0).")

    @property
    def enabled(self) -> bool:
        return self._model is not None

    def tau(self, qpos: np.ndarray) -> np.ndarray:
        """27-vector of gravity-compensation torques for the given full qpos."""
        if self._model is None:
            return np.zeros(KNU)
        n = min(len(qpos), self._model.nq)
        self._data.qpos[:] = 0.0
        self._data.qpos[:n] = np.asarray(qpos)[:n]
        self._data.qvel[:] = 0.0
        self._mj.mj_forward(self._model, self._data)
        return self._gff * np.asarray(self._data.qfrc_bias[6 : 6 + KNU])


class BringUpRamp:
    """Blend joint targets measured-pose -> stance -> live policy at start-up.

    Simplified port of the C++ start-ramp / ramp-hold / policy-blend logic: rise to
    the stance over ``ramp_sec`` while the planner converges, hold it ``hold_sec``,
    then ease into the live policy target over ``blend_sec`` (0 = hard switch).
    Returns the commanded 27-vector for the tick.

    NOTE (skeleton): the C++ additionally rescales the ramp by the measured
    distance-from-home and has a separate live-switch settle/blend — left as TODOs.
    """

    def __init__(self, ramp_sec: float = 5.0, hold_sec: float = 3.0, blend_sec: float = 0.0):
        self.ramp_sec = float(ramp_sec)
        self.hold_sec = float(hold_sec)
        self.blend_sec = float(blend_sec)

    def target(
        self,
        wall: float,
        q_init: np.ndarray,
        stance: np.ndarray,
        policy_target: np.ndarray,
        warming: bool,
    ) -> np.ndarray:
        q_init = np.asarray(q_init, dtype=float)
        stance = np.asarray(stance, dtype=float)
        policy_target = np.asarray(policy_target, dtype=float)

        if self.ramp_sec <= 0.0:
            return q_init.copy() if warming else policy_target.copy()

        t_handover = self.ramp_sec + self.hold_sec
        a = min(1.0, max(0.0, wall / self.ramp_sec))

        if a < 1.0 or warming or wall < t_handover:
            tgt = stance                                   # rising / warmup / scripted hold
        elif self.blend_sec > 0.0 and wall < t_handover + self.blend_sec:
            b = (wall - t_handover) / self.blend_sec       # ease stance -> policy
            tgt = (1.0 - b) * stance + b * policy_target
        else:
            tgt = policy_target                            # full policy authority

        return (1.0 - a) * q_init + a * tgt


def target_clamp(
    tgt: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray = KP,
    tau_estop: np.ndarray = TAU_ESTOP,
    ratio: float = 0.9,
) -> np.ndarray:
    """Bound each target so ``|kp*(tgt-q)| <= ratio*tau_estop`` (≙ C++ --target_clamp).

    Makes a command-side tau-ESTOP impossible for any planner output, and acts as
    the per-tick slew limit the safety layer lacks.
    """
    dmax = ratio * np.asarray(tau_estop) / np.asarray(kp)
    q = np.asarray(q, dtype=float)
    return np.clip(np.asarray(tgt, dtype=float), q - dmax, q + dmax)


_CRC = CRC()


def build_lowcmd(tgt_q: np.ndarray, tau: np.ndarray, mode_machine: int,
                 kp: np.ndarray = KP, kv: np.ndarray = KV, joint_indices=None):
    """Build a unitree_hg ``LowCmd_`` (PR mode) + CRC (≙ C++ build).

    ``mode_pr=0`` (Pitch/Roll series mode), ``mode_machine`` echoed from the latest
    state (required by the real robot). Position-PD: q target + kp/kd, with the
    gravity feed-forward in ``tau``.

    ``joint_indices`` selects which motors to drive (mode=1); the rest stay at the
    default (mode=0), which is what the h12_safety_layer split merge expects from a
    single-half publisher. ``None`` = whole body (all 27).
    """
    cmd = unitree_hg_msg_dds__LowCmd_()
    cmd.mode_pr = 0
    cmd.mode_machine = int(mode_machine)
    for i in range(KNU) if joint_indices is None else joint_indices:
        mc = cmd.motor_cmd[i]
        mc.mode = 1  # enable
        mc.q = float(tgt_q[i])
        mc.dq = 0.0
        mc.tau = float(tau[i])
        mc.kp = float(kp[i])
        mc.kd = float(kv[i])
    cmd.crc = _CRC.Crc(cmd)
    return cmd
