"""Pelvis (free-joint) state reconstruction from H1-2 DDS feedback.

The Python analog of the C++ node's ``fill_state`` / the referenced
``mjpc_dds_bridge.py:pelvis_from_site``. It backs the pelvis free-joint pose out of
the reported IMU-site world pose (``rt/sportmodestate``) and the IMU orientation /
gyro (``rt/lowstate``)::

    base_p = site_p - R(quat) * IMU_OFFSET
    base_v = site_v - (R(quat) * gyro) x (R(quat) * IMU_OFFSET)

Pure NumPy (no mujoco / DDS imports) so it is unit-testable in isolation and emits
plain ``qpos`` / ``qvel`` arrays — which is what ``mujoco_mpc.Agent.set_state``
ships over gRPC, so the pip-mujoco vs agent_server struct-layout difference never
matters on the control path.

State vector layout (robot dofs only; task-object slots are filled by the agent
client)::

    qpos[0:3]   pelvis world position
    qpos[3:7]   pelvis orientation quaternion (w, x, y, z)
    qpos[7:34]  27 joint positions (/lowstate order)
    qvel[0:3]   pelvis world linear velocity
    qvel[3:6]   pelvis angular velocity (== body gyro, free-joint convention)
    qvel[6:33]  27 joint velocities
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .constants import IMU_OFFSET, KNU


@dataclasses.dataclass
class MjpcState:
    """Copyable snapshot of the latest robot feedback (≙ the C++ ``StateData``).

    Populated by the DDS handlers; ``snapshot()`` takes a deep copy under the lock
    so the control loop never reads a half-written frame.
    """

    have_ls: bool = False  # rt/lowstate seen
    have_ss: bool = False  # rt/sportmodestate seen
    q: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(KNU))
    dq: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(KNU))
    quat: np.ndarray = dataclasses.field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    gyro: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    site_p: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    site_v: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    mode_machine: int = 0
    tick: int = 0

    def snapshot(self) -> "MjpcState":
        return MjpcState(
            have_ls=self.have_ls,
            have_ss=self.have_ss,
            q=self.q.copy(),
            dq=self.dq.copy(),
            quat=self.quat.copy(),
            gyro=self.gyro.copy(),
            site_p=self.site_p.copy(),
            site_v=self.site_v.copy(),
            mode_machine=self.mode_machine,
            tick=self.tick,
        )


def quat_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by quaternion ``q`` (w, x, y, z) — matches the C++ ``QuatRot``."""
    w, x, y, z = q
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return np.array(
        [
            v[0] + w * tx + (y * tz - z * ty),
            v[1] + w * ty + (z * tx - x * tz),
            v[2] + w * tz + (x * ty - y * tx),
        ]
    )


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a*b for quaternions in (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Unit quaternion (w, x, y, z) for a rotation of ``angle`` rad about ``axis``."""
    h = 0.5 * angle
    s = np.sin(h)
    a = np.asarray(axis, dtype=float)
    return np.array([np.cos(h), a[0] * s, a[1] * s, a[2] * s])


def apply_imu_offsets(quat: np.ndarray, pitch_off: float = 0.0, roll_off: float = 0.0) -> np.ndarray:
    """Cancel a measured IMU mounting zero-offset (body-frame post-multiply), ≙ the
    C++ pitch/roll offset block. ``pitch_off``/``roll_off`` in radians; 0 = no-op."""
    bq = np.asarray(quat, dtype=float).copy()
    if pitch_off != 0.0:
        bq = quat_mul(bq, axis_angle_quat(np.array([0.0, 1.0, 0.0]), pitch_off))
    if roll_off != 0.0:
        bq = quat_mul(bq, axis_angle_quat(np.array([1.0, 0.0, 0.0]), roll_off))
    n = np.linalg.norm(bq)
    return bq / n if n > 0 else bq


def pelvis_from_site(
    s: MjpcState,
    nominal_base_p: np.ndarray,
    require_sport: bool = True,
    imu_pitch_off: float = 0.0,
    imu_roll_off: float = 0.0,
    ankle_off_l: float = 0.0,
    ankle_off_r: float = 0.0,
):
    """Reconstruct ``(qpos, qvel)`` for the H1-2 free-joint + 27 dofs (≙ ``fill_state``).

    When ``require_sport`` and ``s.have_ss`` the world base pose/vel is derived from
    the IMU-site report; otherwise (twin / debug mode) a nominal standing base is
    held with zero base linear velocity (balance off live IMU + joints).

    Returns ``qpos`` (7 + 27) and ``qvel`` (6 + 27) NumPy arrays.
    """
    roff = quat_rot(s.quat, IMU_OFFSET)          # R * IMU_OFFSET
    ww = quat_rot(s.quat, s.gyro)                # world angular velocity
    cr = np.cross(ww, roff)                       # (R*gyro) x roff

    qpos = np.zeros(7 + KNU)
    qvel = np.zeros(6 + KNU)

    if require_sport and s.have_ss:
        qpos[0:3] = s.site_p - roff               # pelvis = site - R*offset
        qvel[0:3] = s.site_v - cr                 # pelvis world linvel (cross-stream)
    else:
        qpos[0:3] = np.asarray(nominal_base_p, dtype=float)
        qvel[0:3] = 0.0

    qpos[3:7] = apply_imu_offsets(s.quat, imu_pitch_off, imu_roll_off)
    qpos[7 : 7 + KNU] = s.q
    # ankle-roll zero-offset calibration: plan against the corrected roll (idx 5=L,
    # 11=R) so the planner doesn't chase a foot only the encoder thinks is rolled.
    qpos[7 + 5] -= ankle_off_l
    qpos[7 + 11] -= ankle_off_r

    qvel[3:6] = s.gyro                            # free-joint angvel == body gyro
    qvel[6 : 6 + KNU] = s.dq

    # TODO(skeleton): single-stream base-velocity finite-diff + LPF over the sim
    # clock (the C++ --vel_lpf_ms fix that drops the two-stream gyro x r phantom).
    # The cross-stream value above is the A/B baseline; the node applies the LPF.

    return qpos, qvel
