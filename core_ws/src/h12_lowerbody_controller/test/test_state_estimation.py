"""Unit tests for the MJPC pelvis-from-IMU-site reconstruction.

Mirrors the C++ node's note that ``fill_state`` / ``pelvis_from_site`` was
"unit-tested vs ground truth". Pure NumPy, no DDS/mujoco/ROS — runs standalone.
"""

import numpy as np

from h12_lowerbody_controller.mjpc.constants import IMU_OFFSET, KNU
from h12_lowerbody_controller.mjpc.state_estimation import (
    MjpcState,
    axis_angle_quat,
    pelvis_from_site,
    quat_rot,
)


def _state(**kw) -> MjpcState:
    s = MjpcState()
    for k, v in kw.items():
        setattr(s, k, np.asarray(v) if isinstance(v, (list, tuple)) else v)
    return s


def test_quat_rot_identity():
    v = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(quat_rot(np.array([1.0, 0, 0, 0]), v), v, atol=1e-12)


def test_quat_rot_90deg_yaw():
    # +90 deg about z maps x -> y.
    q = axis_angle_quat(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    out = quat_rot(q, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(out, [0.0, 1.0, 0.0], atol=1e-9)


def test_pelvis_identity_with_sportstate():
    q = np.linspace(-0.3, 0.3, KNU)
    dq = np.linspace(0.1, -0.1, KNU)
    site_p = np.array([0.5, -0.2, 1.1])
    site_v = np.array([0.01, 0.02, -0.03])
    s = _state(have_ls=True, have_ss=True, q=q, dq=dq,
               quat=[1.0, 0, 0, 0], gyro=[0.0, 0, 0], site_p=site_p, site_v=site_v)

    qpos, qvel = pelvis_from_site(s, nominal_base_p=[0, 0, 1.03], require_sport=True)

    assert qpos.shape == (7 + KNU,)
    assert qvel.shape == (6 + KNU,)
    # identity orientation -> pelvis = site - IMU_OFFSET, linvel == site_v (gyro=0)
    np.testing.assert_allclose(qpos[0:3], site_p - IMU_OFFSET, atol=1e-12)
    np.testing.assert_allclose(qpos[3:7], [1, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(qpos[7:7 + KNU], q, atol=1e-12)
    np.testing.assert_allclose(qvel[0:3], site_v, atol=1e-12)
    np.testing.assert_allclose(qvel[3:6], 0.0, atol=1e-12)
    np.testing.assert_allclose(qvel[6:6 + KNU], dq, atol=1e-12)


def test_pelvis_debug_mode_holds_nominal_base():
    s = _state(have_ls=True, have_ss=False, q=np.zeros(KNU), dq=np.zeros(KNU),
               quat=[1.0, 0, 0, 0], gyro=[0.0, 0, 0])
    nominal = [0.0, 0.0, 1.03]
    qpos, qvel = pelvis_from_site(s, nominal_base_p=nominal, require_sport=True)
    # no sportstate -> nominal base + zero base linvel
    np.testing.assert_allclose(qpos[0:3], nominal, atol=1e-12)
    np.testing.assert_allclose(qvel[0:3], 0.0, atol=1e-12)


def test_pelvis_gyro_cross_term_subtracted():
    # With a non-zero gyro the base linvel must subtract (R*gyro) x (R*offset).
    gyro = np.array([0.0, 0.0, 1.0])
    site_v = np.array([0.0, 0.0, 0.0])
    s = _state(have_ls=True, have_ss=True, q=np.zeros(KNU), dq=np.zeros(KNU),
               quat=[1.0, 0, 0, 0], gyro=gyro, site_p=[0, 0, 1.1], site_v=site_v)
    qpos, qvel = pelvis_from_site(s, nominal_base_p=[0, 0, 1.03], require_sport=True)
    expected = site_v - np.cross(gyro, IMU_OFFSET)  # identity orientation
    np.testing.assert_allclose(qvel[0:3], expected, atol=1e-12)
    assert not np.allclose(qvel[0:3], 0.0)  # the cross term is actually non-zero here
