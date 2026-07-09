"""Standalone closed-loop test of FamePolicy in pure MuJoCo (no ROS/safety/band).

Replicates the reference deploy as closely as possible but drives the legs with
our FamePolicy class, to localize the fall: if FAME holds here, the bug is in the
ROS plumbing; if it falls here, the bug is in FamePolicy itself.

Feeds the policy the BASE free-joint orientation/ang-vel (d.qpos[3:7], d.qvel[3:6])
exactly like the reference. Run inside the ros container:
  python3 .../reference/standalone_fame_test.py
"""
import os
import numpy as np
import mujoco
from ament_index_python.packages import get_package_share_directory
from h12_lowerbody_rl.policy import FamePolicy, RobotState

POLICY_JOINT_NAMES = [
    "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
XML = "/home/code/CL_Assets/mujoco_assets/h1_2_magpie.xml"
CFG = os.path.join(get_package_share_directory("h12_lowerbody_rl"), "policies", "fame", "fame.yaml")


def quat_rpy(q):
    import math
    w, x, y, z = q
    roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = math.degrees(math.asin(max(-1, min(1, 2 * (w * y - z * x)))))
    return roll, pitch


def main():
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    dt = float(os.environ.get("DT", "0.002"))
    m.opt.timestep = dt
    decim = max(1, round(0.02 / dt))   # keep 50 Hz policy regardless of dt
    print("dt=%.4f  control_decimation=%d" % (dt, decim))

    qpos_adr, qvel_adr, act_idx = [], [], []
    for name in POLICY_JOINT_NAMES:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_adr.append(m.jnt_qposadr[jid])
        qvel_adr.append(m.jnt_dofadr[jid])
        act_idx.append(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name))
    qpos_adr = np.array(qpos_adr); qvel_adr = np.array(qvel_adr); act_idx = np.array(act_idx)
    leg = slice(0, 12); arm = slice(12, 27)

    fp = FamePolicy(CFG)
    print("encoder loaded:", fp.has_encoder)
    kps = fp._kps; kds = fp._kds                      # 12 legs (from fame.yaml)
    arm_kp = float(os.environ.get("ARM_KP", "200"))
    kps_arm = np.full(15, arm_kp); kds_arm = np.full(15, max(1.0, arm_kp * 0.025))
    print("arm hold kp=%.0f" % arm_kp)

    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)

    use_imu = bool(int(os.environ.get("USE_IMU", "0")))
    iq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
    ig = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
    iqa, iga = m.sensor_adr[iq], m.sensor_adr[ig]
    print("orientation source:", "TORSO IMU sensor (ROS /lowstate)" if use_imu else "BASE free-joint (reference)")

    def make_state(step):
        if use_imu:   # exactly what unitree_interface puts in /lowstate.imu_state
            quat = d.sensordata[iqa:iqa + 4].copy().astype(np.float32)
            gyro = d.sensordata[iga:iga + 3].copy().astype(np.float32)
        else:
            quat = d.qpos[3:7].copy().astype(np.float32)   # BASE orientation (reference)
            gyro = d.qvel[3:6].copy().astype(np.float32)    # BASE ang-vel (reference)
        return RobotState(
            q=d.qpos[qpos_adr].copy().astype(np.float32),
            dq=d.qvel[qvel_adr].copy().astype(np.float32),
            quat=quat, gyro=gyro,
            cmd=np.zeros(3, np.float32), height_cmd=1.0, t=step * 0.002,
        )

    fp.reset(make_state(0))
    target = fp._default_angles.copy()
    worst = 0.0
    print("t     base_z  roll   pitch  |legtau|max")
    nsteps = int(10.0 / dt)
    report_every = max(1, nsteps // 20)
    for step in range(nsteps):
        if step % decim == 0:
            cmd = fp.compute(make_state(step))
            target = cmd.target_q
        q_leg = d.qpos[qpos_adr[leg]]; dq_leg = d.qvel[qvel_adr[leg]]
        q_arm = d.qpos[qpos_adr[arm]]; dq_arm = d.qvel[qvel_adr[arm]]
        leg_tau = (target - q_leg) * kps - dq_leg * kds
        arm_tau = (0.0 - q_arm) * kps_arm - dq_arm * kds_arm
        d.ctrl[act_idx[leg]] = np.clip(leg_tau, -200, 200)
        d.ctrl[act_idx[arm]] = np.clip(arm_tau, -200, 200)
        mujoco.mj_step(m, d)
        if step % report_every == 0:
            r, p = quat_rpy(d.qpos[3:7])
            worst = max(worst, abs(r), abs(p))
            print("%4.2f  %6.3f %6.1f %6.1f   %6.1f" % (
                step * dt, d.qpos[2], r, p, float(np.max(np.abs(leg_tau)))))
    verdict = "HELD" if (worst < 25 and d.qpos[2] > 0.7) else "FELL"
    print("RESULT: worst_tilt=%.1f deg final_z=%.3f -> %s" % (worst, d.qpos[2], verdict))


if __name__ == "__main__":
    main()
