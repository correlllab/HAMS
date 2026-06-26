"""Name resolution between the ROS/Unitree control stack and a robosuite-merged
H1-2 MuJoCo model.

When RoboCasa builds a task around the ``H1_2`` robot, every joint/actuator/
sensor/body/site is renamed (legs gain a ``leg`` substring) and prefixed
(``robot0_``), and indices are reordered. The ROS stack, however, speaks the
original Unitree names and assumes a fixed 27-motor order. This module centralises
the translation so the DDS interface, the magpie gripper bridge, and the sensor/
TF bridge all resolve indices by name instead of by hard-coded offsets.

Designed to be testable on the host against the *standalone* robot.xml by passing
``prefix=""`` (names there are already leg-renamed but not robosuite-prefixed).
"""
import mujoco
import numpy as np

# ROS/Unitree lowcmd motor order == the <actuator> order in h1_2_magpie.xml.
ROS_MOTOR_ORDER = [
    "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Must match CL_Assets/robosuite_assets/build_assets.py.
LEG_JOINT_RENAME = {
    f"{side}_{j}_joint": f"{side}_leg_{j}_joint"
    for side in ("left", "right")
    for j in ("hip_yaw", "hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll")
}

# Gripper actuator names per side, in the merged model. The standalone gripper
# defines two <position> actuators (left_finger_actuator / right_finger_actuator);
# robosuite prefixes them gripper0_<side>_.
GRIPPER_ACTUATORS = ["left_finger_actuator", "right_finger_actuator"]


def _id(model, objtype, name):
    return mujoco.mj_name2id(model, objtype, name)


class NameResolver:
    """Resolves ROS names to indices/addresses in a (possibly robosuite-merged)
    MjModel. ``robot_prefix`` is e.g. 'robot0_'; ``gripper_prefix`` formats to a
    side, e.g. 'gripper0_{side}_'. Use ''/'' for the standalone robot.xml."""

    def __init__(self, model, robot_prefix="robot0_", gripper_prefix="gripper0_{side}_"):
        self.model = model
        self.robot_prefix = robot_prefix
        self.gripper_prefix = gripper_prefix

        # body motors: ROS motor i -> (ctrl index, qpos addr, qvel/dof addr,
        # tau sensordata addr)
        self.motor_ctrl = np.empty(len(ROS_MOTOR_ORDER), dtype=int)
        self.motor_qpos = np.empty(len(ROS_MOTOR_ORDER), dtype=int)
        self.motor_qvel = np.empty(len(ROS_MOTOR_ORDER), dtype=int)
        self.motor_tau = np.empty(len(ROS_MOTOR_ORDER), dtype=int)
        for i, ros_name in enumerate(ROS_MOTOR_ORDER):
            sim_joint = robot_prefix + LEG_JOINT_RENAME.get(ros_name, ros_name)
            jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, sim_joint)
            if jid < 0:
                raise KeyError(f"joint not found in model: {sim_joint}")
            self.motor_qpos[i] = model.jnt_qposadr[jid]
            self.motor_qvel[i] = model.jnt_dofadr[jid]
            # actuator name == joint name in the source MJCF
            aid = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, sim_joint)
            if aid < 0:
                raise KeyError(f"actuator not found in model: {sim_joint}")
            self.motor_ctrl[i] = aid
            # <jointactuatorfrc> sensor "<ros_name>_torque" (keeps the original,
            # non-leg-renamed ROS name; see build_assets.py). This reports the
            # APPLIED torque clamped to the joint's actuatorfrcrange — the real
            # robot's "measured torque" semantics — not the raw actuator demand.
            tau_sensor = robot_prefix + ros_name + "_torque"
            sid = _id(model, mujoco.mjtObj.mjOBJ_SENSOR, tau_sensor)
            if sid < 0:
                raise KeyError(f"torque sensor not found in model: {tau_sensor}")
            self.motor_tau[i] = model.sensor_adr[sid]

        # IMU sensors (framequat/gyro/accelerometer) -> sensordata address + dim
        self.sensor_adr = {}
        for base in ("imu_quat", "imu_gyro", "imu_acc"):
            sid = _id(model, mujoco.mjtObj.mjOBJ_SENSOR, robot_prefix + base)
            if sid >= 0:
                self.sensor_adr[base] = (model.sensor_adr[sid], model.sensor_dim[sid])

    def gripper_ctrl(self, side):
        """ctrl indices for [left_finger_actuator, right_finger_actuator] of one
        gripper side ('right'/'left'). Returns list of ints (or -1 if absent)."""
        pre = self.gripper_prefix.format(side=side)
        return [_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, pre + a) for a in GRIPPER_ACTUATORS]

    def site_id(self, name):
        return _id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.robot_prefix + name)

    def body_id(self, name):
        return _id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.robot_prefix + name)

    def strip(self, name):
        """Strip the robot prefix from a sim name for ROS-facing TF frame_ids."""
        if name and name.startswith(self.robot_prefix):
            return name[len(self.robot_prefix):]
        return name


def self_test(robot_xml):
    """Validate the resolver against the standalone robot.xml (prefix='')."""
    m = mujoco.MjModel.from_xml_path(robot_xml)
    r = NameResolver(m, robot_prefix="", gripper_prefix="")
    assert len(r.motor_ctrl) == 27
    assert (r.motor_ctrl >= 0).all(), "all 27 body actuators resolved"
    assert (r.motor_tau >= 0).all(), "all 27 jointactuatorfrc torque sensors resolved"
    # legs renamed correctly
    assert r.motor_qpos[0] >= 0  # left_hip_yaw -> left_leg_hip_yaw
    assert set(r.sensor_adr) == {"imu_quat", "imu_gyro", "imu_acc"}
    print(f"NameResolver self-test OK: 27 motors, sensors={list(r.sensor_adr)}")
    return r


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    self_test(os.path.join(here, "..", "CL_Assets", "robosuite_assets", "robots", "h1_2", "robot.xml"))
