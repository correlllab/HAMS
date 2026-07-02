#!/usr/bin/env python3
"""Proprioceptive floating-base estimator for the MJPC lower-body controller.

Publishes a world-frame pelvis pose+twist as nav_msgs/Odometry on a dedicated
topic (default /h12_deploy_mjpc/base_state) from REAL proprioception only -- the
robot's IMU (orientation + gyro) and joint encoders (q, dq) on /lowstate -- via
leg odometry on the H1-2 MuJoCo kinematics. The MJPC controller consumes this
directly into the planner's floating-base state.

METHOD -- leg odometry, planted-foot constraint:
  A foot in contact is stationary in the world, so the base velocity is whatever
  cancels the joint + angular motion at that foot (evaluated with the base linear
  velocity zeroed), averaged over the *planted* feet. Base HEIGHT comes from the
  same kinematics (lowest planted foot on the floor), auto-calibrated so the
  reference pose matches the model's base height. Base XY is the integral of the
  estimated velocity (translation-invariant for balance). Orientation + gyro come
  straight from the IMU.

CONTACT GATE (audit H3 fix): the upstream estimator applied the planted-foot
constraint to EVERY foot with only a soft load-scaled noise, so a swinging /
sliding foot injected a phantom base velocity into the balance planner. Here each
foot is gated on an actual contact estimate (height above the floor, plus leg
load when tau_est is available) and NON-contact feet are DROPPED from the update
entirely -- not merely down-weighted.

MESSAGE CONTRACT (we own both ends; kept explicit to avoid frame bugs):
  pose.position          = world-frame pelvis position (x, y, z)
  pose.orientation       = pelvis orientation, ROS xyzw (from IMU wxyz)
  twist.linear           = WORLD-frame pelvis linear velocity  -> MuJoCo qvel[0:3]
  twist.angular          = BODY-frame angular velocity (gyro)  -> MuJoCo qvel[3:6]
  header.frame_id="odom", child_frame_id="pelvis"

Offline self-test (no ROS, no robot):
  python3 estimator_node.py --selftest
"""
import argparse
import os

import numpy as np

try:
    import mujoco
except ModuleNotFoundError:
    # The pure helpers (contact gate, RW-EKF) don't need MuJoCo; only the
    # kinematics / node paths do. Deferring keeps them unit-testable anywhere.
    mujoco = None

NJ = 27  # H1-2 handless actuated joints (== /lowstate motor order: legs 0..11, torso 12, arms 13..26)
NLEG = 12
IMU_OFFSET = np.array([-0.04452, -0.01891, 0.27756])  # pelvis -> IMU site, from h1_2_handless.xml
# CL_Assets is bind-mounted at /home/code/CL_Assets in the deploy image.
_DEFAULT_SCENE = "/home/code/CL_Assets/mujoco_assets/scene_h1_2_handless.xml"


def _quat2mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


def load_model_and_calibrate(scene):
    """Load the H1-2 model, find the foot bodies, and calibrate the ankle->floor
    height constant so the reference pose reproduces the model's base height."""
    scene = os.path.expanduser(scene or _DEFAULT_SCENE)
    d_dir = os.path.dirname(os.path.abspath(scene))
    cwd = os.getcwd()
    os.chdir(d_dir)
    try:
        m = mujoco.MjModel.from_xml_path(os.path.basename(scene))
    finally:
        os.chdir(cwd)
    data = mujoco.MjData(m)
    assert m.nq >= 7 + NJ, f"model nq={m.nq} too small (need free base + {NJ} joints)"

    foot_ids = []
    for bid in range(m.nbody):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if "ankle_roll" in nm:
            foot_ids.append((bid, nm))
    if not foot_ids:
        for bid in range(m.nbody):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if "ankle" in nm or "foot" in nm:
                foot_ids.append((bid, nm))
    if not foot_ids:
        raise RuntimeError("no foot bodies found (looked for *ankle_roll* / *ankle* / *foot*)")

    home = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home >= 0:
        home_q = np.array(m.key_qpos[home * m.nq: home * m.nq + m.nq])
    else:
        home_q = np.array(m.qpos0)
    home_base_z = float(home_q[2])

    # base at origin + identity orientation + reference joints -> lowest ankle z.
    data.qpos[:] = 0.0
    data.qpos[3] = 1.0
    data.qpos[7:7 + NJ] = home_q[7:7 + NJ]
    mujoco.mj_forward(m, data)
    ankle_z_home = min(float(data.xpos[bid][2]) for bid, _ in foot_ids)
    height_C = home_base_z + ankle_z_home   # base_height = -min(ankle_z) + C ; at ref -> home_base_z

    # leg-load motor indices per foot (knee, ankle-pitch): left=(3,4) right=(9,10)
    load_idx = [(3, 4) if "left" in nm else (9, 10) for _, nm in foot_ids]

    print(f"[est] model nq={m.nq} nv={m.nv} feet={[nm for _, nm in foot_ids]} "
          f"home_base_z={home_base_z:.3f} C={height_C:.3f}")
    return m, data, foot_ids, load_idx, height_C, home_q


def leg_kinematics(m, data, foot_ids, height_C, q, dq, quat, gyro, res):
    """One proprioceptive sample -> (floor_z, per-foot world height, per-foot world
    linear velocity with base-linvel zeroed). The base velocity that makes a planted
    foot stationary is the negation of that foot's computed world velocity."""
    data.qpos[:] = 0.0
    data.qpos[3:7] = quat                 # base at origin (xy nominal), real orientation
    data.qpos[7:7 + NJ] = q
    data.qvel[:] = 0.0
    data.qvel[3:6] = gyro                 # base angvel = body gyro (base linvel stays 0)
    data.qvel[6:6 + NJ] = dq
    mujoco.mj_forward(m, data)

    foot_z = np.array([float(data.xpos[bid][2]) for bid, _ in foot_ids])
    floor_z = float(foot_z.min())
    base_height = -floor_z + height_C
    vfeet = []
    for bid, _ in foot_ids:
        mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_BODY, bid, res, 0)
        vfeet.append(-res[3:6].copy())    # -(foot world linear velocity), base-lin=0
    return floor_z, foot_z, base_height, np.array(vfeet)


def contact_mask(foot_z, floor_z, tau, load_idx, height_thresh, load_thresh):
    """audit H3: a foot is in contact iff it is near the floor. When tau_est is
    meaningful, also require a minimum leg load. Returns a boolean per foot."""
    near_floor = (foot_z - floor_z) < height_thresh
    if tau is not None and float(np.abs(tau).sum()) > 1e-6:
        loaded = np.array([abs(tau[a]) + abs(tau[b]) > load_thresh for (a, b) in load_idx])
        return near_floor & loaded
    return near_floor


class RWEKF:
    """Random-walk velocity EKF over the planted-foot leg-odometry measurements.
    Only feet flagged in-contact are used; a Mahalanobis gate rejects spikes with a
    force-accept backstop so genuine motion is re-accepted."""

    def __init__(self, amax, r0, r1, lfloor, chi2, rejcap, dt):
        self.I3 = np.eye(3)
        self.P = self.I3 * 0.04
        self.Q = (amax * dt) ** 2 * self.I3
        self.r0, self.r1, self.lfloor = r0, r1, lfloor
        self.chi2, self.rejcap = chi2, rejcap
        self.v = np.zeros(3)
        self.rej = 0

    def update(self, vfeet, contact, tau, load_idx):
        self.P = self.P + self.Q
        for k in range(len(vfeet)):
            if not contact[k]:
                continue                       # audit H3: swing/non-contact foot dropped entirely
            a, b = load_idx[k]
            load_k = (abs(tau[a]) + abs(tau[b])) if tau is not None else 0.0
            rk = (self.r0 + self.r1 / max(load_k, self.lfloor)) ** 2
            S = self.P + rk * self.I3
            inn = vfeet[k] - self.v
            if float(inn @ np.linalg.solve(S, inn)) > self.chi2 and self.rej < self.rejcap:
                self.rej += 1
                self.P = self.P + self.Q   # inflate P while rejecting so the gate self-reopens
                continue
            self.rej = 0
            K = self.P @ np.linalg.inv(S)
            self.v = self.v + K @ inn
            self.P = (self.I3 - K) @ self.P
        return self.v


def _selftest(m, data, foot_ids, load_idx, height_C, home_q):
    res = np.zeros(6)
    qj = np.array(home_q[7:7 + NJ])
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    floor_z, foot_z, h, vfeet = leg_kinematics(m, data, foot_ids, height_C, qj,
                                               np.zeros(NJ), ident, np.zeros(3), res)
    cm = contact_mask(foot_z, floor_z, None, load_idx, 0.04, 5.0)
    ekf = RWEKF(3.0, 0.02, 0.3, 5.0, 11.34, 20, 1.0 /200.0)
    v = ekf.update(vfeet, cm, None, load_idx)
    print(f"[selftest] static ref: base_z={h:.3f} (expect ~{float(home_q[2]):.3f}) "
          f"contacts={cm.tolist()} v={np.round(v, 4)} (expect ~0)")
    dq = np.zeros(NJ); dq[3] = 0.5
    _, foot_z2, _, vfeet2 = leg_kinematics(m, data, foot_ids, height_C, qj, dq, ident,
                                           np.zeros(3), res)
    v2 = ekf.update(vfeet2, cm, None, load_idx)
    print(f"[selftest] joint moving: v={np.round(v2, 4)} (expect nonzero)")
    # a lifted (swing) foot must be excluded by the contact gate
    fz = foot_z.copy(); fz[0] = floor_z + 0.20
    cm_swing = contact_mask(fz, floor_z, None, load_idx, 0.04, 5.0)
    ok = (abs(h - float(home_q[2])) < 0.05 and bool(cm.all()) and np.linalg.norm(v) < 1e-6
          and np.linalg.norm(v2) > 1e-4 and not bool(cm_swing[0]) and bool(cm_swing[1]))
    print(f"[selftest] swing-foot gate: contacts={cm_swing.tolist()} (expect [False, True])")
    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="offline math check (no ROS), then exit")
    ap.add_argument("--scene", default=None, help="H1-2 MuJoCo scene (default: CL_Assets handless)")
    args, _ = ap.parse_known_args()

    if args.selftest:
        m, data, foot_ids, load_idx, height_C, home_q = load_model_and_calibrate(args.scene)
        raise SystemExit(_selftest(m, data, foot_ids, load_idx, height_C, home_q))

    # ROS imports deferred so --selftest runs without a sourced ROS environment.
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from nav_msgs.msg import Odometry
    from unitree_hg.msg import LowState

    class EstimatorNode(Node):
        def __init__(self):
            super().__init__("h12_deploy_mjpc_estimator")
            self.declare_parameter("scene", _DEFAULT_SCENE)
            self.declare_parameter("publish_hz", 200.0)
            self.declare_parameter("base_state_topic", "/h12_deploy_mjpc/base_state")
            self.declare_parameter("lowstate_topic", "/lowstate")
            self.declare_parameter("contact_height_m", 0.04)   # foot within this of floor -> contact
            self.declare_parameter("contact_load_nm", 5.0)     # min knee+ankle |tau| when tau usable
            self.declare_parameter("zero_yaw", False)          # remove initial heading (default: real yaw, as upstream)
            scene = self.get_parameter("scene").value
            self._hz = float(self.get_parameter("publish_hz").value)
            self._hthr = float(self.get_parameter("contact_height_m").value)
            self._lthr = float(self.get_parameter("contact_load_nm").value)
            self._zero_yaw = bool(self.get_parameter("zero_yaw").value)

            self.get_logger().info(f"loading H1-2 kinematics: {scene}")
            (self._m, self._data, self._foot_ids, self._load_idx,
             self._height_C, _home_q) = load_model_and_calibrate(scene)
            self._res = np.zeros(6)
            self._ekf = RWEKF(3.0, 0.02, 0.3, 5.0, 11.34, 20, 1.0 /self._hz)
            self._pos_xy = np.zeros(2)
            self._ls = None
            self._warned_short = False
            # Yaw-align the published base frame to the robot's INITIAL heading so
            # it "faces +x" as the Stabilize task assumes (its CoM / capture-point
            # costs are +x-forward; a yawed spawn otherwise makes the planner fight
            # the heading). Captured on the first sample as a rigid Rz(-yaw0).
            self._yaw0 = None
            self._Rz = np.eye(3)
            self._qcorr = np.array([1.0, 0.0, 0.0, 0.0])

            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
            self.create_subscription(LowState, self.get_parameter("lowstate_topic").value,
                                     self._on_lowstate, qos)
            self._pub = self.create_publisher(
                Odometry, self.get_parameter("base_state_topic").value, 10)
            self.create_timer(1.0 / self._hz, self._tick)
            self.get_logger().info(
                f"estimator ready: publishing {self.get_parameter('base_state_topic').value} "
                f"@ {self._hz:.0f}Hz (contact gate: height<{self._hthr}m + load>{self._lthr}Nm)")

        def _on_lowstate(self, msg: LowState):
            self._ls = msg

        def _tick(self):
            ls = self._ls
            if ls is None:
                return
            if len(ls.motor_state) < NJ and not self._warned_short:
                self.get_logger().warn(
                    f"/lowstate has {len(ls.motor_state)} motors (< {NJ}); reading available")
                self._warned_short = True
            n = min(NJ, len(ls.motor_state))
            q = np.zeros(NJ); dq = np.zeros(NJ); tau = np.zeros(NJ)
            for i in range(n):
                q[i] = ls.motor_state[i].q
                dq[i] = ls.motor_state[i].dq
                tau[i] = ls.motor_state[i].tau_est
            quat = np.asarray(ls.imu_state.quaternion, dtype=float)   # wxyz
            gyro = np.asarray(ls.imu_state.gyroscope, dtype=float)     # body frame

            floor_z, foot_z, base_height, vfeet = leg_kinematics(
                self._m, self._data, self._foot_ids, self._height_C, q, dq, quat, gyro, self._res)
            contact = contact_mask(foot_z, floor_z, tau, self._load_idx, self._hthr, self._lthr)
            v = self._ekf.update(vfeet, contact, tau, self._load_idx)

            dt = 1.0 / self._hz
            self._pos_xy += v[0:2] * dt

            # Capture the initial yaw once, then rotate the published base frame by
            # Rz(-yaw0) so the robot faces +x (matches the Stabilize task frame).
            if self._zero_yaw and self._yaw0 is None:
                self._yaw0 = float(np.arctan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                                              1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2)))
                c, s = float(np.cos(self._yaw0)), float(np.sin(self._yaw0))
                self._Rz = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
                self._qcorr = np.array([np.cos(self._yaw0 / 2.0), 0.0, 0.0,
                                        -np.sin(self._yaw0 / 2.0)])
                self.get_logger().info(
                    "[estimator] yaw-aligned base frame: removing initial yaw %.1f deg"
                    % np.degrees(self._yaw0))
            q_pub = np.zeros(4)
            mujoco.mju_mulQuat(q_pub, self._qcorr, np.asarray(quat, dtype=float))
            pos_pub = self._Rz @ np.array([self._pos_xy[0], self._pos_xy[1], base_height])
            v_pub = self._Rz @ v

            odom = Odometry()
            odom.header.stamp = self.get_clock().now().to_msg()
            odom.header.frame_id = "odom"
            odom.child_frame_id = "pelvis"
            odom.pose.pose.position.x = float(pos_pub[0])
            odom.pose.pose.position.y = float(pos_pub[1])
            odom.pose.pose.position.z = float(pos_pub[2])
            # IMU quat is wxyz; ROS geometry_msgs/Quaternion is xyzw.
            odom.pose.pose.orientation.w = float(q_pub[0])
            odom.pose.pose.orientation.x = float(q_pub[1])
            odom.pose.pose.orientation.y = float(q_pub[2])
            odom.pose.pose.orientation.z = float(q_pub[3])
            # Contract: twist.linear = WORLD-frame linvel; twist.angular = BODY-frame gyro.
            odom.twist.twist.linear.x = float(v_pub[0])
            odom.twist.twist.linear.y = float(v_pub[1])
            odom.twist.twist.linear.z = float(v_pub[2])
            odom.twist.twist.angular.x = float(gyro[0])
            odom.twist.twist.angular.y = float(gyro[1])
            odom.twist.twist.angular.z = float(gyro[2])
            self._pub.publish(odom)

    rclpy.init()
    node = EstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
