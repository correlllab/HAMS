import math

import mujoco
import numpy as np
import threading
import time
import os

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher

from unitree_sdk2py.utils.thread import RecurrentThread

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__SportModeState_ as SportModeState_default)

TOPIC_LOWCMD = 'rt/lowcmd'
TOPIC_LOWSTATE = 'rt/lowstate'
TOPIC_HIGHSTATE = 'rt/sportmodestate'
# 27 motors on main body, 3 sensors on each motor
MOTOR_NUM = 27
MOTOR_SENSOR_NUM = 3

DOMAIN_ID = int(os.getenv("ROS_DOMAIN_ID"))
assert DOMAIN_ID > 0 and isinstance(DOMAIN_ID, int), "Please set ROS_DOMAIN_ID environment variable to a positive value, e.g. export ROS_DOMAIN_ID=1, domain 0 is reserved for the real robot."
def _quat_mul(a, b):
    """Hamilton product of two wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


class SimInterface:
    def __init__(self, model, data, lock=None, resolver=None,
                 truth_sportstate=False, spawn_frame=None):
        # record mujoco model & data
        self.model = model
        self.data = data
        # sim_names.NameResolver for the robosuite-merged model: joints/actuators/
        # sensors are renamed + robot0_-prefixed, so state is read from
        # qpos/qvel, the jointactuatorfrc torque sensors, and the IMU sensors via
        # name-resolved indices, and ctrl is written by resolved actuator index.
        if resolver is None:
            raise ValueError("SimInterface requires a NameResolver")
        self.resolver = resolver
        # Lock shared with the main sim loop to protect MjData access.
        # RecurrentThreads read state and write ctrl from background threads while
        # the main loop calls mj_step — all must hold the lock while touching data.
        self._lock = lock or threading.Lock()

        # initialize state parameters
        self.num_motor = MOTOR_NUM
        self.dt = self.model.opt.timestep
        self.have_imu = "imu_quat" in resolver.sensor_adr

        # SPAWN-FRAME STATE NORMALIZATION (FABEL 2026-07-06). The twin and the
        # REAL robot both hand the controller state in the ROBOT'S HOME frame:
        # the twin spawns at the model origin facing +x, and the real IMU boots
        # with yaw ~= 0 / odometry from 0. RoboCasa instead publishes raw
        # KITCHEN-world pose (spawn xy ~(-2.2,-3.6), arbitrary yaw) -- which
        # poisons every model-frame-referenced cost in the planner (e.g. the
        # Lean task's Body Yaw term aligns the torso with the direction to the
        # planner-model object at ~(1.1,0): a phantom heading the policy then
        # chases, yawing/pivoting the robot off its feet). spawn_frame =
        # {'yaw0': spawn yaw, 'p0': [x0, y0, z_floor]} re-expresses ALL
        # published state in the spawn frame: lowstate IMU quaternion gets the
        # spawn yaw rotated out; truth sportmodestate position/velocity are
        # translated to the spawn origin, yaw-rotated, and z-referenced to the
        # walking-surface top -- byte-identical semantics to the twin bench.
        self.spawn_frame = spawn_frame
        if spawn_frame is not None:
            y0 = float(spawn_frame["yaw0"])
            self._sf_qz = (math.cos(-y0 / 2.0), 0.0, 0.0, math.sin(-y0 / 2.0))
            c, s = math.cos(y0), math.sin(y0)
            # world->spawn-frame rotation Rz(-yaw0)
            self._sf_R = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
            self._sf_p0 = np.array([float(spawn_frame["p0"][0]),
                                    float(spawn_frame["p0"][1]),
                                    float(spawn_frame["p0"][2])])
            print(f"[unitree_interface] SPAWN-FRAME state: yaw0="
                  f"{math.degrees(y0):.1f}deg p0={self._sf_p0.round(3)} -- "
                  "published lowstate quat / sportstate pos are spawn-relative "
                  "(twin/real convention)")

        # initialize channel
        ChannelFactoryInitialize(id=DOMAIN_ID)
        # publish low state
        self.low_state = LowState_default()
        self.low_state_publisher = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self.low_state_publisher.Init()
        self.low_state_thread = RecurrentThread(interval=self.dt, target=self.publish_low_state, name='sim_lowstate')
        self.low_state_thread.Start()

        # TRUTH sportmodestate (TWIN PARITY, 2026-07-04): the digital twin's
        # SimInterface publishes GROUND-TRUTH base position/velocity on
        # rt/sportmodestate (framepos/framelinvel of the IMU site) — the deploy
        # node was validated against truth state, and the twin bench never runs
        # the RW-EKF estimator. On this plant the estimator's leg-odometry xy
        # wanders ~0.3 m under sim foot micro-slip, feeding phantom base
        # velocities into the policy's capture-point terms. This publisher
        # mirrors the twin exactly: position = IMU-site world pos, velocity =
        # IMU-site world linvel (the node back-projects to the pelvis with its
        # kImuOffset). Opt-in; when used, park the estimator on another topic.
        self.imu_site = -1
        if truth_sportstate and self.have_imu:
            qa = resolver.sensor_adr["imu_quat"][0]
            for s in range(model.nsensor):
                if int(model.sensor_adr[s]) == int(qa):
                    self.imu_site = int(model.sensor_objid[s])
                    break
        if truth_sportstate and self.imu_site >= 0:
            self.high_state = SportModeState_default()
            self.high_state_publisher = ChannelPublisher(
                TOPIC_HIGHSTATE, SportModeState_)
            self.high_state_publisher.Init()
            self.high_state_thread = RecurrentThread(
                interval=self.dt, target=self.publish_high_state,
                name='sim_highstate')
            self.high_state_thread.Start()
            print("[unitree_interface] TRUTH rt/sportmodestate publisher ON "
                  "(twin-parity ground-truth base state; estimator must be "
                  "parked on another topic)")
        elif truth_sportstate:
            print("[unitree_interface] truth sportstate requested but no IMU "
                  "site found -- NOT publishing")

        # subscribe to low command
        self.low_cmd_subscriber = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self.low_cmd_subscriber.Init(self.low_cmd_handler, 10)

        # Timeout detection. When no controller command has been received
        # for `timeout` seconds, write zero ctrl on every motor and let the
        # robot go limp. Without an upright tether, the H1-2 falls and lays
        # on the floor — preferable to a stiff pose-hold snapshot, which
        # tries to freeze whatever chaotic mid-fall pose was sampled at the
        # moment of timeout and tends to NaN against contact dynamics.
        # The gap is measured in SIM time once commands have started: with the
        # full sensor set each lidar+camera cycle holds the sim lock up to ~1 s
        # of WALL time, blocking low_cmd_handler with it — a wall-clocked
        # watchdog then fired once per sensor cycle (294 times in one bringup,
        # 2026-07-03), injecting a limp zero-ctrl step into every cycle even
        # though NO sim time passed without a command. A frozen world can't
        # miss commands; a genuinely dead controller still trips this because
        # sim time keeps advancing without new lowcmds.
        self.last_cmd_time = time.time()
        self.last_cmd_simtime = None   # sim clock stamp of the last lowcmd
        self.last_cmd_kp = 0.0         # max commanded stiffness (engagement gate)
        self.timeout = 0.1
        self.timeout_detected = False
        self.timeout_thread = RecurrentThread(
            interval=0.01, target=self.check_cmd_timeout, name='cmd_watchdog'
        )
        self.timeout_thread.Start()

    @property
    def lock(self):
        return self._lock

    def publish_low_state(self):
        if self.data is None:
            return
        r = self.resolver
        sd = self.data.sensordata
        with self._lock:
            self.low_state.tick = int(self.data.time / self.dt)
            # q/dq/tau by name-resolved index (robosuite-merged model).
            # tau_est comes from the <jointactuatorfrc> sensor (clamped to the
            # joint's actuatorfrcrange), matching the real robot's measured-torque
            # semantics. Reading data.actuator_force instead would publish the raw,
            # unclamped PD demand and spuriously trip the safety layer's estop.
            for i in range(self.num_motor):
                self.low_state.motor_state[i].q = float(self.data.qpos[r.motor_qpos[i]])
                self.low_state.motor_state[i].dq = float(self.data.qvel[r.motor_qvel[i]])
                self.low_state.motor_state[i].tau_est = float(sd[r.motor_tau[i]])
            if self.have_imu:
                qa = r.sensor_adr["imu_quat"][0]
                ga = r.sensor_adr["imu_gyro"][0]
                aa = r.sensor_adr["imu_acc"][0]
                if self.spawn_frame is not None:
                    # rotate the spawn yaw out of the world-frame framequat --
                    # the real robot's IMU boots at yaw ~0, the twin spawns at
                    # yaw 0; gyro/accel are body-frame and stay untouched.
                    q = _quat_mul(self._sf_qz, (float(sd[qa]), float(sd[qa + 1]),
                                                float(sd[qa + 2]), float(sd[qa + 3])))
                    for k in range(4):
                        self.low_state.imu_state.quaternion[k] = q[k]
                else:
                    for k in range(4):
                        self.low_state.imu_state.quaternion[k] = float(sd[qa + k])
                for k in range(3):
                    self.low_state.imu_state.gyroscope[k] = float(sd[ga + k])
                for k in range(3):
                    self.low_state.imu_state.accelerometer[k] = float(sd[aa + k])
        # write to low state (DDS publish is thread-safe, no lock needed)
        self.low_state_publisher.Write(self.low_state)

    def publish_high_state(self):
        if self.data is None or self.imu_site < 0:
            return
        v6 = np.zeros(6)   # [angular(3); linear(3)], world frame
        with self._lock:
            mujoco.mj_objectVelocity(self.model, self.data,
                                     mujoco.mjtObj.mjOBJ_SITE,
                                     self.imu_site, v6, 0)
            p = self.data.site_xpos[self.imu_site].copy()
        v = v6[3:6]
        if self.spawn_frame is not None:
            # spawn-relative xy, floor-referenced z, yaw-derotated -- the
            # twin's world (robot home frame). Consistent with the normalized
            # lowstate quaternion: the node's pelvis reconstruction
            # p - R*kImuOffset stays exact under the shared rigid transform.
            p = self._sf_R @ (p - self._sf_p0)
            v = self._sf_R @ v
        for k in range(3):
            self.high_state.position[k] = float(p[k])
            self.high_state.velocity[k] = float(v[k])
        self.high_state_publisher.Write(self.high_state)

    def low_cmd_handler(self, msg: LowCmd_):
        if self.data is None:
            return
        r = self.resolver
        with self._lock:
            self.last_cmd_time = time.time()
            self.last_cmd_simtime = float(self.data.time)
            _kmax = 0.0
            # apply control to each motor (tau + kp*(q*-q) + kd*(dq*-dq))
            for i in range(self.num_motor):
                ci = int(r.motor_ctrl[i])
                q_cur = self.data.qpos[r.motor_qpos[i]]
                dq_cur = self.data.qvel[r.motor_qvel[i]]
                mc = msg.motor_cmd[i]
                if mc.mode == 1:
                    self.data.ctrl[ci] = mc.tau + mc.kp * (mc.q - q_cur) + mc.kd * (mc.dq - dq_cur)
                    if mc.kp > _kmax:
                        _kmax = mc.kp
                else:
                    self.data.ctrl[ci] = 0.0
            # max commanded stiffness this cmd — the harness gates joint-hold
            # release on kp>1 (a STIFF controller command), NOT on last_cmd_time:
            # the safety layer idles at kp=0 zeros, which would release the hold
            # before the controller is driving. Twin-parity (its _stiff["kp"] gate).
            self.last_cmd_kp = _kmax

    def check_cmd_timeout(self):
        # Pre-first-command: wall clock (original behavior — the spawn-limp zero).
        # After commands start: SIM-time gap, so sensor-render world freezes
        # (no sim time elapses) can't fake a dead controller.
        if self.last_cmd_simtime is None:
            timed_out = (time.time() - self.last_cmd_time) > self.timeout
        else:
            timed_out = (float(self.data.time) - self.last_cmd_simtime) > self.timeout
        if timed_out:
            if not self.timeout_detected:
                self.timeout_detected = True
                print('Command timeout! Releasing motors (zero ctrl).')
                # Zero ctrl once on first detection — robot falls limp.
                # Subsequent ticks would just re-write zeros into already-zero
                # ctrl until a new command resets last_cmd_time.
                if self.data is not None:
                    with self._lock:
                        self.data.ctrl[self.resolver.motor_ctrl] = 0.0
        else:
            self.timeout_detected = False
