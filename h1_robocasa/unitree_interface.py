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

TOPIC_LOWCMD = 'rt/lowcmd'
TOPIC_LOWSTATE = 'rt/lowstate'
# 27 motors on main body, 3 sensors on each motor
MOTOR_NUM = 27
MOTOR_SENSOR_NUM = 3

DOMAIN_ID = int(os.getenv("ROS_DOMAIN_ID"))
assert DOMAIN_ID > 0 and isinstance(DOMAIN_ID, int), "Please set ROS_DOMAIN_ID environment variable to a positive value, e.g. export ROS_DOMAIN_ID=1, domain 0 is reserved for the real robot."
class SimInterface:
    def __init__(self, model, data, lock=None, resolver=None):
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

        # Latched last low_cmd. The PD torque is recomputed EVERY sim step from
        # these + the CURRENT state (write_ctrl, called from the sim loop), which
        # emulates the real motor's ~1kHz servo. Computing it only in
        # low_cmd_handler (50Hz) holds a stale torque for ~20ms, so the kd damping
        # term lags the true velocity and FAME oscillates then topples ~2.8s after
        # free-standing -- while the in-loop (per-step PD) stands indefinitely.
        self._cmd_mode = np.zeros(MOTOR_NUM, dtype=np.int32)
        self._cmd_q = np.zeros(MOTOR_NUM, dtype=np.float64)
        self._cmd_dq = np.zeros(MOTOR_NUM, dtype=np.float64)
        self._cmd_kp = np.zeros(MOTOR_NUM, dtype=np.float64)
        self._cmd_kd = np.zeros(MOTOR_NUM, dtype=np.float64)
        self._cmd_tau = np.zeros(MOTOR_NUM, dtype=np.float64)

        # initialize channel
        ChannelFactoryInitialize(id=DOMAIN_ID)
        # publish low state
        self.low_state = LowState_default()
        self.low_state_publisher = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self.low_state_publisher.Init()
        self.low_state_thread = RecurrentThread(interval=self.dt, target=self.publish_low_state, name='sim_lowstate')
        self.low_state_thread.Start()

        # subscribe to low command
        self.low_cmd_subscriber = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self.low_cmd_subscriber.Init(self.low_cmd_handler, 10)

        # Timeout detection. When no controller command has been received
        # for `timeout` seconds, write zero ctrl on every motor and let the
        # robot go limp. Without an upright tether, the H1-2 falls and lays
        # on the floor — preferable to a stiff pose-hold snapshot, which
        # tries to freeze whatever chaotic mid-fall pose was sampled at the
        # moment of timeout and tends to NaN against contact dynamics.
        self.last_cmd_time = time.time()
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
                for k in range(4):
                    self.low_state.imu_state.quaternion[k] = float(sd[qa + k])
                for k in range(3):
                    self.low_state.imu_state.gyroscope[k] = float(sd[ga + k])
                for k in range(3):
                    self.low_state.imu_state.accelerometer[k] = float(sd[aa + k])
        # write to low state (DDS publish is thread-safe, no lock needed)
        self.low_state_publisher.Write(self.low_state)

    def low_cmd_handler(self, msg: LowCmd_):
        if self.data is None:
            return
        # Latch the command only (no data touch, no lock). write_ctrl turns it into
        # a torque every sim step against the CURRENT state.
        self.last_cmd_time = time.time()
        for i in range(self.num_motor):
            mc = msg.motor_cmd[i]
            self._cmd_mode[i] = mc.mode
            self._cmd_q[i] = mc.q
            self._cmd_dq[i] = mc.dq
            self._cmd_kp[i] = mc.kp
            self._cmd_kd[i] = mc.kd
            self._cmd_tau[i] = mc.tau

    def write_ctrl(self):
        """Per-sim-step PD: tau + kp*(q*-q) + kd*(dq*-dq) from the latched command
        and the CURRENT q/dq. Call every step (caller holds the sim lock) so the
        damping tracks the true velocity instead of a 20ms-stale one."""
        if self.data is None:
            return
        r = self.resolver
        if self.timeout_detected:
            self.data.ctrl[r.motor_ctrl] = 0.0
            return
        for i in range(self.num_motor):
            ci = int(r.motor_ctrl[i])
            if self._cmd_mode[i] == 1:
                q_cur = self.data.qpos[r.motor_qpos[i]]
                dq_cur = self.data.qvel[r.motor_qvel[i]]
                self.data.ctrl[ci] = (self._cmd_tau[i]
                                      + self._cmd_kp[i] * (self._cmd_q[i] - q_cur)
                                      + self._cmd_kd[i] * (self._cmd_dq[i] - dq_cur))
            else:
                self.data.ctrl[ci] = 0.0

    def check_cmd_timeout(self):
        current_time = time.time()
        if (current_time - self.last_cmd_time) > self.timeout:
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
