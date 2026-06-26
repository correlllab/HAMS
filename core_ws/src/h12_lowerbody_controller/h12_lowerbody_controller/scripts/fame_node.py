'''FAME standing/squatting policy ROS 2 node.

Runs the RMA FAME policy from policies/fame/policy_3600.pt (env-factor encoder +
history-conditioned actor) against live robot state on /lowstate and publishes
lower-body PD setpoints on safety/lowcmd_lower_in for the h12_safety_layer to
merge with upper-body commands.

It controls only the 12 legs; the torso + arms are driven by the upper-body IK
and are merely *observed* here (the encoder adapts the legs to them). Direct
adaptation of reference/mujoco_deploy_h12_rma.py to a real-robot ROS 2 stack,
the FAME sibling of walking_node.py.
'''

import collections
from pathlib import Path

import numpy as np
import rclpy
import torch
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
from unitree_hg.msg import LowCmd, LowState

from h12_lowerbody_controller.rma import EnvFactorEncoder, EnvFactorEncoderCfg


NUM_LEG_JOINTS = 12
NUM_POLICY_JOINTS = 27          # legs(12) + torso(1) + left arm(7) + right arm(7)
MOTOR_MODE_PR = 1

LATENT_DIM = 8                  # z_t dimension
ET_DIM = 21                     # e_t: 15 upper-body dof + left_xyz(3) + right_xyz(3)
Z_HISTORY = 3                   # z_t history depth (matches obs history length)


def quat_rotate_inverse(q, v):
    '''Rotate v by the inverse of quaternion q ([w, x, y, z]) — FAME/RMA convention.'''
    w, x, y, z = q
    q_conj = np.array([w, -x, -y, -z], dtype=np.float32)
    return np.array(
        [
            v[0] * (q_conj[0] ** 2 + q_conj[1] ** 2 - q_conj[2] ** 2 - q_conj[3] ** 2)
            + v[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3])
            + v[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),
            v[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3])
            + v[1] * (q_conj[0] ** 2 - q_conj[1] ** 2 + q_conj[2] ** 2 - q_conj[3] ** 2)
            + v[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),
            v[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2])
            + v[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1])
            + v[2] * (q_conj[0] ** 2 - q_conj[1] ** 2 - q_conj[2] ** 2 + q_conj[3] ** 2),
        ],
        dtype=np.float32,
    )


def get_gravity_orientation(quat):
    '''Projected gravity as used by the FAME policy (matches mujoco_deploy_h12_rma.py).'''
    return quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))


def rotate_about_z(v, angle):
    '''Rotate a 3-vector about the body z-axis by `angle` rad (xy rotate, z unchanged).'''
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]], dtype=np.float32)


def _default_fame_path(filename: str) -> str:
    pkg_share = get_package_share_directory('h12_lowerbody_controller')
    return str(Path(pkg_share) / 'policies' / 'fame' / filename)


class FameNode(Node):
    def __init__(self):
        super().__init__('fame_node')

        self.declare_parameter('config_path', _default_fame_path('fame.yaml'))
        self.declare_parameter('control_hz', 50.0)
        self.declare_parameter('default_height_cmd', 1.0)
        # The IMU lives in the torso (above the waist-yaw torso_joint), but the
        # policy was trained on pelvis-frame orientation/ang-vel. Correct the
        # torso IMU back to the pelvis frame using the measured waist yaw. Set
        # False to feed the raw torso IMU (A/B the effect, like the old USE_IMU).
        self.declare_parameter('waist_imu_correction', True)
        # Warm start: run FAME band-held for this many control ticks so the
        # zero-seeded obs history flushes and the standing command settles, then
        # drop the elastic band once (keeping the warmed-up state). Set
        # disable_elastic_band False to keep the band on (never drop).
        self.declare_parameter('band_warmup_ticks', 50)
        self.declare_parameter('disable_elastic_band', True)

        config_path = self.get_parameter('config_path').get_parameter_value().string_value
        control_hz = self.get_parameter('control_hz').get_parameter_value().double_value
        self._height_cmd = float(self.get_parameter('default_height_cmd').value)
        self._waist_imu_correction = bool(self.get_parameter('waist_imu_correction').value)
        self._band_warmup_ticks = int(self.get_parameter('band_warmup_ticks').value)
        self._disable_elastic_band = bool(self.get_parameter('disable_elastic_band').value)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        config_dir = Path(config_path).resolve().parent

        def _resolve(key: str, default: str) -> str:
            p = config.get(key, default)
            return p if Path(p).is_absolute() else str(config_dir / p)

        self._kps = np.array(config['kps'], dtype=np.float32)
        self._kds = np.array(config['kds'], dtype=np.float32)
        self._default_angles = np.array(config['default_angles'], dtype=np.float32)          # 12 legs
        self._default_angles_arms = np.array(config['default_angles_arms'], dtype=np.float32)  # 15 arms+torso
        self._ang_vel_scale = float(config['ang_vel_scale'])
        self._dof_pos_scale = float(config['dof_pos_scale'])
        self._dof_vel_scale = float(config['dof_vel_scale'])
        self._action_scale = float(config['action_scale'])
        self._cmd_scale = np.array(config['cmd_scale'], dtype=np.float32)
        self._num_actions = int(config['num_actions'])
        self._num_obs = int(config['num_obs'])
        self._obs_history_len = int(config['obs_history_len'])
        self._left_force = np.array(config.get('left_hand_force', [0.0, 0.0, 0.0]), dtype=np.float32)
        self._right_force = np.array(config.get('right_hand_force', [0.0, 0.0, 0.0]), dtype=np.float32)

        # Padded 27-joint default pose: legs + (torso, arms).
        self._padded_defaults = np.zeros(NUM_POLICY_JOINTS, dtype=np.float32)
        self._padded_defaults[:NUM_LEG_JOINTS] = self._default_angles
        self._padded_defaults[NUM_LEG_JOINTS:] = self._default_angles_arms

        policy_path = _resolve('policy_path', 'policy_3600.pt')
        self._policy = torch.jit.load(policy_path)
        self._policy.eval()

        # Env-factor encoder (e_t -> z_t). Without it z_t stays zeros, which is
        # out of the actor's training distribution.
        self._encoder = None
        encoder_path = _resolve('encoder_path', 'encoder_3600.pt')
        if Path(encoder_path).is_file():
            self._encoder = EnvFactorEncoder(
                EnvFactorEncoderCfg(in_dim=ET_DIM, latent_dim=LATENT_DIM)
            )
            self._encoder.load_state_dict(
                torch.load(encoder_path, map_location='cpu', weights_only=True)
            )
            self._encoder.eval()
        else:
            self.get_logger().warn(
                f'FAME encoder not found at {encoder_path} — z_t will be zeros '
                f'(out-of-distribution).'
            )

        # Single-step proprio obs: cmd(3) + height(1) + omega(3) + gravity(3)
        # + qj(27) + dqj(27) + action(12) = 76.
        self._single_obs_dim = (
            3 + 1 + 3 + 3 + NUM_POLICY_JOINTS + NUM_POLICY_JOINTS + NUM_LEG_JOINTS
        )
        self._cmd = np.zeros(3, dtype=np.float32)
        self._action = np.zeros(self._num_actions, dtype=np.float32)

        # Zero-filled warm-up: start with empty proprio + z_t history and let it
        # fill over the first few control ticks (matches the reference deploy).
        self._obs_history: collections.deque = collections.deque(maxlen=self._obs_history_len)
        for _ in range(self._obs_history_len):
            self._obs_history.append(np.zeros(self._single_obs_dim, dtype=np.float32))
        self._z_history = np.zeros((Z_HISTORY, LATENT_DIM), dtype=np.float32)

        self._lowstate: LowState | None = None

        lowstate_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._lowstate_sub = self.create_subscription(
            LowState, '/lowstate', self._on_lowstate, lowstate_qos
        )
        self._cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self._on_cmd_vel, 10
        )
        self._squat_sub = self.create_subscription(
            Float32, '/lowerbody/squat_cmd', self._on_squat_cmd, 10
        )
        self._cmd_pub = self.create_publisher(LowCmd, '/safety/lowcmd_lower_in', 10)

        self._timer = self.create_timer(1.0 / control_hz, self._tick)

        # Warm-start band handling: the policy starts running on the first
        # /lowstate with the band still holding the robot; the band is dropped
        # only after band_warmup_ticks policy ticks (see _tick / _try_drop_band).
        self._tick_count = 0
        self._band_dropped = not self._disable_elastic_band
        self._band_wait_logged = False
        self._band_cli = self.create_client(Trigger, '/elastic_band/toggle')

        self.get_logger().info(
            f'fame_node ready: config={config_path}, control_hz={control_hz}, '
            f'encoder={"on" if self._encoder is not None else "OFF"}, '
            f'band_warmup_ticks={self._band_warmup_ticks if self._disable_elastic_band else "kept"}'
        )

    def _try_drop_band(self) -> None:
        '''Non-blocking one-shot drop of the elastic band after warm-up.

        Called from the timer callback once the policy has run band-held for
        band_warmup_ticks. Retries each tick until the toggle service is up
        (the policy keeps warming under the band meanwhile); a no-op on stacks
        without the sim band service.
        '''
        if not self._band_cli.service_is_ready():
            if not self._band_wait_logged:
                self.get_logger().info(
                    'warm-up done; waiting for /elastic_band/toggle to drop the band'
                )
                self._band_wait_logged = True
            return
        self._band_dropped = True
        self.get_logger().info(
            f'dropping elastic band after {self._tick_count} warm-up ticks'
        )
        future = self._band_cli.call_async(Trigger.Request())
        future.add_done_callback(
            lambda f: self.get_logger().info(
                f'elastic band toggle: {f.result().message}'
                if f.result() else 'elastic band toggle call failed'
            )
        )

    def _on_lowstate(self, msg: LowState) -> None:
        self._lowstate = msg

    def _on_cmd_vel(self, msg: Twist) -> None:
        # Policy was trained with SI commands in roughly [-1, 1] m/s (linear)
        # and [-1, 1] rad/s (yaw). Clip to keep the policy's observation inside
        # its training distribution.
        self._cmd[0] = msg.linear.x
        self._cmd[1] = msg.linear.y
        self._cmd[2] = msg.angular.z
        np.clip(self._cmd, -1.0, 1.0, out=self._cmd)

    def _on_squat_cmd(self, msg: Float32) -> None:
        self._height_cmd = float(msg.data)

    def _encode(self, q: np.ndarray) -> np.ndarray:
        if self._encoder is None:
            return np.zeros(LATENT_DIM, dtype=np.float32)
        # e_t = 15 upper-body joint positions (raw) + left/right wrist forces.
        # No wrist F/T in sim -> configured (zero) forces; the encoder still
        # adapts the legs to the upper-body configuration the IK produces.
        upper = q[NUM_LEG_JOINTS:NUM_POLICY_JOINTS]
        e_t = np.concatenate([upper, self._left_force, self._right_force]).astype(np.float32)
        with torch.no_grad():
            return self._encoder(torch.from_numpy(e_t).unsqueeze(0)).numpy().squeeze()

    def _tick(self) -> None:
        if self._lowstate is None:
            return

        motor_state = self._lowstate.motor_state
        q = np.array(
            [motor_state[i].q for i in range(NUM_POLICY_JOINTS)], dtype=np.float32
        )
        dq = np.array(
            [motor_state[i].dq for i in range(NUM_POLICY_JOINTS)], dtype=np.float32
        )
        quat = np.asarray(self._lowstate.imu_state.quaternion, dtype=np.float32)
        omega = np.asarray(self._lowstate.imu_state.gyroscope, dtype=np.float32)

        qj = (q - self._padded_defaults) * self._dof_pos_scale
        dqj = dq * self._dof_vel_scale

        gravity_orientation = get_gravity_orientation(quat)
        omega_pelvis = omega
        if self._waist_imu_correction:
            # Rotate the torso-frame IMU readings into the pelvis frame by the
            # measured waist yaw (R_z(psi)), and remove the waist's own yaw rate
            # from the gyro. Exact for this single-DoF (yaw) waist, so the policy
            # sees the pelvis-frame gravity/ang-vel it was trained on.
            psi = float(q[NUM_LEG_JOINTS])       # torso_joint: pelvis->torso yaw
            psi_dot = float(dq[NUM_LEG_JOINTS])
            gravity_orientation = rotate_about_z(gravity_orientation, psi)
            omega_pelvis = rotate_about_z(omega, psi)
            omega_pelvis[2] -= psi_dot
        omega_obs = omega_pelvis * self._ang_vel_scale

        nj = NUM_POLICY_JOINTS
        single_obs = np.zeros(self._single_obs_dim, dtype=np.float32)
        single_obs[0:3] = self._cmd * self._cmd_scale
        single_obs[3:4] = self._height_cmd
        single_obs[4:7] = omega_obs
        single_obs[7:10] = gravity_orientation
        single_obs[10 : 10 + nj] = qj
        single_obs[10 + nj : 10 + 2 * nj] = dqj
        single_obs[10 + 2 * nj : 10 + 2 * nj + NUM_LEG_JOINTS] = self._action
        self._obs_history.append(single_obs)

        z_t = self._encode(q)
        self._z_history[1:, :] = self._z_history[:-1, :]
        self._z_history[0, :] = z_t
        z_flat = np.flip(self._z_history, axis=0).flatten().astype(np.float32)

        proprio = np.concatenate(list(self._obs_history), axis=0)
        actor_obs = np.concatenate([proprio, z_flat], axis=0).astype(np.float32)
        if actor_obs.shape[0] != self._num_obs:
            raise ValueError(
                f'actor_obs dim {actor_obs.shape[0]} != num_obs {self._num_obs}'
            )

        obs_tensor = torch.from_numpy(actor_obs).unsqueeze(0)
        with torch.no_grad():
            self._action = self._policy(obs_tensor).detach().numpy().squeeze()
        target_dof_pos = self._action * self._action_scale + self._default_angles

        cmd_msg = LowCmd()
        for i in range(NUM_LEG_JOINTS):
            motor = cmd_msg.motor_cmd[i]
            motor.mode = MOTOR_MODE_PR
            motor.q = float(target_dof_pos[i])
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = float(self._kps[i])
            motor.kd = float(self._kds[i])

        self._cmd_pub.publish(cmd_msg)

        # Warm start: count policy ticks run under the band, then drop it once.
        self._tick_count += 1
        if not self._band_dropped and self._tick_count >= self._band_warmup_ticks:
            self._try_drop_band()


def main():
    rclpy.init()
    node = FameNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
