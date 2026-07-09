'''Walking policy ROS 2 node.

Runs the TorchScript walking policy from policies/walk/walkingPolicy.pt against
live robot state on /lowstate and publishes lower-body PD setpoints on
safety/lowcmd_lower_in for the h12_safety_layer to merge with upper-body
commands.

Kept as a standalone single-policy node; the generic, switchable controller is
lowerbody_controller_node (see policy.py / policy_manager.py). Direct adaptation
of reference/deploy_mujoco.py to a real-robot ROS 2 stack.
'''

from pathlib import Path

import numpy as np
import rclpy
import torch
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from std_srvs.srv import Trigger
from unitree_hg.msg import LowCmd, LowState


GAIT_PERIOD = 0.8
NUM_LOWER_JOINTS = 12
MOTOR_MODE_PR = 1


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def _default_example_path(filename: str) -> str:
    pkg_share = get_package_share_directory('h12_lowerbody_rl')
    return str(Path(pkg_share) / 'policies' / 'walk' / filename)


class WalkingNode(Node):
    def __init__(self):
        super().__init__('walking_node')

        self.declare_parameter('config_path', _default_example_path('walk.yaml'))
        self.declare_parameter('policy_path', _default_example_path('walkingPolicy.pt'))
        self.declare_parameter('control_hz', 50.0)

        config_path = self.get_parameter('config_path').get_parameter_value().string_value
        policy_path = self.get_parameter('policy_path').get_parameter_value().string_value
        control_hz = self.get_parameter('control_hz').get_parameter_value().double_value

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self._kps = np.array(config['kps'], dtype=np.float32)
        self._kds = np.array(config['kds'], dtype=np.float32)
        self._default_angles = np.array(config['default_angles'], dtype=np.float32)
        self._ang_vel_scale = float(config['ang_vel_scale'])
        self._dof_pos_scale = float(config['dof_pos_scale'])
        self._dof_vel_scale = float(config['dof_vel_scale'])
        self._action_scale = float(config['action_scale'])
        self._cmd_scale = np.array(config['cmd_scale'], dtype=np.float32)
        self._num_actions = int(config['num_actions'])
        self._num_obs = int(config['num_obs'])

        self._policy = torch.jit.load(policy_path)
        self._policy.eval()

        self._cmd = np.array(config['cmd_init'], dtype=np.float32)
        self._action = np.zeros(self._num_actions, dtype=np.float32)
        self._target_dof_pos = self._default_angles.copy()
        self._obs = np.zeros(self._num_obs, dtype=np.float32)

        self._lowstate: LowState | None = None
        self._start_time: Time | None = None

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
        self._cmd_pub = self.create_publisher(LowCmd, '/safety/lowcmd_lower_in', 10)

        self._timer = self.create_timer(1.0 / control_hz, self._tick)

        self.get_logger().info(
            f'walking_node ready: config={config_path}, policy={policy_path}, '
            f'control_hz={control_hz}'
        )

        self._disable_elastic_band_if_present()

    def _disable_elastic_band_if_present(self) -> None:
        cli = self.create_client(Trigger, '/elastic_band/toggle')
        if not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('elastic band toggle service not available — skipping')
            return
        future = cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        result = future.result()
        if result is None:
            self.get_logger().warn('elastic band toggle call timed out')
            return
        self.get_logger().info(f'elastic band toggle: {result.message}')

    def _on_lowstate(self, msg: LowState) -> None:
        self._lowstate = msg
        if self._start_time is None:
            self._start_time = self.get_clock().now()

    def _on_cmd_vel(self, msg: Twist) -> None:
        # Policy was trained with SI commands in roughly [-1, 1] m/s (linear)
        # and [-1, 1] rad/s (yaw). Clip to keep the policy's observation inside
        # its training distribution.
        self._cmd[0] = msg.linear.x
        self._cmd[1] = msg.linear.y
        self._cmd[2] = msg.angular.z
        np.clip(self._cmd, -1.0, 1.0, out=self._cmd)

    def _tick(self) -> None:
        if self._lowstate is None or self._start_time is None:
            return

        motor_state = self._lowstate.motor_state
        q = np.array(
            [motor_state[i].q for i in range(NUM_LOWER_JOINTS)], dtype=np.float32
        )
        dq = np.array(
            [motor_state[i].dq for i in range(NUM_LOWER_JOINTS)], dtype=np.float32
        )
        quat = np.asarray(self._lowstate.imu_state.quaternion, dtype=np.float32)
        omega = np.asarray(self._lowstate.imu_state.gyroscope, dtype=np.float32)

        qj = (q - self._default_angles) * self._dof_pos_scale
        dqj = dq * self._dof_vel_scale
        gravity_orientation = get_gravity_orientation(quat)
        omega_obs = omega * self._ang_vel_scale

        elapsed = (self.get_clock().now() - self._start_time).nanoseconds * 1e-9
        phase = (elapsed % GAIT_PERIOD) / GAIT_PERIOD
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        n = self._num_actions
        self._obs[0:3] = omega_obs
        self._obs[3:6] = gravity_orientation
        self._obs[6:9] = self._cmd * self._cmd_scale
        self._obs[9 : 9 + n] = qj
        self._obs[9 + n : 9 + 2 * n] = dqj
        self._obs[9 + 2 * n : 9 + 3 * n] = self._action
        self._obs[9 + 3 * n : 9 + 3 * n + 2] = np.array([sin_phase, cos_phase])

        obs_tensor = torch.from_numpy(self._obs).unsqueeze(0)
        with torch.no_grad():
            self._action = self._policy(obs_tensor).detach().numpy().squeeze()
        self._target_dof_pos = self._action * self._action_scale + self._default_angles

        cmd_msg = LowCmd()
        for i in range(NUM_LOWER_JOINTS):
            motor = cmd_msg.motor_cmd[i]
            motor.mode = MOTOR_MODE_PR
            motor.q = float(self._target_dof_pos[i])
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = float(self._kps[i])
            motor.kd = float(self._kds[i])

        self._cmd_pub.publish(cmd_msg)


def main():
    rclpy.init()
    node = WalkingNode()
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
