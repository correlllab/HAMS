"""FAME standing/squatting policy ROS 2 node.

Standalone single-policy node — the ROS adaptation of
``reference/mujoco_deploy_h12_rma.py``, mirroring ``walking_node`` for the
walking policy. Runs the RMA FAME policy against ``/lowstate`` and publishes
12-joint lower-body PD setpoints on ``/safety/lowcmd_lower_in`` for the
h12_safety_layer to merge with upper-body commands.

It controls only the 12 legs; the torso + arms are driven by the upper-body IK
and are merely *observed* here (the encoder adapts the legs to them). Reuses
``FamePolicy`` so the RMA observation/encoder math has a single source of truth;
the generic, switchable controller is ``lowerbody_controller_node``.

Interfaces
----------
sub  /lowstate                 (unitree_hg/LowState)   robot state
sub  /cmd_vel                  (geometry_msgs/Twist)   velocity command
sub  /lowerbody/squat_cmd      (std_msgs/Float32)      base-height / squat command
pub  /safety/lowcmd_lower_in   (unitree_hg/LowCmd)     12-joint leg setpoints
"""

import os
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
from unitree_hg.msg import LowCmd, LowState

from h12_lowerbody_controller.policy import (
    NUM_LEG_JOINTS,
    NUM_POLICY_JOINTS,
    FamePolicy,
    RobotState,
)

MOTOR_MODE_PR = 1


def _default_fame_config() -> str:
    return os.path.join(
        get_package_share_directory("h12_lowerbody_controller"),
        "policies", "fame", "fame.yaml",
    )


class FameNode(Node):
    def __init__(self):
        super().__init__("fame_node")

        self.declare_parameter("config_path", _default_fame_config())
        self.declare_parameter("control_hz", 50.0)
        self.declare_parameter("default_height_cmd", 1.0)
        self.declare_parameter("disable_elastic_band", True)
        # Release the band only after frame_task_server finishes its open-loop
        # init (it publishes /left_ee_pose only then); releasing earlier crashes
        # frame_task (base moves mid-init) and free-falls the robot.
        self.declare_parameter("band_wait_for_frame_task", True)
        self.declare_parameter("band_release_topic", "/left_ee_pose")
        self.declare_parameter("band_max_wait", 30.0)

        config_path = self.get_parameter("config_path").get_parameter_value().string_value
        control_hz = self.get_parameter("control_hz").get_parameter_value().double_value
        self._height_cmd = float(self.get_parameter("default_height_cmd").value)

        self._policy = FamePolicy(config_path)
        if not self._policy.has_encoder:
            self.get_logger().warn(
                "FAME encoder not loaded — z_t will be zeros (out-of-distribution). "
                "Check the encoder weight referenced by the config."
            )

        self._lowstate: LowState | None = None
        self._cmd = np.zeros(3, dtype=np.float32)

        lowstate_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(LowState, "/lowstate", self._on_lowstate, lowstate_qos)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(Float32, "/lowerbody/squat_cmd", self._on_squat_cmd, 10)
        self._cmd_pub = self.create_publisher(LowCmd, "/safety/lowcmd_lower_in", 10)

        self.create_timer(1.0 / float(control_hz), self._tick)
        self.get_logger().info(
            f"fame_node ready: config={config_path}, control_hz={control_hz}, "
            f"encoder={'on' if self._policy.has_encoder else 'OFF'}"
        )

        self._band_released = not self.get_parameter("disable_elastic_band").value
        self._band_cli = self.create_client(Trigger, "/elastic_band/toggle")
        self._band_wait_start = time.monotonic()
        self._pending_policy_reset = False  # reset FAME warm-up free-standing at band release
        if not self._band_released:
            if self.get_parameter("band_wait_for_frame_task").value:
                topic = self.get_parameter("band_release_topic").value
                self.create_subscription(PoseStamped, topic, self._on_frame_task_ready, 10)
                self.get_logger().info(
                    f"waiting for frame_task ready ({topic}) before releasing elastic band"
                )
            else:
                self._release_band("disable_elastic_band (no wait)")

    def _on_lowstate(self, msg: LowState) -> None:
        self._lowstate = msg

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._cmd[0] = msg.linear.x
        self._cmd[1] = msg.linear.y
        self._cmd[2] = msg.angular.z
        np.clip(self._cmd, -1.0, 1.0, out=self._cmd)

    def _on_squat_cmd(self, msg: Float32) -> None:
        self._height_cmd = float(msg.data)

    def _on_frame_task_ready(self, _msg: PoseStamped) -> None:
        if not self._band_released:
            self._release_band("frame_task ready (ee_pose received)")

    def _release_band(self, reason: str) -> None:
        if self._band_released:
            return
        self._band_released = True
        self._pending_policy_reset = True  # restart FAME warm-up free-standing
        if not self._band_cli.service_is_ready() and not self._band_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("elastic band toggle service unavailable — band NOT released")
            return
        self.get_logger().info(f"releasing elastic band ({reason})")
        fut = self._band_cli.call_async(Trigger.Request())
        fut.add_done_callback(
            lambda f: self.get_logger().info(
                f"elastic band toggle: {f.result().message}" if f.result() else "elastic band toggle call failed"
            )
        )

    def _tick(self) -> None:
        if not self._band_released and \
                (time.monotonic() - self._band_wait_start) > self.get_parameter("band_max_wait").value:
            self.get_logger().warn("upper-body command not seen within band_max_wait — releasing band anyway")
            self._release_band("band_max_wait timeout")

        if self._lowstate is None:
            return
        msg = self._lowstate
        state = RobotState(
            q=np.array([msg.motor_state[i].q for i in range(NUM_POLICY_JOINTS)], dtype=np.float32),
            dq=np.array([msg.motor_state[i].dq for i in range(NUM_POLICY_JOINTS)], dtype=np.float32),
            quat=np.asarray(msg.imu_state.quaternion, dtype=np.float32),
            gyro=np.asarray(msg.imu_state.gyroscope, dtype=np.float32),
            cmd=self._cmd.copy(),
            height_cmd=self._height_cmd,
            t=self.get_clock().now().nanoseconds * 1e-9,
        )

        if self._pending_policy_reset:
            self._policy.reset(state)
            self._pending_policy_reset = False
            self.get_logger().info("reset FAME for fresh free-standing warm-up (band released)")

        leg = self._policy.compute(state)

        cmd_msg = LowCmd()
        for i in range(NUM_LEG_JOINTS):
            m = cmd_msg.motor_cmd[i]
            m.mode = MOTOR_MODE_PR
            m.q = float(leg.target_q[i])
            m.dq = 0.0
            m.tau = 0.0
            m.kp = float(leg.kp[i])
            m.kd = float(leg.kd[i])
        self._cmd_pub.publish(cmd_msg)


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


if __name__ == "__main__":
    main()
