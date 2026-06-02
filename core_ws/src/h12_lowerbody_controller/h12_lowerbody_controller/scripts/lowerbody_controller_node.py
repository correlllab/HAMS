"""Generic lower-body controller node with seamless policy switching.

Runs one of several lower-body policies (walking, FAME standing/squatting) at a
fixed rate against ``/lowstate`` and publishes 12-joint PD setpoints on
``/safety/lowcmd_lower_in`` for the safety layer to merge with the upper-body IK.

This generalizes the original ``walking_node`` (which was a single hardcoded
policy) into a registry + ``PolicyManager`` that switches policies only when the
robot is standing still with arms home (see policy_manager.py).

Interfaces
----------
sub  /lowstate                 (unitree_hg/LowState)   robot state
sub  /cmd_vel                  (geometry_msgs/Twist)   velocity command
sub  /lowerbody/set_policy     (std_msgs/String)       request a policy by name
sub  /lowerbody/squat_cmd      (std_msgs/Float32)      base-height / squat command (FAME)
pub  /safety/lowcmd_lower_in   (unitree_hg/LowCmd)     12-joint leg setpoints
pub  /lowerbody/active_policy  (std_msgs/String, latched)  current active policy
"""

import os
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger
from unitree_hg.msg import LowCmd, LowState

from h12_lowerbody_controller.policy import (
    NUM_LEG_JOINTS,
    NUM_POLICY_JOINTS,
    FamePolicy,
    RobotState,
    WalkPolicy,
)
from h12_lowerbody_controller.policy_manager import GateConfig, PolicyManager

MOTOR_MODE_PR = 1


def _share(*parts: str) -> str:
    return os.path.join(get_package_share_directory("h12_lowerbody_controller"), *parts)


class LowerBodyControllerNode(Node):
    def __init__(self):
        super().__init__("lowerbody_controller_node")

        self.declare_parameter("control_hz", 50.0)
        self.declare_parameter("active_policy", "fame")
        self.declare_parameter("walk_config", _share("policies", "walk", "walk.yaml"))
        self.declare_parameter("fame_config", _share("policies", "fame", "fame.yaml"))
        self.declare_parameter("default_height_cmd", 1.0)
        self.declare_parameter("disable_elastic_band", True)
        # Hold the band until frame_task_server has FINISHED its open-loop
        # startup routine (Moving home/torso, which assumes a stable base) — it
        # publishes /left_ee_pose only after that completes. Releasing earlier
        # crashes frame_task (base moves mid-init -> e-stop) and free-falls the
        # robot before the control chain is live.
        self.declare_parameter("band_wait_for_frame_task", True)
        self.declare_parameter("band_release_topic", "/left_ee_pose")
        self.declare_parameter("band_max_wait", 30.0)

        control_hz = self.get_parameter("control_hz").value
        active = self.get_parameter("active_policy").value
        walk_cfg = self.get_parameter("walk_config").value
        fame_cfg = self.get_parameter("fame_config").value
        self._height_cmd = float(self.get_parameter("default_height_cmd").value)

        self.get_logger().info("loading lower-body policies...")
        policies = {
            "walk": WalkPolicy(walk_cfg),
            "fame": FamePolicy(fame_cfg),
        }
        if not policies["fame"].has_encoder:
            self.get_logger().warn(
                "FAME encoder not loaded — z_t will be zeros (out-of-distribution). "
                "Check policies/fame/encoder_3800.pt."
            )
        self._manager = PolicyManager(
            policies, active_name=active, gate=GateConfig(),
            log=lambda m: self.get_logger().info(m),
        )

        self._lowstate: LowState | None = None
        self._cmd = np.zeros(3, dtype=np.float32)

        lowstate_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(LowState, "/lowstate", self._on_lowstate, lowstate_qos)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(String, "/lowerbody/set_policy", self._on_set_policy, 10)
        self.create_subscription(Float32, "/lowerbody/squat_cmd", self._on_squat_cmd, 10)
        self._cmd_pub = self.create_publisher(LowCmd, "/safety/lowcmd_lower_in", 10)
        self._active_pub = self.create_publisher(String, "/lowerbody/active_policy", latched)
        self._publish_active()

        self.create_timer(1.0 / float(control_hz), self._tick)
        self.get_logger().info(
            f"lowerbody_controller ready: policies={self._manager.names()}, "
            f"active={self._manager.active_name}, control_hz={control_hz}"
        )

        # Defer band release until the control chain is actually driving the
        # robot (see band_wait_for_upper above).
        self._band_released = not self.get_parameter("disable_elastic_band").value
        self._band_cli = self.create_client(Trigger, "/elastic_band/toggle")
        self._band_wait_start = time.monotonic()
        # Set when the band is released so the next tick restarts the active
        # policy's warm-up from the free-standing state (drops band-held history).
        self._pending_policy_reset = False
        if not self._band_released:
            if self.get_parameter("band_wait_for_frame_task").value:
                topic = self.get_parameter("band_release_topic").value
                self.create_subscription(PoseStamped, topic, self._on_frame_task_ready, 10)
                self.get_logger().info(
                    f"waiting for frame_task ready ({topic}) before releasing elastic band"
                )
            else:
                self._release_band("disable_elastic_band (no wait)")

    # -- callbacks -----------------------------------------------------------
    def _on_lowstate(self, msg: LowState) -> None:
        self._lowstate = msg

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._cmd[0] = msg.linear.x
        self._cmd[1] = msg.linear.y
        self._cmd[2] = msg.angular.z
        np.clip(self._cmd, -1.0, 1.0, out=self._cmd)

    def _on_set_policy(self, msg: String) -> None:
        self._manager.request_switch(msg.data.strip())

    def _on_squat_cmd(self, msg: Float32) -> None:
        self._height_cmd = float(msg.data)

    # -- helpers -------------------------------------------------------------
    def _publish_active(self) -> None:
        self._active_pub.publish(String(data=self._manager.active_name))

    def _on_frame_task_ready(self, _msg: PoseStamped) -> None:
        if not self._band_released:
            self._release_band("frame_task ready (ee_pose received)")

    def _release_band(self, reason: str) -> None:
        if self._band_released:
            return
        self._band_released = True
        self._pending_policy_reset = True  # restart policy warm-up free-standing
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

    def _state_from_lowstate(self, msg: LowState) -> RobotState:
        q = np.array([msg.motor_state[i].q for i in range(NUM_POLICY_JOINTS)], dtype=np.float32)
        dq = np.array([msg.motor_state[i].dq for i in range(NUM_POLICY_JOINTS)], dtype=np.float32)
        quat = np.asarray(msg.imu_state.quaternion, dtype=np.float32)
        gyro = np.asarray(msg.imu_state.gyroscope, dtype=np.float32)
        t = self.get_clock().now().nanoseconds * 1e-9
        return RobotState(
            q=q, dq=dq, quat=quat, gyro=gyro,
            cmd=self._cmd.copy(), height_cmd=self._height_cmd, t=t,
        )

    # -- main loop -----------------------------------------------------------
    def _tick(self) -> None:
        # Fallback: release the band even if the upper-body command never shows
        # up, so a misconfigured stack fails loudly rather than hanging held.
        if not self._band_released and \
                (time.monotonic() - self._band_wait_start) > self.get_parameter("band_max_wait").value:
            self.get_logger().warn("upper-body command not seen within band_max_wait — releasing band anyway")
            self._release_band("band_max_wait timeout")

        if self._lowstate is None:
            return
        state = self._state_from_lowstate(self._lowstate)

        if self._pending_policy_reset:
            self._manager.reset_active(state)
            self._pending_policy_reset = False
            self.get_logger().info("reset active policy for fresh free-standing warm-up (band released)")

        if self._manager.update(state) is not None:  # a switch committed this tick
            self._publish_active()

        leg = self._manager.run(state)

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
    node = LowerBodyControllerNode()
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
