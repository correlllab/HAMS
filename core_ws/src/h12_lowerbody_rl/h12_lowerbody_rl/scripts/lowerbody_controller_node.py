"""Lower-body controller node: band-held idle -> service-activated policy -> switch.

Flow
----
1. The robot starts held by the elastic band, **idle** (no policy drives the legs).
2. A policy is *started* by calling its service (one Trigger service per policy,
   auto-created from the registry):
       ros2 service call /lowerbody/start_fame std_srvs/srv/Trigger
       ros2 service call /lowerbody/start_walk std_srvs/srv/Trigger
   The first activation releases the elastic band (gated on frame_task being
   ready, /left_ee_pose) and then engages the policy.
3. Switching between policies is **seamless**: the request only commits once the
   handover gate passes (robot standing still, arms home), with the incoming
   policy reset for a clean warm-up (see policy_manager.py).

Interfaces
----------
srv  /lowerbody/start_<name>   (std_srvs/Trigger)      start/switch to a policy
sub  /lowerbody/set_policy     (std_msgs/String)       same, as a topic (convenience)
sub  /lowstate                 (unitree_hg/LowState)   robot state
sub  /cmd_vel                  (geometry_msgs/Twist)   velocity command (walk)
sub  /lowerbody/squat_cmd      (std_msgs/Float32)      base-height / squat (FAME)
sub  /left_ee_pose             (geometry_msgs/PoseStamped)  frame_task-ready signal
pub  /safety/lowcmd_lower_in   (unitree_hg/LowCmd)     12-joint leg setpoints
pub  /lowerbody/active_policy  (std_msgs/String, latched)  active policy ("idle" when none)
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

from h12_lowerbody_rl.policy import (
    NUM_LEG_JOINTS,
    NUM_POLICY_JOINTS,
    FamePolicy,
    RobotState,
    WalkPolicy,
)
from h12_lowerbody_rl.policy_manager import GateConfig, PolicyManager

MOTOR_MODE_PR = 1


def _share(*parts: str) -> str:
    return os.path.join(get_package_share_directory("h12_lowerbody_rl"), *parts)


class LowerBodyControllerNode(Node):
    def __init__(self):
        super().__init__("lowerbody_controller_node")

        self.declare_parameter("control_hz", 50.0)
        # "none" -> start idle (band-held), wait for a start service. Otherwise
        # auto-activate that policy at startup (releases the band when ready).
        self.declare_parameter("active_policy", "none")
        self.declare_parameter("walk_config", _share("policies", "walk", "walk.yaml"))
        self.declare_parameter("fame_config", _share("policies", "fame", "fame.yaml"))
        self.declare_parameter("default_height_cmd", 1.0)
        self.declare_parameter("disable_elastic_band", True)
        # Release the band only after frame_task_server has finished its open-loop
        # startup (it publishes /left_ee_pose only then); earlier release crashes
        # frame_task and free-falls the robot.
        self.declare_parameter("band_wait_for_frame_task", True)
        self.declare_parameter("band_release_topic", "/left_ee_pose")
        self.declare_parameter("band_max_wait", 30.0)

        control_hz = float(self.get_parameter("control_hz").value)
        startup_policy = str(self.get_parameter("active_policy").value).strip().lower()
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
            policies, gate=GateConfig(), log=lambda m: self.get_logger().info(m)
        )

        self._lowstate: LowState | None = None
        self._cmd = np.zeros(3, dtype=np.float32)

        # band / activation state
        self._band_released = not bool(self.get_parameter("disable_elastic_band").value)
        self._frame_task_ready = not bool(self.get_parameter("band_wait_for_frame_task").value)
        self._band_cli = self.create_client(Trigger, "/elastic_band/toggle")
        self._awaiting_band_release = False  # policy committed, band not yet released
        self._request_time: float | None = None  # when the pending activation was asked

        lowstate_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                  history=HistoryPolicy.KEEP_LAST, depth=1)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(LowState, "/lowstate", self._on_lowstate, lowstate_qos)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(String, "/lowerbody/set_policy", self._on_set_policy, 10)
        self.create_subscription(Float32, "/lowerbody/squat_cmd", self._on_squat_cmd, 10)
        self.create_subscription(
            PoseStamped, self.get_parameter("band_release_topic").value,
            self._on_frame_task_ready, 10)
        self._cmd_pub = self.create_publisher(LowCmd, "/safety/lowcmd_lower_in", 10)
        self._active_pub = self.create_publisher(String, "/lowerbody/active_policy", latched)
        self._publish_active()

        # one Trigger service per policy: /lowerbody/start_<name>
        self._start_srvs = [
            self.create_service(Trigger, f"/lowerbody/start_{name}", self._make_start_handler(name))
            for name in self._manager.names()
        ]

        self.create_timer(1.0 / control_hz, self._tick)
        self.get_logger().info(
            f"lowerbody_controller ready: policies={self._manager.names()}, "
            f"start services=/lowerbody/start_<{{{'|'.join(self._manager.names())}}}>, "
            f"control_hz={control_hz}. Robot is band-held idle until a policy is started."
        )

        if startup_policy and startup_policy != "none":
            self.get_logger().info(f"active_policy={startup_policy!r}: auto-activating at startup")
            self._request_policy(startup_policy)

    # -- request entry points ------------------------------------------------
    def _request_policy(self, name: str) -> tuple[bool, str]:
        ok, msg = self._manager.request(name)
        if ok and self._manager.is_pending() and self._manager.is_idle():
            self._request_time = time.monotonic()  # start band-release timeout
        return ok, msg

    def _make_start_handler(self, name: str):
        def handler(_req, resp):
            ok, msg = self._request_policy(name)
            resp.success = ok
            resp.message = msg
            return resp
        return handler

    def _on_set_policy(self, msg: String) -> None:
        self._request_policy(msg.data.strip().lower())

    # -- callbacks -----------------------------------------------------------
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
        self._frame_task_ready = True

    # -- helpers -------------------------------------------------------------
    def _publish_active(self) -> None:
        self._active_pub.publish(String(data=self._manager.active_name or "idle"))

    def _release_band(self, reason: str) -> None:
        if self._band_released:
            return
        self._band_released = True
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
        return RobotState(q=q, dq=dq, quat=quat, gyro=gyro,
                          cmd=self._cmd.copy(), height_cmd=self._height_cmd, t=t)

    # -- main loop -----------------------------------------------------------
    def _tick(self) -> None:
        if self._lowstate is None:
            return
        state = self._state_from_lowstate(self._lowstate)

        if self._manager.is_pending():
            if self._manager.is_idle():
                # First activation: engage the policy FIRST so it is actively
                # driving the legs while the band still holds the robot. The band
                # is released afterwards (below), once frame_task is ready —
                # releasing before any policy controls the legs drops the robot.
                self._manager.commit(state)   # resets policy -> clean warm-up
                self._publish_active()
                self._awaiting_band_release = not self._band_released
            else:
                # Switch between active policies: gated handover.
                if self._manager.update_switch(state) is not None:
                    self._publish_active()

        # Release the band only after a policy is committed and driving the legs
        # (gated on frame_task being ready, or the max-wait timeout).
        if self._awaiting_band_release and not self._band_released:
            released = False
            if self._frame_task_ready:
                self._release_band("policy active + frame_task ready")
                released = True
            elif self._request_time is not None and \
                    (time.monotonic() - self._request_time) > self.get_parameter("band_max_wait").value:
                self._release_band("policy active + band_max_wait timeout")
                released = True
            if released:
                # Reset the active policy AT band release so its observation
                # history (which filled with band-held states) is cleared and it
                # warms up fresh on free-standing — otherwise FAME's first free
                # actions are computed from stale band-held obs and it topples.
                # (This is exactly what fame_node does, and why it stays up.)
                self._manager.reset_active(state)
                self._awaiting_band_release = False

        if self._manager.is_idle():
            return  # nothing requested yet -> band-held, legs uncommanded

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
