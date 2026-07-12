"""Lower-body controller node: band-held idle -> FAME stand -> auto stand/walk.

Flow
----
1. The robot starts held by the elastic band, **idle** (no policy drives the legs).
2. At startup the ``active_policy`` (default ``fame``) is auto-activated: the node
   pre-poses the legs, releases the elastic band (gated on frame_task being ready,
   /left_ee_pose), and engages FAME (standing).
3. From then on the policy is chosen **automatically from /cmd_vel** (auto_switch,
   on by default): ||[vx, vy, wz]|| above ``auto_switch_rise_norm`` -> walk
   (locomote); below ``auto_switch_fall_norm`` -> FAME (stand still). The two
   thresholds give hysteresis so it doesn't chatter at the boundary. Each switch
   commits through the gated handover (policy_manager.py): walk->FAME waits for a
   real stop + settle; FAME->walk engages as soon as the base is upright and still,
   so it can start under a held velocity command.

There are no per-policy start services — drive /cmd_vel (e.g. from nav2) and the
controller stands or walks to match. Set ``auto_switch:=false`` to pin the robot to
``active_policy`` (pure stand or pure walk).

Interfaces
----------
sub  /lowstate                 (unitree_hg/LowState)   robot state
sub  /cmd_vel                  (geometry_msgs/Twist)   velocity command -> policy + walk
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
    LegCommand,
    RobotState,
    WalkPolicy,
)
from h12_lowerbody_rl.policy_manager import GateConfig, PolicyManager

MOTOR_MODE_PR = 1

# Policy registry keys (see LowerBodyControllerNode.__init__).
FAME = "fame"   # RMA balance/stand policy (does not locomote)
WALK = "walk"   # TorchScript walk policy (follows /cmd_vel)

# Pre-pose: before engaging the policy, PD-drive the legs to the incoming policy's
# nominal crouch (band-held) and wait until they settle there. The RMA policy warms
# up from its trained default pose; released from the straight-leg spawn it can't
# recover the transition before the band drops. Only once the legs reach nominal do
# we commit the policy (and, later, release the band).
PREPOSE_TOL = 0.15          # rad; max |q - nominal| across the 12 legs to count as posed
PREPOSE_SETTLE_TICKS = 10   # consecutive in-tolerance ticks required (~0.2s @ 50Hz)
PREPOSE_MAX_TICKS = 400     # ~8s @ 50Hz; commit anyway if the legs never settle (safety)


def _share(*parts: str) -> str:
    return os.path.join(get_package_share_directory("h12_lowerbody_rl"), *parts)


class LowerBodyControllerNode(Node):
    def __init__(self):
        super().__init__("lowerbody_controller_node")

        self.declare_parameter("control_hz", 50.0)
        # Policy to auto-activate at startup (releases the band when ready).
        # Default "fame" (stand). "none" stays band-held idle -- but with the start
        # services removed, "none" now leaves the robot idle forever (auto_switch
        # only switches BETWEEN active policies), so set a real policy here.
        self.declare_parameter("active_policy", "fame")
        self.declare_parameter("walk_config", _share("policies", "walk", "walk.yaml"))
        self.declare_parameter("fame_config", _share("policies", "fame", "fame.yaml"))
        self.declare_parameter("default_height_cmd", 1.0)
        # Auto-switch stand<->walk from ||[vx, vy, wz]|| with hysteresis (rise > fall):
        # engage walk above rise, fall back to FAME below fall. auto_switch:=false
        # pins the robot to active_policy. rise sits at the handover gate's cmd_eps
        # (0.10); fall below it so walk->FAME's stop gate can also pass.
        self.declare_parameter("auto_switch", True)
        self.declare_parameter("auto_switch_rise_norm", 0.10)
        self.declare_parameter("auto_switch_fall_norm", 0.05)
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
        self._auto_switch = bool(self.get_parameter("auto_switch").value)
        self._auto_rise = float(self.get_parameter("auto_switch_rise_norm").value)
        self._auto_fall = float(self.get_parameter("auto_switch_fall_norm").value)

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
        self._prepose_pass = 0   # consecutive ticks the legs have held at nominal (pre-engage)
        self._prepose_ticks = 0  # total pre-pose ticks so far (for the settle timeout)

        lowstate_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                  history=HistoryPolicy.KEEP_LAST, depth=1)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(LowState, "/lowstate", self._on_lowstate, lowstate_qos)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(Float32, "/lowerbody/squat_cmd", self._on_squat_cmd, 10)
        self.create_subscription(
            PoseStamped, self.get_parameter("band_release_topic").value,
            self._on_frame_task_ready, 10)
        self._cmd_pub = self.create_publisher(LowCmd, "/safety/lowcmd_lower_in", 10)
        self._active_pub = self.create_publisher(String, "/lowerbody/active_policy", latched)
        self._publish_active()

        self.create_timer(1.0 / control_hz, self._tick)
        self.get_logger().info(
            f"lowerbody_controller ready: policies={self._manager.names()}, "
            f"auto_switch={self._auto_switch} (rise={self._auto_rise}, fall={self._auto_fall}), "
            f"control_hz={control_hz}. Policy follows /cmd_vel; no start services."
        )

        if startup_policy and startup_policy != "none":
            self.get_logger().info(f"active_policy={startup_policy!r}: auto-activating at startup")
            self._request_policy(startup_policy)
        else:
            self.get_logger().warn(
                "active_policy='none' with start services removed -> robot stays "
                "band-held idle forever. Set active_policy to 'fame' or 'walk'."
            )

    # -- request entry points ------------------------------------------------
    def _request_policy(self, name: str, require_stop: bool = True) -> tuple[bool, str]:
        ok, msg = self._manager.request(name, require_stop=require_stop)
        if ok and self._manager.is_pending() and self._manager.is_idle():
            self._request_time = time.monotonic()  # start band-release timeout
        return ok, msg

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

    def _publish_leg(self, leg: LegCommand) -> None:
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

        # Auto-switch stand<->walk from the velocity command. Decide only when
        # settled on a policy (not idle, not mid-handover); the request then commits
        # through the gated handover below. fame->walk uses a relaxed gate
        # (require_stop=False) so it engages under a held cmd; walk->FAME uses the
        # full stop-and-settle gate. request() is idempotent while pending, so
        # re-deciding each tick doesn't reset the gate.
        if self._auto_switch and not self._manager.is_idle() and not self._manager.is_pending():
            cmd_norm = float(np.linalg.norm(self._cmd))
            active = self._manager.active_name
            if active == FAME and cmd_norm > self._auto_rise:
                self._request_policy(WALK, require_stop=False)
            elif active == WALK and cmd_norm < self._auto_fall:
                self._request_policy(FAME, require_stop=True)

        if self._manager.is_pending():
            if self._manager.is_idle():
                # Pre-pose: PD-drive the legs to the incoming policy's nominal
                # crouch (band-held) and wait until they settle BEFORE engaging it.
                # The RMA policy warms up from its trained default pose; from the
                # straight-leg spawn it can't recover the transition before the band
                # drops. The band stays held throughout — the release below is gated
                # on _awaiting_band_release, which we only set once committed.
                pending = self._manager.desired_policy()
                if pending is None:
                    return
                self._publish_leg(pending.nominal_command())
                err = float(np.max(np.abs(state.q[:NUM_LEG_JOINTS] - pending.nominal_lower)))
                self._prepose_ticks += 1
                self._prepose_pass = self._prepose_pass + 1 if err < PREPOSE_TOL else 0
                settled = self._prepose_pass >= PREPOSE_SETTLE_TICKS
                timed_out = self._prepose_ticks >= PREPOSE_MAX_TICKS
                if not settled and not timed_out:
                    return  # keep posing; don't engage the policy or drop the band yet
                self.get_logger().info(
                    f"legs pre-posed to nominal ({'settled' if settled else 'timeout'}, "
                    f"err={err:.3f} rad) — engaging {self._manager.desired_name!r}")
                # Now engage the policy; it drives the legs while the band still holds.
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

        self._publish_leg(self._manager.run(state))


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
