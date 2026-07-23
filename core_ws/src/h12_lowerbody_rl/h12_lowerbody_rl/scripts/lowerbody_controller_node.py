"""Lower-body controller node: band-held idle -> FAME stand -> auto stand/walk.

Flow
----
1. The robot starts held by the elastic band, **idle** (no policy drives the legs).
2. At startup the ``active_policy`` (default ``fame``) is auto-activated: the node
   pre-poses the legs (ramping the target from the measured pose to the policy's
   nominal crouch over ``prepose_ramp_s`` — never a step at full gains), releases
   the elastic band (gated on frame_task being ready, /left_ee_pose), and engages
   FAME (standing). With ``engage_wait_for_confirm``
   (the default) the node HOLDS at the pre-pose crouch and only engages the policy
   after the operator calls the /lowerbody/confirm_engage Trigger service — the
   real-robot arming step. The sim bringup disables it for unattended runs.
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

The stand side of the auto-switch is the startup policy when it can stand:
``active_policy:=almi`` gives almi(stand) <-> walk(locomote), exactly like the
default fame <-> walk. (ALMI can in principle locomote itself, but its
from-stand gait engagement proved unreliable — envelope-tested 2026-07-20 — so
under auto_switch it serves as the stand/manipulation policy and the proven
walk policy handles velocity tracking.) ``auto_switch:=false`` pins the
startup policy outright.

Interfaces
----------
sub  /lowstate                 (unitree_hg/LowState)   robot state
sub  /cmd_vel                  (geometry_msgs/Twist)   velocity command -> policy + walk
sub  /lowerbody/squat_cmd      (std_msgs/Float32)      base-height / squat (FAME)
sub  /left_ee_pose             (geometry_msgs/PoseStamped)  frame_task-ready signal
pub  /safety/lowcmd_lower_in   (unitree_hg/LowCmd)     12-joint leg setpoints
pub  /lowerbody/active_policy  (std_msgs/String, latched)  active policy ("idle" when none)
srv  /lowerbody/confirm_engage (std_srvs/Trigger)      operator go-ahead: engage the
                                                       pre-posed policy (see Flow #2)
"""

import os
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor, SetParametersResult
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger
from unitree_hg.msg import LowCmd, LowState

from h12_lowerbody_rl.policy import (
    NUM_LEG_JOINTS,
    NUM_POLICY_JOINTS,
    AlmiPolicy,
    Almi27Policy,
    FamePolicy,
    LegCommand,
    RobotState,
    WalkPolicy,
    apply_imu_offset,
    imu_offset_quat,
)

# Live-tunable IMU trim parameters (degrees). Declared with a range so
# rqt_reconfigure renders them as sliders; updates apply on the next control
# tick. Only orientation axes exist — no policy observes base position, so
# there is nothing an x/y/z offset could feed into.
IMU_OFFSET_PARAMS = ("imu_offset_roll_deg", "imu_offset_pitch_deg", "imu_offset_yaw_deg")
IMU_OFFSET_RANGE_DEG = 15.0
from h12_lowerbody_rl.policy_manager import GateConfig, PolicyManager

MOTOR_MODE_PR = 1

# Policy registry keys (see LowerBodyControllerNode.__init__).
FAME = "fame"   # RMA balance/stand policy (does not locomote)
WALK = "walk"   # TorchScript walk policy (follows /cmd_vel)
ALMI = "almi"   # ALMI LSTM policy: superb stand (incl. under arm motion), but
                # its from-stand gait engagement is unreliable (LSTM was only
                # ever trained with the command present from episode start) —
                # so under auto_switch it serves as a STAND policy and hands
                # locomotion to walk, exactly like fame does.
ALMI27 = "almi27"  # wrist-inclusive ALMI (27-DoF/77-d obs, trained adversarially
                   # under full arm+wrist motion); same stand role as ALMI.

# Policies that can serve as the auto-switch stand side. The locomotion side
# is always the walk policy (the proven velocity tracker).
STAND_CAPABLE = (FAME, ALMI, ALMI27)

# Pre-pose: before engaging the policy, PD-drive the legs to the incoming policy's
# nominal crouch (band-held) and wait until they settle there. The RMA policy warms
# up from its trained default pose; released from the straight-leg spawn it can't
# recover the transition before the band drops. Only once the legs reach nominal do
# we commit the policy (and, later, release the band).
#
# The pre-pose target is RAMPED (cosine blend over ``prepose_ramp_s``) from the
# measured leg pose captured at the first pre-pose tick to nominal. Stepping
# straight to nominal at full kp jerked the real robot's hanging legs hard enough
# (0.24 rad hip-yaw error x kp 200 -> 4.7 rad/s) to trip the safety layer's dq
# estop (hip yaw limit 0.2 x 23 = 4.6 rad/s) the moment the relay opened.
PREPOSE_TOL = 0.15          # rad; max |q - nominal| across the 12 legs to count as posed
PREPOSE_SETTLE_TICKS = 10   # consecutive in-tolerance ticks required (~0.2s @ 50Hz)
PREPOSE_MAX_TICKS = 400     # ~8s @ 50Hz after the ramp; commit anyway if never settled (safety)


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
        self.declare_parameter("almi_config", _share("policies", "almi", "almi.yaml"))
        self.declare_parameter("almi27_config", _share("policies", "almi27", "almi27.yaml"))
        # IMU mounting/calibration trim (deg): what the raw quaternion reads
        # when the torso is known to be level (e.g. measured hanging plumb).
        # Positive pitch = reads tilted forward, positive roll = tilted right.
        # Applied once in _state_from_lowstate so every policy's gravity obs
        # sees the corrected orientation. DYNAMIC — adjust live during a run
        # via rqt_reconfigure sliders or `ros2 param set` (see the parameter
        # callback below); launch files seed the startup values.
        for pname in IMU_OFFSET_PARAMS:
            self.declare_parameter(pname, 0.0, ParameterDescriptor(
                description="live IMU orientation trim, degrees (raw reading at true level)",
                floating_point_range=[FloatingPointRange(
                    from_value=-IMU_OFFSET_RANGE_DEG, to_value=IMU_OFFSET_RANGE_DEG,
                    step=0.0)]))
        self.declare_parameter("default_height_cmd", 1.0)
        # Auto-switch stand<->walk from ||[vx, vy, wz]||: engage walk above rise,
        # fall back to stand at or below fall. auto_switch:=false pins the robot
        # to active_policy. rise 0.05 so a real velocity command starts walking
        # (small enough to catch nav2's fine-approach commands). fall 0.0 so once
        # walking the robot STAYS in walk for ANY nonzero command and only returns
        # to stand at an exactly-zero command -- it never drops out of walk on
        # small mid-path dips. (The fall check is <=, see _tick, so 0.0 fires at
        # 0.) Both stay <= the handover gate's cmd_eps (0.10) so walk->stand's
        # stop gate can still pass. NOTE: /cmd_vel is latched, so if a publisher
        # stops mid-command the last (nonzero) value holds and the robot keeps
        # walking -- publish an explicit zero to stand.
        self.declare_parameter("auto_switch", True)
        self.declare_parameter("auto_switch_rise_norm", 0.05)
        self.declare_parameter("auto_switch_fall_norm", 0.0)
        # Seconds to ramp the pre-pose target from the measured leg pose to the
        # policy's nominal (cosine blend; see the PREPOSE block above). Starting
        # at the measured pose means zero initial PD error, so full gains apply
        # no step torque. 0 disables the ramp (old step behavior).
        self.declare_parameter("prepose_ramp_s", 2.0)
        # Hold at the pre-pose crouch until an operator confirms engagement via
        # the /lowerbody/confirm_engage Trigger service. Default TRUE — on the
        # real robot the engage is the moment the policy takes authority, so it
        # must be an explicit operator action. The sim bringup passes false to
        # keep unattended runs auto-engaging (the old behavior). While waiting,
        # the pre-pose timeout never auto-commits.
        self.declare_parameter("engage_wait_for_confirm", True)
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
        almi_cfg = self.get_parameter("almi_config").value
        almi27_cfg = self.get_parameter("almi27_config").value
        self._imu_corr = self._build_imu_corr()
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self._height_cmd = float(self.get_parameter("default_height_cmd").value)
        self._auto_switch = bool(self.get_parameter("auto_switch").value)
        self._auto_rise = float(self.get_parameter("auto_switch_rise_norm").value)
        self._auto_fall = float(self.get_parameter("auto_switch_fall_norm").value)
        self._confirm_wait = bool(self.get_parameter("engage_wait_for_confirm").value)
        self._engage_confirmed = False
        self._prepose_ramp_ticks = max(1, int(round(
            float(self.get_parameter("prepose_ramp_s").value) * control_hz)))

        self.get_logger().info("loading lower-body policies...")
        policies = {
            "walk": WalkPolicy(walk_cfg),
            "fame": FamePolicy(fame_cfg),
            "almi": AlmiPolicy(almi_cfg),
            "almi27": Almi27Policy(almi27_cfg),
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
        self._band_release_inflight = False  # a toggle call is dispatched, awaiting result
        self._band_wait_warned = False       # rate-limit the "service not ready" warning
        self._request_time: float | None = None  # when the pending activation was asked
        self._prepose_pass = 0   # consecutive ticks the legs have held at nominal (pre-engage)
        self._prepose_ticks = 0  # total pre-pose ticks so far (ramp phase + settle timeout)
        self._prepose_start_q: np.ndarray | None = None  # measured legs at first pre-pose tick

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
        self._confirm_srv = self.create_service(
            Trigger, "/lowerbody/confirm_engage", self._on_confirm_engage)
        self._publish_active()

        self.create_timer(1.0 / control_hz, self._tick)
        # Auto-switch sides: the startup policy is the stand side when it can
        # stand (fame or almi); walk is always the locomotion side. A startup
        # policy of "walk" gets fame as its stand side (legacy behavior).
        self._stand_policy = startup_policy if startup_policy in STAND_CAPABLE else FAME

        self.get_logger().info(
            f"lowerbody_controller ready: policies={self._manager.names()}, "
            f"auto_switch={self._auto_switch} "
            f"(stand={self._stand_policy!r} <-> walk, rise={self._auto_rise}, fall={self._auto_fall}), "
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

    def _on_confirm_engage(self, _req, resp):
        pending = self._manager.desired_name if self._manager.is_pending() else None
        if self._engage_confirmed:
            resp.success = True
            resp.message = "already confirmed"
            return resp
        self._engage_confirmed = True
        resp.success = True
        resp.message = (f"confirmed — engaging {pending!r} once the pre-pose settles"
                        if pending else "confirmed (no activation pending)")
        self.get_logger().info(f"operator confirm: {resp.message}")
        return resp

    # -- IMU trim ------------------------------------------------------------
    def _build_imu_corr(self):
        r, p, y = (float(self.get_parameter(n).value) for n in IMU_OFFSET_PARAMS)
        return imu_offset_quat(np.radians(r), np.radians(p), np.radians(y))

    def _on_set_parameters(self, params) -> SetParametersResult:
        # Runs on the same single-threaded executor as _tick, so swapping the
        # cached correction here is race-free. Values outside the declared
        # range are already rejected by rclpy before this callback fires.
        touched = [p for p in params if p.name in IMU_OFFSET_PARAMS]
        if touched:
            if any(not np.isfinite(p.value) for p in touched):
                return SetParametersResult(
                    successful=False, reason="IMU trim must be finite")
            # Parameters in `params` are not stored yet — merge them with the
            # current values instead of reading back via get_parameter.
            vals = {n: float(self.get_parameter(n).value) for n in IMU_OFFSET_PARAMS}
            vals.update({p.name: float(p.value) for p in touched})
            self._imu_corr = imu_offset_quat(*(np.radians(vals[n]) for n in IMU_OFFSET_PARAMS))
            self.get_logger().info(
                "IMU trim -> roll %+.3f pitch %+.3f yaw %+.3f deg"
                % tuple(vals[n] for n in IMU_OFFSET_PARAMS))
        return SetParametersResult(successful=True)

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
        # Called every tick while awaiting release. Only mark the band released
        # once the sim's toggle actually succeeds — NOT before the service check.
        # The old code set _band_released=True up front, so if /elastic_band/toggle
        # wasn't discovered yet (a startup race, likelier with the slower non-
        # headless MuJoCo viewer) it gave up permanently and the robot hung from
        # the band forever: it would even switch to walk on /cmd_vel but never
        # move. Now it retries every tick until the service is ready and the call
        # returns success.
        if self._band_released or self._band_release_inflight:
            return
        if not self._band_cli.service_is_ready():
            if not self._band_wait_warned:
                self.get_logger().warn(
                    "elastic band toggle service not ready yet — retrying until it appears")
                self._band_wait_warned = True
            return  # do NOT mark released; retry on a later tick (non-blocking)
        self._band_release_inflight = True
        self.get_logger().info(f"releasing elastic band ({reason})")
        self._band_cli.call_async(Trigger.Request()).add_done_callback(self._on_band_toggled)

    def _on_band_toggled(self, fut) -> None:
        self._band_release_inflight = False
        result = fut.result()
        if result is None:
            self.get_logger().warn("elastic band toggle call failed — will retry")
            return  # leave _band_released False so the next tick retries
        self._band_released = True
        self.get_logger().info(f"elastic band toggle: {result.message}")

    def _state_from_lowstate(self, msg: LowState) -> RobotState:
        q = np.array([msg.motor_state[i].q for i in range(NUM_POLICY_JOINTS)], dtype=np.float32)
        dq = np.array([msg.motor_state[i].dq for i in range(NUM_POLICY_JOINTS)], dtype=np.float32)
        quat = apply_imu_offset(
            np.asarray(msg.imu_state.quaternion, dtype=np.float32), self._imu_corr)
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
            if active == self._stand_policy and cmd_norm > self._auto_rise:
                self._request_policy(WALK, require_stop=False)
            elif active == WALK and cmd_norm <= self._auto_fall:
                # <= (not <) so a fall threshold of 0.0 returns to stand at an
                # exactly-zero command; a norm is never < 0, so strict < would
                # never fire at 0 and the robot would stay in walk forever.
                self._request_policy(self._stand_policy, require_stop=True)

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
                cmd = pending.nominal_command()
                if self._prepose_start_q is None:
                    self._prepose_start_q = state.q[:NUM_LEG_JOINTS].copy()
                if self._prepose_ticks < self._prepose_ramp_ticks:
                    frac = self._prepose_ticks / self._prepose_ramp_ticks
                    alpha = 0.5 * (1.0 - np.cos(np.pi * frac))
                    cmd.target_q = (1.0 - alpha) * self._prepose_start_q + alpha * cmd.target_q
                self._publish_leg(cmd)
                err = float(np.max(np.abs(state.q[:NUM_LEG_JOINTS] - pending.nominal_lower)))
                self._prepose_ticks += 1
                self._prepose_pass = self._prepose_pass + 1 if err < PREPOSE_TOL else 0
                settled = self._prepose_pass >= PREPOSE_SETTLE_TICKS
                timed_out = self._prepose_ticks >= self._prepose_ramp_ticks + PREPOSE_MAX_TICKS
                if not settled and not timed_out:
                    return  # keep posing; don't engage the policy or drop the band yet
                if self._confirm_wait and not self._engage_confirmed:
                    # ARMING GATE: hold the legs at the pre-pose crouch until the
                    # operator explicitly confirms. Applies to the timeout path
                    # too — an unsettled pre-pose must never auto-engage on real.
                    self.get_logger().info(
                        f"legs pre-posed ({'settled' if settled else 'NOT SETTLED (timeout)'}, "
                        f"err={err:.3f} rad) — holding; to engage "
                        f"{self._manager.desired_name!r} run: ros2 service call "
                        "/lowerbody/confirm_engage std_srvs/srv/Trigger",
                        throttle_duration_sec=5.0)
                    return
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
