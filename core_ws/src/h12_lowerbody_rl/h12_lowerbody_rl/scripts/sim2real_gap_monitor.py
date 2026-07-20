"""Read-only sim-vs-real diagnostic for the switchable lower-body RL controller
(walk/FAME/ALMI). Three subscribers span the whole pipeline:

  policy  /safety/lowcmd_lower_in   the active RL policy's raw 12-leg command,
                                     published by lowerbody_controller_node
                                     BEFORE the safety layer touches it
  state   /lowstate                 raw robot feedback (motor q/dq, IMU) --
                                     what the policy actually observes
  final   /lowcmd                   the command actually sent to the
                                     actuators, after safety_node merges/clamps

Cross-referencing them surfaces exactly the kind of gap that a single
environment can hide: control-loop timing jitter/drops (adaptive per-topic
gap detection -- flags any inter-arrival interval that blows past the
topic's own running-baseline period), how much the safety layer alters the
policy's intent (|final - policy| target_q), and how well the real
actuators track what they're told (|measured - final| target_q, near-zero
in an ideal sim, nonzero from real compliance/backlash/latency). Also rolls
up IMU pitch stats.

Run identically in RoboCasa and on the real robot and diff the two reports:

    ros2 run h12_lowerbody_rl sim2real_gap_monitor
    ros2 run h12_lowerbody_rl sim2real_gap_monitor --duration 30 --report-every 5

Waits for the operator's real-robot arming step before it starts measuring:
a service call can't be observed by a third node, so it watches the same
latched /lowerbody/active_policy status topic the controller flips off
"idle" once /lowerbody/confirm_engage has actually taken effect (see
lowerbody_controller_node's engage gate). Data before that point is the
band-held pre-pose hold, not the policy running, so it's discarded rather
than polluting the gap/rate baselines.
"""
import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from unitree_hg.msg import LowCmd, LowState

NUM_LEG_JOINTS = 12
POLICY_TOPIC = "/safety/lowcmd_lower_in"
LOWSTATE_TOPIC = "/lowstate"
LOWCMD_TOPIC = "/lowcmd"
ACTIVE_POLICY_TOPIC = "/lowerbody/active_policy"

GAP_MULTIPLIER = 4.0   # flag an inter-arrival gap this many x the topic's running baseline period
STALE_S = 0.1          # ignore a cross-topic comparison if its reference sample is older than this


def _leg_q(motor_array) -> np.ndarray:
    return np.array([motor_array[i].q for i in range(NUM_LEG_JOINTS)], dtype=np.float64)


def _gravity_x(quat) -> float:
    qw, qx, qy, qz = quat
    return float(2 * (-qz * qx + qw * qy))


class _RateGapTracker:
    """Adaptive per-topic inter-arrival monitor: tracks an EMA of the
    steady-state period and flags/logs any arrival that blows past
    GAP_MULTIPLIER x that baseline -- a real dropped/delayed tick, not just
    normal jitter. summary() reports since the last call, then resets."""

    def __init__(self, name: str, log):
        self.name = name
        self._log = log
        self.last_t = None
        self.ema_dt = None
        self.total_gaps = 0
        self._warmup = 5   # don't flag gaps until the EMA baseline has a few samples to settle on
        self._reset_window()

    def _reset_window(self):
        self.n = 0
        self.dt_sum = 0.0
        self.dt_max = 0.0
        self.window_gaps = 0

    def tick(self, t: float) -> None:
        if self.last_t is None:
            self.last_t = t
            return
        dt = t - self.last_t
        self.last_t = t
        self.n += 1
        self.dt_sum += dt
        self.dt_max = max(self.dt_max, dt)
        if self.ema_dt is None:
            self.ema_dt = dt
            return
        if self._warmup > 0:
            self._warmup -= 1
            self.ema_dt = 0.5 * self.ema_dt + 0.5 * dt
        elif dt < GAP_MULTIPLIER * self.ema_dt:
            self.ema_dt = 0.98 * self.ema_dt + 0.02 * dt
        else:
            self.total_gaps += 1
            self.window_gaps += 1
            self._log(f"[GAP] {self.name}: {dt * 1e3:.1f} ms silence "
                       f"(baseline {self.ema_dt * 1e3:.2f} ms) at t={t:.2f}s")

    def summary(self) -> str:
        if self.n == 0:
            line = f"{self.name}: no samples this window"
        else:
            line = (f"{self.name}: n={self.n} rate~{self.n / self.dt_sum:.1f}Hz "
                     f"dt_mean={1e3 * self.dt_sum / self.n:.2f}ms dt_max={1e3 * self.dt_max:.1f}ms "
                     f"gaps={self.window_gaps} (total={self.total_gaps})")
        self._reset_window()
        return line


class _ErrorAcc:
    """Running mean/rms/max of a per-leg-joint error vector, reset each report."""

    def __init__(self):
        self._reset()

    def _reset(self):
        self.n = 0
        self.sum = np.zeros(NUM_LEG_JOINTS)
        self.sumsq = np.zeros(NUM_LEG_JOINTS)
        self.max = np.zeros(NUM_LEG_JOINTS)

    def add(self, err: np.ndarray) -> None:
        self.n += 1
        self.sum += err
        self.sumsq += err * err
        self.max = np.maximum(self.max, err)

    def summary(self) -> str:
        if self.n == 0:
            line = "n=0"
        else:
            mean = self.sum / self.n
            rms = np.sqrt(self.sumsq / self.n)
            line = (f"n={self.n} mean={np.array2string(mean, precision=4)} "
                    f"rms={np.array2string(rms, precision=4)} "
                    f"max={np.array2string(self.max, precision=4)}")
        self._reset()
        return line


class Sim2RealGapMonitor(Node):
    def __init__(self, report_every: float):
        super().__init__("sim2real_gap_monitor")
        self._t0 = time.monotonic()

        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(LowCmd, POLICY_TOPIC, self._on_policy_cmd, qos)
        self.create_subscription(LowState, LOWSTATE_TOPIC, self._on_lowstate, qos)
        self.create_subscription(LowCmd, LOWCMD_TOPIC, self._on_lowcmd_final, qos)

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL,
                              history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(String, ACTIVE_POLICY_TOPIC, self._on_active_policy, latched)
        self._engaged = False   # gates all data callbacks -- see _on_active_policy

        log = self.get_logger().info
        self._policy_gap = _RateGapTracker(f"policy(ALMI/RL) {POLICY_TOPIC}", log)
        self._lowstate_gap = _RateGapTracker(f"state           {LOWSTATE_TOPIC}", log)
        self._lowcmd_gap = _RateGapTracker(f"final           {LOWCMD_TOPIC}", log)

        self._latest_policy_q = None    # (t, q[12]) most recent policy target_q
        self._latest_state_q = None     # (t, q[12]) most recent measured q
        self._clamp_err = _ErrorAcc()   # |final target_q - policy target_q| -- safety-layer delta
        self._track_err = _ErrorAcc()   # |measured q - final target_q|      -- actuator tracking gap
        self._pitch_deg = []

        self.create_timer(report_every, self._report)
        log(f"listening: policy={POLICY_TOPIC} state={LOWSTATE_TOPIC} final={LOWCMD_TOPIC} "
            f"-- reporting every {report_every:.0f}s, Ctrl+C for a final summary")
        log(f"waiting for {ACTIVE_POLICY_TOPIC} to leave 'idle' (operator confirm_engage) "
            f"before measuring...")

    def _now(self) -> float:
        return time.monotonic() - self._t0

    # -- callbacks ------------------------------------------------------------
    def _on_active_policy(self, msg: String) -> None:
        if not self._engaged and msg.data and msg.data != "idle":
            self._engaged = True
            self.get_logger().info(
                f"engaged: active_policy={msg.data!r} at t={self._now():.2f}s -- "
                f"gap/rate tracking starts now")

    def _on_policy_cmd(self, msg: LowCmd) -> None:
        if not self._engaged:
            return
        t = self._now()
        self._policy_gap.tick(t)
        self._latest_policy_q = (t, _leg_q(msg.motor_cmd))

    def _on_lowstate(self, msg: LowState) -> None:
        if not self._engaged:
            return
        t = self._now()
        self._lowstate_gap.tick(t)
        self._latest_state_q = (t, np.array(
            [msg.motor_state[i].q for i in range(NUM_LEG_JOINTS)], dtype=np.float64))
        gx = _gravity_x(np.asarray(msg.imu_state.quaternion, dtype=np.float64))
        self._pitch_deg.append(np.degrees(np.arcsin(np.clip(gx, -1.0, 1.0))))

    def _on_lowcmd_final(self, msg: LowCmd) -> None:
        if not self._engaged:
            return
        t = self._now()
        self._lowcmd_gap.tick(t)
        final_q = _leg_q(msg.motor_cmd)

        if self._latest_policy_q is not None and (t - self._latest_policy_q[0]) < STALE_S:
            self._clamp_err.add(np.abs(final_q - self._latest_policy_q[1]))
        if self._latest_state_q is not None and (t - self._latest_state_q[0]) < STALE_S:
            self._track_err.add(np.abs(self._latest_state_q[1] - final_q))

    # -- reporting --------------------------------------------------------------
    def _report(self) -> None:
        log = self.get_logger().info
        if not self._engaged:
            log(f"still waiting for {ACTIVE_POLICY_TOPIC} != 'idle' "
                f"(operator hasn't called /lowerbody/confirm_engage yet)")
            return
        log("=" * 70)
        log(f"t={self._now():.1f}s")
        log("  " + self._policy_gap.summary())
        log("  " + self._lowstate_gap.summary())
        log("  " + self._lowcmd_gap.summary())
        log("  safety-layer delta |final - policy| target_q [rad/leg]: "
            + self._clamp_err.summary())
        log("  tracking gap |measured - final| target_q [rad/leg]:     "
            + self._track_err.summary())
        if self._pitch_deg:
            arr = np.asarray(self._pitch_deg)
            log(f"  pitch [deg]: mean={arr.mean():+.3f} std={arr.std():.3f} "
                f"p2p={np.ptp(arr):.3f} n={len(arr)}")
            self._pitch_deg = []


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=0.0,
                         help="stop after this many seconds (0 = run until Ctrl+C)")
    parser.add_argument("--report-every", type=float, default=5.0,
                         help="seconds between summary reports")
    parsed, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)
    node = Sim2RealGapMonitor(parsed.report_every)
    start = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if parsed.duration and (time.monotonic() - start) > parsed.duration:
                break
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except rclpy._rclpy_pybind11.RCLError:
        pass  # context torn down mid-spin by a signal (e.g. SIGTERM) -- shutting down anyway
    finally:
        if rclpy.ok():
            node._report()
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
