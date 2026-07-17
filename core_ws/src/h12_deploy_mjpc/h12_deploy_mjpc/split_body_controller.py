#!/usr/bin/env python3
"""rclpy launcher for the MJPC split-body core PLUS the upper-body pause toggle
(the "split body" controller).

This is a copy of controller_launcher.py: it spawns `mjpc_split_core` (the
fork's OWN thin main, mjpc/deploy/h12_split_controller.cc + deploy_common.cc,
compiled from the submodule) as a SEPARATE process and maps ROS parameters ->
its CLI flags. On top of that it exposes the
`mjpc_deploy/toggle_pause_upperbody` std_srvs/Trigger service (the upper-body
control toggle) and runs the DYNAMIC STARTUP HANDSHAKE (2026-07-17):
frame_task initializes owning the arms, and once the split core is publishing
legs and /frametaskserver/upper_body_paused reads false, this launcher calls
/frametaskserver/toggle_pause_upperbody once and unpauses MJPC on the
response -- exactly one publisher on the safety UPPER channel at every
instant. Because child.wait() blocks the main thread, the node is spun by a
process-wide executor so those callbacks (and the band worker's /clock
polling) are serviced.

The C++ controller must NOT share a process with rclcpp: both rclcpp's
rmw_cyclonedds and unitree_sdk2's bundled CycloneDDS export libddsc.so.0 /
libddscxx.so.0, and one loader namespace resolving both to mismatched builds
corrupts the heap (`free(): invalid pointer` right after ChannelFactory init).
Process isolation is exactly how the native h12 deploy chain runs; this
launcher just adds the ROS-param interface, the pause_upperbody toggle, the sim
elastic-band auto-release, and SIGINT/SIGTERM forwarding (ros2 launch signals
THIS process; the core's handler then does its damping safe-hold + exit summary).
"""

import os
import signal
import subprocess
import sys
import threading
import time

import rclpy
from ament_index_python.packages import get_package_prefix
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger


def _band_worker(node: Node, delay_sec: float, stop: threading.Event) -> None:
    """After delay_sec ON THE NODE CLOCK, call /elastic_band/toggle once.

    With use_sim_time (the sim bringup) the node clock follows /clock, so the
    delay is measured in SIM seconds -- the clock the robot lives on. The
    kitchen sim runs far below 1x real-time, so a wall-clock delay would fire
    at the wrong point of the (plant-clocked) bring-up. Without use_sim_time
    the node clock is system time and this degrades to the old wall behavior.
    The node is spun by the process-wide executor (see main), which services the
    /clock subscription and completes service futures; this worker only polls the
    clock and must NOT spin the node itself (that would double-spin it)."""
    t0 = None
    while not stop.is_set() and rclpy.ok():
        now = node.get_clock().now()
        if t0 is None:
            # with use_sim_time, now() is 0 until the first /clock arrives
            if now.nanoseconds > 0:
                t0 = now
            time.sleep(0.1)
            continue
        if (now - t0) >= Duration(seconds=delay_sec):
            break
        time.sleep(0.1)
    if stop.is_set() or not rclpy.ok():
        return  # controller exited first
    cli = node.create_client(Trigger, "/elastic_band/toggle")
    if not cli.wait_for_service(timeout_sec=2.0):
        node.get_logger().warning(
            "/elastic_band/toggle not available -- band NOT released "
            "(fine on the real robot; set drop_band:=false to silence)")
        return
    fut = cli.call_async(Trigger.Request())
    # the process-wide executor completes the future; just wait for it here
    deadline = time.time() + 5.0
    while not fut.done() and time.time() < deadline and rclpy.ok():
        time.sleep(0.05)
    if fut.done() and fut.result() is not None:
        node.get_logger().info(f"elastic band released: {fut.result().message}")
    else:
        node.get_logger().warning("elastic band toggle call did not complete")


class _UpperPauseBridge:
    """Bridges the ROS toggle_pause_upperbody service to a DDS String the raw-DDS
    core subscribes to ("1" = paused, "0" = active). unitree_sdk2 ships no ros2
    Bool IDL, so the wire type is std_msgs/String. paused=True -> MJPC stops
    publishing the upper channel AND eq-locks the arms to the measured pose in
    its planner (frame_task owns the arms); paused=False -> MJPC drives the arms
    (whole-body). Republished at a low rate so the separately-spawned core
    always has the latest state even if it joins after us or a sample is dropped."""

    def __init__(self, node: Node, dds_topic: str, status_topic: str,
                 initial_paused: bool) -> None:
        self._node = node
        self._paused = bool(initial_paused)
        self._pub = node.create_publisher(String, dds_topic, 10)
        # ROS status: whether MJPC's upper is paused (True = MJPC is NOT driving
        # the arms; the frame_task IK is).
        self._status_pub = node.create_publisher(Bool, status_topic, 10)

    @property
    def paused(self) -> bool:
        return self._paused

    def _publish(self) -> None:
        cmd = String()
        cmd.data = "1" if self._paused else "0"
        self._pub.publish(cmd)
        self._status_pub.publish(Bool(data=self._paused))

    def republish(self) -> None:
        self._publish()

    def set_paused(self, paused: bool, why: str) -> None:
        """Set (not toggle) the pause state and push it to the core at once."""
        if self._paused == bool(paused):
            return
        self._paused = bool(paused)
        state = "PAUSED" if self._paused else "ACTIVE"
        owner = "frame_task owns arms" if self._paused else "mjpc owns arms"
        self._node.get_logger().info(
            f"mjpc upper control -> {state} ({owner}; {why})")
        self._publish()

    def toggle(self, request, response):
        self._paused = not self._paused
        self._publish()
        owner = "frame_task owns arms" if self._paused else "mjpc owns arms"
        state = "PAUSED" if self._paused else "ACTIVE"
        self._node.get_logger().info(
            f"mjpc upper control toggled -> {state} ({owner})")
        response.success = True
        response.message = f"mjpc upper {state.lower()}"
        return response


class _StartupHandshake:
    """Dynamic startup handover: frame_task initializes OWNING the arms (its
    normal init/arm-homing drives them), and once BOTH (a) the split core is
    live (its 200 Hz legs channel is flowing, so an unpause yields upper
    commands on the next tick) and (b) frame_task reports it owns the arms
    (upper_body_paused == False -- its status publisher only starts AFTER the
    blocking homing routine, so this doubles as an init-complete signal), this
    manager calls frame_task's toggle_pause_upperbody ONCE and, on the success
    response, unpauses MJPC.

    Ordering gives exactly-one-publisher at every instant: frame_task's pause
    takes effect inside its service handler BEFORE the response is sent, and
    MJPC is unpaused only ON the response -- so there is never a two-publisher
    overlap, and the no-publisher gap is one service round trip plus at most
    one 5 ms core tick (far under the real safety watchdog's 2 s
    upper_stale_sec; the safety layer latches the last upper command over the
    gap). Waiting for the core's legs channel first means the gap can never
    stretch to the core's multi-second model-load/state-wait.

    frame_task coming up after MJPC is handled for free: MJPC boots paused
    (arms eq-locked to measured, legs balancing) and simply waits for the
    first status message. If the first status says frame_task is ALREADY
    paused (nobody owns the arms -- e.g. a stale start_paused:=true config),
    MJPC claims them directly without toggling. All callbacks run on the one
    executor thread, so the state machine needs no locks; `_done` latches the
    handover so later operator toggles are never fought (a frame_task restart
    that re-takes the arms while MJPC is active is LOGGED loudly, not
    auto-resolved -- pause MJPC first via mjpc_deploy/toggle_pause_upperbody,
    that is the supported handback order)."""

    def __init__(self, node: Node, bridge: _UpperPauseBridge,
                 status_topic: str, toggle_service: str,
                 core_lower_topic: str) -> None:
        self._node = node
        self._bridge = bridge
        self._done = False
        self._in_flight = False
        self._core_ready = False
        self._conflict_logged = False
        self._toggle_cli = node.create_client(Trigger, toggle_service)
        self._toggle_service_name = toggle_service
        self._status_sub = node.create_subscription(
            Bool, status_topic, self._on_frametask_status, 10)
        # Core-liveness gate: the split core's legs channel (DDS
        # rt/safety/lowcmd_lower_in surfaces as this ROS topic -- the same
        # rt/<name> mapping that makes rt/lowstate visible as /lowstate).
        # One message = the core's 200 Hz control loop is emitting commands,
        # so flipping the pause produces upper output within one tick.
        from unitree_hg.msg import LowCmd  # deferred: needs the built workspace
        self._lower_sub = node.create_subscription(
            LowCmd, core_lower_topic, self._on_core_lower, 1)
        self._waiting_log_timer = node.create_timer(10.0, self._log_waiting)

    def _log_waiting(self) -> None:
        if self._done:
            self._waiting_log_timer.cancel()
            return
        self._node.get_logger().info(
            "startup handshake waiting: core %s, frame_task status %s"
            % ("READY" if self._core_ready else "not publishing legs yet",
               "seen" if self._toggle_cli.service_is_ready()
               else "not seen yet (still homing / not up?)"))

    def _on_core_lower(self, _msg) -> None:
        if not self._core_ready:
            self._core_ready = True
            self._node.get_logger().info(
                "split core is publishing the legs channel -- ready to take "
                "the arms once frame_task reports ready")
            # one-shot gate; drop the 200 Hz subscription
            self._node.destroy_subscription(self._lower_sub)

    def _on_frametask_status(self, msg: Bool) -> None:
        frametask_paused = bool(msg.data)
        if self._done:
            # Post-handover conflict watch: frame_task actively owning the
            # arms while MJPC is also active = two publishers on the upper
            # channel. Log once, loudly; never auto-toggle (see class doc).
            if not frametask_paused and not self._bridge.paused:
                if not self._conflict_logged:
                    self._conflict_logged = True
                    self._node.get_logger().error(
                        "BOTH frame_task and MJPC report active on the upper "
                        "channel -- two publishers are fighting. Pause one: "
                        "mjpc_deploy/toggle_pause_upperbody (hand arms to "
                        "frame_task) or %s (hand arms to MJPC)."
                        % self._toggle_service_name)
            else:
                self._conflict_logged = False
            return
        if self._in_flight or not self._core_ready:
            return
        if frametask_paused:
            # frame_task is up but NOT driving the arms: nobody owns them.
            # Claim directly -- no toggle needed, end state is still exactly
            # one publisher.
            self._done = True
            self._bridge.set_paused(
                False, "handshake: frame_task reports paused -> claiming "
                "unowned arms")
            return
        if not self._toggle_cli.service_is_ready():
            return  # discovery pending; retried on the next 100 Hz status msg
        self._in_flight = True
        self._node.get_logger().info(
            "startup handshake: frame_task owns the arms and is ready -- "
            "requesting handover (%s)" % self._toggle_service_name)
        fut = self._toggle_cli.call_async(Trigger.Request())
        fut.add_done_callback(self._on_toggle_response)

    def _on_toggle_response(self, fut) -> None:
        self._in_flight = False
        try:
            resp = fut.result()
        except Exception as err:  # service died mid-call; retry via status
            self._node.get_logger().warning(
                f"handover toggle call failed ({err}); retrying on the next "
                "frame_task status message")
            return
        if not resp.success:
            self._node.get_logger().warning(
                f"frame_task refused the handover ({resp.message}); retrying")
            return
        self._done = True
        self._bridge.set_paused(
            False, f"handshake complete: {resp.message}")
        self._node.get_logger().info(
            "startup handshake DONE: frame_task paused, MJPC owns the arms")


def main() -> None:
    rclpy.init()
    node = Node("mjpc_deploy_splitbody_controller")

    _libdir = os.path.join(get_package_prefix("h12_deploy_mjpc"),
                           "lib", "h12_deploy_mjpc")

    p = node.declare_parameter

    # Upper-body control toggle. The core is a separate raw-DDS process, so the
    # toggle rides a DDS String the core subscribes to: "1" = MJPC stops
    # publishing the upper channel + eq-locks the arms to measured in its
    # planner (frame_task owns the arms), "0" = MJPC drives the arms.
    # rclpy 'mjpc/pause_upperbody' == DDS 'rt/mjpc/pause_upperbody' (matches the
    # core's --pause_upper_topic below). DEFAULT TRUE (2026-07-17, dynamic
    # startup handshake): the system initializes with frame_task owning the
    # arms (it runs its normal init/arm-homing), and the handshake below hands
    # them to MJPC once both sides are ready. The core is told the same boot
    # state via --pause_upper_init so it cannot race the first toggle sample.
    # Registered before the executor spins.
    default_paused = bool(p("default_upper_paused", True).value)
    pause_bridge = _UpperPauseBridge(
        node, "mjpc/pause_upperbody", "mjpc_deploy/upper_body_paused", default_paused)
    node.create_service(
        Trigger, "mjpc_deploy/toggle_pause_upperbody", pause_bridge.toggle)
    node.create_timer(0.2, pause_bridge.republish)  # 5 Hz keep-alive to the core

    # Dynamic startup handshake (2026-07-17): once the core publishes legs AND
    # frame_task reports it owns the arms, pause frame_task via its toggle
    # service, then unpause MJPC on the response. handshake:=false restores the
    # static behavior (whoever default_upper_paused says owns the arms at boot,
    # operator toggles by hand).
    handshake = None
    if bool(p("handshake", True).value):
        if not default_paused:
            node.get_logger().warning(
                "handshake enabled but default_upper_paused is false -- MJPC "
                "would fight frame_task's arm-homing at boot. Forcing the "
                "handshake path: starting paused.")
            default_paused = True
            pause_bridge.set_paused(True, "handshake requires paused boot")
        handshake = _StartupHandshake(
            node, pause_bridge,
            str(p("frametask_status_topic",
                  "frametaskserver/upper_body_paused").value),
            str(p("frametask_toggle_service",
                  "frametaskserver/toggle_pause_upperbody").value),
            str(p("core_lower_topic", "safety/lowcmd_lower_in").value))
    # controller_exe selects which compiled core to run. This launcher defaults to
    # mjpc_split_core (the split-body core, h12_split_controller.cc -- a copy of the
    # legs-only Stabilize lower-body core). Other options: mjpc_lowerbody_core,
    # mjpc_fullbody_core (whole-body Lean). controller_binary (an absolute path)
    # still overrides if set.
    exe = p("controller_exe", "mjpc_split_core").value
    binary = p("controller_binary", os.path.join(_libdir, exe)).value
    # Same knob set + semantics as the fork's CLI (see mjpc/deploy/
    # h12_lower_body_controller.cc); defaults here are the SIM bringup's --
    # config/mjpc_sim.yaml / mjpc_real.yaml override.
    args = [
        binary,
        # WHOLE-BODY task (nu=27) -- the split core actuates legs + torso + arms.
        # The SPLIT task = the Magpie model + 15 inactive upper-joint eq locks;
        # the pause toggle engages them at the measured pose while frame_task
        # owns the arms (plain "Lean H12 Magpie" would only gate the wire).
        "--task", str(p("task", "Lean H12 Magpie Split").value),
        "--strategy", str(p("strategy", 25).value),
        "--gravity_ff", str(p("gravity_ff", 0.85).value),
        # MUST equal 1/lowstate_hz of the plant (RoboCasa 0.002; real 0.001).
        "--twin_dt", str(p("twin_dt", 0.005).value),
        "--sportstate_topic", str(p("sportstate_topic", "rt/sportmodestate_est").value),
        "--imu_pitch_offset_deg", str(p("imu_pitch_offset_deg", 0.0).value),
        "--imu_roll_offset_deg", str(p("imu_roll_offset_deg", 0.0).value),
        "--ankle_roll_offset_l_deg", str(p("ankle_roll_offset_l_deg", 0.0).value),
        "--ankle_roll_offset_r_deg", str(p("ankle_roll_offset_r_deg", 0.0).value),
        "--domain_id", str(p("domain_id",
                             int(os.environ.get("ROS_DOMAIN_ID", "0"))).value),
        # gRPC monitor is compiled out of the container build; the flag is inert
        # there but meaningful on a host build (native monitor attach).
        "--grpc_port", str(p("grpc_port", 0).value),
        "--plan_trajectories", str(p("plan_trajectories", 0).value),
        "--plan_threads", str(p("plan_threads", 0).value),
        "--stale_sec", str(p("stale_sec", 0.05).value),
        # latency-comp sim-time scale = measured RTF. 1.0 (default) is byte-identical
        # to before (real/twin run RTF~1); a below-realtime sim (RoboCasa) passes its
        # measured RTF so the predict-forward horizon stops over-leading by 1/RTF.
        "--latency_rtf", str(p("latency_rtf", 1.0).value),
        # DEBUG plan publish for mjpc_debug_visualizer's blue ghost. Hardcoded, not a
        # ROS param: there is exactly one right topic and the visualizer defaults to
        # the same string, so a param could only ever be set WRONG (silently costing
        # you the ghost). The core still defaults this OFF -- an empty --plan_topic --
        # so the fork's binary, the full-body core and the upper-body core are all
        # byte-unchanged; this launcher is the only thing that arms it.
        # The publish happens on the PLANNER thread after PlanIteration returns, so it
        # costs the 200 Hz control loop nothing. ROS sees it as /mjpc/plan.
        "--plan_topic", "rt/mjpc/plan",
        "--plan_hz", "20.0",
        # Upper-body pause toggle: the core subscribes to this DDS String; the
        # launcher's toggle_pause_upperbody service publishes it (DDS
        # rt/mjpc/pause_upperbody == rclpy /mjpc/pause_upperbody). "1" = MJPC
        # stops publishing the arms and eq-locks them to measured (frame_task
        # owns them); "0" = MJPC drives the arms (whole-body).
        "--pause_upper_topic", "rt/mjpc/pause_upperbody",
        # Boot ownership: the core seeds g_upper_paused from this BEFORE its
        # subscriber can deliver, so it agrees with the bridge state above
        # even if the first keep-alive sample is dropped or late.
        "--pause_upper_init" if default_paused else "--nopause_upper_init",
        #"--straighten_start", "true",
        "--frc_parity", "1"
    ]
    iface = str(p("network_interface", "").value)
    if iface:
        args += ["--network_interface", iface]
    # --arm_aware is a LOWER-BODY-only flag (legs pre-compensate arm motion). The
    # full-body core (mjpc_fullbody_core) owns the arms and hard-codes arm_aware
    # off -- it has no such flag, so passing it would abort on an unknown flag.
    # has_arm_aware:=false (the full-body bench) suppresses it.
    if bool(p("has_arm_aware", True).value):
        args.append("--arm_aware" if bool(p("arm_aware", True).value)
                    else "--noarm_aware")

    drop_band = bool(p("drop_band", True).value)
    band_after = float(p("band_drop_after_secs", 20.0).value)

    if not os.path.isfile(binary):
        node.get_logger().fatal(
            f"controller binary not found at {binary} -- was the package built "
            "in the ros container (mjpc build tree + unitree_sdk2 hydrated)?")
        rclpy.shutdown()
        sys.exit(1)

    # The sourced ROS env puts /opt/ros/humble/lib/... on LD_LIBRARY_PATH, which
    # BEATS the binary's rpath -- so libddsc.so.0 resolves to ROS's CycloneDDS
    # while libddscxx.so.0 stays unitree_sdk2's bundled build: mixed builds
    # corrupt the heap (`free(): invalid pointer` at ChannelFactoryInitialize).
    # Prepend the SDK's own lib dir so BOTH resolve from the same place, exactly
    # as when the binary runs rpath-only outside ROS.
    env = os.environ.copy()
    for cand in ("/opt/unitree_install/lib",
                 os.path.expanduser("~/unitree_install/lib")):
        if os.path.isfile(os.path.join(cand, "libddsc.so.0")):
            env["LD_LIBRARY_PATH"] = cand + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            break

    node.get_logger().info("launching " + " ".join(args))
    child = subprocess.Popen(args, env=env)  # stdout/stderr inherit -> visible in launch log

    # ros2 launch signals THIS process; hand the signal to the core so its
    # damping safe-hold + exit summary run, then exit with its code.
    def _forward(signum, _frame):
        if child.poll() is None:
            child.send_signal(signum)
    signal.signal(signal.SIGINT, _forward)
    signal.signal(signal.SIGTERM, _forward)

    # spin the node for the whole run so the toggle_pause_upperbody service, the
    # 5 Hz pause republish, and the band worker's /clock subscription are
    # serviced. child.wait() blocks the main thread, so the executor gets its own.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(
        target=executor.spin, name="mjpc_deploy_spin", daemon=True)
    spin_thread.start()

    stop = threading.Event()
    band_thread = None
    if drop_band:
        band_thread = threading.Thread(
            target=_band_worker, args=(node, band_after, stop), daemon=True)
        band_thread.start()

    rc = child.wait()
    stop.set()
    if band_thread is not None:
        band_thread.join(timeout=1.0)
    executor.shutdown()
    spin_thread.join(timeout=1.0)
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(rc if rc is not None and rc >= 0 else 1)


if __name__ == "__main__":
    main()
