#!/usr/bin/env python3
"""rclpy parameter launcher for the fork's MJPC lower-body controller binary.

Runs `mjpc_lowerbody_core` (the fork's OWN thin main, mjpc/deploy/
h12_lower_body_controller.cc + deploy_common.cc, compiled from the submodule)
as a SEPARATE process, mapping ROS parameters -> its CLI flags. The C++
controller must NOT share a process with rclcpp: both rclcpp's
rmw_cyclonedds and unitree_sdk2's bundled CycloneDDS export libddsc.so.0 /
libddscxx.so.0, and one loader namespace resolving both to mismatched builds
corrupts the heap (`free(): invalid pointer` right after ChannelFactory init).
Process isolation is exactly how the native h12 deploy chain runs; this
launcher just adds the ROS-param interface, the sim elastic-band auto-release,
and SIGINT/SIGTERM forwarding (ros2 launch signals THIS process; the core's
handler then does its damping safe-hold + exit summary).
"""

import os
import signal
import subprocess
import sys
import threading

import rclpy
from ament_index_python.packages import get_package_prefix
from rclpy.node import Node
from rclpy.duration import Duration
from std_srvs.srv import Trigger


def _band_worker(node: Node, delay_sec: float, stop: threading.Event) -> None:
    """After delay_sec ON THE NODE CLOCK, call /elastic_band/toggle once.

    With use_sim_time (the sim bringup) the node clock follows /clock, so the
    delay is measured in SIM seconds -- the clock the robot lives on. The
    kitchen sim runs far below 1x real-time, so a wall-clock delay would fire
    at the wrong point of the (plant-clocked) bring-up. Without use_sim_time
    the node clock is system time and this degrades to the old wall behavior.
    spin_once() here also services the /clock subscription (nothing else spins
    this node)."""
    t0 = None
    while not stop.is_set() and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        now = node.get_clock().now()
        if t0 is None:
            # with use_sim_time, now() is 0 until the first /clock arrives
            if now.nanoseconds > 0:
                t0 = now
            continue
        if (now - t0) >= Duration(seconds=delay_sec):
            break
    if stop.is_set() or not rclpy.ok():
        return  # controller exited first
    cli = node.create_client(Trigger, "/elastic_band/toggle")
    if not cli.wait_for_service(timeout_sec=2.0):
        node.get_logger().warning(
            "/elastic_band/toggle not available -- band NOT released "
            "(fine on the real robot; set drop_band:=false to silence)")
        return
    fut = cli.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
    if fut.done() and fut.result() is not None:
        node.get_logger().info(f"elastic band released: {fut.result().message}")
    else:
        node.get_logger().warning("elastic band toggle call did not complete")


def main() -> None:
    rclpy.init()
    node = Node("mjpc_deploy_lowerbody_controller")

    _libdir = os.path.join(get_package_prefix("h12_deploy_mjpc"),
                           "lib", "h12_deploy_mjpc")

    p = node.declare_parameter
    # controller_exe selects which compiled core to run (mjpc_lowerbody_core =
    # legs-only Stabilize; mjpc_fullbody_core = whole-body Lean). controller_binary
    # (an absolute path) still overrides if set.
    exe = p("controller_exe", "mjpc_lowerbody_core").value
    binary = p("controller_binary", os.path.join(_libdir, exe)).value
    # Same knob set + semantics as the fork's CLI (see mjpc/deploy/
    # h12_lower_body_controller.cc); defaults here are the SIM bringup's --
    # config/mjpc_sim.yaml / mjpc_real.yaml override.
    args = [
        binary,
        "--task", str(p("task", "Stabilize H12 Magpie").value),
        "--strategy", str(p("strategy", 6).value),
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
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(rc if rc is not None and rc >= 0 else 1)


if __name__ == "__main__":
    main()
