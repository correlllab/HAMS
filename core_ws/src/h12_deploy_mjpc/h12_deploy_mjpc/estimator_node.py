#!/usr/bin/env python3
"""rclpy parameter shell around the fork's FULL proprioceptive base estimator.

The actual estimator is mjpc/deploy/helper_scripts/base_estimator_node.py in the
bind-mounted mujoco_mpc submodule (badinkajink/mujoco_mpc@extended_hw): a
RW-EKF (or complementary) leg-odometry + IMU-fusion filter over rt/lowstate,
publishing SportModeState_ on a side topic, with contact modes, chi2 outlier
rejection, an accel-health probe, --compare and --selftest. This shell ONLY maps
ROS parameters -> the estimator's argv and then runs it unmodified — the fork
stays the single source of truth, and the estimator's unitree_sdk2py DDS I/O
(one-writer + iface auto-pinning) is deliberately NOT converted to rclpy pub/sub.

Replaces the earlier stripped port (velocity-only leg odometry, hardcoded
covariances, never read the accelerometer) — see git history. That version's
"H3 hard contact gate" (drop swing feet entirely) is NOT carried over: the fork
filter handles outliers via load-scaled measurement noise + chi2 rejection;
bench H3 on the twin before considering re-adding it.

Contract: the estimator's IMU_OFFSET must equal the controller core's kImuOffset
(both are {-0.04452, -0.01891, 0.27756}; both ends of the IMU-site -> pelvis
reconstruction must agree or the base pose is wrong).
"""

import os
import runpy
import sys

import rclpy
from rclpy.node import Node

# In-container defaults (docker-compose mounts). MJPC_ROOT can override for a
# host/dev checkout.
_MJPC_ROOT = os.environ.get("MJPC_ROOT", "/home/code/mujoco_mpc")
_DEFAULT_SCRIPT = os.path.join(
    _MJPC_ROOT, "mjpc", "deploy", "helper_scripts", "base_estimator_node.py")
# The estimator only needs SOME H1-2 model with a free base + the 27 joints in
# lowstate motor order + *ankle_roll* foot bodies; use the same staged task XML
# the controller loads (assets live next to it in the hydrated build tree).
_DEFAULT_SCENE = os.path.join(
    os.environ.get("MJPC_TASKS_DIR",
                   os.path.join(_MJPC_ROOT, "build", "mjpc", "tasks")),
    "humanoid_bench", "stabilize", "Stabilize_H12_Magpie.xml")


def main() -> None:
    rclpy.init()
    node = Node("h12_deploy_mjpc_estimator")

    p = node.declare_parameter
    script = p("estimator_script", _DEFAULT_SCRIPT).value
    scene = p("scene", _DEFAULT_SCENE).value
    domain = p("domain_id", int(os.environ.get("ROS_DOMAIN_ID", "0"))).value
    iface = p("iface", "").value               # "" = auto-pin robot subnet
    no_auto_iface = p("no_auto_iface", False).value
    rate = p("rate", 200.0).value              # publish Hz
    # Plant-clock the filter: plant seconds per lowstate tick (0 = wall-clocked,
    # the validated real/twin behavior). Set to the sim's physics timestep on
    # slower-than-realtime plants — see the fork estimator's --tick-dt help.
    tick_dt = p("tick_dt", 0.0).value
    lowstate_topic = p("lowstate_topic", "rt/lowstate").value
    # Side topic by default: in sim nothing else publishes sportmodestate, but a
    # side topic keeps the one-writer rule trivially safe; the controller's
    # sportstate_topic param points here.
    out_topic = p("out_topic", "rt/sportmodestate_est").value
    vel_lpf_ms = p("vel_lpf_ms", 30.0).value
    imu_fusion = p("imu_fusion", True).value
    imu_tau_ms = p("imu_tau_ms", 300.0).value
    static_dq = p("static_dq", 0.3).value
    contact = p("contact", "min-speed").value  # min-speed | mean
    filt = p("filter", "rw-ekf").value         # rw-ekf | complementary
    ekf_amax = p("ekf_amax", 3.0).value
    ekf_r0 = p("ekf_r0", 0.02).value
    ekf_r1 = p("ekf_r1", 0.3).value
    ekf_lfloor = p("ekf_lfloor", 5.0).value
    ekf_chi2 = p("ekf_chi2", 11.34).value
    ekf_rejcap = p("ekf_rejcap", 50).value
    compare = p("compare", False).value
    truth_topic = p("truth_topic", "rt/sportmodestate").value
    selftest = p("selftest", False).value

    node.get_logger().info(
        f"running fork estimator {script} (filter={filt}, out={out_topic}, "
        f"domain={domain})")
    # Params sourced; the estimator is a pure-DDS worker from here on.
    node.destroy_node()
    rclpy.shutdown()

    if not os.path.isfile(script):
        sys.exit(f"[estimator_node] fork estimator not found at {script} — "
                 "is the mujoco_mpc submodule mounted/checked out?")

    argv = [
        script,
        "--scene", str(scene),
        "--domain", str(domain),
        "--rate", str(rate),
        "--tick-dt", str(tick_dt),
        "--lowstate-topic", str(lowstate_topic),
        "--out-topic", str(out_topic),
        "--vel-lpf-ms", str(vel_lpf_ms),
        "--imu-tau-ms", str(imu_tau_ms),
        "--static-dq", str(static_dq),
        "--contact", str(contact),
        "--filter", str(filt),
        "--ekf-amax", str(ekf_amax),
        "--ekf-r0", str(ekf_r0),
        "--ekf-r1", str(ekf_r1),
        "--ekf-lfloor", str(ekf_lfloor),
        "--ekf-chi2", str(ekf_chi2),
        "--ekf-rejcap", str(ekf_rejcap),
        "--truth-topic", str(truth_topic),
    ]
    if iface:
        argv += ["--iface", str(iface)]
    if no_auto_iface:
        argv.append("--no-auto-iface")
    if not imu_fusion:
        argv.append("--no-imu-fusion")
    if compare:
        argv.append("--compare")
    if selftest:
        argv.append("--selftest")

    sys.argv = argv
    # run_name="__main__" fires the script's own entry point; SIGINT from
    # `ros2 launch` lands as KeyboardInterrupt in its loop, same as a terminal ^C.
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
