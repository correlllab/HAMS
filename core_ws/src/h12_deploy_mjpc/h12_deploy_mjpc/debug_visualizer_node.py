#!/usr/bin/env python3
"""rclpy parameter shell around the fork's MJPC plan visualizer.

The actual visualizer is mjpc/deploy/helper_scripts/plan_visualizer.py in the
bind-mounted mujoco_mpc submodule (badinkajink/mujoco_mpc@extended_hw): it renders
the MEASURED robot in its normal colours (joints from rt/lowstate, base from the
RW-EKF estimator's rt/sportmodestate_est) together with a semi-transparent BLUE
GHOST posed at the planner's current best trajectory (rt/mjpc/plan, published by
the deploy core's planner thread), and writes an mp4 on shutdown.

Same shape as estimator_node.py, for the same reasons: this shell ONLY maps ROS
parameters -> the worker's argv and then runs it unmodified -- the fork stays the
single source of truth, and the worker's unitree_sdk2py DDS I/O is deliberately
NOT converted to rclpy pub/sub (rclcpp's rmw_cyclonedds and unitree_sdk2's bundled
CycloneDDS both export libddsc/libddscxx.so.0; one process loading both corrupts
the heap). Since the worker runs only after rclpy is torn down, the two never live
in the same process.

DEBUG ONLY: nothing in the control path reads this node, and it consumes no ground
truth -- only rt/lowstate, the estimator's estimate, and the planner's own plan. It
is a no-op unless the controller core was started with --plan_topic (the sim
launcher passes it; the real robot does not publish plans by default).
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
    _MJPC_ROOT, "mjpc", "deploy", "helper_scripts", "plan_visualizer.py")
# The SAME staged task XML the controller loads: the ghost is posed from the plan's
# raw qpos rows, so the viewer's model must have the controller's exact qpos layout
# (and the build tree's copy is the one with the generated robot body).
_DEFAULT_SCENE = os.path.join(
    os.environ.get("MJPC_TASKS_DIR",
                   os.path.join(_MJPC_ROOT, "build", "mjpc", "tasks")),
    "humanoid_bench", "stabilize", "Stabilize_H12_Magpie.xml")


def main() -> None:
    rclpy.init()
    node = Node("mjpc_debug_visualizer")

    p = node.declare_parameter
    script = p("visualizer_script", _DEFAULT_SCRIPT).value
    scene = p("scene", _DEFAULT_SCENE).value
    domain = p("domain_id", int(os.environ.get("ROS_DOMAIN_ID", "0"))).value
    iface = p("iface", "").value               # "" = auto-pin robot subnet
    no_auto_iface = p("no_auto_iface", False).value
    lowstate_topic = p("lowstate_topic", "rt/lowstate").value
    plan_topic = p("plan_topic", "rt/mjpc/plan").value
    # ESTIMATOR output, matching the controller's sportstate_topic. Never point this
    # at the sim's ground-truth rt/sportmodestate: a viewer that reads truth would
    # draw a robot the real robot can't.
    sportstate_topic = p("sportstate_topic", "rt/sportmodestate_est").value
    out_path = p("out_path", "/tmp/mjpc_plan.mp4").value
    fps = p("fps", 30.0).value                 # render/write rate (lowstate is 200 Hz)
    width = p("width", 960).value
    height = p("height", 720).value
    camera = p("camera", "").value             # "" = free camera tracking the pelvis
    ghost_rgba = p("ghost_rgba", "0.25,0.45,1.0,0.35").value
    # sweep: the ghost's plan step cycles 0 -> end of horizon in real time, so each
    # loop walks from "now" out to the full 1 s lookahead, alpha fading as it goes.
    # NOT step 0: plan row 0 is the measured state bit-for-bit (trajectory.cc:130
    # seeds it from what the deploy core SetState'd), so a ghost parked there is a
    # tautology -- it renders 1:1 forever, including as the robot falls.
    ghost_mode = p("ghost_mode", "sweep").value          # sweep | single
    sweep_rate = p("sweep_rate", 1.0).value              # plan-seconds per wall-second
    ghost_alpha_far = p("ghost_alpha_far", 0.15).value   # alpha at the far end of the sweep
    ghost_step = p("ghost_step", -1).value     # single mode only; -1 = last step
    ghost_trail = p("ghost_trail", 0).value    # >0 = fading trail every Nth step
    plan_stale_sec = p("plan_stale_sec", 1.0).value      # older than this -> NO ghost
    # OFF by default ON PURPOSE: the ghost base is the PLANNER's belief and the
    # measured base is the ESTIMATOR's; that divergence is real signal, not an
    # artifact. Turn on only to compare posture in isolation.
    align_ghost_base = p("align_ghost_base", False).value
    max_seconds = p("max_seconds", 0.0).value  # 0 = until shutdown

    node.get_logger().info(
        f"running fork plan visualizer {script} (plan={plan_topic}, "
        f"out={out_path}, domain={domain})")
    # Params sourced; the visualizer is a pure-DDS worker from here on.
    node.destroy_node()
    rclpy.shutdown()

    if not os.path.isfile(script):
        sys.exit(f"[debug_visualizer_node] fork visualizer not found at {script} — "
                 "is the mujoco_mpc submodule mounted/checked out?")

    argv = [
        script,
        "--scene", str(scene),
        "--domain", str(domain),
        "--lowstate-topic", str(lowstate_topic),
        "--plan-topic", str(plan_topic),
        "--sportstate-topic", str(sportstate_topic),
        "--out", str(out_path),
        "--fps", str(fps),
        "--width", str(width),
        "--height", str(height),
        "--ghost-rgba", str(ghost_rgba),
        "--ghost-mode", str(ghost_mode),
        "--sweep-rate", str(sweep_rate),
        "--ghost-alpha-far", str(ghost_alpha_far),
        "--ghost-step", str(ghost_step),
        "--ghost-trail", str(ghost_trail),
        "--plan-stale-sec", str(plan_stale_sec),
        "--max-seconds", str(max_seconds),
    ]
    if camera:
        argv += ["--camera", str(camera)]
    if iface:
        argv += ["--iface", str(iface)]
    if no_auto_iface:
        argv.append("--no-auto-iface")
    if align_ghost_base:
        argv.append("--align-ghost-base")

    sys.argv = argv
    # run_name="__main__" fires the script's own entry point; SIGINT from
    # `ros2 launch` reaches its handler, which unwinds the loop so the mp4 is
    # finalised (an unreleased VideoWriter = a truncated, unplayable file).
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
