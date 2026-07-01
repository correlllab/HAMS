#!/usr/bin/env python3
"""MJPC DDS control node for the Unitree H1-2 (Python skeleton).

The Python analog of the fork's C++ deploy node (``mjpc/deploy/h12_control_node.cc``).
Where the C++ node embeds ``mjpc::Agent`` in-process, this drives the planner through
the installed ``mujoco_mpc`` gRPC ``Agent`` (the "Python gRPC bridge" the C++ header
references: it proved the control logic but round-trips each plan, so it is far slower
than the embedded planner — fine as a bring-up / digital-twin tool, not for hard
real-time on hardware).

ARCHITECTURE (mirrors the C++ node: a continuous planner + a control loop)
  planner thread : agent.planner_step() forever on the latest state  (≙ async Plan())
  control thread : @--ctrl_hz read DDS state -> pelvis_from_site -> set_state
                   -> get_action -> q* + gravity-FF tau -> LowCmd_ -> safety layer
  (with --plan_rate_hz / --sync_plan the planner runs synchronously in the control
   thread on the fresh state, like the C++ node's proven mode, gRPC rate permitting.)

OUTPUT  --split lower (default): publish ONLY the legs (0..11) to
  rt/safety/lowcmd_lower_in, so this acts as the lower-body controller alongside a
  separate upper-body controller on the safety layer's upper split. The planner
  still ingests whole-body state, so it balances the legs against the measured,
  externally-driven arm/torso pose. --split full restores whole-body output to
  rt/safety/lowcmd_in (≙ the C++ node).

STATE  pelvis (free-joint) is backed out of the reported IMU-site pose; see
  mjpc/state_estimation.py:pelvis_from_site (the analog of the C++ fill_state).
  The digital twin (h1_robocasa) publishes only rt/lowstate, so run it with
  --require_sportstate false (nominal base + IMU/joint balance).

This is a SKELETON: the load-bearing path is implemented; the advanced extras
(latency comp, single-stream velocity LPF, IMU/ankle calibration, live-switch
settle/blend, full B0 metrics, in-process gRPC monitor) are TODO stubs.
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

from h12_lowerbody_controller.mjpc.constants import (
    DEFAULT_BASE_HEIGHT,
    DEFAULT_STANCE,
    KNU,
    LOWER_JOINTS,
    NUM_LOWER,
    TAU_LIMIT,
)
from h12_lowerbody_controller.mjpc.state_estimation import MjpcState, pelvis_from_site
from h12_lowerbody_controller.mjpc.agent_client import MjpcAgentClient
from h12_lowerbody_controller.mjpc.control import (
    BringUpRamp,
    GravityFeedforward,
    build_lowcmd,
    target_clamp,
)


def auto_detect_robot_interface() -> str:
    """Return the interface holding a 192.168.123.x address (wired H1-2 subnet),
    so an empty --network_interface binds the robot link instead of WiFi/Tailscale.
    Mirrors the C++ AutoDetectRobotInterface; "" when no robot-subnet NIC exists
    (-> caller keeps autodetermine/loopback, the right default for the twin)."""
    try:
        import psutil
    except Exception:
        return ""
    for name, addrs in psutil.net_if_addrs().items():
        for a in addrs:
            if a.family == socket.AF_INET and a.address.startswith("192.168.123."):
                return name
    return ""


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # task / planner
    p.add_argument("--task", default="Stabilize H12 Magpie",
                   help="MJPC task id (default: the H1-2 lower-body-only nu=12 stabilize task)")
    p.add_argument("--strategy", type=int, default=6,
                   help="Lean Strategy index (6=stand 8=crouch ...)")
    p.add_argument("--strategy_param", default="Strategy",
                   help="task-parameter key for the strategy (verify with get_task_parameters)")
    p.add_argument("--plan_rate_hz", type=float, default=0.0,
                   help="if >0: plan SYNCHRONOUSLY ~this many iters/sec (fractional) on the fresh "
                        "state in the control thread (≙ C++ proven mode); else async planner thread")
    p.add_argument("--sync_plan", type=int, default=0,
                   help="if >0: N synchronous PlanIteration per control tick (overridden by --plan_rate_hz)")
    # control
    p.add_argument("--ctrl_hz", type=float, default=200.0, help="control / publish rate (Hz)")
    p.add_argument("--gravity_ff", type=float, default=0.85,
                   help="joint gravity feed-forward scale (tau = scale*qfrc_bias); 0 disables")
    p.add_argument("--model_xml", default=os.environ.get("MJPC_GRAVITY_XML", ""),
                   help="local H1-2 mujoco XML for the gravity feed-forward (free-joint + 27 dofs, "
                        "/lowstate order). Empty -> gravity FF disabled (tau=0).")
    p.add_argument("--no_target_clamp", action="store_true",
                   help="disable the torque-aware target clamp (raw planner targets)")
    # ramp
    p.add_argument("--warmup_sec", type=float, default=1.0,
                   help="hold the measured pose while the planner converges before releasing policy")
    p.add_argument("--start_ramp_sec", type=float, default=5.0,
                   help="bring-up ramp: blend measured->stance over this many seconds")
    p.add_argument("--ramp_hold_sec", type=float, default=3.0, help="scripted hold after the ramp")
    p.add_argument("--policy_blend_sec", type=float, default=0.0,
                   help="ease stance->live policy over this many seconds (0 = hard switch)")
    # elastic band (sim overhead tether held by the robocasa twin)
    p.add_argument("--disable_elastic_band", default="true",
                   help="release the sim's overhead elastic band (/elastic_band/toggle) once the "
                        "bring-up ramp completes, so the robot must balance on its own; false = leave it")
    p.add_argument("--band_release_sec", type=float, default=-1.0,
                   help="when to release the band (wall sec from start). <0 = auto: after the bring-up "
                        "ramp (start_ramp_sec + ramp_hold_sec), i.e. once the planner drives the stance")
    # DDS
    p.add_argument("--split", choices=["lower", "full"], default="lower",
                   help="lower = publish ONLY the legs (0..11) to rt/safety/lowcmd_lower_in, to "
                        "coexist with a separate upper-body controller on the upper split; "
                        "full = whole body (27) to rt/safety/lowcmd_in")
    p.add_argument("--lowcmd_topic", default="",
                   help="override the output DDS topic (default chosen by --split)")
    p.add_argument("--lowstate_topic", default="rt/lowstate")
    p.add_argument("--sportstate_topic", default="rt/sportmodestate")
    p.add_argument("--domain_id", type=int, default=int(os.environ.get("ROS_DOMAIN_ID", "1")))
    p.add_argument("--network_interface", default="",
                   help="DDS NIC (empty = auto-pin 192.168.123.x robot subnet, else loopback)")
    p.add_argument("--require_sportstate", default="true",
                   help="true = need rt/sportmodestate (real base pose); false = DEBUG nominal base "
                        "(the twin path: it publishes only rt/lowstate)")
    p.add_argument("--base_height", type=float, default=DEFAULT_BASE_HEIGHT,
                   help="nominal pelvis height (m) when sportmodestate is unavailable")
    # ---- advanced (TODO stubs; flags present for CLI parity with the C++ node) ----
    p.add_argument("--use_twin_time", default="false", help="(TODO) clock the planner by the twin tick")
    p.add_argument("--latency_comp", default="false", help="(TODO) predict state forward by loop delay")
    p.add_argument("--vel_lpf_ms", type=float, default=0.0, help="(TODO) single-stream base-vel LPF tau (ms)")
    p.add_argument("--imu_pitch_offset_deg", type=float, default=0.0, help="IMU pitch zero-offset (deg)")
    p.add_argument("--imu_roll_offset_deg", type=float, default=0.0, help="IMU roll zero-offset (deg)")
    p.add_argument("--ankle_roll_offset_l_deg", type=float, default=0.0, help="(TODO) L ankle-roll calib (deg)")
    p.add_argument("--ankle_roll_offset_r_deg", type=float, default=0.0, help="(TODO) R ankle-roll calib (deg)")
    return p


def parse_args(argv=None):
    """Parse our flags, ignoring any extra args (e.g. ros2 launch's --ros-args)."""
    args, _ = build_parser().parse_known_args(argv)
    return args


def _truthy(s: str) -> bool:
    return str(s).lower() in ("1", "true", "yes", "on")


def run_control(args, exit_event, release_band_cb=None):
    """The DDS control loop (runs in a worker thread under the rclpy node).

    ``release_band_cb`` (the node's elastic-band release) is invoked once, after the
    bring-up ramp, so the sim's overhead tether drops only after the planner is
    holding the stance.
    """
    require_sport = _truthy(args.require_sportstate)
    ctrl_dt = 1.0 / args.ctrl_hz
    nominal_base_p = np.array([0.0, 0.0, args.base_height])
    stance = DEFAULT_STANCE.copy()
    imu_pitch_off = math.radians(args.imu_pitch_offset_deg)
    imu_roll_off = math.radians(args.imu_roll_offset_deg)

    # Elastic band: release once the bring-up ramp has settled the robot at its
    # stance (releasing at t=0 would just drop it before the planner has authority).
    disable_band = _truthy(args.disable_elastic_band)
    band_release_sec = (args.band_release_sec if args.band_release_sec >= 0
                        else args.start_ramp_sec + args.ramp_hold_sec)

    # Output split: a lower-body controller drives ONLY the legs (0..11) on the
    # safety layer's lower split; the planner still ingests whole-body state, so it
    # balances the legs against the (externally-driven) measured arm/torso pose.
    if args.split == "lower":
        out_joints = list(LOWER_JOINTS)
        out_topic = args.lowcmd_topic or "rt/safety/lowcmd_lower_in"
    else:
        out_joints = list(range(KNU))
        out_topic = args.lowcmd_topic or "rt/safety/lowcmd_in"

    # ---- DDS interface resolution (≙ C++): explicit wins; else auto-pin robot NIC ----
    net_if = args.network_interface or auto_detect_robot_interface()
    if net_if:
        print(f"[mjpc] DDS interface '{net_if}'")
    else:
        print("[mjpc] no robot-subnet NIC -> DDS autodetermine (twin/loopback)")
    ChannelFactoryInitialize(args.domain_id, net_if if net_if else None)

    # ---- MJPC gRPC Agent (spawns agent_server) ----
    print(f"[mjpc] starting agent: task='{args.task}' strategy={args.strategy}")
    agent = MjpcAgentClient(
        task_id=args.task, strategy=args.strategy, strategy_param=args.strategy_param,
    )
    try:
        print(f"[mjpc] task params (find the Strategy key here): {agent.task_parameters()}")
    except Exception as exc:
        print(f"[mjpc] (could not read task params: {exc})")
    print(f"[mjpc] server model: nq={agent.nq} nv={agent.nv}")

    gravity = GravityFeedforward(args.model_xml or None, gff=args.gravity_ff)
    print(f"[mjpc] gravity feed-forward: {'ON' if gravity.enabled else 'OFF (tau=0)'}")
    ramp = BringUpRamp(args.start_ramp_sec, args.ramp_hold_sec, args.policy_blend_sec)
    do_clamp = not args.no_target_clamp

    # ---- shared state + DDS handlers ----
    state = MjpcState()
    lock = threading.Lock()

    def on_lowstate(msg):
        with lock:
            for i in range(KNU):
                state.q[i] = msg.motor_state[i].q
                state.dq[i] = msg.motor_state[i].dq
            for k in range(4):
                state.quat[k] = msg.imu_state.quaternion[k]
            for k in range(3):
                state.gyro[k] = msg.imu_state.gyroscope[k]
            state.mode_machine = msg.mode_machine
            state.tick = msg.tick
            state.have_ls = True

    def on_sport(msg):
        with lock:
            for k in range(3):
                state.site_p[k] = msg.position[k]
                state.site_v[k] = msg.velocity[k]
            state.have_ss = True

    ls_sub = ChannelSubscriber(args.lowstate_topic, LowState_)
    ls_sub.Init(on_lowstate, 10)
    ss_sub = ChannelSubscriber(args.sportstate_topic, SportModeState_)
    ss_sub.Init(on_sport, 10)
    cmd_pub = ChannelPublisher(out_topic, LowCmd_)
    cmd_pub.Init()
    print(f"[mjpc] output: split={args.split} -> {out_topic} "
          f"({'legs 0..%d' % (NUM_LOWER - 1) if args.split == 'lower' else 'whole body 0..%d' % (KNU - 1)})")
    if disable_band and release_band_cb is not None:
        print(f"[mjpc] elastic band: will release (/elastic_band/toggle) at t={band_release_sec:.1f}s")
    else:
        print("[mjpc] elastic band: leaving engaged")

    # ---- wait for the first full state ----
    print(f"[mjpc] waiting for {args.lowstate_topic}"
          f"{' + ' + args.sportstate_topic if require_sport else ' (debug: nominal base)'} ...")
    while not exit_event.is_set():
        with lock:
            ready = state.have_ls and (state.have_ss or not require_sport)
        if ready:
            break
        time.sleep(0.02)
    print("[mjpc] state stream up -> starting control loop. Type a Strategy number + Enter (q=quit).")

    # ---- planner cadence ----
    sync_in_ctrl = args.plan_rate_hz > 0.0 or args.sync_plan > 0
    if not sync_in_ctrl:
        agent.start_planner_thread()
        print("[mjpc] planner: async background thread")
    else:
        print(f"[mjpc] planner: synchronous "
              f"({'%.0f iters/s' % args.plan_rate_hz if args.plan_rate_hz > 0 else '%d iters/tick' % args.sync_plan})")

    # ---- live Strategy switch via stdin ----
    def stdin_loop():
        for line in sys.stdin:
            line = line.strip()
            if line in ("q", "quit"):
                exit_event.set()
                break
            if not line:
                continue
            try:
                s = int(line)
                agent.set_strategy(s)
                print(f"[mjpc] >>> Strategy -> {s}")
            except ValueError:
                print("[mjpc] (enter a Strategy number, or q to quit)")

    threading.Thread(target=stdin_loop, name="mjpc-stdin", daemon=True).start()

    # ---- control loop @ ctrl_hz ----
    q_init = None
    action = np.zeros(KNU)
    plan_accum = 0.0
    t0 = time.monotonic()
    next_t = t0
    ticks = 0
    last_plan_count, last_wall = 0, 0.0
    try:
        while not exit_event.is_set():
            with lock:
                cur = state.snapshot()  # deep copy under the lock; never read a half-written frame
            wall = time.monotonic() - t0
            warming = wall < args.warmup_sec
            if q_init is None and cur.have_ls:
                q_init = cur.q.copy()

            # release the sim's elastic band once the bring-up ramp has settled
            # (idempotent; the node-side callback only fires the service once).
            if disable_band and release_band_cb is not None and wall >= band_release_sec:
                release_band_cb()

            qpos, qvel = pelvis_from_site(
                cur, nominal_base_p, require_sport=require_sport,
                imu_pitch_off=imu_pitch_off, imu_roll_off=imu_roll_off,
            )
            agent.set_robot_state(wall, qpos, qvel)

            # synchronous planning on THIS fresh state (≙ C++ proven modes)
            if args.plan_rate_hz > 0.0:
                plan_accum += args.plan_rate_hz * ctrl_dt
                niter = int(plan_accum)
                plan_accum -= niter
                if niter:
                    agent.plan_n(niter)
            elif args.sync_plan > 0:
                agent.plan_n(args.sync_plan)

            if not warming:
                a = np.asarray(agent.get_action(wall))
                # The planner may actuate fewer joints than the full 27 — the
                # "Stabilize H12 Magpie" task is lower-body-only (nu=12). Drive the
                # joints it controls; hold the rest at the measured pose (and for
                # --split lower only legs 0..11 are published anyway).
                n = min(a.shape[0], KNU)
                action = cur.q.copy()
                action[:n] = a[:n]

            tau = gravity.tau(qpos)
            tgt = ramp.target(wall, q_init if q_init is not None else cur.q,
                              stance, action if not warming else cur.q, warming)
            if do_clamp:
                tgt = target_clamp(tgt, cur.q)

            cmd_pub.Write(build_lowcmd(tgt, tau, cur.mode_machine, joint_indices=out_joints))

            # ---- status line ~1/s (B0 metrics summary is a TODO stub) ----
            ticks += 1
            if ticks % max(1, int(args.ctrl_hz)) == 0:
                base_z = qpos[2]
                tilt = math.degrees(math.acos(
                    max(-1.0, min(1.0, 1.0 - 2.0 * (qpos[4] ** 2 + qpos[5] ** 2)))))
                pc = agent.plan_count
                prate = (pc - last_plan_count) / (wall - last_wall) if wall > last_wall else 0.0
                last_plan_count, last_wall = pc, wall
                tau_ff_pct = 100 * np.max(np.abs(tau) / TAU_LIMIT) if gravity.enabled else 0.0
                print(f"[mjpc] t={wall:6.1f}s {'WARMUP' if warming else 'policy'} "
                      f"z={base_z:.3f} tilt={tilt:4.1f} plan={prate:5.0f}/s "
                      f"tau_ff_max%={tau_ff_pct:4.0f}")
                # gRPC planner telemetry: throughput + per-RPC round-trip latency +
                # cost. rate/ctrl_hz = planner iterations per control tick (<1 means
                # the policy is being read on a state staler than this tick); the
                # latencies show where the gRPC time goes (planner_step dominates).
                lat = agent.latencies()
                try:
                    cost = agent.get_total_cost()
                except Exception:
                    cost = float("nan")
                print(f"[mjpc-planner] rate={prate:5.0f}/s ({prate / args.ctrl_hz:4.2f} iter/tick) "
                      f"plan_step={lat['planner_step']:5.1f}ms set_state={lat['set_state']:4.1f}ms "
                      f"get_action={lat['get_action']:4.1f}ms cost={cost:7.3f}")
                # TODO(skeleton): accumulate the full B0 baseline (trackRMSE, |tau|,
                # sat%, base z/tilt stats) and print the summary table on exit.
            next_t += ctrl_dt
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()  # fell behind; resync
    except KeyboardInterrupt:
        pass
    finally:
        print("[mjpc] shutting down ...")
        exit_event.set()
        agent.close()
        try:
            ls_sub.Close()
            ss_sub.Close()
            cmd_pub.Close()
        except Exception:
            pass


class MjpcNode(Node):
    """rclpy node wrapper (≙ h12_safety_layer's SafetyNode): it keeps the process in
    the ROS graph + handles launch/shutdown, while the raw-DDS control loop runs in a
    daemon worker thread. This lets a standard launch_ros ``Node`` start mjpc_node;
    the loop itself still speaks Cyclone DDS via unitree_sdk2py."""

    def __init__(self, args):
        super().__init__("mjpc_node")
        self._band_released = False
        self._band_cli = self.create_client(Trigger, "/elastic_band/toggle")
        self._exit = threading.Event()
        self._worker = threading.Thread(
            target=self._run, args=(args,), name="mjpc-control", daemon=True
        )
        self._worker.start()

    def _run(self, args):
        try:
            run_control(args, self._exit, release_band_cb=self.release_elastic_band)
        except Exception as exc:  # keep the node alive for a clean shutdown
            self.get_logger().error(f"mjpc control loop exited: {exc}")

    def release_elastic_band(self):
        """Disable the sim's overhead tether (same /elastic_band/toggle service the
        RL controllers use). Idempotent: fires the Trigger exactly once; until the
        service is discovered it no-ops, so the worker can call it every tick. The
        spinning executor (rclpy.spin in main) delivers the async response."""
        if self._band_released or not self._band_cli.service_is_ready():
            return
        self._band_released = True
        self.get_logger().info("releasing elastic band (mjpc bring-up complete)")
        fut = self._band_cli.call_async(Trigger.Request())
        fut.add_done_callback(lambda f: self.get_logger().info(
            f"elastic band toggle: {f.result().message}" if f.result()
            else "elastic band toggle call failed"))

    def destroy_node(self):
        self._exit.set()
        if self._worker.is_alive():
            self._worker.join(timeout=3.0)
        return super().destroy_node()


def main(argv=None):
    # Split our flags from ros2 launch's injected --ros-args (≙ SafetyNode.main).
    args, ros_args = build_parser().parse_known_args(argv)
    rclpy.init(args=ros_args)
    node = MjpcNode(args)
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
