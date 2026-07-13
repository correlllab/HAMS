#!/usr/bin/env python3
"""ROS debugging MCP server for the H1 Mac sim.

Runs INSIDE the hams_ros container (where the FastDDS graph is reachable) and
exposes an HTTP MCP endpoint. Colima doesn't forward container ports to the Mac,
so it binds localhost inside the VM and you reach it over the same SSH tunnel as
the noVNC viewers (see mac_vnc_tunnel.sh). Register it on the Mac with:

    claude mcp add --transport http ros_debug http://localhost:6082/mcp

Design: a single warm rclpy node subscribes once to the hot topics (odom,
costmap, plan, scan, clock) and caches the latest message, so status tools answer
instantly instead of paying node-startup + DDS-discovery on every call (the thing
that made `ros2 topic echo --once` slow to poll). Actuation (lower-body mode,
/cmd_vel) uses warm publishers/clients. Generic passthroughs (echo/hz/service
call for arbitrary topics) shell out to the ros2 CLI, which is robust for the
rarely-used cold path. rclpy spins in a background thread; tool handlers read the
thread-safe cache.

Env: HAMS_ROS_MCP_PORT (default 6082), ROS_DOMAIN_ID, RMW_IMPLEMENTATION.
"""
import asyncio
import math
import os
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Trigger

from mcp.server.fastmcp import FastMCP

PORT = int(os.environ.get("HAMS_ROS_MCP_PORT", "6082"))
_LATCHED = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST, depth=1)
# BEST_EFFORT reader is compatible with BOTH best-effort and reliable writers, so
# it receives sensor/odom topics whatever QoS the publisher chose (the sim's /odom
# and /converted_scan are best-effort; a reliable reader silently gets nothing).
_SENSOR = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE,
                     history=HistoryPolicy.KEEP_LAST, depth=10)


class RosDebugNode(Node):
    """Warm subscriber + actuator. Caches latest messages under a lock."""

    def __init__(self):
        super().__init__("ros_debug_mcp")
        self._lock = threading.Lock()
        self._odom = None
        self._costmap = None
        self._plan = None
        self._scan = None
        self._clock = None
        self.create_subscription(Odometry, "/odom", self._cb("_odom"), _SENSOR)
        self.create_subscription(LaserScan, "/converted_scan", self._cb("_scan"), _SENSOR)
        self.create_subscription(Clock, "/clock", self._cb("_clock"), _SENSOR)
        self.create_subscription(OccupancyGrid, "/global_costmap/costmap",
                                 self._cb("_costmap"), _LATCHED)
        self.create_subscription(Path, "/plan", self._cb("_plan"), _SENSOR)
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._fame_cli = self.create_client(Trigger, "/lowerbody/start_fame")
        self._walk_cli = self.create_client(Trigger, "/lowerbody/start_walk")

    def _cb(self, attr):
        def f(msg):
            with self._lock:
                setattr(self, attr, msg)
        return f

    def get(self, attr):
        with self._lock:
            return getattr(self, attr)

    # -- derived state -------------------------------------------------------
    def sim_time(self):
        c = self.get("_clock")
        return None if c is None else c.clock.sec + c.clock.nanosec * 1e-9

    def robot_state(self):
        o = self.get("_odom")
        if o is None:
            return None
        p, q = o.pose.pose.position, o.pose.pose.orientation
        upright = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)  # body-up . world-up
        return {"x": round(p.x, 3), "y": round(p.y, 3), "z": round(p.z, 3),
                "uprightness": round(upright, 3),
                "posture": ("standing" if (upright > 0.85 and p.z > 0.7)
                            else "fallen" if (upright < 0.5 or p.z < 0.4)
                            else "leaning/other")}

    def costmap_summary(self):
        m = self.get("_costmap")
        if m is None:
            return None
        a = np.asarray(m.data, dtype=np.int16)
        return {"width": m.info.width, "height": m.info.height,
                "resolution": round(m.info.resolution, 3),
                "lethal": int((a >= 99).sum()),
                "inflated": int(((a >= 50) & (a < 99)).sum()),
                "free": int((a == 0).sum()),
                "unknown": int((a == -1).sum())}

    def plan_summary(self):
        pl = self.get("_plan")
        if pl is None or not pl.poses:
            return {"has_plan": False, "poses": 0, "length_m": 0.0}
        pts = [(ps.pose.position.x, ps.pose.position.y) for ps in pl.poses]
        length = sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
                     for i in range(1, len(pts)))
        return {"has_plan": True, "poses": len(pts), "length_m": round(length, 2)}

    # -- actuation -----------------------------------------------------------
    def call_trigger(self, which, timeout=8.0):
        cli = self._fame_cli if which == "fame" else self._walk_cli
        if not cli.wait_for_service(timeout_sec=2.0):
            return {"ok": False, "error": f"/lowerbody/start_{which} unavailable"}
        fut = cli.call_async(Trigger.Request())
        t0 = time.time()
        while not fut.done() and time.time() - t0 < timeout:
            time.sleep(0.05)
        if not fut.done():
            return {"ok": False, "error": "service call timed out"}
        r = fut.result()
        return {"ok": bool(r.success), "message": r.message}

    def publish_cmd(self, vx, vy, wz):
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = float(vx), float(vy), float(wz)
        self._cmd_pub.publish(t)


NODE: RosDebugNode = None  # set in main()


def _sh(*args, timeout=10):
    """Run a ros2 CLI command (env already sourced by the launcher) -> stdout."""
    import subprocess
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or out.stderr or "").strip()
    except subprocess.TimeoutExpired:
        return f"(timed out after {timeout}s)"


mcp = FastMCP("ros_debug", host="127.0.0.1", port=PORT)


# ---- hot tools (warm cache, instant) --------------------------------------
@mcp.tool()
def robot_status() -> dict:
    """H1 base state at a glance: position (x,y,z, odom/world frame), uprightness
    (1.0 = vertical, <0.5 = fallen), posture, and current sim time. Instant."""
    st = NODE.robot_state()
    return {"error": "no /odom yet"} if st is None else {
        **st, "sim_time": NODE.sim_time()}


@mcp.tool()
def costmap_summary() -> dict:
    """nav2 GLOBAL costmap cell counts: lethal / inflated / free / unknown, plus
    size and resolution. lethal>0 means the planner is seeing obstacles."""
    s = NODE.costmap_summary()
    return {"error": "no /global_costmap/costmap (nav2 running?)"} if s is None else s


@mcp.tool()
def nav_status() -> dict:
    """Current nav2 global plan: whether one exists, pose count, and path length
    in metres (from /plan)."""
    return NODE.plan_summary()


@mcp.tool()
def scan_status() -> dict:
    """Is /converted_scan (the SLAM/costmap laser input) live? Reports the latest
    scan's stamp and range count, or that it's absent."""
    s = NODE.get("_scan")
    if s is None:
        return {"live": False, "note": "no /converted_scan (pointcloud_to_laserscan / SLAM running?)"}
    return {"live": True, "stamp_sec": s.header.stamp.sec,
            "ranges": len(s.ranges), "range_min": round(s.range_min, 2),
            "range_max": round(s.range_max, 2)}


# ---- actuation ------------------------------------------------------------
@mcp.tool()
def set_lowerbody(mode: str) -> dict:
    """Switch the lower-body controller. mode='fame' -> stand free (FAME);
    mode='walk' -> hand over to the walk policy (call after fame). Needs the sim
    launched with HAMS_LOWERBODY=switch."""
    if mode not in ("fame", "walk"):
        return {"ok": False, "error": "mode must be 'fame' or 'walk'"}
    return NODE.call_trigger(mode)


@mcp.tool()
async def drive(vx: float = 0.0, vy: float = 0.0, wz: float = 0.0,
                duration: float = 4.0) -> dict:
    """Publish /cmd_vel (body frame, m/s and rad/s) at 20 Hz for `duration`
    seconds, then stop. vx>0 forward, wz>0 turn left. The robot must be in walk
    mode (set_lowerbody('walk')). Reports start/end pose so you can see if it moved."""
    duration = max(0.0, min(float(duration), 30.0))
    p0 = NODE.robot_state()
    hz, end = 0.05, time.time() + duration
    while time.time() < end:
        NODE.publish_cmd(vx, vy, wz)
        await asyncio.sleep(hz)
    NODE.publish_cmd(0.0, 0.0, 0.0)
    p1 = NODE.robot_state()
    moved = (round(math.hypot(p1["x"] - p0["x"], p1["y"] - p0["y"]), 3)
             if p0 and p1 else None)
    return {"ok": True, "commanded": {"vx": vx, "vy": vy, "wz": wz, "duration": duration},
            "start": p0, "end": p1, "moved_m": moved}


@mcp.tool()
async def wait_for(condition: str, timeout: float = 60.0) -> dict:
    """Block until a condition holds, then return (polls the warm cache ~5 Hz).
    Replaces sleep-poll loops. condition is one of:
      standing | fallen | stopped | moving | scan_live | nav_has_plan |
      sim_time>=<N>   (e.g. 'sim_time>=20')
    Returns {met: bool, waited_s, ...state}."""
    timeout = max(1.0, min(float(timeout), 600.0))
    t0 = time.time()
    last = None

    def check():
        nonlocal last
        st = NODE.robot_state()
        last = st
        if condition == "standing":
            return bool(st and st["posture"] == "standing")
        if condition == "fallen":
            return bool(st and st["posture"] == "fallen")
        if condition == "scan_live":
            return NODE.get("_scan") is not None
        if condition == "nav_has_plan":
            return NODE.plan_summary()["has_plan"]
        if condition.startswith("sim_time>="):
            try:
                thr = float(condition.split(">=", 1)[1])
            except ValueError:
                return False
            t = NODE.sim_time()
            return t is not None and t >= thr
        if condition in ("stopped", "moving"):
            a = NODE.robot_state()
            time.sleep(0.4)
            b = NODE.robot_state()
            if not a or not b:
                return False
            d = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
            return (d < 0.01) if condition == "stopped" else (d >= 0.02)
        return False

    while time.time() - t0 < timeout:
        if check():
            return {"met": True, "waited_s": round(time.time() - t0, 1),
                    "state": last, "sim_time": NODE.sim_time()}
        await asyncio.sleep(0.2)
    return {"met": False, "waited_s": round(time.time() - t0, 1),
            "state": last, "sim_time": NODE.sim_time(),
            "note": f"condition '{condition}' not met within {timeout}s"}


# ---- generic passthroughs (cold path, ros2 CLI) ---------------------------
@mcp.tool()
def list_topics(filter: str = "") -> list:
    """List active topics (optionally substring-filtered), as 'name  type'."""
    out = []
    for name, types in sorted(NODE.get_topic_names_and_types()):
        if not filter or filter in name:
            out.append(f"{name}  {','.join(types)}")
    return out


@mcp.tool()
def echo_topic(topic: str, timeout: float = 6.0) -> str:
    """One message from any topic (generic, via `ros2 topic echo --once`). Use the
    hot tools (robot_status/costmap_summary/scan_status) for the common ones."""
    return _sh("ros2", "topic", "echo", "--once", topic, timeout=int(timeout) + 2)


@mcp.tool()
def topic_hz(topic: str, window: float = 5.0) -> str:
    """Measured publish rate of a topic over `window` seconds."""
    import subprocess
    try:
        p = subprocess.run(["ros2", "topic", "hz", topic], capture_output=True,
                           text=True, timeout=window + 3)
        for line in (p.stdout or "").splitlines():
            if "average rate" in line:
                return line.strip()
        return "(no messages in window)"
    except subprocess.TimeoutExpired as e:
        for line in ((e.stdout or b"").decode(errors="ignore")).splitlines():
            if "average rate" in line:
                return line.strip()
        return "(no messages in window)"


@mcp.tool()
def call_service(name: str, type: str, args: str = "{}") -> str:
    """Call any service (generic, via `ros2 service call`). type is the full type
    e.g. 'std_srvs/srv/Trigger'; args is a YAML/JSON dict string."""
    return _sh("ros2", "service", "call", name, type, args, timeout=15)


@mcp.tool()
def node_list() -> list:
    """List active ROS nodes."""
    return sorted(f"{ns}/{n}".replace("//", "/") for n, ns in NODE.get_node_names_and_namespaces())


def _spin(executor):
    executor.spin()


def main():
    rclpy.init()
    global NODE
    NODE = RosDebugNode()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(NODE)
    threading.Thread(target=_spin, args=(ex,), daemon=True).start()
    NODE.get_logger().info(f"ros_debug MCP server on http://127.0.0.1:{PORT}/mcp")
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
