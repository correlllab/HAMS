#!/usr/bin/env python3
"""Very basic frontier-based autonomous exploration for the H1 Mac sim.

Loop:
  1. Read the slam_toolbox /map (occupancy grid).
  2. Find frontiers = FREE cells that touch UNKNOWN space (the edge of the map).
  3. Cluster them; send the nearest sizeable cluster's centroid as a nav2 goal.
  4. When the goal finishes (or is unreachable -> blacklisted), pick the next one.
  5. Stop when no frontiers remain (map fully explored).

nav2 plans a collision-free path to each frontier and drives /cmd_vel, which the
walk policy follows -- so the robot must be in walk mode (rob_stand; rob_walk)
with HAMS_SLAM=1 HAMS_NAV2=1. This is deliberately simple: nearest-frontier with a
failure blacklist, no fancy information-gain scoring.

Run:  python3 /home/code/h12_sim_scripts/frontier_explore.py     (or: rob_explore)
"""
import math

import numpy as np
import rclpy
import tf2_ros
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from scipy import ndimage


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__("frontier_explorer")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("base_frame", "pelvis")
        self.declare_parameter("min_frontier_cells", 6)   # ignore tiny frontiers
        self.declare_parameter("blacklist_radius", 0.6)    # m; near a failed goal
        self.declare_parameter("min_goal_distance", 0.7)   # m; skip frontiers underfoot
        self.declare_parameter("goal_timeout", 120.0)      # s; cancel a stuck goal

        self.base_frame = self.get_parameter("base_frame").value
        self.min_cells = int(self.get_parameter("min_frontier_cells").value)
        self.bl_radius = float(self.get_parameter("blacklist_radius").value)
        self.min_dist = float(self.get_parameter("min_goal_distance").value)
        self.goal_timeout = float(self.get_parameter("goal_timeout").value)

        self._map = None
        # slam_toolbox latches /map (transient-local, reliable).
        map_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(OccupancyGrid,
                                 self.get_parameter("map_topic").value,
                                 self._on_map, map_qos)
        self._nav = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._blacklist = []      # [(x, y), ...] map-frame goals that failed
        self._busy = False        # a goal is in flight
        self._goal_xy = None
        self._goal_deadline = None
        self._goal_handle = None
        self._done = False

        self.create_timer(2.0, self._tick)
        self.get_logger().info(
            "frontier_explorer up: waiting for /map + nav2. Robot must be in walk "
            "mode (rob_walk).")

    # -- inputs --------------------------------------------------------------
    def _on_map(self, msg):
        self._map = msg

    def _robot_xy(self):
        try:
            t = self._tf_buffer.lookup_transform("map", self.base_frame,
                                                 rclpy.time.Time())
            return t.transform.translation.x, t.transform.translation.y
        except Exception:
            return None

    # -- frontier detection --------------------------------------------------
    def _frontiers(self):
        """Return [(x, y, n_cells)] frontier-cluster centroids in the map frame."""
        m = self._map
        g = np.array(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
        free = (g >= 0) & (g < 25)          # confidently free
        unknown = g < 0                     # -1 == unexplored
        occupied = g >= 65                   # walls
        # A frontier cell is free and 8-adjacent to unknown, but NOT touching a
        # wall (those edges are just the far side of an obstacle, not open space).
        unknown_adj = ndimage.binary_dilation(unknown, iterations=1)
        wall_adj = ndimage.binary_dilation(occupied, iterations=1)
        frontier = free & unknown_adj & ~wall_adj

        labels, n = ndimage.label(frontier, structure=np.ones((3, 3)))
        res = m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y
        out = []
        for i in range(1, n + 1):
            ys, xs = np.where(labels == i)
            if len(xs) < self.min_cells:
                continue
            cx = ox + (xs.mean() + 0.5) * res
            cy = oy + (ys.mean() + 0.5) * res
            out.append((cx, cy, int(len(xs))))
        return out

    def _blacklisted(self, x, y):
        return any(math.hypot(x - bx, y - by) < self.bl_radius
                   for bx, by in self._blacklist)

    # -- exploration loop ----------------------------------------------------
    def _tick(self):
        if self._done or self._map is None:
            return
        # Time out a stuck goal.
        if self._busy:
            if self._goal_deadline is not None and \
                    self.get_clock().now().nanoseconds * 1e-9 > self._goal_deadline:
                self.get_logger().warn("goal timed out — cancelling + blacklisting")
                if self._goal_xy:
                    self._blacklist.append(self._goal_xy)
                if self._goal_handle is not None:
                    self._goal_handle.cancel_goal_async()
                self._busy = False
            return

        rp = self._robot_xy()
        if rp is None:
            return
        cand = [(x, y, s) for (x, y, s) in self._frontiers()
                if not self._blacklisted(x, y)
                and math.hypot(x - rp[0], y - rp[1]) > self.min_dist]
        if not cand:
            self.get_logger().info("*** No frontiers left — exploration complete. ***")
            self._done = True
            return
        # Nearest sizeable frontier (simplest sensible policy).
        gx, gy, sz = min(cand, key=lambda c: math.hypot(c[0] - rp[0], c[1] - rp[1]))
        self._send_goal(rp, gx, gy, sz)

    def _send_goal(self, rp, gx, gy, sz):
        if not self._nav.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("nav2 /navigate_to_pose not available yet")
            return
        self._busy = True
        self._goal_xy = (gx, gy)
        self._goal_deadline = self.get_clock().now().nanoseconds * 1e-9 + self.goal_timeout
        yaw = math.atan2(gy - rp[1], gx - rp[0])   # face the frontier
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y = float(gx), float(gy)
        ps.pose.orientation.z = math.sin(yaw / 2.0)
        ps.pose.orientation.w = math.cos(yaw / 2.0)
        goal = NavigateToPose.Goal()
        goal.pose = ps
        self.get_logger().info(
            f"exploring frontier at ({gx:.2f}, {gy:.2f}) [{sz} cells, "
            f"{math.hypot(gx - rp[0], gy - rp[1]):.1f} m away]; "
            f"{len(self._blacklist)} blacklisted")
        self._nav.send_goal_async(goal).add_done_callback(self._on_response)

    def _on_response(self, fut):
        gh = fut.result()
        if gh is None or not gh.accepted:
            self.get_logger().warn("goal rejected — blacklisting")
            if self._goal_xy:
                self._blacklist.append(self._goal_xy)
            self._busy = False
            return
        self._goal_handle = gh
        gh.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, fut):
        status = fut.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("frontier reached")
        else:
            self.get_logger().warn(f"goal ended (status {status}) — blacklisting")
            if self._goal_xy:
                self._blacklist.append(self._goal_xy)
        self._goal_handle = None
        self._busy = False


def main():
    rclpy.init()
    node = FrontierExplorer()
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
