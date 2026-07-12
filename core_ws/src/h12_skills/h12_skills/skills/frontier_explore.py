"""SkillFrontierExplore: autonomous frontier-based map exploration.

Drives the base to successive frontiers of the slam_toolbox /map — FREE cells
that touch UNKNOWN space — via nav2's /navigate_to_pose, until the map is fully
explored (or the goal timeout is hit). Policy is deliberately simple:
nearest-sizeable-frontier with a failure blacklist, no information-gain scoring.

Ported from the standalone mac-sim frontier_explore node into the skills mixin so
it runs as the /skill/frontier_explore action alongside the other skills, reusing
SkillsBase's TF buffer + executor-safe action plumbing (_send_action). Requires
nav2 + slam_toolbox up (HAMS_SLAM=1 HAMS_NAV2=1, and the sim started with
HAMS_SIM_ODOM=1 to supply /odom) and the robot in walk mode so nav2's /cmd_vel is
followed.
"""
import math
import time

import numpy as np
from scipy import ndimage

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from rclpy.time import Time

from custom_ros_messages.action import SkillFrontierExplore

from ..base import _Run

# Defaults applied when the corresponding goal field is left at 0.
_DEF_MIN_CELLS = 6
_DEF_BLACKLIST_RADIUS = 0.6
_DEF_MIN_GOAL_DIST = 0.7
_DEF_GOAL_TIMEOUT = 120.0
_MAP_FRAME = 'map'
_BASE_FRAME = 'pelvis'


class FrontierExploreSkill:
    def _frontier_setup(self):
        """Lazily create the /map subscription + nav2 client on first use. slam
        latches /map transient-local, so a late subscriber still gets the map."""
        if getattr(self, '_frontier_ready', False):
            return
        self._frontier_map = None
        map_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(OccupancyGrid, '/map', self._on_frontier_map,
                                 map_qos, callback_group=self._cb_group)
        self._frontier_nav = ActionClient(self, NavigateToPose,
                                           '/navigate_to_pose',
                                           callback_group=self._cb_group)
        self._frontier_ready = True

    def _on_frontier_map(self, msg):
        self._frontier_map = msg

    def _frontier_robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform(_MAP_FRAME, _BASE_FRAME, Time())
            return t.transform.translation.x, t.transform.translation.y
        except Exception:
            return None

    def _frontiers(self, m, min_cells):
        """Return [(x, y, n_cells)] frontier-cluster centroids in the map frame."""
        g = np.array(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
        free = (g >= 0) & (g < 25)          # confidently free
        unknown = g < 0                     # -1 == unexplored
        occupied = g >= 65                  # walls
        # A frontier cell is free and 8-adjacent to unknown, but NOT touching a
        # wall (those edges are the far side of an obstacle, not open space).
        unknown_adj = ndimage.binary_dilation(unknown, iterations=1)
        wall_adj = ndimage.binary_dilation(occupied, iterations=1)
        frontier = free & unknown_adj & ~wall_adj

        labels, n = ndimage.label(frontier, structure=np.ones((3, 3)))
        res = m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y
        out = []
        for i in range(1, n + 1):
            ys, xs = np.where(labels == i)
            if len(xs) < min_cells:
                continue
            cx = ox + (xs.mean() + 0.5) * res
            cy = oy + (ys.mean() + 0.5) * res
            out.append((cx, cy, int(len(xs))))
        return out

    def _frontier_wait(self, gh, secs):
        """Executor-safe short wait that bails on cancel (other executor threads
        keep spinning, so the /map callback still fires)."""
        end = time.monotonic() + secs
        while time.monotonic() < end:
            if gh.is_cancel_requested:
                return
            time.sleep(0.05)

    def _exec_frontier_explore(self, gh):
        req = gh.request
        run = _Run(self, gh, SkillFrontierExplore, 'frontier_explore')

        min_cells = int(req.min_frontier_cells) or _DEF_MIN_CELLS
        bl_radius = float(req.blacklist_radius) or _DEF_BLACKLIST_RADIUS
        min_dist = float(req.min_goal_distance) or _DEF_MIN_GOAL_DIST
        goal_timeout = float(req.goal_timeout) or _DEF_GOAL_TIMEOUT

        self._frontier_setup()
        if not self._frontier_nav.wait_for_server(timeout_sec=10.0):
            return run.abort('nav2 /navigate_to_pose not available')

        blacklist = []          # [(x, y), ...] map-frame goals that failed
        goals_reached = 0

        def blacklisted(x, y):
            return any(math.hypot(x - bx, y - by) < bl_radius
                       for bx, by in blacklist)

        while True:
            # phase() handles cancel + the overall skill deadline (goal.timeout).
            if not run.phase('exploring', 0.0):
                run.result.goals_reached = goals_reached
                return run.result

            m = self._frontier_map
            rp = self._frontier_robot_xy() if m is not None else None
            if m is None or rp is None:
                self._frontier_wait(gh, 0.5)   # wait for /map + TF, then retry
                continue

            cand = [(x, y, s) for (x, y, s) in self._frontiers(m, min_cells)
                    if not blacklisted(x, y)
                    and math.hypot(x - rp[0], y - rp[1]) > min_dist]
            if not cand:
                run.result.goals_reached = goals_reached
                return run.succeed(
                    f'map fully explored ({goals_reached} frontier(s) reached)')

            # Nearest sizeable frontier (simplest sensible policy).
            gx, gy, sz = min(cand, key=lambda c: math.hypot(c[0] - rp[0],
                                                            c[1] - rp[1]))
            run.feedback.frontiers_remaining = len(cand)
            if not run.phase('navigating', 0.5):
                run.result.goals_reached = goals_reached
                return run.result

            yaw = math.atan2(gy - rp[1], gx - rp[0])   # face the frontier
            ps = PoseStamped()
            ps.header.frame_id = _MAP_FRAME
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x, ps.pose.position.y = float(gx), float(gy)
            ps.pose.orientation.z = math.sin(yaw / 2.0)
            ps.pose.orientation.w = math.cos(yaw / 2.0)
            goal = NavigateToPose.Goal()
            goal.pose = ps

            self.get_logger().info(
                f'[frontier_explore] -> ({gx:.2f}, {gy:.2f}) [{sz} cells, '
                f'{math.hypot(gx - rp[0], gy - rp[1]):.1f} m]; '
                f'{len(blacklist)} blacklisted')
            # Bound the single-goal wait by both goal_timeout and the skill's
            # remaining budget so we never blow past the overall deadline.
            resp = self._send_action(
                self._frontier_nav, goal, outer_gh=gh,
                result_timeout=min(goal_timeout, run.remaining()))
            if resp is None or resp.status != GoalStatus.STATUS_SUCCEEDED:
                blacklist.append((gx, gy))   # unreachable / timed out
            else:
                goals_reached += 1
