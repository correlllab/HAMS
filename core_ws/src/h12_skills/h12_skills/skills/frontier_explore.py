"""SkillFrontierExplore: autonomous frontier-based map exploration.

Drives the base toward successive frontiers of the slam_toolbox /map — FREE cells
that touch UNKNOWN space — via nav2's /navigate_to_pose, until the map is fully
explored (or the goal timeout is hit). Policy is deliberately simple:
nearest-sizeable-frontier with a failure blacklist, no information-gain scoring.

ABYSS SAFETY: the goal sent to nav2 is NOT the frontier cell itself. A frontier
sits on the very edge of known space, and in RoboCasa the doorways open onto an
endless void the (downward) lidar cannot see as an obstacle — driving the base
onto that edge walks it off the world. Instead each goal is *backed off* into
known-free space by _safe_goal(): the point on the robot->frontier approach ray
that lies at least _STANDOFF metres inside the free region (clear of both
unknown and walls), oriented to FACE the frontier so the forward lidar still
sweeps the unknown area from safe ground. Pair this with the nav2 planner's
allow_unknown:=false during exploration so no path routes across the void.

Ported from the standalone mac-sim frontier_explore node into the skills mixin so
it runs as the /skill/frontier_explore action alongside the other skills, reusing
SkillsBase's TF buffer + executor-safe action plumbing (_send_action). Requires
nav2 + slam_toolbox up and the robot able to follow nav2's /cmd_vel — under the
switchable lower-body controller (lowerbody:=almi, auto_switch on) ALMI stands at
a waypoint and hands off to the walk policy the moment nav2 commands motion.
"""
import math
import time

import numpy as np
from scipy import ndimage

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from visualization_msgs.msg import Marker, MarkerArray
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
# RViz visualization: one SPHERE marker per detected frontier centroid.
_MARKER_TOPIC = '/frontier_explore/frontiers'

# Abyss-safety standoff: keep the goal at least this far (metres) from the
# nearest UNKNOWN cell (the void beyond doorways the lidar can't see as an
# obstacle). Applied to unknown ONLY — occupied walls are known geometry that
# nav2's own costmap inflation keeps the base off, and requiring a big standoff
# from walls too leaves no viable cell in a tight kitchen. The primary safety is
# allow_unknown:=false on the planner (paths never route through the void); this
# standoff only has to keep the *parked* base (center within the 0.25 m goal
# tolerance, ~0.28 m half-extent) from overhanging the edge. In a freshly-seen
# pocket the deepest known-free point can be <0.45 m from unknown, so 0.35 m is
# the largest value that still lets the robot advance and reveal more.
# Graduated abyss standoffs (m), tried largest-first. In a freshly-seen pocket
# the deepest known-free point can be under 0.45 m from unknown, so a single
# 0.35 m standoff often yields NO valid goal and the skill quits with the map
# barely explored. Falling back to smaller margins keeps the robot moving (and
# revealing more, which then re-opens the larger margins). Door-frame frontiers
# — the actual abyss risk — are excluded by ~wall_adj in _frontiers() at EVERY
# level, so the reduced margins only ever apply to open-floor boundaries where
# the unknown beyond is just unmapped floor.
_STANDOFFS = (0.35, 0.25, 0.18, 0.12)
_STANDOFF = _STANDOFFS[0]
# Small margin (m) from occupied walls, just enough that the goal lands in
# low-cost costmap space nav2 will accept (not jammed against a wall).
_WALL_MARGIN = 0.20
# nav2's controller goal-checker xy tolerance — MUST mirror
# controller_server.goal_checker.xy_goal_tolerance in
# h1_bringup/config/nav2_config.yaml. nav2 reports a goal SUCCEEDED the instant
# the base is within this radius of it, commanding no motion. Kept here so the
# advance gate below can guarantee every goal we issue sits OUTSIDE it.
_NAV_XY_GOAL_TOL = 0.25
# A safe backed-off goal nearer than this to the robot is one nav2 would treat
# as already-reached (within _NAV_XY_GOAL_TOL) — issuing it makes nav2 report
# instant success with zero motion, and since the robot never moves the same
# frontier is re-selected forever (a silent busy-loop). So require every goal to
# sit at least the goal tolerance PLUS a margin away, guaranteeing nav2 actually
# commands motion; a frontier whose deepest safe cell is nearer than this is
# blacklisted instead. MUST exceed _NAV_XY_GOAL_TOL — a smaller value guards
# nothing (this was 0.15 < 0.25, the cause of the instant-success loop).
_MIN_SAFE_ADVANCE = _NAV_XY_GOAL_TOL + 0.10
# Below this base displacement (m) over one nav attempt we treat the robot as
# not having moved at all — used to catch a nav2 "SUCCEEDED" that advanced the
# base nowhere (belt-and-braces if the tolerances above ever drift apart again).
_STUCK_MOVE = 0.05
# Newly-known map cells during one nav attempt above which the attempt counts as
# "made progress" even if nav2 didn't formally reach the goal. With a slow gait
# the robot rarely arrives within the goal timeout, but it reveals map while
# walking toward the frontier — so blacklist a frontier only when an attempt
# reveals essentially NOTHING new (genuinely stuck), not merely because it was
# slow. Prevents premature "fully explored" from timeout-blacklisting.
_PROGRESS_CELLS = 15
# Consecutive rounds where frontiers exist but none is safely reachable before
# the skill gives up (each round waits ~1 s for the map to grow). Distinguishes a
# transient (map still forming) from a genuinely stuck/boxed-in robot.
_MAX_STUCK_ROUNDS = 15


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
        self._frontier_markers = self.create_publisher(
            MarkerArray, _MARKER_TOPIC, 10)
        self._frontier_ready = True

    def _on_frontier_map(self, msg):
        self._frontier_map = msg

    def _known_cells(self, m):
        """Count of explored (non-unknown) cells in an OccupancyGrid — the map
        'size' proxy used to tell real exploration progress from a stuck goal."""
        if m is None:
            return 0
        return int((np.array(m.data, dtype=np.int16) >= 0).sum())

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

    def _publish_frontier_markers(self, frontiers, selected=None):
        """Publish one SPHERE marker per frontier centroid (map frame) to
        ``_MARKER_TOPIC`` for RViz. The currently-targeted frontier is drawn
        larger and green, the rest cyan. A leading DELETEALL clears the previous
        set so stale spheres never linger after the frontiers move."""
        arr = MarkerArray()
        wipe = Marker()
        wipe.header.frame_id = _MAP_FRAME
        wipe.action = Marker.DELETEALL
        arr.markers.append(wipe)
        stamp = self.get_clock().now().to_msg()
        for i, (fx, fy, *_rest) in enumerate(frontiers):
            is_sel = (selected is not None
                      and abs(fx - selected[0]) < 1e-6
                      and abs(fy - selected[1]) < 1e-6)
            mk = Marker()
            mk.header.frame_id = _MAP_FRAME
            mk.header.stamp = stamp
            mk.ns = 'frontiers'
            mk.id = i
            mk.type = Marker.SPHERE
            mk.action = Marker.ADD
            mk.pose.position.x = float(fx)
            mk.pose.position.y = float(fy)
            mk.pose.position.z = 0.15
            mk.pose.orientation.w = 1.0
            mk.scale.x = mk.scale.y = mk.scale.z = 0.35 if is_sel else 0.22
            mk.color.a = 0.9
            mk.color.r, mk.color.g, mk.color.b = (
                (0.1, 1.0, 0.2) if is_sel else (0.1, 0.6, 1.0))
            arr.markers.append(mk)
        self._frontier_markers.publish(arr)

    def _clear_frontier_markers(self):
        """Remove all frontier spheres (called when the skill exits)."""
        arr = MarkerArray()
        wipe = Marker()
        wipe.header.frame_id = _MAP_FRAME
        wipe.action = Marker.DELETEALL
        arr.markers.append(wipe)
        self._frontier_markers.publish(arr)

    def _safe_goal(self, m, fx, fy, rp, standoffs=_STANDOFFS, wall_margin=_WALL_MARGIN):
        """Back a frontier goal off into known-free space.

        Returns (gx, gy, standoff_used) in the map frame: a cell that is
        confidently free, at least ``standoff_used`` metres from any UNKNOWN
        cell (the abyss) AND at least ``wall_margin`` from any OCCUPIED cell (so
        nav2 accepts the goal), chosen as the first such cell walking from the
        frontier centroid (fx, fy) back toward the robot (rp) — the point on the
        approach ray closest to the frontier that is still safely inside known
        space. ``standoffs`` is tried largest-first; the first value that yields
        any safe cell wins, so tight pockets get a smaller (but still positive)
        margin instead of no goal at all. Returns None only if even the smallest
        standoff has no safe cell (caller blacklists the frontier). The robot
        came FROM known-free space, so the ray back toward it is the natural
        safe-corridor direction."""
        g = np.array(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
        H, W = g.shape
        res = m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y

        free_known = (g >= 0) & (g < 25)
        unknown = g < 0
        occupied = g >= 65
        # Separate euclidean distances: the big standoff is from the void
        # (unknown); a small margin keeps the goal off walls (occupied).
        d_unknown = ndimage.distance_transform_edt(~unknown) * res
        d_occ = ndimage.distance_transform_edt(~occupied) * res
        off_walls = free_known & (d_occ >= wall_margin)

        fcx, fcy = (fx - ox) / res, (fy - oy) / res
        rcx, rcy = (rp[0] - ox) / res, (rp[1] - oy) / res
        steps = int(math.hypot(rcx - fcx, rcy - fcy)) + 1
        for standoff in standoffs:
            safe = off_walls & (d_unknown >= standoff)
            if not safe.any():
                continue
            for i in range(steps + 1):
                t = i / steps
                cx = int(round(fcx + t * (rcx - fcx)))
                cy = int(round(fcy + t * (rcy - fcy)))
                if 0 <= cy < H and 0 <= cx < W and safe[cy, cx]:
                    return ox + (cx + 0.5) * res, oy + (cy + 0.5) * res, standoff
            ys, xs = np.where(safe)            # ray missed: nearest safe to centroid
            j = int(((xs - fcx) ** 2 + (ys - fcy) ** 2).argmin())
            return ox + (xs[j] + 0.5) * res, oy + (ys[j] + 0.5) * res, standoff
        return None

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
        stuck_rounds = 0        # consecutive rounds with frontiers but none reachable

        def blacklisted(x, y):
            return any(math.hypot(x - bx, y - by) < bl_radius
                       for bx, by in blacklist)

        while True:
            # phase() handles cancel + the overall skill deadline (goal.timeout).
            if not run.phase('exploring', 0.0):
                run.result.goals_reached = goals_reached
                self._clear_frontier_markers()
                return run.result

            m = self._frontier_map
            rp = self._frontier_robot_xy() if m is not None else None
            if m is None or rp is None:
                self._frontier_wait(gh, 0.5)   # wait for /map + TF, then retry
                continue

            raw = self._frontiers(m, min_cells)
            cand = [(x, y, s) for (x, y, s) in raw
                    if not blacklisted(x, y)
                    and math.hypot(x - rp[0], y - rp[1]) > min_dist]

            # Choose the NEAREST candidate whose safe backed-off goal is far
            # enough to actually command base motion (>= _MIN_SAFE_ADVANCE).
            # A candidate whose deepest safe goal is too close — the robot sits in
            # a small known pocket it can't back off from yet — is DEFERRED, not
            # blacklisted: it becomes reachable once the robot advances elsewhere
            # and reveals more free space. Permanent blacklisting is reserved for
            # frontiers nav actually failed to progress on (below). This is what
            # stops a fresh/boxed-in spawn from blacklisting every frontier and
            # then falsely reporting "fully explored".
            target = None
            deferred = 0
            for (fx, fy, sz) in sorted(
                    cand, key=lambda c: math.hypot(c[0] - rp[0], c[1] - rp[1])):
                safe = self._safe_goal(m, fx, fy, rp)
                if safe is None or math.hypot(safe[0] - rp[0],
                                              safe[1] - rp[1]) < _MIN_SAFE_ADVANCE:
                    deferred += 1
                    continue
                target = (fx, fy, sz, safe)
                break

            # Visualize every candidate frontier in RViz, highlighting the target.
            self._publish_frontier_markers(
                cand, selected=(target[0], target[1]) if target else None)

            if target is None:
                # Nothing actionable this round. Only an empty raw set means the
                # map is genuinely explored — never report that when frontiers
                # remain but were merely filtered out (the false-"explored" bug).
                if not raw:
                    run.result.goals_reached = goals_reached
                    self._clear_frontier_markers()
                    return run.succeed(
                        f'map fully explored ({goals_reached} frontier(s) reached)')
                n_bl = sum(1 for (x, y, s) in raw if blacklisted(x, y))
                n_near = sum(1 for (x, y, s) in raw
                             if not blacklisted(x, y)
                             and math.hypot(x - rp[0], y - rp[1]) <= min_dist)
                self.get_logger().warn(
                    f'[frontier_explore] {len(raw)} frontier(s) remain but none '
                    f'actionable from ({rp[0]:.2f}, {rp[1]:.2f}): {n_bl} blacklisted, '
                    f'{n_near} within min_goal_distance ({min_dist:.2f} m), '
                    f'{deferred} with no safe standoff >= {_MIN_SAFE_ADVANCE:.2f} m. '
                    f'Robot needs to advance to reveal more free space.')
                stuck_rounds += 1
                if stuck_rounds >= _MAX_STUCK_ROUNDS:
                    run.result.goals_reached = goals_reached
                    self._clear_frontier_markers()
                    return run.abort(
                        f'no safely-reachable frontier ({len(raw)} remain: '
                        f'{n_bl} blacklisted, {n_near} too close, {deferred} boxed '
                        f'in). Robot may not be moving — check the lower-body '
                        f'controller is following /cmd_vel.')
                self._frontier_wait(gh, 1.0)   # give the map a moment to grow
                continue
            stuck_rounds = 0

            fx, fy, sz, safe = target
            gx, gy, standoff_used = safe

            run.feedback.frontiers_remaining = len(cand)
            if not run.phase('navigating', 0.5):
                run.result.goals_reached = goals_reached
                self._clear_frontier_markers()
                return run.result

            # Face the frontier FROM the backed-off goal, so on arrival the
            # forward lidar sweeps the unknown area beyond it.
            yaw = math.atan2(fy - gy, fx - gx)
            ps = PoseStamped()
            ps.header.frame_id = _MAP_FRAME
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x, ps.pose.position.y = float(gx), float(gy)
            ps.pose.orientation.z = math.sin(yaw / 2.0)
            ps.pose.orientation.w = math.cos(yaw / 2.0)
            goal = NavigateToPose.Goal()
            goal.pose = ps

            self.get_logger().info(
                f'[frontier_explore] frontier ({fx:.2f}, {fy:.2f}) [{sz} cells] '
                f'-> safe goal ({gx:.2f}, {gy:.2f}) @ {standoff_used:.2f}m standoff, '
                f'{math.hypot(gx - rp[0], gy - rp[1]):.1f} m; '
                f'{len(blacklist)} blacklisted')
            # Bound the single-goal wait by both goal_timeout and the skill's
            # remaining budget so we never blow past the overall deadline.
            known_before = self._known_cells(m)
            resp = self._send_action(
                self._frontier_nav, goal, outer_gh=gh,
                result_timeout=min(goal_timeout, run.remaining()))
            revealed = self._known_cells(self._frontier_map) - known_before
            rp_after = self._frontier_robot_xy() or rp
            moved = math.hypot(rp_after[0] - rp[0], rp_after[1] - rp[1])
            succeeded = resp is not None and resp.status == GoalStatus.STATUS_SUCCEEDED
            if succeeded and (moved >= _STUCK_MOVE or revealed >= _PROGRESS_CELLS):
                goals_reached += 1
            elif succeeded:
                # nav2 reported SUCCEEDED but the base neither moved nor revealed
                # new map: the goal was within nav2's xy_goal_tolerance of the
                # start pose, so nav commanded nothing. Blacklist the frontier so
                # it isn't re-selected forever (the busy-loop this guards). The
                # _MIN_SAFE_ADVANCE gate above should prevent this; this is the
                # backstop if the tolerances drift or nav2 misreports.
                blacklist.append((fx, fy))
                self.get_logger().warn(
                    f'[frontier_explore] blacklisting ({fx:.2f}, {fy:.2f}) '
                    f'— nav succeeded without motion (moved {moved:.2f} m, '
                    f'{revealed} new cells); goal was inside nav goal tolerance')
            elif revealed < _PROGRESS_CELLS:
                # Timed out / aborted AND revealed no new map -> genuinely stuck
                # on this frontier; blacklist so we stop retrying it.
                blacklist.append((fx, fy))
                self.get_logger().info(
                    f'[frontier_explore] blacklisting ({fx:.2f}, {fy:.2f}) '
                    f'— no progress ({revealed} new cells)')
            else:
                # Slow but progressing: the robot revealed map walking toward the
                # frontier. Leave it un-blacklisted and re-plan from the new pose.
                self.get_logger().info(
                    f'[frontier_explore] progress toward ({fx:.2f}, {fy:.2f}): '
                    f'+{revealed} cells (timed out, will re-approach)')
