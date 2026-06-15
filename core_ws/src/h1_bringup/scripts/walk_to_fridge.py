#!/usr/bin/env python3
"""Open-loop walk to the fridge, switch walk->FAME, then open it.

Nav2 is not used yet: FAST-LIO odometry diverges during the walking gait
("No Effective Points" — the sim's downward lidar gives no scan-matching
features in motion), so there is no reliable pose feedback to close the loop.
Until that is fixed this dead-reckons the approach:

  start_walk  ->  drive /cmd_vel forward (heading held via the IMU yaw) for a
  tuned distance  ->  stop  ->  start_fame (gated handover to standing)  ->
  reuse the open_fridge grasp (approach / contact / close / pull).

Run with the robot spawned far and the switchable controller:
  A) docker_run.sh mujoco --spawn far
  B) ros2 launch h1_bringup h1_sim_bringup.launch.py use_rviz:=false \
         use_sliders:=false lowerbody_node:=lowerbody_controller_node
  C) ros2 run h1_bringup walk_to_fridge.py
"""

import math
import time

import rclpy
from rclpy.duration import Duration as RclpyDuration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_srvs.srv import Trigger
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from unitree_hg.msg import LowState

# open_fridge.py is installed alongside this script, so its dir is on sys.path.
# Reusing FridgeOpener gives us move_frame_to / open_gripper / close_gripper
# without touching the (working) open_fridge demo.
from open_fridge import FridgeOpener

# --- tunables -------------------------------------------------------------
WALK_SPEED = 0.4          # m/s commanded forward (cmd_vel linear.x) — open-loop fallback
WALK_TIME = 6.5           # s of forward walking — open-loop fallback
HEADING_KP = 3.0          # rad/s per rad of yaw error (open-loop fallback)
STOP_SETTLE = 2.0         # s to stand still before switching policy
FAME_SETTLE = 5.0         # s to let FAME stabilize the stand before requesting walk

# --- closed-loop navigation (ground-truth world->pelvis TF from the sim) ------
# Goal = the manipulation spot in front of the fridge (world frame). FAST-LIO
# odom diverges during the gait, so we navigate on the sim's ground-truth pose
# (a stand-in for real odom/SLAM on hardware). P-control on /cmd_vel.
NAV_GOAL = (4.25, -1.4, math.pi / 2.0)   # (x, y, yaw) world
NAV_KP_LIN = 1.3          # cmd per metre of position error
NAV_KP_YAW = 1.5          # cmd per rad of yaw error
NAV_POS_TOL = 0.10        # m  — within this of the goal = arrived
NAV_YAW_TOL = 0.12        # rad
NAV_APPROACH_TOL = 0.30   # m  — switch from "drive toward" to fine-position
NAV_VMAX = 0.45           # max commanded forward/lateral (cmd_vel is clipped to 1)
NAV_WMAX = 0.50           # max commanded yaw rate
NAV_TIMEOUT = 90.0        # s (longer for the cross-room diagonal)

# Fridge-door handle in the pelvis frame once standing at the counter (same
# pose the band-held open_fridge demo uses for spawn y=-1.4). Valid only if the
# open-loop walk lands the robot ~on that spot.
HANDLE_PELVIS = (0.533, -0.089, 0.000)   # fallback (band-held spot) if no TF
# Fridge door handle, fixed MuJoCo world pose (low grab point,
# fridge_main_group_fridge_door_handle_1). The sim publishes ground-truth
# world->pelvis TF (far spawn), so we transform this into the LIVE pelvis frame
# and grasp the real handle wherever the open-loop walk lands the robot.
HANDLE_WORLD = (4.339, -0.816, 0.937)
GRASP_OFFSET = 0.214      # right_wrist_yaw_link is this far behind the grasp-centre


def _yaw_from_quat(q):
    """Yaw (rad) from a (w, x, y, z) quaternion."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class WalkToFridge(FridgeOpener):
    def __init__(self):
        super().__init__()
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.walk_cli = self.create_client(Trigger, "/lowerbody/start_walk")
        self.fame_cli = self.create_client(Trigger, "/lowerbody/start_fame")
        self._yaw = None
        self._active = None
        self.create_subscription(LowState, "/lowstate", self._on_lowstate, 10)
        # /lowerbody/active_policy is published LATCHED (transient-local) — match
        # it so we receive the retained value even though we subscribe after the
        # controller already committed the policy (else wait_active never fires).
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(String, "/lowerbody/active_policy", self._on_active, latched)
        self.get_logger().info("Waiting for lower-body start services...")
        self.walk_cli.wait_for_service(timeout_sec=15.0)
        self.fame_cli.wait_for_service(timeout_sec=15.0)
        self.get_logger().info("Ready.")

    def _on_lowstate(self, msg):
        self._yaw = _yaw_from_quat(list(msg.imu_state.quaternion))

    def _on_active(self, msg):
        self._active = msg.data

    def start_policy(self, name):
        cli = self.walk_cli if name == "walk" else self.fame_cli
        future = cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        res = future.result()
        ok = bool(res and res.success)
        self.get_logger().info(
            f"start_{name}: {'ok' if ok else 'FAILED'}"
            + (f" ({res.message})" if res else ""))
        return ok

    def wait_active(self, name, timeout=40.0):
        """Spin until /lowerbody/active_policy == name (band release + commit)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._active == name:
                self.get_logger().info(f"active_policy == {name}")
                return True
        self.get_logger().error(f"timed out waiting for active_policy == {name} "
                                f"(last: {self._active})")
        return False

    def walk_forward(self, duration, speed):
        """Drive /cmd_vel forward for `duration` s, holding the start heading."""
        # grab an initial yaw to hold
        t0 = time.time()
        while self._yaw is None and time.time() - t0 < 3.0:
            rclpy.spin_once(self, timeout_sec=0.1)
        yaw0 = self._yaw if self._yaw is not None else 0.0
        self.get_logger().info(
            f"walking forward {speed} m/s for {duration}s (hold yaw={yaw0:.3f})")
        twist = Twist()
        end = time.time() + duration
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)
            twist.linear.x = speed
            twist.angular.z = (-HEADING_KP * _wrap(self._yaw - yaw0)
                               if self._yaw is not None else 0.0)
            self.cmd_pub.publish(twist)
            time.sleep(0.05)
        self.stop()

    def stop(self):
        self.cmd_pub.publish(Twist())   # zero velocity
        for _ in range(5):
            self.cmd_pub.publish(Twist())
            rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.02)

    def pelvis_world_pose(self):
        """(x, y, yaw) of the pelvis in the world frame, from the sim's
        ground-truth world->pelvis TF. None if the transform isn't available."""
        try:
            tf = self.tf_buffer.lookup_transform(
                "world", "pelvis", rclpy.time.Time(),
                timeout=RclpyDuration(seconds=0.3))
        except Exception as e:
            self.get_logger().warn(f"world->pelvis TF unavailable: {e}", once=True)
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        return (t.x, t.y, _yaw_from_quat([q.w, q.x, q.y, q.z]))

    def walk_to_pose(self, gx, gy, gyaw):
        """Closed-loop nav to a world goal, walking facing the direction of
        travel (turn -> drive -> fine-position) on the ground-truth pelvis pose:
          phase 1: aim at (gx,gy) and walk forward toward it,
          phase 2: turn in place to the goal heading gyaw,
          phase 3: fine forward/lateral/yaw P-control until within tolerance.
        """
        self.get_logger().info(f"nav -> world ({gx:.2f}, {gy:.2f}, yaw={gyaw:.2f})")
        clamp = lambda v, lim: max(-lim, min(lim, v))
        end = time.time() + NAV_TIMEOUT
        phase = 1
        self.get_logger().info("nav phase 1: drive toward goal")
        dist = float("inf")
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            p = self.pelvis_world_pose()
            if p is None:
                time.sleep(0.05)
                continue
            x, y, yaw = p
            ex, ey = gx - x, gy - y
            dist = math.hypot(ex, ey)
            eyaw = _wrap(gyaw - yaw)
            t = Twist()
            if phase == 1:
                if dist < NAV_APPROACH_TOL:
                    phase = 2
                    self.get_logger().info("nav phase 2: turn to goal yaw")
                    continue
                herr = _wrap(math.atan2(ey, ex) - yaw)        # heading toward the goal
                # walk forward only while roughly aimed at the goal (cos gate)
                t.linear.x = clamp(NAV_KP_LIN * dist, NAV_VMAX) * max(0.0, math.cos(herr))
                t.angular.z = clamp(NAV_KP_YAW * herr, NAV_WMAX)
            elif phase == 2:
                if abs(eyaw) < NAV_YAW_TOL:
                    phase = 3
                    self.get_logger().info("nav phase 3: fine-position")
                    continue
                t.angular.z = clamp(NAV_KP_YAW * eyaw, NAV_WMAX)
            else:  # phase 3 — nail the pose
                if dist < NAV_POS_TOL and abs(eyaw) < NAV_YAW_TOL:
                    self.get_logger().info(
                        f"reached goal: dist={dist:.3f} m, eyaw={eyaw:.3f} rad")
                    self.stop()
                    return
                bx = ex * math.cos(yaw) + ey * math.sin(yaw)
                by = -ex * math.sin(yaw) + ey * math.cos(yaw)
                t.linear.x = clamp(NAV_KP_LIN * bx, NAV_VMAX)
                t.linear.y = clamp(NAV_KP_LIN * by, 0.3)
                t.angular.z = clamp(NAV_KP_YAW * eyaw, NAV_WMAX)
            self.cmd_pub.publish(t)
            time.sleep(0.05)
        self.get_logger().warn(f"nav timed out (phase {phase}, dist={dist:.3f} m)")
        self.stop()

    def grasp_and_open(self):
        """Grasp the fridge handle, targeting its fixed world pose transformed
        into the CURRENT pelvis frame (sim ground-truth world->pelvis TF) so it
        works wherever the open-loop walk landed. Falls back to the band-held
        pelvis pose if the TF is unavailable."""
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.1)   # let the tf listener catch up
        handle = self._transform_point(
            HANDLE_WORLD, "world", rclpy.time.Time().to_msg(), "pelvis")
        if handle is None:
            self.get_logger().warn(
                "world->pelvis TF unavailable — falling back to band-held handle pose")
            handle = HANDLE_PELVIS
        hx, hy, hz = handle
        wrist_x = hx - GRASP_OFFSET
        self.get_logger().info(
            f"=== Open fridge: handle (pelvis frame) ({hx:.3f},{hy:.3f},{hz:.3f}) "
            f"wrist_x={wrist_x:.3f} ===")
        self.open_gripper()
        self.get_clock().sleep_for(RclpyDuration(seconds=1.0))
        if not self.move_frame_to(wrist_x - 0.12, hy, hz, duration_sec=4):
            return False
        if not self.move_frame_to(wrist_x, hy, hz, duration_sec=3):
            return False
        self.close_gripper(position_mm=0.0)
        self.get_clock().sleep_for(RclpyDuration(seconds=2.0))
        for px in (wrist_x - 0.15, wrist_x - 0.32, wrist_x - 0.50):
            if not self.move_frame_to(px, hy, hz, duration_sec=2):
                return False
        return True


def main():
    rclpy.init()
    node = WalkToFridge()
    try:
        # 1. Stand on FAME first. FAME reliably catches the band-release (the
        #    proven open_fridge startup); starting straight into walk drops the
        #    robot, so we stabilize standing before ever walking.
        # 1. Stand on FAME. Launch the bringup with lowerbody_policy:=fame so the
        #    legs are FAME-driven from the start — starting idle leaves them limp
        #    and sagged, and FAME can't catch that on band release. If FAME isn't
        #    already active (e.g. launched idle), request it as a fallback.
        node.get_logger().info("=== Stand on FAME ===")
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.1)   # learn the current active policy
        if node._active != "fame" and not node.start_policy("fame"):
            node.get_logger().error("FAME start failed — aborting.")
            return
        if not node.wait_active("fame"):
            node.get_logger().error("FAME not active — aborting.")
            return
        node.get_logger().info(f"settling on FAME for {FAME_SETTLE}s before walking...")
        node.get_clock().sleep_for(RclpyDuration(seconds=FAME_SETTLE))

        # 2. Switch to walking from the stable stand (gated handover).
        node.get_logger().info("=== Switch to walking ===")
        if not node.start_policy("walk") or not node.wait_active("walk"):
            node.get_logger().error("walk policy did not engage — aborting.")
            return

        # 3. Walk to the fridge — closed-loop go-to-pose on the ground-truth
        #    pelvis pose (corrects the open-loop drift/overshoot).
        node.get_logger().info("=== Walk to the fridge (closed-loop nav) ===")
        node.walk_to_pose(*NAV_GOAL)
        # Switch to FAME promptly: at cmd=0 the walk policy marches in place and
        # drifts the base, so don't dwell here before the handover (else the
        # grasp pose shifts off the handle).
        node.get_clock().sleep_for(RclpyDuration(seconds=0.3))

        # 4. Switch back to FAME to stand still for the grasp.
        node.get_logger().info("=== Switch back to FAME standing ===")
        if not node.start_policy("fame") or not node.wait_active("fame"):
            node.get_logger().error("FAME did not engage — aborting.")
            return
        node.get_clock().sleep_for(RclpyDuration(seconds=STOP_SETTLE))

        node.grasp_and_open()
        node.get_logger().info("=== Done ===")
    except Exception as e:
        node.get_logger().error(f"Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
