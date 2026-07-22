#!/usr/bin/env python3
"""Full-body MPC grasp orchestrator (2026-07-20).

Plan: ~/Desktop/h12/docs/grasp_controller_plan_2026-07-20.md (P5). Turns a
perception 6-DOF grasp pose into a controller target stream + gripper sequence for
the full-body grasp MPC (task "Grasp H12 Magpie" on h12_control_node).

Pipeline (all frame math lives HERE in ROS/tf2, not in the MJPC C++):
  perception grasp pose (camera frame)
    -> camera->pelvis TF  -> apply TCP offset (Magpie grasp point) -> world/pelvis frame
    -> STANDOFF (10 cm back along approach axis) -> APPROACH -> HOLD -> RETREAT
    -> stream RAMPED (velocity-capped) position into the MPC via gRPC SetTaskParameters
       ("Reach Active/X/Y/Z" -- the seam lean.cc already reads, lean.h:33 graspgen path)
    -> mink IK per waypoint -> q* (torso-frame, arm sub-chain, legs frozen) = the
       terminal joint seed the deploy PD holds (dual-mode latch; orientation comes
       from q*, NOT an MPC quaternion residual -- see plan method-check)
    -> on dwell: magpie force-close, then set "Grasp Confirmed".

★The RAMPED stream is the P3 balance fix: a step target snaps the arm and tips the
robot (twin: falls 2.7-10.5 s); a velocity-capped approach injects far less momentum
(TOP paper: arm SPEED destabilizes, timing-opt 82->95%).

IK backends (ik_backend param): 'dls' (default, shipped MuJoCo-native damped-least-
squares, joint-limit-only) or 'mink' (OPT-IN QP IK adding HARD self-collision + an
optional virtual-table avoidance, and P4 collision-free path streaming).

Runtime deps (guarded imports; node degrades gracefully if missing):
  - mujoco                       : DLS + mink IK              [q* seed / reachability]
  - mink + a QP solver           : pip install mink daqp qpsolvers
                                   [OPT-IN collision-avoiding IK; ik_backend:=mink]
  - mujoco_mpc agent_pb2 gRPC    : the deploy build's proto   [target stream]
  - magpie_msgs                  : the Magpie gripper services [close]
  - custom_ros_messages          : SkillGrasp/SkillPickPlace  [action interface]
Untested on the live stack -- needs perception + robot + a running grasp MPC.
"""
import math
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped
from std_srvs.srv import Trigger                # gripper/open (release at the plate)
from rcl_interfaces.srv import SetParameters    # push object_name to the adapter
from rcl_interfaces.msg import (Parameter as _ParamMsg, ParameterValue as _ParamVal,
                                ParameterType as _ParamType)

# ---- guarded: skill actions. Reuse the split stack's SkillGrasp / SkillPickPlace
# so the SAME action client/UI drives the whole-body grasp. Interface goes OFF (the
# topic/service path still works) if custom_ros_messages is absent -- e.g. the host
# dev tree; it is container-built. ----
try:
    from custom_ros_messages.action import SkillGrasp, SkillPickPlace
    from rclpy.action import ActionServer, CancelResponse, GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    _HAVE_SKILL_ACTIONS = True
except Exception as e:  # pragma: no cover
    _HAVE_SKILL_ACTIONS = False
    _SKILL_ERR = str(e)

# ---- guarded: MJPC agent gRPC (same fallback the arm_plan_client uses) ----
try:
    import grpc
    try:
        from mujoco_mpc.proto import agent_pb2, agent_pb2_grpc
    except ImportError:
        sys.path.insert(0, os.path.expanduser(   # package root so `mujoco_mpc.proto` resolves
            "~/Desktop/h12/mujoco_mpc/mujoco_mpc/python/build/"
            "lib.linux-x86_64-cpython-310"))
        from mujoco_mpc.proto import agent_pb2, agent_pb2_grpc
    _HAVE_GRPC = True
except Exception as e:  # pragma: no cover
    _HAVE_GRPC = False
    _GRPC_ERR = str(e)

# ---- guarded: Magpie gripper service (magpie_control/gripper_node) ----
try:
    from magpie_msgs.srv import SetGripperForce
    _HAVE_MAGPIE = True
except Exception as e:  # pragma: no cover
    _HAVE_MAGPIE = False
    _MAGPIE_ERR = str(e)


def quat_xyzw_to_wxyz(q):
    """ROS/tf2 (x,y,z,w) -> MuJoCo/mink (w,x,y,z), renormalized."""
    x, y, z, w = q
    v = np.array([w, x, y, z], float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0, 0, 0])


def approach_axis_world(quat_wxyz):
    """GraspGen convention: approach = +Z of the grasp frame. Return the world +Z
    unit axis of the grasp orientation (for the standoff back-off)."""
    w, x, y, z = quat_wxyz
    # third column of R(q) = local +Z in world
    return np.array([2 * (x * z + w * y),
                     2 * (y * z - w * x),
                     1 - 2 * (x * x + y * y)], float)


def top_down_approach_vec(approach_deg, azimuth_deg):
    """Unit approach vector pointing DOWN, tilted (90 - approach_deg) off vertical toward
    azimuth_deg (deg; 180 = toward -x / the pelvis). approach_deg=90 -> straight down."""
    t = math.radians(90.0 - approach_deg)
    a = math.radians(azimuth_deg)
    v = np.array([math.sin(t) * math.cos(a), math.sin(t) * math.sin(a), -math.cos(t)])
    return v / (np.linalg.norm(v) + 1e-9)


def _rot_to_quat_wxyz(R):
    """3x3 rotation matrix -> quaternion (w, x, y, z)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q, float)
    return q / (np.linalg.norm(q) + 1e-9)


def top_down_grasp_quat(approach_deg, azimuth_deg):
    """Grasp orientation whose APPROACH axis (grasp +Z, GraspGen convention) = the
    down-tilted approach vector; the finger-close axis (grasp +X) is left horizontal.
    Returns quat_wxyz -- the IK orientation target for a top-down grasp."""
    z = top_down_approach_vec(approach_deg, azimuth_deg)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, z)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])
    x = ref - np.dot(ref, z) * z
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    return _rot_to_quat_wxyz(np.column_stack([x, y, z]))


def _default_grasp_model():
    """Grasp model for the IK solver: the hams_ros CONTAINER tree first
    (/home/code/mujoco_mpc, where the submodule is mounted + built), else the host
    h12 dev tree. Overridable via the grasp_model_path ROS param."""
    for root in ("/home/code/mujoco_mpc",
                 os.path.expanduser("~/Desktop/h12/mujoco_mpc/mujoco_mpc")):
        p = f"{root}/build/mjpc/tasks/humanoid_bench/grasp/Grasp_H12_Magpie.xml"
        if os.path.exists(p):
            return p
    return "/home/code/mujoco_mpc/build/mjpc/tasks/humanoid_bench/grasp/Grasp_H12_Magpie.xml"


class GraspOrchestrator(Node):
    STATES = ("IDLE", "STANDOFF", "APPROACH", "HOLD", "LIFT", "TRANSPORT",
              "RELEASE", "RETREAT", "DONE")

    def __init__(self):
        super().__init__("grasp_orchestrator")
        # ---- config (override via ROS params for the real cell) ----
        self.declare_parameter("grasp_pose_topic", "/graspgen/grasp_pose")
        self.declare_parameter("grpc_addr", "localhost:10000")   # full-body node monitor
        self.declare_parameter("stream_hz", 50.0)
        self.declare_parameter("approach_speed", 0.05)   # m/s  <- the velocity CAP (P3);
        # halved from 0.10 -> 0.05 for gentler momentum (arm SPEED is what tips the
        # reach; slower = safer for balance + more precise seating). Tune per run.
        self.declare_parameter("standoff_m", 0.10)       # back along approach axis
        self.declare_parameter("dwell_err_m", 0.03)      # settle threshold
        self.declare_parameter("dwell_time_s", 0.4)      # sustained-below-thresh to advance
        self.declare_parameter("tcp_offset_m", [0.17, 0.0, 0.0])  # site->Magpie grasp point
        # ---- TOP-DOWN approach (opt-in) ----------------------------------------
        # Force a near-vertical grasp: approach from ABOVE, jaws descend onto the part,
        # instead of the perception grasp's native (often side) approach. Standard
        # tabletop result -- top-down minimises collision with the support surface (the
        # table). approach_deg = angle from HORIZONTAL (90 = straight down; 82 = the small
        # tilt-from-vertical that keeps the H1-2 wrist in range). azimuth = tilt direction
        # (deg; 180 = toward the pelvis, a natural forearm pose). cone = the accept cone
        # the adapter uses to prefer already-vertical GraspGen candidates.
        self.declare_parameter("top_down", False)
        self.declare_parameter("top_down_approach_deg", 82.0)   # from horizontal; 90=vertical
        self.declare_parameter("top_down_azimuth_deg", 180.0)   # tilt toward pelvis
        self.declare_parameter("top_down_cone_deg", 45.0)       # adapter accept-cone
        # camera->pelvis extrinsic (fill from hand-eye calib; identity = pose already
        # in pelvis frame, e.g. h12_skills publishes pelvis-frame per lean.h:33)
        self.declare_parameter("cam_to_pelvis_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("cam_to_pelvis_quat_wxyz", [1.0, 0.0, 0.0, 0.0])
        self.declare_parameter("grasp_model_path", _default_grasp_model())
        self.declare_parameter("grasp_side", "right")
        # ---- IK backend (opt-in QP with collision avoidance) --------------------
        # 'dls' (default, shipped, validated) = joint-limit-only DLS, NO collision
        # avoidance. 'mink' = QP IK adding HARD self-collision (+ optional virtual
        # table) avoidance; needs `pip install mink daqp qpsolvers` in the container.
        # UNTESTED on the live stack -- validate headless -> twin -> crane first.
        self.declare_parameter("ik_backend", "dls")           # 'dls' | 'mink'
        self.declare_parameter("ik_min_dist_m", 0.02)         # collision buffer (mink)
        self.declare_parameter("ik_detect_dist_m", 0.10)      # start avoiding within (mink)
        self.declare_parameter("ik_qp_solver", "daqp")        # qpsolvers backend (mink)
        # P3 virtual table: pelvis-frame Z of the table top. < 0 (default) = no table
        # (self-collision only). Set it (e.g. from the perceived object/plate height)
        # to steer the reach AROUND the table instead of through it.
        # Real ARPA battery-cell surface = 0.87 m (measured). Baked as the default so
        # the mink virtual table sits at the real height when collision-avoidance is on.
        # (The ARPA model puts the screw-mounting surface ~0.84 m; 0.87 is the measured
        # value -- verify the pelvis-frame Z on the crane before trusting it.) -1 = off.
        self.declare_parameter("table_height_m", 0.87)
        self.declare_parameter("table_center_xy", [0.45, -0.15])  # table box centre (pelvis xy)
        # P4 path mode: stream the mink collision-free waypoints (not the straight
        # ramp) whenever they deviate from the straight line by > this [m]. 0/neg =
        # OFF (always straight ramp). Only meaningful with ik_backend:=mink.
        self.declare_parameter("ik_path_deviation_m", 0.0)
        self.declare_parameter("grip_force_n", 30.0)          # match h12_skills close (GRIP_FORCE_N)
        self.declare_parameter("gripper_force_srv", "gripper/set_force")
        self.declare_parameter("gripper_open_srv", "gripper/open")
        # OPT-IN place: a pelvis-frame [x,y,z] plate point. Empty (default) => v1
        # behaviour (grasp, close, STOP -- no lift). Set it => after grasp the robot
        # lifts, carries above the plate, and opens the gripper. ⚠ balance past
        # 'close' (payload CoM shift + lift-off transient) is UNVALIDATED -- crane only.
        self.declare_parameter("place_point", [])             # [x,y,z] pelvis, or [] = off
        self.declare_parameter("lift_height", 0.10)           # raise this far after grasp [m]
        self.declare_parameter("place_hover_m", 0.10)         # hover this high over the plate [m]
        self.declare_parameter("grip_settle_s", 1.0)          # let the jaws move before advancing

        self.addr = self.get_parameter("grpc_addr").value
        self.stub = None
        if _HAVE_GRPC:
            self.stub = agent_pb2_grpc.AgentStub(grpc.insecure_channel(self.addr))
        else:
            self.get_logger().warn(f"gRPC unavailable ({_GRPC_ERR}); DRY-RUN")

        _table_on = float(self.get_parameter("table_height_m").value) >= 0.0
        self.ik = _make_ik_solver(                       # None if mujoco/model missing
            self.get_logger(), self.get_parameter("grasp_model_path").value,
            side=self.get_parameter("grasp_side").value,
            backend=str(self.get_parameter("ik_backend").value),
            table_pose_fn=(self._table_pose if _table_on else None),
            min_dist=float(self.get_parameter("ik_min_dist_m").value),
            detect_dist=float(self.get_parameter("ik_detect_dist_m").value),
            qp_solver=str(self.get_parameter("ik_qp_solver").value),
            # top-down needs the wrist to actually point down -> weight orientation up
            ori_cost=(0.8 if bool(self.get_parameter("top_down").value) else 0.2))
        self.qstar = None                                # latest terminal seed
        self._reach_path = None                          # P4: mink collision-free waypoints
        self._path_idx = 0
        self._place_enabled = False
        self._place_standoff = None                      # hover point above the plate
        self._place_point_topic = None                   # camera-detected plate centroid (pelvis)
        # Magpie gripper clients (force-close on grasp; Trigger open at the plate)
        self._grip_cli = None
        if _HAVE_MAGPIE:
            self._grip_cli = self.create_client(
                SetGripperForce, self.get_parameter("gripper_force_srv").value)
        else:
            self.get_logger().warn(f"magpie_msgs unavailable ({_MAGPIE_ERR}); gripper DRY-RUN")
        self._grip_open_cli = self.create_client(        # std_srvs/Trigger (always available)
            Trigger, self.get_parameter("gripper_open_srv").value)

        self.state = "IDLE"
        self._lock = threading.Lock()
        self.goal_world = None     # (pos[3], quat_wxyz[4]) final grasp, pelvis/world frame
        self.standoff_world = None
        self.cur = None            # currently-streamed target position (ramped)
        self._below_since = None

        self.sub = self.create_subscription(
            PoseStamped, self.get_parameter("grasp_pose_topic").value,
            self.on_grasp_pose, 1)
        # camera-detected drop point (plate centroid, pelvis) from graspgen_pose_adapter
        self.create_subscription(PointStamped, "/place_point", self._on_place_point, 1)
        dt = 1.0 / float(self.get_parameter("stream_hz").value)
        self.timer = self.create_timer(dt, self.tick)
        self.get_logger().info(f"grasp_orchestrator up; gRPC={self.addr} "
                               f"mink={'yes' if self.ik else 'no'}")

        # ---- action interface: /skill/grasp + /skill/pick_place ----------------
        # Reuse the split stack's SkillGrasp/SkillPickPlace so the SAME client/UI
        # that drives the FrameTask pick drives our whole-body grasp. A goal pushes
        # target_object to the adapter, triggers ONE /perceive, then streams our
        # state machine as feedback (phase/progress) until DONE. The /perceive +
        # /graspgen/grasp_pose path is unchanged -- the action just wraps it.
        self.declare_parameter("perceive_srv", "/graspgen_pose_adapter/perceive")
        self.declare_parameter("adapter_node", "graspgen_pose_adapter")
        self.declare_parameter("action_timeout_s", 300.0)   # used if goal.timeout=0
        self._action_active = False
        self._perceive_cli = self.create_client(
            Trigger, self.get_parameter("perceive_srv").value)
        self._adapter_setparams_cli = self.create_client(
            SetParameters,
            f"/{self.get_parameter('adapter_node').value}/set_parameters")
        if _HAVE_SKILL_ACTIONS:
            cbg = ReentrantCallbackGroup()   # so execute() can block while tick() runs
            self._grasp_action = ActionServer(
                self, SkillGrasp, "/skill/grasp",
                execute_callback=self._execute_grasp,
                goal_callback=self._handle_goal,
                cancel_callback=self._handle_cancel, callback_group=cbg)
            self._pickplace_action = ActionServer(
                self, SkillPickPlace, "/skill/pick_place",
                execute_callback=self._execute_pick_place,
                goal_callback=self._handle_goal,
                cancel_callback=self._handle_cancel, callback_group=cbg)
            self.get_logger().info(
                "action servers up: /skill/grasp (SkillGrasp) + "
                "/skill/pick_place (SkillPickPlace)")
        else:
            self.get_logger().warn(
                f"skill actions OFF (custom_ros_messages: {_SKILL_ERR}); drive via "
                "/graspgen_pose_adapter/perceive + /graspgen/grasp_pose instead")

    def _on_place_point(self, msg: PointStamped):
        # camera-detected plate drop point (pelvis). Read lock-free in on_grasp_pose
        # (atomic ref under the GIL); a slightly stale point is harmless.
        self._place_point_topic = [msg.point.x, msg.point.y, msg.point.z]
        self.get_logger().info(
            f"place point (camera): ({msg.point.x:.3f}, {msg.point.y:.3f}, "
            f"{msg.point.z:.3f}) {msg.header.frame_id!r}")

    def _table_pose(self):
        """P3: virtual-table box CENTRE (pelvis frame) for the IK collision model.
        table_height_m is the table TOP surface z; the 0.02-half-height box is placed
        so its top sits there. Returns None when disabled (height < 0)."""
        h = float(self.get_parameter("table_height_m").value)
        if h < 0.0:
            return None
        cx, cy = [float(v) for v in self.get_parameter("table_center_xy").value]
        return [cx, cy, h - 0.02]

    def _next_path_target(self):
        """P4: the current waypoint along the stored collision-free path and whether
        it is the last one (tick advances the index)."""
        path = self._reach_path
        idx = min(self._path_idx, len(path) - 1)
        return np.array(path[idx], float), idx >= len(path) - 1

    # ---- perception in: transform + plan the standoff, arm the sequence -------
    def on_grasp_pose(self, msg: PoseStamped):
        p = msg.pose.position
        o = msg.pose.orientation
        pos_cam = np.array([p.x, p.y, p.z], float)
        quat_wxyz = quat_xyzw_to_wxyz([o.x, o.y, o.z, o.w])
        # cam->pelvis extrinsic: IDENTITY by default because graspgen_pose_adapter
        # publishes the grasp pose already in the pelvis frame (tf2). Kept a ROS param
        # for a direct-camera setup that skips the adapter.
        R = _quat_to_R(np.array(
            self.get_parameter("cam_to_pelvis_quat_wxyz").value, float))
        t = np.array(self.get_parameter("cam_to_pelvis_xyz").value, float)
        pos = R @ pos_cam + t
        # TCP offset: perception targets the OBJECT grasp point; the reach residual
        # regulates the hand SITE, so shift the target back by the site->TCP vector.
        tcp = np.array(self.get_parameter("tcp_offset_m").value, float)
        if bool(self.get_parameter("top_down").value):
            # TOP-DOWN: override the (often side) perception approach with a near-vertical
            # one -- approach FROM ABOVE so the jaws never clip the table. All geometry is
            # taken from the explicit down vector, so it is self-consistent regardless of
            # the perception grasp's native frame convention. The overridden quat is the
            # IK orientation target (best-effort -- the H1-2 wrist is limited, so the
            # standoff-above is the dominant lever that brings the arm over the part).
            ad = float(self.get_parameter("top_down_approach_deg").value)
            az = float(self.get_parameter("top_down_azimuth_deg").value)
            adir = top_down_approach_vec(ad, az)
            quat_wxyz = top_down_grasp_quat(ad, az)
            pos = pos - adir * float(np.linalg.norm(tcp))   # hand site behind the jaws
            axis = adir                                     # down => standoff goes ABOVE
            self.get_logger().info(f"top-down: approach {ad:.0f} deg from horizontal, "
                                   f"standoff above {pos.round(3)}")
        else:
            pos = pos - _quat_to_R(quat_wxyz) @ tcp
            axis = approach_axis_world(quat_wxyz)
        standoff = pos - float(self.get_parameter("standoff_m").value) * axis
        # P4 (opt-in, mink only): precompute the collision-free hand path to the grasp
        # and stream it INSTEAD of the straight ramp -- but only if it actually bends
        # around something (deviates from the straight line by > the threshold), else
        # the straight ramp is identical and cheaper.
        reach_path = None
        dev_thresh = float(self.get_parameter("ik_path_deviation_m").value)
        if dev_thresh > 0.0 and self.ik is not None:
            try:
                res = self.ik(pos, quat_wxyz, seed_q=self.qstar, return_path=True)
                _ok, full = res[3], res[4]
                if _ok and full and _straight_deviation(full, pos) > dev_thresh:
                    reach_path = _downsample_path(full, 8)
                    self.get_logger().info(
                        f"path mode: {len(reach_path)} collision-free waypoints "
                        f"(bend {_straight_deviation(full, pos) * 100:.0f} cm)")
            except Exception as e:
                self.get_logger().warn(f"path plan failed ({e}); straight ramp")
        # OPT-IN place target: prefer the camera-detected /place_point (plate
        # centroid), else the static place_point param. Hover = point + place_hover_m.
        topic_pt = self._place_point_topic
        place_pt = (list(topic_pt) if topic_pt is not None
                    else list(self.get_parameter("place_point").value or []))
        place_enabled = len(place_pt) == 3
        place_standoff = (np.array(place_pt, float)
                          + np.array([0.0, 0.0, float(self.get_parameter("place_hover_m").value)])
                          ) if place_enabled else None
        with self._lock:
            self.goal_world = (pos, quat_wxyz)
            self.standoff_world = standoff
            self._place_enabled = place_enabled
            self._place_standoff = place_standoff
            self._reach_path = reach_path      # P4: None => straight ramp
            self._path_idx = 0
            self.cur = self._current_hand_or(standoff)
            self.state = "STANDOFF"
            self._below_since = None
        self.get_logger().info(
            f"grasp armed: goal={pos.round(3)} standoff={standoff.round(3)}"
            + (f" -> place hover={place_standoff.round(3)}" if place_enabled
               else " (v1: no lift/place)"))

    # ---- 50 Hz control tick: ramp the target, advance the state machine -------
    def tick(self):
        with self._lock:
            state, goal, standoff, cur = (self.state, self.goal_world,
                                          self.standoff_world, self.cur)
            psa = self._place_standoff
            path_on = self._reach_path is not None
        if state in ("IDLE", "DONE") or goal is None:
            return
        lift_h = float(self.get_parameter("lift_height").value)
        lifted = goal[0] + np.array([0.0, 0.0, lift_h])   # straight up from the grasp
        hover = psa if psa is not None else standoff       # above the plate (place path)
        # P4: during STANDOFF, follow the collision-free waypoints if we have them;
        # every other state (and the default, path-off) is the straight ramp as before.
        if path_on and state == "STANDOFF":
            target, is_last = self._next_path_target()
        else:
            target = {"STANDOFF": standoff, "APPROACH": goal[0], "HOLD": goal[0],
                      "LIFT": lifted, "TRANSPORT": hover, "RELEASE": hover,
                      "RETREAT": lifted}.get(state, standoff)
            is_last = True
        # RAMP: move `cur` toward `target` at <= approach_speed (the velocity cap)
        step = float(self.get_parameter("approach_speed").value) / \
            float(self.get_parameter("stream_hz").value)
        cur, reached = _ramp(cur, target, step)
        self._stream_target(cur, goal[1], arm_hold=(state == "HOLD"))
        # dwell test at the true hand error would need lowstate; here we gate on the
        # commanded ramp reaching the waypoint (a conservative proxy for the demo).
        with self._lock:
            self.cur = cur
            if reached:
                if path_on and state == "STANDOFF" and not is_last:
                    self._path_idx += 1               # next waypoint, stay in STANDOFF
                else:
                    if state == "STANDOFF":
                        self._reach_path = None        # path consumed -> straight from here
                    self._advance_locked(state, goal)

    def _advance_locked(self, state, goal):
        now = self.get_clock().now().nanoseconds * 1e-9
        dwell = float(self.get_parameter("dwell_time_s").value)
        settle = float(self.get_parameter("grip_settle_s").value)
        if state == "STANDOFF":
            self.state = "APPROACH"
        elif state == "APPROACH":
            if self._below_since is None:
                self._below_since = now
            elif now - self._below_since >= dwell:
                self.get_logger().info("dwell -> HOLD: arm latch (freeze arm at q*) + close gripper")
                # ARM the deploy latch first so the right arm is rock-steady at the
                # precise IK q* before the jaws move, THEN force-close the Magpie.
                self._set_params(self._arm_hold_params(active=True))
                self._close_gripper()
                self._below_since = now          # reuse as the grip-settle timer
                self.state = "HOLD"
        elif state == "HOLD":
            # let the jaws finish closing, then STOP (v1) or lift+place (opt-in)
            if now - (self._below_since or now) >= settle:
                if self._place_enabled:
                    self.get_logger().info(
                        "grip settled -> LIFT (disarm latch; MPC carries the part)")
                    self._set_params(self._arm_hold_params(active=False))   # arm back to MPC
                    self._below_since = None
                    self.state = "LIFT"
                else:
                    self.state = "DONE"
                    self.get_logger().info("grasp complete (v1: reach+close, no lift)")
        elif state == "LIFT":
            self.state = "TRANSPORT"             # at the lifted pose -> carry to the plate
        elif state == "TRANSPORT":
            if self._below_since is None:
                self._below_since = now
            elif now - self._below_since >= dwell:
                self.get_logger().info("over the plate -> RELEASE (open gripper)")
                self._open_gripper()
                self._below_since = now
                self.state = "RELEASE"
        elif state == "RELEASE":
            if now - (self._below_since or now) >= settle:
                self.state = "RETREAT"
                self.get_logger().info("released -> RETREAT")
        elif state == "RETREAT":
            self.state = "DONE"
            self.get_logger().info("place complete (grasp -> lift -> plate -> release)")

    # ---- MPC target stream (ramped position; q* seed for the terminal hold) ---
    def _stream_target(self, pos, quat_wxyz, arm_hold=False):
        # q* for the terminal PD-hold: prioritized IK places the wrist at the grasp
        # POSITION to a few mm (validated), orientation best-effort (wrist-limited).
        # Seed from the last q* for the nearest/continuous solution. The accept-gate
        # rejects grasps outside the arm/wrist envelope BEFORE the hand-off.
        if self.ik is not None:
            q, pe, oe, ok = self.ik(pos, quat_wxyz, seed_q=self.qstar)
            if ok:
                self.qstar = q     # deploy holds arm rows at this on the latch (P2 wiring)
            else:
                self.get_logger().warn(
                    f"IK reject: pos_err={pe*1e3:.0f}mm ori_err={math.degrees(oe):.0f}deg "
                    f"(grasp outside arm/wrist envelope)")
        params = {"Reach Active": 1.0, "Reach X": float(pos[0]),
                  "Reach Y": float(pos[1]), "Reach Z": float(pos[2])}
        # Stage the latest IK q* into the node's arm-hold params every tick so the
        # deploy latch (deploy_common.cc residual "Arm Hold" + qR0..6) always has a
        # fresh seed; ARM it (Arm Hold=1 -> right arm freezes at q*) only in HOLD.
        params.update(self._arm_hold_params(active=arm_hold))
        self._set_params(params)

    def _arm_hold_params(self, active):
        """Map the current IK q* (7 right-arm joints) to the node's grasp latch
        params. Arm Hold=1 => deploy_common.cc overrides motor rows 20..26 with these."""
        if self.qstar is None:
            return {}
        p = {f"Arm Hold qR{k}": float(self.qstar[k]) for k in range(7)}
        p["Arm Hold"] = 1.0 if active else 0.0
        return p

    def _set_param(self, name, val):
        self._set_params({name: float(val)})

    def _set_params(self, params):
        if self.stub is None:
            return
        try:
            req = agent_pb2.SetTaskParametersRequest()
            for k, v in params.items():
                req.parameters[k].numeric = float(v)
            self.stub.SetTaskParameters(req, timeout=1.0)
        except Exception as e:  # pragma: no cover
            self.get_logger().warn(f"SetTaskParameters failed: {e}")

    def _close_gripper(self):
        # magpie_control: force-close via the gripper/set_force service (magpie_msgs/
        # SetGripperForce, request field max_force [N]) -- the same close h12_skills uses.
        force = float(self.get_parameter("grip_force_n").value)
        if self._grip_cli is None:
            self.get_logger().info(f"magpie: (dry-run) force-close {force:.0f} N")
            return
        if not self._grip_cli.service_is_ready():
            self.get_logger().warn("magpie: gripper/set_force not ready; skipping close")
            return
        req = SetGripperForce.Request()
        req.max_force = force
        self._grip_cli.call_async(req)
        self.get_logger().info(f"magpie: force-close {force:.0f} N")

    def _open_gripper(self):
        # release the part onto the plate: gripper/open (std_srvs/Trigger).
        if self._grip_open_cli is None or not self._grip_open_cli.service_is_ready():
            self.get_logger().info("magpie: (dry-run) open")
            return
        self._grip_open_cli.call_async(Trigger.Request())
        self.get_logger().info("magpie: open (release part)")

    def _current_hand_or(self, fallback):
        # TODO(real): seed the ramp from the measured hand pos (lowstate/FK). For the
        # dry-run we start at the standoff so the first APPROACH is the full ramp.
        return np.array(fallback, float)

    # ---- action interface (reuse SkillGrasp / SkillPickPlace) -----------------
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _handle_goal(self, goal_request):
        if self._action_active:
            self.get_logger().warn("grasp action busy; rejecting the new goal")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _handle_cancel(self, goal_handle):
        return CancelResponse.ACCEPT

    def _execute_grasp(self, goal_handle):
        return self._run_skill(goal_handle, SkillGrasp, with_place=False)

    def _execute_pick_place(self, goal_handle):
        return self._run_skill(goal_handle, SkillPickPlace, with_place=True)

    def _run_skill(self, goal_handle, ActionT, with_place):
        """Shared executor for both actions: perceive(target) -> run the state
        machine to DONE, streaming phase/progress; cancel/timeout aborts safely."""
        self._action_active = True
        req = goal_handle.request
        target = (req.target_object or "").strip()
        arm = (req.arm or "right").strip().lower()
        place_target = ((getattr(req, "place_target", "") or "").strip()
                        if with_place else "")
        timeout_s = _duration_s(req.timeout) or float(
            self.get_parameter("action_timeout_s").value)
        side = str(self.get_parameter("grasp_side").value)
        try:
            if arm and arm != side:
                self.get_logger().warn(
                    f"action arm={arm!r} but the latch/IK is {side!r}-arm only; "
                    "proceeding on the configured side")
            with self._lock:                        # fresh run
                self.state = "IDLE"
                self.goal_world = None
                self._below_since = None
            self.get_logger().info(
                f"[action] {'pick_place' if with_place else 'grasp'} target={target!r}"
                + (f" place={place_target!r}" if with_place else ""))
            # 1. push the target to the adapter + trigger ONE perceive
            ok, msg = self._perceive(target, place_target, timeout_s)
            if not ok:
                goal_handle.abort()
                return self._mkresult(ActionT, False, msg)
            # 2. wait for the grasp pose to arm the sequence (state leaves IDLE)
            t0 = self._now()
            armed = False
            while self._now() - t0 < timeout_s:
                if goal_handle.is_cancel_requested:
                    self._abort_motion(); goal_handle.canceled()
                    return self._mkresult(ActionT, False, "canceled before grasp")
                with self._lock:
                    st = self.state
                if st != "IDLE":
                    armed = True
                    break
                self._publish_fb(goal_handle, ActionT, "detect", 0.05)
                time.sleep(0.1)
            if not armed:
                self._abort_motion(); goal_handle.abort()
                return self._mkresult(ActionT, False,
                    "timeout waiting for a grasp pose (perception miss / IK reject?)")
            # 3. run the state machine to DONE, streaming feedback
            while self._now() - t0 < timeout_s:
                if goal_handle.is_cancel_requested:
                    self._abort_motion(); goal_handle.canceled()
                    return self._mkresult(ActionT, False, "canceled mid-grasp")
                with self._lock:
                    st = self.state
                phase, prog = self._phase_for(st, with_place)
                self._publish_fb(goal_handle, ActionT, phase, prog)
                if st == "DONE":
                    goal_handle.succeed()
                    return self._mkresult(ActionT, True,
                        "placed on the plate" if with_place
                        else "grasp closed (open-loop; no force-held check)")
                time.sleep(0.1)
            self._abort_motion(); goal_handle.abort()
            return self._mkresult(ActionT, False, "timeout during the grasp motion")
        finally:
            self._action_active = False

    def _perceive(self, target, place_target, timeout_s):
        """Push object_name (+place_object) to the adapter, then call its /perceive
        (Trigger). Returns (ok, message)."""
        if target:
            self._set_adapter_params(target, place_target)
        cli = self._perceive_cli
        if not cli.wait_for_service(timeout_sec=min(5.0, timeout_s)):
            return False, ("adapter /perceive unavailable "
                           "(is graspgen_pose_adapter running?)")
        fut = cli.call_async(Trigger.Request())
        t0 = self._now()
        while not fut.done():
            if self._now() - t0 > min(240.0, timeout_s):   # adapter perceive budget
                return False, "perceive timed out"
            time.sleep(0.05)
        resp = fut.result()
        if resp is None or not resp.success:
            return False, f"perceive failed: {getattr(resp, 'message', 'no response')}"
        return True, "perceived"

    def _set_adapter_params(self, target, place_target):
        """Best-effort remote set of the adapter's object_name/place_object via the
        standard SetParameters service (present on every node, any distro)."""
        cli = self._adapter_setparams_cli
        if not cli.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn("adapter set_parameters unavailable; "
                                   "perceiving with its current object_name")
            return
        req = SetParameters.Request()
        kv = {"object_name": target}
        if place_target:
            kv["place_object"] = place_target
        for name, val in kv.items():
            p = _ParamMsg()
            p.name = name
            p.value = _ParamVal(type=_ParamType.PARAMETER_STRING, string_value=str(val))
            req.parameters.append(p)
        fut = cli.call_async(req)
        t0 = self._now()
        while not fut.done() and self._now() - t0 < 3.0:
            time.sleep(0.02)

    def _abort_motion(self):
        """Safe stop on cancel/timeout: drop the reach cost, disarm the latch (arm
        back to MPC), open the gripper."""
        self._set_params({"Reach Active": 0.0, "Arm Hold": 0.0})
        self._open_gripper()
        with self._lock:
            self.state = "IDLE"
            self.goal_world = None
            self._below_since = None

    def _phase_for(self, state, with_place):
        """Map our state machine onto the action's phase/progress feedback."""
        if with_place:
            m = {"IDLE": ("detect", 0.05), "STANDOFF": ("grasp", 0.15),
                 "APPROACH": ("grasp", 0.30), "HOLD": ("grasp", 0.45),
                 "LIFT": ("carry", 0.60), "TRANSPORT": ("carry", 0.72),
                 "RELEASE": ("release", 0.85), "RETREAT": ("done", 0.95),
                 "DONE": ("done", 1.0)}
        else:
            m = {"IDLE": ("detect", 0.1), "STANDOFF": ("approach", 0.4),
                 "APPROACH": ("approach", 0.6), "HOLD": ("grasp", 0.85),
                 "DONE": ("done", 1.0)}
        return m.get(state, ("detect", 0.0))

    def _publish_fb(self, goal_handle, ActionT, phase, progress):
        fb = ActionT.Feedback()
        fb.phase = phase
        fb.progress = float(progress)
        goal_handle.publish_feedback(fb)

    def _mkresult(self, ActionT, success, message):
        r = ActionT.Result()
        r.success = bool(success)
        r.message = message
        return r


# ---------- helpers (no ROS deps) ----------
def _ramp(cur, target, step):
    cur = np.array(cur, float)
    target = np.array(target, float)
    d = target - cur
    dist = float(np.linalg.norm(d))
    if dist <= step or dist < 1e-6:
        return target.copy(), True
    return cur + (step / dist) * d, False


def _straight_deviation(path, goal):
    """P4: max perpendicular distance [m] of a hand path from the straight line
    path[0] -> goal. Large => the path bends around an obstacle; ~0 => it is straight
    and the plain ramp is equivalent."""
    a = np.asarray(path[0], float)
    b = np.asarray(goal, float)
    ab = b - a
    L = float(np.linalg.norm(ab))
    if L < 1e-6:
        return 0.0
    u = ab / L
    dmax = 0.0
    for p in path:
        p = np.asarray(p, float)
        perp = float(np.linalg.norm((p - a) - np.dot(p - a, u) * u))
        dmax = max(dmax, perp)
    return dmax


def _downsample_path(path, n):
    """P4: n roughly-even waypoints from a dense path (always keeps the last point)."""
    if len(path) <= n:
        return [np.asarray(p, float) for p in path]
    idx = np.linspace(0, len(path) - 1, n).round().astype(int)
    return [np.asarray(path[i], float) for i in idx]


def _quat_to_R(q_wxyz):
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], float)


def _make_ik_solver(logger, model_path, side="right", backend="dls",
                    table_pose_fn=None, min_dist=0.02, detect_dist=0.10,
                    qp_solver="daqp", ori_cost=0.2):
    """Dispatch to an IK backend. BOTH return the same callable
        solve(pos_world, quat_wxyz, seed_q=None, return_path=False)
          -> (q_arm[7], pos_err_m, ori_err_rad, ok)   # return_path=False
          -> (q_arm[7], pos_err_m, ori_err_rad, ok, path)  # return_path=True
    so the orchestrator is backend-agnostic (path is a list of hand xyz waypoints,
    or None for DLS which has no collision-aware path).
      - 'dls'  : the shipped MuJoCo-native prioritized damped-least-squares IK.
                 Joint limits only, NO collision avoidance. Default + fallback.
      - 'mink' : QP IK (github.com/kevinzakka/mink) adding HARD self-collision (+ an
                 optional virtual table) avoidance. Opt-in; needs mink + a QP solver
                 (`pip install mink daqp qpsolvers`) in the container. Untested on the
                 live stack -- validate headless -> twin -> crane before trusting it.
    """
    if backend == "mink":
        solve = _make_mink_solver(logger, model_path, side, table_pose_fn,
                                  min_dist, detect_dist, qp_solver, ori_cost)
        if solve is not None:
            return solve
        logger.warn("mink backend requested but unavailable; falling back to DLS IK")
    return _make_dls_solver(logger, model_path, side)


def _make_dls_solver(logger, model_path, side="right"):
    """MuJoCo-native PRIORITIZED damped-least-squares IK on the arm dofs (torso/legs/
    base frozen): position is the PRIMARY task, orientation solved only in position's
    NULLSPACE so the H1-2's limited wrist (pitch +-0.35, yaw +-1.01) can never corrupt
    the grasp position. Validated headless 2026-07-20: 1.5-4.1 mm (< the 2 cm gate);
    orientation best-effort. Joint-limit clamp only -- NO collision avoidance (use the
    'mink' backend for that)."""
    try:
        import mujoco
    except Exception as e:
        logger.warn(f"mujoco unavailable ({e}); q* seed DISABLED")
        return None
    try:
        cwd = os.getcwd(); os.chdir(os.path.dirname(model_path))
        m = mujoco.MjModel.from_xml_path(model_path)
    except Exception as e:
        logger.warn(f"grasp model load failed ({e}); q* seed DISABLED")
        return None
    finally:
        os.chdir(cwd)
    site = f"{side}_hand"
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
    # arm qpos/dof adr: right = qadr 27..33 / dof 26..32 ; left = qadr 20..26 / dof 19..25
    qadr = list(range(27, 34)) if side == "right" else list(range(20, 27))
    vadr = list(range(26, 33)) if side == "right" else list(range(19, 26))
    logger.info(f"IK solver ready ({side} arm, prioritized DLS, MuJoCo-native)")

    def solve(pos_world, quat_wxyz, seed_q=None, return_path=False,
              iters=400, damp=1e-4, step=0.6, w_ori=0.2):
        d = mujoco.MjData(m)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if kid >= 0:
            mujoco.mj_resetDataKeyframe(m, d, kid)
        if seed_q is not None:               # seed-from-current-q (nearest solution)
            for k, q in enumerate(qadr):
                d.qpos[q] = seed_q[k]
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv)); err = np.zeros(6)
        tgt = np.asarray(pos_world, float); tq = np.asarray(quat_wxyz, float)
        for _ in range(iters):
            mujoco.mj_forward(m, d)
            cq = np.zeros(4); mujoco.mju_mat2Quat(cq, d.site_xmat[sid])
            err[:3] = tgt - d.site_xpos[sid]
            do = np.zeros(3); mujoco.mju_subQuat(do, tq, cq); err[3:] = do
            if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, sid)
            Jp = jacp[:, vadr]; Jr = jacr[:, vadr]
            Jp_pinv = Jp.T @ np.linalg.solve(Jp @ Jp.T + damp * np.eye(3), np.eye(3))
            N = np.eye(len(vadr)) - Jp_pinv @ Jp
            Jr_pinv = Jr.T @ np.linalg.solve(Jr @ Jr.T + damp * np.eye(3), np.eye(3))
            dq = step * (Jp_pinv @ err[:3] + w_ori * (N @ (Jr_pinv @ err[3:])))
            for k, q in enumerate(qadr):
                d.qpos[q] += dq[k]
                for j in range(m.njnt):      # hard joint-limit clamp
                    if m.jnt_qposadr[j] == q and m.jnt_limited[j]:
                        lo, hi = m.jnt_range[j]
                        d.qpos[q] = min(max(d.qpos[q], lo), hi)
        mujoco.mj_forward(m, d)
        cq = np.zeros(4); mujoco.mju_mat2Quat(cq, d.site_xmat[sid])
        pe = float(np.linalg.norm(tgt - d.site_xpos[sid]))
        oe = np.zeros(3); mujoco.mju_subQuat(oe, tq, cq)
        out = (d.qpos[qadr].copy(), pe, float(np.linalg.norm(oe)), pe < 0.02)
        return out + (None,) if return_path else out   # DLS: no collision-aware path
    return solve


def _wrap_model_with_table(model_path, logger):
    """P3: write a temp wrapper XML that `<include>`s the grasp model (by ABSOLUTE
    path, so its own relative includes still resolve) and adds a mocap box geom
    `table_virtual`, parked far below until placed per-solve. mink adds it to the
    collision set. Returns (load_path, geom_name) or (model_path, None) on failure."""
    import tempfile
    try:
        abs_model = os.path.abspath(model_path)
        wrapper = os.path.join(tempfile.gettempdir(), "grasp_ik_table_wrapper.xml")
        with open(wrapper, "w") as f:
            f.write(
                '<mujoco model="grasp_ik_table">\n'
                f'  <include file="{abs_model}"/>\n'
                '  <worldbody>\n'
                '    <body name="table_virtual_body" mocap="true" pos="0 0 -5">\n'
                '      <geom name="table_virtual" type="box" size="0.5 0.5 0.02"'
                ' contype="1" conaffinity="1" group="3" rgba="0.9 0.2 0.2 0.15"/>\n'
                '    </body>\n'
                '  </worldbody>\n'
                '</mujoco>\n')
        return wrapper, "table_virtual"
    except Exception as e:
        logger.warn(f"virtual-table wrap failed ({e}); self-collision only")
        return model_path, None


def _make_mink_solver(logger, model_path, side, table_pose_fn, min_dist,
                      detect_dist, qp_solver, ori_cost=0.2):
    """QP differential IK via mink (kevinzakka/mink). Hard constraints the DLS lacks:
    ConfigurationLimit (joint limits) + CollisionAvoidanceLimit (arm-vs-body self-
    collision, and an optional virtual table). Arm-only motion is enforced by masking
    the solved velocity to the 7 arm dofs each step (legs/torso/base/other-arm stay at
    the 'stand' keyframe -- exactly the DLS contract, now collision-aware). Returns
    None (-> caller falls back to DLS) if mink / a QP solver / the model is missing."""
    try:
        import mink
        import mujoco
    except Exception as e:  # mink or a QP backend not installed
        logger.warn(f"mink unavailable ({e}); use the DLS backend or pip install mink")
        return None
    table_on = table_pose_fn is not None
    load_path, table_geom = (_wrap_model_with_table(model_path, logger)
                             if table_on else (model_path, None))
    try:
        cwd = os.getcwd(); os.chdir(os.path.dirname(os.path.abspath(model_path)))
        model = mujoco.MjModel.from_xml_path(load_path)
    except Exception as e:
        logger.warn(f"mink model load failed ({e}); falling back to DLS")
        return None
    finally:
        os.chdir(cwd)

    site = f"{side}_hand"
    ls = "left" if side == "right" else "right"
    q_arm = list(range(27, 34)) if side == "right" else list(range(20, 27))
    v_arm = list(range(26, 33)) if side == "right" else list(range(19, 26))

    def _gid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    arm_geoms = [g for g in (f"{side}_forearm_pad", f"{side}_wrist_pad",
                             f"{side}_gripper_collision") if _gid(g) >= 0]
    body_geoms = [g for g in ("torso", "head", "helmet", "hip", "back_equipment",
                              f"{ls}_forearm_pad", f"{ls}_wrist_pad",
                              f"{ls}_gripper_collision") if _gid(g) >= 0]
    pairs = []
    if arm_geoms and body_geoms:
        pairs.append((arm_geoms, body_geoms))
    if table_geom and _gid(table_geom) >= 0 and arm_geoms:
        pairs.append((arm_geoms, [table_geom]))
    table_mocap = (model.body_mocapid[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table_virtual_body")]
        if table_geom else -1)

    configuration = mink.Configuration(model)
    frame_task = mink.FrameTask(frame_name=site, frame_type="site",
                                position_cost=1.0, orientation_cost=float(ori_cost),
                                lm_damping=1.0)
    posture_task = mink.PostureTask(model, cost=1e-3)
    tasks = [frame_task, posture_task]
    limits = [mink.ConfigurationLimit(model)]
    if pairs:
        limits.append(mink.CollisionAvoidanceLimit(
            model=model, geom_pairs=pairs,
            minimum_distance_from_collisions=min_dist,
            collision_detection_distance=detect_dist))
    mask = np.zeros(model.nv)
    for i in v_arm:
        mask[i] = 1.0
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    logger.info(f"mink IK ready ({side} arm; {len(pairs)} collision pair-set(s); "
                f"table={'on' if table_geom else 'off'}; solver={qp_solver})")

    def solve(pos_world, quat_wxyz, seed_q=None, return_path=False,
              iters=200, dt=0.02, tol=1.5e-3):
        try:
            configuration.update_from_keyframe("stand")
            if seed_q is not None:
                q = configuration.q.copy()
                for k, qi in enumerate(q_arm):
                    q[qi] = seed_q[k]
                configuration.update(q)
            posture_task.set_target_from_configuration(configuration)
            w, x, y, z = [float(v) for v in quat_wxyz]
            R = _quat_to_R((w, x, y, z))
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(pos_world, float)
            try:                                   # jaxlie-style API; guard variants
                se3 = mink.SE3.from_matrix(T)
            except Exception:
                se3 = mink.SE3.from_rotation_and_translation(
                    mink.SO3.from_matrix(R), np.asarray(pos_world, float))
            frame_task.set_target(se3)
            if table_mocap >= 0 and table_pose_fn is not None:
                tp = table_pose_fn()
                if tp is not None:
                    configuration.data.mocap_pos[table_mocap] = np.asarray(tp, float)
            tgt = np.asarray(pos_world, float)
            path = []
            for _ in range(iters):
                vel = mink.solve_ik(configuration, tasks, dt, qp_solver,
                                    limits=limits, damping=1e-1)
                configuration.integrate_inplace(vel * mask, dt)   # arm-only
                hp = configuration.data.site_xpos[sid].copy()
                path.append(hp)
                if np.linalg.norm(hp - tgt) < tol:
                    break
            d = configuration.data
            pe = float(np.linalg.norm(tgt - d.site_xpos[sid]))
            cq = np.zeros(4); mujoco.mju_mat2Quat(cq, d.site_xmat[sid])
            oe = np.zeros(3); mujoco.mju_subQuat(oe, np.asarray([w, x, y, z]), cq)
            q_out = np.asarray(configuration.q)[q_arm].copy()
            out = (q_out, pe, float(np.linalg.norm(oe)), pe < 0.02)
            return out + (path,) if return_path else out
        except Exception as e:  # any mink API mismatch -> graceful reject, never crash
            logger.warn(f"mink solve failed ({e}); rejecting this grasp")
            bad = (np.zeros(7), 9.9, 9.9, False)
            return bad + (None,) if return_path else bad
    return solve


def _duration_s(dur):
    """builtin_interfaces/Duration -> seconds (0.0 if unset / malformed)."""
    try:
        return float(dur.sec) + float(dur.nanosec) * 1e-9
    except Exception:
        return 0.0


def main(args=None):
    rclpy.init(args=args)
    node = GraspOrchestrator()
    # MultiThreadedExecutor so a blocking action execute() (the ReentrantCallbackGroup
    # loop) runs alongside the 50 Hz tick() + subscription + service-client callbacks.
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
