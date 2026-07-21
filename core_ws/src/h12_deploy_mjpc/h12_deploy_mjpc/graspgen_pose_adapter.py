#!/usr/bin/env python3
"""Perception -> pelvis grasp-pose adapter for the full-body MPC grasp pipeline.

The mature grasp stack (core_ws/src/h12_skills) drives the arm itself over
/frame_task. The MPC stack instead runs a C++ whole-body balancing controller
(h12_control_node, task "Grasp H12 Magpie") that consumes a grasp TARGET over
gRPC via grasp_orchestrator.py. The orchestrator subscribes to a grasp-pose
TOPIC -- but in the h12_skills stack nothing ever PUBLISHES that topic (the skill
consumes the graspgen result in-process and servos the arm). THIS node is the
missing seam: it re-runs the EXISTING h12_skills perception once, picks a grasp,
and publishes it on the topic the orchestrator already listens to.

============================  STRICT OUTPUT CONTRACT  =========================
Publishes:  topic  /graspgen/grasp_pose
            type   geometry_msgs/PoseStamped
            frame  header.frame_id = "pelvis"
The published pose is the RAW chosen GraspGen grasp (highest-scored), expressed
in pelvis:
  * position    = the GraspGen gripper-base grasp point (GraspGen origin);
  * orientation = the GraspGen grasp orientation, GraspGen convention
                  (approach = grasp-frame +Z, finger-close = grasp-frame +X),
                  as a ROS xyzw quaternion.
We do NOT apply the TCP offset, the standoff, or any re-roll here -- the
orchestrator's on_grasp_pose() applies the TCP offset + standoff itself
(grasp_orchestrator.py:123-147). So this node publishes the raw grasp only.

Frame handling (NO hardcoded extrinsic):
  The GraspGen response carries header.frame_id = the working frame the object
  cloud was built in. We transform the chosen grasp into "pelvis" with a tf2
  LOOKUP (SkillsBase._transform_pose), so the real hand-eye calibration is read
  live from the TF tree (cl_realsense static camera TFs + the onboard
  robot_state_publisher). We never bake an extrinsic into code.

  NOTE: the reused h12_skills chain (detect_object_cloud) already deprojects the
  object cloud into "pelvis" and asks graspgen for grasps in "pelvis" (graspgen
  needs a gravity-aligned z-up frame for its approach-cone reasoning, so the
  cloud MUST be pelvis, not a tilted camera optical frame). Hence the returned
  grasp is ALREADY in pelvis and the tf2 transform below is typically the
  identity. It is still done via tf2 so the node stays correct if the working
  frame is ever moved off pelvis, and to honor the "transform via tf2, never
  hardcode an extrinsic" contract literally.

Reused perception (h12_skills/base.py + skills/grasp.py), all guarded:
  * services:  gemini_query  (custom_ros_messages/srv/GeminiQuery)
               sam_segment   (custom_ros_messages/srv/SamSegment)
               graspgen      (custom_ros_messages/srv/GraspGen)
  * head camera topics (SkillsBase.CAMERA_NS = /realsense/head):
               .../color/image_raw/compressed,
               .../aligned_depth_to_color/image_raw/compressedDepth,
               .../color/camera_info
  * chain:     gemini box  ->  sam mask  ->  deproject mask to a pelvis-frame
               object PointCloud2  ->  graspgen 6-DOF grasps (SkillsBase
               .detect_object_cloud + .plan_grasp), exactly what the grasp skill
               runs. Battery-workcell parts (skills/grasp.BATTERY_OBJECTS) route
               through the head-camera YOLO detector + raw box cloud instead of
               gemini/sam (GraspSkill._yolo_box), mirroring the skill's router.

Trigger:  std_srvs/srv/Trigger service  ~/perceive
          (= /graspgen_pose_adapter/perceive). One call runs the chain once and
          publishes one PoseStamped. Target object comes from the ROS param
          `object_name` (default "battery"); the executing arm from `arm`
          (default "right").

Untested on the live stack -- needs the cameras, the vision servers, the TF tree
(pelvis), and a running grasp MPC + orchestrator. Degrades to a clear error if
h12_skills / its deps are missing (mirrors grasp_orchestrator.py's guarded-import
style).
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, PointStamped
from std_srvs.srv import Trigger

# ---- guarded reuse of the mature h12_skills perception stack ----------------
# SkillsBase carries the gemini/sam/graspgen service clients, the head-camera
# caches, the tf2 buffer, and the detect_object_cloud/plan_grasp/_transform_pose
# helpers. GraspSkill adds _yolo_box (the battery-part head-YOLO box provider)
# and BATTERY_OBJECTS (the router set). If the package or a dep (magpie_msgs,
# custom_ros_messages, cv_bridge, ...) is missing, we degrade to a node whose
# ~/perceive returns a clear failure instead of crashing at import.
try:
    from h12_skills.base import SkillsBase
    _HAVE_SKILLS = True
    _SKILLS_ERR = ""
except Exception as e:  # pragma: no cover - depends on the built ROS workspace
    SkillsBase = None
    _HAVE_SKILLS = False
    _SKILLS_ERR = str(e)

try:
    from h12_skills.skills.grasp import GraspSkill, BATTERY_OBJECTS
except Exception:  # pragma: no cover - yolo/battery path optional
    GraspSkill = object
    BATTERY_OBJECTS = ()

# Choose the base class at import time: the full perception node when h12_skills
# is present, otherwise a bare rclpy Node that only reports the missing dep.
if _HAVE_SKILLS:
    class _AdapterBase(SkillsBase, GraspSkill):
        """SkillsBase (perception clients + helpers) + GraspSkill (the head-YOLO
        `_yolo_box` battery-part box provider). No __init__ of its own -- the MRO
        runs SkillsBase.__init__ -> Node.__init__ (GraspSkill is a mixin)."""
        pass
else:  # pragma: no cover
    _AdapterBase = Node


_BATTERY_OBJECTS_LC = frozenset(o.lower() for o in BATTERY_OBJECTS)


class _ShimRun:
    """Minimal stand-in for h12_skills._Run so detect_object_cloud / _gemini_box
    -- which only touch `.remaining()` for their time budget -- can run outside
    an action-server goal. `.remaining()` returns a fixed budget (seconds)."""

    def __init__(self, budget_sec):
        self._budget = float(budget_sec)

    def remaining(self):
        return self._budget


class _ShimGH:
    """Minimal stand-in for an action goal handle: never canceled. The reused
    perception helpers only read `.is_cancel_requested`."""

    is_cancel_requested = False


def _default_grasp_model():
    """Grasp model for the reachability IK: hams_ros CONTAINER tree first, else host."""
    for root in ("/home/code/mujoco_mpc",
                 os.path.expanduser("~/Desktop/h12/mujoco_mpc/mujoco_mpc")):
        p = f"{root}/build/mjpc/tasks/humanoid_bench/grasp/Grasp_H12_Magpie.xml"
        if os.path.exists(p):
            return p
    return "/home/code/mujoco_mpc/build/mjpc/tasks/humanoid_bench/grasp/Grasp_H12_Magpie.xml"


class GraspGenPoseAdapter(_AdapterBase):
    def __init__(self):
        # SkillsBase.__init__(node_name='h12_skills') -> use a DISTINCT name so
        # this node never clashes with a running h12_skills node. Works for the
        # degraded (bare Node) base too: Node.__init__(node_name).
        super().__init__("graspgen_pose_adapter")

        # ---- params (override per cell) ----
        self.declare_parameter("object_name", "battery")   # what to grasp
        self.declare_parameter("place_object", "")          # "" = no plate; e.g. "plate"
        self.declare_parameter("arm", "right")             # executing arm (logging/cone)
        self.declare_parameter("grasp_pose_topic", "/graspgen/grasp_pose")
        self.declare_parameter("pelvis_frame", "pelvis")   # output frame (contract)
        self.declare_parameter("gripper_name", "magpie")   # graspgen gripper model
        # Time budget handed to the (slow, ~minutes for gemini) perception chain.
        self.declare_parameter("perceive_budget_sec", 240.0)
        # Reachability fallback: reuse the ORCHESTRATOR's IK + TCP transform so this
        # filter predicts exactly what the orchestrator will accept/reject.
        self.declare_parameter("tcp_offset_m", [0.17, 0.0, 0.0])
        self.declare_parameter("ik_accept_m", 0.02)
        # TOP-DOWN candidate preference: when set, prefer GraspGen candidates whose
        # approach axis is within top_down_cone_deg of vertical (down) -- the standard
        # tabletop filter (top-down minimises collision with the support surface).
        # Pair with the orchestrator's top_down flag (which then enforces the vertical
        # approach on execution); here we just pick a top-down-friendly grasp POINT.
        self.declare_parameter("top_down", False)
        self.declare_parameter("top_down_cone_deg", 45.0)
        self.declare_parameter("grasp_model_path", _default_grasp_model())

        self._topic = self.get_parameter("grasp_pose_topic").value
        self._pelvis = self.get_parameter("pelvis_frame").value

        # Shared IK solver (same one the orchestrator uses for q*). Disabled ->
        # the selection degrades to argmax(score). Needs mujoco in the ROS env,
        # which the orchestrator requires anyway, so if the pipeline runs, this does.
        self._ik = None
        try:
            from h12_deploy_mjpc.grasp_orchestrator import _make_ik_solver
            self._ik = _make_ik_solver(
                self.get_logger(), self.get_parameter("grasp_model_path").value,
                side=self.get_parameter("arm").value)
        except Exception as e:  # pragma: no cover
            self.get_logger().warn(
                f"IK reachability filter unavailable ({e}); selection = argmax(score)")

        # Latched publisher (TRANSIENT_LOCAL): a late-joining orchestrator still
        # receives the most recent grasp on connect. Compatible with the
        # orchestrator's default (volatile) subscription.
        self._pub = self.create_publisher(
            PoseStamped, self._topic,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # Camera-detected drop point (plate centroid), pelvis frame. Latched so the
        # orchestrator gets it on connect. Only published when place_object is set.
        self._place_pub = self.create_publisher(
            PointStamped, "/place_point",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        # A reentrant group is required so the ~/perceive callback can block on
        # the gemini/sam/graspgen service futures while the executor keeps
        # spinning (SkillsBase's helpers wait on futures, not spins). SkillsBase
        # already made one; reuse it so the clients and the service share a group.
        self._cbg = getattr(self, "_cb_group", None) or ReentrantCallbackGroup()

        self._perceive_srv = self.create_service(
            Trigger, "~/perceive", self.on_perceive, callback_group=self._cbg)

        if _HAVE_SKILLS:
            self.get_logger().info(
                f"graspgen_pose_adapter up; publishing {self._topic} "
                f"(frame={self._pelvis!r}); call ~/perceive to run the chain "
                f"(object_name={self.get_parameter('object_name').value!r}, "
                f"arm={self.get_parameter('arm').value!r})")
        else:  # pragma: no cover
            self.get_logger().error(
                f"h12_skills perception UNAVAILABLE ({_SKILLS_ERR}); ~/perceive "
                f"will report failure. Build/source the h12_skills workspace.")

    def _route_for(self, name):
        """Detection route for ANY named object -- identical for the grasp target and
        the plate. A fine-tuned YOLO class (BATTERY_OBJECTS: Bolt/Nut/Screw/...) uses
        the head-camera YOLO box + raw cloud; anything generic (a plate, a bin, ...)
        uses a Gemini box + SAM mask. Returns (box_provider, use_sam, route_label)."""
        if name.strip().lower() in _BATTERY_OBJECTS_LC and hasattr(self, "_yolo_box"):
            return self._yolo_box, False, "yolo+raw-box"
        return None, True, "gemini+sam"

    # ---- camera-detect the plate -> publish its centroid as the drop point ------
    def _perceive_place(self, run, gh):
        """Detect the plate (place_object) via the gemini+sam route and publish its
        object-cloud CENTROID (pelvis frame) on /place_point. Best-effort: any miss
        logs and returns without publishing, so the grasp still proceeds (and, with
        no place point, the orchestrator stops after close)."""
        name = str(self.get_parameter("place_object").value).strip()
        if not name:
            return
        box_provider, use_sam, route = self._route_for(name)    # same routes as the grasp target
        self.get_logger().info(
            f"perceive: locating plate {name!r} ({route}) for the drop point")
        cloud, _scene, err = self.detect_object_cloud(
            name, run, gh, box_provider=box_provider, use_sam=use_sam)
        if err or cloud is None or len(cloud) == 0:
            self.get_logger().warn(
                f"perceive: plate {name!r} not found ({err or 'empty cloud'}); no place point")
            return
        c = np.asarray(cloud, float).mean(axis=0)               # centroid = drop target (pelvis)
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._pelvis
        msg.point.x, msg.point.y, msg.point.z = float(c[0]), float(c[1]), float(c[2])
        self._place_pub.publish(msg)
        self.get_logger().info(
            f"perceive: plate {name!r} -> place point "
            f"({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}) pelvis")

    # ---- one-shot perceive: chain -> pick highest score -> tf2 -> publish -----
    def on_perceive(self, request, response):
        if not _HAVE_SKILLS:
            response.success = False
            response.message = f"h12_skills perception unavailable: {_SKILLS_ERR}"
            self.get_logger().error(response.message)
            return response

        obj = str(self.get_parameter("object_name").value)
        arm = str(self.get_parameter("arm").value)
        gripper = str(self.get_parameter("gripper_name").value)
        budget = float(self.get_parameter("perceive_budget_sec").value)
        run, gh = _ShimRun(budget), _ShimGH()

        # Camera-detect the plate FIRST (if configured) so /place_point is published
        # before the grasp pose -> the orchestrator has the drop point in hand by the
        # time the grasp arrives. Best-effort: a miss just skips the place.
        self._perceive_place(run, gh)

        # Route selection is identical for the grasp target and the plate (_route_for):
        # YOLO for a fine-tuned battery class, Gemini+SAM for anything generic.
        box_provider, use_sam, route = self._route_for(obj)
        self.get_logger().info(
            f"perceive: object_name={obj!r} arm={arm} route={route}")

        # --- detect -> object cloud (pelvis frame) -----------------------------
        obj_cloud, scene, err = self.detect_object_cloud(
            obj, run, gh, box_provider=box_provider, use_sam=use_sam)
        if err:
            response.success = False
            response.message = err
            self.get_logger().error(f"perceive: {err}")
            return response

        # --- plan grasps: graspgen on the object cloud (returns pelvis-frame) ---
        resp = self.plan_grasp(obj_cloud, frame=self._pelvis,
                               gripper_name=gripper, scene_cloud=scene, arm=arm)
        if resp is None or not resp.grasps:
            response.success = False
            response.message = f"no grasp planned for {obj!r}"
            self.get_logger().error(f"perceive: {response.message}")
            return response

        # --- pick the best-scored grasp the ARM can REACH ----------------------
        # graspgen ranks best-first; walk that order and take the first candidate
        # whose IK is reachable (same solver + TCP shift the orchestrator uses), so
        # we don't hand the orchestrator a top-scored but IK-infeasible grasp -- the
        # H1-2 wrist rejects many. Mirrors the mature h12_skills IK-fallback. If none
        # pass (or IK is disabled), publish the best-scored and let the orchestrator's
        # accept-gate decide.
        scores = [float(s) for s in resp.scores]
        order = list(np.argsort(scores)[::-1]) if scores else list(range(len(resp.grasps)))
        # TOP-DOWN: re-order so candidates whose approach (grasp +Z) is within the cone
        # of vertical (down) come first, best-scored within each group. The IK-reachable
        # walk below then picks the best top-down-friendly reachable grasp.
        if bool(self.get_parameter("top_down").value) and scores:
            from h12_deploy_mjpc.grasp_orchestrator import quat_xyzw_to_wxyz, _quat_to_R
            cos_cone = math.cos(math.radians(
                float(self.get_parameter("top_down_cone_deg").value)))

            def _in_cone(i):
                g = resp.grasps[i].pose
                qw = quat_xyzw_to_wxyz([g.orientation.x, g.orientation.y,
                                        g.orientation.z, g.orientation.w])
                approach = _quat_to_R(qw)[:, 2]              # grasp +Z in pelvis
                return float(-approach[2]) >= cos_cone       # aligned with world-down
            order.sort(key=lambda i: (0 if _in_cone(i) else 1, -scores[i]))
            n_cone = sum(1 for i in order if _in_cone(i))
            self.get_logger().info(
                f"top-down: {n_cone}/{len(order)} candidates within "
                f"{self.get_parameter('top_down_cone_deg').value:.0f} deg of vertical")
        idx = order[0]
        if self._ik is not None:
            from h12_deploy_mjpc.grasp_orchestrator import quat_xyzw_to_wxyz, _quat_to_R
            tcp = np.array(self.get_parameter("tcp_offset_m").value, float)
            picked = None
            for rank, i in enumerate(order):
                g = resp.grasps[i].pose
                qw = quat_xyzw_to_wxyz(
                    [g.orientation.x, g.orientation.y, g.orientation.z, g.orientation.w])
                pos = np.array([g.position.x, g.position.y, g.position.z], float)
                pos_tcp = pos - _quat_to_R(qw) @ tcp     # match the orchestrator's TCP shift
                _, pe, _, ok = self._ik(pos_tcp, qw)
                if ok:
                    picked = i
                    self.get_logger().info(
                        f"perceive: candidate rank {rank + 1}/{len(order)} reachable "
                        f"(score {scores[i]:.3f}, IK {pe * 1e3:.0f}mm)")
                    break
            if picked is not None:
                idx = picked
            else:
                self.get_logger().warn(
                    f"perceive: NONE of {len(order)} candidates IK-reachable; "
                    f"publishing best-scored (orchestrator may reject)")
        chosen = resp.grasps[idx]              # PoseStamped in resp's working frame
        score = scores[idx] if scores else float("nan")

        # --- tf2-transform the RAW chosen grasp into pelvis --------------------
        # See the module docstring: chosen.header.frame_id is the graspgen working
        # frame (pelvis here), and _transform_pose does a live tf2 lookup, so the
        # extrinsic is never hardcoded. Identity when already pelvis.
        src = chosen.header.frame_id or self._pelvis
        pose_pelvis = self._transform_pose(chosen.pose, src, self._pelvis)
        if pose_pelvis is None:
            response.success = False
            response.message = (
                f"tf2 {src!r} -> {self._pelvis!r} unavailable; cannot publish "
                f"grasp (is the robot_state_publisher / camera TF tree up?)")
            self.get_logger().error(f"perceive: {response.message}")
            return response

        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self._pelvis
        out.pose = pose_pelvis
        self._pub.publish(out)

        p, o = out.pose.position, out.pose.orientation
        self.get_logger().info(
            f"perceive: published grasp {idx + 1}/{len(resp.grasps)} "
            f"score={score:.3f} on {self._topic} frame={self._pelvis!r}: "
            f"pos=({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) "
            f"quat_xyzw=({o.x:.3f}, {o.y:.3f}, {o.z:.3f}, {o.w:.3f}) "
            f"gripper_width={resp.gripper_width * 1000:.1f}mm")
        response.success = True
        response.message = (
            f"grasp published (score {score:.3f}, "
            f"{len(resp.grasps)} candidates, route {route})")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GraspGenPoseAdapter()
    # MultiThreadedExecutor + the reentrant callback group let the ~/perceive
    # callback block on the perception service futures while the executor keeps
    # servicing them (same arrangement as the h12_skills node).
    executor = MultiThreadedExecutor()
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
