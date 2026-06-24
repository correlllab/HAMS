#!/usr/bin/env python3
"""graspgen_server — ROS2 service exposing GraspGenX grasp planning.

Serves custom_ros_messages/srv/GraspGen on `graspgen`: a segmented object
PointCloud2 in, ranked 6-DOF grasp poses out. The GraspGenXSampler (heavy GPU
model) is built ONCE per gripper and cached. Checkpoints default to
<package>/weights/graspgen (override with GRASPGENX_CHECKPOINT_DIR); the gripper
assets default to /opt/graspgenx/assets (override with GRASPGENX_ASSETS_DIR), where
the magpie gripper description is staged.

Runs inside the ros container (GraspGenX is pip-installed there). Import of
graspgenx is deferred to __init__ so the module can be inspected without it.
"""

import inspect
import math
import os
from collections import namedtuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster

import trimesh
from graspgenx.samplers import run_planner_on_object
from graspgenx.utils.checkpoint_io import load_model_cfg
from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.utils.collision_filter import filter_colliding_grasps
from custom_ros_messages.srv import GraspGen

from model_server.model_logging import ModelLogger, declare_logging_params
import cv2

# viser web visualizer (graspgenx.utils.viser_utils imports `viser` at module
# load). It's an optional, viz-only dep that may be absent on a pre-rebuild image,
# so guard the import and degrade to RViz-markers-only when it's missing.
try:
    from graspgenx.utils.viser_utils import (
        create_visualizer, visualize_pointcloud, visualize_x_grasp,
        visualize_bbox, get_color_from_score,
    )
    _VISER_OK = True
except Exception:                                 # viser not installed
    _VISER_OK = False


# Default gripper for planning: the magpie description, staged into the GraspGenX
# assets via bind mount (no rebuild). The GraspGen.srv `gripper_name` field
# overrides per request.
DEFAULT_GRIPPER = 'magpie'
MIN_OBJECT_POINTS = 100
# RViz grasp markers: how many of the ranked grasps to draw. Each is drawn as an
# arrow from the GraspGenX pose ORIGIN (the gripper base, where the IK pins
# *_graspgenx_frame) along +Z (approach) to the CONTACT point where the fingers
# close — so the arrow TIP sits on the object, i.e. "where the gripper point is".
# GRASPGEN_MARKER_LENGTH_M is that base->contact arrow length; magpie = 0.193 m
# (its config fingertip = [0,0,0.193]).
N_GRASP_MARKERS = 5
GRASPGEN_MARKER_LENGTH_M = 0.193
# TF frame broadcast for the best (top-ranked) grasp pose, parented to the request
# cloud's frame (e.g. pelvis). Lets RViz / other nodes see and look up the planned
# grasp target. This is the pose the IK's *_graspgenx_frame is driven to.
TARGET_FRAME_ID = 'graspgenx_target_frame'

# Collision filtering against an optional scene cloud (obstacles), done only when
# the request carries a non-empty scene_cloud. A grasp is dropped when the gripper
# SWEEP VOLUME (the jaw region, sampled in _gripper_surface_points — NOT the bulky
# full visual mesh) comes within COLLISION_THRESHOLD_M of any scene point. The
# target object's own points are excluded from the scene first (any scene point
# within OBJECT_EXCLUDE_RADIUS_M of the object cloud) so the object being grasped
# never counts as a collision. The scene is randomly downsampled to MAX_SCENE_POINTS
# before the GPU cdist, and the gripper surface is sampled with NUM_COLLISION_SAMPLES
# points once per gripper.
# NOTE ON DIRECTION: larger = STRICTER (more grasps filtered). A grasp is dropped
# when the gripper sweep volume comes within this distance of an obstacle, so a big
# value (e.g. 5 cm) rejects grasps that merely pass NEAR the scene and can filter
# everything; keep it small (near-contact only). Tune up only if grasps clip obstacles.
COLLISION_THRESHOLD_M = 0.01
# Scene points within this radius of the object cloud are treated as the object
# (not obstacles) and dropped before the collision check. MUST be >=
# COLLISION_THRESHOLD_M: a grasp brings the gripper within COLLISION_THRESHOLD_M
# of the object surface, so any object-surface point left in the scene — e.g. a
# rim just outside the SAM mask — would falsely flag every good grasp as
# colliding with the very object being grasped. Kept a touch larger for seg slop.
OBJECT_EXCLUDE_RADIUS_M = 0.02
MAX_SCENE_POINTS = 8192
NUM_COLLISION_SAMPLES = 2000
# A gripper with no coll_mesh.obj falls back (inside GraspGenX) to a ~1 cm dummy
# box at the gripper base, which makes collision filtering a silent no-op. Treat
# any collision mesh whose largest bounding-box dimension is below this as "no
# real mesh" and skip filtering (with a warning) instead. Real grippers span far
# more (parallel-jaw / magpie fingers reach ~0.11-0.19 m from the base).
DUMMY_MESH_MIN_EXTENT_M = 0.05

# viser visualizer: a web GUI (http://localhost:VISER_PORT) that renders the scene
# cloud, object cloud, and ranked grasps live alongside the RViz markers. The ros
# service runs with network_mode: host, so the port is reachable on the host.
VISER_PORT = 8080

# Per-gripper cache entry: the heavy sampler, its opening width [m], the gripper
# info (sweep volume + collision mesh, for viz/collision), and the gripper surface
# points pre-sampled once for the collision filter (None if sampling failed).
_Gripper = namedtuple('_Gripper', 'sampler width info surf_pts')

# Candidate planner kwargs (filtered to run_planner_on_object's real signature).
PLANNER_KWARGS = dict(
    planner='graspmoe', grasp_threshold=-1.0, num_grasps=200,
    topk_num_grasps=100, moe_num_yaws=36, moe_z_offsets_cm=(-2.0, 0.0),
    moe_outlier_threshold=0.014, moe_outlier_k=20, moe_obb_mode='advanced',
    moe_skip_obb_rule='auto', moe_obb_density='dense-topandside',
    moe_obb_position_spacing_cm=1.0,
)

# GraspGenX checkpoints: point GRASPGENX_CHECKPOINT_DIR at the dir that holds the
# `release/{gen,dis}` checkpoints (required — no implicit fallback).


def mat_to_quat(R):
    """3x3 rotation matrix -> quaternion (x, y, z, w). Shepperd's method."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    return (x, y, z, w)


def _sampler_width(sampler):
    """Best-effort gripper opening width [m] from the sampler/gripper info; falls
    back to 0.08 m. The exact attribute path varies across GraspGenX versions."""
    for path in ('gripper_info', 'gripper'):
        obj = getattr(sampler, path, None)
        for attr in ('width', 'maximum_aperture', 'max_width'):
            val = getattr(obj, attr, None) if obj is not None else None
            if isinstance(val, (int, float)) and val > 0:
                return float(val)
    cfg = getattr(sampler, 'gripper_config', None)
    if isinstance(cfg, dict):
        for k in ('width', 'maximum_aperture'):
            if isinstance(cfg.get(k), (int, float)) and cfg[k] > 0:
                return float(cfg[k])
    return 0.08


class GraspGenServer(Node):
    def __init__(self):
        super().__init__('graspgen_server')
        log, viz, clear = declare_logging_params(self)
        self.logger = ModelLogger(self, 'graspgen', 'model_server', __file__,
                                  log=log, visualize=viz, clear=clear)
        
        self._run_planner = run_planner_on_object
        self._GraspGenXSampler = GraspGenXSampler

        ckpt_root = os.environ['GRASPGENX_CHECKPOINT_DIR']
        checkpoints = os.path.join(ckpt_root, 'release')
        self._assets_dir = os.environ['GRASPGENX_ASSETS_DIR']
        self.get_logger().info(f'Loading GraspGenX model cfg from {checkpoints} ...')
        self._model_cfg = load_model_cfg(
            os.path.join(checkpoints, 'gen'), os.path.join(checkpoints, 'dis'), None, None)

        # Keep only the planner kwargs the installed run_planner_on_object accepts.
        sig = inspect.signature(run_planner_on_object)
        self._kwargs = {k: v for k, v in PLANNER_KWARGS.items() if k in sig.parameters}

        self._samplers = {}           # gripper_name -> _Gripper
        try:
            self._get_sampler(DEFAULT_GRIPPER)   # warm the default + validate load
        except Exception as e:
            # Don't take the node down — keep serving so a valid per-request
            # gripper_name still works (that first call pays the load cost).
            self.get_logger().error(
                f'default gripper {DEFAULT_GRIPPER!r} failed to load ({e}); '
                f'pass a valid gripper_name in the request')

        # Latched (TRANSIENT_LOCAL) so RViz still gets the last grasp markers even
        # if it subscribes after a plan was served. One arrow per ranked grasp, in
        # the request's frame (pelvis), drawn along the GraspGenX approach axis.
        self._marker_pub = self.create_publisher(
            MarkerArray, 'graspgen_markers',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # Republish the object cloud we planned on, so it can be inspected in RViz
        # against the grasp markers (same frame). Also latched.
        self._cloud_pub = self.create_publisher(
            PointCloud2, 'grasp_cloud',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # Republish the scene/obstacle cloud (when a request carries one) on its own
        # topic, so the obstacles the collision filter ran against are visible in
        # RViz alongside the object cloud and grasp markers. Also latched.
        self._scene_pub = self.create_publisher(
            PointCloud2, 'scene_cloud',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # Broadcast the best grasp as a TF frame (TARGET_FRAME_ID). Dynamic (on /tf,
        # NOT /tf_static): the parent is the cloud frame (pelvis), the robot's moving
        # floating base, so the target must carry a live timestamp. A re-publish timer
        # keeps the latest grasp's frame fresh + visible in RViz until the next plan
        # replaces it.
        self._tf_broadcaster = TransformBroadcaster(self)
        self._target_tf = None      # latest TransformStamped (set per plan)
        self._target_tf_timer = self.create_timer(0.1, self._republish_target_frame)

        # Live viser visualizer (web GUI). Started once; each plan resets and
        # redraws its scene. Optional — degrade to RViz markers if viser is absent.
        self._vis = None
        if _VISER_OK:
            try:
                self._vis = create_visualizer(port=VISER_PORT)
                self.get_logger().info(
                    f'viser visualizer at http://localhost:{VISER_PORT}')
            except Exception as e:
                self.get_logger().warn(f'viser init failed ({e}); RViz markers only')
        else:
            self.get_logger().info('viser not installed; RViz markers only')

        self.create_service(GraspGen, 'graspgen', self._plan_cb,
                            callback_group=ReentrantCallbackGroup())
        self.get_logger().info("graspgen_server ready on 'graspgen'")

    def _get_sampler(self, gripper_name):
        name = gripper_name or DEFAULT_GRIPPER
        if name not in self._samplers:
            self.get_logger().info(f'Building GraspGenXSampler (gripper={name}) ...')
            sampler = self._GraspGenXSampler(
                self._model_cfg, name, assets_dir=self._assets_dir)
            # Gripper geometry (for collision mesh + viser). Best-effort: keep
            # serving even if this version exposes no gripper info.
            info = getattr(sampler, 'gripper', None)
            if info is None and hasattr(sampler, 'get_gripper_info'):
                try:
                    info = sampler.get_gripper_info()
                except Exception:
                    info = None
            # Surface points for the collision filter, sampled ONCE per gripper.
            surf_pts = self._gripper_surface_points(info, name)
            self._samplers[name] = _Gripper(
                sampler, _sampler_width(sampler), info, surf_pts)
        return self._samplers[name]

    def _gripper_surface_points(self, info, name):
        """(M, 3) gripper-local surface points for the collision filter, or None to
        disable filtering for this gripper.

        Sampled from the gripper's SWEEP-VOLUME box — the jaw region viser draws —
        NOT the full visual mesh. The visual mesh includes the wide gripper
        body/mount (magpie: 21.8x17.3 cm vs the 11.2x11.6 cm jaw sweep), so
        checking it rejects grasps whose drawn gripper is plainly clear of the
        scene ("looks good but filtered"). Falls back to a real collision mesh
        only when no sweep volume is exposed; skips the ~1 cm dummy-box fallback."""
        sv = getattr(info, 'sweep_volume', None) if info is not None else None
        if sv is not None:
            sv = np.asarray(sv, dtype=np.float64).reshape(-1)
            # sweep_volume = [extents_xyz, offset_xyz]; box of those extents at
            # that offset, in the gripper-base frame.
            if sv.shape[0] >= 6 and np.all(sv[:3] > 0):
                box = trimesh.primitives.Box(extents=sv[:3])
                box.apply_translation(sv[3:6])
                try:
                    sp, _ = trimesh.sample.sample_surface(box, NUM_COLLISION_SAMPLES)
                    return np.asarray(sp, dtype=np.float32)
                except Exception as e:
                    self.get_logger().warn(
                        f'sweep-volume sampling failed for {name!r} ({e})')
        # Fallback: the real collision mesh (skip the ~1 cm dummy-box fallback).
        mesh = getattr(info, 'collision_mesh', None) if info is not None else None
        extent = (float(np.ptp(np.asarray(mesh.bounds), axis=0).max())
                  if mesh is not None else 0.0)
        if mesh is None:
            self.get_logger().warn(
                f'gripper {name!r} has no sweep volume or collision mesh; '
                f'collision filtering disabled for it')
            return None
        if extent < DUMMY_MESH_MIN_EXTENT_M:
            self.get_logger().warn(
                f'gripper {name!r} has no sweep volume and only a '
                f'{extent * 100:.1f} cm dummy collision mesh; collision filtering '
                f'disabled for it')
            return None
        try:
            sp, _ = trimesh.sample.sample_surface(mesh, NUM_COLLISION_SAMPLES)
            return np.asarray(sp, dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(
                f'gripper surface sampling failed for {name!r} ({e}); '
                f'collision filtering disabled for this gripper')
            return None

    def _plan_cb(self, request, response):
        rec = self.logger.start()
        frame = request.object_cloud.header.frame_id
        rec.set(gripper_name=request.gripper_name or DEFAULT_GRIPPER, frame=frame)
        try:
            pts = point_cloud2.read_points_numpy(
                request.object_cloud, field_names=('x', 'y', 'z'), skip_nans=True)
        except Exception as e:                       # malformed cloud
            response.success, response.message = False, f'bad cloud: {e}'
            rec.finish(success=False, message=response.message)
            return response
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        rec.set(n_input_points=int(pts.shape[0]))
        rec.save_array('input_cloud', pts)
        self._cloud_pub.publish(request.object_cloud)   # republish for RViz
        if pts.shape[0] < MIN_OBJECT_POINTS:
            response.success, response.message = False, f'too few points ({pts.shape[0]})'
            rec.finish(success=False, message=response.message)
            return response

        # Optional scene cloud (obstacles, same frame as object_cloud) for
        # collision filtering. Empty/malformed -> skip filtering, don't fail.
        scene_pts = None
        if len(request.scene_cloud.data) > 0:
            try:
                sp = point_cloud2.read_points_numpy(
                    request.scene_cloud, field_names=('x', 'y', 'z'), skip_nans=True)
                scene_pts = np.asarray(sp, dtype=np.float32).reshape(-1, 3)
                rec.set(n_scene_points=int(scene_pts.shape[0]))
                rec.save_array('scene_cloud', scene_pts)        # raw scene points (in frame)
                self._scene_pub.publish(request.scene_cloud)    # republish for RViz
                self.get_logger().info(
                    f'scene cloud: {scene_pts.shape[0]} points (frame={frame})')
            except Exception as e:
                self.get_logger().warn(
                    f'scene_cloud decode failed ({e}); skipping collision filter')
                scene_pts = None

        try:
            gr = self._get_sampler(request.gripper_name)
        except Exception as e:
            response.success, response.message = False, f'sampler load failed: {e}'
            self.get_logger().error(response.message)
            rec.finish(success=False, message=response.message)
            return response
        width = gr.width

        mean = pts.mean(axis=0)
        pc_centered = (pts - mean).astype(np.float32)
        try:
            out = self._run_planner(pc_centered, gr.sampler, **self._kwargs)
        except Exception as e:
            response.success, response.message = False, f'planner failed: {e}'
            self.get_logger().error(response.message)
            rec.finish(success=False, message=response.message)
            return response

        grasps = np.asarray(out[0] if isinstance(out, (tuple, list)) else out)
        # Guard shape BEFORE any len()/indexing — a 0-d/None-ish return must fail
        # cleanly, not crash the service thread (grasps.ndim is safe on 0-d).
        if grasps.ndim != 3 or grasps.shape[0] == 0:
            response.success, response.message = False, 'no grasps produced'
            rec.finish(success=False, message=response.message)
            return response
        n = grasps.shape[0]
        confs = (np.asarray(out[1]).reshape(-1)
                 if isinstance(out, (tuple, list)) and len(out) > 1 else np.ones(n))
        if confs.shape[0] != n:                       # contract drift -> don't drop grasps
            confs = np.ones(n)
        # obb_dict (planner out[3]) is for viz only; tolerate planners that omit it.
        obb_dict = out[3] if isinstance(out, (tuple, list)) and len(out) > 3 else None

        # Rank best-first: the planner returns diff+obb grasps unsorted, but the
        # response contract is "ranked best-first" and the skill executes grasps[0],
        # so sort by confidence before (optionally) collision-filtering.
        order = np.argsort(-confs)
        grasps, confs = grasps[order], confs[order]

        # Collision filtering against the scene cloud, in `frame`. The planner
        # returns grasps in the server-centered frame, so un-center to `frame`
        # (matching scene_pts) before the check.
        if scene_pts is not None and len(scene_pts) and gr.surf_pts is not None:
            grasps_world = grasps.copy().astype(np.float64)
            grasps_world[:, :3, 3] += mean
            scene_obstacles = self._scene_minus_object(scene_pts, pts)
            rec.save_array('scene_obstacles', scene_obstacles)   # scene minus object (collision input)
            rec.save_array('grasps_world', grasps_world)         # poses fed to the filter (debug)
            free = filter_colliding_grasps(
                scene_pc=scene_obstacles, grasp_poses=grasps_world,
                gripper_surface_points=gr.surf_pts,
                collision_threshold=COLLISION_THRESHOLD_M)
            kept, kept_confs = grasps[free], confs[free]
            n_before = len(grasps)
            rec.set(n_scene_obstacle_points=int(len(scene_obstacles)),
                    n_collision_free=int(len(kept)),
                    n_collision=int(n_before - len(kept)))
            self.get_logger().info(
                f'collision filter: {len(kept)}/{n_before} free '
                f'(thr={COLLISION_THRESHOLD_M}m, scene_pts={len(scene_obstacles)})')
            if len(kept) == 0:
                # The success path renders viz at the end; this early return would
                # skip it, leaving the all-collision failure un-inspectable. Render
                # the REJECTED grasps against the obstacle cloud the filter actually
                # used so viser (:8080) / the PNG show WHY everything collided.
                self._render_viser(pts, scene_obstacles, grasps, confs, mean,
                                   gr.info, obb_dict)
                self._save_grasp_viz(rec, pts, grasps, mean, width)
                response.success, response.message = False, \
                    'all grasps in collision with scene'
                rec.finish(success=False, message=response.message)
                return response
            grasps, confs = kept, kept_confs

        for g, c in zip(grasps, confs):
            g = np.asarray(g, dtype=np.float64).reshape(4, 4).copy()
            g[:3, 3] += mean                         # un-center back to the input frame
            qx, qy, qz, qw = mat_to_quat(g[:3, :3])
            ps = PoseStamped()
            ps.header.frame_id = frame
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (
                float(g[0, 3]), float(g[1, 3]), float(g[2, 3]))
            ps.pose.orientation.x = float(qx)
            ps.pose.orientation.y = float(qy)
            ps.pose.orientation.z = float(qz)
            ps.pose.orientation.w = float(qw)
            response.grasps.append(ps)
            response.scores.append(float(c))
        response.gripper_width = float(width)
        response.success = True
        response.message = f'{len(response.grasps)} grasps'
        # Publish the best grasp (grasps[0], best-first) as the target TF frame.
        self._set_target_frame(frame, response.grasps[0].pose)
        # Grasp metrics: count, confidence spread (confs are sorted best-first),
        # gripper opening, and the input sizes the plan was made from.
        best = float(confs[0]) if len(confs) else 0.0
        worst = float(confs[-1]) if len(confs) else 0.0
        rec.set(n_grasps=len(response.grasps), gripper_width=float(width),
                score_best=round(best, 3), score_worst=round(worst, 3))
        self.get_logger().info(
            f"grasp metrics: {len(response.grasps)} grasps "
            f"(gripper={request.gripper_name or DEFAULT_GRIPPER}, "
            f"score best={best:.3f} worst={worst:.3f}, width={width * 1000:.0f}mm, "
            f"object_pts={pts.shape[0]}, "
            f"scene_pts={0 if scene_pts is None else len(scene_pts)})")
        rec.save_array('grasps', grasps)            # raw (M, 4, 4) in centered frame
        rec.save_array('scores', confs)
        self._publish_grasp_markers(frame, grasps, mean)
        self._save_grasp_viz(rec, pts, grasps, mean, width)
        self._render_viser(pts, scene_pts, grasps, confs, mean, gr.info, obb_dict)
        rec.finish(success=True, message=response.message)
        return response

    def _set_target_frame(self, parent_frame, pose):
        """Stash the best grasp `pose` (in `parent_frame`) as the TARGET_FRAME_ID
        transform; _republish_target_frame broadcasts it on /tf with a live stamp
        until the next plan replaces it. No-op if the cloud frame_id is missing."""
        if not parent_frame:
            self.get_logger().warn(
                'object_cloud has no frame_id; skipping target-frame broadcast')
            return
        t = TransformStamped()
        t.header.frame_id = parent_frame
        t.child_frame_id = TARGET_FRAME_ID
        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z
        t.transform.rotation = pose.orientation
        self._target_tf = t

    def _republish_target_frame(self):
        """Re-broadcast the latest grasp target with a fresh stamp so it stays live
        in TF / visible in RViz (a dynamic transform goes stale if not refreshed).
        Best-effort, like the other viz helpers — never let it break the node."""
        t = self._target_tf
        if t is None:
            return
        try:
            t.header.stamp = self.get_clock().now().to_msg()
            self._tf_broadcaster.sendTransform(t)
        except Exception as e:
            self.get_logger().warn(f'target-frame broadcast failed: {e}')

    def _scene_minus_object(self, scene_pts, object_pts):
        """Scene cloud with the target object removed (any scene point within
        OBJECT_EXCLUDE_RADIUS_M of an object point) and randomly downsampled to
        MAX_SCENE_POINTS — the obstacle set the gripper must avoid."""
        scene_pts = np.asarray(scene_pts, dtype=np.float32)
        if len(object_pts):
            from scipy.spatial import cKDTree
            d, _ = cKDTree(np.asarray(object_pts, dtype=np.float32)).query(scene_pts, k=1)
            scene_pts = scene_pts[d > OBJECT_EXCLUDE_RADIUS_M]
        if len(scene_pts) > MAX_SCENE_POINTS:
            idx = np.random.choice(len(scene_pts), MAX_SCENE_POINTS, replace=False)
            scene_pts = scene_pts[idx]
        return scene_pts.astype(np.float32)

    def _render_viser(self, object_pts, scene_pts, grasps, confs, mean,
                      gripper_info, obb_dict=None):
        """Redraw the live viser scene: gray scene cloud, blue object cloud, the
        OBB wireframe, and the ranked grasps (best = thick blue, rest colored by
        score). `grasps` are in the server-centered frame; un-center with `mean`
        to match the clouds. No-op when viser isn't running."""
        if self._vis is None:
            return
        try:
            vis = self._vis
            vis.scene.reset()
            if scene_pts is not None and len(scene_pts):
                visualize_pointcloud(vis, 'scene', scene_pts,
                                     color=[150, 150, 150], size=0.0025)
            visualize_pointcloud(vis, 'object', np.asarray(object_pts),
                                 color=[0, 150, 255], size=0.004)
            if not len(grasps):
                return
            G = grasps.copy().astype(np.float64)
            G[:, :3, 3] += mean                       # centered frame -> world
            # OBB wireframe is best-effort and must NEVER block the grasps: this
            # viser's add_box() has no `wireframe` kwarg (which visualize_bbox
            # passes), so an un-guarded call here raises and aborts the whole
            # render — clouds (already drawn) stay, grasps never get drawn.
            if obb_dict is not None:
                try:
                    T = np.eye(4)
                    T[:3, :3] = obb_dict['R']
                    T[:3, 3] = np.asarray(obb_dict['center']) + mean
                    visualize_bbox(vis, 'obb',
                                   2.0 * np.asarray(obb_dict['half_extent']),
                                   T=T, color=[255, 130, 0])
                except Exception as e:
                    self.get_logger().warn(f'viser OBB draw skipped: {e}')
            colors = get_color_from_score(np.asarray(confs, dtype=np.float32),
                                          use_255_scale=True)
            colors = np.atleast_2d(colors)
            for i in range(len(G)):
                best = (i == 0)                        # grasps are sorted best-first
                color = [0, 100, 255] if best else colors[i]
                visualize_x_grasp(vis, f'grasps/g{i:03d}', G[i], color=color,
                                  gripper_info=gripper_info,
                                  linewidth=5.0 if best else 1.5)
        except Exception as e:                          # viz must never break planning
            self.get_logger().warn(f'viser render failed: {e}')

    def _publish_grasp_markers(self, frame, grasps, mean):
        """Publish the generated grasps as RViz markers on 'graspgen_markers'
        (latched), in `frame` (pelvis). For each ranked grasp:
          - an ARROW from the GraspGenX pose ORIGIN (gripper base, where the IK
            pins *_graspgenx_frame) along +Z to the CONTACT point, so the arrow TIP
            lands on the object = where the fingers close ("the gripper point");
          - a small SPHERE at that contact point.
        The base (arrow tail) is where the driven frame — and ~8 cm behind it, the
        wrist — ends up; the tip is where the gripper actually grasps. This makes
        the base-vs-contact distinction visible so the wrist sitting near the base
        isn't mistaken for the gripper being short. Best grasp (grasps[0], the one
        the skill executes) is bright green; the rest are dim. Mirrors
        _save_grasp_viz's un-centering so the markers match the PNG."""
        try:
            G = np.asarray(
                [np.asarray(g, dtype=np.float64).reshape(4, 4) for g in grasps])
            centers = G[:, :3, 3] + mean          # un-center back to the input frame
            topk = min(N_GRASP_MARKERS, len(G))
            stamp = self.get_clock().now().to_msg()

            arr = MarkerArray()
            # Clear any markers from a previous plan so stale ones don't linger.
            clear = Marker()
            clear.header.frame_id = frame
            clear.action = Marker.DELETEALL
            arr.markers.append(clear)

            for i in range(topk):
                R, base = G[i, :3, :3], centers[i]
                contact = base + GRASPGEN_MARKER_LENGTH_M * R[:, 2]   # +Z approach -> TCP
                best = (i == 0)
                r, g, b = (0.0, 1.0, 0.0) if best else (0.5, 0.5, 0.0)
                a = 1.0 if best else 0.5

                arrow = Marker()
                arrow.header.frame_id = frame
                arrow.header.stamp = stamp
                arrow.ns = 'graspgen_approach'
                arrow.id = i
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.points = [
                    Point(x=float(base[0]), y=float(base[1]), z=float(base[2])),
                    Point(x=float(contact[0]), y=float(contact[1]),
                          z=float(contact[2]))]
                arrow.scale.x = 0.008                  # shaft diameter
                arrow.scale.y = 0.018                  # head diameter
                arrow.scale.z = 0.03                   # head length
                arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = r, g, b, a
                arr.markers.append(arrow)

                dot = Marker()
                dot.header.frame_id = frame
                dot.header.stamp = stamp
                dot.ns = 'graspgen_contact'
                dot.id = i
                dot.type = Marker.SPHERE
                dot.action = Marker.ADD
                dot.pose.position = Point(
                    x=float(contact[0]), y=float(contact[1]), z=float(contact[2]))
                dot.pose.orientation.w = 1.0
                dot.scale.x = dot.scale.y = dot.scale.z = 0.02
                dot.color.r, dot.color.g, dot.color.b, dot.color.a = r, g, b, a
                arr.markers.append(dot)
            self._marker_pub.publish(arr)
        except Exception as e:                          # viz must never break planning
            self.get_logger().warn(f'grasp marker publish failed: {e}')

    def _save_grasp_viz(self, rec, pts, grasps, mean, width):
        """Render an orthographic 'photo' of the object cloud + the generated
        grasps, drawn with cv2 (the ros image's matplotlib 3D backend is broken).
        Two views (front Y-Z, side X-Z) side by side; the top grasps are drawn as
        gripper frames — red = finger-opening axis, blue = approach axis."""
        if not self.logger.visualize:
            return
        
        try:
            G = np.asarray([np.asarray(g, dtype=np.float64).reshape(4, 4) for g in grasps])
            centers = G[:, :3, 3] + mean          # un-center back to the input frame
            # GraspGenX pose: +Z approaches the object, +X is the finger-closing
            # axis (see GraspGen.srv). Build line segments for the top grasps.
            topk = min(5, len(G))
            half = max(0.01, float(width) / 2.0)
            approach_len = 0.06
            segments = []                          # (p_a[3], p_b[3], bgr)
            for i in range(topk):
                R, p = G[i, :3, :3], centers[i]
                segments.append((p - half * R[:, 0], p + half * R[:, 0], (0, 0, 255)))
                segments.append((p - approach_len * R[:, 2], p, (255, 0, 0)))
            panels = [
                self._render_grasp_panel(cv2, pts, centers, segments, ai, bi,
                                         f'{label}  (top {topk}/{len(G)})')
                for (ai, bi), label in (((1, 2), 'front Y-Z'), ((0, 2), 'side X-Z'))
            ]
            cv2.imwrite(rec.path('grasps', 'png'), np.hstack(panels))
        except Exception as e:
            self.get_logger().warn(f'grasp viz failed: {e}')

    @staticmethod
    def _render_grasp_panel(cv2, pts, centers, segments, ai, bi, label,
                            size=520, pad=36):
        """One orthographic view (axes ai=horizontal, bi=vertical, up = up): gray
        cloud dots + colored grasp segments on a white canvas. Uniform scale so the
        two panels share geometry sense."""
        view = np.vstack([pts[:, [ai, bi]], centers[:, [ai, bi]]])
        mins, maxs = view.min(axis=0), view.max(axis=0)
        c = (mins + maxs) / 2.0
        scale = (size - 2 * pad) / (float((maxs - mins).max()) or 0.1)

        def to_px(h, v):
            return (int((h - c[0]) * scale + size / 2.0),
                    int(size / 2.0 - (v - c[1]) * scale))   # flip y so up is up

        img = np.full((size, size, 3), 255, np.uint8)
        for h, v in pts[:, [ai, bi]]:
            cv2.circle(img, to_px(h, v), 1, (170, 170, 170), -1)
        for a, b, color in segments:
            cv2.line(img, to_px(a[ai], a[bi]), to_px(b[ai], b[bi]), color, 2)
        cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 1, cv2.LINE_AA)
        return img


def main():
    rclpy.init()
    node = GraspGenServer()
    # MultiThreadedExecutor (matching gemini/sam servers) so a long GPU plan on the
    # service thread doesn't block discovery, latched-marker publishing, or a second
    # request. The service runs in a ReentrantCallbackGroup (see __init__).
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
