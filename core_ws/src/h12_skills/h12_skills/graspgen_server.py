#!/usr/bin/env python3
"""graspgen_server — ROS2 service exposing GraspGenX grasp planning.

Serves custom_ros_messages/srv/GraspGen on `graspgen`: a segmented object
PointCloud2 in, ranked 6-DOF grasp poses out. The GraspGenXSampler (heavy GPU
model) is built ONCE per gripper and cached. Checkpoints/assets resolve the same
way as graspgenx_smoke_test.py (GRASPGENX_CHECKPOINT_DIR; assets default
/opt/graspgenx/assets, where the magpie gripper description is staged).

Runs inside the ros container (GraspGenX is pip-installed there). Import of
graspgenx is deferred to __init__ so the module can be inspected without it.
"""

import inspect
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import Point, PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray

from custom_ros_messages.srv import GraspGen

from .model_logging import ModelLogger, declare_logging_params
from .perception_utils import mat_to_quat


# Default gripper for planning, env-overridable so it can switch to the magpie
# description once that's staged in the GraspGenX assets (no rebuild). The shipped
# parallel_2f_v1_1002 (a parallel-jaw gripper, like magpie) exists out of the box;
# the GraspGen.srv `gripper_name` field overrides per request.
DEFAULT_GRIPPER = os.environ.get('GRASPGENX_GRIPPER', 'parallel_2f_v1_1002')
MIN_OBJECT_POINTS = 100
# RViz grasp markers: how many of the ranked grasps to draw. Each is drawn as an
# arrow from the GraspGenX pose ORIGIN (the gripper base, where the IK pins
# *_graspgen_site) along +Z (approach) to the CONTACT point where the fingers
# close — so the arrow TIP sits on the object, i.e. "where the gripper point is".
# GRASP_TCP_DEPTH_M is that base->contact distance; magpie = 0.193 m (its config
# fingertip = [0,0,0.193]). Env-overridable for a different gripper.
N_GRASP_MARKERS = 5
GRASP_TCP_DEPTH_M = float(os.environ.get('GRASPGEN_TCP_DEPTH_M', '0.193'))
# Candidate planner kwargs (filtered to run_planner_on_object's real signature,
# exactly as graspgenx_smoke_test.py does).
PLANNER_KWARGS = dict(
    planner='graspmoe', grasp_threshold=-1.0, num_grasps=200,
    topk_num_grasps=100, moe_num_yaws=36, moe_z_offsets_cm=(-2.0, 0.0),
    moe_outlier_threshold=0.014, moe_outlier_k=20, moe_obb_mode='advanced',
    moe_skip_obb_rule='auto', moe_obb_density='dense-topandside',
    moe_obb_position_spacing_cm=1.0,
)


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
        self.logger = ModelLogger(self, 'graspgen', 'h12_skills', __file__,
                                  log=log, visualize=viz, clear=clear)
        from graspgenx.samplers import run_planner_on_object
        from graspgenx.utils.checkpoint_io import load_model_cfg
        self._run_planner = run_planner_on_object
        self._GraspGenXSampler = None  # imported lazily with the first sampler

        ckpt_root = os.environ.get(
            'GRASPGENX_CHECKPOINT_DIR', '/home/code/core_ws/src/h12_skills/weights')
        checkpoints = os.path.join(ckpt_root, 'release')
        self._assets_dir = os.environ.get('GRASPGENX_ASSETS_DIR', '/opt/graspgenx/assets')
        self.get_logger().info(f'Loading GraspGenX model cfg from {checkpoints} ...')
        self._model_cfg = load_model_cfg(
            os.path.join(checkpoints, 'gen'), os.path.join(checkpoints, 'dis'), None, None)

        # Keep only the planner kwargs the installed run_planner_on_object accepts.
        sig = inspect.signature(run_planner_on_object)
        self._kwargs = {k: v for k, v in PLANNER_KWARGS.items() if k in sig.parameters}

        self._samplers = {}           # gripper_name -> (sampler, width_m)
        try:
            self._get_sampler(DEFAULT_GRIPPER)   # warm the default + validate load
        except Exception as e:
            # Don't take the node down — keep serving so a valid per-request
            # gripper_name still works (that first call pays the load cost).
            self.get_logger().error(
                f'default gripper {DEFAULT_GRIPPER!r} failed to load ({e}); set '
                f'GRASPGENX_GRIPPER or pass gripper_name in the request')

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

        self.create_service(GraspGen, 'graspgen', self._plan_cb)
        self.get_logger().info("graspgen_server ready on 'graspgen'")

    def _get_sampler(self, gripper_name):
        name = gripper_name or DEFAULT_GRIPPER
        if name not in self._samplers:
            if self._GraspGenXSampler is None:
                from graspgenx.grasp_server import GraspGenXSampler
                self._GraspGenXSampler = GraspGenXSampler
            self.get_logger().info(f'Building GraspGenXSampler (gripper={name}) ...')
            sampler = self._GraspGenXSampler(
                self._model_cfg, name, assets_dir=self._assets_dir)
            self._samplers[name] = (sampler, _sampler_width(sampler))
        return self._samplers[name]

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

        try:
            sampler, width = self._get_sampler(request.gripper_name)
        except Exception as e:
            response.success, response.message = False, f'sampler load failed: {e}'
            self.get_logger().error(response.message)
            rec.finish(success=False, message=response.message)
            return response

        mean = pts.mean(axis=0)
        pc_centered = (pts - mean).astype(np.float32)
        try:
            out = self._run_planner(pc_centered, sampler, **self._kwargs)
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
        rec.set(n_grasps=len(response.grasps), gripper_width=float(width))
        rec.save_array('grasps', grasps)            # raw (M, 4, 4) in centered frame
        rec.save_array('scores', confs)
        self._publish_grasp_markers(frame, grasps, mean)
        self._save_grasp_viz(rec, pts, grasps, mean, width)
        rec.finish(success=True, message=response.message)
        return response

    def _publish_grasp_markers(self, frame, grasps, mean):
        """Publish the generated grasps as RViz markers on 'graspgen_markers'
        (latched), in `frame` (pelvis). For each ranked grasp:
          - an ARROW from the GraspGenX pose ORIGIN (gripper base, where the IK
            pins *_graspgen_site) along +Z to the CONTACT point, so the arrow TIP
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
                contact = base + GRASP_TCP_DEPTH_M * R[:, 2]   # +Z approach -> TCP
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
            import cv2
        except Exception as e:
            self.get_logger().warn(f'grasp viz needs cv2: {e}')
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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
