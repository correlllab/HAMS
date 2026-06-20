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

from geometry_msgs.msg import PoseStamped
from sensor_msgs_py import point_cloud2

from custom_ros_messages.srv import GraspGen

from .perception_utils import mat_to_quat


# Default gripper for planning, env-overridable so it can switch to the magpie
# description once that's staged in the GraspGenX assets (no rebuild). The shipped
# parallel_2f_v1_1002 (a parallel-jaw gripper, like magpie) exists out of the box;
# the GraspGen.srv `gripper_name` field overrides per request.
DEFAULT_GRIPPER = os.environ.get('GRASPGENX_GRIPPER', 'parallel_2f_v1_1002')
MIN_OBJECT_POINTS = 100
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
        frame = request.object_cloud.header.frame_id
        try:
            pts = point_cloud2.read_points_numpy(
                request.object_cloud, field_names=('x', 'y', 'z'), skip_nans=True)
        except Exception as e:                       # malformed cloud
            response.success, response.message = False, f'bad cloud: {e}'
            return response
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        if pts.shape[0] < MIN_OBJECT_POINTS:
            response.success, response.message = False, f'too few points ({pts.shape[0]})'
            return response

        try:
            sampler, width = self._get_sampler(request.gripper_name)
        except Exception as e:
            response.success, response.message = False, f'sampler load failed: {e}'
            self.get_logger().error(response.message)
            return response

        mean = pts.mean(axis=0)
        pc_centered = (pts - mean).astype(np.float32)
        try:
            out = self._run_planner(pc_centered, sampler, **self._kwargs)
        except Exception as e:
            response.success, response.message = False, f'planner failed: {e}'
            self.get_logger().error(response.message)
            return response

        grasps = np.asarray(out[0] if isinstance(out, (tuple, list)) else out)
        # Guard shape BEFORE any len()/indexing — a 0-d/None-ish return must fail
        # cleanly, not crash the service thread (grasps.ndim is safe on 0-d).
        if grasps.ndim != 3 or grasps.shape[0] == 0:
            response.success, response.message = False, 'no grasps produced'
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
        return response


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
