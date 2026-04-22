#!/usr/bin/env python3
"""Detect red cube via vision pipeline, then move right arm to it."""

import sys
import time
import struct
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, Point, Quaternion
from builtin_interfaces.msg import Duration

from custom_ros_messages.srv import UpdateTrackedObject, UpdateBeliefs, Query
from custom_ros_messages.action import FrameTask


class CubeMover(Node):
    def __init__(self):
        super().__init__('cube_mover')

        # Service clients
        self.track_cli = self.create_client(UpdateTrackedObject, '/vp_update_tracked_object')
        self.beliefs_cli = self.create_client(UpdateBeliefs, '/vp_update_beliefs')
        self.query_cli = self.create_client(Query, '/vp_query_tracked_objects')

        # Action client
        self.frame_task_cli = ActionClient(self, FrameTask, '/frame_task')

        self.get_logger().info('Waiting for services...')
        self.track_cli.wait_for_service(timeout_sec=10.0)
        self.beliefs_cli.wait_for_service(timeout_sec=10.0)
        self.query_cli.wait_for_service(timeout_sec=10.0)
        self.get_logger().info('Waiting for frame_task action server...')
        self.frame_task_cli.wait_for_server(timeout_sec=10.0)
        self.get_logger().info('All services ready.')

    def call_service(self, client, request, name='service'):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        if future.result() is None:
            self.get_logger().error(f'{name} call failed or timed out')
            return None
        return future.result()

    def add_track_string(self, obj_name, action='add'):
        req = UpdateTrackedObject.Request()
        req.object = obj_name
        req.action = action
        result = self.call_service(self.track_cli, req, 'UpdateTrackedObject')
        if result:
            self.get_logger().info(f'Track "{obj_name}": {result.result} — {result.message}')
        return result

    def update_beliefs(self, camera='/realsense/head'):
        req = UpdateBeliefs.Request()
        req.camera_name_space = camera
        result = self.call_service(self.beliefs_cli, req, 'UpdateBeliefs')
        if result:
            self.get_logger().info(f'UpdateBeliefs: success={result.success} — {result.message}')
        return result

    def query_object(self, query_str, confidence=0.3):
        req = Query.Request()
        req.query = query_str
        req.confidence_threshold = confidence
        req.pc_name = ''
        result = self.call_service(self.query_cli, req, 'Query')
        if result:
            self.get_logger().info(
                f'Query "{query_str}": success={result.success}, '
                f'{len(result.clouds)} clouds, names={list(result.names)}, '
                f'probs={list(result.probabilities)}'
            )
            if result.success and result.clouds:
                cloud = result.clouds[0]
                self.get_logger().info(
                    f'  Cloud frame: {cloud.header.frame_id}, '
                    f'points: {cloud.width * cloud.height}, '
                    f'point_step: {cloud.point_step}'
                )
                centroid = self._compute_centroid(cloud)
                if centroid is not None:
                    self.get_logger().info(f'  Centroid (pelvis frame): x={centroid[0]:.3f} y={centroid[1]:.3f} z={centroid[2]:.3f}')
                return result, centroid
        return result, None

    def _compute_centroid(self, cloud):
        n_points = cloud.width * cloud.height
        if n_points == 0:
            return None
        data = bytes(cloud.data)
        step = cloud.point_step
        xs, ys, zs = [], [], []
        for i in range(n_points):
            offset = i * step
            x = struct.unpack_from('f', data, offset + 0)[0]
            y = struct.unpack_from('f', data, offset + 4)[0]
            z = struct.unpack_from('f', data, offset + 8)[0]
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                xs.append(x)
                ys.append(y)
                zs.append(z)
        if not xs:
            return None
        return (np.mean(xs), np.mean(ys), np.mean(zs))

    def send_frame_task(self, frame_name, x, y, z, duration_sec=3):
        goal = FrameTask.Goal()
        goal.frame_names = [frame_name]
        pose = Pose()
        pose.position = Point(x=float(x), y=float(y), z=float(z))
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        goal.frame_targets = [pose]
        goal.duration = Duration(sec=duration_sec, nanosec=0)

        self.get_logger().info(f'Sending frame_task: {frame_name} → ({x:.3f}, {y:.3f}, {z:.3f}) over {duration_sec}s')
        send_future = self.frame_task_cli.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return False

        self.get_logger().info('Goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=duration_sec + 10.0)
        result = result_future.result()
        status = result.status if result else GoalStatus.STATUS_UNKNOWN
        status_str = {
            GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED',
            GoalStatus.STATUS_ABORTED: 'ABORTED',
            GoalStatus.STATUS_CANCELED: 'CANCELED',
        }.get(status, f'UNKNOWN({status})')
        self.get_logger().info(f'frame_task result: {status_str}')
        return status == GoalStatus.STATUS_SUCCEEDED


def main():
    rclpy.init()
    node = CubeMover()

    try:
        # Step 1: Add red cube tracking
        node.get_logger().info('=== Step 1: Add track string ===')
        node.add_track_string('red cube')
        time.sleep(1.0)

        # Step 2: Update beliefs from head camera
        node.get_logger().info('=== Step 2: Update beliefs ===')
        node.update_beliefs('/realsense/head')
        time.sleep(2.0)

        # Step 3: Query for red cube
        node.get_logger().info('=== Step 3: Query red cube ===')
        result, centroid = node.query_object('red cube', confidence=0.3)


        assert centroid is not None
        node.get_logger().info(f'VP centroid: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})')
        

        # Known cube position in world frame: (0.35, -0.2, 1.0)
        # Pelvis is at world origin for fixed-pelvis scene, at height ~1.05
        # So cube in pelvis frame ≈ (0.35, -0.2, 1.0 - 1.05) = (0.35, -0.2, -0.05)
        # Use VP centroid if available, else fallback to known geometry
        
        cx, cy, cz = centroid
        node.get_logger().info(f'Using VP centroid for target')
       

        # Step 4: Move right arm above the cube first
        node.get_logger().info('=== Step 4: Move arm above cube ===')
        above_z = max(cz + 0.15, 0.0)  # above the cube
        ok = node.send_frame_task('right_wrist_yaw_link', cx, cy, above_z, duration_sec=4)
        if not ok:
            node.get_logger().error('Failed to move above cube')

        time.sleep(1.0)

        # Step 5: Lower arm toward the cube
        node.get_logger().info('=== Step 5: Lower arm to cube ===')
        target_z = max(cz, -0.10)  # clamp to avoid joint limit abort
        ok = node.send_frame_task('right_wrist_yaw_link', cx, cy, target_z, duration_sec=3)
        if not ok:
            node.get_logger().warn('Lower motion aborted (may be at joint limit)')

        node.get_logger().info('=== Done ===')

    except Exception as e:
        node.get_logger().error(f'Exception: {e}')
        import traceback
        traceback.print_exc()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
