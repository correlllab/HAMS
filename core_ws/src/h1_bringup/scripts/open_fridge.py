#!/usr/bin/env python3
"""Detect the fridge handle via the vision pipeline, then drive the right arm to open it."""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, Point, Quaternion, PointStamped, PoseStamped
from builtin_interfaces.msg import Duration

from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_point

from custom_ros_messages.srv import UpdateTrackedObject, UpdateBeliefs, Query
from custom_ros_messages.action import FrameTask
from nav2_msgs.action import NavigateToPose
from magpie_msgs.srv import SetGripperPosition
import struct
import numpy as np


TARGET_QUERY = 'fridge handle'
CAMERA_NS = '/realsense/head'
ARM_FRAME = 'right_wrist_yaw_link'
GRIPPER_SRV = '/gripper/right/set_position'


class FridgeOpener(Node):
    def __init__(self):
        super().__init__('fridge_opener')

        self.track_cli = self.create_client(UpdateTrackedObject, '/vp_update_tracked_object')
        self.beliefs_cli = self.create_client(UpdateBeliefs, '/vp_update_beliefs')
        self.query_cli = self.create_client(Query, '/vp_query_tracked_objects')
        self.frame_task_cli = ActionClient(self, FrameTask, '/frame_task')
        self.gripper_cli = self.create_client(SetGripperPosition, GRIPPER_SRV)
        self.nav_cli = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info('Waiting for VP services and frame_task action server...')
        self.track_cli.wait_for_service(timeout_sec=10.0)
        self.beliefs_cli.wait_for_service(timeout_sec=10.0)
        self.query_cli.wait_for_service(timeout_sec=10.0)
        self.frame_task_cli.wait_for_server(timeout_sec=10.0)
        self.gripper_cli.wait_for_service(timeout_sec=10.0)
        self.nav_cli.wait_for_server(timeout_sec=10.0)
        self.get_logger().info('All endpoints ready.')

    def _call(self, client, request, name):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        if future.result() is None:
            self.get_logger().error(f'{name} call failed or timed out')
            return None
        return future.result()

    def update_beliefs(self, camera):
        req = UpdateBeliefs.Request()
        req.camera_name_space = camera
        return self._call(self.beliefs_cli, req, 'UpdateBeliefs')

    def query_centroid(self, q, target_frame='pelvis'):
        req = Query.Request()
        req.query = q
        req.confidence_threshold = 0.3
        req.pc_name = ''
        result = self._call(self.query_cli, req, 'Query')
        if not result or not result.success or not result.clouds:
            return None
        cloud = result.clouds[0]
        centroid = _centroid_from_cloud(cloud)
        if centroid is None:
            return None
        return self._transform_point(centroid, cloud.header.frame_id, cloud.header.stamp, target_frame)

    def _transform_point(self, xyz, source_frame, stamp, target_frame):
        pt = PointStamped()
        pt.header.frame_id = source_frame
        pt.header.stamp = stamp
        pt.point.x, pt.point.y, pt.point.z = xyz
        # Spin briefly so the tf listener can collect recent transforms.
        rclpy.spin_once(self, timeout_sec=0.1)
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, stamp,
                timeout=RclpyDuration(seconds=1.0)
            )
        except TransformException as e:
            self.get_logger().error(f'TF lookup {source_frame!r} -> {target_frame!r} failed: {e}')
            return None
        transformed = do_transform_point(pt, transform)
        return (float(transformed.point.x), float(transformed.point.y), float(transformed.point.z))

    def _frame_task_feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        lin = ', '.join(f'{e:.4f}' for e in fb.errors_linear)
        ang = ', '.join(f'{e:.4f}' for e in fb.errors_angular)
        print(f'\rframe_task feedback: errors_linear=[{lin}] errors_angular=[{ang}]', end='', flush=True)

    def move_frame_to(self, x, y, z, duration_sec=3):
        goal = FrameTask.Goal()
        goal.frame_names = [ARM_FRAME]
        pose = Pose()
        pose.position = Point(x=float(x), y=float(y), z=float(z))
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        goal.frame_targets = [pose]
        goal.duration = Duration(sec=duration_sec, nanosec=0)

        self.get_logger().info(f'frame_task: {ARM_FRAME} -> ({x:.3f}, {y:.3f}, {z:.3f}) in {duration_sec}s')
        send_future = self.frame_task_cli.send_goal_async(goal, feedback_callback=self._frame_task_feedback_cb)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error('frame_task goal rejected')
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=duration_sec + 10.0)
        result = result_future.result()
        status = result.status if result else GoalStatus.STATUS_UNKNOWN
        return status == GoalStatus.STATUS_SUCCEEDED

    def _nav_feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        print(
            f'\rnav feedback: distance_remaining={fb.distance_remaining:.3f} m '
            f'recoveries={fb.number_of_recoveries}',
            end='', flush=True,
        )

    def navigate_to(self, x, y, yaw=0.0, frame='map', timeout_sec=120.0):
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position = Point(x=float(x), y=float(y), z=0.0)
        half = 0.5 * float(yaw)
        goal.pose.pose.orientation = Quaternion(
            x=0.0, y=0.0, z=float(np.sin(half)), w=float(np.cos(half))
        )

        self.get_logger().info(f'navigate_to_pose -> ({x:.3f}, {y:.3f}, yaw={yaw:.3f}) in {frame!r}')
        send_future = self.nav_cli.send_goal_async(goal, feedback_callback=self._nav_feedback_cb)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error('navigate_to_pose goal rejected')
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout_sec)
        result = result_future.result()
        status = result.status if result else GoalStatus.STATUS_UNKNOWN
        return status == GoalStatus.STATUS_SUCCEEDED

    def set_gripper(self, position_mm, speed=1.0):
        req = SetGripperPosition.Request()
        req.position = float(position_mm)
        req.speed = float(speed)
        result = self._call(self.gripper_cli, req, 'SetGripperPosition')
        if result is None:
            return False
        self.get_logger().info(
            f'gripper set: success={result.success} actual_position={result.actual_position:.2f} mm — {result.message}'
        )
        return result.success

    def open_gripper(self, position_mm=85.0, speed=1.0):
        return self.set_gripper(position_mm, speed)

    def close_gripper(self, position_mm=0.0, speed=1.0):
        return self.set_gripper(position_mm, speed)


def _centroid_from_cloud(cloud):
    n = cloud.width * cloud.height
    if n == 0:
        return None
    data = bytes(cloud.data)
    step = cloud.point_step
    xs, ys, zs = [], [], []
    for i in range(n):
        off = i * step
        x = struct.unpack_from('f', data, off + 0)[0]
        y = struct.unpack_from('f', data, off + 4)[0]
        z = struct.unpack_from('f', data, off + 8)[0]
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
            xs.append(x); ys.append(y); zs.append(z)
    if not xs:
        return None
    return (float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs)))


def main():
    rclpy.init()
    node = FridgeOpener()
    try:
        node.get_logger().info('=== Navigate to (0, 0, 0) ===')
        print(f"{node.nav_cli=}")
        if not node.navigate_to(0.0, 0.0, yaw=0.0):
            node.get_logger().error('Navigation failed')


        node.get_logger().info('=== Update beliefs from head camera ===')
        node.update_beliefs("/realsense/head")
        node.get_clock().sleep_for(RclpyDuration(seconds=2.0))

        node.get_logger().info('=== Query handle centroid ===')
        centroid = node.query_centroid(TARGET_QUERY, target_frame='pelvis')
        if centroid is None:
            node.get_logger().error(f'No {TARGET_QUERY} detected — aborting.')
            return
        hx, hy, hz = centroid
        node.get_logger().info(f'{TARGET_QUERY} centroid (pelvis frame): ({hx:.3f}, {hy:.3f}, {hz:.3f})')

        # hx = 1.0
        # hy = -0.1
        # hz = 0.1

        node.get_logger().info('=== Open gripper ===')
        if not node.open_gripper():
            node.get_logger().error('Gripper open failed.')
            return
        node.get_clock().sleep_for(RclpyDuration(seconds=1.0))

        node.get_logger().info('=== Approach handle ===')
        approach_x = hx - 0.5  # 25 cm in front of handle along robot +x
        if not node.move_frame_to(approach_x, hy, hz, duration_sec=4):
            node.get_logger().error('Approach motion failed.')
            return

        node.get_logger().info('=== Contact handle ===')
        if not node.move_frame_to(hx-0.25, hy, hz, duration_sec=2):
            node.get_logger().error('Contact motion failed.')
            return

        node.get_logger().info('=== Close gripper on handle ===')
        if not node.close_gripper(position_mm=0.0, speed=1.0):
            node.get_logger().error('Gripper close failed.')
            return
        node.get_clock().sleep_for(RclpyDuration(seconds=5.0))

        node.get_logger().info('=== Pull door open ===')
        # Naive: pull straight back along -x by 25 cm.
        if not node.move_frame_to(0, hy, hz, duration_sec=4):
            node.get_logger().error('Pull motion failed.')
            return

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
