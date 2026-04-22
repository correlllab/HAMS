#!/usr/bin/env python3
"""Detect red cube via vision pipeline, then push it off the table.

Usage (after sourcing core_ws):
    ros2 run h1_bringup push_red_cube
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, Point, Quaternion
from builtin_interfaces.msg import Duration

from custom_ros_messages.srv import UpdateTrackedObject, UpdateBeliefs, Query
from custom_ros_messages.action import FrameTask


class CubePusher(Node):
    def __init__(self):
        super().__init__('cube_pusher')
        self.track_cli = self.create_client(UpdateTrackedObject, '/vp_update_tracked_object')
        self.beliefs_cli = self.create_client(UpdateBeliefs, '/vp_update_beliefs')
        self.query_cli = self.create_client(Query, '/vp_query_tracked_objects')
        self.frame_task_cli = ActionClient(self, FrameTask, '/frame_task')

        self.get_logger().info('Waiting for services...')
        self.track_cli.wait_for_service(timeout_sec=30.0)
        self.beliefs_cli.wait_for_service(timeout_sec=30.0)
        self.query_cli.wait_for_service(timeout_sec=30.0)
        self.frame_task_cli.wait_for_server(timeout_sec=30.0)
        self.get_logger().info('All services ready.')

    def call_service(self, client, request, name='service'):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)
        if future.result() is None:
            self.get_logger().error(f'{name} timed out')
            return None
        return future.result()

    def add_track_string(self, obj_name):
        req = UpdateTrackedObject.Request()
        req.object = obj_name
        req.action = 'add'
        r = self.call_service(self.track_cli, req, 'Track')
        if r:
            self.get_logger().info(f'Track "{obj_name}": {r.message}')

    def update_beliefs(self, camera='/realsense/head'):
        req = UpdateBeliefs.Request()
        req.camera_name_space = camera
        r = self.call_service(self.beliefs_cli, req, 'Beliefs')
        if r:
            self.get_logger().info(f'Beliefs: {r.message}')

    def query_object(self, query_str, confidence=0.3):
        req = Query.Request()
        req.query = query_str
        req.confidence_threshold = confidence
        req.pc_name = ''
        r = self.call_service(self.query_cli, req, 'Query')
        if r and r.success and r.clouds:
            n = r.clouds[0].width * r.clouds[0].height
            self.get_logger().info(f'Detected "{query_str}": {n} pts, conf={r.probabilities[0]:.2f}')
            return True
        return False

    def move(self, label, x, y, z, dur=3):
        goal = FrameTask.Goal()
        goal.frame_names = ['right_wrist_yaw_link']
        goal.frame_targets = [Pose(
            position=Point(x=float(x), y=float(y), z=float(z)),
            orientation=Quaternion(w=1.0)
        )]
        goal.duration = Duration(sec=dur)

        self.get_logger().info(f'{label}: ({x:.3f}, {y:.3f}, {z:.3f}) {dur}s')
        f = self.frame_task_cli.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, f, timeout_sec=10.0)
        gh = f.result()
        if not gh or not gh.accepted:
            self.get_logger().error(f'  REJECTED')
            return False
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf, timeout_sec=dur + 10.0)
        r = rf.result()
        ok = r and r.status == GoalStatus.STATUS_SUCCEEDED
        self.get_logger().info(f'  {"OK" if ok else "ABORT"}')
        return ok


def main():
    rclpy.init()
    node = CubePusher()

    # Scene geometry (scene_handless_pelvis_fixed.xml):
    #   Table at world (0.45, 0, 0.5), surface z = 1.025
    #   Red cube at world (0.45, -0.1, 1.07) — on the table
    #   Pelvis welded at world origin, z ≈ 1.05
    #   → Cube in pelvis frame: (0.45, -0.1, 0.02)
    #   → Table right edge y = -0.35 in pelvis frame
    #   → Default right wrist rests at ~(0.33, -0.37, 0.03)
    CUBE_X, CUBE_Y, CUBE_Z = 0.45, -0.1, 0.02
    TABLE_EDGE_Y = -0.35
    PUSH_HEIGHT = 0.05  # slightly above cube center to avoid table collision

    try:
        # --- Detect red cube via Vision Pipeline ---
        node.get_logger().info('=== DETECT RED CUBE ===')
        node.add_track_string('red cube')
        time.sleep(1.0)
        node.update_beliefs('/realsense/head')
        time.sleep(2.0)
        detected = node.query_object('red cube')
        if detected:
            node.get_logger().info('VP confirmed red cube on table!')
        else:
            node.get_logger().warn('VP miss — proceeding with known geometry')

        # --- Move arm to push position ---
        # From default rest (0.33, -0.37, 0.03), move in small steps:

        # Step 1: Forward to cube x, keep y near default (arm's side)
        node.move('Forward', CUBE_X, -0.30, PUSH_HEIGHT, dur=3)
        time.sleep(0.3)

        # Step 2: Swing inward past center to inside of cube
        node.move('Inside', CUBE_X, CUBE_Y + 0.12, PUSH_HEIGHT, dur=3)
        time.sleep(0.3)

        # Step 3: PUSH — sweep outward through cube, past table edge
        node.get_logger().info('=== PUSHING CUBE OFF TABLE ===')
        ok = node.move('PUSH', CUBE_X, TABLE_EDGE_Y - 0.10, PUSH_HEIGHT, dur=2)

        if ok:
            node.get_logger().info('*** Cube pushed off the table! ***')
        else:
            node.get_logger().warn('Push motion incomplete')

        node.get_logger().info('=== Done ===')

    except Exception as e:
        node.get_logger().error(f'{e}')
        import traceback
        traceback.print_exc()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
