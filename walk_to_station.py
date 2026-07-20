#!/usr/bin/env python3
"""Station-keep the ALMI-standing robot at the grasp station: closed loop on
pelvis x, y AND yaw (from /robocasa/object_poses __pelvis__), then stop and
stand. The grasp pipeline needs the robot AT the station FACING the fridge —
x alone is not enough (ALMI's gait drifts y/yaw while walking).
Exits 0 when parked within tolerance, 1 on timeout. Runs INSIDE hams_ros."""
import json
import math
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

TARGET = (3.00, -2.787, 0.0)   # world x, y, yaw(rad) — frozen-tier station, facing +x
POS_TOL = 0.05
YAW_TOL = math.radians(6)
TIMEOUT_S = 300

rclpy.init()
n = Node('walk_to_station')
state = {'p': None}
n.create_subscription(String, '/robocasa/object_poses',
                      lambda m: state.__setitem__('p', json.loads(m.data).get('__pelvis__')), 10)
pub = n.create_publisher(Twist, '/cmd_vel', 10)

def spin(sec):
    t0 = time.time()
    while time.time() - t0 < sec:
        rclpy.spin_once(n, timeout_sec=0.05)

def yaw_of(q):  # wxyz
    w, x, y, z = q
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

spin(3.0)
if state['p'] is None:
    print('no pelvis feed'); sys.exit(1)

t0 = time.time()
parked = False
cmd = Twist()
while time.time() - t0 < TIMEOUT_S:
    p = state['p']
    yaw = yaw_of(p[3:7])
    ex, ey = TARGET[0] - p[0], TARGET[1] - p[1]
    eyaw = (TARGET[2] - yaw + math.pi) % (2*math.pi) - math.pi
    if math.hypot(ex, ey) <= POS_TOL and abs(eyaw) <= YAW_TOL:
        parked = True
        break
    # world error -> robot frame
    bx =  math.cos(yaw)*ex + math.sin(yaw)*ey
    by = -math.sin(yaw)*ex + math.cos(yaw)*ey
    # yaw first when badly misaligned, else translate with small yaw correction
    if abs(eyaw) > math.radians(25):
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.angular.z = 0.35 if eyaw > 0 else -0.35
    else:
        cmd.linear.x = max(-0.25, min(0.25, 1.2*bx)) if abs(bx) > POS_TOL/2 else 0.0
        cmd.linear.y = max(-0.15, min(0.15, 1.2*by)) if abs(by) > POS_TOL/2 else 0.0
        cmd.angular.z = max(-0.3, min(0.3, 1.0*eyaw))
        # keep ||cmd|| above the 0.1 gait threshold while any error remains
        if math.hypot(cmd.linear.x, cmd.linear.y) < 0.12 and abs(cmd.angular.z) < 0.12:
            if abs(bx) >= abs(by):
                cmd.linear.x = 0.12 if bx >= 0 else -0.12
            else:
                cmd.linear.y = 0.12 if by >= 0 else -0.12
    pub.publish(cmd)
    spin(0.4)

cmd.linear.x = 0.0; cmd.linear.y = 0.0; cmd.angular.z = 0.0
for _ in range(15):
    pub.publish(cmd)
    spin(0.2)
spin(8.0)
p = state['p']; yaw = yaw_of(p[3:7])
print(f'parked={parked} x={p[0]:.3f} y={p[1]:.3f} yaw={math.degrees(yaw):.1f}deg')
ok = math.hypot(TARGET[0]-p[0], TARGET[1]-p[1]) <= 2*POS_TOL and abs(yaw) <= 2*YAW_TOL
sys.exit(0 if ok else 1)
