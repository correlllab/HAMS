#!/usr/bin/env python3
"""Per-trial telemetry recorder: pelvis pose (sway), right graspgenx-frame xyz
(gripper trajectory), door pose — ~10 Hz wall — until SIGTERM. CSV to stdout.
Runs INSIDE hams_ros alongside a trial; the harness redirects to
trial_NN_telemetry.csv."""
import json
import math
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

run = {'on': True}
signal.signal(signal.SIGTERM, lambda *_: run.__setitem__('on', False))
signal.signal(signal.SIGINT, lambda *_: run.__setitem__('on', False))

rclpy.init()
n = Node('trial_recorder')
buf = Buffer(); TransformListener(buf, n)
state = {'d': None}
n.create_subscription(String, '/robocasa/object_poses',
                      lambda m: state.__setitem__('d', m.data), 10)

print('t_wall,pelvis_x,pelvis_y,pelvis_z,pelvis_yaw_deg,grip_x,grip_y,grip_z,door_x,door_y,door_z', flush=True)
t0 = time.time()
last = 0.0
while run['on']:
    rclpy.spin_once(n, timeout_sec=0.05)
    now = time.time()
    if now - last < 0.1:
        continue
    last = now
    px = py = pz = yaw = gx = gy = gz = dx = dy = dz = float('nan')
    if state['d']:
        try:
            d = json.loads(state['d'])
            p = d.get('__pelvis__')
            if p:
                px, py, pz = p[:3]
                w, x, y, z = p[3:7]
                yaw = math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
            do = d.get('door_obj')
            if do:
                dx, dy, dz = do[:3]
        except Exception:
            pass
    try:
        tf = buf.lookup_transform('pelvis', 'right_graspgenx_frame', Time(),
                                  timeout=Duration(seconds=0.05))
        gx = tf.transform.translation.x
        gy = tf.transform.translation.y
        gz = tf.transform.translation.z
    except Exception:
        pass
    print(f'{now-t0:.2f},{px:.4f},{py:.4f},{pz:.4f},{yaw:.2f},{gx:.4f},{gy:.4f},{gz:.4f},{dx:.4f},{dy:.4f},{dz:.4f}', flush=True)
