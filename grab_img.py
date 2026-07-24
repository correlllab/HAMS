#!/usr/bin/env python3
"""Grab one head-camera frame + report GT screw feed. Saves PNG to /tmp/out.png."""
import sys, time, threading
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2

OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/out.png'
TOPIC = sys.argv[2] if len(sys.argv) > 2 else '/realsense/head/color/image_raw/compressed'

rclpy.init()
n = Node('grab_img')
state = {'img': None, 'poses': None}

def on_img(msg):
    arr = np.frombuffer(msg.data, np.uint8)
    state['img'] = cv2.imdecode(arr, cv2.IMREAD_COLOR)

def on_poses(msg):
    state['poses'] = msg.data

from rclpy.qos import qos_profile_sensor_data
n.create_subscription(CompressedImage, TOPIC, on_img, qos_profile_sensor_data)
n.create_subscription(String, '/robocasa/object_poses', on_poses, 10)

t0 = time.time()
while time.time() - t0 < 12.0 and (state['img'] is None or state['poses'] is None):
    rclpy.spin_once(n, timeout_sec=0.2)

if state['img'] is not None:
    cv2.imwrite(OUT, state['img'])
    print(f'saved {OUT} shape={state["img"].shape}')
else:
    print('NO IMAGE received on', TOPIC)

if state['poses'] is not None:
    import json
    d = json.loads(state['poses'])
    screws = sorted(k for k in d if k.startswith('screw'))
    print(f'object_poses: {len(d)} keys, {len(screws)} screws, __pelvis__={"__pelvis__" in d}')
else:
    print('NO object_poses received')
rclpy.shutdown()
