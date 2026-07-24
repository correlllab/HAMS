#!/usr/bin/env python3
"""Find the gripper CLOSE-END (fingertips) precisely: the tip sites live at finger-z
0.0763 in the finger frames. Measure the finger frames' full transforms in the wrist
frame, apply the tip offset, and report the closed-tip midpoint (for the red marker).
Also drive a top-down pose over screw_27 and report the fingertip DEPTH in pelvis
(how low the tips actually reach) vs the table, to size the nail."""
import time, threading, numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from rclpy.duration import Duration
from h12_skills.grasp_benchmark import GRASP_FRAMES
import battery_bench as bb

rclpy.init()
from h12_skills.grasp_benchmark import GraspBenchmark
b = GraspBenchmark('right', gt_name='', box_source='gt')
ex = MultiThreadedExecutor(num_threads=6); ex.add_node(b)
threading.Thread(target=ex.spin, daemon=True).start()
for _ in range(100):
    if b._obj_poses.get('__pelvis__') is not None: break
    time.sleep(0.2)
frame = GRASP_FRAMES['right']

def T(parent, child):
    try:
        t = b.tf_buffer.lookup_transform(parent, child, Time(), timeout=Duration(seconds=2.0))
        tr = t.transform.translation; q = t.transform.rotation
        from h12_skills.perception_utils import pose_to_matrix
        from geometry_msgs.msg import Pose
        p = Pose(); p.position.x, p.position.y, p.position.z = tr.x, tr.y, tr.z; p.orientation = q
        return pose_to_matrix(p)
    except Exception as e:
        print('  no tf', parent, child, e); return None

TIP_L = np.array([0.003, -0.02027, 0.0763, 1.0])   # tip_left  site in left finger frame
TIP_R = np.array([0.003,  0.02027, 0.0763, 1.0])   # tip_right site in right finger frame

# home + close so fingers are at the grip config
b.go_home(duration_sec=4.0); time.sleep(2.0)
b.set_gripper('right', 20.0); time.sleep(2.0)      # near-closed to see the pinch point
Ml = T('right_wrist_yaw_link', 'rg_left_finger'); Mr = T('right_wrist_yaw_link', 'rg_right_finger')
if Ml is not None and Mr is not None:
    tl = (Ml @ TIP_L)[:3]; tr = (Mr @ TIP_R)[:3]
    print('rg_left_finger origin  (wrist):', np.round(Ml[:3,3],4).tolist())
    print('rg_right_finger origin (wrist):', np.round(Mr[:3,3],4).tolist())
    print('LEFT  TIP (wrist):', np.round(tl,4).tolist())
    print('RIGHT TIP (wrist):', np.round(tr,4).tolist())
    print('==> CLOSE-END TIP MIDPOINT (wrist frame, for red marker):', np.round(0.5*(tl+tr),4).tolist())
    print('    (current red marker is at x=0.169; camera hand_cam at mount ~x=0.124)')

# now a top-down grasp pose over screw_27: how deep do the TIPS reach in pelvis?
nail = b.gt_pos_pelvis('screw_27')
tabletop_world = 0.910  # screw base / table top
print('\nscrew_27 (pelvis):', np.round(nail,4).tolist())
_, tip = bb._screw_grasp_pose(b, 'screw_27')
over = np.asarray(tip) + np.array([0,0,0.10])
b.move_frame_to(frame, bb._top_down_pose(over), duration_sec=6.0, do_plan=bb.DO_PLAN)
cmd = np.array([nail[0], nail[1], -0.052])
b.move_frame_to(frame, bb._top_down_pose(cmd), duration_sec=6.0, do_plan=False)
# fingertip in pelvis via the finger frames
Pl = b.frame_pose_pelvis('rg_left_finger'); Pr = b.frame_pose_pelvis('rg_right_finger')
Mlp = T('pelvis', 'rg_left_finger'); Mrp = T('pelvis', 'rg_right_finger')
if Mlp is not None and Mrp is not None:
    tlp = (Mlp @ TIP_L)[:3]; trp = (Mrp @ TIP_R)[:3]
    tipmid = 0.5*(tlp+trp)
    print('FINGERTIP MIDPOINT reachable depth (pelvis):', np.round(tipmid,4).tolist())
    print('  finger-frame midpoint (pelvis):', np.round(0.5*(np.array([Pl.position.x,Pl.position.y,Pl.position.z])+np.array([Pr.position.x,Pr.position.y,Pr.position.z])),4).tolist())
    # convert tip pelvis-z to world-ish: table top world 0.910 = pelvis z of nail base
    print(f'  nail base pelvis-z = {nail[2]:.4f} (table). Tips reach pelvis-z {tipmid[2]:.4f}')
    print(f'  => tips are {1000*(tipmid[2]-nail[2]):.0f} mm ABOVE the table; a nail must stand at least this tall to be tip-gripped')
rclpy.shutdown()
