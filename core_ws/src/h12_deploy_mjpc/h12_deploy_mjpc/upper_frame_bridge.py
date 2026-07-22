#!/usr/bin/env python3
"""upper_frame_bridge — serve the FrameTask/NamedConfig actions with the upper-body MPC.

Drop-in replacement for h12_ros2_controller's frame_task_server that routes the
"FrameTask job" through the sampling-MPC upper-body controller
(h12_upper_body_controller) instead of Pinocchio-IK -> lowcmd. GraspSkill is
UNMODIFIED: it still calls the /frame_task and /named_config actions. This node:

  1. Serves the identical FrameTask + NamedConfig ActionServers (names 'frame_task'
     / 'named_config', resolving to /frame_task, /named_config).
  2. On a FrameTask goal, runs MuJoCo-native prioritized/weighted-DLS IK on the
     UPPER model (torso + the addressed arm(s), base/legs frozen) to turn the
     pelvis-frame Cartesian frame pose(s) into the 15 upper joint targets J0..J14.
  3. Pushes them to the upper MPC over gRPC as SetTaskParameters
     "Goal Active"/"Goal Sec"/"Goal J0".."Goal J14" (default :10001), letting the
     MPC do the balance-aware transport + quiescence hold-latch (P6.2 twin gate:
     EE bias 13 mm, held wobble 0.22 mm std).

The MPC writes motor rows 12..26 -> rt/safety/lowcmd_upper_in; the RL lower body
(ALMI) owns legs 0..11 -> lowcmd_lower_in; the split safety layer merges them
(legs-from-lower, torso+arms-from-upper). Legs are NEVER commanded here.

The upper MPC's decision variables are ONLY torso+arms, so unlike the whole-body
grasp task it CANNOT trade balance for reach by diving the legs -- the RL lower
body holds balance and absorbs the (small) CoM shift of the arm reach.

Guarded imports (grpc / mujoco / custom_ros_messages are image/mount-provided) so
the node imports on a bare host (DRY-RUN) and only the live path needs the full
stack -- matching grasp_orchestrator's pattern in this package.
"""
import os
import sys
import math
import time

import numpy as np

# rclpy guarded too: keeps UpperIK + helpers importable on a bare host (bench /
# headless IK validation) where ROS isn't installed. The Node only builds when
# rclpy is present (main() checks _HAVE_RCLPY).
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionServer, CancelResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    _HAVE_RCLPY = True
except Exception as e:  # pragma: no cover
    _HAVE_RCLPY, _RCLPY_ERR = False, str(e)
    Node = object   # so `class UpperFrameBridge(Node)` still DEFINES on a bare host

# --- guarded: gRPC to the running upper MPC ---------------------------------
try:
    import grpc
    try:
        from mujoco_mpc.proto import agent_pb2, agent_pb2_grpc
    except ImportError:
        sys.path.insert(0, os.path.expanduser(   # the fork's built python extension
            "~/Desktop/h12/mujoco_mpc/mujoco_mpc/python/build/"
            "lib.linux-x86_64-cpython-310"))
        from mujoco_mpc.proto import agent_pb2, agent_pb2_grpc
    _HAVE_GRPC = True
except Exception as e:  # pragma: no cover
    _HAVE_GRPC, _GRPC_ERR = False, str(e)

# --- guarded: mujoco for the IK ---------------------------------------------
try:
    import mujoco
    _HAVE_MJ = True
except Exception as e:  # pragma: no cover
    _HAVE_MJ, _MJ_ERR = False, str(e)

# --- guarded: the action types (CONTAINER-only, like grasp_orchestrator) -----
try:
    from custom_ros_messages.action import FrameTask, NamedConfig
    _HAVE_ACT = True
except Exception as e:  # pragma: no cover
    _HAVE_ACT, _ACT_ERR = False, str(e)


def _default_upper_model():
    return os.path.expanduser(
        "~/Desktop/h12/mujoco_mpc/mujoco_mpc/build/mjpc/tasks/"
        "humanoid_bench/upper/Upper_H12_Magpie.xml")


# Upper-model joint index maps (qpos / dof), confirmed from the XML DFS order:
# base 0-6, legs 7-18, torso qadr 19 (dof 18), L arm 20-26 (dof 19-25),
# R arm 27-33 (dof 26-32). Goal J0..J14 order = torso, L arm(7), R arm(7).
TORSO_Q, TORSO_V = 19, 18
LARM_Q, LARM_V = list(range(20, 27)), list(range(19, 26))
RARM_Q, RARM_V = list(range(27, 34)), list(range(26, 33))
GOAL_QADR = [TORSO_Q] + LARM_Q + RARM_Q      # 15 qpos addresses, J0..J14 order

# GraspSkill sends targets for the '*_graspgenx_frame' (base.py GRASP_FRAMES); the
# bridge servos the MuJoCo '{side}_hand' site. Both are rigidly on wrist_yaw_link;
# the hand site sits at [0, +0.001613, +0.052585] m in graspgenx-frame coords (i.e.
# ~52.6 mm along graspgenx +Z toward the fingertips) with a fixed relative rotation
# R_rel = R_graspgenx^T (quat 0.5,-0.5,-0.5,-0.5). Apply this so the IK target lands
# where GraspSkill intends, not ~5.3 cm short. (h1_2_magpie_ros.urdf graspgenx joint
# xyz=0.117415,0,-0.001613 rpy=1.5708,0,1.5708 ; hand site pos=0.17,0,0.)
GRASPGENX_OFFSET = np.array([0.0, 0.001613, 0.052585])   # hand-site pos in graspgenx frame
GRASPGENX_QREL = np.array([0.5, -0.5, -0.5, -0.5])       # hand rot in graspgenx frame (wxyz)


def _quat_mul(a, b):
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array([w0*w1 - x0*x1 - y0*y1 - z0*z1,
                     w0*x1 + x0*w1 + y0*z1 - z0*y1,
                     w0*y1 - x0*z1 + y0*w1 + z0*x1,
                     w0*z1 + x0*y1 - y0*x1 + z0*w1])


def _quat_rot(q, v):
    w, x, y, z = q
    u = np.array([x, y, z], float)
    v = np.asarray(v, float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


class UpperIK:
    """Weighted-DLS IK on the upper model: solve torso + the addressed arm(s) so the
    hand site(s) reach pelvis-frame targets, base/legs frozen at the stand keyframe.
    Position-primary (orientation down-weighted -- the H1-2 wrist can't guarantee an
    arbitrary approach quat). Returns the full 15-vec J0..J14."""

    def __init__(self, logger, model_path):
        self.ok = False
        self.logger = logger
        if not _HAVE_MJ:
            logger.warn(f"mujoco unavailable ({_MJ_ERR}); IK DISABLED (DRY-RUN)")
            return
        try:
            cwd = os.getcwd()
            os.chdir(os.path.dirname(model_path))
            self.m = mujoco.MjModel.from_xml_path(model_path)
        except Exception as e:
            logger.warn(f"upper model load failed ({e}); IK DISABLED")
            return
        finally:
            os.chdir(cwd)
        self.sid = {s: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_hand")
                    for s in ("left", "right")}
        self.kid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "stand")
        d = mujoco.MjData(self.m)
        if self.kid >= 0:
            mujoco.mj_resetDataKeyframe(self.m, d, self.kid)
        mujoco.mj_forward(self.m, d)
        self.base_pos = d.qpos[0:3].copy()
        self.base_quat = d.qpos[3:7].copy()          # wxyz (pelvis-frame -> world)
        self.home15 = d.qpos[GOAL_QADR].copy()        # J0..J14 at stand
        if self.sid["left"] < 0 or self.sid["right"] < 0:
            logger.warn("hand sites missing on upper model; IK DISABLED")
            return
        self.ok = True
        logger.info(f"UpperIK ready: sites L={self.sid['left']} R={self.sid['right']}, "
                    f"base@{self.base_pos.round(3)}")

    def _pelvis_to_world(self, pos_p, quat_p):
        pw = self.base_pos + _quat_rot(self.base_quat, np.asarray(pos_p, float))
        qw = _quat_mul(self.base_quat, np.asarray(quat_p, float))
        return pw, qw

    def solve(self, frames, iters=400, damp=1e-4, step=0.6, w_ori=0.4, w_post=0.4):
        """frames = [(side, pos_pelvis(3), quat_pelvis_wxyz(4))].
        PRIORITIZED IK: hand POSITION is the primary task (a grasp must HIT the
        object); orientation (H1-2 wrist can't guarantee an arbitrary approach quat)
        AND a posture-toward-home bias live in position's NULLSPACE, so an infeasible
        approach angle can't corrupt position and the torso YAW stays near 0 unless
        position genuinely needs it. Returns (J15, max_pos_err_m, max_ori_err, ok)."""
        d = mujoco.MjData(self.m)
        if self.kid >= 0:
            mujoco.mj_resetDataKeyframe(self.m, d, self.kid)
        q_home = d.qpos.copy()
        qadr = [TORSO_Q]
        vadr = [TORSO_V]
        sides = set(s for s, _, _ in frames)
        if "left" in sides:
            qadr += LARM_Q
            vadr += LARM_V
        if "right" in sides:
            qadr += RARM_Q
            vadr += RARM_V
        qadr = np.array(qadr)
        vadr = np.array(vadr)
        nd = len(vadr)
        tgts = [(self.sid[s], *self._pelvis_to_world(p, q)) for s, p, q in frames]
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            mujoco.mj_forward(self.m, d)
            Ep, Eo, Jp_rows, Jr_rows = [], [], [], []
            for sid, pw, qw in tgts:
                cq = np.zeros(4)
                mujoco.mju_mat2Quat(cq, d.site_xmat[sid])
                Ep.append(pw - d.site_xpos[sid])
                eo = np.zeros(3)
                mujoco.mju_subQuat(eo, qw, cq)        # log map cur->tgt
                Eo.append(eo)
                mujoco.mj_jacSite(self.m, d, jacp, jacr, sid)
                Jp_rows.append(jacp[:, vadr])
                Jr_rows.append(jacr[:, vadr])
            ep = np.concatenate(Ep)
            eo = np.concatenate(Eo)
            Jp = np.vstack(Jp_rows)
            Jr = np.vstack(Jr_rows)
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(eo) < 1e-3:
                break
            Jp_pinv = Jp.T @ np.linalg.solve(Jp @ Jp.T + damp*np.eye(Jp.shape[0]),
                                             np.eye(Jp.shape[0]))
            N = np.eye(nd) - Jp_pinv @ Jp               # position nullspace
            Jr_pinv = Jr.T @ np.linalg.solve(Jr @ Jr.T + damp*np.eye(Jr.shape[0]),
                                             np.eye(Jr.shape[0]))
            post = q_home[qadr] - d.qpos[qadr]          # pull toward home (torso ~0)
            dq_null = w_ori * (Jr_pinv @ eo) + w_post * post
            dq = step * (Jp_pinv @ ep + N @ dq_null)
            for k, q in enumerate(qadr):
                d.qpos[q] += dq[k]
                for j in range(self.m.njnt):            # hard joint-limit clamp
                    if self.m.jnt_qposadr[j] == q and self.m.jnt_limited[j]:
                        lo, hi = self.m.jnt_range[j]
                        d.qpos[q] = min(max(d.qpos[q], lo), hi)
        mujoco.mj_forward(self.m, d)
        max_pe = max_oe = 0.0
        for sid, pw, qw in tgts:
            cq = np.zeros(4)
            mujoco.mju_mat2Quat(cq, d.site_xmat[sid])
            max_pe = max(max_pe, float(np.linalg.norm(pw - d.site_xpos[sid])))
            oe = np.zeros(3)
            mujoco.mju_subQuat(oe, qw, cq)
            max_oe = max(max_oe, float(np.linalg.norm(oe)))
        return d.qpos[GOAL_QADR].copy(), max_pe, max_oe, (max_pe < 0.02)


class UpperFrameBridge(Node):
    def __init__(self):
        super().__init__("upper_frame_bridge")
        self.declare_parameter("grpc_addr", "localhost:10001")   # upper MPC monitor
        self.declare_parameter("upper_model_path", _default_upper_model())
        self.declare_parameter("settle_margin_sec", 1.0)
        self.declare_parameter("default_goal_sec", 1.5)
        addr = self.get_parameter("grpc_addr").value
        model_path = self.get_parameter("upper_model_path").value
        self.settle = float(self.get_parameter("settle_margin_sec").value)

        self.ik = UpperIK(self.get_logger(), model_path)
        self.stub = None
        if _HAVE_GRPC:
            self.stub = agent_pb2_grpc.AgentStub(grpc.insecure_channel(addr))
            self.get_logger().info(f"gRPC -> upper MPC @ {addr}")
        else:
            self.get_logger().warn(f"gRPC unavailable ({_GRPC_ERR}); DRY-RUN (no goals sent)")

        # Named joint configs J0..J14, RECONCILED with h12_ros2_controller
        # utility/named_config.py (J0=torso, J1-7=L arm, J8-14=R arm). frame_task's
        # ENABLED_JOINTS are arms-only so torso=0; GraspSkill requests only 't_pose'
        # (grasp.py) but 'home' is kept for parity (= named_config zeros).
        self.named = {
            "home": [0.0] * 15,
            "t_pose": [0.0,
                       0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0,     # L arm: shoulder_roll +1.5
                       0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0],   # R arm: shoulder_roll -1.5
        }

        cbg = ReentrantCallbackGroup()
        if _HAVE_ACT:
            self._fs = ActionServer(self, FrameTask, "frame_task",
                                    execute_callback=self._on_frame_task,
                                    cancel_callback=lambda _g: CancelResponse.ACCEPT,
                                    callback_group=cbg)
            self._ns = ActionServer(self, NamedConfig, "named_config",
                                    execute_callback=self._on_named_config,
                                    cancel_callback=lambda _g: CancelResponse.ACCEPT,
                                    callback_group=cbg)
            self.get_logger().info("serving /frame_task + /named_config (upper-MPC backend)")
        else:
            self.get_logger().warn(f"custom_ros_messages unavailable ({_ACT_ERR}); "
                                   "actions OFF (import-only / bench)")

    # ------------------------------------------------------------------ helpers
    def _side_of(self, frame_name):
        return "left" if "left" in (frame_name or "").lower() else "right"

    @staticmethod
    def _graspgenx_to_hand(pos, quat):
        """A target given for '*_graspgenx_frame' -> the equivalent '{side}_hand'
        site target (position + orientation), applying the fixed rigid offset."""
        q = np.asarray(quat, float)
        hp = np.asarray(pos, float) + _quat_rot(q, GRASPGENX_OFFSET)
        hq = _quat_mul(q, GRASPGENX_QREL)
        return list(hp), list(hq)

    def _set_goal(self, j15, goal_sec):
        if self.stub is None:
            self.get_logger().warn("DRY-RUN: Goal not sent")
            return
        try:
            req = agent_pb2.SetTaskParametersRequest()
            req.parameters["Goal Sec"].numeric = float(goal_sec)
            for i, v in enumerate(j15):
                req.parameters[f"Goal J{i}"].numeric = float(v)
            req.parameters["Goal Active"].numeric = 1.0   # rising edge latches the seg
            self.stub.SetTaskParameters(req, timeout=1.0)
        except Exception as e:  # pragma: no cover
            self.get_logger().warn(f"SetTaskParameters failed: {e}")

    def _settle_loop(self, gh, dur, publish):
        """Block dur+settle while streaming feedback; honor cancel. `publish(frac)`
        emits one feedback msg. Returns True if it ran to completion (not canceled)."""
        t0 = time.time()
        total = dur + self.settle
        while time.time() - t0 < total:
            if gh.is_cancel_requested:
                return False
            publish(min(1.0, (time.time() - t0) / max(dur, 1e-3)))
            time.sleep(0.1)
        return True

    # ------------------------------------------------------------------ actions
    def _on_frame_task(self, gh):
        goal = gh.request
        result = FrameTask.Result()
        frames = []
        for nm, ps in zip(list(goal.frame_names), list(goal.frame_targets)):
            pos = [ps.position.x, ps.position.y, ps.position.z]
            quat = [ps.orientation.w, ps.orientation.x, ps.orientation.y, ps.orientation.z]
            if "graspgenx" in (nm or "").lower():   # GraspSkill's GRASP_FRAMES target
                pos, quat = self._graspgenx_to_hand(pos, quat)
            frames.append((self._side_of(nm), pos, quat))
        dur = goal.duration.sec + goal.duration.nanosec * 1e-9
        if dur <= 0:
            dur = float(self.get_parameter("default_goal_sec").value)

        if not (self.ik.ok and frames):
            self.get_logger().warn("frame_task: IK unavailable or empty frames -> abort")
            gh.abort()
            result.success = False
            return result

        j15, pe, oe, ok = self.ik.solve(frames)
        self.get_logger().info(
            f"frame_task {list(goal.frame_names)}: IK pos_err={pe*1000:.1f}mm "
            f"ori_err={math.degrees(oe):.1f}deg reach<2cm={ok} -> Goal(sec={dur:.2f})")
        self._set_goal(j15, dur)

        fb = FrameTask.Feedback()
        n = max(1, len(frames))

        def publish(frac):
            # taper reported error from the IK residual toward ~0 as the MPC transports
            fb.errors_linear = [float(pe * (1.0 - 0.9 * frac))] * n
            fb.errors_angular = [float(oe * (1.0 - 0.9 * frac))] * n
            gh.publish_feedback(fb)

        if not self._settle_loop(gh, dur, publish):
            gh.canceled()
            result.success = False
            return result
        gh.succeed()
        result.success = ok
        return result

    def _on_named_config(self, gh):
        goal = gh.request
        result = NamedConfig.Result()
        name = goal.config_name
        dur = goal.duration.sec + goal.duration.nanosec * 1e-9
        if dur <= 0:
            dur = float(self.get_parameter("default_goal_sec").value)
        if name not in self.named:
            self.get_logger().warn(f"named_config '{name}' unknown "
                                   f"(have {list(self.named)}) -> abort")
            gh.abort()
            result.success = False
            return result
        self.get_logger().info(f"named_config '{name}' -> Goal(sec={dur:.2f})")
        self._set_goal(self.named[name], dur)

        fb = NamedConfig.Feedback()

        def publish(frac):
            fb.joint_error = float(0.5 * (1.0 - frac))   # coarse taper toward 0
            gh.publish_feedback(fb)

        if not self._settle_loop(gh, dur, publish):
            gh.canceled()
            result.success = False
            return result
        gh.succeed()
        result.success = True
        return result


def main(args=None):
    if not _HAVE_RCLPY:
        raise SystemExit(f"rclpy unavailable ({_RCLPY_ERR}); this node needs ROS 2")
    rclpy.init(args=args)
    node = UpperFrameBridge()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
