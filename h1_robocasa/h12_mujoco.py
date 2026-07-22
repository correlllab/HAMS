import argparse
import math
import os
import random
import threading
import time

import mujoco
import numpy as np
import rclpy.executors

from mujoco_ros_bridge import init_ros, shutdown_ros
import mujoco.viewer

import h1_2_robosuite  # registers H1_2 + Magpie grippers, patches RoboCasa
from magpie_hand_bridge import MagpieHandBridge
from measurement_bridge import MeasurementBridge
from mujoco_env import ElasticBand
from mujoco_ros_bridge import RosSensorBridge
from sim_names import NameResolver
from unitree_interface import SimInterface

# Viewer free-camera spawn pose, anchored to the robot so the passive viewer
# opens framed on the robot wherever it spawns (RoboCasa places the base by
# layout/seed and place_robot_collision_free may back it off), instead of
# MuJoCo's default whole-scene framing. Top-down (straight down) view: lookat
# follows the robot base body, elevation looks straight down, and azimuth (the
# in-plane rotation of the top-down image) is offset from the base yaw so the
# robot's facing points a consistent way in frame at any spawn orientation.
# Spawn-time only — the user can orbit/zoom freely after.
VIEW_CAM_DISTANCE  = 3.0     # m, camera height above lookat (straight down)
VIEW_CAM_AZIMUTH   = 90.0    # deg, ADDED to robot yaw — rotates the top-down image
VIEW_CAM_ELEVATION = -90.0   # deg, -90 = look straight down
VIEW_CAM_LOOKAT_DZ = 0.0     # m, focal point offset along z (irrelevant for top-down)

NUM_LEG_JOINTS = 12
NUM_MOTORS = 27
# Stance-hold gains for the legs when no lower-body controller is running
# (see SimInterface.set_hold). Stiff enough to hold the spawn stance against
# gravity; MuJoCo clamps the result to each actuator's ctrlrange anyway.
LEG_HOLD_KP = 300.0
LEG_HOLD_KD = 10.0
# Stance: the shipped near-straight STANDING stance (user requirement: the robot
# stands upright — no lean, no squat). The handle sits ~0.23 m below the shoulders;
# grasps must therefore approach HORIZONTALLY (a vertical handle allows it, and a
# flat approach needs no downward wrist pitch), with the spawn laterally aligned so
# the reach isn't cross-body.
INIT_LEG_POS = np.array([
    0.0, -0.16, 0.0, 0.36, -0.2, 0.0,
    0.0, -0.16, 0.0, 0.36, -0.2, 0.0,
], dtype=float)
# HAMS_STANCE=almi: spawn with the legs ALREADY at the ALMI policy's trained
# nominal crouch (hip -0.4 / knee 0.8 / ankle -0.4, matching policies/almi/
# almi.yaml default_angles). Engaging the policy from its own nominal removes the
# pre-pose transition that otherwise makes it take a backward settle-walk right
# after the elastic band releases — the robot starts AT the grasp station and
# stays there. Default unset = the standing stance above.
if os.environ.get('HAMS_STANCE', '').strip().lower() == 'almi':
    INIT_LEG_POS = np.array([
        0.0, -0.4, 0.0, 0.8, -0.4, 0.0,
        0.0, -0.4, 0.0, 0.8, -0.4, 0.0,
    ], dtype=float)
INIT_ARM_POS = np.array([
    0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
], dtype=float)


def _initial_motor_qpos():
    """Build the 27-motor spawn pose without depending on controller configs."""
    qpos = np.concatenate([INIT_LEG_POS, INIT_ARM_POS])
    if qpos.shape != (NUM_MOTORS,):
        raise ValueError(f"initial motor pose must contain {NUM_MOTORS} entries")
    return qpos


def _dump_reach_geometry(env, model, data, tag=""):
    """One-shot ground-truth reach geometry for the grasp-comparison setup: print
    the robot's arm/base anchors and every candidate fixture handle/door geom in
    world coords, so the static-arm spawn standoff is set from real numbers rather
    than guesses. Diagnostic only — no side effects. Gated by HAMS_DUMP_REACH."""
    if os.environ.get('HAMS_DUMP_REACH', '0').strip().lower() in ('0', 'off', 'false', 'no', ''):
        return
    try:
        pfx = env.robots[0].robot_model.naming_prefix
        def _bpos(name):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            return None if bid < 0 else data.xpos[bid].copy()
        base = _bpos(f"{pfx}pelvis")
        fwd = np.zeros(3)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{pfx}pelvis")
        if bid >= 0:
            mujoco.mju_rotVecQuat(fwd, np.array([1.0, 0.0, 0.0]), data.xquat[bid])
        print(f"[reach-geom]{tag} facing +x(world)={fwd.round(3)} === robot anchors (world xyz) ===")
        for nm in ("pelvis", "torso_link",
                   "right_shoulder_pitch_link", "left_shoulder_pitch_link",
                   "right_elbow_link", "left_elbow_link",
                   "right_wrist_yaw_link", "left_wrist_yaw_link"):
            p = _bpos(f"{pfx}{nm}")
            if p is not None:
                fp = float((p - base).dot(fwd)) if base is not None else 0.0
                print(f"[reach-geom]  {nm:28s} {p.round(3)}  fproj={fp:+.3f}")
        # graspgenx TCP frames (sites), if present
        for snm in (f"{pfx}right_graspgenx_frame", f"{pfx}left_graspgenx_frame",
                    "right_graspgenx_frame", "left_graspgenx_frame"):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, snm)
            if sid >= 0:
                sp = data.site_xpos[sid].copy()
                fp = float((sp - base).dot(fwd)) if base is not None else 0.0
                print(f"[reach-geom]  site {snm:28s} {sp.round(3)}  fproj={fp:+.3f}")
        kws = ("handle", "door", "fridge", "refriger", "drawer")
        cands = []
        for gid in range(model.ngeom):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            low = nm.lower()
            if not any(k in low for k in kws) or nm.startswith(pfx):
                continue
            gp = data.geom_xpos[gid].copy()
            fproj = float((gp - base).dot(fwd)) if base is not None else 0.0
            cands.append((fproj, nm, gp))
        cands.sort(key=lambda t: (-t[0], t[1]))
        print(f"[reach-geom]{tag} === {len(cands)} fixture handle/door/fridge geoms (nearest-front first) ===")
        for fproj, nm, gp in cands:
            z = float(gp[2])
            flag = "  <== IN FRONT @ grasp height" if 0.1 < fproj and 0.6 < z < 1.7 else ""
            print(f"[reach-geom]  fproj={fproj:+.3f} z={z:.3f}  {nm:40s} {gp.round(3)}{flag}")
    except Exception as e:
        print(f"[reach-geom] dump skipped: {e}")


def _frame_viewer_on_robot(handle, data, body_id):
    """Set the passive viewer free camera's spawn pose as a function of the robot
    base body's world pose. lookat tracks the body position; the orbit azimuth is
    offset from the body's yaw, so framing is invariant to spawn orientation."""
    pos = data.xpos[body_id]
    w, x, y, z = (float(v) for v in data.xquat[body_id])
    yaw_deg = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    cam = handle.cam
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (float(pos[0]), float(pos[1]), float(pos[2]) + VIEW_CAM_LOOKAT_DZ)
    cam.distance = VIEW_CAM_DISTANCE
    cam.azimuth = yaw_deg + VIEW_CAM_AZIMUTH
    cam.elevation = VIEW_CAM_ELEVATION


def _draw_band(handle, anchor, body_pos):
    """Draw the elastic band (anchor sphere + capsule to the torso) in the passive
    viewer's user scene. Mirrors the legacy MujocoEnv.draw_elastic_band."""
    scn = handle.user_scn
    scn.ngeom = 0
    color = np.array([0.5, 0.6, 1.0, 0.8])
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.02, 0.0, 0.0]), anchor, np.eye(3).flatten(), color)
        scn.ngeom += 1
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE,
                            np.zeros(3), np.zeros(3), np.eye(3).flatten(), color)
        mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE,
                             0.01, anchor, body_pos)
        scn.ngeom += 1


class VideoRecorder:
    """Offscreen third-person video of the sim, streamed straight to an mp4 via
    imageio-ffmpeg (frames are NOT accumulated in RAM). Intended for headless
    benchmark runs (MUJOCO_GL=egl) where no passive viewer is up but we still
    want a recording of each trial. The free camera re-frames on the robot base
    every captured frame so the robot stays in shot as it sways on the tether."""

    def __init__(self, model, data, body_id, path, fps=30, width=640, height=480):
        import os as _os
        import imageio
        self._body_id = body_id
        self._path = path
        self._frames = 0
        _os.makedirs(_os.path.dirname(_os.path.abspath(path)), exist_ok=True)
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._opt = mujoco.MjvOption()
        self._opt.geomgroup[0] = 0            # hide collision geoms, like the viewer
        # yuv420p + even dims => the mp4 plays in every browser / QuickTime.
        # Fragmented mp4 (frag_keyframe+empty_moov) writes the index incrementally
        # instead of only in a trailing moov atom, so the benchmark's `docker rm
        # -f` on the sim between episodes (a hard kill, no clean shutdown) still
        # leaves a PLAYABLE file with every frame flushed so far — a plain mp4
        # killed mid-run is an unplayable ~48-byte stub. imageio already passes a
        # -pix_fmt, so set the pixel format via ffmpeg_params to avoid a dup.
        self._writer = imageio.get_writer(
            path, fps=fps, macro_block_size=None, codec='libx264',
            pixelformat='yuv420p',
            output_params=['-movflags', '+frag_keyframe+empty_moov+default_base_moof'])
        self._frame_on_robot(data)

    def _frame_on_robot(self, data):
        if self._body_id < 0:
            return
        pos = data.xpos[self._body_id]
        w, x, y, z = (float(v) for v in data.xquat[self._body_id])
        yaw = math.degrees(math.atan2(2.0 * (w * z + x * y),
                                      1.0 - 2.0 * (y * y + z * z)))
        self._cam.lookat[:] = (float(pos[0]), float(pos[1]), float(pos[2]))
        self._cam.distance = 2.2
        self._cam.azimuth = yaw + 150.0       # oblique 3/4 view onto the robot's workspace
        self._cam.elevation = -20.0

    def capture(self, data):
        self._frame_on_robot(data)
        self._renderer.update_scene(data, camera=self._cam, scene_option=self._opt)
        self._writer.append_data(self._renderer.render())
        self._frames += 1

    def close(self):
        for closer in (getattr(self._writer, 'close', None),
                       getattr(self._renderer, 'close', None)):
            try:
                if closer is not None:
                    closer()
            except Exception:
                pass
        print(f"[h12_mujoco] video: wrote {self._frames} frames to {self._path}",
              flush=True)


def sim_loop(task, viewer=True, layout=None, style=None, seed=None, record_video=None,
             cam_width=256, cam_height=256, hold_legs=True):
    """Launch a RoboCasa task env around the H1-2 and run the shared-MjData loop.

    Builds the env with robots='H1_2' and steps the env's *own* MjData with our
    loop (never env.step()), driving the full sim<->ROS bridge layer:
      - SimInterface     : DDS rt/lowstate + rt/lowcmd (27 body motors, name-resolved)
      - RosSensorBridge  : /clock, head RGBD, livox lidar + IMU
      - MagpieHandBridge : /{left,right}/gripper/* (x2)
      - MeasurementBridge: /robocasa/{task_goal,success,reward} + /elastic_band/toggle
    An elastic-band tether holds the robot upright until the ROS walking policy
    drives it; RoboCasa's _check_success/reward/lang are read off the shared env.
    """

    # BatteryWorkcell: not a RoboCasa kitchen env — a self-contained MuJoCo
    # scene (raw h1_2_magpie robot + ARPA battery cell) loaded directly and
    # duck-typed to the same env surface (see battery_env.py). Raw model =
    # NO robosuite name prefixes; `pfx` below comes out '' and every
    # f"{pfx}..." lookup resolves the raw names. layout/style/seed are
    # kitchen-only and ignored here (the scene is fully authored).
    _battery = (task == 'BatteryWorkcell')
    if _battery:
        import battery_env
        env = battery_env.BatteryWorkcellEnv(
            os.environ.get('HAMS_BATTERY_XML', '').strip() or None)
    else:
        create_kwargs = {}
        if layout is not None:
            create_kwargs["layout_ids"] = layout
        if style is not None:
            create_kwargs["style_ids"] = style
        if seed is not None:
            create_kwargs["seed"] = seed

        env = h1_2_robosuite.make_kitchen_env(task, **create_kwargs)

    # robosuite.make() constructs but does NOT reset; reset() builds env.sim, runs
    # _load_model + _reset_internal (placement patch fires here, sets ep_meta /
    # init_robot_base_pos), and places the task objects.
    env.reset()

    model = env.sim.model._model
    data = env.sim.data._data
    # ROS<->sim name map for the DDS / sensor bridges. The battery scene's raw
    # model carries no robosuite prefixes, which NameResolver supports by design.
    resolver = (NameResolver(model, robot_prefix="") if _battery
                else NameResolver(model))

    # Clean, contact-free initial state. RoboCasa's zero-action settling loop
    # perturbs the pose during reset, so restore a baked-in bent-knee stance,
    # zero all velocities, and re-place the pelvis. Upper-body motors remain at
    # zero. place_robot_collision_free auto-fits floor clearance AND backs the
    # base away (the robot's -x) until no robot geom penetrates a fixture, keeping
    # the least-penetrating spot if it can't fully clear.
    try:
        init_qpos = _initial_motor_qpos()
        data.qpos[resolver.motor_qpos] = init_qpos
        data.qvel[:] = 0.0
        # HAMS_SPAWN_BACKOFF backs the robot further off the counter into open
        # floor (default 0 = at the counter for manipulation). Set ~1.0 for nav2,
        # which needs the robot's start cell to be free to plan.
        _spawn_backoff = float(os.environ.get("HAMS_SPAWN_BACKOFF", "0") or 0)
        # HAMS_SPAWN_ARMS_CLEAR: ignore the arm/gripper geoms when backing the
        # robot off the fixture at spawn, so it parks as close as its TORSO/legs
        # allow (the zero-pose hands jut ~0.31 m and otherwise shove it out of
        # arm reach). The arms are homed by bringup right after, so their spawn
        # overlap is transient. Default off preserves the shipped spawn distance.
        _arms_clear = os.environ.get('HAMS_SPAWN_ARMS_CLEAR', '0').strip().lower() \
            not in ('0', 'off', 'false', 'no', '')
        # HAMS_SPAWN_FORWARD: start the spawn this many metres closer to the fixture
        # (along the robot's facing) so a STATIC arm can reach it. This is
        # COLLISION-AWARE — place_robot_collision_free then backs the torso off
        # again if it would actually overlap the fixture/door, so it can't clip
        # through anything; it just removes the mobile-robot approach gap.
        _base_pos = np.array(env.init_robot_base_pos, dtype=float)
        _base_quat = h1_2_robosuite._euler_to_wxyz(getattr(env, "init_robot_base_ori", None))
        _spawn_fwd = float(os.environ.get('HAMS_SPAWN_FORWARD', '0') or 0)
        if _spawn_fwd:
            _fwd = -h1_2_robosuite._backward_dir(_base_quat)   # world forward, toward the fixture
            _base_pos[:2] += _spawn_fwd * _fwd[:2]
            print(f"[h12_mujoco] HAMS_SPAWN_FORWARD={_spawn_fwd}m: starting spawn "
                  f"{_spawn_fwd}m closer to the fixture (torso back-off still applies)")
        # HAMS_SPAWN_LATERAL: slide the spawn this many metres along the robot's
        # LEFT (+) / right (-) so the grasp arm's shoulder lines up with the target,
        # removing the cross-body reach that the wrist can't fold into. Collision-
        # aware back-off still applies afterwards.
        _spawn_lat = float(os.environ.get('HAMS_SPAWN_LATERAL', '0') or 0)
        if _spawn_lat:
            _fwd = -h1_2_robosuite._backward_dir(_base_quat)
            _left = np.cross(np.array([0.0, 0.0, 1.0]), _fwd)  # world robot-left
            _n = np.linalg.norm(_left[:2])
            if _n > 1e-9:
                _base_pos[:2] += _spawn_lat * (_left[:2] / _n)
                print(f"[h12_mujoco] HAMS_SPAWN_LATERAL={_spawn_lat}m: sliding spawn "
                      f"{_spawn_lat}m to the robot's {'left' if _spawn_lat > 0 else 'right'}")
        h1_2_robosuite.place_robot_collision_free(
            env, _base_pos, _base_quat,
            extra_backoff=_spawn_backoff,
            ignore_arm_geoms=_arms_clear,
        )
        if _arms_clear:
            print("[h12_mujoco] HAMS_SPAWN_ARMS_CLEAR=1: spawned close to the fixture "
                  "(arm/gripper spawn overlap ignored; torso/legs kept clear)")
        print("[h12_mujoco] initial stance from baked-in sim defaults")
    except Exception as e:
        print(f"[h12_mujoco] clean reset skipped: {e}")

    # One-shot ground-truth reach dump (HAMS_DUMP_REACH=1) for setting the static
    # arm's spawn standoff from real handle/shoulder positions.
    _dump_reach_geometry(env, model, data, tag=" spawn")

    # Elastic-band balance tether on the torso: holds the free-floating biped
    # upright until the ROS walking policy takes over. Anchor a fixed point above
    # the torso's spawn; apply the spring force each step (SPACE / the
    # /elastic_band/toggle service disables it).
    band = None
    band_body_id = -1
    base_dof = 0
    try:
        band_body_id = resolver.body_id("torso_link")            # robot0_torso_link
        fj = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT,
            f"{env.robots[0].robot_model.naming_prefix}{h1_2_robosuite.FREE_JOINT_NAME}")
        base_dof = int(model.jnt_dofadr[fj])
        anchor = data.xpos[band_body_id].copy()
        anchor[2] += 0.01
        # Stiff enough to hold the LIMP robot near-upright through the lowerbody
        # controller's band-held idle phase (~20s of frame_task arm-home startup
        # before FAME engages); at stiffness=1e3 it pitched forward to ~60deg and
        # FAME inherited a toppling robot. Static tilt ~ gravity_torque/stiffness.
        band = ElasticBand(anchor, length=0, stiffness=1e4, damping=1e3)
        print(f"[h12_mujoco] elastic band on torso anchored at {anchor.round(3)}")
    except Exception as e:
        print(f"[h12_mujoco] elastic band setup skipped: {e}")

    print(f"[h12_mujoco] RoboCasa task {task!r} built around H1_2: "
          f"nq={model.nq} nu={model.nu}; resolver mapped {len(resolver.motor_ctrl)} motors")

    # Shared lock: the DDS + ROS bridge threads touch MjData (read sensors, write
    # ctrl, render) while the main loop runs mj_step. All MjData access is
    # serialized through sim_lock.
    sim_lock = threading.Lock()
    pfx = env.robots[0].robot_model.naming_prefix   # "robot0_"

    # Body carrying the pelvis free joint — its world pose is ground-truth base
    # odometry, which the bridge publishes as odom -> pelvis TF + /odom (feeds
    # SLAM without needing FAST-LIO). OFF by default so the shared sim bridge does
    # not inject a second odom->pelvis anchor that competes with the real stack's
    # FAST-LIO (camera_init) TF; the Mac nav/SLAM demo opts in with HAMS_SIM_ODOM=1.
    odom_base_body_id = -1
    _sim_odom_on = os.environ.get('HAMS_SIM_ODOM', '0').strip().lower() not in ('0', 'off', 'false', 'no', '')
    if _sim_odom_on:
        try:
            _fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    f"{pfx}{h1_2_robosuite.FREE_JOINT_NAME}")
            odom_base_body_id = int(model.jnt_bodyid[_fj])
        except Exception as e:
            print(f"[h12_mujoco] base odom body resolve skipped: {e}")
    print(f"[h12_mujoco] ground-truth sim odom {'ON (HAMS_SIM_ODOM=1)' if _sim_odom_on else 'OFF'}")

    # DDS control bridge: publishes rt/lowstate, subscribes rt/lowcmd, drives the
    # 27 body motors by name via the resolver (grippers handled by the hand
    # bridges below; gripper ctrl indices are disjoint from the body motors).
    sim_interface = SimInterface(model, data, lock=sim_lock, resolver=resolver)  # noqa: F841

    # Keep the legs at their spawn stance whenever no lower-body controller is
    # driving them (i.e. MJPC/walking off — as the grasp benchmark runs it). Without
    # this the leg motors get zero torque, the robot flops on the elastic band, and
    # the pelvis lands somewhere different every episode (~15 cm of spread), which
    # randomises what the arm can reach. Inert as soon as a real controller takes
    # the joints over, so it costs nothing in a normal walking bringup.
    if hold_legs:
        sim_interface.set_hold(np.arange(NUM_LEG_JOINTS), INIT_LEG_POS,
                               kp=LEG_HOLD_KP, kd=LEG_HOLD_KD)
        print(f"[h12_mujoco] holding {NUM_LEG_JOINTS} leg joints at the spawn "
              f"stance (kp={LEG_HOLD_KP}, kd={LEG_HOLD_KD}) while nothing commands them")

    # Optional RIGID body freeze for arm-only experiments (the grasp comparison).
    # The elastic band is a soft spring, so the whole robot still sways. When
    # HAMS_FREEZE_BODY=1, kinematically pin the pelvis free joint + the 12 leg
    # joints + the torso joint to their spawn pose every sim step (velocities
    # zeroed), so ONLY the 14 arm joints move. Default OFF preserves the sim's
    # shipped elastic-band behavior; this is opt-in and gated like HAMS_SIM_ODOM.
    # Spawn-at-ready: HAMS_STATE_PATH points at a saved qpos/qvel snapshot of a
    # CONVERGED standing state (written via /hams/save_state, paired with the
    # controller's LSTM snapshot). Restoring it before the freeze/scene
    # snapshots makes "ready" the spawn state itself: the freeze pin, the
    # /hams/reset_scene restore point, and the policy warmup all inherit it.
    _state_path = os.environ.get('HAMS_STATE_PATH', '').strip()
    _state_save = {'req': False}
    if _state_path and os.path.isfile(_state_path):
        try:
            _snap = np.load(_state_path)
            data.qpos[:] = _snap['qpos']
            data.qvel[:] = _snap['qvel']
            mujoco.mj_forward(model, data)
            print(f"[h12_mujoco] STATE RESTORED from {_state_path} (spawn-at-ready)", flush=True)
        except Exception as e:
            print(f"[h12_mujoco] state restore failed ({e}); normal spawn", flush=True)

    _freeze_body = os.environ.get('HAMS_FREEZE_BODY', '0').strip().lower() \
        not in ('0', 'off', 'false', 'no', '')
    _frz_base_qadr = _frz_base_dof = -1
    _frz_base_qpos = _frz_lower_qidx = _frz_lower_vidx = _frz_lower_qpos = None
    try:
        # Resolve the pin indices UNCONDITIONALLY so the freeze can also be
        # toggled at runtime (/hams/freeze_body) even when it starts off.
        _frzfj = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT,
            f"{pfx}{h1_2_robosuite.FREE_JOINT_NAME}")
        _frz_base_qadr = int(model.jnt_qposadr[_frzfj])
        _frz_base_dof = int(model.jnt_dofadr[_frzfj])
        _frz_base_qpos = data.qpos[_frz_base_qadr:_frz_base_qadr + 7].copy()
        # legs (0..11) + torso (12) — everything except the 14 arm motors (13..26)
        _frz_lower_qidx = np.asarray(resolver.motor_qpos[:NUM_LEG_JOINTS + 1])
        _frz_lower_vidx = np.asarray(resolver.motor_qvel[:NUM_LEG_JOINTS + 1])
        _frz_lower_qpos = data.qpos[_frz_lower_qidx].copy()
        if _freeze_body:
            if band is not None:
                band.enabled = False   # the rigid pin replaces the soft tether
            print("[h12_mujoco] HAMS_FREEZE_BODY=1: pelvis + legs + torso pinned "
                  "rigidly every step; elastic band disabled — ONLY the arms move")
    except Exception as e:
        print(f"[h12_mujoco] freeze setup failed, staying dynamic: {e}")
        _freeze_body = False

    # Full-scene snapshot for the in-place trial reset (/hams/reset_scene):
    # restoring qpos/qvel to the spawn state puts the door, objects, AND robot
    # back exactly where a fresh env would — without a container rebuild or a
    # policy cold-start. Applied under the freeze pin by the main loop.
    _scene_qpos0 = data.qpos.copy()
    _scene_reset = {'req': False}

    # Reliable between-trial ARM RESET for the grasp sweep. The upper-body
    # differential-IK controller can leave the arm in a raised config it cannot
    # recover from (named_config 'home' then fails), so consecutive trials start
    # from a bad pose and stop reaching. On a /hams/reset_arm message we pin the 14
    # arm joints (+ their PD targets) to the spawn/home pose for HAMS_ARM_RESET_HOLD
    # seconds, which deterministically returns the arm home regardless of a stuck
    # controller setpoint. Opt-in; harmless when unused.
    _arm_qidx = np.asarray(resolver.motor_qpos[NUM_LEG_JOINTS + 1:])   # 14 arm motors
    _arm_vidx = np.asarray(resolver.motor_qvel[NUM_LEG_JOINTS + 1:])
    _arm_cidx = np.asarray(resolver.motor_ctrl[NUM_LEG_JOINTS + 1:])
    _arm_home_q = np.asarray(INIT_ARM_POS[1:], dtype=float)           # skip torso[0]
    _arm_reset_hold = float(os.environ.get('HAMS_ARM_RESET_HOLD', '2.5') or 2.5)
    # Ramp duration (sim-s) for the reset motion; 0 = legacy one-step snap.
    _arm_reset_ramp = float(os.environ.get('HAMS_ARM_RESET_RAMP', '0') or 0.0)
    _arm_reset = {'until': -1.0, 'from_q': None}
    # Runtime freeze toggle (/hams/freeze_body, std_msgs/Bool). Enables the
    # freeze-hold handover protocol for standing-policy experiments: hold the
    # robot rigidly pinned through setup (cannot fall or sway, unlike the elastic
    # band), engage the policy while pinned, then release the pin and the policy
    # is already in control. Re-freezing captures the CURRENT pose as the new pin.
    # 'req' carries the requested state; the main loop applies it (capturing the
    # pin pose under the sim step, race-free). Initialized from HAMS_FREEZE_BODY.
    _frz_toggle = {'req': None, 'on': _freeze_body}

    init_ros()

    # Sensor bridge: /clock, RGBD cameras (head + both hands), livox lidar + IMU.
    # Pass robosuite-prefixed MuJoCo lookup names; ROS-facing frame_ids stay
    # unprefixed (ctor defaults). cameras: (mujoco name, /realsense/<ns>, frame_id).
    # Head rides the robot prefix; the eye-in-hand gripper cameras ride the gripper
    # prefixes (same as the hand bridges below).
    # RGBD camera rendering (3x 256x256 offscreen renders per frame) is the
    # heaviest per-step cost on CPU. HAMS_CAMERAS=0 drops them to speed the sim up
    # for locomotion/SLAM (lidar + odom are unaffected).
    # HAMS_CAMERAS: '1' (default) = all three RGBD cams, '0' = none,
    # 'head' = head only — the offscreen renders are the heaviest per-step
    # cost, and GT-perception grasp sweeps only consume the head camera, so
    # dropping the two hand cams roughly halves the per-step render bill.
    _cam_mode = os.environ.get('HAMS_CAMERAS', '1').strip().lower()
    _cameras_on = _cam_mode not in ('0', 'off', 'false', 'no')
    # Hand-camera MuJoCo names: robosuite prefixes the (identical) gripper MJCFs
    # as gripper0_<side>_hand_cam; the raw battery-scene model keeps the gripper
    # files' own names — right hand 'hand_cam', left hand 'leftg_hand_cam'.
    _lh_cam, _rh_cam = (("leftg_hand_cam", "hand_cam") if _battery else
                        ("gripper0_left_hand_cam", "gripper0_right_hand_cam"))
    _cameras = ([
        (f"{pfx}head_cam",          "head",       "camera_color_optical_frame"),
    ] if _cam_mode == 'head' else [
        (f"{pfx}head_cam",          "head",       "camera_color_optical_frame"),
        (_lh_cam,  "left_hand",  "left_hand_camera_color_optical_frame"),
        (_rh_cam, "right_hand", "right_hand_camera_color_optical_frame"),
    ]) if _cameras_on else []
    print(f"[h12_mujoco] RGBD cameras {'ON' if _cameras_on else 'OFF (HAMS_CAMERAS=0)'}")
    ros_bridge = RosSensorBridge(
        model, data,
        cameras=_cameras,
        # Camera render size for all 3 RGBD cameras (--cam-width/--cam-height).
        # 256x256 is the RoboCasa default and is enough for the gemini detector to
        # box the target (the earlier "cheese not found" was a PROMPT problem —
        # "wedge of cheese" returned [] where "cheese" boxes it — not resolution).
        # Raising this costs host RAM on an already-tight box, so keep the default
        # unless a detection genuinely needs the pixels.
        cam_width=cam_width, cam_height=cam_height,
        # MID-360 fidelity: 360x56 @ 10Hz ~= 201k pts/s (real ~200k), 0.1m near /
        # 40m far range, per-point offset_time for FAST-LIO deskew. el_rays/rate
        # are the knobs to dial back if the ray cast (main thread) hurts RTF.
        # HAMS_LIDAR=0: grasp-sweep mode — no SLAM/nav consumer exists, and the
        # 360x56 @ 10 Hz raycast (~201k rays/s) runs on the MAIN sim thread. A
        # token 4x2 @ 0.2 Hz keeps the topics/publishers alive (surface intact)
        # at effectively zero step cost. Default 1 = full MID-360 fidelity.
        lidar_az_rays=(360 if os.environ.get('HAMS_LIDAR', '1') != '0' else 4),
        lidar_el_rays=(56 if os.environ.get('HAMS_LIDAR', '1') != '0' else 2),
        lidar_rate_hz=(10.0 if os.environ.get('HAMS_LIDAR', '1') != '0' else 0.2),
        lidar_max_range=40.0, lidar_min_range=0.1,
        lidar_body=f"{pfx}livox_link",
        lidar_exclude_body=f"{pfx}torso_link",
        imu_quat_sensor=f"{pfx}livox_imu_quat",
        imu_gyro_sensor=f"{pfx}livox_imu_gyro",
        imu_acc_sensor=f"{pfx}livox_imu_acc",
        base_body_id=odom_base_body_id,   # -> odom -> pelvis TF + /odom
        # Drop the robot's own legs/arms from the lidar (self-returns otherwise map
        # a phantom obstacle under the robot and nav2 rejects the start). Default on
        # for this locomotion/SLAM/nav sim; HAMS_LIDAR_SELF_FILTER=0 restores them
        # as occluders.
        lidar_self_prefixes=(
            (pfx, "gripper0_")
            if os.environ.get("HAMS_LIDAR_SELF_FILTER", "1").strip().lower()
            not in ("0", "off", "false", "no")
            else ()
        ),
        sim_lock=sim_lock,
    )

    # Gripper bridges. Kitchen envs: robosuite-prefixed gripper0_<side>_ names.
    # Battery scene (raw combined XML): the right hand keeps the gripper file's
    # unprefixed actuator names but the left hand's were renamed with an _L
    # suffix (joints leftg_*) to coexist in one flat model, and BOTH sides'
    # sensors were renamed <side>_hand_* — so each side gets an explicit name
    # map instead of a prefix.
    if _battery:
        _rh_names = {
            "act_left": "left_finger_actuator", "act_right": "right_finger_actuator",
            "pos_left": "right_hand_left_finger_pos", "pos_right": "right_hand_right_finger_pos",
            "vel_left": "right_hand_left_finger_vel", "vel_right": "right_hand_right_finger_vel",
            "trq_left": "right_hand_left_finger_torque", "trq_right": "right_hand_right_finger_torque",
        }
        _lh_names = {
            "act_left": "left_finger_actuator_L", "act_right": "right_finger_actuator_L",
            "pos_left": "left_hand_left_finger_pos", "pos_right": "left_hand_right_finger_pos",
            "vel_left": "left_hand_left_finger_vel", "vel_right": "left_hand_right_finger_vel",
            "trq_left": "left_hand_left_finger_torque", "trq_right": "left_hand_right_finger_torque",
        }
        hand_right = MagpieHandBridge(model, data, side="right", sim_lock=sim_lock,
                                      names=_rh_names)
        hand_left = MagpieHandBridge(model, data, side="left", sim_lock=sim_lock,
                                     names=_lh_names)
    else:
        hand_right = MagpieHandBridge(model, data, side="right", sim_lock=sim_lock,
                                      name_prefix="gripper0_right_")
        hand_left = MagpieHandBridge(model, data, side="left", sim_lock=sim_lock,
                                     name_prefix="gripper0_left_")

    measurement = MeasurementBridge(env, elastic_band=band, task_name=task)
    print(f"[h12_mujoco] task goal: {measurement.publish_goal()!r}")

    # /hams/reset_arm listener (see the arm-reset block above). A tiny node so the
    # subscription rides the background executor; it only stamps a deadline that the
    # main loop enforces under sim_lock.
    from std_msgs.msg import Empty as _EmptyMsg
    from std_msgs.msg import Bool as _BoolMsg
    reset_node = rclpy.create_node('hams_arm_reset')
    reset_node.create_subscription(
        _EmptyMsg, '/hams/reset_arm',
        lambda _m: (_arm_reset.__setitem__('from_q', None),
                    _arm_reset.__setitem__('until', float(data.time) + _arm_reset_hold)), 10)
    # Runtime body-freeze toggle: True = pin (re-capturing the CURRENT pose as
    # the pin), False = release. Applied by the main loop (race-free).
    reset_node.create_subscription(
        _BoolMsg, '/hams/freeze_body',
        lambda m: _frz_toggle.__setitem__('req', bool(m.data)), 10)
    # In-place trial reset: restore the ENTIRE spawn state (door + objects +
    # robot) under the freeze pin — replaces a container rebuild. Main-loop
    # applied. Send while frozen; the pin is re-snapshotted to the spawn pose.
    reset_node.create_subscription(
        _EmptyMsg, '/hams/reset_scene',
        lambda _m: _scene_reset.__setitem__('req', True), 10)
    # Write the CURRENT qpos/qvel to HAMS_STATE_PATH (spawn-at-ready capture).
    reset_node.create_subscription(
        _EmptyMsg, '/hams/save_state',
        lambda _m: _state_save.__setitem__('req', True), 10)

    # Background executor serves the gripper services/timers/action + the band
    # toggle service. ros_bridge.tick() is driven from the main loop instead (its
    # MuJoCo renderer context is thread-affine), so it is NOT added here.
    executor = rclpy.executors.MultiThreadedExecutor()
    for node in (measurement, hand_right, hand_left, reset_node):
        executor.add_node(node)
    executor_thread = threading.Thread(target=executor.spin, daemon=True, name="ros_bridge_exec")
    executor_thread.start()
    print("[h12_mujoco] ROS bridges up: rt/lowstate+rt/lowcmd, /clock, "
          "RGBD /realsense/{head,left_hand,right_hand}, livox lidar+imu, "
          "/{left,right}/gripper/*, /elastic_band/toggle")

    # SPACE in the viewer toggles the band (ElasticBand.key_callback).
    handle = (
        mujoco.viewer.launch_passive(model, data, key_callback=band.key_callback)
        if viewer and band is not None
        else (mujoco.viewer.launch_passive(model, data) if viewer else None)
    )
    if handle is not None:
        handle.opt.geomgroup[0] = 0   # hide collision geoms by default
        # Frame the free camera on the robot's spawn pose (function of robot
        # position + yaw). band_body_id is the torso; if the band setup failed,
        # resolve the torso directly so framing still works.
        cam_body_id = band_body_id
        if cam_body_id < 0:
            try:
                cam_body_id = resolver.body_id("torso_link")
            except Exception:
                cam_body_id = -1
        if cam_body_id >= 0:
            try:
                with handle.lock():
                    _frame_viewer_on_robot(handle, data, cam_body_id)
            except Exception as e:
                print(f"[h12_mujoco] viewer camera framing skipped: {e}")
    # Optional offscreen third-person recording (headless benchmark runs). Frame
    # on the torso; if the band setup resolved it, reuse that id, else look it up.
    recorder = None
    steps_per_frame = 1
    if record_video:
        rec_body_id = band_body_id
        if rec_body_id < 0:
            try:
                rec_body_id = resolver.body_id("torso_link")
            except Exception:
                rec_body_id = -1
        try:
            recorder = VideoRecorder(model, data, rec_body_id, record_video)
            steps_per_frame = max(1, round(1.0 / (30.0 * model.opt.timestep)))
            print(f"[h12_mujoco] recording video -> {record_video} "
                  f"(1 frame / {steps_per_frame} sim steps, ~30 fps)", flush=True)
        except Exception as e:
            print(f"[h12_mujoco] video recording disabled: {e}", flush=True)
            recorder = None

    # Fall logger: watch the torso's uprightness + height and log once when the
    # robot falls, including how long it stood after the elastic band was released.
    _fall_logged = False
    _band_release_t = None
    _band_prev_on = bool(band.enabled) if band is not None else False

    _step_idx = 0
    try:
        while True:
            start_time = time.time()
            if viewer and not handle.is_running():
                break
            with sim_lock:
                if band is not None:
                    # Re-write every step; MuJoCo persists xfrc_applied, so we must
                    # zero it when the band is disabled or the last force lingers.
                    data.xfrc_applied[band_body_id, :3] = (
                        band.evaluate_force(data.xpos[band_body_id], data.qvel[base_dof:base_dof + 3])
                        if band.enabled else 0.0
                    )
                if sim_interface is not None:
                    # Recompute the DDS command's PD torque every sim step (emulates
                    # the real motor's ~1kHz servo) instead of only on each 50Hz
                    # /lowcmd, which held a stale-damping torque and toppled FAME.
                    sim_interface.write_ctrl()
                mujoco.mj_step(model, data)
                # Runtime freeze toggle request (from /hams/freeze_body).
                # Re-freezing captures the CURRENT pose as the new pin.
                if _frz_toggle['req'] is not None and _frz_base_qadr >= 0:
                    _rq = _frz_toggle['req']
                    _frz_toggle['req'] = None
                    if _rq and not _frz_toggle['on']:
                        _frz_base_qpos = data.qpos[_frz_base_qadr:_frz_base_qadr + 7].copy()
                        _frz_lower_qpos = data.qpos[_frz_lower_qidx].copy()
                        if band is not None:
                            band.enabled = False
                        print(f"[h12_mujoco] body FROZEN at t={data.time:.2f}s "
                              "(runtime toggle)", flush=True)
                    elif not _rq and _frz_toggle['on']:
                        print(f"[h12_mujoco] body RELEASED at t={data.time:.2f}s "
                              "(runtime toggle)", flush=True)
                    _frz_toggle['on'] = _rq
                # In-place scene reset (under the pin): full spawn-state restore.
                # Re-snapshots the freeze pin to the spawn pose so the frozen
                # robot is teleported home (fallen/wandered robots recover
                # without a rebuild). Forces the freeze ON — the caller releases
                # it via /hams/freeze_body once its policy is ready.
                if _state_save['req']:
                    _state_save['req'] = False
                    if _state_path:
                        try:
                            np.savez(_state_path, qpos=data.qpos.copy(), qvel=data.qvel.copy())
                            print(f"[h12_mujoco] STATE SAVED to {_state_path} at t={data.time:.2f}s", flush=True)
                        except Exception as e:
                            print(f"[h12_mujoco] state save failed: {e}", flush=True)
                if _scene_reset['req']:
                    _scene_reset['req'] = False
                    data.qpos[:] = _scene_qpos0
                    data.qvel[:] = 0.0
                    if _frz_base_qadr >= 0:
                        _frz_base_qpos = data.qpos[_frz_base_qadr:_frz_base_qadr + 7].copy()
                        _frz_lower_qpos = data.qpos[_frz_lower_qidx].copy()
                    _frz_toggle['on'] = True
                    _frz_toggle['req'] = None
                    _fall_logged = False        # re-arm the fall logger
                    mujoco.mj_forward(model, data)
                    print(f"[h12_mujoco] SCENE RESET to spawn state at t={data.time:.2f}s "
                          "(frozen; release via /hams/freeze_body)", flush=True)
                # Between-trial arm reset window (works FROZEN OR NOT): pin the 14
                # arm joints to home and overwrite their PD setpoint, so a stuck
                # controller command can't drag the arm back off home while the
                # window is open. Unfrozen, the lower body keeps its own dynamics.
                # HAMS_ARM_RESET_RAMP > 0 (sim-s) interpolates qpos to home over
                # the ramp window instead of snapping in one step: on a FREE
                # dynamic base the instantaneous 14-joint teleport is a violent
                # impulse that launches the robot; the ramp keeps the reset
                # deterministic but quasi-static. Default 0 = legacy snap.
                _arm_pinned = float(data.time) < _arm_reset['until']
                if _arm_pinned:
                    if _arm_reset_ramp > 0.0:
                        _t_left = _arm_reset['until'] - float(data.time)
                        _t_in = (_arm_reset_hold - _t_left)
                        if _t_in <= 1e-9:                     # window just opened
                            _arm_reset['from_q'] = data.qpos[_arm_qidx].copy()
                        _a = min(1.0, max(0.0, _t_in / _arm_reset_ramp))
                        _from = _arm_reset.get('from_q')
                        if _from is None:
                            _from = data.qpos[_arm_qidx].copy()
                            _arm_reset['from_q'] = _from
                        _q_t = (1.0 - _a) * _from + _a * _arm_home_q
                        data.qpos[_arm_qidx] = _q_t
                        data.qvel[_arm_vidx] = 0.0
                        data.ctrl[_arm_cidx] = _q_t
                    else:
                        data.qpos[_arm_qidx] = _arm_home_q
                        data.qvel[_arm_vidx] = 0.0
                        data.ctrl[_arm_cidx] = _arm_home_q
                if _frz_toggle['on']:
                    # Re-pin the pelvis + legs + torso to the pin pose after the
                    # step so the body is perfectly static; only the arm DOFs
                    # (and scene objects) evolve. mj_forward refreshes xpos/sensors
                    # for the viewer + bridges to match the pinned pose.
                    data.qpos[_frz_base_qadr:_frz_base_qadr + 7] = _frz_base_qpos
                    data.qvel[_frz_base_dof:_frz_base_dof + 6] = 0.0
                    data.qpos[_frz_lower_qidx] = _frz_lower_qpos
                    data.qvel[_frz_lower_vidx] = 0.0
                if _frz_toggle['on'] or _arm_pinned:
                    mujoco.mj_forward(model, data)
                # --- fall logger ---
                if band is not None and _band_prev_on and not band.enabled:
                    _band_release_t = float(data.time)
                    _band_prev_on = False
                    print(f"[fall-logger] elastic band released at t={data.time:.2f}s", flush=True)
                if band_body_id >= 0 and not _fall_logged:
                    _upz = float(data.xmat[band_body_id][8])   # torso z-axis . world up (1=upright)
                    _torso_h = float(data.xpos[band_body_id][2])
                    if _upz < 0.5 or _torso_h < 0.5:
                        _fall_logged = True
                        _since = (f"{data.time - _band_release_t:.2f}s after band release"
                                  if _band_release_t is not None
                                  else f"{data.time:.2f}s (band still on)")
                        print(f"[fall-logger] *** ROBOT FELL at sim t={data.time:.2f}s — stood "
                              f"{_since} (uprightness={_upz:.2f}, torso_h={_torso_h:.2f}) ***",
                              flush=True)
                env.update_state()                  # REQUIRED before _check_success
                ros_bridge.tick()                   # /clock + camera + lidar + imu
                done = measurement.tick()
            if done:
                print("[h12_mujoco] task success (debounced).")
            # Offscreen recording runs on the main thread (MuJoCo renderer is
            # thread-affine) outside sim_lock: qpos is only written by mj_step
            # here, so the read is consistent without blocking the bridge threads.
            if recorder is not None and _step_idx % steps_per_frame == 0:
                try:
                    recorder.capture(data)
                except Exception as e:
                    print(f"[h12_mujoco] video capture failed, stopping recorder: {e}",
                          flush=True)
                    recorder.close()
                    recorder = None
            _step_idx += 1
            if viewer:
                if band is not None and band.enabled:
                    _draw_band(handle, band.point, data.xpos[band_body_id])
                else:
                    handle.user_scn.ngeom = 0   # clear overlay when band off
                handle.sync()
            time.sleep(max(0, model.opt.timestep - (time.time() - start_time)))
    finally:
        if recorder is not None:
            recorder.close()
        if handle is not None:
            handle.close()
        executor.shutdown()
        ros_bridge.shutdown()
        for node in (measurement, hand_right, hand_left, ros_bridge):
            node.destroy_node()
        shutdown_ros()


def _runnable_tasks():
    """Return (env_classes, runnable).

    `env_classes` maps every robosuite-registered env name -> its class. This is
    the same REGISTERED_ENVS that robosuite.make() resolves against, and it DOES
    include RoboCasa's abstract bases (OpenDoor/CloseDoor/ManipulateDoor/
    PickPlace) — RoboCasa's metaclass deliberately omits those five from its own
    REGISTERED_KITCHEN_ENVS, but the robosuite parent metaclass still registers
    them. We need them here so an abstract base name can be looked up and expanded
    to its concrete subclasses.

    `runnable` is the set of concrete benchmark task names from RoboCasa's curated
    ATOMIC_/COMPOSITE_TASK_DATASETS (intersected with what's registered, to
    tolerate version skew). Those bases can't be instantiated directly — they
    require a `fixture_id` only a concrete subclass supplies — so they are never
    in `runnable`. Falls back to REGISTERED_KITCHEN_ENVS (which already excludes
    the five bases) if the curated lists are unavailable.
    """
    import robocasa  # noqa: F401  -> registers all kitchen envs on import
    from robosuite.environments.base import REGISTERED_ENVS
    env_classes = dict(REGISTERED_ENVS)
    try:
        from robocasa.utils.dataset_registry import (
            ATOMIC_TASK_DATASETS,
            COMPOSITE_TASK_DATASETS,
        )
        runnable = (set(ATOMIC_TASK_DATASETS) | set(COMPOSITE_TASK_DATASETS)) & set(env_classes)
    except Exception:
        from robocasa.environments.kitchen.kitchen import REGISTERED_KITCHEN_ENVS
        runnable = set(REGISTERED_KITCHEN_ENVS) & set(env_classes)
    return env_classes, runnable


def _random_task(seed=None):
    """Pick a random *runnable* (concrete) RoboCasa kitchen env name."""
    _, runnable = _runnable_tasks()
    if seed is not None:
        random.seed(seed)
    return random.choice(sorted(runnable))


def _resolve_task(name, seed=None):
    """Resolve a user-supplied --task name to a concrete runnable task.

    If `name` is already a concrete benchmark task, return it unchanged. If it's
    an abstract base (e.g. OpenDoor, ManipulateDoor) — registered but not directly
    instantiable — randomly pick one of its concrete runnable subclasses instead
    of crashing.
    """
    env_classes, runnable = _runnable_tasks()
    if name in runnable:
        return name
    if name not in env_classes:
        raise SystemExit(f"[h12_mujoco] unknown RoboCasa task {name!r}")
    base = env_classes[name]
    concrete = sorted(
        n for n in runnable
        if isinstance(env_classes.get(n), type)
        and issubclass(env_classes[n], base) and env_classes[n] is not base
    )
    if not concrete:
        raise SystemExit(f"[h12_mujoco] {name!r} is abstract with no runnable concrete tasks")
    if seed is not None:
        random.seed(seed)
    choice = random.choice(concrete)
    print(f"[h12_mujoco] {name!r} is an abstract task base; "
          f"randomly selected concrete subclass {choice!r} "
          f"(from {len(concrete)} options)")
    return choice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the H1-2 RoboCasa MuJoCo sim")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the MuJoCo passive viewer (default: viewer on)",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="RoboCasa env name (e.g. TurnOnToasterOven). "
             "Omit to launch a random registered kitchen task.",
    )
    parser.add_argument("--layout", type=int, default=None, help="RoboCasa kitchen layout id")
    parser.add_argument("--style", type=int, default=None, help="RoboCasa kitchen style id")
    parser.add_argument("--seed", type=int, default=None, help="episode seed")
    parser.add_argument("--record-video", default=None, metavar="PATH",
                        help="Write an offscreen third-person mp4 of the run to "
                             "PATH (streamed; no passive viewer required).")
    # Bigger frames cost host RAM on an already-tight box; 256 is enough for the
    # gemini detector to box the target (see sim_loop). Raise deliberately.
    parser.add_argument("--cam-width", type=int, default=256,
                        help="RGBD camera render width (default 256)")
    parser.add_argument("--cam-height", type=int, default=256,
                        help="RGBD camera render height (default 256)")
    parser.add_argument("--no-hold-legs", action="store_true",
                        help="Leave the legs limp when no lower-body controller is "
                             "running (default: hold them at the spawn stance, which "
                             "keeps the robot's pose reproducible across episodes)")
    args = parser.parse_args()

    task = args.task
    if task is None:
        task = _random_task(seed=args.seed)
        print(f"[h12_mujoco] no --task given; randomly selected {task!r}")
    elif task == 'BatteryWorkcell':
        # Not a RoboCasa registry env — resolved by sim_loop via battery_env.
        print("[h12_mujoco] BatteryWorkcell: loading the self-contained "
              "battery scene (no RoboCasa)")
    else:
        task = _resolve_task(task, seed=args.seed)

    sim_loop(task, viewer=not args.headless,
             layout=args.layout, style=args.style, seed=args.seed,
             record_video=args.record_video,
             cam_width=args.cam_width, cam_height=args.cam_height,
             hold_legs=not args.no_hold_legs)
