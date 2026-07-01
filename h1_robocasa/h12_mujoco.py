import argparse
import math
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


def sim_loop(task, viewer=True, layout=None, style=None, seed=None):
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
    resolver = NameResolver(model)  # ROS<->sim name map for the DDS / sensor bridges

    # Clean, contact-free initial state. RoboCasa's zero-action settling loop
    # perturbs the pose during reset, so reset every actuated joint to 0 (all-zero
    # spawn pose), zero all velocities, and re-place the pelvis. At the zero pose
    # the arms jut forward and can overlap the counter, so place_robot_collision_free
    # auto-fits floor clearance AND backs the base away (the robot's -x) until no
    # robot geom penetrates a fixture, keeping the least-penetrating spot if it
    # can't fully clear. For an upright standing spawn instead, write
    # h1_2_robosuite.nominal_stance_vector() to the motor qpos here.
    try:
        data.qpos[resolver.motor_qpos] = 0.0
        data.qvel[:] = 0.0
        h1_2_robosuite.place_robot_collision_free(
            env, env.init_robot_base_pos,
            h1_2_robosuite._euler_to_wxyz(getattr(env, "init_robot_base_ori", None)),
        )
    except Exception as e:
        print(f"[h12_mujoco] clean reset skipped: {e}")

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

    # DDS control bridge: publishes rt/lowstate, subscribes rt/lowcmd, drives the
    # 27 body motors by name via the resolver (grippers handled by the hand
    # bridges below; gripper ctrl indices are disjoint from the body motors).
    sim_interface = SimInterface(model, data, lock=sim_lock, resolver=resolver)  # noqa: F841

    init_ros()

    # Sensor bridge: /clock, RGBD cameras (head + both hands), livox lidar + IMU.
    # Pass robosuite-prefixed MuJoCo lookup names; ROS-facing frame_ids stay
    # unprefixed (ctor defaults). cameras: (mujoco name, /realsense/<ns>, frame_id).
    # Head rides the robot prefix; the eye-in-hand gripper cameras ride the gripper
    # prefixes (same as the hand bridges below).
    ros_bridge = RosSensorBridge(
        model, data,
        cameras=[
            (f"{pfx}head_cam",          "head",       "camera_color_optical_frame"),
            ("gripper0_left_hand_cam",  "left_hand",  "left_hand_camera_color_optical_frame"),
            ("gripper0_right_hand_cam", "right_hand", "right_hand_camera_color_optical_frame"),
        ],
        cam_width=256, cam_height=256,   # all 3 cameras render at 256x256 (RoboCasa default)
        # MID-360 fidelity: 360x56 @ 10Hz ~= 201k pts/s (real ~200k), 0.1m near /
        # 40m far range, per-point offset_time for FAST-LIO deskew. el_rays/rate
        # are the knobs to dial back if the ray cast (main thread) hurts RTF.
        lidar_az_rays=360, lidar_el_rays=56,
        lidar_rate_hz=10.0, lidar_max_range=40.0, lidar_min_range=0.1,
        lidar_body=f"{pfx}livox_link",
        lidar_exclude_body=f"{pfx}torso_link",
        imu_quat_sensor=f"{pfx}livox_imu_quat",
        imu_gyro_sensor=f"{pfx}livox_imu_gyro",
        imu_acc_sensor=f"{pfx}livox_imu_acc",
        sim_lock=sim_lock,
    )

    # Gripper bridges (gripper0_<side>_ prefixed actuators/sensors).
    hand_right = MagpieHandBridge(model, data, side="right", sim_lock=sim_lock,
                                  name_prefix="gripper0_right_")
    hand_left = MagpieHandBridge(model, data, side="left", sim_lock=sim_lock,
                                 name_prefix="gripper0_left_")

    measurement = MeasurementBridge(env, elastic_band=band, task_name=task)
    print(f"[h12_mujoco] task goal: {measurement.publish_goal()!r}")

    # Background executor serves the gripper services/timers/action + the band
    # toggle service. ros_bridge.tick() is driven from the main loop instead (its
    # MuJoCo renderer context is thread-affine), so it is NOT added here.
    executor = rclpy.executors.MultiThreadedExecutor()
    for node in (measurement, hand_right, hand_left):
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
    # Fall logger: watch the torso's uprightness + height and log once when the
    # robot falls, including how long it stood after the elastic band was released.
    _fall_logged = False
    _band_release_t = None
    _band_prev_on = bool(band.enabled) if band is not None else False

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
            if viewer:
                if band is not None and band.enabled:
                    _draw_band(handle, band.point, data.xpos[band_body_id])
                else:
                    handle.user_scn.ngeom = 0   # clear overlay when band off
                handle.sync()
            time.sleep(max(0, model.opt.timestep - (time.time() - start_time)))
    finally:
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
    args = parser.parse_args()

    task = args.task
    if task is None:
        task = _random_task(seed=args.seed)
        print(f"[h12_mujoco] no --task given; randomly selected {task!r}")
    else:
        task = _resolve_task(task, seed=args.seed)

    sim_loop(task, viewer=not args.headless,
             layout=args.layout, style=args.style, seed=args.seed)
