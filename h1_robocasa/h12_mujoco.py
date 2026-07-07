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


def _draw_band(handle, anchor, body_pos, slack=False):
    """Draw the elastic band (anchor sphere + capsule to the torso) in the passive
    viewer's user scene. Mirrors the legacy MujocoEnv.draw_elastic_band.

    slack=True colors it GREEN (a tension-only strap paid out past the
    torso-to-anchor distance exerts nothing = robot free-standing), blue while
    taut -- the operator's cue to pay out with '.' until it turns GREEN, exactly
    as on the twin."""
    scn = handle.user_scn
    scn.ngeom = 0
    color = (np.array([0.3, 1.0, 0.4, 0.8]) if slack
             else np.array([0.5, 0.6, 1.0, 0.8]))
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


def sim_loop(task, viewer=True, layout=None, style=None, seed=None, no_sensors=False,
             band_auto_release=False, band_release_force=30.0,
             band_release_sustain=2.0, band_release_min_time=15.0,
             harness=False, harness_kp=300.0, harness_kd=60.0,
             harness_xy_kp=1000.0, harness_xy_kd=200.0,
             harness_hold_kp=80.0, harness_hold_kd=8.0,
             harness_slack=0.05, harness_fade=3.0, harness_release_torque=15.0,
             harness_payout_rate=0.02, harness_payout_max=1.5, harness_payout_quiet=20.0,
             harness_payout_chunk=0.10, harness_payout_settle=6.0,
             harness_holdout=15.0, harness_hands_fade=10.0,
             harness_grip_boost=5.0, harness_force_release_after=0.0,
             rigid_feet=True, foot_solref=0.008, sim_dt=0.0,
             sim_implicitfast=True, twin_contact=False,
             elliptic_contact=False,
             multiccd=True, truth_sportstate=False, plane_floor=False,
             spawn_crouch=0.0, auto_release=False, auto_release_delay=2.0,
             auto_release_fade=3.0, real_hands=False, slowmo=1.0):
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

    # SURGICAL FOOT<->FLOOR CONTACT OVERRIDE (parity fix, 2026-07-03).
    # The assembled robosuite model runs GLOBAL opt impratio=20 / cone=elliptic /
    # integrator=Euler (grasp-tuned) vs the planner's belief of rigid no-slip
    # pyramidal contact at impratio=100. So the stance foot creeps laterally under
    # load and the strat-6 stand folds ~5s after policy blend (confirmed: opt dump
    # + gravity_ff A/B both fold, lateral signature). impratio/cone are GLOBAL and
    # grasp-critical, so instead of touching them we harden ONLY the foot geoms and
    # give them contact PRIORITY, so their (stiff, grippy) params win the foot<->
    # ground pair while every other contact keeps robosuite's manipulation-tuned
    # defaults. Post-compile geom_* mutation is read live by the collision pipeline.
    if rigid_feet:
        _n = 0
        for _side in ("left", "right"):
            _bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                     f"{env.robots[0].robot_model.naming_prefix}{_side}_ankle_roll_link")
            if _bid < 0:
                continue
            for _g in range(model.ngeom):
                if model.geom_bodyid[_g] == _bid and model.geom_contype[_g]:
                    model.geom_priority[_g] = 2          # win the pair vs the floor (priority 0)
                    model.geom_condim[_g] = 4            # normal + 2 slide + TORSIONAL (kills yaw creep)
                    # PLANNER-PARITY friction (2026-07-04, wobble root cause R2):
                    # with PYRAMIDAL cones all pyramid rows share one regularizer
                    # Rpy = 2*mu^2*R0/impratio -- the contact NORMAL is over-
                    # stiffened by ~impratio/(2*mu^2). Planner-model feet (mu=1)
                    # get ~50x; the old mu=2.0 here got ~12.5x = a 4x planner<->
                    # plant contact-impedance MISMATCH. mu=1.0 carries the SAME
                    # artifact magnitude as the planner's belief (docs: high
                    # impratio + pyramidal is "not recommended" -- but it is the
                    # validated twin/planner combo, so match it, don't fix it).
                    model.geom_friction[_g] = [1.0, 0.06, 0.01]  # slide / torsional / rolling
                    # --foot-solref: 0.008 was the original anti-sink hammer
                    # (pre-multiccd). Under pyramidal impratio-100 the normal is
                    # ~50x over-stiffened, so 0.008 at dt 0.002 = an effective
                    # ~1 ms contact timeconstant (chatter territory); the twin's
                    # quiet feet are default 0.02 at dt 0.005. Tunable for the
                    # L4 discriminating take.
                    model.geom_solref[_g] = [foot_solref, 1.0]
                    _n += 1
        print(f"[h12_mujoco] rigid-feet: hardened {_n} foot collision geoms "
              "(priority 2, condim 4, friction [1,0.06,0.01] planner-parity, "
              f"solref [{foot_solref},1]) to counter the global impratio=20/"
              "elliptic foot-slip")

    # FLOOR-MANIFOLD PARITY (2026-07-05, T-box twin A/B): the kitchen walking
    # surface is a THIN BOX (floor_*_room_g*, 0.02 half-thickness) while the
    # twin + the planner's internal model stand on an infinite PLANE. Putting
    # the SAME thin box under the twin (T1b chain otherwise identical, multiccd
    # on) flipped its 218-s free stand into a 2-s forward face-plant -- the
    # box-mesh dynamic contact (even at 4 pts/foot under multiccd) is the last
    # RoboCasa<->twin delta. Fix = make the floor a PLANE for contact: convert
    # the walking-surface boxes to planes in-place (top face height preserved;
    # planes are static-body geoms so fixture pairs stay excluded; movables
    # rest at the same height). geom_type/pos/rbound are read live by the
    # collision pipeline; rbound=0 marks the geom unbounded for broadphase.
    if plane_floor:
        import re as _re
        _nf = 0
        for _g in range(model.ngeom):
            _gn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, _g) or ""
            if (_re.match(r"floor_\d+_room_g\d+$", _gn)
                    and model.geom_type[_g] == mujoco.mjtGeom.mjGEOM_BOX):
                if abs(model.geom_quat[_g][0] - 1.0) > 1e-6 or \
                   np.any(np.abs(model.geom_quat[_g][1:]) > 1e-6):
                    print(f"[h12_mujoco] plane-floor: SKIP '{_gn}' (rotated box)")
                    continue
                _top = float(model.geom_pos[_g][2] + model.geom_size[_g][2])
                model.geom_type[_g] = mujoco.mjtGeom.mjGEOM_PLANE
                model.geom_pos[_g][2] = _top
                model.geom_rbound[_g] = 0.0
                _nf += 1
                print(f"[h12_mujoco] plane-floor: '{_gn}' box->PLANE, surface "
                      f"z={_top:.3f} (twin/planner manifold parity)")
        if _nf == 0:
            print("[h12_mujoco] plane-floor: *** no floor_*_room_g* box found — "
                  "flag had no effect ***")

    # INTEGRATOR PARITY (2026-07-03). robosuite assembles the model with the
    # MuJoCo-default EULER integrator; the twin, the planner's Stabilize/Lean
    # rollout models, and real's planner belief all use IMPLICITFAST (both task
    # XMLs: "Euler diverges on biped balance per the docs"). Euler integrates the
    # joint damping=10/armature=0.1 + Coriolis EXPLICITLY, giving a velocity-decay
    # response the planner's implicitfast rollouts don't predict -> tracking error
    # accumulates over the blend -> the residual tilt wobble. Unlike impratio/cone
    # this is ODE time-stepping, NOT a contact/grasp parameter, so matching it to
    # the rest of the stack is manipulation-safe. Post-compile opt mutation is read
    # live by mj_step. --keep-euler leaves robosuite's default for an A/B.
    if sim_implicitfast:
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        print("[h12_mujoco] integrator: Euler -> implicitfast (parity with twin/"
              "planner; the whole stack is implicitfast, robosuite defaulted Euler)")

    # GLOBAL CONTACT PARITY (2026-07-03) -- OPT-IN, balance-only. impratio and cone
    # are GLOBAL <option> settings with NO per-geom equivalent, so the surgical
    # rigid-feet fix (friction/solref/priority) could NOT touch them: the plant
    # stayed at robosuite's grasp-tuned impratio=20 / cone=elliptic while the twin+
    # planner run impratio=100 / pyramidal (stiff, near-no-slip friction). This flag
    # matches the twin's contact model globally -- DEFAULT OFF so manipulation runs
    # keep robosuite's grasp-tuned 20/elliptic; the balance-only bringup
    # (use_skills:=false) passes --twin-contact. Confirms whether the residual lean
    # is the contact globals the local foot fix can't reach.
    if elliptic_contact:
        # DOCS' NO-SLIP RECIPE (root-caused 2026-07-06 via /mujoco-docs
        # XMLreference option-impratio + option-cone). impratio "determines the
        # ratio of frictional-to-normal constraint impedance FOR ELLIPTIC FRICTION
        # CONES ... larger than 1 ... prevents slip WITHOUT increasing the actual
        # friction coefficient. For pyramidal cones ... NOT recommended." So the
        # anti-slip lever ONLY works on the ELLIPTIC cone. RoboCasa is natively
        # elliptic but ships impratio=20 (grasp-tuned, weak anti-slip); the
        # planner/twin (Stabilize_H12_Magpie.xml) run impratio=100, so the stance
        # foot creeps and the controller -- which plans on rigid no-slip feet --
        # topples with the lateral-drift->one-knee-strut signature. FIX = raise the
        # plant's ELLIPTIC impratio 20->100 to the planner's value (no cone change
        # = grasp parity kept) + force the Newton solver (the recipe's third leg:
        # "elliptic cones, large impratio, Newton"; MuJoCo's default, enforced here
        # so an elliptic cone can never land on a dual solver). This is the CLEAN
        # single flag; the --twin-contact branch below is the PRIOR FAILED attempt
        # (pyramidal makes impratio inert -> it over-stiffens the NORMAL instead of
        # hardening friction -> tilt 13->21, the slip it was meant to fix).
        model.opt.impratio = 100.0
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        print("[h12_mujoco] elliptic-contact (NO-SLIP recipe): impratio 20->100, "
              "cone ELLIPTIC (native/grasp-safe), solver Newton -- friction now "
              "100x harder than normal so the stance foot stops creeping; matches "
              "the planner's impratio=100 belief. Balance-only; grasp omits the flag.")
    elif twin_contact:
        # PRIOR FAILED ATTEMPT, kept for A/B only: pyramidal + impratio 100. Per the
        # MuJoCo docs impratio is INERT on pyramidal cones (each basis vector mixes
        # normal+friction) -- it over-stiffens the NORMAL ~impratio/(2*mu^2) instead
        # of hardening friction, which is WHY this made slip/tilt WORSE (tilt 13->21,
        # 2026-07-04 A/B). Prefer --elliptic-contact.
        model.opt.impratio = 100.0
        model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        print("[h12_mujoco] twin-contact: impratio 20->100, cone elliptic->PYRAMIDAL "
              "(A/B only -- impratio inert on pyramidal per MuJoCo docs; prefer "
              "--elliptic-contact)")

    # PLANT TIMESTEP PARITY (2026-07-04, T2-vs-T3b twin delta-walk: THE stand
    # discriminator). The twin runs dt 0.005; robosuite assembles 0.002. Under
    # the deploy chain the same twin STANDS at 0.005 (even paced to RTF 0.5,
    # T3b: slack rope + honest gate) and HANGS at 0.002 (T2: 550 N deep crouch)
    # -- the validated stack's gains/gff/weights are implicitly tuned to the
    # 0.005 plant's numerical dissipation; a 0.002 plant is "livelier" than
    # everything downstream expects. Runtime opt.timestep override is the
    # documented-supported path; the lowstate publisher interval and
    # tick=int(t/dt) follow it automatically (set BEFORE SimInterface).
    # Pass --sim-dt 0.005 for the balance bench (node twin_dt must match);
    # 0 = keep robosuite's assembled dt (grasp default). BONUS: 2.5x fewer
    # steps -> RTF rises ~0.25 -> ~0.5.
    if sim_dt > 0.0:
        model.opt.timestep = sim_dt
        print(f"[h12_mujoco] plant timestep OVERRIDDEN to {sim_dt} (twin parity; "
              f"pass twin_dt {sim_dt} to the node + estimator tick_dt)")

    # CONTACT MANIFOLD PARITY (2026-07-04 root cause, P2). The twin/planner floor
    # is an infinite PLANE: the dedicated plane-mesh collider returns up to 3
    # contact points per foot. The kitchen floor is BOX geoms: box-mesh goes
    # through generic convex CCD which returns ONE contact point per foot unless
    # multiccd is enabled (default-on only since MuJoCo 3.8; this container pins
    # 3.3.1 where it is OFF). One point per foot = no ankle moment transmits and
    # the contact point migrates -- the docs' literal "sliding or wobbling". This
    # was the structural floor-GEOMETRY gap rigid-feet could only partially paper
    # over. multiccd gives box-mesh 4-5 points = a real support polygon per foot,
    # same class as the twin's 3-point plane contact. --no-multiccd for A/B.
    if multiccd:
        model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
        print("[h12_mujoco] multiccd ON: box-floor x mesh-foot now yields a multi-"
              "point contact manifold (was 1 pt/foot at mujoco 3.3.1 defaults = "
              "no ankle moments; twin's plane-mesh collider gives 3 pts/foot)")

    # STARTUP PARITY CHECKS (2026-07-04). (C) The whole external chain -- node
    # twin_dt, estimator tick_dt, lowstate tick=int(time/dt) -- assumes the
    # assembled timestep is 0.002. robosuite owns <option> assembly, so verify
    # loudly instead of silently drifting the plant clock 2x. (D) The Magpie
    # gripper bodies are gravcomp'd (weightless statics, mesh-derived mass) while
    # the controller model carries 0.506 kg real load per hand -- log the actual
    # subtree mass so the residual is a number, not a mystery.
    _dt_expect = sim_dt if sim_dt > 0.0 else 0.002
    if abs(model.opt.timestep - _dt_expect) > 1e-9:
        print(f"[h12_mujoco] *** TIMESTEP MISMATCH: model.opt.timestep="
              f"{model.opt.timestep} != {_dt_expect} expected by the external "
              f"chain (node twin_dt / estimator tick_dt / lowstate tick). Fix "
              f"the configs before trusting any run. ***")
    else:
        print(f"[h12_mujoco] timestep check: {model.opt.timestep} (node twin_dt "
              f"/ estimator tick_dt must match)")
    try:
        _gc_n = int(np.count_nonzero(model.body_gravcomp))
        _gmass = {"left": 0.0, "right": 0.0}
        for _b in range(model.nbody):
            _bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, _b) or ""
            if "gripper0" in _bn:
                _side = "right" if "right" in _bn else "left"
                _gmass[_side] += float(model.body_mass[_b])
        print(f"[h12_mujoco] gripper mass probe: left={_gmass['left']:.3f}kg "
              f"right={_gmass['right']:.3f}kg (controller model believes 0.506kg "
              f"welded/uncompensated per hand); gravcomp bodies={_gc_n}; "
              f"total model mass={float(np.sum(model.body_mass)):.3f}kg")
    except Exception as _e:
        print(f"[h12_mujoco] gripper mass probe skipped: {_e}")

    # --real-hands: the ONE robot-model delta vs the twin (2026-07-06 static
    # audit: masses, inertials, foot mesh, joint armature/damping/frictionloss
    # all match the twin to the digit). RoboCasa's hands are gravcomp'd
    # (weightless), so the deploy controller -- which feed-forwards gravity for a
    # ~0.5 kg hand -- over-torques the wrists against hands that carry no weight
    # (the twin's hands DO carry ~0.3 kg, which is why it holds). Removing the
    # gravcomp gives the hands real weight = twin parity. Opt-in so grasp runs
    # (which want weightless statics) are untouched.
    if real_hands:
        _rn = 0
        for _b in range(model.nbody):
            _bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, _b) or ""
            if ("gripper0" in _bn or _bn.endswith("_hand")) and model.body_gravcomp[_b]:
                model.body_gravcomp[_b] = 0.0
                _rn += 1
        print(f"[h12_mujoco] real-hands: gravcomp OFF on {_rn} hand/gripper "
              "bodies (twin parity -- hands now carry real weight; controller "
              "feed-forwards for weighted, not weightless, hands)")

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
        # SPAWN CROUCH (--spawn-crouch K>0): start the legs KNEE-BENT (knee=+K,
        # hipP=-K/2, ankP=-K/2) instead of straight. A straight-knee spawn hands
        # the policy the hyperextension / one-leg-strut singularity the instant it
        # takes over (the deploy ramp's knee=0 "home" basin); the stand keyframe
        # is itself bent (knee 0.35). The joint hold then FREEZES this good bent
        # pose until the controller rises from it -- the twin's --crouch, ported.
        if spawn_crouch > 0.0:
            # Leg joints are RENAMED in the robosuite model (sim_names.
            # LEG_JOINT_RENAME) -- the earlier name reconstruction
            # ("robot0_left_knee_joint") resolved to -1 and set NOTHING while the
            # print still fired (silent no-op -> straight-leg spawn). Drive the
            # crouch through the RESOLVER's validated motor->qpos map instead.
            # ROS_MOTOR_ORDER indices: 1/3/4 = L hipP/knee/ankP, 7/9/10 = R.
            # Re-plant by the change in the robot's LOWEST body (naming-agnostic).
            _base_q = int(model.jnt_qposadr[fj])
            _rbody = [b for b in range(model.nbody)
                      if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
                          or "").startswith(resolver.robot_prefix)]
            _low = lambda: (min(float(data.xpos[b][2]) for b in _rbody)
                            if _rbody else None)
            _foot0 = _low()
            for _mi, _val in ((1, -spawn_crouch / 2.0), (3, spawn_crouch),
                              (4, -spawn_crouch / 2.0), (7, -spawn_crouch / 2.0),
                              (9, spawn_crouch), (10, -spawn_crouch / 2.0)):
                data.qpos[resolver.motor_qpos[_mi]] = _val
            mujoco.mj_forward(model, data)
            _foot1 = _low()
            if _foot0 is not None and _foot1 is not None:
                data.qpos[_base_q + 2] -= (_foot1 - _foot0)   # keep feet on floor
                mujoco.mj_forward(model, data)
            print(f"[h12_mujoco] SPAWN CROUCH: knees {spawn_crouch:.2f} rad, "
                  f"hipP/ankP -{spawn_crouch/2.0:.2f} (via resolver map; feet "
                  "re-planted); joint hold freezes it until the controller rises")
        if harness:
            # Twin-validated gantry-harness handoff (the REAL bring-up procedure:
            # support + hold straight while the controller stands the robot up,
            # release only once it carries itself). TENSION-ONLY strap anchored
            # 0.5 m above the torso spawn with a small slack budget — it can
            # catch a sag/fall but exerts NOTHING once the robot stands at/above
            # its spawn height, unlike the hang band (a point spring the policy
            # provably winds up and hangs on: 223-309 N measured 2026-07-03).
            # The limp robot first settles into a knee-bent crouch on the strap
            # (= the physically-reached crouch start of the twin winch recipe);
            # the deploy core's plant-clocked ramp then stands it up and the
            # strap slackens. Tilt steadying comes from the posture assist
            # applied in the sim loop below — a point strap cannot constrain
            # tilt, and the supported robot otherwise swings like a pendulum.
            # 3000/300 strap + 0.8 m anchor height = the twin rig's exact
            # constants (h1_mujoco crouch-rehearsal harness). Stiff strap =
            # webbing, not bungee: the limp pre-controller robot settles only
            # ~15-20 cm onto it (a knee-bent crouch), and the damping is low
            # enough not to poison the slack signal with sway-rate force.
            anchor = data.xpos[band_body_id].copy()
            anchor[2] += 0.8
            band = ElasticBand(anchor, length=0.8 + harness_slack,
                               stiffness=3000, damping=300, tension_only=True,
                               posture_kp=harness_kp, posture_kd=harness_kd,
                               release_fade=harness_fade)
            print(f"[h12_mujoco] HARNESS on torso: anchor {anchor.round(3)}, "
                  f"slack {harness_slack*100:.0f}cm, hands kp/kd "
                  f"{harness_kp:.0f}/{harness_kd:.0f}, fade {harness_fade:.1f}s")
        else:
            # MANUAL harness (no --harness auto logic): the SAME tension-only
            # catch strap as the automated harness, but driven ENTIRELY by the
            # viewer keys (U/L hoist, arrows move anchor, ./,  rope, T tilt
            # assist, SPACE eased release) plus a live engagement readout in the
            # loop below. Twin-parity manual bring-up on RoboCasa: lower to a
            # knee-bent crouch, start the controller, watch the readout, pay out
            # (.) to slack + SPACE to release once it carries itself. Posture
            # assist starts OFF (press T for steadying hands).
            anchor = data.xpos[band_body_id].copy()
            anchor[2] += 0.8
            # posture assist ON by default (steadying hands 300/60) + the joint
            # hold in the loop = the limp robot is held UPRIGHT at its spawn pose
            # until the controller engages, instead of folding forward on the
            # point strap. Press T to toggle the tilt assist off.
            band = ElasticBand(anchor, length=0.8, stiffness=3000, damping=300,
                               tension_only=True, posture_kp=300.0, posture_kd=60.0,
                               release_fade=3.0)
            print(f"[h12_mujoco] MANUAL harness on torso: anchor {anchor.round(3)} "
                  "(held upright: joint-hold + tilt assist ON until controller; T toggles assist)")
            band.print_instructions()
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
    sim_interface = SimInterface(model, data, lock=sim_lock, resolver=resolver,
                                 truth_sportstate=truth_sportstate)  # noqa: F841

    init_ros()

    # Sensor bridge: /clock, RGBD cameras (head + both hands), livox lidar + IMU.
    # Pass robosuite-prefixed MuJoCo lookup names; ROS-facing frame_ids stay
    # unprefixed (ctor defaults). cameras: (mujoco name, /realsense/<ns>, frame_id).
    # Head rides the robot prefix; the eye-in-hand gripper cameras ride the gripper
    # prefixes (same as the hand bridges below).
    # --no-sensors keeps the bridge (it owns /clock, which every use_sim_time
    # node needs) but drops the RGBD renders + lidar ray cast: those hold the
    # sim lock 50-60 ms per frame, stalling rt/lowstate past the controller's
    # stale watchdog and dragging RTF far below 1x -- pointless when nothing
    # consumes the topics (lower-body-only bringup, use_skills:=false).
    ros_bridge = RosSensorBridge(
        model, data,
        cameras=[] if no_sensors else [
            (f"{pfx}head_cam",          "head",       "camera_color_optical_frame"),
            ("gripper0_left_hand_cam",  "left_hand",  "left_hand_camera_color_optical_frame"),
            ("gripper0_right_hand_cam", "right_hand", "right_hand_camera_color_optical_frame"),
        ],
        cam_width=256, cam_height=256,   # all 3 cameras render at 256x256 (RoboCasa default)
        cam_rate_hz=0.0 if no_sensors else 15.0,
        # MID-360 fidelity: 360x56 @ 10Hz ~= 201k pts/s (real ~200k), 0.1m near /
        # 40m far range, per-point offset_time for FAST-LIO deskew. el_rays/rate
        # are the knobs to dial back if the ray cast (main thread) hurts RTF.
        lidar_az_rays=360, lidar_el_rays=56,
        lidar_rate_hz=0.0 if no_sensors else 10.0,
        lidar_max_range=40.0, lidar_min_range=0.1,
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
    # Slack-triggered band auto-release (the twin harness "handoff-timing"
    # doctrine, translated): a robot that is carrying itself leaves the band
    # SLACK, so release when |band force| stays under band_release_force for
    # band_release_sustain consecutive SIM seconds (after band_release_min_time,
    # so the controller's bring-up owns the early phase). A wall/sim TIMER
    # release is dishonest under support -- it fires mid-sway and dumps the
    # robot from whatever posture the tether wound it into.
    _band_slack_t0 = None
    _band_log_t = -1e9
    _t_cmd0 = None   # sim time of controller engagement (harness gate anchor)
    _hold_q0 = None  # latched spawn pose for the pre-engagement joint hold
    _holdout_done = False   # FIX A (2026-07-04): stiff hands persist past engagement
    _hands_fade_t0 = None   # start of the one-way xy-pin gain fade (post-holdout)
    _payout_last_t = None   # sim time of the last payout CHUNK (settle pacing)
    _payout_quiet_t0 = None  # quiet-sustain anchor before a chunk may fire
    _auto_rel_t0 = None      # manual --auto-release: sim time pay-out started
    _auto_rel_kp0 = None     # posture (kp, kd) captured at auto-release start
    _auto_rel_len0 = None    # rope rest length at auto-release start
    _auto_rel_lent = None    # rope target rest length (slack + catch margin)
    # VIEWER RENDER THROTTLE (2026-07-06): handle.sync() runs mjv_updateScene over
    # the whole RoboCasa kitchen (nq~151, ~9 ms) and, called every physics step,
    # bottlenecks the loop to ~67 Hz / RTF 0.33 — vs 170 Hz / RTF 0.85 headless.
    # That 0.33 RTF is a fidelity handicap: the deploy stack is validated at RTF~1
    # (twin/real run real-time) and the controller couples to the plant over async
    # DDS, so a slow, choppy sim is NOT a faithful test. Cap the on-screen refresh
    # at _viewer_max_hz wall-Hz so physics + /clock + rt/lowstate keep stepping
    # every iteration while only the render is decimated — recovers most of the RTF
    # and keeps the manual crouch/release workflow watchable. Headless is still the
    # most faithful test (RTF 0.85); this makes HEAD a much smaller handicap.
    _viewer_max_hz = 30.0
    _last_render = 0.0
    try:
        while True:
            start_time = time.time()
            if viewer and not handle.is_running():
                break
            with sim_lock:
                if band is not None:
                    # Re-write every step; MuJoCo persists xfrc_applied, so we must
                    # zero it when the band is disabled or the last force lingers.
                    if band.enabled:
                        _scale = band.fade_scale(data.time)   # 1.0 unless a fade is running
                        _f = band.evaluate_force(
                            data.xpos[band_body_id], data.qvel[base_dof:base_dof + 3])
                        data.xfrc_applied[band_body_id, :3] = _scale * _f
                        if not harness:
                            # MANUAL viewer mode: apply the steadying-hands tilt
                            # assist (T toggles band.posture_kp/kd) that the
                            # automated path owns, and surface a SIGNAL for when
                            # the MJPC controller takes over -- RoboCasa is too
                            # slow/choppy to eyeball the arm-jitter cue you use on
                            # the twin, so read the state instead of watching it.
                            if band.posture_kp > 0.0 or band.posture_kd > 0.0:
                                _Rt = data.xmat[band_body_id].reshape(3, 3)
                                _v6 = np.zeros(6)
                                mujoco.mj_objectVelocity(
                                    model, data, mujoco.mjtObj.mjOBJ_BODY,
                                    band_body_id, _v6, 0)
                                _tqm = (band.posture_kp
                                        * np.cross(_Rt[:, 2], np.array([0.0, 0.0, 1.0]))
                                        - band.posture_kd * _v6[:3])
                                data.xfrc_applied[band_body_id, 3:6] = _scale * _tqm
                            # Pre-engagement JOINT HOLD: pin the body motors at
                            # their spawn pose until the controller's first lowcmd
                            # (then the DDS bridge owns ctrl and the node's warmup
                            # hold-at-measured-q replaces this seamlessly). Without
                            # it the limp robot folds forward on the point strap
                            # (the strap holds height, not pose) -> the ~40deg sag
                            # you saw. Mirrors the automated harness hold + the
                            # twin's pose freeze; motors-alive PD keeps the
                            # accelerometer live.
                            if _t_cmd0 is None:
                                if _hold_q0 is None:
                                    _hold_q0 = np.array(data.qpos[resolver.motor_qpos])
                                data.ctrl[resolver.motor_ctrl] = (
                                    harness_hold_kp
                                    * (_hold_q0 - data.qpos[resolver.motor_qpos])
                                    - harness_hold_kd * data.qvel[resolver.motor_qvel])
                            # engagement latch: a fresh STIFF rt/lowcmd (kp>1) =
                            # controller actually DRIVING. Gate on kp, NOT just a
                            # fresh cmd — the safety layer idles kp=0 zeros, which
                            # would drop the joint hold before the controller
                            # takes over (the "everything slacks then almost
                            # falls" gap). Twin-parity (its stiff["kp"]>1 gate).
                            if (_t_cmd0 is None and data.time > 1.0
                                    and time.time() - sim_interface.last_cmd_time < 0.5
                                    and getattr(sim_interface, "last_cmd_kp", 0.0) > 1.0):
                                _t_cmd0 = data.time
                                print("\n" + "=" * 66
                                      + f"\n  ★ CONTROLLER ENGAGED at sim t={_t_cmd0:.1f}s "
                                        "— MJPC is now driving the legs."
                                        "\n    When tilt is small & steady and the strap reads "
                                        "SLACK,\n    press . to pay out, then SPACE to release.\n"
                                      + "=" * 66 + "\n", flush=True)
                            # AUTO HONEST-RELEASE (--auto-release): the manual
                            # gesture can't produce a clean slack-strap test --
                            # by the time you eyeball 'green' the knee has already
                            # over-extended into the strut (the fixed-anchor strap
                            # unloads as it rises and the MPC chases it straight).
                            # So once the controller is DRIVING, wait a short delay
                            # then pay the rope out to slack + fade the tilt assist
                            # to zero over a fade -> the legs take their own weight
                            # EARLY, before that feedback can build. A ~35cm catch
                            # margin remains (target length = disp-at-start + 0.35):
                            # a genuine free-stand goes green and holds; one that
                            # can't sags onto the catch. Hands-free, repeatable.
                            if (auto_release and _t_cmd0 is not None
                                    and data.time - _t_cmd0 >= auto_release_delay):
                                if _auto_rel_t0 is None:
                                    _auto_rel_t0 = data.time
                                    _auto_rel_kp0 = (band.posture_kp, band.posture_kd)
                                    _auto_rel_len0 = band.length
                                    _auto_rel_lent = float(np.linalg.norm(
                                        data.xpos[band_body_id] - band.point)) + 0.35
                                    print(f"[h12_mujoco] AUTO-RELEASE at sim "
                                          f"t={data.time:.1f}s: paying rope to slack "
                                          f"+ fading tilt assist over "
                                          f"{auto_release_fade:.1f}s (legs take load "
                                          f"now; ~35cm catch remains)", flush=True)
                                _ar = min(1.0, (data.time - _auto_rel_t0)
                                          / max(1e-6, auto_release_fade))
                                band.length = (_auto_rel_len0
                                               + _ar * (_auto_rel_lent - _auto_rel_len0))
                                band.posture_kp = (1.0 - _ar) * _auto_rel_kp0[0]
                                band.posture_kd = (1.0 - _ar) * _auto_rel_kp0[1]
                            if data.time - _band_log_t >= 1.0:
                                _band_log_t = data.time
                                _Rt = data.xmat[band_body_id].reshape(3, 3)
                                _tilt = float(np.degrees(np.arccos(
                                    np.clip(_Rt[2, 2], -1.0, 1.0))))
                                _zt = float(data.xpos[band_body_id][2])
                                _disp = float(np.linalg.norm(
                                    data.xpos[band_body_id] - band.point))
                                _slack = _disp <= band.length
                                _tn = 0.0 if _slack else float(np.linalg.norm(_f))
                                _eng = "LIVE " if _t_cmd0 is not None else "waiting"
                                _asst = ("assist %.0fNm" % float(np.linalg.norm(
                                    data.xfrc_applied[band_body_id, 3:6]))
                                    if (band.posture_kp > 0.0 or band.posture_kd > 0.0)
                                    else "assist off")
                                # TRUTH-vs-ESTIMATE diagnostic (2026-07-06): the plant's
                                # ACTUAL torso world xy. Compare its DRIFT to the
                                # estimator terminal's 'xy=[..]' (leg-odometry estimate).
                                # If both drift ~+0.2 laterally -> REAL foot slip (contact
                                # work). If true xy stays put but the estimate drifts ->
                                # PHANTOM: the RW-EKF leg-odometry is inventing lateral
                                # translation from RoboCasa foot micro-motion and the
                                # controller chases a ghost (plant-input, not the
                                # controller/estimator logic which is clean on twin/real).
                                _txy = data.xpos[band_body_id][:2]
                                print(f"[manual] t={data.time:5.1f}s  ctrl={_eng}  "
                                      f"tilt={_tilt:4.1f}°  z={_zt:.2f}  "
                                      f"trueXY=[{_txy[0]:+.2f},{_txy[1]:+.2f}]  "
                                      f"strap={'SLACK(green)' if _slack else 'TAUT %.0fN' % _tn}  "
                                      f"{_asst}", flush=True)
                        if harness:
                            # Steadying hands (the twin harness posture assist): a
                            # point strap cannot constrain TILT, so also apply a
                            # world-frame upright spring + angular damping on the
                            # torso — the part of a real gantry bring-up the
                            # operator's hands do. Gate + log use the RAW (pre-
                            # fade) magnitudes: they measure what the robot still
                            # NEEDS, not what the fade happens to grant.
                            # Engagement latch first (it switches the hands' law):
                            # first rt/lowcmd stream = controller alive. min-time
                            # counts from HERE, not sim start — the ROS container's
                            # build/launch wall time is variable, and an absolute
                            # gate could open during the scripted HOLD phase
                            # (quiet + slack but pre-policy, exactly the dishonest
                            # early release the doctrine forbids).
                            if (_t_cmd0 is None and data.time > 1.0
                                    and time.time() - sim_interface.last_cmd_time < 0.5):
                                _t_cmd0 = data.time
                                print(f"[h12_mujoco] harness: controller engaged at "
                                      f"sim t={_t_cmd0:.1f}s — stiff hands HELD "
                                      f"through bring-up (holdout "
                                      f"{harness_holdout:.0f}s + {harness_hands_fade:.0f}s"
                                      f" grip fade); release gate opens at "
                                      f"t={_t_cmd0 + band_release_min_time:.1f}s",
                                      flush=True)
                            _R = data.xmat[band_body_id].reshape(3, 3)
                            _vel6 = np.zeros(6)   # [angular(3); linear(3)], world frame
                            mujoco.mj_objectVelocity(
                                model, data, mujoco.mjtObj.mjOBJ_BODY,
                                band_body_id, _vel6, 0)
                            # BRING-UP GRIP BOOST (take H4 lesson): the xy pin
                            # holds POSITION but not TILT, and the operating
                            # posture gain (300 Nm/rad) is weaker than gravity's
                            # tipping stiffness (~mgh 400+) — the robot still
                            # drifted to 13-14 deg tilt DURING the held bring-up
                            # and toppled when the grip faded. The real operator
                            # grips the shoulders HARD (rotationally stiff)
                            # through bring-up and lightens after. Boost the
                            # posture gains during pre-handoff, fading back to
                            # the operating gains with the same grip fade — the
                            # honest gate only arms after handoff, when the gain
                            # is back at 1x, so its thresholds are unchanged.
                            _grip = 1.0
                            if _t_cmd0 is None or not _holdout_done:
                                _gp = 1.0
                                if _hands_fade_t0 is not None:
                                    _gp = max(0.0, 1.0
                                              - (data.time - _hands_fade_t0)
                                              / harness_hands_fade)
                                _grip = 1.0 + (harness_grip_boost - 1.0) * _gp
                            _tq = (_grip * band.posture_kp
                                   * np.cross(_R[:, 2], np.array([0.0, 0.0, 1.0]))
                                   - min(_grip, 2.0) * band.posture_kd * _vel6[:3])
                            # XY hands: the strap can't stop a pendulum swing, and
                            # take-1 proved a limp hang drifts ~30 cm off the feet
                            # into a taut-forever geometry the policy cannot undo.
                            # The real operator holds the robot IN PLACE over its
                            # feet, not just upright.
                            # FIX A (2026-07-04 root cause, P1): the hands used to
                            # drop to damping-only at the FIRST lowcmd — but that is
                            # the START of the node's scripted bring-up; full policy
                            # authority only arrives ~11 s later (warmup 1 + ramp <=5
                            # + hold 3 + blend 4.5, deploy_common.h). In between the
                            # robot is a PD statue deploy_common itself documents as
                            # unsurvivable unsupported: it tipped ~1 deg/s (posture
                            # spring 300 < mgh ~400 Nm/rad; damping hands are blind
                            # to slow drift) and the policy inherited an 8-17 deg
                            # lean at blend = the 1-in-5 coin flip (takes FB/G5/G6).
                            # The twin's winch and the real operator's hands both
                            # hold through this exact window. So: keep the STIFF xy
                            # spring for --harness-holdout plant-seconds, then FADE
                            # its gain to zero over --harness-hands-fade (one-way,
                            # never re-stiffen). A quiet-gated binary handoff was
                            # tried first (take H1): with the full pin held, pin and
                            # policy settle into a bounded 45-110 Nm limit cycle
                            # that never quiets — the operator's real move is to
                            # LIGHTEN the grip gradually, not to hold rigid until
                            # the robot is perfectly still under the grip. The
                            # honest release machinery downstream (strap slack +
                            # hands + assist thresholds, sustained) is unchanged
                            # and remains the actual release arbiter.
                            if _t_cmd0 is not None and not _holdout_done:
                                if (_hands_fade_t0 is None
                                        and data.time - _t_cmd0 >= harness_holdout):
                                    _hands_fade_t0 = data.time
                                    print(f"[h12_mujoco] harness: holdout done at sim "
                                          f"t={data.time:.1f}s — fading xy-pin gain "
                                          f"to zero over {harness_hands_fade:.0f}s "
                                          f"(operator lightening grip)", flush=True)
                                if (_hands_fade_t0 is not None
                                        and data.time - _hands_fade_t0
                                            >= harness_hands_fade):
                                    _holdout_done = True
                                    print(f"[h12_mujoco] harness: HANDS OFF at sim "
                                          f"t={data.time:.1f}s — xy hands are "
                                          f"damping-only touch; payout + honest "
                                          f"gate now armed", flush=True)
                            if _t_cmd0 is None or not _holdout_done:
                                _pin = 1.0
                                if _hands_fade_t0 is not None:
                                    _pin = max(0.0, 1.0 - (data.time - _hands_fade_t0)
                                               / harness_hands_fade)
                                _fxy = (_pin * harness_xy_kp
                                        * (band.point[:2] - data.xpos[band_body_id][:2])
                                        - harness_xy_kd * _vel6[3:5])
                                # Pre-engagement JOINT HOLD: gentle PD pinning all
                                # body motors at their spawn pose until the first
                                # lowcmd ONLY (then the controller owns ctrl and
                                # its warmup hold-at-measured-q replaces this
                                # seamlessly; the xy hands above persist through
                                # the holdout, the joint hold must not — it would
                                # overwrite the controller's ctrl). Without it the
                                # legs are jello — takes 1-2: the feet skated
                                # forward under the torso pin and the robot set
                                # into a backward plank (joints ~at stand targets,
                                # so the ramp was a no-op and the strap stayed
                                # taut forever). The twin freezes the pose until
                                # the node's first stiff command for the same
                                # reason; motors-alive PD is the honest version
                                # (physics + accelerometer stay live) and matches
                                # the real robot, which is never limp in the
                                # gantry.
                                if _t_cmd0 is None:
                                    if _hold_q0 is None:
                                        _hold_q0 = np.array(
                                            data.qpos[resolver.motor_qpos])
                                    data.ctrl[resolver.motor_ctrl] = (
                                        harness_hold_kp
                                        * (_hold_q0 - data.qpos[resolver.motor_qpos])
                                        - harness_hold_kd
                                        * data.qvel[resolver.motor_qvel])
                            else:
                                _fxy = -harness_xy_kd * _vel6[3:5]
                            data.xfrc_applied[band_body_id, 0] = _scale * (_f[0] + _fxy[0])
                            data.xfrc_applied[band_body_id, 1] = _scale * (_f[1] + _fxy[1])
                            data.xfrc_applied[band_body_id, 2] = _scale * _f[2]
                            data.xfrc_applied[band_body_id, 3:] = _scale * _tq
                            _tension = float(np.linalg.norm(_f))
                            _assist = float(np.linalg.norm(_tq))
                            _hands_f = float(np.linalg.norm(_fxy))
                            if data.time - _band_log_t >= 1.0:
                                _band_log_t = data.time
                                print(f"[h12_mujoco] harness: t={data.time:.1f}s "
                                      f"tension={_tension:.0f}N assist={_assist:.1f}Nm "
                                      f"hands={_hands_f:.0f}N scale={_scale:.2f} "
                                      f"rope={band.length:.2f} "
                                      f"z={data.xpos[band_body_id][2]:.2f}", flush=True)
                            # Honest handoff signal (twin doctrine, /mjpc-fk-analyze
                            # handoff-timing): strap SLACK and residual hands torque
                            # under threshold, sustained — only a robot that is
                            # already carrying and centering itself looks like this.
                            # AUTO-PAYOUT, CHUNKED (2026-07-04, takes H2/L1/L2/L4
                            # lesson): the operator's '.' pays 10 cm AT ONCE, then
                            # waits and watches — the sudden support drop FORCES the
                            # legs to catch the load, with the posture assist and
                            # the (longer) rope still armed as a catch. The first
                            # auto-payout was a continuous 2 cm/s ramp — the one
                            # thing the operator never does: it hands the policy a
                            # gently descending equilibrium to surf, and every take
                            # rode it down to the cap. So: pay a discrete chunk,
                            # only while quiet (sustained), then hold for a settle
                            # period before the next press. A robot that CAN carry
                            # itself goes slack after a few chunks -> the honest
                            # gate below fires; one that can't sags onto the longer
                            # catch (capped) and never falsely releases.
                            if (harness_payout_chunk > 0.0
                                    and not band.release_requested
                                    and _t_cmd0 is not None
                                    and _holdout_done
                                    and data.time - _t_cmd0 >= band_release_min_time
                                    and _tension > 5.0
                                    and band.length < harness_payout_max):
                                if _assist <= harness_payout_quiet:
                                    if _payout_quiet_t0 is None:
                                        _payout_quiet_t0 = data.time
                                    elif (data.time - _payout_quiet_t0 >= 1.0
                                          and (_payout_last_t is None
                                               or data.time - _payout_last_t
                                                   >= harness_payout_settle)):
                                        band.length = min(
                                            band.length + harness_payout_chunk,
                                            harness_payout_max)
                                        _payout_last_t = data.time
                                        _payout_quiet_t0 = None
                                        print(f"[h12_mujoco] harness: PAYOUT chunk "
                                              f"+{harness_payout_chunk*100:.0f}cm at "
                                              f"sim t={data.time:.1f}s -> rope "
                                              f"{band.length:.2f} (tension was "
                                              f"{_tension:.0f}N; settling "
                                              f"{harness_payout_settle:.0f}s)",
                                              flush=True)
                                else:
                                    _payout_quiet_t0 = None
                            # (hands <= 15 N: post-engagement the xy hands are
                            # damping-only, so quiet == near-zero; 15 N tolerates
                            # residual micro-sway without passing a real lean.)
                            if (not band.release_requested
                                    and _t_cmd0 is not None
                                    and _holdout_done
                                    and data.time - _t_cmd0 >= band_release_min_time
                                    and _tension <= 5.0
                                    and _hands_f <= 15.0
                                    and _assist <= harness_release_torque):
                                if _band_slack_t0 is None:
                                    _band_slack_t0 = data.time
                                elif data.time - _band_slack_t0 >= band_release_sustain:
                                    band.request_release()
                                    print(f"[h12_mujoco] harness RELEASING at sim "
                                          f"t={data.time:.1f}s (slack + assist "
                                          f"{_assist:.1f}Nm <= "
                                          f"{harness_release_torque:.1f}Nm for "
                                          f"{band_release_sustain:.1f}s) — easing off "
                                          f"over {band.release_fade:.1f}s", flush=True)
                            elif not band.release_requested:
                                _band_slack_t0 = None
                            # BENCH PROBE (2026-07-04): FORCED release, bypassing
                            # the honest gate. The Lean strat-6 policy EXPLOITS
                            # available support (H2/H3/H5: follows the payout
                            # down instead of standing taller; takeF/F3 saw the
                            # same pre-parity) so the slack gate is structurally
                            # unreachable for it under a compliant harness. The
                            # actual parity question — CAN the policy free-stand
                            # on this plant once support vanishes, as it does on
                            # twin/real — is answered by cutting support at the
                            # assisted equilibrium and measuring survival. NOT
                            # an honest handoff; logs shout FORCED.
                            if (harness_force_release_after > 0.0
                                    and not band.release_requested
                                    and _t_cmd0 is not None
                                    and _holdout_done
                                    and data.time - _t_cmd0
                                        >= harness_force_release_after):
                                band.request_release()
                                print(f"[h12_mujoco] harness FORCED RELEASE at sim "
                                      f"t={data.time:.1f}s (bench probe, gate "
                                      f"BYPASSED; tension was {_tension:.0f}N, "
                                      f"assist {_assist:.1f}Nm) — fading off over "
                                      f"{band.release_fade:.1f}s", flush=True)
                            if not band.enabled:   # fade just completed this step
                                data.xfrc_applied[band_body_id, :] = 0.0
                                print(f"[h12_mujoco] harness FULLY RELEASED at sim "
                                      f"t={data.time:.1f}s — robot is free-standing",
                                      flush=True)
                        elif band_auto_release:
                            # Gate on SPRING DISPLACEMENT (torso-to-anchor), not
                            # total force: the damper term (1e3 N.s/m) adds ~50 N
                            # per 5 cm/s of sway, so a total-force gate punishes
                            # motion rather than measuring load-bearing. Slack ==
                            # the torso is AT the anchor. band_release_force is
                            # interpreted through the spring: d_thr = F/k.
                            _d = float(np.linalg.norm(
                                data.xpos[band_body_id] - band.point))
                            _d_thr = band_release_force / band.stiffness
                            if data.time - _band_log_t >= 1.0:
                                _band_log_t = data.time
                                _fmag = float(np.linalg.norm(_f))
                                print(f"[h12_mujoco] band: t={data.time:.1f}s "
                                      f"disp={_d*100:.1f}cm (thr {_d_thr*100:.1f}) "
                                      f"|F|={_fmag:.0f}N", flush=True)
                            if data.time >= band_release_min_time and _d <= _d_thr:
                                if _band_slack_t0 is None:
                                    _band_slack_t0 = data.time
                                elif data.time - _band_slack_t0 >= band_release_sustain:
                                    band.enabled = False
                                    data.xfrc_applied[band_body_id, :3] = 0.0
                                    print(f"[h12_mujoco] band AUTO-RELEASED at sim t={data.time:.1f}s "
                                          f"(slack: disp={_d*100:.1f}cm <= {_d_thr*100:.1f}cm "
                                          f"for {band_release_sustain:.1f}s)", flush=True)
                            else:
                                _band_slack_t0 = None
                    else:
                        data.xfrc_applied[band_body_id, :] = 0.0
                        if harness and data.time - _band_log_t >= 1.0:
                            # Post-release verdict feed: the free-stand either
                            # holds (z ~1.0, small tilt) or the fall shows here.
                            _band_log_t = data.time
                            _R22 = float(data.xmat[band_body_id].reshape(3, 3)[2, 2])
                            _tilt = math.degrees(math.acos(max(-1.0, min(1.0, _R22))))
                            print(f"[h12_mujoco] FREE: t={data.time:.1f}s "
                                  f"z={data.xpos[band_body_id][2]:.2f} "
                                  f"tilt={_tilt:.1f}deg", flush=True)
                mujoco.mj_step(model, data)
                env.update_state()                  # REQUIRED before _check_success
                ros_bridge.tick()                   # /clock + camera + lidar + imu
                done = measurement.tick()
            if done:
                print("[h12_mujoco] task success (debounced).")
            if viewer:
                if band is not None and band.enabled:
                    # GREEN once the tension-only strap is slack (torso closer to
                    # the anchor than the paid-out rope length) -- the free-stand cue.
                    _slack = (band.tension_only and float(np.linalg.norm(
                        data.xpos[band_body_id] - band.point)) < band.length)
                    _draw_band(handle, band.point, data.xpos[band_body_id], slack=_slack)
                else:
                    handle.user_scn.ngeom = 0   # clear overlay when band off
                # Throttle the expensive on-screen refresh (see _viewer_max_hz note
                # above). The band overlay is updated every step (cheap); only the
                # mjv_updateScene push is capped, so physics/clock/lowstate never
                # wait on the render.
                _now_r = time.time()
                if _now_r - _last_render >= 1.0 / _viewer_max_hz:
                    handle.sync()
                    _last_render = _now_r
            # Real-time throttle. slowmo > 1 stretches the wall-target per step so
            # the WHOLE plant (physics + /clock + rt/lowstate) runs uniformly slower
            # in wall-clock at the SAME sim-dt — deliberate slow-motion for watching.
            # The controller keeps its fixed wall ctrl_hz loop but its phase rides
            # the plant tick (deploy_common.cc: phase_t = tick*twin_dt), so it tracks
            # the slowed sim-time correctly. NOTE: this is a VIEWING/DIAGNOSTIC aid,
            # not a more-faithful test — real runs at RTF 1.0 (slowmo=1.0); slowing
            # moves AWAY from real conditions. It only makes the robot "perfect" if
            # the failure was a timing/rate artifact; a control-margin failure just
            # collapses in slow motion (which is itself the useful signal).
            time.sleep(max(0, slowmo * model.opt.timestep - (time.time() - start_time)))
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
    parser.add_argument(
        "--band-auto-release",
        action="store_true",
        help="Release the elastic band on the SLACK signal instead of only the "
             "/elastic_band/toggle service: |band force| <= --band-release-force "
             "for --band-release-sustain consecutive sim seconds (after "
             "--band-release-min-time). A self-supporting robot leaves the band "
             "slack, so this releases exactly at the honest handoff moment "
             "(same doctrine as the twin harness residual gate).")
    parser.add_argument("--band-release-force", type=float, default=30.0,
                        help="slack threshold in N (robot weight ~500N; slack ~ <30N)")
    parser.add_argument("--band-release-sustain", type=float, default=2.0,
                        help="sim seconds the slack must persist before release")
    parser.add_argument("--band-release-min-time", type=float, default=15.0,
                        help="earliest sim time (s) the auto-release may fire "
                             "(bring-up = ~13.5 plant-s to a live policy)")
    parser.add_argument(
        "--harness",
        action="store_true",
        help="Replace the hang band with the twin-validated gantry HARNESS "
             "handoff: tension-only strap anchored above the torso (catch-only, "
             "exerts nothing once the robot stands) + steadying-hands posture "
             "assist (upright spring + angular damping on the torso), released "
             "by a quadratic fade once the honest slack signal fires — strap "
             "slack AND assist <= --harness-release-torque for "
             "--band-release-sustain sim-s (after --band-release-min-time). "
             "This is the REAL bring-up procedure: hold straight while the "
             "controller stands the robot up, release once it carries itself.")
    parser.add_argument("--harness-kp", type=float, default=300.0,
                        help="steadying-hands upright spring (N*m/rad); 300/60 "
                             "are the twin's proven-stable gains")
    parser.add_argument("--harness-kd", type=float, default=60.0,
                        help="steadying-hands angular damping (N*m*s/rad)")
    parser.add_argument("--harness-xy-kp", type=float, default=1000.0,
                        help="hands horizontal hold spring (N/m) BEFORE the "
                             "controller engages (holds the limp robot over its "
                             "feet); drops to damping-only after engagement")
    parser.add_argument("--harness-xy-kd", type=float, default=200.0,
                        help="hands horizontal damping (N*s/m; the post-"
                             "engagement light touch)")
    parser.add_argument("--harness-hold-kp", type=float, default=80.0,
                        help="pre-engagement joint-hold PD kp (per motor): pins "
                             "the spawn pose until the first lowcmd so the limp "
                             "robot can't plank/skate; controller owns ctrl after")
    parser.add_argument("--harness-hold-kd", type=float, default=8.0,
                        help="pre-engagement joint-hold PD kd")
    parser.add_argument("--harness-slack", type=float, default=0.05,
                        help="strap slack budget below the torso spawn (m): the "
                             "strap only catches a sag/fall deeper than this")
    parser.add_argument("--harness-fade", type=float, default=3.0,
                        help="release fade duration (sim s, quadratic ease-out; "
                             "an instant cut is a step disturbance)")
    parser.add_argument("--harness-release-torque", type=float, default=15.0,
                        help="residual hands torque (N*m) under which the robot "
                             "counts as self-centering (twin handoff doctrine)")
    parser.add_argument("--harness-payout-rate", type=float, default=0.02,
                        help="auto rope payout (m per sim-s) once the release "
                             "window is open and the strap still carries load -- "
                             "the automated '.'-payout: hand the load to the legs "
                             "quasi-statically at the policy's own stance height. "
                             "0 disables (strap length fixed).")
    parser.add_argument("--harness-payout-max", type=float, default=1.5,
                        help="absolute rope-length cap for auto payout (a hanging "
                             "robot lowers to this and never passes the gate)")
    parser.add_argument("--harness-payout-quiet", type=float, default=20.0,
                        help="pause payout while |hands torque| exceeds this (N*m) "
                             "-- never pay rope into a wobble")
    parser.add_argument("--harness-payout-chunk", type=float, default=0.10,
                        help="payout CHUNK size (m) -- the operator's '.' press: "
                             "a sudden support drop that FORCES the legs to catch, "
                             "with posture assist + the longer rope still armed. "
                             "0 disables payout. (The old continuous 2 cm/s ramp "
                             "let the policy surf a descending equilibrium to the "
                             "cap -- takes H2/L1/L2/L4.)")
    parser.add_argument("--harness-payout-settle", type=float, default=6.0,
                        help="wait this many plant seconds after each payout chunk "
                             "before the next (operator watches it settle)")
    parser.add_argument("--harness-holdout", type=float, default=15.0,
                        help="keep the STIFF xy hands for this many PLANT seconds "
                             "after the first lowcmd (>= the node's scripted "
                             "bring-up: warmup 1 + ramp <=5 + hold 3 + blend 4.5 "
                             "~= 13.5s), then hand off to damping-only once the "
                             "posture is also quiet for 1s. The real operator / "
                             "twin winch hold through exactly this window; "
                             "releasing at first lowcmd left a PD statue tipping "
                             "1 deg/s into the policy blend (the 1-in-5 stands).")
    parser.add_argument("--harness-hands-fade", type=float, default=10.0,
                        help="after the holdout, fade the xy-pin GAIN to zero over "
                             "this many plant seconds (operator lightening grip). "
                             "A binary quiet-gated handoff deadlocks: the full pin "
                             "and the policy settle into a 45-110 Nm limit cycle "
                             "that never quiets (take H1).")
    parser.add_argument("--harness-grip-boost", type=float, default=5.0,
                        help="posture-gain multiplier during pre-handoff (fades "
                             "back to 1x with the grip fade). 300 Nm/rad < mgh "
                             "~400: without the boost the held robot still tilt-"
                             "drifts to 13-14 deg during bring-up and topples at "
                             "the fade (take H4). 5x = 1500 Nm/rad, a firm "
                             "operator grip on the shoulders.")
    parser.add_argument("--harness-force-release-after", type=float, default=0.0,
                        help="BENCH PROBE: force the band release this many plant "
                             "seconds after engagement (0 = off), BYPASSING the "
                             "honest gate. For measuring free-stand ABILITY when "
                             "the policy exploits support and the slack gate is "
                             "unreachable (Lean strat-6 follows payout down).")
    parser.add_argument(
        "--soft-feet", dest="rigid_feet", action="store_false",
        help="Disable the surgical foot<->floor contact hardening (leaves the "
             "stance feet on robosuite's global impratio=20/elliptic soft cone, "
             "which lets them creep laterally -> the strat-6 stand folds). Default "
             "is rigid feet ON: foot geoms get priority/condim4/high-friction/stiff "
             "solref so the foot-ground pair matches the planner's no-slip belief "
             "without touching the grasp-tuned global contact model.")
    parser.set_defaults(rigid_feet=True)
    parser.add_argument("--sim-dt", type=float, default=0.0,
                        help="override the plant timestep (0 = keep robosuite's "
                             "assembled 0.002). Pass 0.005 for the balance bench: "
                             "the twin runs 0.005 and the deploy-chain stand is "
                             "dt-sensitive (twin@0.002 HANGS, twin@0.005 STANDS "
                             "even at RTF 0.5 -- T2 vs T3b). Node twin_dt and "
                             "estimator tick_dt must match. Also raises RTF 2.5x.")
    parser.add_argument("--foot-solref", type=float, default=0.008,
                        help="rigid-feet contact timeconstant (s). 0.008 = the "
                             "original anti-sink hammer; 0.02 = the twin's "
                             "default foot softness (under pyramid impratio-100 "
                             "the normal is ~50x stiffer than this nominal).")
    parser.add_argument(
        "--keep-euler", dest="sim_implicitfast", action="store_false",
        help="Keep robosuite's default Euler integrator instead of matching the "
             "rest of the stack's implicitfast (twin/planner/real all implicitfast; "
             "the task XMLs warn Euler diverges on biped balance). Default overrides "
             "to implicitfast -- ODE time-stepping only, not a contact/grasp param.")
    parser.set_defaults(sim_implicitfast=True)
    parser.add_argument(
        "--twin-contact", dest="twin_contact", action="store_true",
        help="Match the twin/planner GLOBAL contact model: impratio 20->100 + cone "
             "elliptic->pyramidal (stiff near-no-slip friction; impratio/cone have NO "
             "per-geom form so the rigid-feet fix can't reach them). DEFAULT OFF -- "
             "manipulation keeps robosuite's grasp-tuned 20/elliptic; the balance-only "
             "bringup (use_skills:=false) opts in for a clean stand.")
    parser.add_argument(
        "--no-multiccd", dest="multiccd", action="store_false",
        help="Disable the multiCCD contact-manifold override. Default ON: the "
             "kitchen floor is BOX geoms and box-mesh convex collision returns "
             "ONE contact point per foot at mujoco 3.3.1 (no ankle moments, "
             "migrating contact = 'sliding or wobbling'); multiccd yields 4-5 "
             "points/foot, matching the class of the twin's 3-point plane-mesh "
             "contact. Affects only coincident-surface contacts; grasp point "
             "contacts are unchanged, but use this flag for a pure-robosuite A/B.")
    parser.set_defaults(multiccd=True)
    parser.add_argument(
        "--elliptic-contact", dest="elliptic_contact", action="store_true",
        help="THE NO-SLIP FIX (standalone; MuJoCo docs' recipe 'elliptic cones, "
             "large impratio, Newton'). Raises the plant's native ELLIPTIC cone "
             "impratio 20->100 + forces Newton, matching the planner's impratio=100 "
             "belief so the stance foot stops creeping. impratio only hardens "
             "friction on ELLIPTIC cones (inert on pyramidal per the docs), so this "
             "is the correct lever -- NOT --twin-contact (pyramidal, which just "
             "over-stiffens the normal and made tilt worse). DEFAULT OFF -- "
             "manipulation keeps robosuite's grasp-tuned impratio 20; the "
             "balance-only bringup (use_skills:=false) opts in for a clean stand.")
    parser.add_argument(
        "--plane-floor", dest="plane_floor", action="store_true",
        help="Convert the kitchen's thin-box walking-surface geoms "
             "(floor_*_room_g*) to infinite PLANES at the same top height -- "
             "twin/planner floor-manifold parity. The T-box twin A/B "
             "(2026-07-05) showed the box floor alone flips the deploy-chain "
             "stand from 218 s (plane) to a 2-s forward face-plant, even with "
             "multiccd's 4 contacts/foot. Static-fixture contacts are "
             "unaffected (static-static pairs are always excluded); movables "
             "rest at the same height. Bench-only; default OFF.")
    parser.add_argument(
        "--spawn-crouch", type=float, default=0.0,
        help="Spawn the robot KNEE-BENT (knee=K rad, hipP/ankP=-K/2, feet "
             "re-planted on the floor) instead of straight-legged. A straight "
             "spawn hands the policy the knee hyperextension/strut singularity "
             "when it takes over; the stand keyframe is itself bent (0.35). The "
             "joint hold freezes this bent pose until the controller rises from "
             "it (the twin's --crouch). Try 0.4-0.6. Default 0 = straight.")
    parser.add_argument(
        "--auto-release", dest="auto_release", action="store_true",
        help="MANUAL harness: once the controller engages, auto pay the rope out "
             "to slack and fade the tilt assist to zero (hands-free honest "
             "release). Loads the legs EARLY, before the fixed-anchor strap-unload "
             "feedback drives the knee into the strut -- the clean slack-strap "
             "free-stand test the hand-timed gesture can't produce.")
    parser.add_argument(
        "--auto-release-delay", type=float, default=2.0,
        help="Sim seconds after controller engagement before auto-release begins "
             "(default 2.0 -- lets the bring-up warmup settle first).")
    parser.add_argument(
        "--auto-release-fade", type=float, default=3.0,
        help="Sim seconds over which the rope pays out to slack + tilt assist "
             "fades to zero (default 3.0).")
    parser.add_argument(
        "--real-hands", dest="real_hands", action="store_true",
        help="Remove gravcomp from the hand/gripper bodies so they carry real "
             "weight (twin parity). RoboCasa gravcomps the hands weightless, but "
             "the deploy controller feed-forwards gravity for a real ~0.5kg hand "
             "-> wrist over-torque. The one robot-model delta vs the twin. "
             "Balance-only; grasp runs want the weightless default.")
    parser.add_argument(
        "--truth-sportstate", dest="truth_sportstate", action="store_true",
        help="Publish GROUND-TRUTH base position/velocity on rt/sportmodestate "
             "(IMU-site pos/linvel), exactly like the digital twin's "
             "SimInterface -- the deploy node's validated state source on sim "
             "benches. The RW-EKF estimator belongs to the REAL chain; when "
             "this is on, point the estimator's out_topic elsewhere "
             "(e.g. rt/sportmodestate_est) so the two don't fight.")
    parser.add_argument(
        "--no-sensors",
        action="store_true",
        help="Skip the RGBD camera renders + livox lidar ray cast (they hold the "
             "sim lock 50-60ms and drag RTF far below 1x). /clock, rt/lowstate, "
             "grippers and the elastic band keep working. Use for lower-body-only "
             "bringups (use_skills:=false) where nothing consumes vision topics.")
    parser.add_argument(
        "--slowmo", dest="slowmo", type=float, default=1.0,
        help="Deliberately run the WHOLE plant slower in wall-clock at the SAME "
             "sim-dt: the real-time target per step is stretched by this factor, so "
             "physics + /clock + rt/lowstate all slow together uniformly (2.0 = "
             "half speed). The controller's phase rides the plant tick so it tracks "
             "correctly -- clean slow-motion for WATCHING the crouch/release on this "
             "heavy scene. DIAGNOSTIC ONLY: real runs at 1.0; slowing moves away "
             "from real, and only 'fixes' failures that were timing artifacts (a "
             "margin failure just collapses slowly -- itself the useful signal).")
    args = parser.parse_args()

    task = args.task
    if task is None:
        task = _random_task(seed=args.seed)
        print(f"[h12_mujoco] no --task given; randomly selected {task!r}")
    else:
        task = _resolve_task(task, seed=args.seed)

    sim_loop(task, viewer=not args.headless,
             layout=args.layout, style=args.style, seed=args.seed,
             no_sensors=args.no_sensors,
             band_auto_release=args.band_auto_release,
             band_release_force=args.band_release_force,
             band_release_sustain=args.band_release_sustain,
             band_release_min_time=args.band_release_min_time,
             harness=args.harness, harness_kp=args.harness_kp,
             harness_kd=args.harness_kd,
             harness_xy_kp=args.harness_xy_kp, harness_xy_kd=args.harness_xy_kd,
             harness_hold_kp=args.harness_hold_kp,
             harness_hold_kd=args.harness_hold_kd,
             harness_slack=args.harness_slack,
             harness_fade=args.harness_fade,
             harness_release_torque=args.harness_release_torque,
             harness_payout_rate=args.harness_payout_rate,
             harness_payout_max=args.harness_payout_max,
             harness_payout_quiet=args.harness_payout_quiet,
             harness_payout_chunk=args.harness_payout_chunk,
             harness_payout_settle=args.harness_payout_settle,
             harness_holdout=args.harness_holdout,
             harness_hands_fade=args.harness_hands_fade,
             harness_grip_boost=args.harness_grip_boost,
             harness_force_release_after=args.harness_force_release_after,
             rigid_feet=args.rigid_feet, foot_solref=args.foot_solref,
             sim_dt=args.sim_dt,
             sim_implicitfast=args.sim_implicitfast,
             twin_contact=args.twin_contact,
             elliptic_contact=args.elliptic_contact, multiccd=args.multiccd,
             truth_sportstate=args.truth_sportstate,
             plane_floor=args.plane_floor,
             spawn_crouch=args.spawn_crouch,
             auto_release=args.auto_release,
             auto_release_delay=args.auto_release_delay,
             auto_release_fade=args.auto_release_fade,
             real_hands=args.real_hands)
