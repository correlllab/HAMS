# Audit: `h1_robocasa` simulator vs `correlllab/h1_mujoco@mjpc_twin` (the "twin")
https://github.com/correlllab/h1_mujoco/tree/mjpc_twin

**Purpose.** Enumerate every way the local RoboCasa-embedded H1‑2 simulator
(`HAMS/h1_robocasa/`) diverges from the known‑good standalone MuJoCo "twin"
(`h1_mujoco`, branch `mjpc_twin`), so a downstream debugging agent can explain
why the *same* shared MJPC lower‑body controller "stands for a couple seconds
then falls" on the RoboCasa plant while it stands on the twin.

**Scope.** This is a **plant/sim** audit (sim‑vs‑sim). The MJPC controller
(`mujoco_mpc` + the `core_ws` deploy node + safety layer) is *shared* by both
sims and is out of scope — except where the sim's **state‑publishing** feeds it
different inputs (see §6). Both sims speak the same Unitree DDS protocol
(`rt/lowcmd` → 27 motors, `rt/lowstate`, `rt/sportmodestate`).

> Method note: findings are from static reading of both codebases + the twin's
> XML/README/deps fetched from GitHub raw. Claims sourced from the local code's
> own comments are labelled **(dev‑asserted)**; independently verified items are
> labelled **(verified)**. Anything needing a compiled model is flagged.

---

## 0. Bottom line — most likely balance‑breakers, ranked

1. **Default timestep mismatch (verified).** Twin plant runs **dt = 0.005**
   (hardcoded in `mujoco_env.py:23`). RoboCasa/robosuite assembles **dt = 0.002**
   and the override (`--sim-dt 0.005`) is **default OFF**. The deploy stack's
   gains/gravity‑ff/weights are tuned to the 0.005 plant; the dev notes the same
   twin *hangs* at 0.002 and *stands* at 0.005 (T2 vs T3b). → **A default
   RoboCasa launch runs the wrong plant timestep.**
2. **Ground‑truth base state is not published by default (verified).** The twin's
   `SimInterface` **always** publishes ground‑truth base pos/vel on
   `rt/sportmodestate`. The local `SimInterface` only does so with
   **`--truth-sportstate` (default OFF)**; otherwise the controller consumes the
   RW‑EKF leg‑odometry estimate, which drifts ~0.3 m under sim foot micro‑slip →
   the policy chases a phantom base velocity. **(dev‑asserted, code‑confirmed off‑by‑default.)**
3. **Contact model globals differ and parity is default OFF (verified).** Twin
   declares **`impratio=100`, cone default = pyramidal**; robosuite assembles
   **`impratio=20`, cone=elliptic**. `--elliptic-contact` matches impratio but
   keeps elliptic; `--twin-contact` matches the literal twin (pyramidal+100) but
   is a documented failed A/B; `--plane-floor` matches the floor. **All default
   OFF** → stance foot creeps laterally; box‑floor manifold vs plane. (§4)
4. **Upper‑body mass differs by ~1.68 kg (verified).** RoboCasa gravcomps the
   gripper/hand bodies **weightless**, while the twin's hands carry a real
   **0.506 kg each** — AND the twin has a **0.67 kg `back_equipment` box on the
   torso that the local model lacks entirely**. The controller feed‑forwards
   gravity for real, weighted hands → over‑torques the weightless local wrists;
   the missing back box shifts the twin's torso CoM back/heavier. `--real-hands`
   (default OFF) restores hand weight but **cannot add the back box**. (§5c)
5. **Three different MuJoCo versions across the stack (verified).** MJPC planner
   builds **3.2.3**, the twin sim runs **3.6.0**, the local RoboCasa plant pins
   **3.3.1**. Contact/solver defaults differ across these. (§5, §11)
6. **The whole robot lives in a RoboCasa kitchen (verified).** ~151 nq, dozens of
   fixtures/objects, box floor, robosuite‑assembled `<option>`, name‑renamed +
   `robot0_`‑prefixed joints — vs the twin's bare robot‑on‑a‑plane. (§2, §7)

The local `h12_mujoco.py` is *explicitly* a twin‑parity port and contains ~15
runtime patches to claw back parity — **but 5 of the most important are opt‑in
and OFF in the default launch** (see the flag matrix, §3). That gap is the first
thing to check.

---

## 1. Architecture: standalone twin vs RoboCasa‑embedded plant

| | **Twin (`h1_mujoco@mjpc_twin`)** | **Local (`h1_robocasa`)** |
|---|---|---|
| Model source | Loads scene MJCF directly: `unitree_robots/h1_2/scene_handless_magpie.xml` (`MujocoEnv(xml_path)`) | robosuite **assembles** the model: `H1_2` robot (`CL_Assets/robosuite_assets/robots/h1_2/robot.xml`) + Magpie grippers, merged into a **RoboCasa kitchen** task env |
| World | Robot + infinite plane floor (+ optional box‑floor probe) | Full kitchen: counters, appliances, objects, walls, thin **box** floor |
| Stepping | Own loop calls `mujoco.mj_step` (`MujocoEnv.sim_step`) | Own loop calls `mujoco.mj_step`; **never** `env.step()` (robosuite controllers never run) |
| Joint/actuator names | Original Unitree names (`left_knee_joint`, …) | 12 leg joints **renamed** (`left_leg_knee_joint`) + **all** entities `robot0_`‑prefixed; grippers `gripper0_{side}_` → resolved by `sim_names.NameResolver` |
| DDS domain | `ChannelFactoryInitialize(id=0)` (real‑robot domain) | `ROS_DOMAIN_ID` env, **asserted > 0** (domain 1 in `launch_robocasa.sh`); 0 reserved for real robot |
| Base placement | Fixed spawn / hang band on pelvis free joint | RoboCasa layout/seed places the pelvis free joint; `place_robot_collision_free` backs the base off until no geom penetrates a fixture |
| Extra I/O | pyvista viz, force interface, skin | ROS 2 bridges: `/clock`, RGBD ×3, Livox lidar+IMU, gripper services, RoboCasa task measurement |

**Consequence for debugging:** the two sims do **not** share a model file. The
robot subtree is claimed near‑identical (§5) but the *surrounding* `<option>`,
floor, contacts, and DOF count are assembled by robosuite/RoboCasa and only
partially patched back to twin values at runtime.

---

## 2. The correct apples‑to‑apples twin config

For the lower‑body/magpie MJPC deploy, the twin equivalent of the local sim is:

```
python h12_mujoco.py --magpie            # handless + 2×0.506 kg magpie grippers, plane floor, dt 0.005
                     [--crouch 0.7]      # bent-knee bring-up rehearsal, pose frozen till first stiff cmd
                     [--auto-harness]    # automated gantry handoff (mirrors local --harness verbatim)
```

- Twin **default** (`--handless`, no `--magpie`) has **no grippers** — do **not**
  compare against that; it lacks the ~1 kg of wrist mass the controller expects.
- The local sim **always** has the two Magpie grippers (default_gripper in
  `h1_2_robosuite.py`), so the magpie twin config is the right baseline.
- The twin's automated harness `_auto_harness_loop` (`h12_mujoco.py` ref) is a
  **verbatim port** of the local `--harness` path (same constants:
  3000/300 strap, 300/60 posture, 5× grip boost, 15 s holdout, 10 s fade,
  10 cm/6 s chunked payout, slack gate). Handoff logic is **not** a difference.

---

## 3. Runtime parity‑flag matrix — **the most actionable table**

`h12_mujoco.py` mutates the robosuite‑assembled model at startup to approach twin
parity. `launch_robocasa.sh` / `docker-compose.yml` pass **no flags** by default
(`command: launch_robocasa.sh`, args just forwarded). So the **effective default
plant** is:

| Parity dimension | Flag | Default | Default‑run value | Twin value | Gap at default? |
|---|---|---|---|---|---|
| Integrator | `--keep-euler` disables | **implicitfast ON** | implicitfast | implicitfast | ✅ matched |
| Foot contact hardening | `--soft-feet` disables | **rigid‑feet ON** | priority/condim4/µ=1/solref | (plane feet) | ~ partial |
| Contact manifold | `--no-multiccd` disables | **multiccd ON** | multi‑point box contact | 3‑pt plane | ~ partial |
| **Plant timestep** | `--sim-dt 0.005` | **0 (OFF)** | **0.002** | **0.005** | ❌ **MISMATCH** |
| **Global contact** | `--elliptic-contact` / `--twin-contact` | **OFF** | **impratio 20, elliptic** | **impratio 100, pyramidal** | ❌ **MISMATCH** |
| **Floor manifold** | `--plane-floor` | **OFF** | **box floor** | **plane** | ❌ **MISMATCH** |
| **Hand mass** | `--real-hands` | **OFF** | **gravcomp'd weightless** | **weighted** | ❌ **MISMATCH** |
| **Base state source** | `--truth-sportstate` | **OFF** | **estimator (RW‑EKF)** | **ground truth** | ❌ **MISMATCH** |
| Spawn pose | `--spawn-crouch K` | **0 (OFF)** | straight‑leg (all‑zero) | qpos0 (or `--crouch`) | context‑dependent |
| Sensors | `--no-sensors` | OFF (sensors ON) | RGBD+lidar hold sim‑lock ~50–60 ms | n/a | RTF/latency risk |

**Takeaway:** even with all the default‑ON patches, a bare `launch_robocasa.sh`
leaves **five** twin‑deltas live: **dt, global contact cone/impratio, box floor,
weightless hands, and estimator‑not‑truth base state**. Full parity requires,
e.g.:
`--sim-dt 0.005 --elliptic-contact --plane-floor --real-hands --truth-sportstate`
(plus parking the estimator on another topic when truth is on). Confirm what the
failing runs actually passed.

---

## 4. Physics / solver `<option>` deltas

All values below are what the **local runtime applies** vs **what the twin XML
declares**. The local model's `<option>` is owned by robosuite assembly and then
mutated in `sim_loop()`.

| Option | Twin | Local default run | Local with full flags | Notes |
|---|---|---|---|---|
| `timestep` | **0.005** (`mujoco_env.py:23`) | **0.002** (robosuite) | 0.005 (`--sim-dt`) | External chain (`node twin_dt`, estimator `tick_dt`, `lowstate tick=int(t/dt)`) assumes one value; a mismatch silently drifts the plant clock 2.5×. `h12_mujoco.py:319` prints a MISMATCH warning. |
| `integrator` | implicitfast | robosuite default **Euler** → patched to **implicitfast** | implicitfast | `h12_mujoco.py:228`. Euler integrates damping/Coriolis explicitly → response the planner's implicitfast rollouts don't predict. |
| `cone` | **pyramidal** (unset in XML → MuJoCo default) | **elliptic** (robosuite native) | elliptic (`--elliptic-contact`) or pyramidal (`--twin-contact`) | **verified.** Note the twin literally runs *pyramidal*, so its `impratio=100` is largely inert (docs: impratio hardens friction only on elliptic). `--elliptic-contact` deliberately deviates from the literal twin to get real anti‑slip. |
| `impratio` | **100** (declared, `h1_2_handless_magpie.xml:9`) | **20** (robosuite grasp‑tuned) | 100 (`--elliptic-contact`/`--twin-contact`) | **verified.** `--twin-contact` (pyramidal+100) reproduces the twin exactly but is a documented **failed** balance A/B (over‑stiffens the normal). |
| `solver` | Newton (MuJoCo default) | robosuite default | forced Newton by `--elliptic-contact` | high‑impratio elliptic needs Newton. |
| `multiccd` flag | n/a (plane collider gives 3 pts/foot natively) | **ON** (patched; off at 3.3.1 default) | ON | box‑mesh narrowphase yields **1 contact/foot** at MuJoCo 3.3.1 without it → no ankle moment. |
| Floor geom | **plane** | **box** (`floor_*_room_g*`, 0.02 half‑thickness) | plane (`--plane-floor` converts in place) | **dev‑asserted key delta:** same thin box under the twin flipped its 218 s stand into a 2 s face‑plant even with multiccd. |
| Foot geoms | plane‑side default (µ, solref default ~0.02) | patched: `priority=2, condim=4, friction=[1,0.06,0.01], solref=[0.008,1]` | same | `h12_mujoco.py:151`. Surgical, per‑geom (can't reach global impratio/cone). `--foot-solref` tunable. |

---

## 5. Robot‑model deltas (mass, actuators, joints, gravcomp)

*Verified by static parse of both MJCF trees (local `CL_Assets` robot.xml +
`build_assets.py` + magpie gripper XMLs vs the twin's
`scene_handless_magpie.xml → h1_2_handless_magpie.xml` chain fetched from
GitHub). NB: the twin repo's `h1_2.xml` is a **different**, full‑dexterous‑hand
model and is NOT what the magpie scene loads — do not diff against it.*

**5a. Actuators — IDENTICAL (verified).** Both sides have 27 `<motor>` (torque)
actuators, **no `gear`** (→ gear 1), **no `ctrlrange`**, **no `kp/kv`**, **no
motor `forcerange`**. Force is clamped only by each joint's `actuatorfrcrange`,
which is **identical on all 27**: hip_yaw/pitch/roll ±200, knee ±300,
ankle_pitch ±60, ankle_roll ±40, torso ±200, shoulder_pitch/roll ±40,
shoulder_yaw/elbow ±18, wrist_roll/pitch/yaw ±19. Order matches
`ROS_MOTOR_ORDER` element‑for‑element on both. **→ The torque law and its limits
are the same; actuators are not the bug.**

**5b. Joints — IDENTICAL where it matters (verified).** damping/armature/
frictionloss = **10 / 0.1 / 0.2** for all 27 body joints on both (local inlines
per‑joint; twin uses a `<default>`). Free base joint frictionloss = 0 on both.
All 12 leg + torso ranges identical. **Only arm delta:** `left_shoulder_roll`
range is local `-0.19..3.4` vs twin `-0.38..3.4` — not balance‑relevant.

**5c. Mass / gravcomp — the largest model delta (verified).** All shared link
inertials (pelvis→wrist) are byte‑identical. The differences are all in the
upper‑body payload:

| Item | Twin | Local | Effect |
|---|---|---|---|
| Per‑hand gripper mass | **0.506 kg** lumped on the mount geom, **no gravcomp → real weight**, CoM ~6.7 cm past the wrist | Magpie meshes (ρ=1000), **`gravcomp=1` on every gripper body → weightless** | Controller feed‑forwards gravity for real ~0.5 kg hands → over‑torques the weightless local wrists |
| `back_equipment` torso box | **0.67 kg** box (0.05×0.26×0.28 m) at torso `-0.095 0 0.18`, `contype=0` | **absent entirely** | Twin's torso CoM sits ~1 cm back / heavier; local torso is lighter and CoM‑forward |
| `gravcomp` anywhere | **none** (grippers + back box fully gravity‑loaded) | on `left_hand`, `right_hand`, and **all** gripper bodies | — |
| Static total mass | **≈ 68.67 kg** | not statically computable (mesh‑density grippers, gravcomp'd) — read the runtime `body_mass` probe (`h12_mujoco.py:328`) | Twin carries ~1.68 kg of upper‑body weight local either cancels (hands) or lacks (back box) |

`--real-hands` (default OFF) zeroes the local hand gravcomp but **cannot add the
missing 0.67 kg back box** — that delta persists even with the flag.

**5d. Gripper structure / DOF count — different mechanism (verified).** Twin
grippers are **rigid welds: 0 extra DOF (nq=34, nv=33)**. Local merges the
**full articulated Magpie**: `finger_joint1/2` + `hinge_2/3` per side = **12
gripper DOF**, each closed by two `<connect>` 4‑bar **equality constraints** per
hand (all gravcomp'd). → Local adds constraint‑solver dynamics and potential
near‑hand jitter the twin does not have, and `nq/nv` differ solely from these
gripper DOF.

**5e. Collision coverage — twin collides far more of the body (verified).** Twin
gives nearly every link a collision mesh **plus knee collision cylinders**
(`size 0.04 0.1` at each knee) and collidable torso/hip/arm meshes → it can
self‑collide, cross legs, and knee‑strike the floor. Local `build_assets.py`
forces all visual (group‑1) geoms to `contype=0/conaffinity=0`, so it collides
**only** on the pelvis sphere (r 0.05), hip_pitch, the feet, and the two hand
cylinders — legs/torso/arms mostly **cannot** self‑collide or hit the floor.
This changes leg‑crossing / near‑fall behavior (not a clean‑stand differentiator
by itself). Local also `<contact>`‑excludes torso↔shoulder_roll (moot: those are
visual‑only); twin has no `<contact>` block.

**5f. Feet — static‑identical, runtime‑hardened (verified).** The foot ground
geom is the same `ankle_roll_link` **mesh** at condim 3 on both; because condim 3
uses only `friction[0]=1.0`, the raw foot friction is effectively equal (twin
declares `1 0.001 0.001`, local inherits `1 0.005 0.0001`). The real foot delta
is the default `rigid_feet` runtime hardening (§4): local → priority 2 / condim 4
/ friction `[1,0.06,0.01]` / solref `[0.008,1]` vs twin's untouched condim 3 /
solref `[0.02,1]`.

**5g. Sensors — shared 86‑block identical, local appends more (verified).** Both
have the same first 86 sensors in the same order: jointpos×27 → jointvel×27 →
jointactuatorfrc×27 → `imu_quat`(framequat) → `imu_gyro`(gyro, noise 5e‑4) →
`imu_acc`(accel, noise 1e‑2) → `frame_pos`(framepos) → `frame_vel`(framelinvel),
all on site `imu`. So the twin's **fixed‑index** reads (`sensordata[i]`,
`[i+27]`, `[i+54]`) and the local's name‑resolved reads land on the same data.
Local **appends** 3 Livox IMU sensors + per‑gripper `force_ee`/`torque_ee`/finger
sensors **after** the shared block (so they don't shift the first 86).

**Known model‑level delta (dev‑asserted, `h12_mujoco.py:343`):** a static audit
claims masses, inertials, foot mesh, and joint armature/damping/frictionloss all
match the twin "to the digit," with **one exception — gravcomp on the hands**:

- RoboCasa sets `body_gravcomp > 0` on the gripper/hand bodies → they are
  **weightless statics** (mesh‑derived mass exists but gravity is cancelled).
- The twin's hands carry real weight (~0.3–0.5 kg/hand).
- The MJPC controller feed‑forwards gravity for a **real** ~0.5 kg hand, so
  against weightless hands it **over‑torques the wrists**.
- `--real-hands` zeroes gravcomp on hand/gripper bodies (parity). **Default OFF.**
- The startup gripper‑mass probe (`h12_mujoco.py:328`) prints per‑side gripper
  mass + gravcomp body count so the residual is a measured number.

**MuJoCo version (verified):** local RoboCasa container pins **`mujoco==3.3.1`**
(`docker/BUILD.md:145`, "to match RoboCasa's hard pin"); twin runs
**`mujoco==3.6.0`** (`pyproject.toml`, `requirements.txt`). This alone changes
contact/solver defaults between the two plants.

---

## 6. Control‑interface deltas (`unitree_interface.py`)

Both `SimInterface`s implement the same Unitree torque law
`ctrl = tau + kp·(q*−q) + kd·(dq*−dq)` for 27 motors and publish `rt/lowstate` at
the plant dt. Differences that matter:

| Aspect | Twin | Local | Impact |
|---|---|---|---|
| **`rt/sportmodestate`** | **Always** published from IMU‑site framepos/framelinvel (ground truth) | **Only** with `--truth-sportstate` (default OFF); else the RW‑EKF estimator owns the topic | **HIGH** — default local feeds the controller a **drifting leg‑odometry estimate**, not truth. Dev notes ~0.3 m lateral drift → phantom capture‑point velocity. |
| State read for q/dq | `data.sensordata[i]` / `[i+27]` (jointpos/jointvel **sensors**) | `data.qpos[motor_qpos[i]]` / `data.qvel[...]` **directly** (name‑resolved) | Numerically equal for direct joint sensors; different mechanism. Local bypasses any sensor noise/filtering. |
| `tau_est` source | `sensordata[i+54]` (jointactuatorfrc sensor) | `sensordata[motor_tau[i]]` = `<jointactuatorfrc>` **clamped to actuatorfrcrange** (real "measured torque" semantics; avoids spuriously tripping the safety estop with raw PD demand) | Equivalent intent; local is explicit about clamping. |
| Motor→ctrl index | Fixed `data.ctrl[i]` (actuator order == motor order) | **Name‑resolved** via `NameResolver` (robosuite reorders/renames) | Local **must** resolve or it writes the wrong joints; `sim_names.ROS_MOTOR_ORDER` defines the 27‑order. Verify the resolver maps all 27 (+ IMU sensors) at launch. |
| Watchdog timeout | **Wall‑clock** gap always; rewrites zeros every tick while timed out | **Sim‑time** gap after first cmd (wall‑clock before); zeros **once** on detection | Local fix: sensor renders hold the sim‑lock ~1 s wall and were false‑tripping a wall watchdog (294×/bringup). A frozen world can't "miss" a command. |
| Engagement gate | (twin harness sniffs `stiff["kp"]>1` in the loop) | `last_cmd_kp` tracked in handler; harness gates joint‑hold release on **kp>1** (safety layer idles kp=0 zeros) | Same doctrine. |
| DDS domain | id=0 | id=`ROS_DOMAIN_ID`>0 | Coexistence only. |
| Inspire `ShadowInterface` | present (inspire‑hand mapping) | absent (magpie grippers handled by `magpie_hand_bridge.py`) | Local uses a different hand stack. |

---

## 7. Scene / world deltas (RoboCasa kitchen)

- **DOF explosion:** twin `nq` ≈ robot only (~7 + 27 + gripper hinges). Local
  `nq` ≈ **151** (kitchen fixtures + task objects). Bigger `mjv_updateScene`
  (~9 ms) and more broadphase/contacts per step → the viewer‑render throttle
  (`_viewer_max_hz=30`, `h12_mujoco.py:583`) exists solely to keep RTF up.
- **Spawn placement:** RoboCasa positions the base by layout/seed; at the
  all‑zero pose the arms jut ~0.31 m forward into the counter, so
  `place_robot_collision_free` **backs the base off** (up to 0.5 m) to clear
  fixtures — the robot may not spawn where the twin does relative to the world.
- **Contact zoo:** the robot can contact counters/objects the twin scene has no
  analog for; only the **foot↔floor** pair is patched to parity.
- **Camera framing / RTF:** local decimates rendering and can run `--slowmo`;
  twin decouples render at 30 Hz too. Both note RTF≈1 is the validated regime and
  a slow/choppy sim is itself an unfaithful test.

---

## 8. Initial‑state / spawn‑pose deltas

| | Twin | Local |
|---|---|---|
| Default spawn | Model `qpos0` (natural), hang‑band on pelvis; `--crouch K` sets knee=+K, hipP/ankP=−K/2 and re‑plants feet, pose **frozen** until first stiff cmd | Resets **all 27 motor qpos to 0** + zero qvel after RoboCasa's settling loop, re‑places pelvis; `--spawn-crouch K` mirrors the twin crouch via the resolver map |
| Bent‑knee stance available | `NOMINAL_STANCE`‑equivalent via `--crouch` | `nominal_stance_vector()` exists (`h1_2_robosuite.py:63`, legs = policy `default_angles`, knee 0.36) but **not the default**; default is all‑zero straight‑leg |
| Foot‑on‑floor guarantee | crouch re‑plants to ankle‑body height 0.047 | `place_robot_clear` lifts lowest robot AABB point to clearance 0.02 |

**Note (verified):** **neither** model file contains a `<keyframe>` — the
hypothesized bent‑knee "stand" qpos (~knee 0.35) lives only in the controller,
not in either sim. **Both sims spawn straight‑legged by default**, so the
straight‑knee spawn (near the knee‑hyperextension / one‑leg‑strut singularity at
handoff) is **common to both and not a differentiator by itself**. It only
becomes a delta if the runs used *different* crouch settings — e.g. a default
RoboCasa run (`--spawn-crouch 0`) compared against a twin `--crouch 0.7` run.
Match the crouch on both sides when A/B‑ing.

---

## 9. File inventory (what exists on each side)

**Overlapping (compared above):** `h12_mujoco.py`, `mujoco_env.py`,
`unitree_interface.py`.

**Local‑only (RoboCasa/ROS integration — not balance‑critical, but they hold the
sim lock and shape I/O):**
- `h1_2_robosuite.py` — registers `H1_2` as a robosuite robot + Magpie grippers;
  RoboCasa placement monkeypatches; `NullBase` (keeps pelvis freejoint) vs the
  MobileBase path that would weld the robot to the world.
- `sim_names.py` — `NameResolver` (ROS↔robosuite name/index map; 27‑motor order).
- `mujoco_ros_bridge.py` — `/clock`, RGBD, Livox lidar+IMU.
- `magpie_hand_bridge.py` — `/{left,right}/gripper/*` services (×2).
- `measurement_bridge.py` — `/robocasa/{task_name,task_goal,success,reward}`.
- `test_spawn_collision.py` — synthetic‑scene collision unit test.
- `mujoco_env.py` here is **only** the `ElasticBand` class (the twin's
  `mujoco_env.py` is the full `MujocoEnv` + `ElasticBand` + `EndEffectorForce`).

**Twin‑only (not ported; mostly irrelevant to the balance audit):**
- `augment.py`, `replay.py` — trajectory augmentation/replay.
- `h12_skin.py` — tactile skin.
- `pv_interface.py` — pyvista viz.
- `unitree_interface.py::ShadowInterface` — inspire‑hand shadow control.
- Large model tree: `unitree_robots/`, `inspire/`, `magpie/`, `archive/`,
  `utility/` (the twin repo *ships* the MJCF/meshes; the local sim gets its robot
  from `CL_Assets`).

---

## 10. ElasticBand / harness parity

The harness/handoff machinery is a **verbatim port** and is **not** a source of
divergence (constants match: 3000/300 strap, 300/60 posture, 5× grip boost,
15 s holdout, 10 s grip fade, 10 cm/6 s chunked payout, slack+assist release
gate, 3 s quadratic release fade). Minor mechanical differences:
- Twin band reads pelvis free joint `qpos[:3]`/`qvel[:3]`; local reads the
  **torso body** `xpos`/`qvel[base_dof:+3]`.
- Twin `sim_step` zeroes `xfrc_applied[:]` every step; local zeroes the band
  body's `xfrc_applied` each step in the loop. Same net effect (fresh spring
  force, no integral wind‑up).
- The posture‑assist "steadying hands" upright spring + angular damping is the
  same law on both.

---

## 11. MuJoCo version triad (verified)

| Component | MuJoCo | Source |
|---|---|---|
| MJPC planner (controller's internal model / rollouts) | **3.2.3** | `docker/BUILD.md:199` (CMake FetchContent; "matches the mjpc C++ server ABI") |
| Twin sim (`h1_mujoco`) | **3.6.0** | `pyproject.toml`, `requirements.txt` |
| Local RoboCasa plant | **3.3.1** | `docker/BUILD.md:145` ("RoboCasa's hard pin") |

Default contact/solver behavior (incl. multiccd availability, elliptic‑cone
regularization, implicitfast details) shifts across 3.2.3 → 3.3.1 → 3.6.0. The
controller plans in 3.2.3, the twin validates in 3.6.0, the local plant runs
3.3.1 — **no two links in the chain share a MuJoCo build.**

---

## 12. Suggested reproduction / triage for the downstream agent

1. **Confirm the failing run's flags.** If it was a bare `launch_robocasa.sh`,
   the five default‑OFF deltas (§3) are all live — start there.
2. **A/B the timestep first** (`--sim-dt 0.005` + matching node `twin_dt` +
   estimator `tick_dt`): the dev's own T2/T3b evidence says this is the single
   biggest stand discriminator.
3. **Turn on truth state** (`--truth-sportstate`, park the estimator elsewhere)
   and diff the printed `trueXY` (`[manual]` log line) against the estimator's
   `xy=[…]` — if truth stays put but the estimate drifts, the failure is
   estimator‑input, not control.
4. **Add contact + floor + hands parity** (`--elliptic-contact --plane-floor
   --real-hands`) and re‑test the free stand under `--harness`/`--auto-release`.
5. **Cross‑check the compiled model** in‑container (§5 lists the exact fields;
   most are already verified identical). Confirm at runtime: the printed
   `body_mass` gripper probe + `np.sum(model.body_mass)`, that `model.opt`
   really reads `impratio/cone/integrator/timestep` as expected, and whether any
   RoboCasa arena default overrode foot/floor `friction[0]` away from 1.0. The
   actuators, `actuatorfrcrange`, joint damping/armature/frictionloss, and the
   86‑sensor prefix are **verified identical** — deprioritize those.
6. Remember the plant is MuJoCo **3.3.1** while the twin is **3.6.0** and the
   planner is **3.2.3** — if parity flags all match and it still diverges,
   suspect version‑dependent contact/solver defaults.

---

## 13. Empirical: soak-loop stand test on the default RoboCasa plant (measured 2026-07-06)

**Setup.** Ran the whole-body bench (`ros2 launch h1_bringup h1_sim_fullbody_bench.launch.py`
— the `mjpc_fullbody_core` nu=27 Lean/strategy-6 controller, described in the launch
docstring as "the validated 281 s real-robot stand") against a **bare** headless RoboCasa
sim (`launch_robocasa.sh --headless`, `MUJOCO_GL=egl`). No parity flags were passed, so
**all five default-OFF deltas of §3 were live** (dt=0.002, no `--truth-sportstate`,
elliptic cone / impratio=20 contact, box floor, weightless hands). `hams_ros` image built at
`MJPC_REF=730d81e`; both containers on `ROS_DOMAIN_ID=67`, host-net. A driver loop brought the
stack up, watched the sim's **ground-truth** base height (`[manual]` log `z=`), recorded
time-to-fall, tore down, and repeated x6.

**Fall metric.** On this plant the robot does **not tip** — it **sags vertically** (knees
buckle, base height collapses from the ~1.03 m stand) and the elastic assist band catches it.
So "fall" = ground-truth `z < 0.90 m` sustained; corroborated by band tension climbing toward
its ~650 N payout cap.

**Result: it failed to hold the stand on every trial** — 5/6 crossed the fall line; the 6th
was a ROS-side process death while the robot hovered borderline. Each trial spawned a
*different* RoboCasa kitchen task (the source of the wide time-to-fall spread):

| Trial | RoboCasa task | Outcome | Time-to-fall (sim-s) | min z (m) | max tilt | peak band load |
|---|---|---|---|---|---|---|
| 1 | RestockPantry | fell | 141.7 | 0.88 | 35.8° | 569 N |
| 2 | PlateStoreDinner | fell | 16.5 | 0.86 | 19.8° | 689 N |
| 3 | TurnOffSimmeredSauceHeat | fell | 79.6 | 0.87 | 15.7° | 609 N |
| 4 | DateNight | fell | 122.7 | 0.84 | 19.7° | 671 N |
| 5 | TurnOffSimmeredSauceHeat | fell | 66.6 | 0.87 | 8.1° | 576 N |
| 6 | TurnSinkSpout | ros stack exited @ ~13 min wall; robot borderline (z≈0.89, never crossed 0.90) | — | 0.89 | 12.9° | 549 N |

**Observations.**
- **100% loss-of-stand** among the 5 trials that reached a stand verdict. Time-to-fall spans
  **16.5–141.7 sim-s** (median ≈ 80, mean ≈ 85) — a wide spread driven by RoboCasa placing the
  robot in a **different scene/task/pose each episode** (e.g. `PlateStoreDinner` lost balance in
  ~16 s; `RestockPantry` held 142 s).
- **Failure mode is vertical collapse, not tipping.** min z reaches 0.84–0.88 every time while
  tilt is usually small (trial 5: 8.1°, near-pure sag), with occasional forward tip
  (trial 1: 35.8°). The band bore **570–690 N** (near its ~650 N cap) in every trial → the
  plant cannot hold the whole-body stand unassisted under default flags.
- Because this controller is the shared, twin-validated stand core, the falls are
  **plant-attributable**, consistent with §0's ranking: a bare RoboCasa launch runs the wrong
  timestep (§0.1), feeds the estimator's drifting base state instead of truth (§0.2), and uses
  mismatched contact globals (§0.3).

**Caveats.** Headless sim runs at ~0.25x realtime, so each trial is ~2–13 min wall; the fall
times above are **sim-seconds** (from the sim `[manual]` clock), not wall-clock. This is the
**whole-body bench**, not the lower-body squat controller of the §0 top-line quote — but both
drive the identical RoboCasa plant, so the loss-of-stand result corroborates the same
plant deltas. Next step per §12: re-run this loop with `--sim-dt 0.005 --truth-sportstate
--elliptic-contact --plane-floor --real-hands` and compare the fall-rate / time-to-fall.

---

## 14. FABEL addendum (2026-07-06/07): corrections measured by live A/B on both plants

**Method.** Every §0/§3 delta was exercised live (17 RoboCasa trials + 5 twin
cross-tests, `FABEL_SILVERBULLETS.txt` has the full lab notebook). The twin
itself was run **on this host, against this exact ROS stack** (clone staged
in-container, DDS domain patched, mujoco 3.6.0) — the control experiment §13
never ran.

**14.1 The §13 attribution is invalid.** The identical bench (committed
`mjpc_fullbody.yaml`, `MJPC_REF=730d81e`) **hangs the twin exactly like
RoboCasa** (rides the harness payout to the rope cap at 330–440 N; and the
lower-body chain tuck-hangs at ~590 N on both plants). The stack in this
working tree stands **no** plant; §13's falls are therefore not
plant-attributable. The 218 s twin free stand (T1b) and 281 s real stand
belong to an earlier fork-state/config. Any future plant A/B must first
re-validate the twin baseline at the current ref.

**14.2 New, verified plant deltas the audit missed (fixed this session, all
opt-in flags in `h12_mujoco.py`):**
- **World-frame state poison (`--spawn-frame-state`)** — the biggest real
  finding. RoboCasa published raw kitchen-world pose: spawn yaw was **180°**
  and xy ≈ (−2.2, −3.6), while twin and real both hand the controller yaw≈0 /
  xy≈0 (twin spawns at the model origin; the real IMU boots yaw≈0). The Lean
  task's Body Yaw cost (w=40, lean.cc:2135) aligns the torso with the
  direction to the **planner-model object** at ~(1.1, 0) — a yaw'd/offset
  robot carries a permanent phantom heading error it can only fix by pivoting
  itself off its feet. This, not contact, drove the lateral-drift →
  one-knee-strut fold. The flag publishes lowstate quat yaw-normalized and
  truth sportstate spawn-relative (z referenced to the floor top); post-fix
  the post-release drift is zero.
- **Airborne spawn strap pre-load (`--settle-spawn`)** — `place_robot_clear`
  spawns the robot 2 cm off the floor; the band/harness anchor is captured
  pre-drop, so the strap starts 60–160 N loaded (twin: 5–15 N). A supported
  start is an attractor the policy exploits (tuck-hang). Pose-frozen 0.3 s
  settle before anchor capture restores twin geometry.
- **Free base joint sky-hook** — the twin's `<default>` joint class reaches
  its FREE joint: `dof_damping[0:6]=10, armature=0.1` (only frictionloss is
  zeroed), and the **planner model carries the same**; the robosuite base is
  0/0/0. `--base-damping` closes it (measured: not the stand-breaker).
- **`--real-hands` is half a fix** — robosuite's `inertiagrouprange="0 0"`
  drops the visual-mesh copies: assembled grippers are 0.253 kg/side vs the
  twin/planner 0.506 kg; with the missing 0.67 kg back box the plant is
  ~1.18 kg light (67.49 vs 68.67 kg). `--twin-mass` closes both.
- **Air vs vacuum** — robosuite base.xml sets density 1.2 / viscosity 2e-5;
  twin is vacuum. `--vacuum` (negligible at stand speeds).

**14.3 Corrections to existing sections.**
- §0.1/§4 dt: correct and **worse than stated** — §13 also ran the config
  side at twin_dt/tick_dt = 0.005 against the 0.002 plant, so the bring-up
  choreography compressed 2.5× and the planner's finite-diff base velocity
  scaled to 0.4×. `--sim-dt 0.005` is mandatory with the committed yamls.
- §0.2 truth-state: **not required for a stand.** Note the committed yaml has
  the controller on `rt/sportmodestate_est`, so `--truth-sportstate` alone
  changes nothing (nobody reads the topic) — flipping `sportstate_topic` is
  part of that A/B.
- §3/§4 multiccd: **already ON at robosuite assembly** (robocasa passes
  `enable_multiccd=True`); the runtime patch is a no-op, so multiccd cannot
  explain any A/B attributed to it. Solver is likewise already Newton.
- §4 `--twin-contact`: the launch docstring "pair with `--twin-contact`" is
  stale — the failed-A/B verdict stands even post-yaw-fix (T8). But
  `--elliptic-contact` was not the bullet either (both bracketed).
- §5e: collision coverage is also left/right asymmetric — `robot.xml` gives
  left_hip_pitch a collision mesh but not the right.
- §6 safety layer: `network.domain_id: 0` in the config is a fallback
  (env `ROS_DOMAIN_ID` wins); clip `position_offset` shrinks the absolute
  URDF range (docs/config.md mis-describes it as a tether to measured q);
  the only live clip for this stack is `torque_ratio` on tau_ff
  (ankle-pitch 36 Nm) — **exonerated** by a 2.0 A/B on the twin (TW-D2).
- §13 caveat to add: at ~0.25× RTF the *fullbody* yaml's `plan_threads: 3`
  comment logic inverts once `--no-sensors --sim-dt 0.005` push RTF≈1; the
  twin/real value is 12 (also bracketed: not the stand-breaker).
- Plumbing footnote: `mjpc_lowerbody_core` publishes
  `rt/safety/lowcmd_lower_in` (split safety) — it cannot be benched under
  `h1_sim_fullbody_bench.launch.py` (full-mode safety listens on
  `rt/safety/lowcmd_in`); use `h1_sim_bringup.launch.py use_skills:=false
  use_nav:=false use_rviz:=false use_sliders:=false`.
