# CLAUDE.md — HAMS developer guide for coding agents

Everything below was verified against the code on 2026-07-12. Where older docs or
script comments disagree with the code, the code wins (known stale spots are
flagged inline).

## 1. Project overview

HAMS (Humanoid Agent Modular Stack) is the software stack for a Unitree H1-2
humanoid: MuJoCo/RoboCasa and Isaac Lab simulators, a ROS 2 Humble workspace
(drivers, upper-body IK control, lower-body MPC/RL locomotion, safety layer,
perception model servers, and an LLM-callable skills layer), all running in
separate Docker containers that share one DDS domain. The same `rt/lowcmd` /
`rt/lowstate` Unitree DDS wire format is used against the sims and the real
robot, so code moves between them unchanged.

## 2. Repo layout

| Path | What it is |
|---|---|
| `docker/` | x86 setup — the fully-featured, preferred platform (Dockerfiles, `docker-compose.yml`, `scripts/`) |
| `docker/mac/` | self-contained Apple-Silicon (arm64, CPU-only) port — limited: no Isaac, no ML/vision stack, FastDDS instead of CycloneDDS |
| `core_ws/` | the ROS 2 workspace — where the bulk of development happens (see §4) |
| `h1_robocasa/` | RoboCasa/MuJoCo simulator entry point (`h12_mujoco.py`) and its ROS/DDS bridges |
| `CL_isaaclab_sim/` | Isaac Lab simulator (`sim_main.py`, DDS bridges, tasks) |
| `CL_Assets/` | URDF meshes, MuJoCo XML, Isaac USD (Git-LFS) |
| `mujoco_mpc/` | MuJoCo-MPC fork submodule (built standalone, outside colcon — see §8) |
| `unitree_sdk2_python/` | vendored Unitree DDS SDK (Python) |
| `tools/` | standalone debug tools (ROS MCP server for Claude Code) |
| `docs/` | `NAVIGATION_DEMO.md` (mac SLAM/nav2/frontier demo), `ROS_MCP_DEBUG.md` |
| `docker/BUILD.md` | deep dive on the image/build system — read it before touching any Dockerfile |

Most of the tree is **git submodules** (`git submodule update --init --recursive`
is mandatory) and large assets are **Git-LFS** (`git lfs install`).

## 3. Containers: always use the provided scripts

Never `docker run` the images by hand — the scripts wire up `docker/.env`, DDS
domain safety checks, bind-mount pre-creation, and stable container names.
They can be invoked from any directory.

### x86 (preferred platform)

| Script | Verified arguments | Notes |
|---|---|---|
| `docker/scripts/docker_build.sh [profile ...]` | `isaac`, `robocasa`, `ros` (no args = all three) | `hams_base` is built first automatically for `robocasa`/`ros`; `isaac` is self-contained. Pins the MJPC build to the `mujoco_mpc` submodule SHA (`MJPC_REF`). |
| `docker/scripts/docker_run.sh <profile> [cmd...]` | `isaac`, `robocasa`, `ros`; then optionally `bash` (shell instead of default launcher) or any command; a leading `-flag` (e.g. `--headless`) is forwarded to the default launcher | Container names: `hams_ros`, `hams_sim_robocasa`, `hams_sim_isaac`. |

Default launchers (compose `command`): `docker/scripts/launch_isaac.sh`,
`launch_robocasa.sh`, `launch_ros.sh`. The `ros` launcher colcon-builds
`core_ws` (if stale) and **drops to a shell — it does not launch bringup**.

### mac (`docker/mac`, limited)

| Script | Verified arguments | Notes |
|---|---|---|
| `docker/mac/scripts/docker_build_mac.sh [service ...]` | `robocasa`, `ros` (no args = both) | **No `isaac`** — Isaac Sim needs an NVIDIA GPU. arm64 base built from `docker/mac/BaseDockerfile.arm64`. |
| `docker/mac/scripts/docker_run_mac.sh <service> [cmd...]` | `robocasa`, `ros` | Starts ONE service. For the paired sim prefer `docker compose -f docker/mac/docker-compose.yml up` so both start together (RoboCasa running alone for a few seconds lets the robot collapse — motor command timeout). |

Mac feature toggles are env vars read by the compose file: `HAMS_DISPLAY=vnc`
(MuJoCo viewer → noVNC :6080), `HAMS_RVIZ=vnc` (RViz → :6081), `HAMS_LOWERBODY=fame|walk|switch`,
`HAMS_SLAM=1`, `HAMS_NAV2=1`, `HAMS_SIM_ODOM=1`, `HAMS_CAMERAS=0`,
`HAMS_SPAWN_BACKOFF=<m>`, `HAMS_CMD_TIMEOUT=<sim-s>`, `HAMS_ROS_MCP=1`.
GUIs stream over noVNC via `docker/mac/scripts/mac_vnc_tunnel.sh` (no XQuartz).

### DDS domain safety (both platforms)

`ROS_DOMAIN_ID=0` is the **real robot's command bus**. The run scripts refuse to
start a sim on it (the x86 `ros` profile asks for interactive confirmation; mac
always rejects). Unset/empty defaults to `1` (the sim domain). Set it in
`docker/.env` (copy from `docker/.env.example`; also holds `GEMINI_API_KEY`,
which the `ros` container needs for the vision servers and `h12_skills`).

## 4. `core_ws/src` packages

19 ROS packages in 16 directories. Note two name mismatches: dir `FAST_LIO` →
package `fast_lio`; dir `unitree_ros2` is not itself a package (it nests
`unitree_api`, `unitree_go`, `unitree_hg`, `unitree_ros2_example`).

| Directory | Role | What it does | Key entry points |
|---|---|---|---|
| `h1_bringup` | **bringup** | Launch-only package that starts the whole robot (sim or real). See §6. | 7 launch files, `config/*.yaml` |
| `h12_ros2_controller` | controller (upper body) | Pinocchio-based arm IK: `/frame_task` + `/dual_arm` action servers, joint state publisher, hand controller | `frame_task_server`, `dual_arm_server`, `joint_state_publisher`, `hand_controller_node` |
| `h12_deploy_mjpc` | controller (lower body, MPC) | MJPC balance/locomotion controller (C++ cores from the `mujoco_mpc` fork, run as separate processes) + RW-EKF base estimator | `mjpc_lowerbody_core`, `controller_launcher.py`, `estimator_node.py` |
| `h12_lowerbody_rl` | controller (lower body, RL) | TorchScript walking + FAME RMA stand/squat policies; switchable stand↔walk controller; consumes `/lowstate` + `/cmd_vel`, emits 12-joint PD setpoints | `walking_node`, `fame_node`, `lowerbody_controller_node` |
| `h12_safety_layer` | safety | Merges upper/lower command channels into `/lowcmd` with limit checks (YAML-configured) | `safety_node` |
| `h12_skills` | **skills** | Action servers for the 12 `/skill/*` atomic skills an LLM can call (frontier exploration lives here). See §7. | `skills` (SkillsNode) |
| `model_server` | model server | ROS service servers wrapping ML models: Gemini VLM, SAM3 segmentation, GraspGenX grasp generation, YOLO detection | `gemini_server`, `sam_server`, `graspgen_server`, `yolo_server` |
| `custom_ros_messages` | interfaces | 18 actions (12 `Skill*`, `FrameTask`, `NamedConfig`, `DualArm`, …), msgs, srvs (`GeminiQuery`, `SamSegment`, `GraspGen`, …) | rosidl only |
| `magpie_control` | driver (gripper/sensors) | Dynamixel gripper node (`/left|right/gripper/*` services, DeliGrasp action), F/T + tactile sensor nodes | `gripper_node`, `ft_sensor_node` |
| `magpie_msgs` | interfaces | Gripper/manipulation interfaces (`GripperState`, `SetGripperPosition/Force`, `DeliGrasp`, …) | rosidl only |
| `estop` | safety (hardware) | Serial e-stop bridge → ROS Bool (real robot only; launched by real bringup, not sim) | `estop_node` |
| `cl_realsense` | perception util | Point-cloud accumulator (tf2 + Open3D) for the RealSense cams; ships the real-camera launch file | `pc_acc`; `launch/h12_rs_cams.launch.py` |
| `FAST_LIO` (`fast_lio`) | SLAM (vendored) | LiDAR-inertial odometry/mapping from the Livox cloud | `fastlio_mapping` |
| `livox_ros_driver2` | driver (vendored) | Livox MID360 lidar driver (builds via its own `build.sh humble`) | `livox_ros_driver2_node` |
| `unitree_ros2/*` | interfaces (vendored) | Unitree IDL packages — `unitree_hg` `LowCmd_`/`LowState_` are the stack's spine | rosidl only |

Third-party/vendored (do not refactor): `FAST_LIO`, `livox_ros_driver2`,
`unitree_ros2`, plus the top-level `unitree_sdk2_python` and `mujoco_mpc`.

## 5. Ownership & scoping convention

Development is **scoped to a single package**. Each package (the ROS 2
controller, the MJPC deploy package, the lower-body RL package, the skills
package, the model servers, the simulators, …) has its own owner; several are
separate git submodules with their own history. Rules of thumb:

- Confine a change to the package it belongs to. If a task seems to require
  edits across packages, stop and coordinate first.
- Shared infra — `h1_bringup`, `custom_ros_messages`/`magpie_msgs`,
  `docker/`, the safety layer, the simulators — affects everyone. Treat
  changes there as interface changes: minimal, deliberate, announced.
- Submodule packages must be committed and pushed in the submodule repo, then
  the pin bumped here. Never leave a submodule pointing at an unpushed commit.

## 6. Bringup: how the robot starts

`h1_bringup` is launch-only. Sim vs real is **separate launch files** (there is
no `sim:=true` arg):

| Launch file | Scenario |
|---|---|
| `h1_sim_bringup.launch.py` | **x86 sim** — full stack: nav (via include), robot/joint state publishers, `frame_task_server`, `safety_node`, gemini/sam servers, MJPC estimator + lowerbody controller, graspgen + skills (`use_skills`, default true), rviz (`use_rviz`) |
| `h1_sim_bringup_mac.launch.py` | **mac sim** — trimmed: state publishers, `frame_task_server`, `safety_node`; lower body/SLAM/nav2 gated by `HAMS_*` env vars, not launch args |
| `h1_real_robot_bringup.launch.py` | **real robot, onboard PC** (native, `ROS_DOMAIN_ID=0`) — aggregates the three below-listed real files |
| `h1_real_drivers.launch.py` | real: Livox MID360, RealSense cams, left+right `gripper_node` |
| `h1_real_controller.launch.py` | real: estop, state publishers, staggered `safety_node` + `frame_task_server` |
| `h1_real_desktop_bringup.launch.py` | real: companion x86 desktop — model servers, skills, MJPC controller. Leg control is interlocked behind `start_position_verified:=true` (default **false**) |
| `h1_navigation.launch.py` | shared nav stack (FAST-LIO → pointcloud_to_laserscan → slam_toolbox → nav2); included by sim and real bringups |

**Wiring a new package into bringup:** add it as `exec_depend` in
`h1_bringup/package.xml`, add a `launch_ros.actions.Node(...)` entry to the
relevant launch file (put node params in `h1_bringup/config/<file>.yaml` — the
top-level YAML key must match the node `name`), and gate optional nodes with
`IfCondition(LaunchConfiguration(...))`. See the `h12_deploy_mjpc` entries in
`h1_sim_bringup.launch.py` as the model. **Mac only:** also add the package to
the `PKGS` list in `docker/mac/scripts/launch_ros_mac.sh` or the slim image
won't build it.

## 7. Skills (`h12_skills`) — the LLM-callable layer

Each skill is a mixin class in `h12_skills/skills/` with one
`_exec_<name>()` method; `SkillsNode` registers one action server per entry in
the `SKILL_ACTIONS` table (`base.py`). All types are
`custom_ros_messages/action/Skill*`; every goal has a `timeout` Duration
(0 → 300 s default); every result is `success` + `message`; feedback is
`phase` + `progress`.

| Action server | Status |
|---|---|
| `/skill/grasp`, `/skill/pick_place`, `/skill/frontier_explore` | **implemented** |
| `/skill/open_door`, `/skill/close_door`, `/skill/open_lid`, `/skill/close_lid`, `/skill/navigate`, `/skill/press`, `/skill/slide_rack`, `/skill/turn_lever`, `/skill/twist_knob` | stubs — goal accepted, then aborted with "not implemented" |

Skills consume the `frame_task` IK action, the gripper services, and the
`model_server` perception services (`gemini_query`, `sam_segment`, `graspgen` —
this is why the `ros` container needs `GEMINI_API_KEY`). There is currently no
autonomous LLM orchestrator in-repo that calls them; they are invoked over ROS
(by a human, an external agent, or the `tools/ros_mcp_server.py` debug MCP).

Verified invocation examples (from a shell inside `hams_ros`):

```bash
ros2 action send_goal /skill/grasp custom_ros_messages/action/SkillGrasp \
  "{target_object: 'vertical fridge handle', arm: 'right', timeout: {sec: 60, nanosec: 0}}" \
  --feedback

ros2 action send_goal /skill/frontier_explore custom_ros_messages/action/SkillFrontierExplore \
  "{min_frontier_cells: 6, blacklist_radius: 0.6, min_goal_distance: 0.7, goal_timeout: 120.0, timeout: {sec: 1800, nanosec: 0}}" \
  --feedback

# non-skill but useful: move arms to a named posture first
ros2 action send_goal /named_config custom_ros_messages/action/NamedConfig \
  "{config_name: 'home', duration: {sec: 0, nanosec: 0}}"
```

## 8. Do-not-break rules

### The simulators are shared infra — don't modify them casually

`CL_isaaclab_sim` and `h1_robocasa` should generally not be modified unless
necessary, and their **default behaviors must be preserved**. Everyone's
sim-vs-real workflow depends on them behaving like the real robot.

### Simulators mimic the real robot's topic surface

They should publish only what the real robot publishes and consume only what
it consumes. Verified current surface:

**RoboCasa (`h1_robocasa/h12_mujoco.py`)** — publishes `/clock`,
`/realsense/{head,left_hand,right_hand}/color/image_raw[/compressed]`,
`.../aligned_depth_to_color/image_raw[/compressedDepth]`, `.../color/camera_info`,
`/livox/lidar` (CustomMsg), `/livox/pointcloud`, `/livox/imu`,
`/{left,right}/gripper/state`, and `rt/lowstate` over Unitree DDS; consumes
`rt/lowcmd` and the `/{left,right}/gripper/*` services/DeliGrasp action.
(The topic list in `launch_robocasa.sh`'s header comment and the root README is
stale — trust the bridge code in `mujoco_ros_bridge.py`.)

**Isaac (`CL_isaaclab_sim/sim_main.py`)** — publishes `rt/lowstate` and
`rt/inspire/state` over DDS plus compressed left-hand camera topics over ROS;
consumes `rt/lowcmd`, `rt/inspire/cmd`, `rt/reset_pose/cmd`. Lidar/IMU ROS
publishers are advertised but currently never fed (publish code commented out).

### Ground truth must NEVER be on by default

Sims may *optionally* publish ground-truth data (true base/object poses), but
it must be opt-in. Rationale: any node that consumes ground truth "cheats" —
it works in sim and silently fails on the real robot, where no such topic
exists. Keeping it off by default keeps the sim honest and protects
sim-to-real transfer.

- **RoboCasa complies:** ground-truth `/odom` + `odom→pelvis` TF is gated by
  `HAMS_SIM_ODOM` (default `'0'` = off, `h12_mujoco.py:184`). Only the mac nav
  demo opts in. Keep it that way.
- **Known deviations (do not make worse):** Isaac publishes full scene ground
  truth (`rt/sim_state`, robot + object poses as JSON) unconditionally every
  loop — there is no off switch, and `docker/scripts/dds_bridge.py` even relays
  it across domains. RoboCasa always publishes privileged task signals
  (`/robocasa/success`, `/robocasa/reward`, `/robocasa/task_goal`). Never make
  robot-stack code *depend* on any of these, and don't add new ungated
  ground-truth publishers.

### Other invariants

- MuJoCo versions are pinned per image (`3.2.3` in `ros` to match the MJPC ABI,
  `3.3.1` in `robocasa`) — never bump one without the other side of its pin.
- `numpy<2`, `setuptools==59.6.0`, `wheel<0.44` are deliberately re-clamped in
  the Dockerfiles; add new pip deps at the END of a Dockerfile (layer-order
  discipline). Details in `docker/BUILD.md`.
- After bumping the `mujoco_mpc` submodule: rebuild the `ros` image at the
  matching `MJPC_REF` and wipe stale `core_ws/build/h12_deploy_mjpc`.

## 9. Running the robot — the standard three-terminal flow

Everyone runs their package through bringup, which starts the whole robot.

### x86

```bash
# Terminal 1 — simulator (start FIRST so /clock is publishing)
docker/scripts/docker_run.sh robocasa            # or: isaac; add --headless if no display

# Terminal 2 — ROS workspace (auto colcon-builds core_ws, drops to a shell)
docker/scripts/docker_run.sh ros
ros2 launch h1_bringup h1_sim_bringup.launch.py   # manual step — the container does NOT launch it
# variants: use_skills:=false use_rviz:=false use_nav:=false

# Terminal 3 — drive skills: attach to the running ROS container
docker exec -it hams_ros /bin/bash
source /opt/ros/humble/setup.bash
source /home/code/core_ws/install/setup.bash
ros2 action send_goal /skill/grasp custom_ros_messages/action/SkillGrasp \
  "{target_object: 'vertical fridge handle', arm: 'right', timeout: {sec: 60, nanosec: 0}}" --feedback
```

### mac

```bash
# Terminals 1+2 in one: start BOTH services together (bringup auto-launches in ros)
docker compose -f docker/mac/docker-compose.yml up
# or two terminals: docker/mac/scripts/docker_run_mac.sh robocasa  /  ... ros
# (start ros within seconds of robocasa or the robot collapses)

# Terminal 3
docker exec -it hams_ros bash                    # fallback: colima ssh -- docker exec -it hams_ros bash
source /home/code/h12_sim_scripts/robot_cli.sh   # rob_pose, rob_grip, rob_stand, rob_explore, ...
```

The mac `ros` container auto-runs
`ros2 launch /home/code/core_ws/src/h1_bringup/launch/h1_sim_bringup_mac.launch.py`
(launched by path; only skills built on mac are available — no `h12_skills`,
`model_server`, or nav unless `HAMS_SLAM/HAMS_NAV2` are set).

### Real robot (for awareness — coordinate before touching)

Onboard PC, native (no docker): `export ROS_DOMAIN_ID=0`, source
`core_ws/install/setup.bash`, then
`ros2 launch h1_bringup h1_real_robot_bringup.launch.py`. Companion desktop:
`ros2 launch h1_bringup h1_real_desktop_bringup.launch.py`
(`start_position_verified:=true` only after physically verifying the robot's
pose — it gates leg commands).

## 10. Getting-started checklist for a new agent

1. `git submodule update --init --recursive && git lfs install` (most of the
   tree is submodules; builds fail cryptically without this).
2. `cp docker/.env.example docker/.env`; set `GEMINI_API_KEY` and
   `ROS_DOMAIN_ID` (any non-zero; 0 is the real robot).
3. Place SAM3 weights at `core_ws/src/model_server/weights/sam3.pt`
   (gated HF download — see root `README.md`).
4. `docker/scripts/docker_build.sh robocasa ros` (add `isaac` if needed).
5. Run the three-terminal flow in §9; confirm with `ros2 topic hz /lowstate`
   and `ros2 action list | grep skill` inside `hams_ros`.
6. Trigger `/skill/grasp` or the frontier explorer (§7).
7. Read first: root `README.md` (run flows, DDS tuning, mac port),
   `docker/BUILD.md` (build system), your target package's source.

### Common pitfalls

- **Start RoboCasa before bringup** — ROS nodes latch onto sim `/clock`; and if
  you restart the `robocasa` container, restart `hams_ros` too (sim time resets
  to 0 → TF extrapolation errors, frozen SLAM map).
- **x86 `ros` container ≠ bringup.** It only builds and gives you a shell.
- Rebuild is incremental across container restarts (build/install are
  host-mounted); `colcon build --symlink-install` inside the container forces
  it; wipe `container_cache/msgs_ws/` for a clean message rebuild.
- MJPC C++ iteration: `docker exec -it hams_ros /home/code/h12_sim_scripts/rebuild_mjpc.sh`
  (`--install` to also refresh assets/proto) — not colcon.
- 9 of the 12 `/skill/*` actions are stubs (§7) — a goal that immediately
  aborts with "not implemented" is expected, not a regression.
- Talking to the sim from the host bypasses `docker/.env`; run
  `set -a; source docker/.env; set +a` first or topics will be invisible.
- Laggy point clouds/images on x86 are usually kernel UDP buffer drops — see
  "Network / DDS tuning" in the root README (sysctls + IRQ pinning).
- Stale name alert: `docker/mac/scripts/launch_ros_mac.sh` `PKGS` still lists
  `h12_lowerbody_controller`; the package is now `h12_lowerbody_rl`. If the mac
  lower-body stack won't build/launch, fix that list.
- Isaac's `launch_isaac.sh` hardcodes its task on the `exec` line — the
  script's task/`--headless` parsing is dead code; pass flags by invoking
  `sim_main.py` directly if you need a different task.
