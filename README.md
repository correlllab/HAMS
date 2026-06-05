# Humanoid_Simulation

Simulation stack for the Correll Lab **Unitree H1‑2** humanoid. A physics
simulator (MuJoCo, or Isaac Lab) publishes robot state + sensors, and a ROS 2
(Humble) workspace runs perception, planning, and control. Each piece runs in
its own Docker container and they talk over a shared **CycloneDDS** domain.

```
┌──────────────────────── Docker, network_mode: host, nvidia ────────────────────────┐
│                                                                                     │
│  humanoid_sim_mujoco                         humanoid_ros                           │
│  (h1_mujoco/h12_mujoco.py)                   (ros2 launch h1_bringup …)             │
│    ├ rt/lowstate (DDS) ─────────────────►  frame_task_server   (arm IK, upper body) │
│    ├ /realsense/head/*, /lidar, /tf, /clock ►  vp_node          (CLIP/SAM/Gemini)   │
│    │                                         lowerbody_controller (walk / FAME RL)  │
│    │                                         nav2 + slam, joint/robot_state, rviz   │
│    │                                              │ upper cmd        │ leg cmd      │
│    │                                              ▼                  ▼              │
│    └ rt/lowcmd (DDS) ◄──────────────────────  safety_node  (merge + clip + estop)  │
│        ▲ magpie grippers (/gripper/*)                                               │
│                                                                                     │
│  humanoid_sim_isaac  (alternative to mujoco) — Isaac Sim 5.1 + IsaacLab 2.3.2       │
└─────────────────────────────────────────────────────────────────────────────────── ┘
```

The MuJoCo and Isaac containers are **interchangeable** physics backends; you run
one or the other, never both. Everything shares `ROS_DOMAIN_ID` (default `1`;
`0` is reserved for the real robot).

## Repository layout

| Path | What it is |
|------|------------|
| `docker/` | `BaseDockerfile` + `{Mujoco,Ros,Isaac}Dockerfile`, `docker-compose.yml`, and `scripts/` (`docker_build.sh`, `docker_run.sh`, `launch_*.sh`). |
| `h1_mujoco/` ⊂ | MuJoCo sim: `h12_mujoco.py` (entry), `mujoco_ros_bridge.py` (camera/lidar/TF/`/clock`), `unitree_interface.py` (`rt/lowstate`↔`rt/lowcmd` DDS), `magpie_hand_bridge.py`, `scene_builder.py` (Robocasa kitchen). |
| `CL_isaaclab_sim/` ⊂ | Isaac Lab tasks + `sim_main.py` (alternative backend). |
| `CL_Assets/` ⊂ | Robot URDF (`ros_assets/h1_2_magpie_ros.urdf`), MuJoCo XML (`mujoco_assets/h1_2_magpie.xml`), meshes, scene/object assets. |
| `core_ws/` | The ROS 2 workspace — `src/` (below), plus `build/`/`install/`/`log/`. |
| `container_cache/` | Persists `msgs_ws` build artifacts across `--rm` container runs. |

`⊂` = git submodule. Most `core_ws/src` packages are submodules too:

| `core_ws/src/` package | Role |
|------------------------|------|
| `h1_bringup` | Top-level launch (`h1_sim_bringup.launch.py`), nav launch, and tasks (`scripts/open_fridge.py`, `slider_debugger.py`). *In‑repo.* |
| `h12_ros2_controller` ⊂ | Arm control: `frame_task_server` (Pink differential IK + QP) on the `/frame_task` action; joint_state_publisher. |
| `h12_ros2_model` ⊂ | Robot description helper. |
| `h12_lowerbody_controller` | **Lower-body RL** — walking + FAME standing/squat policies, the switchable controller (see below). *In‑repo.* |
| `h12_safety_layer` ⊂ | Merges lower (legs) + upper (torso/arms) commands → `rt/lowcmd`; clipping + e‑stop. |
| `vision_pipeline` ⊂ | CLIP/SAM2/SAM3 + Gemini perception; `/vp_*` services. |
| `custom_ros_messages`, `magpie_msgs` ⊂ | IDL: `FrameTask` action, `Query`/`UpdateBeliefs` srvs, gripper msgs. |
| `unitree_ros2` ⊂, `livox_ros_driver2` ⊂, `FAST_LIO` | Unitree DDS msgs (`LowState`/`LowCmd`), Livox driver, LiDAR‑inertial SLAM. |

## Docker images

`humanoid_sim_base` (CUDA 12.2 / Ubuntu 22.04 / ROS Humble / CycloneDDS 0.10.x
from source / torch cu130 / pin·pink·mink) is the shared base for **mujoco** and
**ros**. **isaac** is self‑contained (conda Python 3.11, Isaac Sim 5.1).

## Quickstart

**Prerequisites:** Docker + Compose v2, NVIDIA Container Toolkit, `git lfs install`,
and `git submodule update --init --recursive`. The vision pipeline needs a
`core_ws/src/vision_pipeline/vision_pipeline/API_KEYS.py` containing
`GEMINI_KEY = "..."` (untracked — add your own).

```bash
# 1. build (base is built automatically for mujoco/ros)
docker/scripts/docker_build.sh mujoco ros        # or: isaac, or no args = all

export ROS_DOMAIN_ID=1                            # match across all terminals

# 2. terminal A — physics sim (start first so /clock is publishing)
docker/scripts/docker_run.sh mujoco              # windowed; --headless for no GUI

# 3. terminal B — ROS workspace shell (auto-builds core_ws on first run), then bringup
docker/scripts/docker_run.sh ros
ros2 launch h1_bringup h1_sim_bringup.launch.py   # args: use_rviz/use_sliders/use_nav/lowerbody_policy
```

Scripts resolve their own path (run from anywhere). `ROS_DOMAIN_ID` of `0`/unset
is normalized to `1`. Run `docker/scripts/docker_run.sh mujoco bash` to drop to a
shell instead of auto-launching.

## Lower-body RL control

The legs are driven by a TorchScript RL policy. `h12_lowerbody_controller` wraps
this as a **switchable controller** at 50 Hz that publishes leg PD setpoints to
the safety layer (`/safety/lowcmd_lower_in`); the arms come from the IK
(`frame_task_server`).

**Policies** (`policies/<name>/` holds weights + `*.yaml`):
- `walk` — locomotion policy (velocity command via `/cmd_vel`).
- `fame` — RMA standing/squatting policy (env-factor encoder + history); base
  height via `/lowerbody/squat_cmd`. Trimmed inference encoder lives in
  `h12_lowerbody_controller/rma/`.

**Operating model** — the bringup runs `lowerbody_controller_node`:
1. Robot starts **held by the elastic band, idle** (`lowerbody_policy:=none`).
2. **Start a policy** (releases the band once `frame_task` is ready, then engages):
   ```bash
   ros2 service call /lowerbody/start_fame std_srvs/srv/Trigger    # stand
   ros2 service call /lowerbody/start_walk std_srvs/srv/Trigger    # walk
   ```
3. **Switching** is gated — it commits only when the robot is standing still and
   upright, resetting the incoming policy for a clean handover.

Status on `/lowerbody/active_policy`; `/lowerbody/set_policy` (String) is an
equivalent topic. Set `lowerbody_policy:=fame` to auto-engage at launch instead
of starting idle. `walking_node` / `fame_node` are standalone single-policy
nodes; `reference/standalone_fame_test.py` is a no‑ROS MuJoCo regression harness.

## Working example — open the fridge

![open_fridge demo — robot head-camera view](docs/open_fridge.gif)

*Robot's head-camera view of the `open_fridge` demo: the gripper grasps the
fridge handle and swings the door open (~50°). Full clip:
[docs/open_fridge.mp4](docs/open_fridge.mp4).*

The robot stays band-held (stable for manipulation) while the demo runs. Match
`ROS_DOMAIN_ID` in every terminal.

```bash
export ROS_DOMAIN_ID=1

# terminal A — MuJoCo physics (add --record to capture the head-cam view)
docker/scripts/docker_run.sh mujoco --headless --record /home/code/h1_mujoco/open_fridge.mp4

# terminal B — ROS bringup
docker/scripts/docker_run.sh ros
ros2 launch h1_bringup h1_sim_bringup.launch.py use_rviz:=false use_sliders:=false

# terminal C — run the demo
docker exec -it humanoid_sim_ros bash
ros2 run h1_bringup open_fridge.py
```

`open_fridge.py`: query the handle (`vision_pipeline`, Gemini — logged for
reference) → reach the model-derived handle pose, grasp it (`/frame_task` IK +
gripper), and pull the door open along its hinge arc. The robot stays band-held,
so it does not navigate; the grasp uses the static handle pose rather than the
noisy vision centroid for a repeatable demo.

**Recording the demo** — pass `--record <path.mp4|.gif>` to the MuJoCo sim. It
saves the robot's head-camera view to a host-visible path under `h1_mujoco/`
(here `h1_mujoco/open_fridge.mp4`), encoded when the sim exits — stop it with
`Ctrl-C` so the file is finalized. Requires `--headless` (the offscreen egl
renderer); `.gif` output embeds straight into Markdown like the clip above.

## Isaac container — WIP

`docker/scripts/docker_run.sh isaac` runs `launch_isaac.sh` (Isaac Sim 5.1 +
IsaacLab 2.3.2). Its OmniGraph DDS bridge currently pins itself to domain 1
regardless of the host setting — non-default-domain bridging is a known gap.

## Troubleshooting

- **GUI** (rviz / MuJoCo viewer / sliders won't open): `xhost +local:docker` once per session.
- **Host can't see sim topics:** export the same non-zero `ROS_DOMAIN_ID`.
- **Clean message-workspace rebuild:** delete `container_cache/msgs_ws/` before relaunching.
