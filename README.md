# Humanoid Simulation

Dual-backend simulator for the Unitree H1-2 humanoid: both backends speak the
real robot's CycloneDDS protocol on `rt/lowstate` / `rt/lowcmd`, so a single
controller binary targets either physics engine or the actual robot with no
code changes.

---

## Quick Start

**Prerequisites** — Linux + NVIDIA GPU (driver 570.x recommended) + Docker
with the NVIDIA container runtime + ~10 GB free disk for the MuJoCo image.

```bash
# 1. Clone with submodules.
git clone --recurse-submodules https://github.com/correlllab/Humanoid_Simulation.git
cd Humanoid_Simulation

# 2. Build the MuJoCo backend (smallest path; ~10 min on a fresh host).
./docker/scripts/docker_build.sh mujoco

# 3. Run a working demo.
./docker/scripts/docker_run.sh mujoco
```

You should see the MuJoCo passive viewer with the H1-2 robot suspended in a
Robocasa kitchen via an elastic band on `torso_link`. The container also
publishes `rt/lowstate` on CycloneDDS domain 1 plus camera, lidar, IMU and
`/clock` on ROS 2 — see *Testing → Smoke Test* to verify.

For Isaac Lab GPU sim, the ROS 2 control stack, training, or troubleshooting,
see below.

> **Tip — bake the warmed Isaac image after the first run.** The first
> `./docker/scripts/docker_run.sh isaac` spends 20–40 minutes compiling Kit
> shaders and warming texture caches before the viewport opens. The shader and
> asset caches live at `./CL_isaaclab_sim/.isaac_cache/` on the host
> (bind-mounted from the container) and persist across runs *on this machine*.
> To share the warmup with teammates or carry it to another machine, snapshot
> the running container into a new image while it's still up:
>
> ```bash
> # From a second terminal, after the Isaac viewport is open and idle:
> docker commit humanoid_sim_isaac humanoid_sim_isaac:latest
> # Optional: also archive the bind-mounted cache for clean clones.
> tar -C CL_isaaclab_sim -czf isaac_cache_warm.tgz .isaac_cache
> ```
>
> Subsequent `./docker/scripts/docker_run.sh isaac` invocations reuse the
> committed image and skip the long compile entirely.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Supported Backends](#supported-backends)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Robot Models & Assets](#robot-models--assets)
- [Tasks / Environments](#tasks--environments)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Development](#development)
- [Testing](#testing)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This repository wraps two independent physics simulators behind a common DDS
wire format so the same controller stack works against either:

- **NVIDIA Isaac Lab** (Isaac Sim `5.1.0.0` + IsaacLab `v2.3.2`) — GPU
  PhysX 5, USD scene format, manager-based RL environment, OmniGraph-based
  ROS 2 publishing using the bundled `isaacsim.ros2.bridge` extension.
- **MuJoCo `3.6.0`** — lightweight CPU/EGL physics with MJCF scenes and a
  passive GLFW viewer.

Both processes publish on **CycloneDDS domain 1** with `unitree_sdk2py`'s
`rt/lowstate` / `rt/lowcmd` topics, matching the protocol the real Unitree H1-2
hardware uses on domain 0. The same code path that drives a real robot drives
the sim — switch domains, switch targets.

A separate ROS 2 Humble container runs the control / perception side: a
Pinocchio-based IK frame-task action server, a vision pipeline (YOLO + SAM +
Gemini), a joint-limit / e-stop safety layer, and RViz. Sensor data
(RealSense head camera, "Livox" half-sphere lidar, IMU, `/tf`, `/clock`) flows
from sim → ROS 2 over the standard ROS 2 RTPS wire on `ROS_DOMAIN_ID=1`.

The system is split across three Docker images. The MuJoCo and ROS images
share a common base (`humanoid_sim_base`) with CUDA 12.2, ROS 2 Humble,
CycloneDDS 0.10.x, `unitree_sdk2py`, PyTorch cu130 and Pinocchio. The Isaac
image is **self-contained** (Isaac Sim 5.1 needs Python 3.11; the base image
is pinned to 3.10 for ROS 2 Humble apt packages).

---

## Features

- **Two simulation backends** behind a common DDS wire format (`H12Interface`
  / `SimInterface`) so controllers run unmodified against either.
- **CycloneDDS-on-domain-1** wire compatibility with real Unitree H1-2 hardware
  (Unitree HG protocol, 27-motor body).
- **Watchdog pose-hold** in both backends: if no `rt/lowcmd` arrives for
  100 ms the sim freezes joints with a stiff PD (`kp=80`, `kd=3`) instead of
  letting the robot collapse / NaN.
- **Sensor parity across backends** — both publish `/clock`,
  `/realsense/head/...`, `/livox/lidar`, `/livox/imu` (see *Backend matrix*
  for per-backend topic-format differences).
- **Layered Docker build**: shared `humanoid_sim_base` (CUDA 12.2 + ROS 2
  Humble + CycloneDDS 0.10.x + PyTorch cu130 + pin/pink/mink) → `mujoco` and
  `ros` variants. The `isaac` image is built directly on `nvidia/cuda` to
  carry its own Python 3.11.
- **Headless support** — both backends auto-switch to `--headless` when no
  `DISPLAY` is reachable (CI / SSH / cloud).
- **Elastic band tether** in MuJoCo — keeps the H1-2 upright in the kitchen
  scene; `SPACE` toggles, arrow keys move the anchor, `,` / `.` adjust rest
  length.
- **Shader cache persistence** for Isaac Sim — first-boot ~20–40 min compile
  is bind-mounted from `./CL_isaaclab_sim/.isaac_cache/` so subsequent boots
  are fast.
- **OmniGraph ROS bridge in Isaac** — the Isaac container has no `rclpy`
  installed; sensor / clock publishing is authored as an OmniGraph action
  graph using nodes from the bundled `isaacsim.ros2.bridge` extension.
- **ROS 2 control stack** — differential IK frame-task server (Pinocchio),
  joint-limit safety layer, RealSense + LiDAR drivers, FAST-LIO SLAM, vision
  pipeline.

---

## Supported Backends

| Backend | Engine version | OS | GPU | Parallel envs | Determinism | Status | Notable limitations |
|---------|----------------|----|-----|---------------|-------------|--------|---------------------|
| MuJoCo | `mujoco==3.6.0` | Ubuntu 22.04 (Docker) | Optional (EGL); CPU works | `num_envs=1` (single process) | `mj_step` is deterministic on a given build; no domain randomization | First-class | Robocasa kitchen scene assembled at launch; H1-2 held upright by an elastic band on `torso_link`. Depth is published as `16UC1` (mm) and additionally as `compressedDepth`/`compressed`. |
| Isaac Lab | Isaac Sim `5.1.0.0` + IsaacLab `v2.3.2` | Ubuntu 22.04 (Docker) | **Required** (PhysX GPU); cu128 wheels include `sm_120` so RTX 30 / 40 / 50 (Blackwell) all work | `num_envs=1` configured, framework supports more | PhysX is GPU-deterministic per seed within same hardware; not bit-exact across GPUs / driver versions | First-class | First boot compiles Kit shaders for 20–40 min. Only one task currently registered (`Isaac-Stack-RgyBlock-H12-27dof-Inspire-Joint`). `sim_main.py` argparse default is the not-registered `Isaac-PickPlace-Cylinder-...`; `launch_isaac.sh` overrides this — don't pass `--task` without setting it. Depth is `32FC1` (metres); no `compressed` variants. Hand cameras (`cl_load_rs` Kit ext) are defined but not loaded by `sim_main.py`. |
| ROS 2 control stack | ROS 2 Humble | Ubuntu 22.04 (Docker) | Required for vision pipeline (CUDA Torch + SAM2/SAM3) | n/a (control side) | n/a | First-class | Bringup expects a sim already running on DDS domain 1; vision pipeline needs `GEMINI_KEY` env var for Gemini-based perception (defaults to `""` stub if unset). |

Build commands:

| Backend | Build | Run |
|---------|-------|-----|
| MuJoCo | `./docker/scripts/docker_build.sh mujoco` | `./docker/scripts/docker_run.sh mujoco` |
| Isaac Lab | `./docker/scripts/docker_build.sh isaac` | `./docker/scripts/docker_run.sh isaac` |
| ROS 2 control | `./docker/scripts/docker_build.sh ros` | `./docker/scripts/docker_run.sh ros` |
| All three | `./docker/scripts/docker_build.sh` (no args) | (run profiles individually) |

Only one sim backend runs at a time (both bind `rt/*` on domain 1).

---

## Architecture

```mermaid
flowchart LR
    subgraph SIM["Sim container (one of)"]
        direction TB
        ISAAC["Isaac Lab<br/>sim_main.py"]
        MJ["MuJoCo<br/>h12_mujoco.py"]
    end

    subgraph BRIDGE["Per-backend interface modules"]
        direction TB
        H12["H12Interface (Isaac)<br/>SimInterface (MuJoCo)<br/>→ rt/lowstate, rt/lowcmd"]
        ROSB_ISAAC["isaac_omnigraph_ros_bridge.py<br/>(OmniGraph + isaacsim.ros2.bridge)"]
        ROSB_MJ["mujoco_ros_bridge.py<br/>(rclpy + cv_bridge)"]
    end

    subgraph DDS["CycloneDDS domain 1"]
        LS["rt/lowstate"]
        LC["rt/lowcmd"]
    end

    subgraph CTRL["ROS 2 container"]
        direction TB
        IK["frame_task_server<br/>(Pinocchio IK)"]
        SAFE["safety_node"]
        VP["vp_node<br/>(YOLO + SAM + Gemini)"]
        RVIZ["rviz2 / wrist_slider_gui"]
    end

    ISAAC --> H12
    MJ --> H12
    ISAAC --> ROSB_ISAAC
    MJ --> ROSB_MJ
    H12 --> LS
    LC --> H12
    ROSB_ISAAC --> CTRL
    ROSB_MJ --> CTRL
    IK --> LC
    LS --> IK
```

**Backend abstraction** — there is no single Python ABC that both Isaac and
MuJoCo implement; each backend has a parallel module that produces the same
DDS wire format and the same ROS 2 topic set:

| Concept | Isaac | MuJoCo |
|---------|-------|--------|
| Sim entry point | `CL_isaaclab_sim/sim_main.py` | `h1_mujoco/h12_mujoco.py` |
| DDS interface | `CL_isaaclab_sim/h12_interface.py` (`H12Interface`) | `h1_mujoco/unitree_interface.py` (`SimInterface`) |
| ROS 2 sensor bridge | `CL_isaaclab_sim/isaac_omnigraph_ros_bridge.py` (no `rclpy`) | `h1_mujoco/mujoco_ros_bridge.py` (`rclpy.Node`) |
| Action provider | `action_provider/h12_dds_action_provider.py` (subclasses `ActionProvider` ABC in `action_provider/action_base.py`) | none — DDS handler writes `data.ctrl` directly |
| Physics step | `RobotController.step()` calls `env.step(action)` at `--step_hz` (default 100 Hz) | Main thread calls `mujoco.mj_step` at `model.opt.timestep` (5 ms = 200 Hz) |
| Scene format | USD (`assets/env_assets/h1_2_*.usd`) | MJCF (`assets/h1_2_handless.xml` + Robocasa kitchen assembled at launch by `h1_mujoco/scene_builder.py`) |
| Joint count | 47 articulated joints (26 body + 21 finger; waist absent in DDS map) | 27 actuators driven by DDS |
| Watchdog | 100 ms timeout → stiff PD pose-hold | 100 ms timeout → stiff PD pose-hold |

**Selection** — there is no runtime backend flag in the application code; the
backend is chosen by *which docker compose profile is launched*
(`docker_run.sh isaac` vs `docker_run.sh mujoco`). Each container only contains
one engine. The two backends are otherwise interchangeable from the
controller's point of view because they publish the same DDS topics on
domain 1.

**Adding a new backend** (e.g. PyBullet, Genesis, Drake):
1. Create `docker/<NewBackend>Dockerfile`. Inherit from `humanoid_sim_base`
   if you need ROS 2 Humble + Python 3.10; otherwise build standalone like
   `IsaacDockerfile` does.
2. Add a `<newbackend>` service block to `docker/docker-compose.yml` with
   `profiles: [<newbackend>]` and an entry in `VALID_PROFILES` of
   `docker/scripts/docker_build.sh`.
3. Implement an interface module that publishes `LowState_` and subscribes to
   `LowCmd_` on CycloneDDS domain 1, mapping motor slots 0..26 to the
   engine's joints (mirror `h1_mujoco/unitree_interface.py` or
   `CL_isaaclab_sim/h12_interface.py`).
4. Implement a sensor bridge that publishes the same ROS 2 topic set as
   `mujoco_ros_bridge.py` / `isaac_omnigraph_ros_bridge.py`.
5. Add `docker/scripts/launch_<newbackend>.sh` and reuse the URDF / mesh
   assets in `assets/`.
6. Match the watchdog: if no `rt/lowcmd` for 100 ms, hold the last joint pose
   with `kp=80`, `kd=3`.

---

## Tech Stack

| Layer | Components |
|-------|------------|
| Languages | Python 3.10 (base / mujoco / ros), Python 3.11 (isaac), C++ (ROS 2 packages, Livox SDK2) |
| Physics | MuJoCo 3.6.0; PhysX 5 via Isaac Sim 5.1.0 |
| RL framework | Isaac Lab `ManagerBasedRLEnv` (`isaaclab` via `isaacsim[all,extscache]`); no separate RL trainer in this repo |
| Control / IK | Pinocchio (apt + pip `pin`), `pink`, `mink`, `pin-pink`, `qpsolvers`, `proxsuite`, `quadprog` |
| Robot SDK | `unitree_sdk2_python` (Unitree HG protocol on CycloneDDS) |
| ROS 2 | Humble — `ros-humble-ros-base`, `rmw_cyclonedds_cpp`, `cv_bridge`, `tf2-ros`, `rviz2`, `pcl-ros` |
| DDS | CycloneDDS 0.10.x built from source, plus PyPI `cyclonedds` wheel pointing at `CYCLONEDDS_HOME` |
| Perception | OpenAI CLIP, Ultralytics YOLO 8.3.52, SAM2 + SAM3, Hugging Face transformers 4.47.1, Google `genai` 1.1.0 (Gemini), Open3D |
| GPU | CUDA 12.2 base; PyTorch cu130 in base / mujoco / ros (Blackwell sm_120 support); PyTorch 2.7.0+cu128 in isaac (cu128 already has sm_120) |
| Container | Docker + `nvidia` runtime + `docker compose` profiles |

**Notable pins** (don't bump casually): `mujoco==3.6.0`, `numpy>=2.2.6` (mujoco)
vs `numpy<2` (ros + isaac transitively), `setuptools==59.6.0`, `wheel<0.44`,
`pip==23` + `setuptools==65` + `cmake<4` (Isaac install staging),
`isaacsim[all,extscache]==5.1.0.0`, `IsaacLab==v2.3.2`, CycloneDDS
`releases/0.10.x`, PyTorch `2.7.0+cu128` (isaac) and `cu130` (others). See
*Configuration* and the Dockerfile comments for why.

---

## Prerequisites

| Requirement | Version | Required for |
|-------------|---------|--------------|
| OS | Ubuntu 22.04 (host or any Linux that runs the NVIDIA container runtime) | All |
| NVIDIA driver | 570.x recommended; 595 has known instability | All GPU-using profiles |
| Docker | with NVIDIA Container Toolkit | All |
| `docker compose` | v2 (subcommand, not `docker-compose`) | All |
| Python | 3.10/3.11 (provided inside containers; not needed on host) | All |
| CUDA | 12.2 in base image; cu130 / cu128 wheels for Torch | All |
| Disk | ~25–30 GB for Isaac image, ~10 GB for MuJoCo, ~15 GB for ROS | Per profile |
| GPU memory | 8 GB+ for Isaac Sim, 2 GB+ for MuJoCo (only for offscreen rendering / EGL) | Isaac always; MuJoCo for headless cam |
| X11 display | Optional, needed for native viewers (Isaac viewport, MuJoCo GLFW, RViz) | Per profile |

---

## Installation

```bash
git clone --recurse-submodules https://github.com/correlllab/Humanoid_Simulation.git
cd Humanoid_Simulation
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

Then choose a backend (or build all three with no arguments). The base image
(`humanoid_sim_base`) is automatically built first whenever `mujoco` or `ros`
is selected, because they inherit from it; `isaac` is self-contained and
does not.

```bash
./docker/scripts/docker_build.sh                    # all (isaac + mujoco + ros)
./docker/scripts/docker_build.sh mujoco             # smallest, fastest
./docker/scripts/docker_build.sh isaac              # GPU sim only
./docker/scripts/docker_build.sh ros                # control stack only
./docker/scripts/docker_build.sh mujoco ros         # any subset
```

Approximate build times on a fast connection: base ~6 min, mujoco ~3 min, ros
~15 min, isaac ~25 min (Isaac Sim wheels are multi-GB).

**Isaac Sim EULA** is auto-accepted via `OMNI_KIT_ACCEPT_EULA=yes` in
`docker/IsaacDockerfile`. By using this repo you accept it.

**No host-side Python install is required** — `assets/`, `CL_isaaclab_sim/`,
`h1_mujoco/`, and `core_ws/` are bind-mounted into the containers at runtime
so code edits take effect without rebuilding.

---

## Configuration

| Name | Where | Type | Default | Purpose | Backend |
|------|-------|------|---------|---------|---------|
| `ROS_DOMAIN_ID` | env / compose | int | `1` | Sets ROS 2 / Cyclone domain. Sims are pinned to 1; real hardware to 0. `docker_run.sh` rewrites `0`/empty → `1` | All |
| `MUJOCO_GL` | env / compose | str | `glfw` (windowed); `egl` (when `--headless`) | OpenGL backend for offscreen vs. windowed rendering | MuJoCo |
| `DISPLAY` | env | str | host `$DISPLAY` | X11 forwarding for native viewers | All viewer modes |
| `XAUTHORITY` | env | path | `$HOME/.Xauthority` | X11 auth | All viewer modes |
| `OMNI_KIT_ACCEPT_EULA` | Dockerfile | str | `yes` | Auto-accept Isaac EULA | Isaac |
| `OMNI_KIT_ALLOW_ROOT` | Dockerfile | int | `1` | Allow Isaac Sim to run as root in the container | Isaac |
| `CYCLONEDDS_HOME` | Dockerfile | path | `/cyclonedds/install` | Points the PyPI cyclonedds wheel at the from-source CycloneDDS 0.10.x | All |
| `RMW_IMPLEMENTATION` | env (base + isaac) | str | `rmw_cyclonedds_cpp` | ROS 2 middleware | All |
| `GEMINI_KEY` | env | str | `""` (stub written by RosDockerfile) | API key for Gemini perception backend | ROS (vision pipeline) |
| `HEADLESS` | env / arg | flag | unset | Force `--headless` even with a display | Isaac, MuJoCo |
| `PYTHONUNBUFFERED` | Dockerfile / launcher | str | `1` (Isaac launcher) | Stream Isaac logs in real time | Isaac |
| `http_proxy` / `https_proxy` | build arg | str | unset | Optional HTTP proxy for the build; cleared in the final layer of `BaseDockerfile` | Build only |

**Sim CLI flags** (`launch_isaac.sh` / `launch_mujoco.sh`):

| Flag | Default | Backend | Meaning |
|------|---------|---------|---------|
| `--task <name>` | `Isaac-Stack-RgyBlock-H12-27dof-Inspire-Joint` | Isaac (`launch_isaac.sh`) | Gym task ID. Note `sim_main.py`'s argparse default (`Isaac-PickPlace-Cylinder-...`) is **not** a registered task; the launcher overrides it. |
| `--reset-cache` | off | Isaac (`launch_isaac.sh`) | Wipes `~/.cache/ov/texturecache` (forces shader recompile) |
| `--headless` | auto-set if no `DISPLAY` | Both | No native viewer; Isaac runs Kit headless, MuJoCo skips the GLFW viewer and switches `MUJOCO_GL=egl` |
| `--step_hz <int>` | `100` | Isaac (`sim_main.py`) | Control loop rate |
| `--physics_dt <float>` | unset (uses task default `0.005`) | Isaac (`sim_main.py`) | Physics dt override |
| `--render_interval <int>` | task default | Isaac (`sim_main.py`) | Frames between renders |
| `--solver_iterations <int>` | task default | Isaac (`sim_main.py`) | PhysX position-iteration count |
| `--no_render` | off | Isaac (`sim_main.py`) | Disables rendering entirely (sets `render_interval=1_000_000`) |
| `--seed <int>` | `42` | Isaac (`sim_main.py`) | RNG seed for env / scene resets |
| `--device cuda` / `--enable_cameras` | always set by `launch_isaac.sh` | Isaac | Forwarded to `AppLauncher` |
| (no `--headless`) | viewer on | MuJoCo | GLFW passive viewer (requires X11) |

**ROS 2 bringup args** — `ros2 launch h1_bringup h1_sim_bringup.launch.py`:

| Argument | Default | Purpose |
|----------|---------|---------|
| `use_rviz` | `true` | Start RViz with `sim.rviz` |
| `rviz_config` | `<bringup>/rviz/sim.rviz` | Override RViz config |
| `config` | `<bringup>/config/sim_network.yaml` | Sets `domain_id: 1` and loosens estop limits for sim |

---

## Usage

### Run a demo

```bash
# MuJoCo (loads the Robocasa kitchen scene; H1-2 held upright by the elastic
# band tether on torso_link):
./docker/scripts/docker_run.sh mujoco

# Isaac Lab (first boot compiles shaders for 20-40 min; cached afterward):
./docker/scripts/docker_run.sh isaac
```

### Switch backends

Stop the running sim (Ctrl-C), then:

```bash
./docker/scripts/docker_run.sh <backend>     # mujoco | isaac
```

Both publish on the same DDS domain 1 with the same `rt/lowstate` /
`rt/lowcmd` topics, so your controller does not need to know which is running.

### Run the ROS 2 control stack

```bash
# Terminal 1: start a sim (mujoco or isaac).
./docker/scripts/docker_run.sh mujoco

# Terminal 2: start the ROS 2 container; launch_ros.sh runs colcon build (if
# needed), sources the overlay, and drops to bash.
./docker/scripts/docker_run.sh ros
# inside the container:
ros2 launch h1_bringup h1_sim_bringup.launch.py
```

This brings up:

1. `robot_state_publisher` (consumes `assets/h1_2_handless_ros.urdf`)
2. `sim_joint_state_publisher` (h1_bringup wrapper that pins DDS domain 1)
3. `frame_task_server` (Pinocchio IK action server, accepts
   `FrameTask.action`) — started 2 s after the others
4. `vp_node` (vision pipeline)
5. `safety_node` (joint-limit / collision guard)
6. `rviz2 -d sim.rviz`
7. `wrist_slider_gui` — cv2 trackbar panel for `left_wrist_yaw_link` and
   `right_wrist_yaw_link` 6-DoF target poses (delayed 3 s so RViz initialises
   Qt before cv2 imports its bundled Qt plugins)

### Run a specific Isaac task

```bash
./docker/scripts/docker_run.sh isaac \
    /home/code/h12_sim_scripts/launch_isaac.sh \
    Isaac-Stack-RgyBlock-H12-27dof-Inspire-Joint
```

(`Isaac-Stack-RgyBlock-H12-27dof-Inspire-Joint` is currently the only
registered task — see *Tasks / Environments*.)

### Drop to a shell

```bash
./docker/scripts/docker_run.sh isaac bash
./docker/scripts/docker_run.sh mujoco bash
./docker/scripts/docker_run.sh ros bash
```

### Train a policy

There is **no training loop in this repository**. `sim_main.py` runs the env
forward and forwards external DDS commands; it does not call `env.step` with
a learned policy. To train, point an external Isaac Lab training script at
`Isaac-Stack-RgyBlock-H12-27dof-Inspire-Joint` (config:
`StackRgyBlockH1227dofInspireBaseFixEnvCfg`).

### Visualize / record video

- **MuJoCo viewer** (default): launches the passive GLFW viewer; press P to
  pause, SPACE to toggle the elastic band, arrow keys to move its anchor,
  `,` / `.` to adjust rest length.
- **Headless + RViz / Foxglove Studio**: omit `DISPLAY` (or pass
  `--headless`); subscribe externally to `/realsense/head/color/image_raw`,
  `/livox/lidar`, `/livox/imu`, `/tf`, `/clock` on `ROS_DOMAIN_ID=1`.
- **Isaac viewport** (default): comes up automatically; `sim_main.py`
  destroys it on shutdown.
- **Recording** is not built in. Use OBS or `ffmpeg` against the X11 display,
  or `ros2 bag record` for sensor data.

### Run in headless mode

Both backends auto-detect when no `DISPLAY` is set and switch to headless
(`MUJOCO_GL=egl` for MuJoCo, `--headless` for Isaac). To force it explicitly:

```bash
HEADLESS=1 ./docker/scripts/docker_run.sh isaac
./docker/scripts/docker_run.sh mujoco /home/code/h12_sim_scripts/launch_mujoco.sh --headless
```

---

## Robot Models & Assets

| File (in `assets/`) | Backend | Purpose |
|--------------------|---------|---------|
| `h1_2.urdf` / `h1_2_ros.urdf` | ROS / MuJoCo | Full H1-2 with Inspire hands |
| `h1_2_handless.urdf` / `h1_2_handless_ros.urdf` | ROS / MuJoCo | H1-2 without hands; `_ros.urdf` adds `camera_link` and `lidar_link` joints |
| `h1_2_handless.xml` | MuJoCo | MJCF body for the handless robot — merged into the Robocasa kitchen at launch |
| `h1_2_*_collision.srdf` / `h1_2_*sphere*.urdf` | Both (via h12_ros2_controller) | Sphere-swept collision proxies |
| `meshes/*.STL` | Both | STL meshes referenced by URDF/MJCF |
| `magpie/*.xml` | MuJoCo (in progress) | UR5e + Magpie eflesh gripper scenes — partial port from upstream Magpie |
| `Payload/` | Isaac (USD) | Misc. payload meshes (USD-format Geometry/Materials/Physics layers) |
| `env_assets/h1_2_26dof_with_inspire_rev_1_0_with_CL_realsense.usd` | Isaac | H1-2 USD with attached RealSense D455 head camera |
| `env_assets/rsd455.usd` | Isaac | Standalone RealSense D455 USD |
| `env_assets/ikea_table_usd/` | Isaac | Table prop |

**Adding a new robot** — drop the URDF/MJCF and meshes into `assets/`,
register an `ArticulationCfg` in `CL_isaaclab_sim/robots/unitree.py` (Isaac)
or include the new MJCF body in a scene (`assets/scene_*.xml`), then either
re-use `H12Interface` / `SimInterface` (if it's a 27-motor variant) or fork
them and adjust `NUM_MOTORS` and the motor → joint index mapping
(`ISAAC_TO_DDS_INDICES` for Isaac).

---

## Tasks / Environments

Isaac Lab tasks are registered via `gym.register` under
`CL_isaaclab_sim/tasks/h1-2_tasks/`.

| Task ID | Robot | Hand | Object(s) | Status |
|---------|-------|------|-----------|--------|
| `Isaac-Stack-RgyBlock-H12-27dof-Inspire-Joint` | H1-2 | Inspire | red / green / yellow blocks on a table | Registered, default in `launch_isaac.sh` |
| `Isaac-PickPlace-Cylinder-...` | — | — | — | **Not registered.** `sim_main.py`'s argparse default still references it; `tasks/__init__.py` blacklists `pick_place`. Don't pass it to `--task`. |

**Stack-RgyBlock environment** —
`StackRgyBlockH1227dofInspireBaseFixEnvCfg`
(`tasks/h1-2_tasks/stack_rgyblock_h12_27dof_inspire/stack_rgyblock_h12_27dof_inspire_joint_env_cfg.py`):

| Manager | Contents |
|---------|----------|
| Scene | `TableRedGreenYellowBlockSceneCfg` — robot (`h12_27dof_inspire_base_fix`, init at `(-4.2, -3.7, 0.76)`, rot `(0.7071, 0, 0, -0.7071)`), table, three colored blocks, head camera |
| Actions | `JointPositionActionCfg(joint_names=[".*"], scale=1.0, use_default_offset=True)` — direct joint position control over all 47 actuated DoFs |
| Observations (`PolicyCfg`) | `robot_joint_state` (DDS-ordered 26-joint pos/vel/torque), `robot_inspire_state` (finger joints), `camera_image`. `enable_corruption=False`, `concatenate_terms=False`. |
| Rewards | `mdp.compute_reward` (single term, weight 1.0) |
| Terminations | `mdp.reset_object_estimate` |
| Sim | `dt=0.005`, `decimation=2`, `episode_length_s=20.0`, PhysX CCD on, 16 position-iters, 4 substeps |
| Resets | Per-block uniform pose perturbation in `[-0.05, +0.05]` m on x/y; `reset_object_self` + `reset_all_self` events registered on a custom `SimpleEventManager` |

**MuJoCo "task"** is a single passive arena: the H1-2 (`h1_2_handless.xml`)
is merged into a Robocasa kitchen at launch by `h1_mujoco/scene_builder.py`
and held upright by an elastic-band tether. There is no manager-based env
layer — "task" semantics live in whatever controller you connect to
`rt/lowcmd`.

---

## API Reference

### Backend-side (sim processes)

`H12Interface(env)` — `CL_isaaclab_sim/h12_interface.py`

```python
from h12_interface import H12Interface

h12 = H12Interface(env)        # binds rt/lowstate, rt/lowcmd on domain 1
action = h12.get_action()      # (1, num_joints) torch tensor (PD command)
h12.shutdown()                 # detaches env, threads exit
```

`SimInterface(model, data, lock=None)` — `h1_mujoco/unitree_interface.py`

```python
from unitree_interface import SimInterface

sim_interface = SimInterface(mujoco_env.model, mujoco_env.data, lock=sim_lock)
# Publishes rt/lowstate every model.opt.timestep; writes incoming rt/lowcmd
# directly to data.ctrl. Watchdog runs at 100 Hz; pose-hold engages after
# 100 ms of silence on rt/lowcmd. No public action-pull method — interaction
# is via DDS topics only.
```

`isaac_omnigraph_ros_bridge` (Isaac) — module-level
`build(env, ...)` and `shutdown()`. No tick callback needed; the OmniGraph
graph runs each playback tick. Topics: `/clock`,
`/realsense/head/color/image_raw` (rgb8),
`/realsense/head/aligned_depth_to_color/image_raw` (32FC1 metres),
`/realsense/head/color/camera_info`, `/livox/lidar`, `/livox/imu`.

`RosSensorBridge` (MuJoCo) — `h1_mujoco/mujoco_ros_bridge.py`. Constructor
takes `(model, data, ...)`; call `tick()` once per sim step from the main
thread (MuJoCo's EGL renderer is thread-affine). Same topic set as the Isaac
bridge plus the additional compressed variants
(`.../image_raw/compressed` jpeg, `.../image_raw/compressedDepth` 16UC1 PNG).
Depth is `16UC1` mm here (vs Isaac's `32FC1` metres).

`ActionProvider` (ABC) — `CL_isaaclab_sim/action_provider/action_base.py`

```python
class ActionProvider(ABC):
    def __init__(self, name: str): ...
    @abstractmethod
    def get_action(self, env) -> Optional[torch.Tensor]: ...
    def start(self) / stop() / cleanup(): ...
```

Concrete: `H12DdsActionProvider(h12_interface)` — wraps `H12Interface` so the
`RobotController` can pull commands once per tick.

### Controller-side (DDS clients)

```python
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_

ChannelFactoryInitialize(1)                               # domain 1 = sim
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(lambda msg: print(msg.tick), 10)

pub = ChannelPublisher("rt/lowcmd", LowCmd_)
pub.Init()
pub.Write(LowCmd_())                                      # 27 motor cmds + IMU
```

DDS topic contract: `LowState_` carries 27 motor states (q, dq, tau_est) +
IMU; `LowCmd_` carries 27 motor commands (`mode`, `q`, `dq`, `tau`, `kp`,
`kd`). Both interfaces use the standard Unitree CRC; messages with bad CRC
are silently dropped on the receiving side.

### Backend interface contract (informal)

A backend module must:

- Initialize CycloneDDS on domain 1.
- Publish `rt/lowstate` (`LowState_` IDL) at the physics tick rate.
- Subscribe to `rt/lowcmd` (`LowCmd_` IDL) and apply
  `tau + kp*(q - q_meas) + kd*(dq - dq_meas)` per motor when `mode == 1`.
- Implement a 100 ms watchdog that snapshots joint positions and applies a
  stiff PD pose-hold (`kp=80`, `kd=3`) when commands stop arriving.
- (Optionally) instantiate a `RosSensorBridge`-equivalent that publishes
  `/clock`, `/realsense/head/color/image_raw`,
  `/realsense/head/aligned_depth_to_color/image_raw`,
  `/realsense/head/color/camera_info`, `/livox/lidar`, `/livox/imu` on the
  default RTPS middleware.

---

## Project Structure

```
Humanoid_Simulation/
├── docker/
│   ├── BaseDockerfile             # CUDA 12.2 + ROS 2 Humble + CycloneDDS 0.10.x +
│   │                              # uv + unitree_sdk2py + Torch cu130 + pin/pink/mink
│   ├── IsaacDockerfile            # Self-contained: CUDA 12.2 + Python 3.11 +
│   │                              # CycloneDDS 0.10.x + Torch 2.7.0+cu128 +
│   │                              # Isaac Sim 5.1.0 + IsaacLab v2.3.2 (pink-ik)
│   ├── MujocoDockerfile           # base → MuJoCo 3.6.0 + EGL/OSMesa
│   ├── RosDockerfile              # base → colcon, rviz2, Livox SDK2, full vision ML stack
│   ├── docker-compose.yml         # 'isaac' / 'mujoco' / 'ros' profiles, GPU + X11 + bind mounts
│   └── scripts/
│       ├── docker_build.sh        # Builds base then selected profiles (default: all)
│       ├── docker_run.sh          # xhost + docker compose run for the chosen profile
│       ├── launch_isaac.sh        # Runs sim_main.py inside the isaac container
│       ├── launch_mujoco.sh       # Runs h12_mujoco.py; auto-headless if no DISPLAY
│       └── launch_ros.sh          # colcon build (idempotent) + drop to bash
│
├── assets/                        # Robot URDFs/MJCFs, meshes, scenes (bind-mounted ro)
│   ├── h1_2*.urdf, *.xml          # H1-2 URDF/MJCF (kitchen scene assembled at launch)
│   ├── meshes/                    # STL meshes
│   ├── magpie/                    # UR5e + Magpie gripper assets (in-progress port)
│   ├── Payload/                   # USD payload layers
│   └── env_assets/                # Isaac USDs (H1-2+RealSense, IKEA table, rsd455)
│
├── CL_isaaclab_sim/               # Isaac Lab task library + sim entry (submodule)
│   ├── sim_main.py                # AppLauncher + env + H12Interface + omnigraph bridge
│   ├── h12_interface.py           # DDS interface — publishes rt/lowstate, applies rt/lowcmd
│   ├── isaac_omnigraph_ros_bridge.py   # OmniGraph ROS publishers (camera, lidar, IMU, clock).
│   │                                   # No rclpy — uses bundled isaacsim.ros2.bridge.
│   ├── robots/unitree.py          # H1-2 + Inspire-hand ArticulationCfg
│   ├── tasks/                     # Manager-based env configs and gym registrations
│   │   ├── h1-2_tasks/stack_rgyblock_h12_27dof_inspire/
│   │   ├── common_config/         # H12RobotPresets, CameraPresets
│   │   ├── common_observations/   # h12_27dof_state, ISAAC_TO_DDS_INDICES, get_robot_imu_data
│   │   ├── common_event/          # custom SimpleEventManager
│   │   ├── common_scene/          # base_scene_stack_rgyblock and friends
│   │   ├── common_rewards/, common_termination/
│   │   └── utils/                 # parse_cfg, hydra glue
│   ├── action_provider/           # ActionProvider ABC + H12DdsActionProvider
│   ├── layeredcontrol/            # RobotController (step_hz, action plumbing)
│   ├── exts/isaac_exts/
│   │   ├── cl_load_rs/            # Hand-camera Kit ext (defined; not loaded by sim_main)
│   │   └── cl_reset_button/       # Reset button Kit ext (in progress; just prints)
│   ├── dds/                       # (empty init module — placeholder)
│   ├── doc/                       # Isaac Sim 4.5 / 5.0 install notes (Chinese + English)
│   └── .isaac_cache/              # Bind-mounted shader/texture cache (gitignored)
│
├── h1_mujoco/                     # MuJoCo H1-2 simulator (submodule)
│   ├── h12_mujoco.py              # Entry: scene load + sim loop + viewer
│   ├── scene_builder.py           # Assembles Robocasa kitchen + H1-2 into one MJCF
│   ├── mujoco_env.py              # MujocoEnv class, ElasticBand
│   ├── unitree_interface.py       # DDS interface, watchdog pose-hold
│   └── mujoco_ros_bridge.py       # rclpy-based ROS 2 sensor publishers
│
└── core_ws/                       # ROS 2 Humble workspace (colcon)
    └── src/
        ├── h1_bringup/            # This repo's launch + rviz + slider GUI + DDS-domain wrappers
        ├── h12_ros2_controller/   # Pinocchio IK + frame_task action server (submodule)
        ├── h12_ros2_model/        # H1-2 URDF + robot_description (submodule)
        ├── h12_safety_layer/      # Joint-limit / e-stop guard (submodule)
        ├── h12_realsense/         # RealSense D455 driver (submodule)
        ├── vision_pipeline/       # YOLO + SAM + Gemini perception (submodule)
        ├── custom_ros_messages/   # FrameTask.action + DDS msg bridges (submodule)
        ├── magpie_control/, magpie_msgs/   # UR5 + Magpie gripper stack (submodule)
        ├── FAST_LIO/              # LiDAR-inertial SLAM (upstream submodule, ROS2 branch)
        ├── livox_ros_driver2/     # Livox MID360 driver (upstream submodule)
        └── unitree_ros2/          # Unitree HG/GO/API ROS 2 message defs (upstream submodule)
```

`core_ws/src/*` are git submodules; **do not patch them in-place** — fix
build / env issues in `docker/RosDockerfile` instead.

---

## Development

There is no host-side Python package — all work happens inside the containers.
Common loops:

```bash
# Edit code on the host (bind-mounted into the container).
$EDITOR CL_isaaclab_sim/sim_main.py

# Re-run inside the container without rebuilding the image:
./docker/scripts/docker_run.sh isaac

# For the ROS workspace, colcon build is idempotent on launch_ros.sh: if any
# package.xml is newer than install/setup.bash, it rebuilds.
./docker/scripts/docker_run.sh ros
# or, inside the running container:
cd /home/code/core_ws && colcon build --symlink-install
```

There is **no host-side lint / format / typecheck / pre-commit configuration**
checked into this repo. Lint and format inside the relevant submodule's own
toolchain if needed.

To rebuild a single profile after Dockerfile changes:

```bash
docker compose -f docker/docker-compose.yml --profile isaac build isaac
```

---

## Testing

There is **no test suite at the top level** of this repository — no `pytest`
config, no GitHub Actions / CI, no automated determinism harness. Per-submodule
`test/` directories exist (e.g. `core_ws/src/h1_bringup/test/`) but contain
only the default ament/colcon scaffolding (`test_copyright`, `test_pep257`,
`test_flake8`).

The recommended manual smoke test:

### 1. DDS — verify `rt/lowstate`

```bash
# Terminal 1: start MuJoCo sim.
./docker/scripts/docker_run.sh mujoco

# Terminal 2: exec into the running container.
docker exec -it humanoid_sim_mujoco bash
source /opt/ros/humble/setup.bash

python3 -c "
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
ChannelFactoryInitialize(1)
sub = ChannelSubscriber('rt/lowstate', LowState_)
sub.Init(lambda msg: print(f'tick={msg.tick}  motor[0].q={msg.motor_state[0].q:.4f}'), 10)
import time; time.sleep(3)
"
# Expected: tick increments, motor[0].q shows a plausible joint angle.
```

### 2. DDS — verify `rt/lowcmd`

```bash
python3 -c "
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
ChannelFactoryInitialize(1)
pub = ChannelPublisher('rt/lowcmd', LowCmd_)
pub.Init()
pub.Write(LowCmd_())
print('lowcmd sent — check sim logs for cmd received')
"
```

### 3. ROS 2 sensor topics

```bash
# Inside the sim container (ROS_DOMAIN_ID=1 is set):
ros2 topic hz /realsense/head/color/image_raw                              # ~10 Hz
ros2 topic echo --once /realsense/head/color/camera_info | head -15        # K matrix non-zero
ros2 topic hz /realsense/head/aligned_depth_to_color/image_raw             # ~10 Hz
ros2 topic hz /livox/imu                                                   # ~100 Hz
ros2 topic hz /livox/lidar                                                 # ~5 Hz
ros2 topic hz /clock                                                       # every sim step
```

Note the per-backend differences:

| Topic | MuJoCo | Isaac |
|-------|--------|-------|
| `/realsense/head/color/image_raw` | `rgb8` | `rgb8` |
| `/realsense/head/color/image_raw/compressed` | jpeg published | **not published** |
| `/realsense/head/aligned_depth_to_color/image_raw` | `16UC1` (mm) | `32FC1` (metres) |
| `.../compressedDepth` | 16UC1 PNG published | **not published** |

### 4. ROS 2 control stack (optional)

```bash
./docker/scripts/docker_run.sh ros
# inside:
ros2 launch h1_bringup h1_sim_bringup.launch.py
# Expected: rviz opens with robot model, TF tree connected,
# frame_task_server starts after ~2s,
# wrist slider GUI appears after ~3s.
```

---

## Reproducibility

- **Seeds** — Isaac: pass `--seed <int>` to `launch_isaac.sh` (default 42).
  MuJoCo: no built-in seed flag (deterministic given a fixed scene and command
  stream; add seeding manually if you need stochastic resets).
- **Determinism** — MuJoCo `mj_step` is deterministic on a given build; PhysX
  GPU is deterministic per seed on the *same* hardware but is **not** bit-exact
  across different GPUs / driver versions.
- **Pinned versions** — every external dep is pinned in the Dockerfiles. Major
  pins to avoid bumping casually:
  - CycloneDDS `releases/0.10.x` (PyPI `cyclonedds` wheel needs the removed
    `dds/ddsi/q_radmin.h`)
  - Isaac Sim `5.1.0.0`, IsaacLab `v2.3.2`, Python `3.11` (Sim 5.x dropped
    Python 3.10 wheels)
  - PyTorch `2.7.0+cu128` in the Isaac container (cu128 has native sm_120 /
    Blackwell; the previous cu130 + ld.so.conf dance is no longer needed for
    Sim 5.1)
  - PyTorch `cu130` in the base / mujoco / ros containers (sm_120 support)
  - `numpy<2` in ROS (apt scipy / pinocchio ABI) and as a transitive
    constraint in the IK pip layer; MuJoCo runs with `numpy>=2.2.6`
  - `pip==23` + `setuptools==65` in the Isaac install staging layer (newer
    pip / setuptools break IsaacLab's `./isaaclab.sh`)
  - `cmake<4` (Isaac install) — egl_probe's `CMakeLists.txt` requires
    `cmake_minimum_required` ≥ 3.5 and cmake 4.0 hard-rejects below that
  - `setuptools==59.6.0` + `wheel<0.44` in ROS (colcon's `literal_eval`
    chokes on `SpecifierSet` from setuptools 70+)
- **Submodule pinning** — `core_ws/src/*` and `CL_isaaclab_sim` / `h1_mujoco`
  are pinned by SHA in `.gitmodules` + the parent commit, so
  `git submodule update --init --recursive` produces a fixed tree.

---

## Troubleshooting

**First Isaac boot stalls for 20–40 min.** Shader compile. Output is cached in
`./CL_isaaclab_sim/.isaac_cache/ov/`, bind-mounted into the container.
Subsequent boots are fast. To wipe: pass `--reset-cache` to `launch_isaac.sh`,
or `rm -rf CL_isaaclab_sim/.isaac_cache/ov/texturecache` on the host.

**`undefined symbol: shm_set_data_state` when starting Isaac.** This was the
old failure mode when `isaacsim.ros2.bridge`'s bundled `librmw_cyclonedds_cpp`
was placed on the global `LD_LIBRARY_PATH`. The current `IsaacDockerfile`
deliberately does **not** prepend that path (see the comment block near the
end). If you re-introduce it manually, expect this crash.

**`Isaac-PickPlace-Cylinder-...` not found.** That task isn't actually
registered in this checkout — `tasks/__init__.py` blacklists `pick_place`
(blocked on Pinocchio compatibility). The only registered task is
`Isaac-Stack-RgyBlock-H12-27dof-Inspire-Joint`, which is what
`launch_isaac.sh` defaults to (overriding `sim_main.py`'s argparse default).

**`MUJOCO WARNING: Nan, Inf or huge value in QACC ...`.** The robot is
penetrating kitchen geometry at startup. Retune the spawn pose in
`h1_mujoco/scene_builder.py` (`SPAWN_POSES[(layout_id, style_id)]`) to a
clear aisle. The elastic-band tether on `torso_link` keeps the robot upright
during the first sim steps, so a stable `rt/lowcmd` stream is not strictly
required, but sending one within 100 ms still avoids the watchdog kicking in.

**`module 'numpy.lib.stride_tricks' has no attribute 'broadcast_to'` in Isaac.**
A transitive dep bumped past `numpy 2.0`. The Isaac stack should resolve to
the wheel that pins it; rebuild the Isaac image, or
`uv pip install --system --force-reinstall "numpy<2"` inside the container if
you've broken it locally.

**Colcon build fails with `literal_eval: malformed node or string` on
setuptools.** Setuptools ≥ 70 returns `SpecifierSet` instead of a string.
Pin `setuptools==59.6.0`, `wheel<0.44` (the ROS Dockerfile already does this
in step 7).

**`no kernel image is available for execution on the device`.** PyTorch was
resolved to a wheel without sm_120 support. The Isaac image installs cu128
(which has sm_120) and the others install cu130 (also sm_120). Verify with
`python3 -c "import torch; print(torch.version.cuda)"` inside the container.

**`xhost: must be on local machine` on remote SSH.** Harmless. The container
still starts. Use `--headless` for fully unattended runs.

**`xhost +local:docker` warning, container still starts.** Expected on remote
sessions. Pass `--headless` to suppress the X11 path entirely.

**NVIDIA driver 595 instability.** Use 570.x.

**`rt/*` traffic visible on host network.** The compose services run with
`privileged: true` and `network_mode: host` for CycloneDDS multicast
discovery. Side effect: DDS traffic is on your host interfaces. Acceptable on
a robotics workstation; not acceptable on a multi-tenant box.

**`livox_ros_driver2` build fails on cmake policy / `colcon build` flag
diffs.** `launch_ros.sh` patches the upstream `build.sh` in-place to inject
`--symlink-install`. The patch is idempotent. If it ever doesn't apply, run
`colcon build --symlink-install` from `/home/code/core_ws` directly inside
the container.

---

## Contributing

The top-level repo is a Docker + asset shell over a stack of submodules. To
contribute:

- **Top-level changes** (Dockerfiles, scripts, scenes, this README,
  `core_ws/src/h1_bringup`): PR against this repo.
- **Backend internals** (`CL_isaaclab_sim/`, `h1_mujoco/`,
  `core_ws/src/h12_*`, `core_ws/src/vision_pipeline`,
  `core_ws/src/magpie_*`, `core_ws/src/custom_ros_messages`): PR against the
  respective `correlllab/<repo>` submodule, then bump the submodule SHA here.
- **Upstream submodules** (`FAST_LIO`, `livox_ros_driver2`, `unitree_ros2`):
  do **not** patch in-place. Work around build / env issues in
  `docker/RosDockerfile` (or `launch_ros.sh`) instead.
- **Adding a new backend**: see *Architecture → Adding a new backend*. The
  acceptance criterion is wire-compatibility with `rt/lowstate` / `rt/lowcmd`
  on domain 1 and the standard ROS 2 sensor topic set.

There is no separate `CONTRIBUTING.md` at this time.

---

## License

The top-level repo does not currently ship a top-level `LICENSE` file. The
submodules carry their own:

| Submodule | License |
|-----------|---------|
| `CL_isaaclab_sim` | Apache 2.0 (Unitree Robotics; see `CL_isaaclab_sim/LICENSE`) |
| `h1_mujoco` | MIT (Correll Lab; see `h1_mujoco/LICENSE`) |
| `core_ws/src/h1_bringup` | MIT (see `core_ws/src/h1_bringup/LICENSE`) |
| `core_ws/src/FAST_LIO` | upstream HKU-MARS license |
| `core_ws/src/livox_ros_driver2` | upstream Livox license |
| `core_ws/src/unitree_ros2` | upstream Unitree license |
| Isaac Sim | NVIDIA Omniverse EULA (auto-accepted by `OMNI_KIT_ACCEPT_EULA=yes` in `docker/IsaacDockerfile`) |
| MuJoCo | Apache 2.0 |

Robot meshes, USDs, and the H1-2 description are derived from Unitree
Robotics; see their public release for redistribution terms.

---

## Acknowledgments

- **Unitree Robotics** — H1-2 robot, `unitree_sdk2_python`, Inspire hand
  description, and reference task scaffolding (much of `CL_isaaclab_sim` is
  derived from Unitree's `unitree_sim_isaaclab`).
- **NVIDIA** — Isaac Sim and Isaac Lab.
- **Google DeepMind** — MuJoCo.
- **HKU-MARS** — FAST-LIO.
- **Livox** — MID360 LiDAR + SDK2.
- **Eclipse CycloneDDS**.
- **Meta AI** — SAM2 / SAM3.
- **OpenAI** — CLIP.
- **Ultralytics** — YOLO.
- **Stéphane Caron / Inria** — Pinocchio, Pink, mink, Pin-Pink.

The Correll Lab maintains the `correlllab/*` submodules (`h1_mujoco`,
`h12_ros2_controller`, `h12_ros2_model`, `h12_safety_layer`,
`h12_realsense`, `vision_pipeline`, `custom_ros_messages`, `magpie_*`,
`CL_isaaclab_sim`).
