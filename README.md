# Humanoid Simulation

Dual-backend simulation framework for Unitree humanoid robots (H1-2, G1), supporting GPU-accelerated training in NVIDIA Isaac Lab and lightweight physics-only simulation in MuJoCo. Both backends publish on the same DDS topics the real robot uses, so controllers run unmodified against either simulator or hardware.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Repository Structure](#repository-structure)
- [Docker Image Layers](#docker-image-layers)
- [Communication Architecture (DDS)](#communication-architecture-dds)
- [Isaac Simulation Backend](#isaac-simulation-backend)
- [MuJoCo Simulation Backend](#mujoco-simulation-backend)
- [ROS 2 Stack](#ros-2-stack)
- [Assets](#assets)
- [Quick Start](#quick-start)
- [Smoke Test](#smoke-test)
- [Known Issues & Notes](#known-issues--notes)

---

## Architecture Overview

```
┌──────────────────────────────┐       DDS Domain 1        ┌──────────────────────────────┐
│    Isaac sim container       │────rt/lowstate────────────►│    External controller       │
│    sim_main.py               │                            │    (ROS2, policy, SDK)       │
│                              │◄───rt/lowcmd──────────────│                              │
│    or                        │                            │    (must also initialize on  │
│                              │                            │     DDS domain 1)            │
│    MuJoCo sim container      │                            │                              │
│    h12_mujoco.py             │                            │                              │
│                              │                            │                              │
│    ROS 2 sensor bridge       │──/realsense/head/color/──►│                              │
│    (ros_bridge.py)           │  /realsense/head/depth/    │                              │
│                              │  /realsense/head/cam_info  │                              │
│                              │  /livox/lidar, /livox/imu  │                              │
│                              │  /clock                    │                              │
└──────────────────────────────┘                            └──────────────────────────────┘
```

Only one sim runs at a time. Both backends use DDS domain 1. Real robot hardware lives on domain 0; the sim is a drop-in replacement for hardware when the controller is switched to domain 1.

---

## Repository Structure

```
Humanoid_Simulation/
├── docker/
│   ├── docker-compose.yml      # 'isaac', 'mujoco', 'ros' profiles, GPU passthrough, volume mounts
│   ├── BaseDockerfile          # Shared base: CUDA 12.2, conda/py3.11, CycloneDDS 0.10.x, unitree_sdk2py
│   ├── IsaacDockerfile         # 2-stage: builder installs Isaac Sim + IsaacLab + teleimager; slim runtime
│   ├── MujocoDockerfile        # 2-stage: builder installs MuJoCo stack; slim runtime
│   ├── RosDockerfile           # apt-based ROS 2 Humble (ros-base + rviz2 + pinocchio) + ML stack + Livox SDK2 (NOT from base; avoids libstdc++ ABI clash)
│   └── scripts/
│       ├── docker_build.sh     # Builds base → selected profiles (edit PROFILES array to choose)
│       ├── docker_run.sh       # xhost + docker compose run for the selected profile
│       ├── launch_isaac.sh     # Entry point inside Isaac container: runs sim_main.py
│       ├── launch_mujoco.sh    # Entry point inside MuJoCo container: runs h12_mujoco.py
│       └── launch_ros.sh       # Entry point inside ROS container: colcon build + source + bash
├── assets/
│   ├── h1_2_handless.urdf      # H1-2 (no hands) URDF for MuJoCo
│   ├── h1_2_handless_ros.urdf  # H1-2 (no hands) URDF for ROS (camera_link, lidar_link included)
│   ├── h1_2_handless.xml       # H1-2 (no hands) MJCF body
│   ├── scene_handless*.xml     # MuJoCo scenes (free base + pelvis-fixed)
│   ├── meshes/                 # STL meshes referenced by URDF/MJCF
│   ├── magpie/                 # Magpie gripper assets
│   └── env_assets/             # Isaac-specific USDs: RealSense D455, IKEA table, H1-2 w/ camera
├── CL_isaaclab_sim/            # Core Isaac task library (tasks, robots, DDS, action providers) — submodule
│   ├── exts/cl_load_rs/        # RealSense camera Kit extension, loaded by sim_main.py at runtime
│   └── .isaac_cache/           # Gitignored. Isaac shader + texture cache, bind-mounted into container.
│                               # Persists across runs so first-boot shader compile (~20-40 min) is one-time.
├── h1_mujoco/                  # MuJoCo H1-2 simulator (submodule: correlllab/h1_mujoco@unified_sim)
├── core_ws/                    # ROS 2 Humble workspace (colcon). All src/ entries are submodules.
│   └── src/
│       ├── h1_bringup/         # This repo's glue: launch files, rviz config, sim wrappers
│       ├── h12_ros2_controller # Pinocchio IK + frame_task action server
│       ├── h12_ros2_model      # H1-2 URDF + robot_description
│       ├── h12_safety_layer    # Joint-limit / collision guard
│       ├── h12_realsense       # RealSense D455 driver wrapper
│       ├── vision_pipeline     # YOLO + Gemini-based perception
│       ├── custom_ros_messages # FrameTask.action + DDS msg bridges
│       ├── magpie_{control,msgs} # Dexterous gripper + UR5 stack
│       ├── FAST_LIO            # LiDAR-inertial SLAM (upstream, ROS2 branch)
│       └── livox_ros_driver2   # Livox MID360 driver (upstream)

Third-party dependencies (not tracked as submodules — cloned inside the Docker images at build time):
    - cyclonedds        (eclipse-cyclonedds, pinned to releases/0.10.x)       built in docker/BaseDockerfile
    - unitree_sdk2py    (unitreerobotics/unitree_sdk2_python)                 installed in docker/BaseDockerfile
    - IsaacLab          (isaac-sim/IsaacLab)                                  installed in docker/IsaacDockerfile
    - teleimager        (unitreerobotics/teleimager, WebRTC camera streaming) installed in docker/IsaacDockerfile
```

---

## Docker Image Layers

Build order: `docker/BaseDockerfile` → `docker/IsaacDockerfile` or `docker/MujocoDockerfile`. The base image is shared; both simulators pull from it.

### Base (`humanoid_sim_base`)

| Layer | Contents |
|-------|----------|
| `nvidia/cuda:12.2.0-devel-ubuntu22.04` | CUDA toolkit, gcc-12, Vulkan, OpenGL/EGL |
| Miniconda | Python 3.11 in `humanoid_sim_env` conda env |
| CycloneDDS | Built from source, pinned to `releases/0.10.x`. Post-0.10 removes `dds/ddsi/q_radmin.h`, which the PyPI `cyclonedds` wheel (a transitive dep of `unitree_sdk2py`) still needs to compile. Installed to `/cyclonedds/install` (`CYCLONEDDS_HOME`). |
| unitree_sdk2py | Pip-installed, then source tree copied over the install so sub-packages (e.g. `b2`) present in `__init__.py` are actually on disk. Upstream `setup.py` only declares the top-level package. |
| ROS 2 Humble | Via RoboStack conda channel: `rclpy`, `sensor_msgs_py`, `tf2_ros`, `geometry_msgs`. Shared by both simulators for sensor/TF publishing. |

### Isaac (`humanoid_sim_isaac`)

**Stage 1 — builder** (from base):
1. PyTorch `cu130` — required for RTX 50-series support.
2. Isaac Sim 5.1.0 from NVIDIA PyPI (`isaacsim[all,extscache]`). Bumped from 5.0.0 for `wp.transform_compose` in Isaac Lab's `fabric.py`.
3. `pip==23` + `setuptools==65` downgrade — IsaacLab install scripts break on newer pip/setuptools.
4. IsaacLab cloned from `isaac-sim/IsaacLab` HEAD, installed via `./isaaclab.sh --install`.
5. `CL_isaaclab_sim/requirements.txt` (`rerun-sdk`, `pyzmq`, `onnxruntime`, `pynput`).
6. teleimager (WebRTC camera streaming). `pyproject.toml` is patched to allow Python 3.11; installed non-editable so the source tree can be deleted.

**Stage 2 — runtime** (`cuda:12.2.0-runtime-ubuntu22.04`, slim):
- Copies `/home/code`, `/cyclonedds`, `/opt/conda` from the builder.
- Sets `OMNI_KIT_ALLOW_ROOT=1`, `OMNI_KIT_ACCEPT_EULA=yes`.
- `LD_LIBRARY_PATH` includes the Isaac Sim ROS2 bridge libs.
- Assets and `CL_isaaclab_sim` are mounted at runtime by docker-compose (not baked in), so code edits don't require a rebuild.

### MuJoCo (`humanoid_sim_mujoco`)

**Stage 1 — builder**: Installs `mujoco==3.6.0` + `numpy`. `unitree-sdk2py` and ROS 2 are already installed in the base image.

**Stage 2 — runtime**: Same slim cuda:12.2.0-runtime base. Hardcoded `MUJOCO_GL=egl` — MuJoCo runs headless (no GLFW viewer); visualize externally via RViz/Foxglove over ROS 2.

### ROS (`humanoid_sim_ros`)

Single-stage apt-based Ubuntu 22.04 + ROS 2 Humble. **Deliberately does NOT inherit from base** — the conda `libstdc++` in the base image clashes with the apt-built `colcon`/`rviz2` ABI. Includes `ros-humble-ros-base`, `ros-humble-rviz2`, `ros-humble-pinocchio`, `ros-humble-rmw-cyclonedds-cpp`, and builds Livox SDK2 from source. Uses `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` and `ROS_DOMAIN_ID=1` so it interoperates with the sims on DDS domain 1. Also includes a full ML stack for the vision pipeline: PyTorch cu130, transformers, ultralytics (YOLO), CLIP, SAM2/SAM3, and Google Genai.

---

## Communication Architecture (DDS)

All inter-process communication uses **CycloneDDS**, matching the protocol the real Unitree robots use. This means the same controller binary can target either the simulator or hardware with no modification — you just re-init the controller on a different DDS domain.

### DDS Domains

| Domain | Who lives here |
|--------|---------------|
| **0** | Real robot hardware |
| **1** | Simulators (Isaac or MuJoCo), and any controller running against the sim |

A controller can only talk to one domain at a time. To swap between sim and hardware, restart the controller with a different domain.

### Topics

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `rt/lowstate` | `LowState_` | sim → controller | Body joint positions, velocities, IMU |
| `rt/lowcmd` | `LowCmd_` | controller → sim | Body joint position/torque commands |

### Action Source

`sim_main.py` uses the DDS action provider (`H12DdsActionProvider`) — live joint commands from an external controller via `rt/lowcmd` on CycloneDDS domain 1.

---

## Isaac Simulation Backend

### Entry point

`docker/scripts/launch_isaac.sh` is the container entrypoint for the `isaac` compose profile. It activates conda, sets `PYTHONUNBUFFERED=1` (so `docker logs` stream in real time), and runs `sim_main.py` with the PickPlace-Cylinder task for H1-2 + Inspire hand. Pass a task name as a positional arg to override. `--reset-cache` wipes `~/.cache/ov/texturecache` (which is bind-mounted to host `CL_isaaclab_sim/.isaac_cache/ov/`, so this deletes host files). `--headless` runs without a viewport (auto-enabled if no `DISPLAY` is set).

`sim_main.py` accepts additional tuning flags: `--step_hz` (control loop rate, default 100), `--physics_dt`, `--render_interval`, `--solver_iterations`, `--no_render`, `--seed`.

### Task library (`CL_isaaclab_sim/tasks/`)

Tasks follow Isaac Lab's manager-based env pattern. Each task defines:

| Component | What it configures |
|-----------|-------------------|
| `ObjectTableSceneCfg` | Robot USD, table, task object, cameras, lighting |
| `ActionsCfg` | Joint position control groups (body / hand) |
| `ObservationsCfg` | What the policy receives (joint states, images) |
| `RewardsCfg` | Shaped reward functions |
| `TerminationsCfg` | Episode-end conditions |

**Available tasks** (registered via `gym.register`):

| Task name | Robot | Hand | Object |
|-----------|-------|------|--------|
| `Isaac-PickPlace-Cylinder-H12-27dof-Inspire-Joint` | H1-2 | Inspire | Cylinder |
| `Isaac-PickPlace-RedBlock-H12-27dof-Inspire-Joint` | H1-2 | Inspire | Red block |
| `Isaac-Stack-RgyBlock-H12-27dof-Inspire-Joint` | H1-2 | Inspire | RGB blocks |

### Cameras

RealSense D455 cameras are placed at the chest and both wrists (1280×720). The `cl_load_rs` Kit extension under `CL_isaaclab_sim/exts/cl_load_rs/` is loaded at runtime by `sim_main.py` to provide the RealSense asset; `teleimager` serves frames over WebRTC.

---

## MuJoCo Simulation Backend

Lighter-weight alternative for headless testing or CI. Uses MuJoCo 3.6.0 instead of Isaac Sim; runs headless (no native viewer) and publishes `rt/lowstate` on DDS domain 1 plus ROS 2 sensor topics and TF.

### Entry point

`docker/scripts/launch_mujoco.sh` (compose default command) runs `h12_mujoco.py`. Flags are forwarded:

| Flag | Meaning |
|------|---------|
| `--fixed` | Pin pelvis to world (no balance needed). Loads `scene_handless_pelvis_fixed.xml` instead of `scene_handless.xml`. **This is the default when no args are given** (the free-standing scene has startup velocity transients that trip the safety estop). |
| `--force link1 link2` | Enable external-force interface on the named links. |
| `--viewer` | Launch with MuJoCo's passive GLFW viewer (requires X11 display; auto-sets `MUJOCO_GL=glfw`). |

### Visualization

By default the sim runs fully headless on EGL (the GLFW viewer and EGL can't share a GL context). Use `--viewer` to get the GLFW passive viewer, or visualize externally via **RViz** or **Foxglove Studio** over ROS 2:
- Fixed frame: `world`
- Displays: Image (`/realsense/head/color/image_raw`), Camera (with `/realsense/head/color/camera_info`), PointCloud2 (`/livox/lidar`), Imu (`/livox/imu`)
- TF is published by `robot_state_publisher` + `sim_joint_state_publisher` in the ROS container (not by the sim itself)

### ROS 2 topics

Defaults `ROS_DOMAIN_ID=1` to match the Unitree DDS domain.

| Topic | Type | Rate | Frame |
|-------|------|------|-------|
| `/clock` | `rosgraph_msgs/Clock` | every step | — |
| `/realsense/head/color/image_raw` | `sensor_msgs/Image` (rgb8) | 10 Hz | `camera_link` |
| `/realsense/head/color/image_raw/compressed` | `sensor_msgs/CompressedImage` (jpeg) | 10 Hz | `camera_link` |
| `/realsense/head/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` (16UC1 mm) | 10 Hz | `camera_link` |
| `/realsense/head/aligned_depth_to_color/image_raw/compressedDepth` | `sensor_msgs/CompressedImage` (16UC1 png) | 10 Hz | `camera_link` |
| `/realsense/head/color/camera_info` | `sensor_msgs/CameraInfo` | 10 Hz | `camera_link` |
| `/livox/lidar` | `sensor_msgs/PointCloud2` | 5 Hz | `lidar_link` |
| `/livox/imu` | `sensor_msgs/Imu` (co-located with lidar) | 100 Hz | `lidar_link` |

Mount poses are defined in [assets/h1_2_handless_ros.urdf](assets/h1_2_handless_ros.urdf) (`camera_joint`, `livox_joint` → `camera_link`, `lidar_link`) and mirrored in the MJCF.

Camera and LiDAR rates/resolutions are tunable via `ros_bridge.py` constructor kwargs — reduce for CI or slower hosts (defaults: camera 640×480 @ 10 Hz, lidar 72×12 rays @ 5 Hz, IMU 100 Hz).

### MJCF models (`assets/`)

| File | Description |
|------|-------------|
| `scene_handless.xml` | H1-2 without hands (default) |
| `scene_handless_pelvis_fixed.xml` | H1-2 without hands, pelvis pinned to world |
| `h1_2_handless.xml` | Robot body definition (no hands) |

---

## ROS 2 Stack

The `ros` compose profile runs a separate container that builds `core_ws/` via `colcon`, sources `install/setup.bash`, and drops into a bash shell. It's the control-side counterpart to the sim containers — controllers, perception, and visualization live here while the sim runs in its own container. All communication is via DDS on domain 1.

### Top-level launch

[core_ws/src/h1_bringup/launch/h1_sim_bringup.launch.py](core_ws/src/h1_bringup/launch/h1_sim_bringup.launch.py) brings up a complete control stack against a running sim:

- `robot_state_publisher` + `sim_joint_state_publisher` (wrapper in h1_bringup that initializes DDS domain 1 before calling upstream `joint_state_publisher`)
- `frame_task_server` (pinocchio IK action server; accepts `FrameTask.action` goals, config: `sim_network.yaml` with `domain_id: 1`)
- `vp_node` (vision pipeline)
- `safety_node` (joint limits)
- `rviz2` with `sim.rviz` — single rviz window (upstream launches' own rviz invocations are inlined-out here, since the submodules are read-only)
- `wrist_slider_gui` — cv2 trackbar panel to nudge `left_wrist_yaw_link` / `right_wrist_yaw_link` target poses (x/y/z/rpy, world frame) via `FrameTask` goals. Delayed 3s so rviz2 initializes Qt before cv2 imports its bundled plugins (otherwise Qt conflict blocks rviz).

Launch args: `use_rviz` (default `true`), `rviz_config` (default `sim.rviz`), `config` (default `sim_network.yaml` — sets `domain_id: 1` for `frame_task_server`).

### Packages (all read-only submodules)

| Package | Role |
|---------|------|
| `h1_bringup` | This repo's launch/rviz/slider glue + DDS domain wrappers (`sim_joint_state_publisher`) |
| `h12_ros2_controller` | Pinocchio-based IK, `frame_task` action server, controller core |
| `h12_ros2_model` | URDF + robot description |
| `h12_safety_layer` | Joint-limit & collision guard |
| `h12_realsense` | RealSense D455 driver wrapper |
| `vision_pipeline` | YOLO fine-tuning + Gemini perception |
| `custom_ros_messages` | Action/msg defs (`FrameTask`, `LowState` bridges, etc.) |
| `magpie_control`, `magpie_msgs` | Dexterous gripper + UR5 arm stack |
| `FAST_LIO` | LiDAR-inertial SLAM (upstream, ROS2 branch) |
| `livox_ros_driver2` | Livox MID360 driver (upstream) |

Upstream submodules must not be patched in-place — fix build/env issues in the Dockerfile instead.

---

## Assets

Assets are kept outside the Docker images (too large to bake in) and mounted at runtime:

| Path | Contents | Mounted as |
|------|----------|------------|
| `assets/` (URDF/MJCF + meshes) | Shared robot geometry for MuJoCo scenes and ROS bringup | `/home/code/assets` (MuJoCo ro, ROS ro) |
| `assets/env_assets/` | Isaac USDs: H1-2 with RealSense, RealSense D455, IKEA table | `/home/code/cl_assets` (Isaac only) |
| `h1_mujoco/` | `h12_mujoco.py` and MuJoCo-specific Python (submodule: correlllab/h1_mujoco@unified_sim) | `/home/code/h1_mujoco` (MuJoCo only) |
| `CL_isaaclab_sim/` | Task library + `exts/cl_load_rs` | `/home/code/CL_isaaclab_sim` (Isaac only) |
| `docker/scripts/` | Launchers (build, run, per-sim entry points) | `/home/code/h12_sim_scripts` (all containers) |
| `CL_isaaclab_sim/.isaac_cache/` | Shader/texture cache | `/root/.cache/ov` + `/root/.nv` (Isaac only) |

The handless URDF/MJCF and meshes at the top of `assets/` are committed; they're referenced directly by the MuJoCo scene files at runtime.

---

## Quick Start

```bash
# Clone and initialize submodules
git clone --recurse-submodules https://github.com/correlllab/Humanoid_Simulation.git
cd Humanoid_Simulation

# Build images. Edit docker_build.sh to select which profiles to build.
# By default it builds base + isaac. Uncomment lines for mujoco/ros as needed.
# First build takes 30–60 min.
./docker/scripts/docker_build.sh

# Run Isaac simulation (auto-starts launch_isaac.sh; first boot compiles shaders, 20–40 min)
./docker/scripts/docker_run.sh isaac

# Run MuJoCo simulation (auto-starts launch_mujoco.sh; defaults to pelvis-fixed scene)
./docker/scripts/docker_run.sh mujoco

# Run the ROS 2 control stack against a running sim (rviz + frame_task_server + wrist slider GUI)
./docker/scripts/docker_run.sh ros
# then inside the container:
ros2 launch h1_bringup h1_sim_bringup.launch.py

# Debug: drop to a shell inside the container
./docker/scripts/docker_run.sh isaac bash

# Run a specific launcher with args in one shot
./docker/scripts/docker_run.sh isaac /home/code/h12_sim_scripts/launch_isaac.sh Isaac-Stack-RgyBlock-H12-27dof-Inspire-Joint
./docker/scripts/docker_run.sh mujoco /home/code/h12_sim_scripts/launch_mujoco.sh --fixed
```

**GPU driver note**: NVIDIA driver 595 has known instability issues. Use driver 570.x.

**Controller note**: Configure your controller to initialize DDS on domain 1 (`ChannelFactoryInitialize(1)`) when talking to a sim.

---

## Smoke Test

Quick sanity checks to verify each communication channel. Start the sim in one terminal, then run checks from a second shell inside the same container (or the ROS container for ROS topics).

### 1. DDS: verify `rt/lowstate`

```bash
# Terminal 1: start MuJoCo sim
./docker/scripts/docker_run.sh mujoco

# Terminal 2: exec into the running container
docker exec -it $(docker ps -qf "ancestor=humanoid_sim_mujoco") bash
source /opt/conda/etc/profile.d/conda.sh && conda activate humanoid_sim_env

python3 -c "
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
ChannelFactoryInitialize(1)
sub = ChannelSubscriber('rt/lowstate', LowState_)
sub.Init(lambda msg: print(f'tick={msg.tick}  motor[0].q={msg.motor_state[0].q:.4f}'), 10)
import time; time.sleep(3)
"
# Expected: tick increments, motor[0].q shows a plausible joint angle
```

### 2. DDS: verify `rt/lowcmd`

```bash
# In the same exec shell (sim already running):
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

### 3. ROS 2 topics (image, IMU, lidar, clock)

```bash
# From the sim container shell (ROS_DOMAIN_ID=1 is already set):
# Image (expect ~10 Hz, encoding=rgb8):
ros2 topic hz /realsense/head/color/image_raw
ros2 topic echo --once /realsense/head/color/image_raw --no-arr | head -10

# Compressed image (expect ~10 Hz, format=jpeg):
ros2 topic hz /realsense/head/color/image_raw/compressed

# Depth (expect ~10 Hz, encoding=16UC1):
ros2 topic hz /realsense/head/aligned_depth_to_color/image_raw

# Camera info (K matrix non-zero, width/height match image):
ros2 topic echo --once /realsense/head/color/camera_info | head -15

# IMU (expect ~100 Hz, orientation + angular_velocity + linear_acceleration):
ros2 topic hz /livox/imu
ros2 topic echo --once /livox/imu | head -20

# LiDAR (expect ~5 Hz, point_step > 0):
ros2 topic hz /livox/lidar
ros2 topic echo --once /livox/lidar --no-arr | head -10

# Clock (every sim step):
ros2 topic hz /clock
```

### 4. ROS 2 control stack (optional — requires ros container)

```bash
# Terminal 3: start the ROS container (sim must already be running)
./docker/scripts/docker_run.sh ros
# Inside the container:
ros2 launch h1_bringup h1_sim_bringup.launch.py
# Expected: rviz opens with robot model, TF tree connected, wrist slider GUI appears after ~3s
```

---

## Known Issues & Notes

- **First-boot shader compile**: Isaac Sim takes 20–40 minutes the first time. Output is cached to `./CL_isaaclab_sim/.isaac_cache/ov/` (bind-mounted into the container), so subsequent boots are much faster. To wipe it and force a recompile, delete `CL_isaaclab_sim/.isaac_cache/ov/texturecache/` or pass `--reset-cache` to `launch_isaac.sh`.
- **RTX 50 series**: Needs PyTorch `cu130`. Already pinned in all Dockerfiles.
- **CycloneDDS pin**: Do not bump past `releases/0.10.x` — post-0.10 removes a header (`dds/ddsi/q_radmin.h`) that the PyPI `cyclonedds` wheel still expects.
- **Isaac pip/setuptools pin**: Isaac image pins `pip==23` and `setuptools==65` — IsaacLab install scripts break on newer versions.
- **ROS setuptools pin**: ROS image pins `setuptools==59.6.0` — newer versions (69+) return `SpecifierSet` instead of a string from `setup.py`, which breaks colcon's `literal_eval` during workspace builds.
- **ROS numpy pin**: `numpy<2` in the ROS container — system scipy and pinocchio are built against numpy 1.x ABI.
- **Privileged containers**: All compose services run with `privileged: true` and `network_mode: host`. This is required for CycloneDDS multicast discovery; it also means the sim shares the host's network namespace (so DDS traffic appears on your host interfaces).
- **X11 forwarding**: `xhost: must be on local machine` warnings are harmless for local use; `xhost +local:docker` may fail on remote sessions but the container still starts.
