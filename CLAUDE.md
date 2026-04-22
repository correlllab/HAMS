# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Dual-backend simulation framework for Unitree humanoid robots (H1-2, G1). Both an Isaac Lab backend (GPU training) and a MuJoCo backend (lightweight/headless) publish on the same CycloneDDS topics (`rt/lowstate`, `rt/lowcmd`) on **domain 1** that the real robot uses on domain 0. Controllers run unmodified against either simulator or hardware.

## Build & Run

Everything runs in Docker. There is no local build — all dependencies live in container images.

```bash
# Build images (base → selected profiles). Edit PROFILES array in script to choose.
# Default: base + isaac. First build takes 30-60 min.
./docker/scripts/docker_build.sh

# Run a simulator
./docker/scripts/docker_run.sh isaac          # Isaac Lab (first boot: 20-40 min shader compile)
./docker/scripts/docker_run.sh mujoco         # MuJoCo (headless, visualize via RViz)

# Run the ROS 2 control stack (in a separate terminal, against a running sim)
./docker/scripts/docker_run.sh ros
# then: ros2 launch h1_bringup h1_sim_bringup.launch.py

# Drop to a shell inside any container
./docker/scripts/docker_run.sh isaac bash
./docker/scripts/docker_run.sh mujoco bash

# Run with specific args
./docker/scripts/docker_run.sh mujoco /home/code/h12_sim_scripts/launch_mujoco.sh --fixed
./docker/scripts/docker_run.sh isaac /home/code/h12_sim_scripts/launch_isaac.sh Isaac-Stack-RgyBlock-H12-27dof-Inspire-Joint
```

There are no tests or linters configured in this repository.

## Architecture

### Docker Image Hierarchy

```
BaseDockerfile (CUDA 12.2, conda py3.11, CycloneDDS 0.10.x, unitree_sdk2py, ROS 2 Humble via RoboStack)
├── IsaacDockerfile  (2-stage: Isaac Sim 5.1.0 + IsaacLab + teleimager)
├── MujocoDockerfile (2-stage: mujoco 3.6.0)
└── RosDockerfile    (standalone apt-based, NOT from base — conda libstdc++ ABI clashes with colcon)
```

### Sim Entry Points

- **Isaac**: `CL_isaaclab_sim/sim_main.py` — creates gymnasium env, H12Interface (DDS), RosSensorBridge, loads RealSense extension
- **MuJoCo**: `h1_mujoco/h12_mujoco.py` — loads MJCF scene, runs `mj_step()` loop, publishes DDS + ROS topics

Both simulators have parallel files: `h12_interface.py` / `unitree_interface.py` (DDS pub/sub) and `ros_bridge.py` (ROS 2 sensor topics, TF, clock).

### DDS Communication

Domain 0 = real robot, Domain 1 = simulators. Topics: `rt/lowstate` (sim→controller), `rt/lowcmd` (controller→sim). Uses `unitree_sdk2py.idl.unitree_hg.msg.dds_` types. Any new node talking to a sim must call `ChannelFactoryInitialize(id=1)`.

### ROS 2 Workspace (`core_ws/`)

Built by `colcon` inside the ROS container. The bringup launch is `core_ws/src/h1_bringup/launch/h1_sim_bringup.launch.py` which starts robot_state_publisher, sim_joint_state_publisher, frame_task_server (Pinocchio IK), vision_pipeline, safety_layer, rviz2, and wrist_slider_gui.

### Submodule Layout

Most of the codebase is submodules. `h1_bringup` (under `core_ws/src/`) is the only ROS package owned by this repo — it contains launch files, rviz config, and DDS domain wrapper nodes.

## Critical Constraints

- **Never edit files in `core_ws/src/` submodules** — all packages there are upstream. Fix build issues in Dockerfiles, not source.
- **CycloneDDS must stay on `releases/0.10.x`** — post-0.10 removes `dds/ddsi/q_radmin.h` needed by the PyPI cyclonedds wheel.
- **Isaac image pins `pip==23` + `setuptools==65`** — IsaacLab install breaks on newer versions.
- **ROS container is apt-based (not conda)** — avoids libstdc++ ABI clash with colcon/rviz2.
- **`numpy<2` in ROS container** — scipy/pinocchio ABI compatibility.
- **PyTorch `cu130`** in all images — required for RTX 50-series GPUs.

## Key Paths

| Path | Purpose |
|------|---------|
| `docker/docker-compose.yml` | Profiles: `isaac`, `mujoco`, `ros`. GPU passthrough, volume mounts |
| `docker/scripts/launch_*.sh` | Container entrypoints (conda activate, env vars, run sim) |
| `assets/` | Robot URDFs, MJCFs, meshes, scene XMLs |
| `CL_isaaclab_sim/tasks/` | Isaac Lab task definitions (PickPlace, Stack variants) |
| `CL_isaaclab_sim/exts/cl_load_rs/` | RealSense camera Kit extension |
| `CL_isaaclab_sim/.isaac_cache/` | Shader cache (bind-mounted, gitignored) |
| `core_ws/src/h1_bringup/` | This repo's ROS package: launch, rviz config, sim wrappers |
| `core_ws/src/h1_bringup/config/sim_network.yaml` | frame_task_server config (domain_id: 1) |
