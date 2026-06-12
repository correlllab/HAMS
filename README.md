# Humanoid_Simulation

Humanoid sim stack for the Correll Lab H1 robot: MuJoCo, ROS 2, and Isaac Sim
running in separate containers and sharing a CycloneDDS ROS domain.

## Layout

- `docker/` — Dockerfiles, `docker-compose.yml`, and the build/run scripts.
- `core_ws/` — ROS 2 workspace (bringup, IK, perception, safety). Submodules.
- `h1_mujoco/` — MuJoCo simulator entry point and bridges.
- `CL_Assets/` — URDF, MuJoCo XML, and Isaac USD assets.

## Prerequisites

- Docker (with Compose v2) and the NVIDIA Container Toolkit.
- Git LFS (`git lfs install`) — required to fetch the large binary assets
  (URDF meshes, MuJoCo XML, Isaac USD) tracked via LFS.
- `git submodule update --init --recursive` to populate `core_ws/src`.
- Copy `docker/.env.example` to `docker/.env` and fill in your `GEMINI_API_KEY`
  and `ROS_DOMAIN_ID` (see [Configuration](#configuration)).

A few things worth knowing before you run anything:

- The build/run scripts can be invoked from any directory — they resolve their
  own location, so `docker/scripts/docker_run.sh mujoco` works just as well
  from `/tmp` as from the repo root.
- `ROS_DOMAIN_ID` is read from `docker/.env` and forwarded into the MuJoCo and
  ROS containers. If it is unset or `0`, it is normalized to `1` (domain 0 is
  reserved for the real robot).
- Isaac currently overrides this and pins its DDS bridge to channel 1 — see
  the Isaac section below.

## Configuration

Runtime settings live in `docker/.env` (git-ignored — it holds your API key).
Copy the template once and edit it; you never need to `export` these variables
in your shell:

```bash
cp docker/.env.example docker/.env
# then edit docker/.env:
#   GEMINI_API_KEY=...   # https://aistudio.google.com/apikey
#   ROS_DOMAIN_ID=1      # any non-zero value; 0 is reserved for the real robot
```

Both `docker compose` and `docker/scripts/docker_run.sh` load `docker/.env`
automatically, so every container and every terminal sees the same values.
`ROS_DOMAIN_ID` is passed to all containers; `GEMINI_API_KEY` is passed only to
the `ros` container (the vision pipeline's Gemini backbone and `h12_skills` need
it). Because the run scripts `source` the file, `docker/.env` takes precedence
over any value left exported in your shell.

## Build the containers

```bash
docker/scripts/docker_build.sh             # all three
docker/scripts/docker_build.sh mujoco ros  # subset
docker/scripts/docker_build.sh isaac       # isaac only
```

The MuJoCo and ROS images both inherit from `humanoid_sim_base`, which is
built first automatically when either profile is selected. Isaac is
self-contained and does not use the base.

## Run the MuJoCo container

```bash
docker/scripts/docker_run.sh mujoco            # windowed viewer
docker/scripts/docker_run.sh mujoco --headless # no DISPLAY / SSH / CI
docker/scripts/docker_run.sh mujoco bash       # drop to a shell instead

# to use a custom ROS domain (e.g. several devs on one network),
# set ROS_DOMAIN_ID in docker/.env (see Configuration above)
```

Once it's up, MuJoCo publishes `rt/lowstate` over CycloneDDS plus
`/head/color/image_raw`, `/head/depth/image_raw`, `/head/color/camera_info`,
`/lidar/points`, and `/tf` on the chosen ROS domain (default `1`).

## Run the ROS container and bringup

The ROS launcher only builds the workspace and drops to a shell, so bringup
is a manual step. `ROS_DOMAIN_ID` (all containers) and `GEMINI_API_KEY` (the
`ros` container, for the vision pipeline's Gemini backbone and `h12_skills`)
both come from `docker/.env`, so there is nothing to export — just open two
terminals:

```bash
# terminal A — start MuJoCo first so /clock is publishing
docker/scripts/docker_run.sh mujoco

# terminal B — ROS workspace shell (auto-builds core_ws on first run)
docker/scripts/docker_run.sh ros

# inside the ROS container
ros2 launch h1_bringup h1_sim_bringup.launch.py
```

Bringup starts `joint_state_publisher`, `robot_state_publisher`, the
`frame_task_server` IK solver, the `safety_node`, and `rviz2`.

## Isaac container — WIP / TODO

The Isaac profile builds and runs the same way as the others:

```bash
docker/scripts/docker_build.sh isaac
docker/scripts/docker_run.sh  isaac
```

The launcher is `docker/scripts/launch_isaac.sh`. Task selection, asset
paths, and the OmniGraph DDS bridge are not yet documented here. Note that
the bridge currently `unset`s `ROS_DOMAIN_ID` and pins itself to channel 1
regardless of the host setting — bridging to a non-default domain is a
known gap.

## Troubleshooting

- X11 / GUI: run `xhost +local:docker` once per session if rviz, the MuJoCo
  viewer, or the slider GUI fail to open.
- Talking to the sim from the host (`ros2 topic list`, standalone `rviz2`)
  bypasses the run scripts, so it won't pick up `docker/.env` on its own —
  load it into your shell first: `set -a; source docker/.env; set +a`.
- For a clean rebuild of the message workspace, wipe
  `container_cache/msgs_ws/` on the host before relaunching.

## Working example — open the fridge

End-to-end run of the fridge-opening demo across three terminals. All three
share `ROS_DOMAIN_ID` from `docker/.env` (terminals A and B via the run scripts,
terminal C via the already-running container), so there's nothing to export.

```bash
# terminal A — MuJoCo (start first so /clock is publishing)
docker/scripts/docker_run.sh mujoco

# terminal B — ROS workspace shell, then launch bringup
docker/scripts/docker_run.sh ros
ros2 launch h1_bringup h1_sim_bringup.launch.py

# terminal C — exec into the running ROS container and run the demo
docker exec -it humanoid_sim_ros bash
source /opt/ros/humble/setup.bash
source /home/code/core_ws/install/setup.bash
ros2 run h1_bringup open_fridge.py
```
