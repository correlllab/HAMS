# HAMS

HAMS (Humanoid Agent Modular Stack) for the Correll Lab H1 robot: MuJoCo, ROS 2,
and Isaac Sim running in separate containers and sharing a CycloneDDS ROS domain.

## Layout

- `docker/` — Dockerfiles, `docker-compose.yml`, and the build/run scripts.
- `core_ws/` — ROS 2 workspace (bringup, IK, perception, safety). Submodules.
- `h1_robocasa/` — RoboCasa/MuJoCo simulator entry point and bridges.
- `CL_Assets/` — URDF, MuJoCo XML, and Isaac USD assets.

## Prerequisites

- Docker (with Compose v2) and the NVIDIA Container Toolkit.
- Git LFS (`git lfs install`) — required to fetch the large binary assets
  (URDF meshes, MuJoCo XML, Isaac USD) tracked via LFS.
- `git submodule update --init --recursive` to populate `core_ws/src`.
- Copy `docker/.env.example` to `docker/.env` and fill in your `GEMINI_API_KEY`
  and `ROS_DOMAIN_ID` (see [Configuration](#configuration)).
- SAM3 weights (`sam3.pt`) — needed by the perception/`sam_server` pipeline.
  They are **not** tracked in git; download them from the gated
  [`facebook/sam3`](https://huggingface.co/facebook/sam3) repo and place the
  file at `core_ws/src/model_server/weights/sam3.pt`:

  ```bash
  huggingface-cli login                         # one-time; accept the facebook/sam3 license
  cd core_ws/src/model_server/weights
  huggingface-cli download facebook/sam3 --local-dir .cache
  mv .cache/sam3.pt sam3.pt                      # name it exactly sam3.pt
  ```

  See `core_ws/src/model_server/weights/README.md` for details and the
  auto-download fallback.

A few things worth knowing before you run anything:

- The build/run scripts can be invoked from any directory — they resolve their
  own location, so `docker/scripts/docker_run.sh robocasa` works just as well
  from `/tmp` as from the repo root.
- `ROS_DOMAIN_ID` is read from `docker/.env` and forwarded into the RoboCasa and
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
docker/scripts/docker_build.sh robocasa ros  # subset
docker/scripts/docker_build.sh isaac       # isaac only
```

The RoboCasa and ROS images both inherit from `hams_base`, which is
built first automatically when either profile is selected. Isaac is
self-contained and does not use the base.

## Run the RoboCasa container

```bash
docker/scripts/docker_run.sh robocasa            # windowed viewer
docker/scripts/docker_run.sh robocasa --headless # no DISPLAY / SSH / CI
docker/scripts/docker_run.sh robocasa bash       # drop to a shell instead

# to use a custom ROS domain (e.g. several devs on one network),
# set ROS_DOMAIN_ID in docker/.env (see Configuration above)
```

Once it's up, RoboCasa publishes `rt/lowstate` over CycloneDDS plus
`/head/color/image_raw`, `/head/depth/image_raw`, `/head/color/camera_info`,
`/lidar/points`, and `/tf` on the chosen ROS domain (default `1`).

## Run the ROS container and bringup

The ROS launcher only builds the workspace and drops to a shell, so bringup
is a manual step. `ROS_DOMAIN_ID` (all containers) and `GEMINI_API_KEY` (the
`ros` container, for the vision pipeline's Gemini backbone and `h12_skills`)
both come from `docker/.env`, so there is nothing to export — just open two
terminals:

```bash
# terminal A — start RoboCasa first so /clock is publishing
docker/scripts/docker_run.sh robocasa

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
# terminal A — RoboCasa (start first so /clock is publishing)
./docker/scripts/docker_run.sh robocasa --task OpenFridge --layout 1 --style 1 --seed 42

# terminal B — ROS workspace shell, then launch bringup
docker/scripts/docker_run.sh ros
ros2 launch h1_bringup h1_sim_bringup.launch.py

# terminal C — exec into the running ROS container and run the demo
docker exec -it hams_ros bash
source /opt/ros/humble/setup.bash
source /home/code/core_ws/install/setup.bash
ros2 action send_goal /skill/grasp custom_ros_messages/action/SkillGrasp \
  "{target_object: 'vertical fridge handle', arm: 'right', timeout: {sec: 60, nanosec: 0}}" \
  --feedback
```

## macOS (Apple Silicon) — headless CPU port

On Apple-Silicon Macs there is a self-contained, CPU-only port that runs **just
ROS 2 + MuJoCo/RoboCasa** — Isaac is dropped and there is no NVIDIA/EGL path, so
MuJoCo renders in software (OSMesa/llvmpipe). It uses its own standalone compose
file, `docker/docker-compose.mac.yml`, and runs the ROS layer on **FastDDS**
instead of CycloneDDS (an arm64 in-process coexistence fix). None of the x86
instructions above apply — use this section instead.

### Prerequisites

- [Colima](https://github.com/abiosoft/colima) + the Docker CLI
  (`brew install colima docker docker-compose`). No Docker Desktop, no NVIDIA
  toolkit, and no XQuartz (see the GUI note below).
- Git LFS and submodules, exactly as in [Prerequisites](#prerequisites) above.
- No `docker/.env` is needed; the Mac compose only reads `ROS_DOMAIN_ID`
  (optional, defaults to `1`), `HAMS_DISPLAY`, and `HAMS_RVIZ`.
- Start the VM with enough resources — one software-GL viewer alone uses ~5
  cores, and running both the MuJoCo viewer and RViz wants headroom:

  ```bash
  colima start --cpu 12 --memory 24     # sized for a 14-core / 48 GB Mac; scale to yours
  ```

### Build and run

```bash
# build both arm64 images (robocasa + ros); the base image builds automatically
docker compose -f docker/docker-compose.mac.yml build

# headless (no viewer) — the default
docker compose -f docker/docker-compose.mac.yml up
```

The first run is slow (colcon builds the message + controller workspaces into
named volumes); later runs are cached and fast. Both containers use
`network_mode: host` + `ipc: host` so FastDDS's shared-memory transport works
across them.

> **Always bring both containers up together.** If RoboCasa runs alone for more
> than a few seconds it releases the robot's motors ("Command timeout"), the H1
> collapses under gravity, and when the ROS controller then connects it reads the
> fallen pose and trips its e-stop. `docker compose … up` (both services) engages
> the controller before the robot can fall.

### Viewing the GUIs (MuJoCo viewer + RViz)

MuJoCo's and RViz's OpenGL windows **cannot** be forwarded to XQuartz —
Apple-Silicon XQuartz has broken indirect GLX. Instead each is rendered
in-container with software GL into a virtual display and streamed to your browser
over noVNC. Enable them per-viewer:

```bash
# MuJoCo viewer on :6080, RViz on :6081 (set either or both)
HAMS_DISPLAY=vnc HAMS_RVIZ=vnc docker compose -f docker/docker-compose.mac.yml up -d

# open the SSH tunnel to both noVNC ports (Colima does not forward container ports)
./docker/scripts/mac_vnc_tunnel.sh     # --open also opens the browser; --stop closes the tunnel
```

Then open:

- **MuJoCo viewer** → <http://localhost:6080/vnc.html>
- **RViz** (RobotModel + TF) → <http://localhost:6081/vnc.html>

`HAMS_DISPLAY` (RoboCasa/MuJoCo) and `HAMS_RVIZ` (ROS/RViz) are independent — set
either or both; the defaults are headless.

### Driving the robot

Bringup starts automatically in the `ros` container (`joint_state_publisher`,
`robot_state_publisher`, the `frame_task_server` IK solver, `safety_node`). To
command the H1, source the helper and send joint-space postures or gripper
commands:

```bash
docker exec -it hams_ros bash          # if the host docker CLI is flaky: colima ssh, then docker exec …
source /home/code/h12_sim_scripts/robot_cli.sh

rob_poses                # list postures
rob_pose t_pose          # MOVE: home t_pose arms_front arms_overhead elbow_only … (rob_poses lists all)
rob_grip right close     # open/close a gripper
rob_home
```

Watch the motion in either viewer. (`rob_pose` takes a name and moves the robot;
`rob_poses` only prints the list.)

### Lower-body controller (standing free)

By default an elastic-band tether holds the robot upright and only the upper body
is controlled. Set `HAMS_LOWERBODY=fame` to run the RMA lower-body policy, which
releases the tether and **balances the robot standing unsupported**:

```bash
HAMS_DISPLAY=vnc HAMS_RVIZ=vnc HAMS_LOWERBODY=fame \
  docker compose -f docker/docker-compose.mac.yml up
```

`HAMS_LOWERBODY=fame` **stands** (and squats via `/lowerbody/squat_cmd`) but holds
position — it does not locomote. For **stand *and* walk**, use the switchable
controller instead:

```bash
HAMS_DISPLAY=vnc HAMS_RVIZ=vnc HAMS_LOWERBODY=switch \
  docker compose -f docker/docker-compose.mac.yml up
```

It starts band-held idle; `rob_stand` engages FAME (stand free), `rob_walk` hands
over to the TorchScript walk policy, and `rob_go <vx> <vy> <wz>` drives it. The
walk policy **does stay upright** now, handed over from a settled FAME stance — the
earlier "falls a few seconds after the tether releases" was a too-tight motor
watchdog on the slow sim (see gotchas). Launching the raw policy directly
(`HAMS_LOWERBODY=walk`) still just marches in place; use `switch`. (`torch` is
already in the ros image; building `unitree_hg` needs `rosidl-generator-dds-idl`,
which the image now includes.)

### Navigation demo (SLAM + nav2 + frontier exploration)

The robot can map the kitchen and drive itself around autonomously — 2D SLAM from
the lidar, nav2 planning collision-free paths on a costmap that includes the full
3D cloud, and a frontier explorer sending goals. Full walkthrough:
**[docs/NAVIGATION_DEMO.md](docs/NAVIGATION_DEMO.md)**. In short:

```bash
HAMS_DISPLAY=vnc HAMS_RVIZ=vnc HAMS_CAMERAS=0 \
HAMS_LOWERBODY=switch HAMS_SLAM=1 HAMS_NAV2=1 HAMS_SPAWN_BACKOFF=1.5 \
  docker compose -f docker/docker-compose.mac.yml up -d
# then, inside hams_ros:
#   source /home/code/h12_sim_scripts/robot_cli.sh
#   rob_stand    # FAME stand (wait ~15 s sim time for the tether to release)
#   rob_explore  # walk handover + autonomous frontier exploration
```

Watch the map, costmap, and green plan build in RViz (<http://localhost:6081/vnc.html>).

### macOS gotchas

- **The host `docker` CLI socket is intermittent** under Colima — `docker …` may
  fail with "Cannot connect to the Docker daemon" while the VM and containers are
  perfectly healthy. Use `colima ssh -- docker …` as the reliable fallback, or
  `colima stop && colima start` to relink the socket (this restarts the containers).
- **Cartesian reaching is disabled.** The `/frame_task` action currently crashes
  the controller node, so `rob_reach` is a no-op stub — drive the robot in joint
  space with `rob_pose` for now.
- **Two viewers use two displays.** RoboCasa renders on X display `:99` and RViz
  on `:100`; they must differ because the containers share one network namespace
  (`network_mode: host`). The launchers already handle this.
- **Restart the two containers coherently.** The RoboCasa container owns `/clock`.
  Restarting it alone resets sim time to 0 while the ROS side keeps its old clock →
  TF extrapolation errors and a frozen `0×0` SLAM map. After restarting/recreating
  `robocasa`, restart `hams_ros` too so it resyncs to the fresh clock.
- **`HAMS_SPAWN_BACKOFF` is baked at container-create.** `docker restart` reuses the
  old value; use `docker compose up --force-recreate --no-deps robocasa` (with the
  env set) to change it. For nav, `1.5`–`2.0` keeps the robot off the counters.
- **The sim motor watchdog is sim-time based.** The low-level interface zeroes the
  motors if no `rt/lowcmd` arrives within `HAMS_CMD_TIMEOUT` (default `0.5`) seconds
  **of sim time** — measured in sim time on purpose, because a wall-clock timeout is
  far too tight on a sim running ~0.2× real-time (it used to drop the robot ~0.7 s
  after the tether released). Bump `HAMS_CMD_TIMEOUT` if a heavier scene still trips it.
