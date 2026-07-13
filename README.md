# HAMS

HAMS (Humanoid Agent Modular Stack) for the Correll Lab H1 robot: MuJoCo, ROS 2,
and Isaac Sim running in separate containers and sharing a CycloneDDS ROS domain.

## Layout

- `docker/` — x86 Dockerfiles, `docker-compose.yml`, and the build/run scripts.
- `docker/mac/` — self-contained Apple-Silicon (arm64/CPU) port: its own
  Dockerfiles, `docker-compose.yml`, and `scripts/` (see the macOS section below).
- `core_ws/` — ROS 2 workspace (bringup, IK, perception, safety). Submodules.
- `h1_robocasa/` — RoboCasa/MuJoCo simulator entry point and bridges.
- `CL_Assets/` — URDF, MuJoCo XML, and Isaac USD assets.
- `tools/` — standalone debugging tools (e.g. the ROS MCP server).

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

Once it's up, RoboCasa publishes `rt/lowstate` over CycloneDDS plus, on the
chosen ROS domain (default `1`): `/clock`; the RealSense topics
`/realsense/{head,left_hand,right_hand}/color/image_raw` and
`.../aligned_depth_to_color/image_raw` (each raw and compressed) with
`.../color/camera_info`; the Livox topics `/livox/lidar` (CustomMsg),
`/livox/pointcloud`, and `/livox/imu`; and `/{left,right}/gripper/state`.

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

## Network / DDS tuning

The stack moves bursty, high-bandwidth sensor data (camera images, Livox point
clouds) over CycloneDDS. On a gigabit link the wire has headroom, but the
**default 208 KB kernel UDP receive buffer** overflows on bursts, so the kernel
silently drops datagrams. Lost DDS fragments mean whole point-cloud/image
samples stall or drop — topics go laggy even though the network isn't full. The
tell is `RcvbufErrors` climbing in `/proc/net/snmp`.

Three settings fix this and **all are required**:

1. **Host kernel buffers.** The containers run `network_mode: host`, so these
   are the host's `net.core` sysctls. CycloneDDS can only *request* a large
   socket buffer — the kernel clamps it to `net.core.rmem_max` — so this must be
   raised or step 2 has no effect. For the current session:

   ```bash
   sudo sysctl -w net.core.rmem_max=2147483647
   sudo sysctl -w net.core.rmem_default=16777216
   sudo sysctl -w net.core.netdev_max_backlog=10000
   ```

   To persist across reboots, put them in `/etc/sysctl.d/10-cyclonedds.conf`:

   ```
   net.core.rmem_max=2147483647
   net.core.rmem_default=16777216
   net.core.netdev_max_backlog=10000
   ```

   then apply with `sudo sysctl --system`.

2. **CycloneDDS config.** `core_ws/cyclonedds.xml` requests a 16 MB receive
   buffer. The `ros` profile wires it in automatically — `docker-compose.yml`
   bind-mounts it to `/home/code/cyclonedds.xml` and sets
   `CYCLONEDDS_URI=file:///home/code/cyclonedds.xml`. Edit the file and restart
   the container to change the tuning (e.g. to pin the DDS network interface).

3. **Pin the NIC IRQ off the real-time cores.** `eno1` has a single RX queue, so
   *all* inbound DDS traffic is processed in one `NET_RX` softirq on whichever core
   holds its IRQ. `irqbalance` parks that IRQ on core 11 — inside the MJPC planner's
   `0-11` set — so under a balance run the robot's 500 Hz `/lowstate` stream arrives
   at niraj at only ~400 Hz with multi-second stalls (tf fades in rviz, the lowerbody
   controller trips its state-stale safe-hold). The wire and the robot are fine; the
   loss is this single-core contention. Move the IRQ onto a dedicated core off the RT
   set (0–12) and stop irqbalance from moving it back. For the current session:

   ```bash
   sudo systemctl stop irqbalance
   echo 13 | sudo tee /proc/irq/$(awk -F: '/eno1/{gsub(/ /,"",$1);print $1}' /proc/interrupts)/smp_affinity_list
   ```

   To persist, `sudo systemctl disable --now irqbalance` and run `scripts/pin_net_irq.sh`
   at boot (it derives the IRQ, pins it, and prints a ready-made systemd unit). The
   launch's CPU map reserves core 13 for exactly this — `OTHER_CPUS` starts at 14 so
   no ROS node lands on the network core, and the estimator has core 12. Confirm the
   softirq moved (counts should now grow on core 13, not 11):

   ```bash
   grep NET_RX /proc/softirqs
   ```

Verify under load — `RcvbufErrors` should stop climbing:

```bash
docker exec hams_ros grep '^Udp:' /proc/net/snmp   # watch the RcvbufErrors column
```

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

# send the arms to the 'home' named config before grasping
ros2 action send_goal /named_config custom_ros_messages/action/NamedConfig \
  "{config_name: 'home', duration: {sec: 0, nanosec: 0}}"

ros2 action send_goal /skill/grasp custom_ros_messages/action/SkillGrasp \
  "{target_object: 'vertical fridge handle', arm: 'right', timeout: {sec: 60, nanosec: 0}}" \
  --feedback
```

## macOS (Apple Silicon) — headless CPU port

On Apple-Silicon Macs there is a self-contained, CPU-only port that runs **just
ROS 2 + MuJoCo/RoboCasa** — Isaac is dropped and there is no NVIDIA/EGL path, so
MuJoCo renders in software (OSMesa/llvmpipe). It uses its own standalone compose
file, `docker/mac/docker-compose.yml`, and runs the ROS layer on **FastDDS**
instead of CycloneDDS (an arm64 in-process coexistence fix). None of the x86
instructions above apply — use this section instead.

### Prerequisites

- [Colima](https://github.com/abiosoft/colima) + the Docker CLI
  (`brew install colima docker docker-compose`). No Docker Desktop, no NVIDIA
  toolkit, and no XQuartz (see the GUI note below).
- Git LFS and submodules, exactly as in [Prerequisites](#prerequisites) above.
- No `docker/.env` is needed; the Mac compose reads `ROS_DOMAIN_ID` (optional,
  defaults to `1`) plus the `HAMS_*` feature toggles (`HAMS_DISPLAY`,
  `HAMS_RVIZ`, `HAMS_LOWERBODY`, `HAMS_SLAM`, `HAMS_NAV2`, …) documented below.
- Start the VM with enough resources — one software-GL viewer alone uses ~5
  cores, and running both the MuJoCo viewer and RViz wants headroom:

  ```bash
  colima start --cpu 12 --memory 24     # sized for a 14-core / 48 GB Mac; scale to yours
  ```

### Build and run

```bash
# build both arm64 images (robocasa + ros); the base image builds automatically
docker/mac/scripts/docker_build_mac.sh      # or: docker_build_mac.sh robocasa | ros

# headless (no viewer) — the default; brings BOTH services up together
docker compose -f docker/mac/docker-compose.yml up
```

`docker/mac/scripts/docker_run_mac.sh [robocasa|ros]` is the Mac counterpart of the
x86 `docker/scripts/docker_run.sh` — it starts one service with a stable name
(`hams_sim_robocasa` / `hams_ros`), a `bash`/`<cmd>` override, and the same
`HAMS_DISPLAY`/`HAMS_RVIZ` env passthrough. Because it runs a single service, use
it in two terminals (start `ros` promptly) or for one-off shells; prefer
`docker compose … up` for the everyday paired sim so RoboCasa never runs alone
(see the collapse warning below).

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
HAMS_DISPLAY=vnc HAMS_RVIZ=vnc docker compose -f docker/mac/docker-compose.yml up -d

# open the SSH tunnel to both noVNC ports (Colima does not forward container ports)
./docker/mac/scripts/mac_vnc_tunnel.sh     # --open also opens the browser; --stop closes the tunnel
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
  docker compose -f docker/mac/docker-compose.yml up
```

`HAMS_LOWERBODY=fame` **stands** (and squats via `/lowerbody/squat_cmd`) but holds
position — it does not locomote. For **stand *and* walk**, use the switchable
controller instead:

```bash
HAMS_DISPLAY=vnc HAMS_RVIZ=vnc HAMS_LOWERBODY=switch \
  docker compose -f docker/mac/docker-compose.yml up
```

It starts band-held idle and auto-switches stand↔walk from the commanded velocity:
`rob_stand` settles it into FAME (stand free) and `rob_go <vx> <vy> <wz>` drives the
TorchScript walk policy — any non-zero `/cmd_vel` engages walk automatically, so
there is no manual `rob_walk` handover step (`rob_walk` just prints a reminder). The
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
HAMS_DISPLAY=vnc HAMS_RVIZ=vnc HAMS_CAMERAS=0 HAMS_SIM_ODOM=1 \
HAMS_LOWERBODY=switch HAMS_SLAM=1 HAMS_NAV2=1 HAMS_SPAWN_BACKOFF=1.5 \
  docker compose -f docker/mac/docker-compose.yml up -d
# then, inside hams_ros:
#   source /home/code/h12_sim_scripts/robot_cli.sh
#   rob_stand    # FAME stand (wait ~15 s sim time for the tether to release)
#   rob_explore  # walk handover + the /skill/frontier_explore action (autonomous mapping)
```

`rob_explore` is a shorthand; send the action directly if you want to tune the
frontier params or watch feedback. Any field left at `0` falls back to the node
default (shown in the comments):

```bash
ros2 action send_goal /skill/frontier_explore custom_ros_messages/action/SkillFrontierExplore \
  "{min_frontier_cells: 6, blacklist_radius: 0.6, min_goal_distance: 0.7, goal_timeout: 120.0, timeout: {sec: 1800, nanosec: 0}}" \
  --feedback
# min_frontier_cells : ignore frontier clusters smaller than this   (0 -> 6)
# blacklist_radius   : m; blacklist a failed goal within this radius (0 -> 0.6)
# min_goal_distance  : m; skip frontiers closer than this            (0 -> 0.7)
# goal_timeout       : s; cancel a single nav goal after this        (0 -> 120)
# timeout            : overall exploration budget (Duration)         (0 -> node default)
# Feedback streams phase / progress / frontiers_remaining; result is
# success / message / goals_reached. Ctrl-C cancels the goal.
```

`HAMS_SIM_ODOM=1` publishes the sim's ground-truth base odometry (`/odom` +
`odom→pelvis` TF) that SLAM consumes; it is off by default so the shared sim
bridge stays silent on the real/x86 stack, and the nav demo opts in.

Watch the map, costmap, and green plan build in RViz (<http://localhost:6081/vnc.html>).

### ROS debugging MCP server (optional)

`HAMS_ROS_MCP=1` starts an in-container MCP server that exposes ROS-inspection and
robot-driving tools (`robot_status`, `wait_for`, `costmap_summary`, `drive`, …) to
Claude Code over the same SSH tunnel as the viewers. Setup and tool list:
**[docs/ROS_MCP_DEBUG.md](docs/ROS_MCP_DEBUG.md)**.

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
