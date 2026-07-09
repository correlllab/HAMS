# Navigation demo — autonomous SLAM + nav2 exploration (macOS / Apple-Silicon sim)

This walks the H1-2 through a RoboCasa kitchen **autonomously**: it builds a map
with SLAM, plans collision-free paths around the furniture with nav2, and drives
itself to unexplored frontiers — all on the headless CPU sim, no GPU.

It runs on top of the macOS (Apple-Silicon) port; see the **macOS (Apple
Silicon)** section of the top-level [`README.md`](../README.md) for the one-time
Colima/Docker setup, image builds, and the noVNC tunnel.

## What it demonstrates

- **2D SLAM** — the Livox lidar cloud is flattened to a laser scan
  (`pointcloud_to_laserscan`) and fed to `slam_toolbox`, which builds an
  occupancy `/map` against the sim's ground-truth `/odom`.
- **nav2** — a Smac/RegulatedPurePursuit stack plans a collision-free path on
  the map **and** a live costmap (2D scan + the full 3D Livox cloud, so it sees
  counters and appliances the flat slice misses), then drives `/cmd_vel`.
- **Locomotion** — the switchable lower-body controller follows `/cmd_vel` with
  the TorchScript walk policy, handed over from a stable FAME stand.
- **Frontier exploration** — `frontier_explore.py` repeatedly sends the nearest
  unexplored frontier as a nav2 goal until the reachable space is mapped.

## Run it

Start **both** containers with the nav stack enabled (RoboCasa first so `/clock`
is publishing before the ROS nodes latch onto sim time):

```bash
# from the repo root
HAMS_DISPLAY=vnc HAMS_RVIZ=vnc HAMS_CAMERAS=0 \
HAMS_LOWERBODY=switch HAMS_SLAM=1 HAMS_NAV2=1 HAMS_SPAWN_BACKOFF=1.5 \
  docker compose -f docker/docker-compose.mac.yml up -d

# open the noVNC tunnel (Colima doesn't forward container ports)
./docker/scripts/mac_vnc_tunnel.sh
```

Then drive it:

```bash
docker exec -it hams_ros bash        # host docker CLI flaky? use: colima ssh -- docker exec -it hams_ros bash
source /home/code/h12_sim_scripts/robot_cli.sh

rob_stand      # engage FAME; wait ~15 s of sim time for the tether to release and the stance to settle
rob_explore    # hand over to walk + start autonomous frontier exploration (Ctrl-C to stop)
```

Watch it live:

- **MuJoCo viewer** (robot in the kitchen) — <http://localhost:6080/vnc.html>
- **RViz** (map, costmap, green planned path) — <http://localhost:6081/vnc.html>

You'll see the map fill in, the costmap mark the counters/appliances, a green
plan appear to each frontier, and the robot walk to it. Unreachable frontiers are
blacklisted after a timeout and the explorer moves on; it prints
`No frontiers left — exploration complete` when done.

## The env knobs

| Variable | For the nav demo | Notes |
|---|---|---|
| `HAMS_LOWERBODY` | `switch` | Switchable controller (`rob_stand`→`rob_walk`). `fame` stands only; `walk` launches the raw policy (use `switch`). |
| `HAMS_SLAM` | `1` | Adds `pointcloud_to_laserscan` + `slam_toolbox` → `/map`. |
| `HAMS_NAV2` | `1` | Adds the nav2 stack (implies SLAM). Send goals via RViz "2D Nav Goal", `/navigate_to_pose`, or `rob_explore`. |
| `HAMS_SPAWN_BACKOFF` | `~1.5` | Metres to back the robot into open floor at spawn so nav2 has room. **Baked at container create — change it with `--force-recreate`, not `docker restart`** (see gotchas). |
| `HAMS_CAMERAS` | `0` | Drops the 3 RGBD renders — the heaviest per-step CPU cost. Roughly doubles sim rate; the nav demo doesn't need them. |
| `HAMS_DISPLAY` / `HAMS_RVIZ` | `vnc` | MuJoCo viewer on 6080, RViz on 6081. |

## Manual driving (without exploration)

```bash
rob_stand                 # FAME stand
rob_walk                  # hand over to the walk policy
rob_go 0.3 0 0.2 6        # vx vy wz secs — forward + gentle left turn for 6 s
rob_stop                  # zero velocity, back to FAME
```

Or set a single goal in RViz with the **2D Nav Goal** tool and let nav2 drive.

## How the pipeline fits together

```
MuJoCo (robocasa)                         ROS (hams_ros)
  Livox cloud ─/livox/pointcloud──►  pointcloud_to_laserscan ─/converted_scan─► slam_toolbox ─/map─┐
  free-joint pose ─/odom, TF──────►  ...............................................................│
                                     nav2 (global+local costmap: /map + scan + 3D cloud) ◄─────────┘
                                       └─ plan ─► controller ─/cmd_vel─► lowerbody_controller_node
  rt/lowcmd ◄──────────────────────────────────────────────────────────────────────┘  (walk policy)
```

## Gotchas specific to the nav demo

- **`HAMS_SPAWN_BACKOFF` needs a recreate, not a restart.** It's an environment
  variable baked into the container at create time. `docker restart` reuses the
  old value; to change it, recreate the sim service:
  ```bash
  HAMS_DISPLAY=vnc HAMS_CAMERAS=0 HAMS_SPAWN_BACKOFF=1.5 \
    docker compose -f docker/docker-compose.mac.yml up -d --force-recreate --no-deps robocasa
  ```
  The layout is randomized each launch, so a recreate also re-rolls the kitchen —
  handy if the robot spawns cramped against a counter (it'll then lean into it
  once FAME releases). `1.5`–`2.0` reliably clears the fixtures.
- **Restart the two containers coherently.** The sim owns `/clock`; if you restart
  it, its sim time resets to 0 and the still-running ROS side sees the clock jump
  backward (TF extrapolation errors, a frozen `0x0` SLAM map). After recreating or
  restarting `robocasa`, restart `hams_ros` too so it resyncs.
- **`rob_stand` before `rob_walk`/`rob_explore`.** The walk policy is stable when
  handed over from a settled FAME stance; give it ~15 s of sim time after
  `rob_stand` before commanding motion.
- **Some exploration goals time out on the slow sim.** At ~0.2× real-time, nav2
  planning + walking to a 2 m frontier can exceed the 120 s goal timeout; the
  explorer blacklists it and picks another. That's expected, not a failure.
