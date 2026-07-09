# ROS debugging MCP server (macOS / Apple-Silicon sim)

A small MCP server that gives an assistant (Claude Code) first-class tools to
inspect and drive the simulated H1 — `robot_status`, `wait_for`, `costmap_summary`,
`drive`, etc. — instead of shelling into the container and parsing `ros2` CLI output.

## Why it lives in the container

The ROS 2 graph (FastDDS, `network_mode: host`) runs inside the Colima VM's
containers, and Colima exposes no reachable VM IP and forwards no ports — the same
wall the noVNC viewers hit. So the server **runs inside the `ros` container** (where
DDS is directly reachable) and exposes a single HTTP port, which you reach from the
Mac over the existing SSH tunnel. Only the MCP protocol crosses the boundary; DDS
never has to.

A warm `rclpy` node subscribes once to the hot topics (`/odom`, the global costmap,
`/plan`, `/converted_scan`, `/clock`) and caches the latest message, so status calls
answer instantly instead of paying node-startup + DDS-discovery on every query.

## Enable it

Add `HAMS_ROS_MCP=1` when you launch (it starts on `localhost:6082` inside the VM):

```bash
HAMS_DISPLAY=vnc HAMS_RVIZ=vnc HAMS_CAMERAS=0 \
HAMS_LOWERBODY=switch HAMS_SLAM=1 HAMS_NAV2=1 HAMS_SPAWN_BACKOFF=1.5 \
HAMS_ROS_MCP=1 \
  docker compose -f docker/docker-compose.mac.yml up -d

./docker/scripts/mac_vnc_tunnel.sh        # forwards 6082 alongside the viewer ports
```

The first launch `pip install`s the `mcp` package inside the container (logged to
`/tmp/ros_mcp.log`). `HAMS_ROS_MCP` is baked at container-create — set it on a
`compose up` (recreate), not a bare `docker restart`.

## Register it with Claude Code (one-time, on the Mac)

```bash
claude mcp add --transport http ros_debug http://localhost:6082/mcp
claude mcp list          # verify it connects
```

Optionally allowlist the tools so they run without a prompt (in your settings, e.g.
`.claude/settings.json`):

```json
{ "permissions": { "allow": ["mcp__ros_debug__.*"] } }
```

To remove: `claude mcp remove ros_debug`.

## Tools

| Tool | What it does |
|---|---|
| `robot_status()` | base position (x,y,z), uprightness (1.0=vertical, <0.5=fallen), posture, sim time |
| `costmap_summary()` | global-costmap cell counts: lethal / inflated / free / unknown |
| `nav_status()` | current nav2 plan: exists?, pose count, path length (m) |
| `scan_status()` | is `/converted_scan` live? stamp + range count |
| `set_lowerbody(mode)` | `'fame'` stand free, or `'walk'` hand over to the walk policy |
| `drive(vx,vy,wz,duration)` | publish `/cmd_vel` for N seconds, then stop; reports start/end pose |
| `wait_for(condition,timeout)` | block until `standing`/`fallen`/`stopped`/`moving`/`scan_live`/`nav_has_plan`/`sim_time>=N` |
| `list_topics(filter)` / `echo_topic(topic)` / `topic_hz(topic)` | generic topic inspection |
| `call_service(name,type,args)` | call any service |
| `node_list()` | list active nodes |

The first eight are the warm/fast path; the generic passthroughs shell out to the
`ros2` CLI for the occasional arbitrary topic.

## Notes & caveats

- **In-container reach only.** The server sees everything on the ROS graph. It does
  **not** reach the *other* container's MuJoCo `:99` viewer or the RoboCasa
  fall-logger stdout (a second endpoint or docker-socket access would be needed).
- **Sensor QoS.** `/odom` and `/converted_scan` are best-effort publishers; the
  server subscribes best-effort so it receives them regardless of publisher QoS.
- **Headless runs.** An unauthenticated localhost HTTP MCP server works in
  non-interactive Claude Code runs as long as the tunnel and server are up; there's
  no OAuth to complete.
- **Port.** Override with `HAMS_ROS_MCP_PORT` (compose passes it through); update the
  `claude mcp add` URL and `mac_vnc_tunnel.sh` `PORTS` to match.
