# Humanoid Simulation Runbook

This runbook explains how to bring up the H1-2 simulation stack, what each
piece does, and how to debug the common failure modes. It assumes you are on a
Linux/NVIDIA machine with Docker, Docker Compose, NVIDIA Container Toolkit, Git
LFS, and the repo submodules already initialized.

## Mental Model

This repo is not a standalone MuJoCo desktop app. It is a robotics stack split
across containers:

- MuJoCo container: runs the physics simulation, robot model, kitchen scene,
  simulated camera/lidar, Unitree-style low-level DDS interface, and Magpie hand
  bridge.
- ROS container: runs ROS 2 controllers, safety relay, RViz, slider GUI, IK
  action servers, optional navigation, optional perception, and demos.
- RViz: visualizes ROS topics such as `/tf`, `/joint_states`, images, point
  clouds, and the robot model. RViz is not the MuJoCo native viewer.
- Slider Debugger: a small Tk GUI that sends wrist targets to `/frame_task` and
  gripper aperture commands to the Magpie gripper services.

MuJoCo itself is installed inside the Docker image as the Python `mujoco`
package. You do not need a separate host MuJoCo installation for this stack.

## Directory Map

- `docker/`: Dockerfiles, Compose config, and build/run scripts.
- `h1_mujoco/`: MuJoCo runtime code.
  - `h12_mujoco.py`: main simulator entry point.
  - `scene_builder.py`: merges the H1-2 robot into a Robocasa kitchen scene.
  - `mujoco_ros_bridge.py`: publishes camera, lidar, TF, and `/clock` to ROS 2.
  - `unitree_interface.py`: publishes `rt/lowstate` and consumes `rt/lowcmd`.
  - `magpie_hand_bridge.py`: exposes simulated hands as ROS gripper
    services/actions.
- `CL_Assets/`: robot assets, meshes, URDF, MuJoCo XML, USD files.
- `core_ws/`: ROS 2 workspace.
  - `h1_bringup`: launch files, slider GUI, RH56 demo, fridge demo.
  - `h12_ros2_controller`: IK and upper-body frame task controller.
  - `h12_safety_layer`: validates and merges low-level commands.
  - `h12_lowerbody_controller`: learned walking/lower-body policy.
  - `vision_pipeline`: optional perception stack.
  - `custom_ros_messages`, `magpie_msgs`: message/action/service definitions.

## First-Time Setup Check

From the repo root:

```bash
git status --short --branch
git submodule status --recursive
git lfs version
```

If submodules are missing:

```bash
git submodule update --init --recursive
git lfs pull
git submodule foreach --recursive 'git lfs pull || true'
```

If the repo uses SSH submodule URLs and GitHub SSH is not configured, either add
an SSH key to GitHub or use an HTTPS rewrite for the initialization command:

```bash
git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive
```

## Build Images

Build the two images used for the MuJoCo/ROS workflow:

```bash
docker/scripts/docker_build.sh mujoco ros
```

If your shell has not picked up Docker group membership yet:

```bash
newgrp docker
```

or:

```bash
sg docker -c 'docker/scripts/docker_build.sh mujoco ros'
```

The images are large. Docker image layers and BuildKit cache live under
Docker's configured data root, usually `/var/lib/docker`, unless Docker has a
custom `data-root` configured.

## Recommended Slider Smoke Test

This is the best mode for debugging upper-body wrist control. It keeps the base
fixed, disables lower-body walking, disables navigation, and disables vision.

Terminal A, MuJoCo:

```bash
export ROS_DOMAIN_ID=1
docker/scripts/docker_run.sh mujoco --headless --fixed-base
```

Terminal B, ROS:

```bash
export ROS_DOMAIN_ID=1
docker/scripts/docker_run.sh ros
source /home/code/core_ws/install/setup.bash
ros2 launch h1_bringup h1_sim_bringup.launch.py \
  use_nav:=false \
  use_vision:=false \
  use_walking:=false \
  use_rviz:=true \
  use_sliders:=true
```

If launch file edits are not reflected, rebuild only `h1_bringup` inside the
ROS container:

```bash
colcon build --symlink-install --packages-select h1_bringup
source install/setup.bash
```

Check that the wrist action server is alive:

```bash
ros2 action list | grep frame_task
```

Expected output:

```text
/frame_task
```

The slider GUI does not stream continuously. Move sliders, then click `Send`.

## Full Bringup Modes

The bringup launch file exposes these switches:

- `use_rviz:=true|false`: show or hide RViz.
- `use_sliders:=true|false`: show or hide the slider GUI.
- `use_nav:=true|false`: start or skip Nav2/navigation.
- `use_vision:=true|false`: start or skip the vision pipeline.
- `use_walking:=true|false`: start or skip the lower-body walking policy.

For a quiet upper-body session:

```bash
ros2 launch h1_bringup h1_sim_bringup.launch.py \
  use_nav:=false \
  use_vision:=false \
  use_walking:=false \
  use_rviz:=true \
  use_sliders:=true
```

For a fuller sim session with walking/navigation:

```bash
ros2 launch h1_bringup h1_sim_bringup.launch.py \
  use_nav:=true \
  use_vision:=false \
  use_walking:=true \
  use_rviz:=true \
  use_sliders:=true
```

Only enable `use_vision:=true` when the perception stack and API keys are ready.

## Runtime Data Flow

Wrist slider path:

```text
Slider Debugger
  -> /frame_task action
  -> frame_task_server
  -> h12_ros2_controller IK
  -> rt/safety/lowcmd_upper_in
  -> h12_safety_layer
  -> rt/lowcmd
  -> MuJoCo SimInterface
  -> robot joints move
  -> /lowstate, /joint_states, /tf
  -> RViz updates
```

Gripper slider path:

```text
Slider Debugger
  -> /gripper/right/set_position or /gripper/left/set_position
  -> MagpieHandBridge
  -> MuJoCo hand actuators
```

Sensor path:

```text
MuJoCo scene
  -> mujoco_ros_bridge.py
  -> /realsense/head/*, /livox/*, /tf, /clock
  -> RViz and ROS nodes
```

## Important Topics, Actions, And Services

Core sim topics:

```bash
ros2 topic list | grep -E 'lowstate|lowcmd|clock|realsense|livox|tf|joint_states'
```

Useful actions:

```bash
ros2 action list
```

Expected during slider testing:

```text
/frame_task
/named_config
/gripper/left/deligrasp
/gripper/right/deligrasp
```

Useful gripper checks:

```bash
ros2 topic echo /gripper/right/state
ros2 service list | grep gripper
```

## RViz Notes

If the camera/depth images move but the robot model appears stale:

- Set RViz `Global Options -> Fixed Frame` to `pelvis`.
- Make sure `RobotModel`, `TF`, and `/joint_states` are visible.
- Check:

```bash
ros2 topic echo /joint_states --once
ros2 topic echo /tf --once
```

RViz displays ROS state. It is possible for camera images to update while the
RobotModel display is hidden, stale, or using a disconnected fixed frame.

## Common Warnings

These are usually harmless for the slider smoke test:

- `robosuite_models` not installed.
- `mimicgen` not installed.
- RViz message filter queue drops during startup.
- Camera image QoS reliability mismatch warnings.
- Nav2 costmap warnings when `use_nav:=true`.

These matter:

- `frame_task_server ... process has died`: slider wrist goals will not work.
- `Emergency stop triggered`: controller or safety layer stopped command flow.
- `Command timeout! Releasing motors`: MuJoCo received no low-level command for
  too long and released the motors.
- `Nan, Inf or huge value in QACC`: MuJoCo dynamics became unstable, often
  after the robot fell or hit a hard limit.

## Why Use `--fixed-base`

The normal MuJoCo model has a floating base:

```xml
<joint name="floating_base_joint" type="free" ... />
```

If no controller publishes quickly enough, `unitree_interface.py` releases the
motors after a short timeout. The robot can fall, hit a joint limit, and trigger
an emergency stop before the upper-body controller is ready.

`--fixed-base` removes the floating-base joint in the assembled kitchen scene.
This is not a realistic whole-body simulation mode, but it is very useful for:

- verifying ROS/MuJoCo communication;
- debugging `/frame_task`;
- testing wrist trajectories;
- testing the RH56 grasp demo's upper-body logic.

Use floating-base mode later when testing walking/balance behavior.

## Current RH56 State

The current MuJoCo hand model is still Magpie, not RH56/Inspire. The RH56 demo
in `h1_bringup` is a ROS-side integration stub:

- It uses `rh56_fk_cache.npz` to map grasp width to RH56 hand posture.
- It sends synchronized wrist targets through `/frame_task`.
- It can use the Magpie gripper aperture as a placeholder hand backend.

Run the demo only after `/frame_task` is alive:

```bash
ros2 run h1_bringup rh56_grasp_demo.py \
  --mode plane \
  --fingers 4 \
  --side right \
  --hand-backend magpie \
  --start-width-mm 110 \
  --target-width-mm 40 \
  --steps 10 \
  --segment-duration 0.45
```

Research path:

1. Validate H1-2 wrist trajectory with fixed-base MuJoCo and Magpie placeholder.
2. Tune grasp target defaults such as `--grasp-x`, `--grasp-y`, `--grasp-z`,
   and `--plane-rpy-deg`.
3. Port RH56/Inspire MuJoCo XML assets into the sim.
4. Add a bridge from `/right_hand_cmd` and `/left_hand_cmd` to RH56 actuators.
5. Switch RH56 demo backend from `magpie` to `inspire-topic` or `both`.

## Useful Debug Commands

List running containers:

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
```

Enter the ROS container:

```bash
docker exec -it humanoid_sim_ros bash
source /opt/ros/humble/setup.bash
source /home/code/core_ws/install/setup.bash
```

Enter the MuJoCo container:

```bash
docker exec -it humanoid_sim_mujoco bash
```

Watch actions/topics:

```bash
ros2 action list
ros2 topic list
ros2 node list
```

Check Docker image/build storage:

```bash
docker images | grep humanoid
docker system df
```

Prevent host sleep during long builds:

```bash
systemd-inhibit --what=sleep --why="Docker build" docker/scripts/docker_build.sh mujoco ros
```

## Files Edited For The Current Smoke-Test Workflow

These local changes make the current workflow easier to debug:

- `h1_mujoco/h12_mujoco.py`: adds `--fixed-base`.
- `h1_mujoco/scene_builder.py`: removes the floating-base joint when
  `fixed_base=True`.
- `core_ws/src/h1_bringup/launch/h1_sim_bringup.launch.py`: adds
  `use_vision` and `use_walking` launch switches.
- `docker/MujocoDockerfile`: removes optional warning-only GitHub clones from
  the mandatory MuJoCo image build.
- `docker/RosDockerfile`: splits fragile pip layers and cleans stale Pillow
  metadata before pinning Pillow.

