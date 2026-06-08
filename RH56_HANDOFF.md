# RH56 / H1-2 Grasp Integration Handoff

Date: 2026-06-08
Branch: `rh56-grasp-integration`
Primary repo: `correlllab/Humanoid_Simulation`
Source reference repo: `/Users/tanxuan/Code/RH56/rh56_controller`, branch `h12-sim-grasp-sync`

## Goal

Bring the RH56 antipodal grasp workflow into the H1-2 simulation stack so the H1-2 arm can act as the wrist carrier for the RH56 hand. The important behavior is width-space synchronization: as grasp width closes, the wrist target should update with the hand posture so the active fingertip line/plane stays aligned with the contact plane.

This is not just an animation requirement. The intended production path is a control/planning path that can later be mapped to the real H1-2 and RH56 hand.

## Current State

A first ROS-side demo has been added to `h1_bringup`:

- `core_ws/src/h1_bringup/scripts/rh56_grasp_demo.py`
- `core_ws/src/h1_bringup/data/rh56_fk_cache.npz`
- `core_ws/src/h1_bringup/setup.py`
- `core_ws/src/h1_bringup/package.xml`

The demo computes RH56 width-space closure from a precomputed FK cache, then sends short `/frame_task` wrist goals while streaming matching hand-width commands.

The FK cache avoids importing MuJoCo in the ROS container. This is deliberate because the ROS container has scipy/pinocchio/pink/mink, but MuJoCo is mainly in the MuJoCo container.

## Important Limitation

The current `Humanoid_Simulation` MuJoCo model is still H1-2 + Magpie, not H1-2 + RH56/Inspire.

Evidence found in the repo:

- `h1_mujoco/h12_mujoco.py` starts `MagpieHandBridge` for left and right hands.
- `CL_Assets/mujoco_assets/h1_2_magpie.xml` includes `magpie_gripper.xml` and exposes Magpie finger actuators.
- The available MuJoCo hand service is `/gripper/{left,right}/set_position`, an aperture command in millimeters.
- `h12_ros2_controller` has `/right_hand_cmd` and `/left_hand_cmd` topics for 6-float Inspire-style commands, but the current MuJoCo bridge does not appear to consume those commands into an RH56 hand model.

Because of this, the demo defaults to `--hand-backend magpie`. That lets us validate the H1-2 wrist trajectory and timing first. Full RH56 hand simulation still needs a MuJoCo asset/bridge pass.

## Running On Linux

This stack is designed for Linux with Docker, NVIDIA Container Toolkit, and X11. Running it on macOS is not expected to be production-faithful, and Docker Desktop on Mac will likely hit NVIDIA/CUDA/runtime or GUI limitations.

Recommended first setup on the Linux machine:

```bash
cd /path/to/Humanoid_Simulation
git checkout rh56-grasp-integration
git submodule update --init --recursive
```

If SSH submodule URLs fail, use the HTTPS rewrite for initialization:

```bash
git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --recursive
```

Build the relevant containers:

```bash
docker/scripts/docker_build.sh mujoco ros
```

Run across terminals with the same ROS domain:

```bash
export ROS_DOMAIN_ID=1
```

Terminal A:

```bash
docker/scripts/docker_run.sh mujoco
```

Terminal B:

```bash
docker/scripts/docker_run.sh ros
ros2 launch h1_bringup h1_sim_bringup.launch.py
```

Terminal C:

```bash
docker exec -it humanoid_sim_ros bash
source /opt/ros/humble/setup.bash
source /home/code/core_ws/install/setup.bash
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

The default planner-world grasp target is:

- `--grasp-x 0.35`: forward in planner world
- `--grasp-y -0.25`: right side of the robot, because planner-world +Y is left
- `--grasp-z 0.10`: height in planner world

The script converts planner world to H1-2 pelvis frame before sending `/frame_task`.

## What To Check First

1. Confirm MuJoCo and ROS containers can see each other:

```bash
ros2 topic list
ros2 action list
```

Expected relevant endpoints include:

- `/frame_task`
- `/named_config`
- `/right_ee_pose`
- `/left_ee_pose`
- `/gripper/right/set_position`

2. Run the demo with conservative widths and watch the MuJoCo viewer.

3. If the wrist target is unreachable or moves in the wrong direction, adjust only these first:

- `--grasp-x`
- `--grasp-y`
- `--grasp-z`
- `--plane-rpy-deg`

4. If wrist motion works but hand motion still looks like Magpie aperture only, that is expected for the current MuJoCo model.

## Technical Notes

The RH56 demo preserves the core control idea from `rh56_controller`:

- `width -> closure result`
- closure result gives RH56 finger controls and hand-base pose
- hand-base pose is converted to H1-2 `right_wrist_yaw_link` or `left_wrist_yaw_link`
- H1-2 planner-world coordinates are converted to pelvis frame before calling `/frame_task`

The current lightweight implementation supports line and plane grasps. Cylinder support is intentionally not copied yet because the cylinder planner in `rh56_controller` relies on proximal-joint positions that are not included in the current FK cache.

## Next Development Steps

1. Validate H1-2 wrist trajectory in the Linux MuJoCo/ROS stack.
2. Tune default `--grasp-x/y/z` so the demo starts in a reachable, visible pose.
3. Decide the hand simulation path:
   - Short-term: keep Magpie aperture as a placeholder.
   - Real RH56 sim: port the RH56/Inspire MuJoCo XML from `rh56_controller/h1_mujoco/inspire` and add a bridge that maps `/right_hand_cmd` / `/left_hand_cmd` to the RH56 actuators.
4. Once RH56 hand assets are in MuJoCo, switch the demo default from `--hand-backend magpie` to `--hand-backend inspire-topic` or `both`.
5. Add a documented launch path for the RH56 grasp demo once the first Linux run is verified.

## Current Commit Plan

Suggested commit once reviewed:

```bash
git add RH56_HANDOFF.md \
  core_ws/src/h1_bringup/scripts/rh56_grasp_demo.py \
  core_ws/src/h1_bringup/data/rh56_fk_cache.npz \
  core_ws/src/h1_bringup/setup.py \
  core_ws/src/h1_bringup/package.xml

git commit -m "Add RH56 grasp demo handoff and H1-2 integration stub"
git push -u origin rh56-grasp-integration
```

## Known Mac Note

On the Mac, `docker/scripts/docker_run.sh mujoco` failed with:

```text
docker/scripts/docker_run.sh: line 39: docker: command not found
```

Even after installing Docker Desktop, this stack is expected to be much more reliable on Linux/NVIDIA because the compose files use NVIDIA runtime, CUDA images, and X11 viewer assumptions.
