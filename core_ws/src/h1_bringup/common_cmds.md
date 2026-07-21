# Common commands

Handy `ros2 action send_goal` snippets for driving the H1-2 stack. Run these
from a shell **inside `hams_ros`** with the workspace sourced:

```bash
docker exec -it hams_ros /bin/bash
source /opt/ros/humble/setup.bash
source /home/code/core_ws/install/setup.bash
```

Quick sanity checks:

```bash
ros2 topic hz /lowstate          # sim/robot is alive and publishing state
ros2 action list | grep -E 'skill|frame_task|named_config'
```

**Timeouts:** every `/skill/*` goal takes a `timeout` Duration. **Omit it (or send
zero) and the skill runs to completion** — it then ends only by finishing,
failing, or being canceled, never on a clock. Give a non-zero `timeout` when you
actually want a bound. Ctrl-C on `send_goal` cancels the in-flight goal.

---

## Frontier exploration (`/skill/frontier_explore`)

Autonomous frontier-based exploration over the slam_toolbox `/map`. Needs nav2 +
slam_toolbox up and the robot in **walk mode** so nav2's `/cmd_vel` is followed.
All numeric goal fields accept `0` to fall back to the node default.

```bash
ros2 action send_goal /skill/frontier_explore custom_ros_messages/action/SkillFrontierExplore \
  "{min_frontier_cells: 6, blacklist_radius: 0.4, min_goal_distance: 0.3, goal_timeout: 120.0, timeout: {sec: 3000, nanosec: 0}}" \
  --feedback
```

Field reference:

| Field | Meaning | Default (0 →) |
|---|---|---|
| `min_frontier_cells` | ignore frontier clusters smaller than this | 6 |
| `blacklist_radius` (m) | blacklist a failed goal within this radius | 0.6 |
| `min_goal_distance` (m) | skip frontiers closer than this to the robot | 0.7 |
| `goal_timeout` (s) | cancel a single nav goal after this | 120 |
| `timeout` (Duration) | overall exploration budget | node default |

---

## Grasp (`/skill/grasp`)

Pick up a named object. Does **not** lift after grasping — the skill computes the
approach and grasp width itself. `arm` is `"left"` or `"right"`; `""` (or
`"none"`) auto-selects the arm currently closest to the object.

```bash
ros2 action send_goal /skill/grasp custom_ros_messages/action/SkillGrasp \
  "{target_object: 'vertical fridge handle', arm: 'right', timeout: {sec: 60, nanosec: 0}}" \
  --feedback
```

```bash
# left arm, tabletop object
ros2 action send_goal /skill/grasp custom_ros_messages/action/SkillGrasp \
  "{target_object: 'mug', arm: 'left', timeout: {sec: 60, nanosec: 0}}" \
  --feedback
```

```bash
# top-down + visual servo: auto arm, steep from above (battery-workcell part).
# No timeout -> runs to completion; cancel with Ctrl-C to stop it.
ros2 action send_goal /skill/grasp custom_ros_messages/action/SkillGrasp \
  "{target_object: 'screw', arm: '', top_down: true, visual_servo: true}" \
  --feedback
```

Optional goal fields (omit / `false` / `0` = off):

| Field | Meaning |
|---|---|
| `top_down` | skip GraspGenX; build ONE synthetic grasp above the object — the tier-1 orientation (forward azimuth, fingers across pelvis ±Y, level wrist) pitched `TOP_DOWN_PITCH_DEG` = 80° below horizontal, base backed 0.1146 m off along that approach so the fingers close on the cloud centroid. Still steeper than the arm's ~45° practical reach, and there is no second candidate to fall back to, so an unreachable pose aborts the skill. |
| `min_downward_pitch_deg` | keep only GraspGenX grasps approaching at least this far below horizontal (a *minimum* pitch, not proximity to vertical) |
| `visual_servo` | at the pre-grasp, close the loop on the arm's hand camera: translate (orientation frozen) until the object is centered at `VISUAL_SERVO_DEPTH_M` = 0.10 m, then close **there** instead of running the contact descent. Tolerance 5 mm lateral / 2 mm range (the arm's own IK settles to ~1.2 mm, so tighter than ~4 mm lateral is unreachable and just burns the budget), 90 s budget (~50 corrections). If it can't converge or sees no detection the skill **aborts** — there is no open-loop fallback, so the result tells you the grasp was never aligned. Only works for the 7 YOLO battery classes — the hand detector knows nothing else. |

Small battery-workcell parts (`Bolt`, `BusBar`, `InteriorScrew`, `Nut`,
`OrageCover`, `Screw`, `ScrewHole`) are routed to the head-camera YOLO detector
and the raw bounding-box cloud instead of Gemini + SAM — no goal field needed,
the match is on `target_object`.

The full sequence runs: pre-grasp standoff → contact move (`APPROACH_DIST −
CONTACT_OFFSET` = 7.5 cm along the approach) → force close at
`base.GRIP_FORCE_N`. It does **not** lift afterwards.

---

## Named arm configs (`/named_config`)

Drive the 14 arm joints to a predefined posture (defined in
`h12_ros2_controller/utility/named_config.py`). `plan: false` (or omitted) servos
directly; `plan: true` runs the planner first. `duration` of `0` uses the
controller default settle time.

Available configs: `home`, `t_pose`, `arms_front_45`, `arms_overhead`,
`arms_asym`, `arms_front_yaw`, `elbow_only`, `init_1`, `init_2`, `init_3`.

```bash
# arms down to the home baseline
ros2 action send_goal /named_config custom_ros_messages/action/NamedConfig \
  "{config_name: 'home', duration: {sec: 0, nanosec: 0}}" \
  --feedback
```

```bash
# T-pose, planned, over 4 s
ros2 action send_goal /named_config custom_ros_messages/action/NamedConfig \
  "{plan: true, config_name: 't_pose', duration: {sec: 4, nanosec: 0}}" \
  --feedback
```

---

## Frame task / arm IK (`/frame_task`)

Command one or more end-effector frames to a Cartesian target. Targets are
`geometry_msgs/Pose` in the **`pelvis`** frame; orientation is a quaternion
(`w: 1.0` = identity). `frame_names` and `frame_targets` must be the same length.
Common frames: `left_wrist_yaw_link`, `right_wrist_yaw_link`.

```bash
# move the right wrist to a point in front of the robot (pelvis frame)
ros2 action send_goal /frame_task custom_ros_messages/action/FrameTask \
  "{plan: false,
    frame_names: ['right_wrist_yaw_link'],
    frame_targets: [{position: {x: 0.30, y: -0.20, z: 0.10},
                     orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}],
    duration: {sec: 3, nanosec: 0}}" \
  --feedback
```

```bash
# dual-arm: both wrists to symmetric targets, planned
ros2 action send_goal /frame_task custom_ros_messages/action/FrameTask \
  "{plan: true,
    frame_names: ['left_wrist_yaw_link', 'right_wrist_yaw_link'],
    frame_targets: [{position: {x: 0.24, y: 0.20, z: 0.09},
                     orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}},
                    {position: {x: 0.24, y: -0.20, z: 0.09},
                     orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}],
    duration: {sec: 4, nanosec: 0}}" \
  --feedback
```
