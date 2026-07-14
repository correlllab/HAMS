# Grasp-synthesis benchmark (RoboCasa CheesyBread)

Compares four parallel-jaw grasp-synthesis methods on the RoboCasa
`CheesyBread` task (single right arm, robot held in place by the elastic-band
tether — no walking). Each episode: home the arms → perceive → synthesize a
grasp → reach → close → lift 15 cm → judge success against ground truth.

| method | perception | synthesis |
|---|---|---|
| `centroid` | head-cam SAM cloud, from home pose | cloud centroid, fixed top-down grasp |
| `topdown_antipodal` | wrist-cam SAM cloud (wrist parked above the object) | top-layer PCA, fingers close across the minor axis |
| `graspgenx` | gemini box → SAM → cloud | GraspGenX 6-DOF grasps, executed as ranked |
| `vlm_judge` | gemini box → SAM → cloud | GraspGenX top-2 + PCA short/long side; Gemini picks from an annotated contact sheet (magpie pickup-pipeline style) |

**Success metric** is ground truth, not vision: the sim's `MeasurementBridge`
publishes every task object's MuJoCo pose on `/robocasa/object_poses`, and an
episode succeeds when the cheese's z rises ≥ 8 cm (`--success-dz`) after the
close+lift and stays there through a 2 s hold. `plan_time_s` (perception +
synthesis) and `exec_time_s` (reach + close + lift) are recorded per episode.

## Moving parts

- `core_ws/src/h12_skills/h12_skills/grasp_benchmark.py` — the episode runner
  (ROS node, `ros2 run h12_skills grasp_benchmark`). Builds on `SkillsBase`
  (frame_task IK, grippers, gemini/sam/graspgen clients). All methods emit
  poses in the GraspGenX gripper-base convention and execute through the same
  drive-`{left,right}_graspgenx_frame` path, so only synthesis differs.
- `h1_robocasa/measurement_bridge.py` — publishes `/robocasa/object_poses`
  (ground truth for the lift check).
- `h1_robocasa/mujoco_ros_bridge.py` — broadcasts TF for the eye-in-hand
  cameras (`pelvis → {left,right}_hand_camera_color_optical_frame`) straight
  from MuJoCo's camera matrices, so the wrist cloud can be expressed in pelvis.
- `run_benchmark.sh` — host orchestrator: fresh sim + fresh bringup per
  (method, seed) episode (sim time restarts with the sim, so bringup nodes
  must restart too), runs the episode, collects JSON.
- `summarize.py` — aggregates `core_ws/benchmark_results/*.json` into a table.

## Prerequisites

1. Images built: `docker/scripts/docker_build.sh robocasa ros`
2. NVIDIA container toolkit installed (compose runs with `runtime: nvidia`).
3. `docker/.env` with a `GEMINI_API_KEY` (vlm_judge + graspgenx detection).
4. SAM3 weights at `core_ws/src/model_server/weights/sam3.pt` (gated
   HuggingFace download — see root README).

## Run

```bash
# all four methods, seeds 42..44 (12 episodes):
benchmarks/grasp_synthesis/run_benchmark.sh

# one method, more seeds:
benchmarks/grasp_synthesis/run_benchmark.sh -m graspgenx -s "42 43 44 45 46"

# summarize existing results:
python3 benchmarks/grasp_synthesis/summarize.py core_ws/benchmark_results --csv results.csv
```

Single episode by hand (sim + bringup already running):

```bash
docker exec -it hams_ros bash
source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash
# bringup for benchmarking: NO rviz / slider GUI (they land on the host X
# display and fight the sim + SAM3 + GraspGenX for the GPU — a full default
# bringup has frozen a 16 GB desktop), no nav (robot is tethered):
ros2 launch h1_bringup h1_sim_bringup.launch.py \
    use_rviz:=false use_sliders:=false use_nav:=false model_visualization:=false
# then, in another shell:
ros2 run h12_skills grasp_benchmark -- --method vlm_judge \
    --object "wedge of cheese" --gt-name cheese \
    --out /home/code/core_ws/benchmark_results/manual.json
```

## Notes / caveats

- The grasp executes in the **pelvis** frame directly (`move_frame_to`), no
  world-frame servoing: the robot is tethered and FAST-LIO/odometry isn't
  required for the benchmark.
- `vlm_judge` degrades gracefully: if GraspGenX returns nothing it judges the
  two PCA candidates; if Gemini fails it defaults to the first candidate.
- Every method returns a ranked candidate list; execution walks it until a
  pre-grasp is IK-reachable (`MAX_ATTEMPTS = 5`).
- CheesyBread's full task success (cheese ON bread) is also recorded as
  `task_success`, but the benchmark's primary metric is the grasp+lift.
