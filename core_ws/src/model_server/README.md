# model_server

Self-contained ROS 2 (ament_python) package hosting the project's heavy
model-inference service servers. Each server is a single, dependency-isolated file
— it imports nothing from other workspace perception packages, so they can be
retired independently.

## Executables

| Executable | Service | Interface | Notes |
|---|---|---|---|
| `gemini_server` | `/gemini_query` | `custom_ros_messages/srv/GeminiQuery` | Image/text → Gemini text. Needs `GEMINI_API_KEY` (or `GOOGLE_API_KEY`). |
| `sam_server` | `/sam_segment` | `custom_ros_messages/srv/SamSegment` | SAM3 promptable segmentation. Loads `weights/sam3.pt`. |
| `graspgen_server` | `/graspgen` | `custom_ros_messages/srv/GraspGen` | GraspGenX 6-DOF grasp planning. Loads `weights/graspgen/release/{gen,dis}`. |

Run with `ros2 run model_server <executable>`.

## Weights

All checkpoints live under `weights/`, tracked with Git LFS:

```
weights/
  sam3.pt                                    # SAM3 (facebook/sam3)
  graspgen/release/gen/{config.yaml,epoch_736.pth}   # GraspGenX generator
  graspgen/release/dis/{config.yaml,epoch_1056.pth}  # GraspGenX discriminator
```

Servers resolve weights relative to this package via `__file__` (works with
`colcon build --symlink-install`). Override the root with
`MODEL_SERVER_WEIGHTS_DIR`. GraspGenX additionally honors
`GRASPGENX_CHECKPOINT_DIR` and needs the gripper description staged at
`GRASPGENX_ASSETS_DIR` (default `/opt/graspgenx/assets`, inside the ros container).

## Dependencies

ROS deps are in `package.xml`. The model libraries are pip-installed in the ros
container and have no rosdep keys: `google-genai`, `pillow` (gemini); `torch`,
`pillow`, `sam3` (sam); `torch`, `graspgenx` (graspgen).
