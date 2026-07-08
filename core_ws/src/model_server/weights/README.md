# model_server weights

Large model checkpoints, tracked with Git LFS (patterns in `.gitattributes`).
Config YAMLs and this README stay as normal git text.

## SAM3 — `sam3.pt`

Promptable segmentation model loaded by `sam_server` (`SAM3_MODEL`).
Source (gated): https://huggingface.co/facebook/sam3

Not tracked in git — it's too large, so download it yourself and place it at:

    core_ws/src/model_server/weights/sam3.pt

Accept the model license on the HuggingFace page, authenticate, then fetch the
checkpoint into this directory:

    huggingface-cli login                         # one-time; accept the facebook/sam3 license
    cd core_ws/src/model_server/weights
    huggingface-cli download facebook/sam3 --local-dir .cache
    mv .cache/sam3.pt sam3.pt                      # name it exactly sam3.pt (what SAM3_MODEL expects)

Alternatively, set `SAM3_MODEL = None` in `sam_server.py` to let SAM3
auto-download from HuggingFace on first run (also requires HF auth).

## YOLO-World — `yolo_world_battery_best.pt`

Open-vocabulary detection model loaded by `yolo_server` (`YOLO_MODEL`). This is
our fine-tuned YOLO-World checkpoint for the battery workcell (a full Ultralytics
export, e.g. `runs/.../weights/best.pt` renamed), loaded directly at startup from:

    core_ws/src/model_server/weights/yolo_world_battery_best.pt

Base model / source: https://docs.ultralytics.com/models/yolo-world/

If this file is absent, `ultralytics` treats `YOLO_MODEL` as a checkpoint name and
tries to download it — so to run the stock open-vocabulary base instead, set
`YOLO_MODEL = "yolov8x-worldv2.pt"` in `yolo_server.py` (auto-downloaded on first
run). A fine-tuned export saved as a `{'state_dict': ...}` dict is also accepted —
it is loaded (non-strict) on top of the base architecture.

## GraspGenX — `graspgen/release/`

6-DOF grasp generation model (generator + discriminator) loaded by
`graspgen_server` (`GRASPGENX_CHECKPOINT_DIR` -> `release/{gen,dis}`).

- Source: https://huggingface.co/adithyamurali/GraspGenXModel
- License: NVIDIA Open Model License
  (https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)

Layout:

    graspgen/release/gen/{config.yaml, epoch_736.pth}    generator
    graspgen/release/dis/{config.yaml, epoch_1056.pth}   discriminator
