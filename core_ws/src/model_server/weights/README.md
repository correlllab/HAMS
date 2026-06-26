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

## GraspGenX — `graspgen/release/`

6-DOF grasp generation model (generator + discriminator) loaded by
`graspgen_server` (`GRASPGENX_CHECKPOINT_DIR` -> `release/{gen,dis}`).

- Source: https://huggingface.co/adithyamurali/GraspGenXModel
- License: NVIDIA Open Model License
  (https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)

Layout:

    graspgen/release/gen/{config.yaml, epoch_736.pth}    generator
    graspgen/release/dis/{config.yaml, epoch_1056.pth}   discriminator
