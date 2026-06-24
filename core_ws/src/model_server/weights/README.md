# model_server weights

Large model checkpoints, tracked with Git LFS (patterns in `.gitattributes`).
Config YAMLs and this README stay as normal git text.

## SAM3 — `sam3.pt`

Promptable segmentation model loaded by `sam_server` (`SAM3_MODEL`).
Source: https://huggingface.co/facebook/sam3

## GraspGenX — `graspgen/release/`

6-DOF grasp generation model (generator + discriminator) loaded by
`graspgen_server` (`GRASPGENX_CHECKPOINT_DIR` -> `release/{gen,dis}`).

- Source: https://huggingface.co/adithyamurali/GraspGenXModel
- License: NVIDIA Open Model License
  (https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)

Layout:

    graspgen/release/gen/{config.yaml, epoch_736.pth}    generator
    graspgen/release/dis/{config.yaml, epoch_1056.pth}   discriminator
