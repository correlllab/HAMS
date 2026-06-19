#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Headless GraspGenX inference smoke test — NO ROS, NO viser.

Loads the released GraspGenX checkpoints (downloaded into h12_skills/weights/) and
runs the grasp planner on a shipped sample point cloud, printing the resulting
6-DOF grasps. Used to validate that the model loads + runs inference *inside the
ros container* (outside the ROS ecosystem) before wiring up ROS integration.

Run inside the ros container, on GPU, e.g.:
    export GRASPGENX_CHECKPOINT_DIR=/home/code/core_ws/src/h12_skills/weights
    python3 /home/code/core_ws/src/h12_skills/graspgenx_smoke_test.py
"""
import argparse
import inspect
import os
import sys

import numpy as np
import trimesh.transformations as tra


def main() -> int:
    default_ckpt_root = os.environ.get(
        "GRASPGENX_CHECKPOINT_DIR",
        "/home/code/core_ws/src/h12_skills/weights",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", default=os.path.join(default_ckpt_root, "release"),
                    help="checkpoint root containing gen/ and dis/")
    ap.add_argument("--assets_dir", default="/opt/graspgenx/assets")
    ap.add_argument("--sample_data_dir",
                    default="/opt/graspgenx/assets/sample_data/object_pc")
    ap.add_argument("--gripper_name", default="parallel_2f_v1_1002")
    ap.add_argument("--planner", default="graspmoe", choices=["graspmoe", "diffusion"])
    ap.add_argument("--num_grasps", type=int, default=200)
    ap.add_argument("--max_items", type=int, default=1)
    args = ap.parse_args()

    import torch
    print(f"torch {torch.__version__} | cuda? {torch.cuda.is_available()} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'}")
    print(f"arch_list: {torch.cuda.get_arch_list() if torch.cuda.is_available() else '-'}")

    from graspgenx.grasp_server import GraspGenXSampler
    from graspgenx.samplers import run_planner_on_object
    from graspgenx.utils.checkpoint_io import load_model_cfg
    from graspgenx.utils.scene_loaders import collect_object_items

    print(f"Loading model cfg from {args.checkpoints} ...")
    model_cfg = load_model_cfg(
        os.path.join(args.checkpoints, "gen"),
        os.path.join(args.checkpoints, "dis"),
        None, None,
    )
    print(f"Building GraspGenXSampler (gripper={args.gripper_name}) ...")
    sampler = GraspGenXSampler(model_cfg, args.gripper_name, assets_dir=args.assets_dir)

    items = collect_object_items(args.sample_data_dir, min_obj_points=100)
    print(f"Found {len(items)} sample object(s) in {args.sample_data_dir}")
    if not items:
        print("NO SAMPLES FOUND")
        return 2

    # Build planner kwargs, filtered to whatever run_planner_on_object accepts.
    candidate_kwargs = dict(
        planner=args.planner, grasp_threshold=-1.0, num_grasps=args.num_grasps,
        topk_num_grasps=100, moe_num_yaws=36, moe_z_offsets_cm=(-2.0, 0.0),
        moe_outlier_threshold=0.014, moe_outlier_k=20, moe_obb_mode="advanced",
        moe_skip_obb_rule="auto", moe_obb_density="dense-topandside",
        moe_obb_position_spacing_cm=1.0,
    )
    sig = inspect.signature(run_planner_on_object)
    kwargs = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}

    n_ok = 0
    for item in items[:args.max_items]:
        pc = np.asarray(item["pc"], dtype=np.float32)
        T = tra.translation_matrix(-pc.mean(axis=0))
        pc_centered = tra.transform_points(pc, T)
        out = run_planner_on_object(pc_centered, sampler, **kwargs)
        grasps = np.asarray(out[0]) if isinstance(out, (tuple, list)) else np.asarray(out)
        conf = np.asarray(out[1]) if isinstance(out, (tuple, list)) and len(out) > 1 else None
        if grasps.shape[0] > 0:
            n_ok += 1
            crange = f"[{conf.min():.3f}, {conf.max():.3f}]" if conf is not None and conf.size else "n/a"
            print(f"[{item['name']}] grasps={grasps.shape} conf={crange}")
            print("  grasp[0] (4x4):\n" + np.array2string(grasps[0], precision=3, suppress_small=True))
        else:
            print(f"[{item['name']}] grasps=0")

    if n_ok > 0:
        print(f"\nSUCCESS: produced grasps for {n_ok}/{min(args.max_items, len(items))} object(s)")
        return 0
    print("\nFAIL: no grasps produced")
    return 3


if __name__ == "__main__":
    sys.exit(main())
