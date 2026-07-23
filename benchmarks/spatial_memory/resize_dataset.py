#!/usr/bin/env python3
"""Create a paired observation-resolution variant of a frozen benchmark.

The episode order, queries, poses, timestamps, and oracle relevance labels are
copied unchanged. RGB, depth, and camera intrinsics are resized together so the
derived dataset differs from its source only in sensor resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _scale_intrinsics(matrix: list, scale_x: float, scale_y: float) -> list:
    scaled = [[float(value) for value in row] for row in matrix]
    if len(scaled) != 3 or any(len(row) != 3 for row in scaled):
        raise ValueError("camera intrinsics must be a 3x3 matrix")
    scaled[0][0] *= scale_x
    scaled[0][1] *= scale_x
    scaled[0][2] *= scale_x
    scaled[1][0] *= scale_y
    scaled[1][1] *= scale_y
    scaled[1][2] *= scale_y
    return scaled


def _resize_depth(path: Path, size: int) -> None:
    depth = np.load(path, allow_pickle=False)
    if depth.ndim != 2:
        raise ValueError(f"depth image must be 2D: {path}")
    source_height, source_width = depth.shape
    y_indices = np.minimum(
        (np.arange(size, dtype=np.float64) * source_height / size).astype(int),
        source_height - 1,
    )
    x_indices = np.minimum(
        (np.arange(size, dtype=np.float64) * source_width / size).astype(int),
        source_width - 1,
    )
    resized = depth[y_indices[:, None], x_indices[None, :]]
    np.save(path, resized)


def _resize_episode(episode_dir: Path, source_size: int, target_size: int) -> None:
    scale = target_size / source_size
    for image_path in sorted((episode_dir / "color").glob("*.png")):
        with Image.open(image_path) as image:
            if image.size != (source_size, source_size):
                raise ValueError(
                    f"expected {source_size}x{source_size} RGB image: {image_path}"
                )
            image.convert("RGB").resize(
                (target_size, target_size), Image.Resampling.LANCZOS
            ).save(image_path)

    for depth_path in sorted((episode_dir / "depth").glob("*.npy")):
        _resize_depth(depth_path, target_size)

    contact_sheet = episode_dir / "contact_sheet.jpg"
    if contact_sheet.is_file():
        with Image.open(contact_sheet) as image:
            resized_size = (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            )
            image.convert("RGB").resize(
                resized_size, Image.Resampling.LANCZOS
            ).save(contact_sheet, quality=90)

    observations_path = episode_dir / "observations.jsonl"
    observations = []
    with open(observations_path, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            observation = json.loads(line)
            observation["camera_intrinsics"] = _scale_intrinsics(
                observation["camera_intrinsics"], scale, scale
            )
            observations.append(observation)
    observations_path.write_text(
        "".join(json.dumps(item) + "\n" for item in observations),
        encoding="utf-8",
    )

    for metadata_path in sorted((episode_dir / "frame_meta").glob("*.json")):
        metadata = _read_json(metadata_path)
        camera = metadata["camera"]
        camera["width"] = target_size
        camera["height"] = target_size
        camera["intrinsics"] = _scale_intrinsics(
            camera["intrinsics"], scale, scale
        )
        metadata["resolution_ablation_source_size"] = source_size
        _write_json(metadata_path, metadata)

    oracle_path = episode_dir / "oracle" / "episode.json"
    if oracle_path.is_file():
        oracle = _read_json(oracle_path)
        oracle["paired_resolution_ground_truth"] = {
            "source_image_size": source_size,
            "target_image_size": target_size,
            "target_visible_labels_inherited": True,
            "target_pixel_counts_measured_at_source_resolution": True,
        }
        _write_json(oracle_path, oracle)


def resize_dataset(source: Path, output: Path, image_size: int) -> Path:
    """Create one immutable paired dataset and return its output path."""
    source = Path(source).resolve()
    output = Path(output).resolve()
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    manifest_path = source / "benchmark_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"benchmark manifest not found: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("benchmark_id") not in {
        "robocasa_object_relocation_v1",
        "robocasa_object_relocation_v2",
    }:
        raise ValueError("unsupported benchmark dataset")
    source_size = int(manifest.get("capture", {}).get("image_size", 0))
    if source_size <= 0:
        raise ValueError("source manifest does not define a valid image size")
    if image_size >= source_size:
        raise ValueError("derived image_size must be smaller than the source")

    temporary = output.with_name(f".{output.name}.building")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        shutil.copytree(source / "episodes", temporary / "episodes")
        for episode in manifest["episodes"]:
            _resize_episode(
                temporary / "episodes" / episode["episode_id"],
                source_size,
                image_size,
            )

        capture = manifest["capture"]
        capture["image_size"] = image_size
        visibility_fraction = float(capture.get("visibility_threshold_fraction", 0))
        if visibility_fraction > 0:
            capture["effective_visibility_threshold_pixels"] = math.ceil(
                image_size * image_size * visibility_fraction
            )
        manifest["resolution_ablation"] = {
            "source_dataset": source.name,
            "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "source_image_size": source_size,
            "target_image_size": image_size,
            "rgb_resampling": "Pillow LANCZOS",
            "depth_resampling": "nearest-neighbor",
            "episode_queries_poses_and_oracle_labels": "copied unchanged",
        }
        _write_json(temporary / "benchmark_manifest.json", manifest)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-size", required=True, type=int)
    args = parser.parse_args()
    output = resize_dataset(args.source, args.output, args.image_size)
    print(f"[memory-resolution] paired dataset: {output}")


if __name__ == "__main__":
    main()
