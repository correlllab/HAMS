import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from benchmarks.spatial_memory.resize_dataset import resize_dataset


class SpatialMemoryResizeDatasetTests(unittest.TestCase):
    def _source_dataset(self, root: Path) -> Path:
        source = root / "source_4px"
        episode = source / "episodes" / "episode_000_mug"
        for relative in ("color", "depth", "frame_meta", "oracle", "robot_xy"):
            (episode / relative).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), color="red").save(episode / "color/000000.png")
        Image.new("RGB", (8, 4), color="blue").save(
            episode / "contact_sheet.jpg"
        )
        np.save(episode / "depth/000000.npy", np.arange(16).reshape(4, 4))
        (episode / "robot_xy/000000.txt").write_text("1 2 3\n", encoding="utf-8")
        observation = {
            "observation_id": "obs_000000",
            "frame_idx": 0,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "image_path": "color/000000.png",
            "depth_path": "depth/000000.npy",
            "pose_path": "robot_xy/000000.txt",
            "robot_pose": [1, 2, 3],
            "camera_intrinsics": [[4, 0, 2], [0, 4, 2], [0, 0, 1]],
        }
        (episode / "observations.jsonl").write_text(
            json.dumps(observation) + "\n", encoding="utf-8"
        )
        query = {
            "query_id": "static",
            "track": "static",
            "checkpoint_frame": 0,
            "text": "Find the mug",
            "relevant_observation_ids": ["obs_000000"],
            "stale_observation_ids": [],
        }
        (episode / "queries.jsonl").write_text(
            json.dumps(query) + "\n", encoding="utf-8"
        )
        frame_metadata = {
            "camera": {
                "width": 4,
                "height": 4,
                "intrinsics": [[4, 0, 2], [0, 4, 2], [0, 0, 1]],
            }
        }
        (episode / "frame_meta/000000.json").write_text(
            json.dumps(frame_metadata), encoding="utf-8"
        )
        oracle = {"frames": [{"target_pixel_count": 4, "target_visible": True}]}
        (episode / "oracle/episode.json").write_text(
            json.dumps(oracle), encoding="utf-8"
        )
        manifest = {
            "schema_version": 2,
            "benchmark_id": "robocasa_object_relocation_v2",
            "capture": {
                "image_size": 4,
                "visibility_threshold_fraction": 0.1,
                "effective_visibility_threshold_pixels": 2,
            },
            "episode_count": 1,
            "episodes": [{"episode_id": "episode_000_mug", "query_count": 1}],
        }
        (source / "benchmark_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return source

    def test_creates_paired_dataset_with_scaled_sensor_geometry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._source_dataset(root)
            output = resize_dataset(source, root / "derived_2px", 2)
            episode = output / "episodes/episode_000_mug"

            with Image.open(episode / "color/000000.png") as image:
                self.assertEqual(image.size, (2, 2))
            self.assertEqual(
                np.load(episode / "depth/000000.npy").shape, (2, 2)
            )
            observation = json.loads(
                (episode / "observations.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(
                observation["camera_intrinsics"],
                [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]],
            )
            manifest = json.loads(
                (output / "benchmark_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["capture"]["image_size"], 2)
            self.assertEqual(
                manifest["resolution_ablation"]["source_image_size"], 4
            )
            self.assertEqual(
                (episode / "queries.jsonl").read_text(encoding="utf-8"),
                (source / "episodes/episode_000_mug/queries.jsonl").read_text(
                    encoding="utf-8"
                ),
            )
            oracle = json.loads(
                (episode / "oracle/episode.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                oracle["paired_resolution_ground_truth"]
                ["target_visible_labels_inherited"]
            )

    def test_refuses_to_replace_existing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._source_dataset(root)
            output = root / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                resize_dataset(source, output, 2)


if __name__ == "__main__":
    unittest.main()
