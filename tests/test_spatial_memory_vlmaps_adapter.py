import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"RGB-D test runtime is unavailable: {exc}") from exc

from benchmarks.spatial_memory.vlmaps_adapter import (
    VLMapsAdapter,
    _semantic_query,
)


class _FakeLSegBackend:
    name = "fake_lseg"

    def encode_image(self, rgb):
        features = np.zeros((*rgb.shape[:2], 2), dtype=np.float32)
        features[..., 0] = 1.0
        return features

    def encode_text(self, text):
        if text == "other":
            return np.asarray([0.0, 1.0], dtype=np.float32)
        return np.asarray([1.0, 0.0], dtype=np.float32)


class SpatialMemoryVLMapsAdapterTests(unittest.TestCase):
    def test_normalizes_benchmark_instruction_to_object_phrase(self):
        self.assertEqual(
            _semantic_query(
                "Find the current location of the red mug; prefer newest evidence."
            ),
            "red mug",
        )
        self.assertEqual(
            _semantic_query("Where was the bowl before it moved?"),
            "bowl",
        )

    def test_incremental_rgbd_fusion_returns_world_locations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            episode = Path(temp_dir) / "episode"
            state = Path(temp_dir) / "state"
            (episode / "color").mkdir(parents=True)
            (episode / "depth").mkdir()
            adapter = VLMapsAdapter(
                cell_size=1.0,
                depth_sample_rate=1,
                backend=_FakeLSegBackend(),
            )
            adapter.reset(episode, state)

            intrinsics = [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]
            for frame_idx, translation_x in enumerate((0.0, 3.0)):
                stem = f"{frame_idx:06d}"
                Image.fromarray(
                    np.full((2, 2, 3), 127, dtype=np.uint8)
                ).save(episode / "color" / f"{stem}.png")
                np.save(
                    episode / "depth" / f"{stem}.npy",
                    np.ones((2, 2), dtype=np.float32),
                )
                camera_to_world = np.eye(4)
                camera_to_world[0, 3] = translation_x
                adapter.ingest({
                    "observation_id": f"obs_{stem}",
                    "image_path": f"color/{stem}.png",
                    "depth_path": f"depth/{stem}.npy",
                    "camera_intrinsics": intrinsics,
                    "camera_to_world": camera_to_world.tolist(),
                    "camera_coordinate_frame": "opencv_x_right_y_down_z_forward",
                })

            candidates = adapter.query("Find the current location of the mug", top_k=2)
            self.assertEqual(len(candidates), 2)
            self.assertEqual(
                {item.observation_id for item in candidates},
                {"obs_000000", "obs_000001"},
            )
            for candidate in candidates:
                self.assertEqual(
                    len(candidate.metadata["predicted_world_xyz"]), 3
                )
                self.assertEqual(candidate.metadata["feature_backend"], "fake_lseg")
                self.assertNotIn("confidence_0_1", candidate.metadata)
            diagnostics = adapter.query_diagnostics()
            self.assertTrue(diagnostics["incremental_3d_fusion"])
            self.assertFalse(diagnostics["dynamic_point_removal"])
            self.assertGreaterEqual(diagnostics["component_count"], 2)

    def test_rejects_rgb_only_v1_observation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = VLMapsAdapter(backend=_FakeLSegBackend())
            root = Path(temp_dir)
            adapter.reset(root, root / "state")
            with self.assertRaisesRegex(RuntimeError, "benchmark-v2 RGB-D"):
                adapter.ingest({
                    "observation_id": "obs_000000",
                    "image_path": "color/000000.png",
                })


if __name__ == "__main__":
    unittest.main()
