import tempfile
import unittest
from pathlib import Path

from benchmarks.spatial_memory.latest_only_adapter import LatestOnlyAdapter


class SpatialMemoryBaselineTests(unittest.TestCase):
    def test_latest_only_returns_newest_ingested_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter = LatestOnlyAdapter()
            adapter.reset(root, root / "state")
            for frame_idx in range(5):
                adapter.ingest({
                    "observation_id": f"obs_{frame_idx:06d}",
                    "frame_idx": frame_idx,
                    "timestamp": f"2026-01-01T00:00:{frame_idx:02d}+00:00",
                    "robot_pose": [float(frame_idx), 0.0, 0.0],
                })

            results = adapter.query("Find the mug", top_k=3)

            self.assertEqual(
                [item.observation_id for item in results],
                ["obs_000004", "obs_000003", "obs_000002"],
            )
            self.assertEqual([item.score for item in results], [1.0, 0.8, 0.6])
            self.assertTrue(adapter.query_diagnostics()["query_agnostic"])

    def test_latest_only_resets_between_episodes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter = LatestOnlyAdapter()
            adapter.reset(root, root / "state_a")
            adapter.ingest({"observation_id": "obs_000000", "frame_idx": 0})
            adapter.reset(root, root / "state_b")
            self.assertEqual(adapter.query("anything", top_k=3), [])


if __name__ == "__main__":
    unittest.main()
