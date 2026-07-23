import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.spatial_memory.quota import (
    completed_episode_ids,
    quota_summary,
    query_counts,
)


class SpatialMemoryQuotaTests(unittest.TestCase):
    def test_estimates_calls_and_safe_episode_prefix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir)
            (dataset / "benchmark_manifest.json").write_text(
                json.dumps({
                    "episodes": [
                        {"episode_id": "episode_000", "query_count": 5},
                        {"episode_id": "episode_001", "query_count": 5},
                        {"episode_id": "episode_002", "query_count": 5},
                        {"episode_id": "episode_003", "query_count": 9},
                    ]
                }),
                encoding="utf-8",
            )

            self.assertEqual(query_counts(dataset, max_episodes=2), [5, 5])
            self.assertEqual(
                quota_summary(dataset, call_limit=20),
                {
                    "scheduled_calls": 24,
                    "selected_episodes": 4,
                    "safe_prefix_episodes": 3,
                    "safe_prefix_calls": 15,
                },
            )

    def test_rejects_invalid_query_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir)
            (dataset / "benchmark_manifest.json").write_text(
                json.dumps({"episodes": [{"episode_id": "bad"}]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "query_count"):
                query_counts(dataset)

    def test_resume_counts_only_unfinished_episode_calls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            dataset.mkdir()
            episodes = [
                {"episode_id": "episode_000", "query_count": 5},
                {"episode_id": "episode_001", "query_count": 7},
                {"episode_id": "episode_002", "query_count": 9},
            ]
            (dataset / "benchmark_manifest.json").write_text(
                json.dumps({"episodes": episodes}),
                encoding="utf-8",
            )
            resume = root / "report"
            checkpoint_root = resume / "checkpoints"
            (checkpoint_root / "episodes").mkdir(parents=True)
            (checkpoint_root / "run.json").write_text(
                json.dumps({
                    "episode_ids": [item["episode_id"] for item in episodes],
                }),
                encoding="utf-8",
            )
            (checkpoint_root / "episodes/episode_000.json").write_text(
                json.dumps({"episode_id": "episode_000"}),
                encoding="utf-8",
            )

            self.assertEqual(completed_episode_ids(resume), {"episode_000"})
            self.assertEqual(
                quota_summary(dataset, call_limit=20, resume_output=resume),
                {
                    "scheduled_calls": 16,
                    "selected_episodes": 2,
                    "safe_prefix_episodes": 2,
                    "safe_prefix_calls": 16,
                },
            )


if __name__ == "__main__":
    unittest.main()
