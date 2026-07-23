import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.spatial_memory.compare import (
    _metric_stats,
    compare,
    validate_comparable,
)


def fake_report(path: Path, adapter_spec: str, values: list[float]) -> dict:
    episodes = []
    for index, value in enumerate(values):
        episodes.append({
            "episode_id": f"episode_{index:03d}_mug",
            "summary": {
                "static_recall_at_k": value,
                "query_latency_ms_p50": 2.0 + index,
            },
            "queries": [{
                "query_id": "static_end_lap1",
                "track": "static",
                "text": "Find the mug",
                "checkpoint_frame": 7,
                "relevant_observation_ids": ["obs_000000"],
                "stale_observation_ids": [],
            }],
        })
    return {
        "schema_version": 1,
        "benchmark_id": "robocasa_object_relocation_v1",
        "dataset": str(path.parent),
        "adapter": adapter_spec,
        "adapter_spec": adapter_spec,
        "adapter_kwargs": {},
        "top_k": 3,
        "created_at": "2026-01-01T00:00:00+00:00",
        "aggregate": {"episode_count": len(episodes)},
        "episodes": episodes,
    }


class SpatialMemoryComparisonTests(unittest.TestCase):
    def test_nonnegative_metric_confidence_interval_is_clamped(self):
        for metric, kind in (
            ("update_lag_frames_at_k", "frames"),
            ("static_location_error_m_top1", "meters"),
        ):
            with self.subTest(kind=kind):
                report = {
                    "episodes": [
                        {"summary": {metric: value}}
                        for value in (0.0, 0.0, 1.0)
                    ]
                }
                stats = _metric_stats(report, metric, kind)
                self.assertEqual(stats["ci95_low"], 0.0)

    def test_writes_comparison_with_confidence_intervals(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "dataset"
            dataset.mkdir()
            paths = []
            for adapter, values in (
                ("latest_only", [0.0, 1.0]),
                ("embodied_agent", [1.0, 1.0]),
            ):
                path = dataset / f"{adapter}.json"
                path.write_text(
                    json.dumps(fake_report(path, adapter, values)), encoding="utf-8"
                )
                paths.append(path)

            output = compare(dataset, paths, dataset / "comparison")
            payload = json.loads(
                (output / "comparison.json").read_text(encoding="utf-8")
            )
            latest_stats = payload["methods"][0]["metrics"]["static_recall_at_k"]
            self.assertEqual(latest_stats["mean"], 0.5)
            self.assertEqual(latest_stats["count"], 2)
            self.assertIsNotNone(latest_stats["ci95_low"])
            self.assertIn("Latest-only", (output / "comparison.html").read_text())
            self.assertIn("95% CI", (output / "comparison.md").read_text())

    def test_rejects_different_episode_sets(self):
        first = fake_report(Path("a.json"), "latest_only", [0.0, 1.0])
        second = fake_report(Path("b.json"), "embodied_agent", [1.0])
        with self.assertRaisesRegex(ValueError, "same episodes"):
            validate_comparable([first, second])

    def test_custom_labels_distinguish_resolution_variants(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "dataset"
            dataset.mkdir()
            paths = []
            for resolution in (256, 512):
                path = dataset / f"faiss_{resolution}.json"
                path.write_text(
                    json.dumps(fake_report(path, "embodied_agent", [1.0, 1.0])),
                    encoding="utf-8",
                )
                paths.append(path)
            output = compare(
                dataset,
                paths,
                dataset / "resolution_comparison",
                labels=["FAISS (256x256)", "FAISS (512x512)"],
            )
            payload = json.loads(
                (output / "comparison.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [method["label"] for method in payload["methods"]],
                ["FAISS (256x256)", "FAISS (512x512)"],
            )

    def test_custom_label_count_must_match_results(self):
        first = fake_report(Path("a.json"), "latest_only", [1.0])
        second = fake_report(Path("b.json"), "embodied_agent", [1.0])
        from benchmarks.spatial_memory.compare import build_comparison

        with self.assertRaisesRegex(ValueError, "labels"):
            build_comparison(
                [first, second], Path("."), labels=["only one label"]
            )


if __name__ == "__main__":
    unittest.main()
