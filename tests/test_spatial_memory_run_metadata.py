import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.spatial_memory.run_metadata import RunMetadataCollector


class SpatialMemoryRunMetadataTests(unittest.TestCase):
    def test_collects_workload_latency_storage_and_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            episode_dir = dataset / "episodes/episode_000/color"
            episode_dir.mkdir(parents=True)
            (dataset / "benchmark_manifest.json").write_text("{}", encoding="utf-8")
            (episode_dir / "000000.png").write_bytes(b"rgb")
            state = root / "state/episode_000/index"
            state.mkdir(parents=True)
            (state / "vectors.bin").write_bytes(b"index")
            records = [{"episode_id": "episode_000", "frame_count": 1, "query_count": 1}]
            results = [{
                "frame_count": 1,
                "query_count": 1,
                "adapter_reset_latency_ms": 1.0,
                "ingest_latency_ms": [2.0],
                "queries": [{"query_latency_ms": 3.0}],
            }]

            with patch(
                "benchmarks.spatial_memory.run_metadata._reset_gpu_peak"
            ), patch(
                "benchmarks.spatial_memory.run_metadata._gpu_metadata",
                return_value={"available": False, "device_count": 0, "devices": []},
            ), patch(
                "benchmarks.spatial_memory.run_metadata._git_state",
                return_value={"commit": "abc", "dirty": False, "dirty_entry_count": 0},
            ):
                collector = RunMetadataCollector(dataset, records, root)
                metadata = collector.finish(
                    status="failed",
                    episode_results=results,
                    state_root=root / "state",
                    adapter_metadata={"vlm": {"logical_calls": 1}},
                    failure=RuntimeError("quota exhausted"),
                )

            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["workload"]["completed_query_count"], 1)
            self.assertEqual(metadata["latency"]["adapter_reset"]["p50_ms"], 1.0)
            self.assertEqual(metadata["latency"]["ingest"]["p50_ms"], 2.0)
            self.assertEqual(metadata["latency"]["query"]["p95_ms"], 3.0)
            self.assertEqual(
                metadata["resources"]["storage"]["index_bytes"], len(b"index")
            )
            self.assertEqual(metadata["failure"]["type"], "RuntimeError")
            self.assertEqual(metadata["adapter"]["vlm"]["logical_calls"], 1)
            json.dumps(metadata)


if __name__ == "__main__":
    unittest.main()
