import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.spatial_memory.adapter import MemoryAdapter
from benchmarks.spatial_memory.evaluate import _completed_status, evaluate


class InterruptingAdapter(MemoryAdapter):
    name = "interrupting"

    def reset(self, episode_dir: Path, state_dir: Path) -> None:
        state_dir.mkdir(parents=True)

    def ingest(self, observation: dict) -> None:
        pass

    def query(self, text: str, top_k: int):
        raise KeyboardInterrupt()

    def run_metadata(self) -> dict:
        return {"test_counter": 1}

    def close(self) -> None:
        pass


class SpatialMemoryEvaluateMetadataTests(unittest.TestCase):
    def test_vlm_api_failure_marks_completed_run_as_error(self):
        self.assertEqual(
            _completed_status({"vlm": {
                "failed_calls": 1,
                "api_errors": 3,
                "telemetry_available": True,
            }}),
            "completed_with_errors",
        )
        self.assertEqual(_completed_status({}), "completed")

    def test_interrupted_run_preserves_standalone_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            episode = dataset / "episodes/episode_000"
            (episode / "color").mkdir(parents=True)
            (episode / "robot_xy").mkdir()
            (episode / "color/000000.png").write_bytes(b"image")
            (episode / "robot_xy/000000.txt").write_text("0 0 0\n", encoding="utf-8")
            observation = {
                "observation_id": "obs_000000",
                "frame_idx": 0,
                "timestamp": "2026-01-01T00:00:00+00:00",
                "image_path": "color/000000.png",
                "pose_path": "robot_xy/000000.txt",
                "robot_pose": [0.0, 0.0, 0.0],
            }
            query = {
                "query_id": "static",
                "track": "static",
                "checkpoint_frame": 0,
                "text": "Find the mug",
                "relevant_observation_ids": ["obs_000000"],
                "stale_observation_ids": [],
            }
            (episode / "observations.jsonl").write_text(
                json.dumps(observation) + "\n", encoding="utf-8"
            )
            (episode / "queries.jsonl").write_text(
                json.dumps(query) + "\n", encoding="utf-8"
            )
            (dataset / "benchmark_manifest.json").write_text(
                json.dumps({
                    "benchmark_id": "robocasa_object_relocation_v2",
                    "episodes": [{
                        "episode_id": "episode_000",
                        "frame_count": 1,
                        "query_count": 1,
                    }],
                }),
                encoding="utf-8",
            )
            output = root / "output"
            args = SimpleNamespace(
                dataset=str(dataset),
                output=str(output),
                adapter="interrupting",
                adapter_kwargs="{}",
                top_k=1,
                max_episodes=None,
                keep_state=False,
            )

            with patch(
                "benchmarks.spatial_memory.evaluate._load_adapter",
                return_value=InterruptingAdapter(),
            ), self.assertRaises(KeyboardInterrupt):
                evaluate(args)

            self.assertFalse((output / "results.json").exists())
            metadata = json.loads(
                (output / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "interrupted")
            self.assertEqual(metadata["failure"]["type"], "KeyboardInterrupt")
            self.assertEqual(metadata["adapter"]["test_counter"], 1)


if __name__ == "__main__":
    unittest.main()
