import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.spatial_memory.adapter import MemoryAdapter, MemoryCandidate
from benchmarks.spatial_memory.evaluate import (
    _completed_status,
    _merge_attempt_metadata,
    evaluate,
)


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


class ResumableAdapter(MemoryAdapter):
    name = "resumable"

    def __init__(self, interrupt_episode: str | None = None):
        self.interrupt_episode = interrupt_episode
        self.current_episode = None
        self.reset_episodes = []
        self.latest_observation = None

    def reset(self, episode_dir: Path, state_dir: Path) -> None:
        self.current_episode = episode_dir.name
        self.reset_episodes.append(self.current_episode)
        self.latest_observation = None
        state_dir.mkdir(parents=True)

    def ingest(self, observation: dict) -> None:
        self.latest_observation = observation

    def query(self, text: str, top_k: int):
        if self.current_episode == self.interrupt_episode:
            raise KeyboardInterrupt()
        return [
            MemoryCandidate(
                observation_id=self.latest_observation["observation_id"],
                score=1.0,
            )
        ]

    def close(self) -> None:
        pass


def write_dataset(root: Path, episode_count: int) -> Path:
    dataset = root / "dataset"
    episode_records = []
    for index in range(episode_count):
        episode_id = f"episode_{index:03d}"
        episode = dataset / "episodes" / episode_id
        (episode / "color").mkdir(parents=True)
        (episode / "robot_xy").mkdir()
        (episode / "color/000000.png").write_bytes(b"image")
        (episode / "robot_xy/000000.txt").write_text(
            "0 0 0\n", encoding="utf-8"
        )
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
        episode_records.append({
            "episode_id": episode_id,
            "frame_count": 1,
            "query_count": 1,
        })
    (dataset / "benchmark_manifest.json").write_text(
        json.dumps({
            "benchmark_id": "robocasa_object_relocation_v2",
            "episodes": episode_records,
        }),
        encoding="utf-8",
    )
    return dataset


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

    def test_resume_metadata_sums_vlm_usage_and_cost_across_attempts(self):
        pricing = {
            "input_usd_per_million_tokens": 1.5,
            "cached_input_usd_per_million_tokens": 0.15,
            "output_usd_per_million_tokens": 9.0,
        }
        attempts = []
        for index, values in enumerate(((100, 25, 1), (200, 50, 0))):
            prompt, output, errors = values
            attempts.append({
                "status": "failed" if index == 0 else "completed",
                "started_at": f"2026-01-01T00:00:0{index}+00:00",
                "completed_at": f"2026-01-01T00:00:0{index + 1}+00:00",
                "wall_time_seconds": 2.0 + index,
                "adapter": {
                    "vlm": {
                        "logical_calls": 1,
                        "api_attempts": 1 + errors,
                        "successful_calls": 1,
                        "responses_with_usage_metadata": 1,
                        "api_errors": errors,
                        "prompt_tokens": prompt,
                        "candidate_tokens": output,
                        "thought_tokens": 0,
                        "total_tokens": prompt + output,
                        "error_types": {"QuotaError": errors} if errors else {},
                        "telemetry_available": True,
                        "pricing_assumption": pricing,
                    }
                },
            })

        merged = _merge_attempt_metadata(
            attempts,
            episode_results=[],
            loaded_episode_count=1,
        )

        usage = merged["adapter"]["vlm"]
        self.assertEqual(merged["wall_time_seconds"], 5.0)
        self.assertEqual(merged["status"], "completed_with_errors")
        self.assertEqual(usage["logical_calls"], 2)
        self.assertEqual(usage["prompt_tokens"], 300)
        self.assertEqual(usage["candidate_tokens"], 75)
        self.assertEqual(usage["error_types"], {"QuotaError": 1})
        self.assertAlmostEqual(
            usage["estimated_standard_cost_usd"],
            (300 * 1.5 + 75 * 9.0) / 1_000_000,
        )

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

    def test_resume_skips_checkpointed_episodes_and_validates_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = write_dataset(root, episode_count=2)
            output = root / "output"
            first_args = SimpleNamespace(
                dataset=str(dataset),
                output=str(output),
                resume=None,
                adapter="resumable",
                adapter_kwargs="{}",
                top_k=1,
                max_episodes=None,
                keep_state=False,
            )
            first_adapter = ResumableAdapter(interrupt_episode="episode_001")
            with patch(
                "benchmarks.spatial_memory.evaluate._load_adapter",
                return_value=first_adapter,
            ), self.assertRaises(KeyboardInterrupt):
                evaluate(first_args)

            self.assertEqual(
                first_adapter.reset_episodes,
                ["episode_000", "episode_001"],
            )
            self.assertTrue(
                (output / "checkpoints/episodes/episode_000.json").is_file()
            )
            self.assertFalse(
                (output / "checkpoints/episodes/episode_001.json").exists()
            )

            mismatched_args = SimpleNamespace(
                **{
                    **vars(first_args),
                    "output": None,
                    "resume": str(output),
                    "top_k": 2,
                }
            )
            with patch(
                "benchmarks.spatial_memory.evaluate._load_adapter",
                return_value=ResumableAdapter(),
            ), self.assertRaisesRegex(RuntimeError, "top_k"):
                evaluate(mismatched_args)

            resumed_args = SimpleNamespace(
                **{
                    **vars(first_args),
                    "output": None,
                    "resume": str(output),
                }
            )
            resumed_adapter = ResumableAdapter()
            with patch(
                "benchmarks.spatial_memory.evaluate._load_adapter",
                return_value=resumed_adapter,
            ), patch(
                "benchmarks.spatial_memory.evaluate.write_reports",
                return_value=(output / "report.html", output / "summary.md"),
            ):
                self.assertEqual(evaluate(resumed_args), output)

            self.assertEqual(resumed_adapter.reset_episodes, ["episode_001"])
            report = json.loads(
                (output / "results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [item["episode_id"] for item in report["episodes"]],
                ["episode_000", "episode_001"],
            )
            resumption = report["run_metadata"]["resumption"]
            self.assertTrue(resumption["resumed"])
            self.assertEqual(resumption["attempt_count"], 2)
            self.assertEqual(resumption["loaded_episode_count_this_attempt"], 1)
            self.assertEqual(resumption["checkpointed_episode_count"], 2)
            self.assertEqual(
                len(list((output / "checkpoints/attempts").glob("*.json"))),
                2,
            )


if __name__ == "__main__":
    unittest.main()
