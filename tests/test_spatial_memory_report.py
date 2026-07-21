import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.spatial_memory.report import write_reports


class SpatialMemoryReportTests(unittest.TestCase):
    def test_writes_readable_reports_with_relative_images_and_escaped_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "dataset"
            episode = dataset / "episodes" / "episode_000_mug"
            output = dataset / "reports" / "adapter" / "run"
            (episode / "color").mkdir(parents=True)
            output.mkdir(parents=True)
            (episode / "color" / "000000.png").write_bytes(b"image")
            (episode / "contact_sheet.jpg").write_bytes(b"sheet")
            (episode / "observations.jsonl").write_text(
                json.dumps({
                    "observation_id": "obs_000000",
                    "image_path": "color/000000.png",
                }) + "\n",
                encoding="utf-8",
            )
            manifest = {
                "capture": {"image_size": 512},
                "episodes": [{
                    "episode_id": "episode_000_mug",
                    "target_language": "mug",
                    "surface_a": "counter_a",
                    "surface_b": "counter_b",
                    "visible_frames_lap1": 1,
                    "visible_frames_lap2": 1,
                }]
            }
            report = {
                "adapter": "fake_adapter",
                "created_at": "2026-01-01T00:00:00+00:00",
                "top_k": 1,
                "aggregate": {
                    "episode_count": 1,
                    "static_recall_at_k": 1.0,
                    "live_current_recall_at_k": 1.0,
                    "live_current_top1_accuracy": 1.0,
                    "live_stale_top1_rate": 0.0,
                    "update_lag_frames_at_k": 0,
                    "history_recall_at_k": 0.0,
                },
                "episodes": [{
                    "episode_id": "episode_000_mug",
                    "frame_count": 1,
                    "query_count": 1,
                    "summary": {"static_recall_at_k": 1.0},
                    "queries": [{
                        "query_id": "static_end_lap1",
                        "track": "static",
                        "checkpoint_frame": 0,
                        "text": "Find <the mug>",
                        "query_latency_ms": 1.0,
                        "adapter_diagnostics": {
                            "rerank_attempted": True,
                            "rerank_valid": True,
                            "recall_count": 12,
                        },
                        "faiss_recall_pool_metrics": {
                            "recall_at_k": 1.0,
                            "relevant_coverage_at_k": 1.0,
                        },
                        "relevant_observation_ids": ["obs_000000"],
                        "stale_observation_ids": [],
                        "metrics": {"recall_at_k": 1.0, "stale_top1": 0.0},
                        "candidates": [{
                            "rank": 1,
                            "observation_id": "obs_000000",
                            "score": 0.5,
                            "metadata": {
                                "score_label": "VLM confidence",
                                "faiss_rank": 7,
                                "faiss_score": 0.1234,
                                "rerank_reasoning": "The mug is visible.",
                            },
                        }],
                    }],
                }],
            }

            html_path, markdown_path = write_reports(
                report, manifest, dataset, output
            )
            html_text = html_path.read_text(encoding="utf-8")
            markdown_text = markdown_path.read_text(encoding="utf-8")
            self.assertIn("Find &lt;the mug&gt;", html_text)
            self.assertNotIn("Find <the mug>", html_text)
            self.assertIn("../../../episodes/episode_000_mug/color/000000.png", html_text)
            self.assertIn("Current-location Recall@K", html_text)
            self.assertIn("images: 512×512", html_text)
            self.assertIn("Rank 1", html_text)
            self.assertIn("Frame 000000", html_text)
            self.assertIn("VLM rerank valid · 12 candidates", html_text)
            self.assertIn("FAISS pool: hit", html_text)
            self.assertIn("FAISS Rank 7 · cosine 0.1234", html_text)
            self.assertIn("The mug is visible.", html_text)
            self.assertNotIn("<strong>#1</strong>", html_text)
            self.assertIn("# Spatial-memory benchmark report", markdown_text)
            self.assertIn("Image resolution: 512×512", markdown_text)
            self.assertIn("counter_a` → `counter_b", markdown_text)
            self.assertIn("| valid |", markdown_text)


if __name__ == "__main__":
    unittest.main()
