import unittest

from benchmarks.spatial_memory.metrics import score_query, summarize_episode


class SpatialMemoryMetricTests(unittest.TestCase):
    def test_live_query_distinguishes_current_and_stale_results(self):
        query = {
            "checkpoint_frame": 4,
            "relevant_observation_ids": ["obs_000004"],
            "stale_observation_ids": ["obs_000003"],
        }
        metrics = score_query(
            query,
            ["obs_000003", "obs_000004", "obs_000002"],
            top_k=3,
        )
        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertEqual(metrics["relevant_coverage_at_k"], 1.0)
        self.assertEqual(metrics["checkpoint_recall_at_k"], 1.0)
        self.assertEqual(metrics["top1_relevant"], 0.0)
        self.assertEqual(metrics["stale_top1"], 1.0)
        self.assertAlmostEqual(metrics["stale_fraction_at_k"], 1.0 / 3.0)

    def test_update_lag_starts_at_first_visible_current_frame(self):
        results = [
            {
                "track": "live_current",
                "checkpoint_frame": 7,
                "first_visible_current_frame": 7,
                "query_latency_ms": 2.0,
                "candidates": [],
                "metrics": {
                    "recall_at_k": 0.0,
                    "top1_relevant": 0.0,
                    "reciprocal_rank": 0.0,
                    "stale_top1": 1.0,
                    "stale_fraction_at_k": 0.5,
                },
            },
            {
                "track": "live_current",
                "checkpoint_frame": 9,
                "first_visible_current_frame": 7,
                "query_latency_ms": 3.0,
                "candidates": [],
                "metrics": {
                    "recall_at_k": 1.0,
                    "top1_relevant": 1.0,
                    "reciprocal_rank": 1.0,
                    "stale_top1": 0.0,
                    "stale_fraction_at_k": 0.0,
                },
            },
        ]
        summary = summarize_episode(results, [10.0, 20.0])
        self.assertEqual(summary["update_lag_frames_at_k"], 2)
        self.assertEqual(summary["update_lag_frames_top1"], 2)
        self.assertEqual(summary["live_stale_top1_rate"], 0.5)

    def test_reports_vlm_valid_and_fallback_rates(self):
        results = []
        for rerank_valid in (True, False):
            results.append({
                "track": "absent",
                "checkpoint_frame": 0,
                "query_latency_ms": 1.0,
                "candidates": [],
                "adapter_diagnostics": {
                    "rerank_attempted": True,
                    "rerank_valid": rerank_valid,
                },
                "faiss_recall_pool_metrics": {
                    "recall_at_k": 1.0,
                    "relevant_coverage_at_k": 0.5,
                },
                "metrics": {
                    "recall_at_k": None,
                    "top1_relevant": None,
                    "reciprocal_rank": 0.0,
                    "stale_top1": 0.0,
                    "stale_fraction_at_k": 0.0,
                },
            })
        summary = summarize_episode(results, [1.0])
        self.assertEqual(summary["vlm_valid_response_rate"], 0.5)
        self.assertEqual(summary["vlm_fallback_rate"], 0.5)
        self.assertEqual(summary["vlm_faiss_recall_pool_hit_rate"], 1.0)
        self.assertEqual(summary["vlm_faiss_recall_pool_coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
