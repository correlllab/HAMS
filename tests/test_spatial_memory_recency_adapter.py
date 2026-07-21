import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def raw_candidate(frame_idx, score):
    return SimpleNamespace(
        memory_id=f"mem_{frame_idx:06d}",
        retrieval_score=score,
        frame_idx=frame_idx,
        timestamp=f"2026-01-01T00:00:{frame_idx:02d}+00:00",
        robot_pose=[float(frame_idx), 0.0, 0.0],
        image_path=f"/tmp/{frame_idx:06d}.png",
    )


class SpatialMemoryRecencyAdapterTests(unittest.TestCase):
    def setUp(self):
        try:
            from benchmarks.spatial_memory import embodied_agent_recency_adapter as module
        except ImportError as error:
            self.skipTest(f"EmbodiedAgent runtime is not importable: {error}")
        self.module = module

    def test_recency_prior_can_promote_newer_recalled_frames(self):
        adapter = object.__new__(self.module.EmbodiedAgentRecencyAdapter)
        adapter.episode_dir = Path("/tmp")
        adapter.memory = object()
        adapter.worker = SimpleNamespace(index_dir=Path("/tmp/index"))
        adapter.model = "siglip_base"
        adapter.recall_k = 12
        adapter.recency_weight = 0.75
        adapter._latest_frame_idx = 9
        adapter._last_query_diagnostics = {}
        recalled = [
            raw_candidate(0, 0.90),
            raw_candidate(5, 0.89),
            raw_candidate(9, 0.88),
        ]

        with patch.object(
            self.module, "retrieve_memory_candidates", return_value=recalled
        ) as retrieve:
            result = adapter.query("Find the current mug", top_k=2)

        self.assertEqual(
            [item.observation_id for item in result],
            ["obs_000009", "obs_000005"],
        )
        self.assertEqual(result[0].metadata["faiss_rank"], 3)
        self.assertEqual(result[0].metadata["normalized_recency_score"], 1.0)
        self.assertEqual(retrieve.call_args.kwargs["top_k"], 12)
        diagnostics = adapter.query_diagnostics()
        self.assertEqual(diagnostics["faiss_order"][0], "mem_000000")
        self.assertEqual(diagnostics["final_order"][0], "mem_000009")


if __name__ == "__main__":
    unittest.main()
