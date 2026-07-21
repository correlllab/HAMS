import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class FakeGemini:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def rerank_memory_candidates(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def raw_candidate(frame_idx, score):
    return SimpleNamespace(
        memory_id=f"mem_{frame_idx:06d}",
        retrieval_score=score,
        frame_idx=frame_idx,
        timestamp=f"2026-01-01T00:00:{frame_idx:02d}+00:00",
        robot_pose=[float(frame_idx), 0.0, 0.0],
        image_path=f"/tmp/{frame_idx:06d}.png",
    )


class SpatialMemoryVLMAdapterTests(unittest.TestCase):
    def setUp(self):
        try:
            from benchmarks.spatial_memory import embodied_agent_vlm_adapter as module
        except ImportError as error:
            self.skipTest(f"EmbodiedAgent runtime is not importable: {error}")
        self.module = module
        self.recalled = [raw_candidate(1, 0.9), raw_candidate(2, 0.8)]

    def make_adapter(self, response):
        adapter = object.__new__(self.module.EmbodiedAgentVLMAdapter)
        adapter.episode_dir = Path("/tmp")
        adapter.memory = object()
        adapter.worker = SimpleNamespace(index_dir=Path("/tmp/index"))
        adapter.model = "siglip_base"
        adapter.recall_k = 12
        adapter.vlm_model = "fake-vlm"
        adapter.gemini = FakeGemini(response)
        adapter._last_query_diagnostics = {}
        return adapter

    def test_promotes_valid_vlm_choice_and_preserves_faiss_rank(self):
        response = {
            "ranked_ids": ["mem_000002", "mem_000001"],
            "candidates_analysis": [
                {
                    "memory_id": "mem_000001",
                    "object_location": "absent",
                    "confidence": 0.1,
                    "reasoning": "No target.",
                },
                {
                    "memory_id": "mem_000002",
                    "object_location": "counter",
                    "confidence": 0.95,
                    "reasoning": "Target visible.",
                },
            ],
        }
        adapter = self.make_adapter(response)
        with patch.object(
            self.module, "retrieve_memory_candidates", return_value=self.recalled
        ) as retrieve:
            result = adapter.query("Find the target", top_k=1)

        self.assertEqual(result[0].observation_id, "obs_000002")
        self.assertEqual(result[0].metadata["faiss_rank"], 2)
        self.assertEqual(result[0].metadata["score_label"], "VLM confidence")
        self.assertTrue(adapter.query_diagnostics()["rerank_valid"])
        self.assertEqual(retrieve.call_args.kwargs["top_k"], 12)

    def test_invalid_vlm_response_falls_back_to_faiss_order(self):
        adapter = self.make_adapter({"ranked_ids": ["mem_000002"]})
        with patch.object(
            self.module, "retrieve_memory_candidates", return_value=self.recalled
        ):
            result = adapter.query("Find the target", top_k=1)

        self.assertEqual(result[0].observation_id, "obs_000001")
        self.assertEqual(result[0].metadata["score_label"], "FAISS cosine (fallback)")
        diagnostics = adapter.query_diagnostics()
        self.assertFalse(diagnostics["rerank_valid"])
        self.assertEqual(diagnostics["fallback_reason"], "invalid_vlm_response")


if __name__ == "__main__":
    unittest.main()
