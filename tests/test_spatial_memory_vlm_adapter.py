import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class FakeGemini:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.usage = {
            "logical_calls": 0,
            "api_attempts": 0,
            "successful_calls": 0,
            "responses_with_usage_metadata": 0,
            "prompt_tokens": 0,
            "candidate_tokens": 0,
            "thought_tokens": 0,
            "total_tokens": 0,
        }

    def rerank_memory_candidates(self, **kwargs):
        self.calls.append(kwargs)
        self.usage.update({
            "logical_calls": self.usage["logical_calls"] + 1,
            "api_attempts": self.usage["api_attempts"] + 1,
            "successful_calls": self.usage["successful_calls"] + 1,
            "responses_with_usage_metadata": (
                self.usage["responses_with_usage_metadata"] + 1
            ),
            "prompt_tokens": self.usage["prompt_tokens"] + 100,
            "candidate_tokens": self.usage["candidate_tokens"] + 20,
            "thought_tokens": self.usage["thought_tokens"] + 5,
            "total_tokens": self.usage["total_tokens"] + 125,
        })
        return self.response

    def usage_summary(self):
        return dict(self.usage)


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
        adapter.vlm_model = "gemini-3.5-flash"
        with patch.object(
            self.module, "retrieve_memory_candidates", return_value=self.recalled
        ) as retrieve:
            result = adapter.query("Find the target", top_k=1)

        self.assertEqual(result[0].observation_id, "obs_000002")
        self.assertEqual(result[0].metadata["faiss_rank"], 2)
        self.assertEqual(result[0].metadata["score_label"], "VLM confidence")
        self.assertTrue(adapter.query_diagnostics()["rerank_valid"])
        self.assertEqual(retrieve.call_args.kwargs["top_k"], 12)
        diagnostics = adapter.query_diagnostics()
        self.assertEqual(diagnostics["vlm_usage_delta"]["logical_calls"], 1)
        self.assertEqual(diagnostics["vlm_usage_delta"]["prompt_tokens"], 100)
        metadata = adapter.run_metadata()
        self.assertEqual(metadata["vlm"]["total_tokens"], 125)
        self.assertAlmostEqual(
            metadata["vlm"]["estimated_standard_cost_usd"],
            (100 * 1.5 + 25 * 9.0) / 1_000_000,
        )

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
