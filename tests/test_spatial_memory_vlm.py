import unittest

from benchmarks.spatial_memory.rerank_validation import validate_rerank_response


def valid_response():
    return {
        "ranked_ids": ["mem_000002", "mem_000001"],
        "candidates_analysis": [
            {
                "memory_id": "mem_000001",
                "object_location": "not visible",
                "confidence": 0.1,
                "reasoning": "Only a countertop is visible.",
            },
            {
                "memory_id": "mem_000002",
                "object_location": "left side of the counter",
                "confidence": 0.95,
                "reasoning": "The target mug is clearly visible.",
            },
        ],
        "reason": "Candidate two shows the target.",
    }


class SpatialMemoryVLMValidationTests(unittest.TestCase):
    def test_accepts_complete_exact_candidate_permutation(self):
        result = validate_rerank_response(
            valid_response(), ["mem_000001", "mem_000002"]
        )
        self.assertIsNotNone(result)
        ranked_ids, analysis = result
        self.assertEqual(ranked_ids, ["mem_000002", "mem_000001"])
        self.assertEqual(analysis["mem_000002"]["confidence"], 0.95)

    def test_rejects_missing_or_duplicate_ranked_ids(self):
        response = valid_response()
        response["ranked_ids"] = ["mem_000002", "mem_000002"]
        self.assertIsNone(
            validate_rerank_response(response, ["mem_000001", "mem_000002"])
        )

    def test_rejects_incomplete_analysis(self):
        response = valid_response()
        response["candidates_analysis"] = response["candidates_analysis"][:1]
        self.assertIsNone(
            validate_rerank_response(response, ["mem_000001", "mem_000002"])
        )

    def test_rejects_out_of_range_confidence(self):
        response = valid_response()
        response["candidates_analysis"][0]["confidence"] = 1.2
        self.assertIsNone(
            validate_rerank_response(response, ["mem_000001", "mem_000002"])
        )


if __name__ == "__main__":
    unittest.main()
