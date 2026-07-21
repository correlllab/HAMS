"""Pure validation helpers for VLM memory reranking responses."""

from __future__ import annotations

import math
from typing import Any


def validate_rerank_response(
    response: Any,
    candidate_ids: list[str],
) -> tuple[list[str], dict[str, dict]] | None:
    """Accept only a complete ranking and analysis for every recalled candidate."""
    if not isinstance(response, dict):
        return None
    ranked_ids = response.get("ranked_ids")
    if (
        not isinstance(ranked_ids, list)
        or len(ranked_ids) != len(candidate_ids)
        or any(not isinstance(memory_id, str) for memory_id in ranked_ids)
        or len(set(ranked_ids)) != len(ranked_ids)
        or set(ranked_ids) != set(candidate_ids)
    ):
        return None
    raw_analysis = response.get("candidates_analysis")
    if not isinstance(raw_analysis, list) or len(raw_analysis) != len(candidate_ids):
        return None

    expected = set(candidate_ids)
    analysis_by_id: dict[str, dict] = {}
    for item in raw_analysis:
        if not isinstance(item, dict):
            return None
        memory_id = item.get("memory_id")
        confidence = item.get("confidence")
        if (
            not isinstance(memory_id, str)
            or memory_id not in expected
            or memory_id in analysis_by_id
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
            or not isinstance(item.get("object_location"), str)
            or not isinstance(item.get("reasoning"), str)
        ):
            return None
        normalized = dict(item)
        normalized["confidence"] = float(confidence)
        analysis_by_id[memory_id] = normalized
    if set(analysis_by_id) != expected:
        return None
    return list(ranked_ids), analysis_by_id
