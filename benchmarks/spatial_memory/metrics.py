"""Pure metric calculations for spatial-memory benchmark query records."""

from __future__ import annotations

import math
import statistics


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def score_query(query: dict, candidate_ids: list[str], top_k: int) -> dict:
    ranked = candidate_ids[:top_k]
    relevant = set(query.get("relevant_observation_ids", []))
    stale = set(query.get("stale_observation_ids", []))
    relevant_ranks = [
        rank for rank, observation_id in enumerate(ranked, start=1)
        if observation_id in relevant
    ]
    stale_count = sum(observation_id in stale for observation_id in ranked)
    relevant_count = sum(observation_id in relevant for observation_id in ranked)
    checkpoint_id = f"obs_{int(query['checkpoint_frame']):06d}"
    checkpoint_is_relevant = checkpoint_id in relevant
    return {
        "recall_at_k": float(bool(relevant_ranks)) if relevant else None,
        "relevant_coverage_at_k": (
            relevant_count / len(relevant) if relevant else None
        ),
        "checkpoint_recall_at_k": (
            float(checkpoint_id in ranked) if checkpoint_is_relevant else None
        ),
        "checkpoint_top1": (
            float(bool(ranked) and ranked[0] == checkpoint_id)
            if checkpoint_is_relevant else None
        ),
        "top1_relevant": float(bool(ranked) and ranked[0] in relevant) if relevant else None,
        "reciprocal_rank": 1.0 / relevant_ranks[0] if relevant_ranks else 0.0,
        "stale_top1": float(bool(ranked) and ranked[0] in stale),
        "stale_fraction_at_k": stale_count / len(ranked) if ranked else 0.0,
    }


def _mean(records: list[dict], key: str) -> float | None:
    values = [record[key] for record in records if record.get(key) is not None]
    return statistics.fmean(values) if values else None


def summarize_episode(query_results: list[dict], ingestion_ms: list[float]) -> dict:
    by_track: dict[str, list[dict]] = {}
    for result in query_results:
        by_track.setdefault(result["track"], []).append(result)

    live = by_track.get("live_current", [])
    update_lag_at_k = None
    update_lag_top1 = None
    if live:
        first_visible = int(live[0]["first_visible_current_frame"])
        successful_at_k = [item for item in live if item["metrics"]["recall_at_k"] == 1.0]
        successful_top1 = [item for item in live if item["metrics"]["top1_relevant"] == 1.0]
        if successful_at_k:
            update_lag_at_k = min(
                int(item["checkpoint_frame"]) - first_visible for item in successful_at_k
            )
        if successful_top1:
            update_lag_top1 = min(
                int(item["checkpoint_frame"]) - first_visible for item in successful_top1
            )

    def track_metric(track: str, key: str) -> float | None:
        return _mean(
            [item["metrics"] for item in by_track.get(track, [])], key
        )

    query_latencies = [float(item["query_latency_ms"]) for item in query_results]
    absent_scores = [
        float(item["candidates"][0]["score"])
        for item in by_track.get("absent", []) if item["candidates"]
    ]
    absent_confidences = [
        float(item["candidates"][0]["metadata"]["confidence_0_1"])
        for item in by_track.get("absent", [])
        if item.get("candidates")
        and isinstance(item["candidates"][0].get("metadata"), dict)
        and item["candidates"][0]["metadata"].get("confidence_0_1") is not None
    ]
    vlm_queries = [
        item for item in query_results
        if item.get("adapter_diagnostics", {}).get("rerank_attempted") is True
    ]
    vlm_valid = [
        float(item["adapter_diagnostics"].get("rerank_valid") is True)
        for item in vlm_queries
    ]
    recall_pool_records = [
        item["faiss_recall_pool_metrics"]
        for item in vlm_queries
        if (item.get("faiss_recall_pool_metrics") or {}).get("recall_at_k")
        is not None
    ]
    return {
        "static_recall_at_k": track_metric("static", "recall_at_k"),
        "static_coverage_at_k": track_metric("static", "relevant_coverage_at_k"),
        "static_mrr": track_metric("static", "reciprocal_rank"),
        "live_current_recall_at_k": track_metric("live_current", "recall_at_k"),
        "live_current_coverage_at_k": track_metric(
            "live_current", "relevant_coverage_at_k"
        ),
        "live_latest_visible_frame_recall_at_k": track_metric(
            "live_current", "checkpoint_recall_at_k"
        ),
        "live_latest_visible_frame_top1_accuracy": track_metric(
            "live_current", "checkpoint_top1"
        ),
        "live_current_top1_accuracy": track_metric("live_current", "top1_relevant"),
        "live_stale_top1_rate": track_metric("live_current", "stale_top1"),
        "live_stale_fraction_at_k": track_metric("live_current", "stale_fraction_at_k"),
        "update_lag_frames_at_k": update_lag_at_k,
        "update_lag_frames_top1": update_lag_top1,
        "history_recall_at_k": track_metric("history", "recall_at_k"),
        "history_coverage_at_k": track_metric("history", "relevant_coverage_at_k"),
        "history_mrr": track_metric("history", "reciprocal_rank"),
        "absent_top1_score": statistics.fmean(absent_scores) if absent_scores else None,
        "absent_top1_confidence": (
            statistics.fmean(absent_confidences) if absent_confidences else None
        ),
        "absent_false_positive_rate_at_0_5": (
            statistics.fmean(value >= 0.5 for value in absent_confidences)
            if absent_confidences else None
        ),
        "vlm_valid_response_rate": statistics.fmean(vlm_valid) if vlm_valid else None,
        "vlm_fallback_rate": (
            1.0 - statistics.fmean(vlm_valid) if vlm_valid else None
        ),
        "vlm_faiss_recall_pool_hit_rate": _mean(
            recall_pool_records, "recall_at_k"
        ),
        "vlm_faiss_recall_pool_coverage": _mean(
            recall_pool_records, "relevant_coverage_at_k"
        ),
        "ingest_latency_ms_p50": percentile(ingestion_ms, 0.50),
        "ingest_latency_ms_p95": percentile(ingestion_ms, 0.95),
        "query_latency_ms_p50": percentile(query_latencies, 0.50),
        "query_latency_ms_p95": percentile(query_latencies, 0.95),
    }


def aggregate_episode_summaries(summaries: list[dict]) -> dict:
    metric_names = sorted({key for summary in summaries for key in summary})
    aggregate = {"episode_count": len(summaries)}
    for name in metric_names:
        values = [summary[name] for summary in summaries if summary.get(name) is not None]
        aggregate[name] = statistics.fmean(values) if values else None
        aggregate[f"{name}_observed_episodes"] = len(values)
    return aggregate
