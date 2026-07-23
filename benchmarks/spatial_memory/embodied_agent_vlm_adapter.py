"""EmbodiedAgent FAISS recall followed by strict Gemini image reranking."""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

from memory.retrieval import retrieve_memory_candidates

from .adapter import MemoryCandidate
from .embodied_agent_adapter import EmbodiedAgentAdapter
from .metrics import percentile
from .rerank_validation import validate_rerank_response


GEMINI_STANDARD_PRICING = {
    "gemini-3.5-flash": {
        "input_usd_per_million_tokens": 1.50,
        "output_usd_per_million_tokens": 9.00,
        "cached_input_usd_per_million_tokens": 0.15,
        "source": "https://ai.google.dev/gemini-api/docs/pricing",
        "verified_on": "2026-07-22",
        "note": "Standard paid-tier list price; actual free-tier charge may be zero.",
    },
}

USAGE_COUNTERS = (
    "logical_calls",
    "api_attempts",
    "successful_calls",
    "failed_calls",
    "successful_api_responses",
    "responses_with_usage_metadata",
    "api_errors",
    "parse_failures",
    "prompt_tokens",
    "candidate_tokens",
    "thought_tokens",
    "cached_content_tokens",
    "tool_use_prompt_tokens",
    "total_tokens",
)


def _empty_usage() -> dict:
    return {**{key: 0 for key in USAGE_COUNTERS}, "error_types": {}, "last_error": None}


def _merge_usage(total: dict, addition: dict) -> None:
    for key in USAGE_COUNTERS:
        total[key] = int(total.get(key, 0)) + int(addition.get(key, 0))
    for name, count in (addition.get("error_types") or {}).items():
        errors = total.setdefault("error_types", {})
        errors[name] = int(errors.get(name, 0)) + int(count)
    if addition.get("last_error"):
        total["last_error"] = str(addition["last_error"])


def _usage_delta(before: dict, after: dict) -> dict:
    delta = {
        key: max(0, int(after.get(key, 0)) - int(before.get(key, 0)))
        for key in USAGE_COUNTERS
    }
    error_types = {}
    for name, count in (after.get("error_types") or {}).items():
        difference = int(count) - int((before.get("error_types") or {}).get(name, 0))
        if difference > 0:
            error_types[name] = difference
    delta["error_types"] = error_types
    delta["last_error"] = after.get("last_error") if delta["api_errors"] else None
    delta["retry_count"] = max(0, delta["api_attempts"] - delta["logical_calls"])
    return delta


def _latency_summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "total": sum(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


class EmbodiedAgentVLMAdapter(EmbodiedAgentAdapter):
    """Streaming SigLIP/FAISS memory with Gemini reranking at each query."""

    name = "embodied_agent_siglip_faiss_gemini"

    def __init__(
        self,
        model: str = "siglip_base",
        device: str = "auto",
        ready_timeout: float = 180.0,
        recall_k: int = 12,
        vlm_model: str = "gemini-3.5-flash",
        max_retries: int = 3,
    ):
        super().__init__(model=model, device=device, ready_timeout=ready_timeout)
        if int(recall_k) <= 0:
            raise ValueError("recall_k must be positive")
        self.recall_k = int(recall_k)
        self.vlm_model = vlm_model
        self.max_retries = int(max_retries)
        self.gemini = None
        self._last_query_diagnostics: dict = {}
        self._completed_vlm_usage = _empty_usage()
        self._faiss_recall_latency_ms: list[float] = []
        self._vlm_rerank_latency_ms: list[float] = []
        self._vlm_telemetry_available = False

    def reset(self, episode_dir: Path, state_dir: Path) -> None:
        super().reset(episode_dir, state_dir)
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            self.close()
            raise RuntimeError(
                "EmbodiedAgentVLMAdapter requires GEMINI_API_KEY or GOOGLE_API_KEY"
            )
        try:
            from agent.gemini_client import GeminiClient

            self.gemini = GeminiClient(
                api_key=api_key,
                model_name=self.vlm_model,
                log_dir=str(state_dir / "vlm_logs"),
                max_retries=self.max_retries,
            )
            self._vlm_telemetry_available = hasattr(self.gemini, "usage_summary")
        except Exception:
            self.close()
            raise
        self._last_query_diagnostics = {}

    @staticmethod
    def _candidate_metadata(candidate) -> dict:
        return {
            "memory_id": candidate.memory_id,
            "retrieval_score": float(candidate.retrieval_score),
            "frame_idx": candidate.frame_idx,
            "timestamp": candidate.timestamp,
            "robot_pose": candidate.robot_pose,
        }

    def _ensure_telemetry(self) -> None:
        if not hasattr(self, "_completed_vlm_usage"):
            self._completed_vlm_usage = _empty_usage()
        if not hasattr(self, "_faiss_recall_latency_ms"):
            self._faiss_recall_latency_ms = []
        if not hasattr(self, "_vlm_rerank_latency_ms"):
            self._vlm_rerank_latency_ms = []
        if not hasattr(self, "_vlm_telemetry_available"):
            self._vlm_telemetry_available = hasattr(
                getattr(self, "gemini", None), "usage_summary"
            )

    def query(self, text: str, top_k: int) -> list[MemoryCandidate]:
        self._ensure_telemetry()
        if (
            self.episode_dir is None
            or self.memory is None
            or self.worker is None
            or self.gemini is None
        ):
            raise RuntimeError("adapter must be reset before query")

        fetch_k = max(self.recall_k, int(top_k))
        recall_started = time.perf_counter()
        recalled = retrieve_memory_candidates(
            query=text,
            index_dir=str(self.worker.index_dir),
            capture_out_dir=str(self.episode_dir),
            top_k=fetch_k,
            model=self.model,
            episodic_memory=self.memory,
            embedding_worker=self.worker,
        )
        faiss_recall_latency_ms = (time.perf_counter() - recall_started) * 1000.0
        self._faiss_recall_latency_ms.append(faiss_recall_latency_ms)
        if not recalled:
            self._last_query_diagnostics = {
                "rerank_attempted": False,
                "rerank_valid": False,
                "fallback_reason": "empty_faiss_recall",
                "recall_k_requested": fetch_k,
                "recall_count": 0,
                "vlm_model": self.vlm_model,
                "stage_latency_ms": {
                    "faiss_recall": faiss_recall_latency_ms,
                    "vlm_rerank": None,
                },
            }
            return []

        candidate_ids = [candidate.memory_id for candidate in recalled]
        faiss_rank = {
            candidate.memory_id: rank
            for rank, candidate in enumerate(recalled, start=1)
        }
        usage_before = self._current_vlm_usage()
        rerank_started = time.perf_counter()
        raw_response = self.gemini.rerank_memory_candidates(
            query=text,
            candidates=[self._candidate_metadata(candidate) for candidate in recalled],
            image_paths=[candidate.image_path for candidate in recalled],
        )
        vlm_rerank_latency_ms = (time.perf_counter() - rerank_started) * 1000.0
        self._vlm_rerank_latency_ms.append(vlm_rerank_latency_ms)
        usage_after = self._current_vlm_usage()
        validated = validate_rerank_response(raw_response, candidate_ids)
        analysis_by_id: dict[str, dict] = {}
        rerank_valid = validated is not None
        if validated is not None:
            ranked_ids, analysis_by_id = validated
            candidate_by_id = {candidate.memory_id: candidate for candidate in recalled}
            ranked = [candidate_by_id[memory_id] for memory_id in ranked_ids]
        else:
            ranked_ids = candidate_ids
            ranked = recalled

        self._last_query_diagnostics = {
            "rerank_attempted": True,
            "rerank_valid": rerank_valid,
            "fallback_reason": None if rerank_valid else "invalid_vlm_response",
            "recall_k_requested": fetch_k,
            "recall_count": len(recalled),
            "vlm_model": self.vlm_model,
            "faiss_order": candidate_ids,
            "final_rerank_order": ranked_ids,
            "raw_response": raw_response,
            "stage_latency_ms": {
                "faiss_recall": faiss_recall_latency_ms,
                "vlm_rerank": vlm_rerank_latency_ms,
            },
            "vlm_usage_delta": _usage_delta(usage_before, usage_after),
        }

        result = []
        for vlm_rank, candidate in enumerate(ranked[:top_k], start=1):
            analysis = analysis_by_id.get(candidate.memory_id, {})
            score = (
                float(analysis["confidence"])
                if analysis else float(candidate.retrieval_score)
            )
            result.append(MemoryCandidate(
                observation_id=f"obs_{int(candidate.frame_idx):06d}",
                score=score,
                metadata={
                    "memory_id": candidate.memory_id,
                    "image_path": candidate.image_path,
                    "robot_pose": candidate.robot_pose,
                    "timestamp": candidate.timestamp,
                    "score_label": (
                        "VLM confidence" if analysis else "FAISS cosine (fallback)"
                    ),
                    "faiss_rank": faiss_rank[candidate.memory_id],
                    "faiss_score": float(candidate.retrieval_score),
                    "vlm_rank": vlm_rank if rerank_valid else None,
                    "rerank_valid": rerank_valid,
                    **(
                        {"confidence_0_1": float(analysis["confidence"])}
                        if analysis else {}
                    ),
                    "rerank_object_location": analysis.get("object_location"),
                    "rerank_reasoning": analysis.get("reasoning"),
                },
            ))
        return result

    def query_diagnostics(self) -> dict:
        return dict(self._last_query_diagnostics)

    def _current_vlm_usage(self) -> dict:
        if self.gemini is None or not hasattr(self.gemini, "usage_summary"):
            return _empty_usage()
        return self.gemini.usage_summary()

    def run_metadata(self) -> dict:
        self._ensure_telemetry()
        usage = dict(self._completed_vlm_usage)
        usage["error_types"] = dict(self._completed_vlm_usage["error_types"])
        if self.gemini is not None:
            _merge_usage(usage, self._current_vlm_usage())
        usage["retry_count"] = max(
            0, int(usage["api_attempts"]) - int(usage["logical_calls"])
        )
        output_tokens = int(usage["candidate_tokens"]) + int(usage["thought_tokens"])
        usage["output_tokens_including_thoughts"] = output_tokens
        pricing = GEMINI_STANDARD_PRICING.get(self.vlm_model)
        estimated_cost = None
        if pricing is not None and int(usage["responses_with_usage_metadata"]) > 0:
            cached = int(usage["cached_content_tokens"])
            uncached = max(0, int(usage["prompt_tokens"]) - cached)
            estimated_cost = (
                uncached * pricing["input_usd_per_million_tokens"]
                + cached * pricing["cached_input_usd_per_million_tokens"]
                + output_tokens * pricing["output_usd_per_million_tokens"]
            ) / 1_000_000.0
        usage["estimated_standard_cost_usd"] = estimated_cost
        usage["pricing_assumption"] = pricing
        usage["telemetry_available"] = self._vlm_telemetry_available
        return {
            "stage_latency_ms": {
                "faiss_recall": _latency_summary(self._faiss_recall_latency_ms),
                "vlm_rerank": _latency_summary(self._vlm_rerank_latency_ms),
            },
            "vlm": usage,
        }

    def close(self) -> None:
        self._ensure_telemetry()
        if getattr(self, "gemini", None) is not None:
            _merge_usage(self._completed_vlm_usage, self._current_vlm_usage())
        self.gemini = None
        self._last_query_diagnostics = {}
        super().close()
