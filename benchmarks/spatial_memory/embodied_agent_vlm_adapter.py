"""EmbodiedAgent FAISS recall followed by strict Gemini image reranking."""

from __future__ import annotations

import os
from pathlib import Path

from memory.retrieval import retrieve_memory_candidates

from .adapter import MemoryCandidate
from .embodied_agent_adapter import EmbodiedAgentAdapter
from .rerank_validation import validate_rerank_response


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

    def query(self, text: str, top_k: int) -> list[MemoryCandidate]:
        if (
            self.episode_dir is None
            or self.memory is None
            or self.worker is None
            or self.gemini is None
        ):
            raise RuntimeError("adapter must be reset before query")

        fetch_k = max(self.recall_k, int(top_k))
        recalled = retrieve_memory_candidates(
            query=text,
            index_dir=str(self.worker.index_dir),
            capture_out_dir=str(self.episode_dir),
            top_k=fetch_k,
            model=self.model,
            episodic_memory=self.memory,
            embedding_worker=self.worker,
        )
        if not recalled:
            self._last_query_diagnostics = {
                "rerank_attempted": False,
                "rerank_valid": False,
                "fallback_reason": "empty_faiss_recall",
                "recall_k_requested": fetch_k,
                "recall_count": 0,
                "vlm_model": self.vlm_model,
            }
            return []

        candidate_ids = [candidate.memory_id for candidate in recalled]
        faiss_rank = {
            candidate.memory_id: rank
            for rank, candidate in enumerate(recalled, start=1)
        }
        raw_response = self.gemini.rerank_memory_candidates(
            query=text,
            candidates=[self._candidate_metadata(candidate) for candidate in recalled],
            image_paths=[candidate.image_path for candidate in recalled],
        )
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

    def close(self) -> None:
        self.gemini = None
        self._last_query_diagnostics = {}
        super().close()
