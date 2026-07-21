"""SigLIP/FAISS retrieval with a query-independent recency prior."""

from __future__ import annotations

import math

from memory.retrieval import retrieve_memory_candidates

from .adapter import MemoryCandidate
from .embodied_agent_adapter import EmbodiedAgentAdapter


class EmbodiedAgentRecencyAdapter(EmbodiedAgentAdapter):
    """Blend normalized FAISS similarity with normalized observation recency."""

    name = "embodied_agent_siglip_faiss_recency"

    def __init__(
        self,
        model: str = "siglip_base",
        device: str = "auto",
        ready_timeout: float = 180.0,
        recall_k: int = 12,
        recency_weight: float = 0.25,
    ):
        super().__init__(model=model, device=device, ready_timeout=ready_timeout)
        if int(recall_k) <= 0:
            raise ValueError("recall_k must be positive")
        if not math.isfinite(float(recency_weight)) or not 0.0 <= float(
            recency_weight
        ) <= 1.0:
            raise ValueError("recency_weight must be between zero and one")
        self.recall_k = int(recall_k)
        self.recency_weight = float(recency_weight)
        self._latest_frame_idx = -1
        self._last_query_diagnostics: dict = {}

    def reset(self, episode_dir, state_dir) -> None:
        super().reset(episode_dir, state_dir)
        self._latest_frame_idx = -1
        self._last_query_diagnostics = {}

    def ingest(self, observation: dict) -> None:
        super().ingest(observation)
        self._latest_frame_idx = int(observation["frame_idx"])

    def query(self, text: str, top_k: int) -> list[MemoryCandidate]:
        if self.episode_dir is None or self.memory is None or self.worker is None:
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
        recalled = [item for item in recalled if item.frame_idx is not None]
        if not recalled:
            self._last_query_diagnostics = {
                "strategy": "faiss_plus_linear_recency",
                "recall_count": 0,
                "recency_weight": self.recency_weight,
                "faiss_order": [],
                "final_order": [],
            }
            return []

        scores = [float(item.retrieval_score) for item in recalled]
        low, high = min(scores), max(scores)
        span = high - low
        faiss_rank = {
            item.memory_id: rank for rank, item in enumerate(recalled, start=1)
        }
        scored = []
        for item in recalled:
            semantic_score = (
                (float(item.retrieval_score) - low) / span if span > 1e-12 else 0.5
            )
            recency_score = (int(item.frame_idx) + 1) / max(
                self._latest_frame_idx + 1, 1
            )
            combined_score = (
                (1.0 - self.recency_weight) * semantic_score
                + self.recency_weight * recency_score
            )
            scored.append((combined_score, item, semantic_score, recency_score))
        scored.sort(
            key=lambda record: (-record[0], faiss_rank[record[1].memory_id])
        )

        self._last_query_diagnostics = {
            "strategy": "faiss_plus_linear_recency",
            "recall_count": len(recalled),
            "recall_k_requested": fetch_k,
            "recency_weight": self.recency_weight,
            "faiss_order": [item.memory_id for item in recalled],
            "final_order": [record[1].memory_id for record in scored],
        }
        return [
            MemoryCandidate(
                observation_id=f"obs_{int(item.frame_idx):06d}",
                score=float(combined_score),
                metadata={
                    "memory_id": item.memory_id,
                    "image_path": item.image_path,
                    "robot_pose": item.robot_pose,
                    "timestamp": item.timestamp,
                    "score_label": "semantic + recency",
                    "faiss_rank": faiss_rank[item.memory_id],
                    "faiss_score": float(item.retrieval_score),
                    "normalized_semantic_score": float(semantic_score),
                    "normalized_recency_score": float(recency_score),
                    "recency_weight": self.recency_weight,
                },
            )
            for combined_score, item, semantic_score, recency_score in scored[:top_k]
        ]

    def query_diagnostics(self) -> dict:
        return dict(self._last_query_diagnostics)

    def close(self) -> None:
        self._latest_frame_idx = -1
        self._last_query_diagnostics = {}
        super().close()
