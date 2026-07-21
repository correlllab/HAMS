"""Query-agnostic newest-frame baseline for benchmark sanity checks."""

from __future__ import annotations

from pathlib import Path

from .adapter import MemoryAdapter, MemoryCandidate


class LatestOnlyAdapter(MemoryAdapter):
    """Return the most recently ingested observations, newest first."""

    name = "latest_only"

    def __init__(self):
        self.observations: list[dict] = []
        self._last_query_diagnostics: dict = {}

    def reset(self, episode_dir: Path, state_dir: Path) -> None:
        self.close()
        state_dir.mkdir(parents=True, exist_ok=False)

    def ingest(self, observation: dict) -> None:
        self.observations.append(dict(observation))

    def query(self, text: str, top_k: int) -> list[MemoryCandidate]:
        selected = list(reversed(self.observations[-int(top_k):]))
        count = len(self.observations)
        self._last_query_diagnostics = {
            "query_agnostic": True,
            "strategy": "newest_frames_first",
            "ingested_count": count,
        }
        return [
            MemoryCandidate(
                observation_id=observation["observation_id"],
                score=(int(observation["frame_idx"]) + 1) / max(count, 1),
                metadata={
                    "score_label": "recency rank",
                    "timestamp": observation.get("timestamp"),
                    "robot_pose": observation.get("robot_pose"),
                },
            )
            for observation in selected
        ]

    def query_diagnostics(self) -> dict:
        return dict(self._last_query_diagnostics)

    def close(self) -> None:
        self.observations = []
        self._last_query_diagnostics = {}
