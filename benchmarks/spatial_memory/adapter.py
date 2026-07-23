"""Small adapter contract implemented by each memory algorithm under test."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemoryCandidate:
    """One ranked result returned by a benchmark adapter."""

    observation_id: str
    score: float
    metadata: dict | None = None


class MemoryAdapter(ABC):
    """Streaming interface between the dataset and a memory implementation.

    Implementations receive only sensor observations. The evaluator owns the
    query relevance/oracle files and never passes them through this interface.
    """

    name = "memory_adapter"

    @abstractmethod
    def reset(self, episode_dir: Path, state_dir: Path) -> None:
        """Start an empty memory for one episode."""

    @abstractmethod
    def ingest(self, observation: dict) -> None:
        """Incrementally ingest exactly one sensor observation."""

    @abstractmethod
    def query(self, text: str, top_k: int) -> list[MemoryCandidate]:
        """Return ranked candidates from observations ingested so far."""

    def query_diagnostics(self) -> dict:
        """Optional metadata about the most recent query (rerank/fallback, etc.)."""
        return {}

    def run_metadata(self) -> dict:
        """Optional run-level telemetry such as stage latency and API usage."""
        return {}

    @abstractmethod
    def close(self) -> None:
        """Flush and release episode resources."""
