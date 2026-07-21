"""EmbodiedAgent SigLIP + live FAISS adapter for the neutral benchmark API."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from agent.episodic_memory import EpisodicMemory
from memory.embedding import EmbeddingWorker
from memory.retrieval import retrieve_memory_candidates

from .adapter import MemoryAdapter, MemoryCandidate


class EmbodiedAgentAdapter(MemoryAdapter):
    name = "embodied_agent_siglip_faiss"

    def __init__(
        self,
        model: str = "siglip_base",
        device: str = "auto",
        ready_timeout: float = 180.0,
    ):
        self.model = model
        self.device = device
        self.ready_timeout = float(ready_timeout)
        self.episode_dir: Path | None = None
        self.memory: EpisodicMemory | None = None
        self.worker: EmbeddingWorker | None = None

    def reset(self, episode_dir: Path, state_dir: Path) -> None:
        self.close()
        self.episode_dir = Path(episode_dir)
        state_dir.mkdir(parents=True, exist_ok=False)
        self.memory = EpisodicMemory(str(state_dir / "memory"))
        self.worker = EmbeddingWorker(
            index_dir=str(state_dir / "index"),
            model_name=self.model,
            device=self.device,
        )
        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            if self.worker.is_ready:
                return
            if not self.worker._thread.is_alive():
                raise RuntimeError("EmbeddingWorker exited before becoming ready")
            time.sleep(0.2)
        raise TimeoutError(
            f"embedding model did not become ready within {self.ready_timeout:.0f}s"
        )

    def ingest(self, observation: dict) -> None:
        if self.episode_dir is None or self.memory is None or self.worker is None:
            raise RuntimeError("adapter must be reset before ingest")
        frame_idx = int(observation["frame_idx"])
        image_path = (self.episode_dir / observation["image_path"]).resolve()
        pose = [float(value) for value in observation["robot_pose"]]
        entry = self.memory.create_entry(
            memory_id=f"mem_{frame_idx:06d}",
            image_path=str(image_path),
            robot_pose=pose,
            timestamp=observation["timestamp"],
            embedding_model=self.model,
            source_type="benchmark_observe",
            episode_id=self.episode_dir.name,
        )
        self.memory.add_entry(entry)
        rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        self.worker.enqueue(
            rgb=rgb,
            frame_path=str(image_path),
            robot_xy=np.asarray(pose[:2], dtype=float),
            robot_yaw=float(pose[2]),
        )
        # Every benchmark checkpoint observes a fully committed live memory. This
        # intentionally measures embed/index update latency as part of ingest.
        self.worker.flush()
        if self.worker.failed:
            raise RuntimeError(
                f"embedding failed while ingesting {observation['observation_id']}"
            )

    def query(self, text: str, top_k: int) -> list[MemoryCandidate]:
        if self.episode_dir is None or self.memory is None or self.worker is None:
            raise RuntimeError("adapter must be reset before query")
        raw = retrieve_memory_candidates(
            query=text,
            index_dir=str(self.worker.index_dir),
            capture_out_dir=str(self.episode_dir),
            top_k=top_k,
            model=self.model,
            episodic_memory=self.memory,
            embedding_worker=self.worker,
        )
        return [
            MemoryCandidate(
                observation_id=f"obs_{int(candidate.frame_idx):06d}",
                score=float(candidate.retrieval_score),
                metadata={
                    "memory_id": candidate.memory_id,
                    "image_path": candidate.image_path,
                    "robot_pose": candidate.robot_pose,
                    "timestamp": candidate.timestamp,
                },
            )
            for candidate in raw
            if candidate.frame_idx is not None
        ]

    def close(self) -> None:
        if self.worker is not None:
            self.worker.stop()
        self.worker = None
        self.memory = None
        self.episode_dir = None
