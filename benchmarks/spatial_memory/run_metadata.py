"""Reproducibility, resource, and API metadata for benchmark runs."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .metrics import percentile


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dataset_storage(dataset_dir: Path) -> dict:
    episodes = dataset_dir / "episodes"
    files = [item for item in episodes.rglob("*") if item.is_file()]
    rgb = sum(item.stat().st_size for item in files if item.suffix.lower() in {".png", ".jpg", ".jpeg"})
    depth = sum(item.stat().st_size for item in files if item.suffix.lower() == ".npy")
    manifest = dataset_dir / "benchmark_manifest.json"
    return {
        "payload_bytes": sum(item.stat().st_size for item in files)
        + (manifest.stat().st_size if manifest.is_file() else 0),
        "rgb_bytes": rgb,
        "depth_bytes": depth,
    }


def _state_storage(state_root: Path) -> dict:
    categories = {"index_bytes": 0, "memory_metadata_bytes": 0, "vlm_log_bytes": 0}
    total = 0
    if state_root.exists():
        for item in state_root.rglob("*"):
            if not item.is_file():
                continue
            size = item.stat().st_size
            total += size
            parts = set(item.relative_to(state_root).parts)
            if "index" in parts:
                categories["index_bytes"] += size
            elif "memory" in parts:
                categories["memory_metadata_bytes"] += size
            elif "vlm_logs" in parts:
                categories["vlm_log_bytes"] += size
    return {"adapter_state_bytes": total, **categories}


def _git_state(path: Path, environment_prefix: str) -> dict | None:
    environment_commit = os.environ.get(f"{environment_prefix}_GIT_COMMIT")
    if environment_commit:
        dirty_value = os.environ.get(f"{environment_prefix}_GIT_DIRTY", "")
        return {
            "commit": environment_commit,
            "dirty": dirty_value == "1",
            "dirty_entry_count": None,
        }
    if not path.exists():
        return None
    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()
        return {
            "commit": commit,
            "dirty": bool(status),
            "dirty_entry_count": len(status),
        }
    except (OSError, subprocess.SubprocessError):
        return None


def _package_versions() -> dict:
    versions = {}
    for package in ("torch", "transformers", "faiss-cpu", "google-genai", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _torch_module():
    try:
        import torch

        return torch
    except ImportError:
        return None


def _reset_gpu_peak() -> dict:
    torch = _torch_module()
    if torch is None:
        return {"attempted": False, "successful": False, "error": None}
    try:
        if not torch.cuda.is_available():
            return {"attempted": False, "successful": False, "error": None}
        # On recent GPUs, device_count/current_device can succeed before the CUDA
        # memory allocator exists. Explicit initialization avoids an otherwise
        # surprising ``Invalid device argument`` from reset_peak_memory_stats.
        torch.cuda.init()
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
        return {"attempted": True, "successful": True, "error": None}
    except Exception as error:
        # Telemetry must never prevent the benchmark itself from running.
        return {
            "attempted": True,
            "successful": False,
            "error": f"{type(error).__name__}: {error}"[:500],
        }


def _gpu_metadata() -> dict:
    torch = _torch_module()
    if torch is None:
        return {"available": False, "device_count": 0, "devices": []}
    try:
        available = bool(torch.cuda.is_available())
    except Exception as error:
        return {
            "available": False,
            "device_count": 0,
            "devices": [],
            "error": f"{type(error).__name__}: {error}"[:500],
        }
    devices = []
    if available:
        try:
            device_count = int(torch.cuda.device_count())
        except Exception as error:
            return {
                "available": True,
                "device_count": 0,
                "devices": [],
                "error": f"{type(error).__name__}: {error}"[:500],
            }
        for index in range(device_count):
            try:
                properties = torch.cuda.get_device_properties(index)
                devices.append({
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
                })
            except Exception as error:
                devices.append({
                    "index": index,
                    "name": "unavailable",
                    "error": f"{type(error).__name__}: {error}"[:500],
                })
    return {
        "available": available,
        "device_count": len(devices),
        "cuda_version": getattr(torch.version, "cuda", None),
        "devices": devices,
    }


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _latency_summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "total_ms": sum(values),
        "mean_ms": statistics.fmean(values) if values else None,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
    }


def _discover_repo(hams_root: Path, name: str, container_path: str) -> Path | None:
    candidates = [Path(container_path), hams_root.parent / name]
    return next((path for path in candidates if path.exists()), None)


class RunMetadataCollector:
    """Collect metadata without exposing environment variables or credentials."""

    def __init__(
        self,
        dataset_dir: Path,
        episode_records: list[dict],
        hams_root: Path,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.episode_records = episode_records
        self.hams_root = Path(hams_root)
        self.started_at = _utc_now()
        self.started_perf = time.perf_counter()
        self.dataset_storage = _dataset_storage(self.dataset_dir)
        self.gpu_peak_reset = _reset_gpu_peak()

    def finish(
        self,
        status: str,
        episode_results: list[dict],
        state_root: Path,
        adapter_metadata: dict,
        failure: BaseException | None = None,
    ) -> dict:
        ingest_latencies = [
            float(value)
            for episode in episode_results
            for value in episode.get("ingest_latency_ms", [])
        ]
        query_latencies = [
            float(query["query_latency_ms"])
            for episode in episode_results
            for query in episode.get("queries", [])
        ]
        reset_latencies = [
            float(episode["adapter_reset_latency_ms"])
            for episode in episode_results
            if episode.get("adapter_reset_latency_ms") is not None
        ]
        embodied_root = _discover_repo(
            self.hams_root, "EmbodiedAgent", "/opt/EmbodiedAgent"
        )
        vlmaps_root = _discover_repo(self.hams_root, "vlmaps", "/opt/vlmaps")
        gpu_metadata = _gpu_metadata()
        gpu_metadata["peak_stats_reset"] = self.gpu_peak_reset
        metadata = {
            "schema_version": 1,
            "status": status,
            "started_at": self.started_at,
            "completed_at": _utc_now(),
            "wall_time_seconds": time.perf_counter() - self.started_perf,
            "workload": {
                "scheduled_episode_count": len(self.episode_records),
                "scheduled_frame_count": sum(
                    int(item.get("frame_count", 0)) for item in self.episode_records
                ),
                "scheduled_query_count": sum(
                    int(item.get("query_count", 0)) for item in self.episode_records
                ),
                "completed_episode_count": len(episode_results),
                "completed_frame_count": sum(
                    int(item.get("frame_count", 0)) for item in episode_results
                ),
                "completed_query_count": sum(
                    int(item.get("query_count", 0)) for item in episode_results
                ),
            },
            "latency": {
                "adapter_reset": _latency_summary(reset_latencies),
                "ingest": _latency_summary(ingest_latencies),
                "query": _latency_summary(query_latencies),
            },
            "resources": {
                "process_peak_rss_bytes": _peak_rss_bytes(),
                "gpu": gpu_metadata,
                "storage": {
                    **self.dataset_storage,
                    **_state_storage(Path(state_root)),
                },
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "containerized": Path("/.dockerenv").exists(),
                "container_image": os.environ.get("SPATIAL_RUNTIME_IMAGE"),
                "argv": list(sys.argv),
                "packages": _package_versions(),
            },
            "software": {
                "hams": _git_state(self.hams_root, "SPATIAL_HAMS"),
                "embodied_agent": (
                    _git_state(embodied_root, "SPATIAL_EMBODIED_AGENT")
                    if embodied_root else None
                ),
                "vlmaps": (
                    _git_state(vlmaps_root, "SPATIAL_VLMAPS")
                    if vlmaps_root else None
                ),
            },
            "adapter": adapter_metadata,
        }
        if failure is not None:
            metadata["failure"] = {
                "type": type(failure).__name__,
                "message": str(failure)[:1000],
            }
        return metadata
