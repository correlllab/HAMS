#!/usr/bin/env python3
"""Stream a neutral benchmark dataset through a selected memory adapter."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from .adapter import MemoryAdapter
from .metrics import (
    aggregate_episode_summaries,
    score_location_query,
    score_query,
    summarize_episode,
)
from .report import write_reports


BUILTIN_ADAPTERS = {
    "latest_only": (
        "benchmarks.spatial_memory.latest_only_adapter:LatestOnlyAdapter"
    ),
    "embodied_agent": (
        "benchmarks.spatial_memory.embodied_agent_adapter:EmbodiedAgentAdapter"
    ),
    "embodied_agent_recency": (
        "benchmarks.spatial_memory.embodied_agent_recency_adapter:EmbodiedAgentRecencyAdapter"
    ),
    "embodied_agent_vlm": (
        "benchmarks.spatial_memory.embodied_agent_vlm_adapter:EmbodiedAgentVLMAdapter"
    ),
    "vlmaps": "benchmarks.spatial_memory.vlmaps_adapter:VLMapsAdapter",
}


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _faiss_recall_pool_metrics(query: dict, diagnostics: dict) -> dict | None:
    """Score an adapter-reported FAISS pool without exposing oracle data to it."""
    raw_ids = diagnostics.get("faiss_order")
    if not isinstance(raw_ids, list) or not raw_ids:
        return None
    observation_ids = []
    for memory_id in raw_ids:
        if not isinstance(memory_id, str) or not memory_id.startswith("mem_"):
            return None
        try:
            observation_ids.append(f"obs_{int(memory_id[4:]):06d}")
        except ValueError:
            return None
    return score_query(query, observation_ids, len(observation_ids))


def _load_adapter(spec: str, adapter_kwargs: dict) -> MemoryAdapter:
    target = BUILTIN_ADAPTERS.get(spec, spec)
    if ":" not in target:
        raise ValueError(
            "adapter must be a built-in name or import path like package.module:Class"
        )
    module_name, class_name = target.rsplit(":", maxsplit=1)
    adapter_class = getattr(importlib.import_module(module_name), class_name)
    adapter = adapter_class(**adapter_kwargs)
    if not isinstance(adapter, MemoryAdapter):
        raise TypeError(f"{target} does not implement MemoryAdapter")
    return adapter


def _validate_episode_contract(episode_dir: Path) -> tuple[list[dict], list[dict]]:
    observations = _read_jsonl(episode_dir / "observations.jsonl")
    queries = _read_jsonl(episode_dir / "queries.jsonl")
    if not observations:
        raise RuntimeError(f"episode has no observations: {episode_dir}")
    expected = list(range(len(observations)))
    actual = [int(item["frame_idx"]) for item in observations]
    if actual != expected:
        raise RuntimeError(f"episode frames must be contiguous from zero: {episode_dir}")
    observation_ids = {item["observation_id"] for item in observations}
    if len(observation_ids) != len(observations):
        raise RuntimeError(f"duplicate observation id: {episode_dir}")
    for observation in observations:
        for key in ("image_path", "pose_path"):
            path = (episode_dir / observation[key]).resolve()
            if episode_dir.resolve() not in path.parents or not path.is_file():
                raise RuntimeError(f"invalid observation {key}: {path}")
        if observation.get("depth_path") is not None:
            depth_path = (episode_dir / observation["depth_path"]).resolve()
            if episode_dir.resolve() not in depth_path.parents or not depth_path.is_file():
                raise RuntimeError(f"invalid observation depth_path: {depth_path}")
    for query in queries:
        if int(query["checkpoint_frame"]) not in expected:
            raise RuntimeError(f"invalid query checkpoint: {query}")
        unknown = (
            set(query.get("relevant_observation_ids", []))
            | set(query.get("stale_observation_ids", []))
        ) - observation_ids
        if unknown:
            raise RuntimeError(f"query references unknown observations: {sorted(unknown)}")
    return observations, queries


def evaluate_episode(
    adapter: MemoryAdapter,
    episode_dir: Path,
    state_dir: Path,
    top_k: int,
) -> dict:
    observations, queries = _validate_episode_contract(episode_dir)
    queries_by_frame: dict[int, list[dict]] = {}
    for query in queries:
        queries_by_frame.setdefault(int(query["checkpoint_frame"]), []).append(query)

    ingestion_ms: list[float] = []
    query_results: list[dict] = []
    ingested_ids: set[str] = set()
    adapter.reset(episode_dir, state_dir)
    try:
        for observation in observations:
            started = time.perf_counter()
            adapter.ingest(observation)
            ingestion_ms.append((time.perf_counter() - started) * 1000.0)
            ingested_ids.add(observation["observation_id"])
            for query in queries_by_frame.get(int(observation["frame_idx"]), []):
                started = time.perf_counter()
                candidates = adapter.query(query["text"], top_k)
                query_latency_ms = (time.perf_counter() - started) * 1000.0
                candidate_ids = [candidate.observation_id for candidate in candidates]
                if len(candidate_ids) != len(set(candidate_ids)):
                    raise RuntimeError(
                        f"adapter returned duplicate candidates for {query['query_id']}"
                    )
                unavailable = set(candidate_ids) - ingested_ids
                if unavailable:
                    raise RuntimeError(
                        "adapter returned observations that were not ingested at this "
                        f"checkpoint: {sorted(unavailable)}"
                    )
                candidate_records = [
                    {
                        "rank": rank,
                        "observation_id": candidate.observation_id,
                        "score": candidate.score,
                        "metadata": candidate.metadata,
                    }
                    for rank, candidate in enumerate(candidates, start=1)
                ]
                diagnostics = adapter.query_diagnostics()
                result = {
                    **query,
                    "query_latency_ms": query_latency_ms,
                    "adapter_diagnostics": diagnostics,
                    "faiss_recall_pool_metrics": _faiss_recall_pool_metrics(
                        query, diagnostics
                    ),
                    "candidates": candidate_records,
                    "metrics": {
                        **score_query(query, candidate_ids, top_k),
                        **score_location_query(query, candidate_records),
                    },
                }
                query_results.append(result)
                print(
                    f"[memory-eval] {episode_dir.name} {query['query_id']} "
                    f"top1={candidate_records[0]['observation_id'] if candidate_records else 'none'} "
                    f"recall@{top_k}={result['metrics']['recall_at_k']} "
                    f"stale_top1={result['metrics']['stale_top1']}"
                )
    finally:
        adapter.close()

    return {
        "episode_id": episode_dir.name,
        "frame_count": len(observations),
        "query_count": len(query_results),
        "summary": summarize_episode(query_results, ingestion_ms),
        "ingest_latency_ms": ingestion_ms,
        "queries": query_results,
    }


def evaluate(args) -> Path:
    dataset_dir = Path(args.dataset).resolve()
    with open(dataset_dir / "benchmark_manifest.json", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("benchmark_id") not in {
        "robocasa_object_relocation_v1",
        "robocasa_object_relocation_v2",
    }:
        raise RuntimeError("unsupported benchmark dataset")

    episode_records = manifest["episodes"]
    if args.max_episodes is not None:
        episode_records = episode_records[:args.max_episodes]
    if not episode_records:
        raise RuntimeError("no episodes selected")

    adapter_kwargs = json.loads(args.adapter_kwargs)
    if not isinstance(adapter_kwargs, dict):
        raise ValueError("--adapter-kwargs must decode to a JSON object")
    adapter = _load_adapter(args.adapter, adapter_kwargs)

    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = dataset_dir / "reports" / adapter.name / timestamp
    if output_dir.exists():
        raise RuntimeError(f"report output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    state_root = output_dir / "state"
    state_root.mkdir()

    episode_results = []
    try:
        for episode_record in episode_records:
            episode_id = episode_record["episode_id"]
            episode_results.append(evaluate_episode(
                adapter=adapter,
                episode_dir=dataset_dir / "episodes" / episode_id,
                state_dir=state_root / episode_id,
                top_k=args.top_k,
            ))
        report = {
            "schema_version": 1,
            "benchmark_id": manifest["benchmark_id"],
            "dataset": str(dataset_dir),
            "adapter": adapter.name,
            "adapter_spec": args.adapter,
            "adapter_kwargs": adapter_kwargs,
            "top_k": args.top_k,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "aggregate": aggregate_episode_summaries(
                [item["summary"] for item in episode_results]
            ),
            "episodes": episode_results,
        }
        with open(output_dir / "results.json", "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
            file.write("\n")
        html_report, markdown_report = write_reports(
            report=report,
            manifest=manifest,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
        )
        if not args.keep_state:
            shutil.rmtree(state_root)
        print(json.dumps(report["aggregate"], indent=2))
        print(f"[memory-eval] results: {output_dir / 'results.json'}")
        print(f"[memory-eval] readable HTML: {html_report}")
        print(f"[memory-eval] Markdown summary: {markdown_report}")
        return output_dir
    except Exception:
        adapter.close()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a spatial-memory benchmark")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--adapter",
        default="embodied_agent",
        help="built-in name or import path package.module:Class",
    )
    parser.add_argument("--adapter-kwargs", default="{}")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--keep-state", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.top_k <= 0:
        raise SystemExit("top-k must be positive")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise SystemExit("max-episodes must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
