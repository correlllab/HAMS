#!/usr/bin/env python3
"""Stream a neutral benchmark dataset through a selected memory adapter."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
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
from .run_metadata import RunMetadataCollector


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

CHECKPOINT_SCHEMA_VERSION = 1
VLM_USAGE_COUNTERS = (
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


def _adapter_run_metadata(adapter: MemoryAdapter) -> dict:
    try:
        metadata = adapter.run_metadata()
        return metadata if isinstance(metadata, dict) else {}
    except Exception as error:
        return {
            "telemetry_error": {
                "type": type(error).__name__,
                "message": str(error)[:1000],
            }
        }


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    """Atomically replace one JSON file so an interrupted write is never resumed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_config(
    *,
    manifest_path: Path,
    manifest: dict,
    adapter: MemoryAdapter,
    adapter_spec: str,
    adapter_kwargs: dict,
    top_k: int,
    episode_records: list[dict],
) -> dict:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "benchmark_id": manifest["benchmark_id"],
        "dataset_manifest_sha256": _file_sha256(manifest_path),
        "adapter": adapter.name,
        "adapter_spec": adapter_spec,
        "adapter_kwargs": adapter_kwargs,
        "top_k": int(top_k),
        "episode_ids": [item["episode_id"] for item in episode_records],
    }


def _load_episode_checkpoints(
    checkpoint_root: Path,
    episode_records: list[dict],
) -> list[dict]:
    """Load only a contiguous completed prefix of the selected episode list."""
    episode_root = checkpoint_root / "episodes"
    results = []
    found_gap = False
    for record in episode_records:
        episode_id = record["episode_id"]
        path = episode_root / f"{episode_id}.json"
        if not path.is_file():
            found_gap = True
            continue
        if found_gap:
            raise RuntimeError(
                "resume checkpoints are not a contiguous episode prefix; "
                f"unexpected checkpoint: {path}"
            )
        result = _read_json(path)
        if result.get("episode_id") != episode_id:
            raise RuntimeError(f"checkpoint episode mismatch: {path}")
        if int(result.get("frame_count", -1)) != int(record.get("frame_count", -2)):
            raise RuntimeError(f"checkpoint frame count mismatch: {path}")
        if int(result.get("query_count", -1)) != int(record.get("query_count", -2)):
            raise RuntimeError(f"checkpoint query count mismatch: {path}")
        results.append(result)
    return results


def _next_attempt_path(checkpoint_root: Path) -> Path:
    attempt_root = checkpoint_root / "attempts"
    attempt_root.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in attempt_root.glob("attempt_*.json"):
        try:
            indices.append(int(path.stem.removeprefix("attempt_")))
        except ValueError:
            continue
    return attempt_root / f"attempt_{max(indices, default=0) + 1:03d}.json"


def _load_attempt_metadata(checkpoint_root: Path) -> list[dict]:
    attempt_root = checkpoint_root / "attempts"
    return [
        _read_json(path)
        for path in sorted(attempt_root.glob("attempt_*.json"))
    ] if attempt_root.is_dir() else []


def _stage_latency_from_results(episode_results: list[dict]) -> dict:
    values = {"faiss_recall": [], "vlm_rerank": []}
    for episode in episode_results:
        for query in episode.get("queries", []):
            diagnostics = query.get("adapter_diagnostics")
            stages = (
                diagnostics.get("stage_latency_ms")
                if isinstance(diagnostics, dict) else None
            )
            if not isinstance(stages, dict):
                continue
            for name in values:
                value = stages.get(name)
                if isinstance(value, (int, float)):
                    values[name].append(float(value))

    def summarize(samples: list[float]) -> dict:
        ordered = sorted(samples)
        if not ordered:
            return {
                "count": 0,
                "total": 0,
                "mean": None,
                "p50": None,
                "p95": None,
            }

        def percentile(quantile: float) -> float:
            position = (len(ordered) - 1) * quantile
            lower = int(position)
            upper = min(lower + 1, len(ordered) - 1)
            fraction = position - lower
            return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

        return {
            "count": len(ordered),
            "total": sum(ordered),
            "mean": sum(ordered) / len(ordered),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
        }

    return {name: summarize(samples) for name, samples in values.items()}


def _merge_adapter_attempts(
    attempts: list[dict],
    episode_results: list[dict],
) -> dict:
    adapters = [
        item.get("adapter")
        for item in attempts
        if isinstance(item.get("adapter"), dict)
    ]
    if not adapters:
        return {}
    merged = copy.deepcopy(adapters[-1])
    vlm_attempts = [
        item["vlm"] for item in adapters if isinstance(item.get("vlm"), dict)
    ]
    if not vlm_attempts:
        return merged

    current = copy.deepcopy(vlm_attempts[-1])
    for key in VLM_USAGE_COUNTERS:
        current[key] = sum(int(item.get(key, 0)) for item in vlm_attempts)
    current["retry_count"] = max(
        0, int(current["api_attempts"]) - int(current["logical_calls"])
    )
    current["output_tokens_including_thoughts"] = (
        int(current["candidate_tokens"]) + int(current["thought_tokens"])
    )
    error_types = {}
    for item in vlm_attempts:
        for name, count in (item.get("error_types") or {}).items():
            error_types[name] = int(error_types.get(name, 0)) + int(count)
    current["error_types"] = error_types
    current["last_error"] = next(
        (item.get("last_error") for item in reversed(vlm_attempts) if item.get("last_error")),
        None,
    )
    current["telemetry_available"] = all(
        bool(item.get("telemetry_available", True)) for item in vlm_attempts
    )
    pricing = current.get("pricing_assumption")
    if (
        isinstance(pricing, dict)
        and int(current["responses_with_usage_metadata"]) > 0
    ):
        cached = int(current["cached_content_tokens"])
        uncached = max(0, int(current["prompt_tokens"]) - cached)
        current["estimated_standard_cost_usd"] = (
            uncached * float(pricing["input_usd_per_million_tokens"])
            + cached * float(pricing["cached_input_usd_per_million_tokens"])
            + int(current["output_tokens_including_thoughts"])
            * float(pricing["output_usd_per_million_tokens"])
        ) / 1_000_000.0
    else:
        current["estimated_standard_cost_usd"] = None
    current["attempt_count"] = len(vlm_attempts)
    merged["vlm"] = current
    merged["stage_latency_ms"] = _stage_latency_from_results(episode_results)
    return merged


def _merge_attempt_metadata(
    attempts: list[dict],
    episode_results: list[dict],
    loaded_episode_count: int,
) -> dict:
    """Combine attempt-level API usage while retaining final workload/resource data."""
    if not attempts:
        raise ValueError("at least one attempt is required")
    merged = copy.deepcopy(attempts[-1])
    merged["started_at"] = attempts[0]["started_at"]
    merged["wall_time_seconds"] = sum(
        float(item.get("wall_time_seconds", 0.0)) for item in attempts
    )
    merged["adapter"] = _merge_adapter_attempts(attempts, episode_results)
    if str(merged.get("status", "")).startswith("completed"):
        merged["status"] = _completed_status(merged["adapter"])
    merged["resumption"] = {
        "resumed": len(attempts) > 1,
        "attempt_count": len(attempts),
        "loaded_episode_count_this_attempt": int(loaded_episode_count),
        "checkpointed_episode_count": len(episode_results),
        "attempts": [
            {
                "attempt": index,
                "status": item.get("status"),
                "started_at": item.get("started_at"),
                "completed_at": item.get("completed_at"),
                "wall_time_seconds": item.get("wall_time_seconds"),
                "failure": item.get("failure"),
            }
            for index, item in enumerate(attempts, start=1)
        ],
    }
    return merged


def _completed_status(adapter_metadata: dict) -> str:
    vlm = adapter_metadata.get("vlm")
    if isinstance(vlm, dict) and (
        int(vlm.get("failed_calls", 0)) > 0
        or int(vlm.get("api_errors", 0)) > 0
        or not bool(vlm.get("telemetry_available", True))
    ):
        return "completed_with_errors"
    return "completed"


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
    reset_started = time.perf_counter()
    adapter.reset(episode_dir, state_dir)
    reset_latency_ms = (time.perf_counter() - reset_started) * 1000.0
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
        "adapter_reset_latency_ms": reset_latency_ms,
        "summary": summarize_episode(query_results, ingestion_ms),
        "ingest_latency_ms": ingestion_ms,
        "queries": query_results,
    }


def evaluate(args) -> Path:
    dataset_dir = Path(args.dataset).resolve()
    manifest_path = dataset_dir / "benchmark_manifest.json"
    with open(manifest_path, encoding="utf-8") as file:
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

    resume_dir = getattr(args, "resume", None)
    if resume_dir and args.output:
        raise ValueError("--resume and --output cannot be used together")
    if resume_dir:
        output_dir = Path(resume_dir).resolve()
    elif args.output:
        output_dir = Path(args.output).resolve()
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = dataset_dir / "reports" / adapter.name / timestamp

    expected_config = _checkpoint_config(
        manifest_path=manifest_path,
        manifest=manifest,
        adapter=adapter,
        adapter_spec=args.adapter,
        adapter_kwargs=adapter_kwargs,
        top_k=args.top_k,
        episode_records=episode_records,
    )
    checkpoint_root = output_dir / "checkpoints"
    config_path = checkpoint_root / "run.json"
    if resume_dir:
        if not output_dir.is_dir():
            raise RuntimeError(f"resume output does not exist: {output_dir}")
        if (output_dir / "results.json").exists():
            raise RuntimeError(f"benchmark run is already complete: {output_dir}")
        if not config_path.is_file():
            raise RuntimeError(f"resume checkpoint config not found: {config_path}")
        saved_config = _read_json(config_path)
        if saved_config != expected_config:
            mismatches = sorted(
                key
                for key in set(saved_config) | set(expected_config)
                if saved_config.get(key) != expected_config.get(key)
            )
            raise RuntimeError(
                "resume configuration does not match the checkpoint: "
                + ", ".join(mismatches)
            )
    else:
        if output_dir.exists():
            raise RuntimeError(f"report output already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        (checkpoint_root / "episodes").mkdir(parents=True)
        _write_json(config_path, expected_config)

    state_root = output_dir / "state"
    state_root.mkdir(exist_ok=True)
    episode_results = _load_episode_checkpoints(checkpoint_root, episode_records)
    loaded_episode_count = len(episode_results)
    completed_ids = {item["episode_id"] for item in episode_results}
    prior_attempts = _load_attempt_metadata(checkpoint_root)
    attempt_path = _next_attempt_path(checkpoint_root)
    if loaded_episode_count:
        print(
            f"[memory-eval] resume loaded {loaded_episode_count}/"
            f"{len(episode_records)} completed episode checkpoints"
        )

    metadata_collector = RunMetadataCollector(
        dataset_dir=dataset_dir,
        episode_records=episode_records,
        hams_root=Path(__file__).resolve().parents[2],
    )

    try:
        for episode_record in episode_records:
            episode_id = episode_record["episode_id"]
            if episode_id in completed_ids:
                continue
            episode_state = state_root / episode_id
            if episode_state.exists():
                shutil.rmtree(episode_state)
            episode_result = evaluate_episode(
                adapter=adapter,
                episode_dir=dataset_dir / "episodes" / episode_id,
                state_dir=episode_state,
                top_k=args.top_k,
            )
            episode_results.append(episode_result)
            completed_ids.add(episode_id)
            checkpoint_path = checkpoint_root / "episodes" / f"{episode_id}.json"
            _write_json(checkpoint_path, episode_result)
            print(
                f"[memory-eval] checkpoint saved: {len(episode_results)}/"
                f"{len(episode_records)} episodes ({checkpoint_path})"
            )
        adapter_metadata = _adapter_run_metadata(adapter)
        attempt_metadata = metadata_collector.finish(
            status=_completed_status(adapter_metadata),
            episode_results=episode_results,
            state_root=state_root,
            adapter_metadata=adapter_metadata,
        )
        _write_json(attempt_path, attempt_metadata)
        run_metadata = _merge_attempt_metadata(
            [*prior_attempts, attempt_metadata],
            episode_results,
            loaded_episode_count,
        )
        report = {
            "schema_version": 1,
            "benchmark_id": manifest["benchmark_id"],
            "dataset": str(dataset_dir),
            "adapter": adapter.name,
            "adapter_spec": args.adapter,
            "adapter_kwargs": adapter_kwargs,
            "top_k": args.top_k,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_metadata": run_metadata,
            "aggregate": aggregate_episode_summaries(
                [item["summary"] for item in episode_results]
            ),
            "episodes": episode_results,
        }
        _write_json(output_dir / "run_metadata.json", run_metadata)
        _write_json(output_dir / "results.json", report)
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
    except BaseException as error:
        adapter.close()
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        attempt_metadata = metadata_collector.finish(
            status=status,
            episode_results=episode_results,
            state_root=state_root,
            adapter_metadata=_adapter_run_metadata(adapter),
            failure=error,
        )
        _write_json(attempt_path, attempt_metadata)
        run_metadata = _merge_attempt_metadata(
            [*prior_attempts, attempt_metadata],
            episode_results,
            loaded_episode_count,
        )
        _write_json(output_dir / "run_metadata.json", run_metadata)
        print(
            f"[memory-eval] {status} run metadata: "
            f"{output_dir / 'run_metadata.json'}"
        )
        print(f"[memory-eval] resume this run with --resume {output_dir}")
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
    parser.add_argument(
        "--resume",
        default=None,
        help="resume an incomplete report directory after validating its checkpoint",
    )
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
