#!/usr/bin/env python3
"""Estimate scheduled VLM calls for a frozen spatial-memory benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def completed_episode_ids(resume_output: Path | None) -> set[str]:
    if resume_output is None:
        return set()
    checkpoint_root = Path(resume_output) / "checkpoints"
    config_path = checkpoint_root / "run.json"
    if not config_path.is_file():
        raise ValueError(f"resume checkpoint config not found: {config_path}")
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    selected = config.get("episode_ids")
    if not isinstance(selected, list) or not all(
        isinstance(item, str) for item in selected
    ):
        raise ValueError(f"invalid resume checkpoint config: {config_path}")
    completed = set()
    found_gap = False
    for episode_id in selected:
        path = checkpoint_root / "episodes" / f"{episode_id}.json"
        if not path.is_file():
            found_gap = True
            continue
        if found_gap:
            raise ValueError(
                "resume checkpoints are not a contiguous episode prefix"
            )
        with open(path, encoding="utf-8") as file:
            result = json.load(file)
        if result.get("episode_id") != episode_id:
            raise ValueError(f"checkpoint episode mismatch: {path}")
        completed.add(episode_id)
    return completed


def query_counts(
    dataset_dir: Path,
    max_episodes: int | None = None,
    completed_ids: set[str] | None = None,
) -> list[int]:
    manifest_path = Path(dataset_dir) / "benchmark_manifest.json"
    with open(manifest_path, encoding="utf-8") as file:
        manifest = json.load(file)
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("benchmark manifest does not contain an episode list")
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        episodes = episodes[:max_episodes]
    completed_ids = completed_ids or set()
    counts = []
    for episode in episodes:
        episode_id = episode.get("episode_id")
        if episode_id in completed_ids:
            continue
        count = episode.get("query_count")
        if not isinstance(count, int) or count < 0:
            raise ValueError(
                f"invalid query_count for episode {episode.get('episode_id')!r}"
            )
        counts.append(count)
    return counts


def quota_summary(
    dataset_dir: Path,
    call_limit: int,
    max_episodes: int | None = None,
    resume_output: Path | None = None,
) -> dict[str, int]:
    if call_limit <= 0:
        raise ValueError("call_limit must be positive")
    completed_ids = completed_episode_ids(resume_output)
    counts = query_counts(
        dataset_dir,
        max_episodes=max_episodes,
        completed_ids=completed_ids,
    )
    safe_episodes = 0
    safe_calls = 0
    for count in counts:
        if safe_calls + count > call_limit:
            break
        safe_calls += count
        safe_episodes += 1
    return {
        "scheduled_calls": sum(counts),
        "selected_episodes": len(counts),
        "safe_prefix_episodes": safe_episodes,
        "safe_prefix_calls": safe_calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--call-limit", required=True, type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--resume-output")
    args = parser.parse_args()
    summary = quota_summary(
        Path(args.dataset),
        call_limit=args.call_limit,
        max_episodes=args.max_episodes,
        resume_output=Path(args.resume_output) if args.resume_output else None,
    )
    print(
        summary["scheduled_calls"],
        summary["selected_episodes"],
        summary["safe_prefix_episodes"],
        summary["safe_prefix_calls"],
    )


if __name__ == "__main__":
    main()
