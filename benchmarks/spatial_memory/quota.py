#!/usr/bin/env python3
"""Estimate scheduled VLM calls for a frozen spatial-memory benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def query_counts(dataset_dir: Path, max_episodes: int | None = None) -> list[int]:
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
    counts = []
    for episode in episodes:
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
) -> dict[str, int]:
    if call_limit <= 0:
        raise ValueError("call_limit must be positive")
    counts = query_counts(dataset_dir, max_episodes=max_episodes)
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
    args = parser.parse_args()
    summary = quota_summary(
        Path(args.dataset),
        call_limit=args.call_limit,
        max_episodes=args.max_episodes,
    )
    print(
        summary["scheduled_calls"],
        summary["selected_episodes"],
        summary["safe_prefix_episodes"],
        summary["safe_prefix_calls"],
    )


if __name__ == "__main__":
    main()
