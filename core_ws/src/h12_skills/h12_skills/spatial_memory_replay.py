#!/usr/bin/env python3
"""Replay a frozen RoboCasa episode through the H12 memory backend.

This is the integration test for machines where H12 locomotion is unavailable:
it feeds the exact image, pose, and timestamp stream that the ROS node would
receive, then runs every checkpoint query without importing benchmark adapters.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image

from .skills.spatial_memory import EmbodiedAgentMemory


def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _observation_id(memory_id: str) -> str:
    return f'obs_{int(memory_id.removeprefix("mem_")):06d}'


def replay_episode(
    episode_dir: Path,
    data_dir: Path,
    embodied_agent_root: Path,
    model: str = 'siglip_base',
    device: str = 'auto',
    top_k: int = 3,
    rerank: bool = False,
    verify_restart: bool = False,
) -> dict:
    """Replay one benchmark episode and return simple query-level checks."""
    observations = _jsonl(episode_dir / 'observations.jsonl')
    queries = _jsonl(episode_dir / 'queries.jsonl')
    if not observations or not queries:
        raise RuntimeError(f'invalid benchmark episode: {episode_dir}')
    if data_dir.exists() and any(data_dir.iterdir()):
        raise RuntimeError(f'replay data directory must be empty: {data_dir}')

    by_checkpoint = {}
    for query in queries:
        by_checkpoint.setdefault(int(query['checkpoint_frame']), []).append(query)

    memory = EmbodiedAgentMemory(
        data_dir=str(data_dir),
        embodied_agent_root=str(embodied_agent_root),
        model=model,
        device=device,
    )
    records = []
    try:
        if not memory.wait_until_ready():
            raise RuntimeError('embedding model did not become ready')
        for observation in observations:
            frame_idx = int(observation['frame_idx'])
            image = Image.open(episode_dir / observation['image_path']).convert('RGB')
            memory_id = memory.add_frame(
                image,
                tuple(observation['robot_pose']),
                timestamp=observation['timestamp'],
                metadata={
                    'episode_id': episode_dir.name,
                    'replay_source_observation_id': observation['observation_id'],
                },
            )
            expected_id = f'mem_{frame_idx:06d}'
            if memory_id != expected_id:
                raise RuntimeError(
                    f'non-empty replay state: expected {expected_id}, got {memory_id}')
            checkpoint_queries = by_checkpoint.get(frame_idx, [])
            if not checkpoint_queries:
                continue
            memory.flush()
            for query in checkpoint_queries:
                outcome = memory.query(
                    query['text'], top_k=top_k, rerank=rerank)
                ranked = [_observation_id(hit.memory_id) for hit in outcome.hits]
                relevant = set(query.get('relevant_observation_ids', []))
                stale = set(query.get('stale_observation_ids', []))
                record = {
                    'query_id': query['query_id'],
                    'track': query['track'],
                    'ranked_observation_ids': ranked,
                    'recall_at_k': (
                        bool(relevant.intersection(ranked)) if relevant else None),
                    'top1_relevant': (
                        bool(ranked and ranked[0] in relevant) if relevant else None),
                    'stale_top1': bool(ranked and ranked[0] in stale),
                    'rerank_attempted': outcome.rerank_attempted,
                    'rerank_valid': outcome.rerank_valid,
                    'fallback_reason': outcome.fallback_reason,
                }
                records.append(record)
                print(json.dumps(record, sort_keys=True))
    finally:
        memory.close()

    scored = [record for record in records if record['recall_at_k'] is not None]
    summary = {
        'episode': episode_dir.name,
        'query_count': len(records),
        'scored_query_count': len(scored),
        'recall_at_k': (
            sum(record['recall_at_k'] for record in scored) / len(scored)
            if scored else None),
        'top1_accuracy': (
            sum(record['top1_relevant'] for record in scored) / len(scored)
            if scored else None),
        'stale_top1_rate': (
            sum(record['stale_top1'] for record in records) / len(records)
            if records else None),
    }
    if verify_restart:
        restarted = EmbodiedAgentMemory(
            data_dir=str(data_dir),
            embodied_agent_root=str(embodied_agent_root),
            model=model,
            device=device,
        )
        try:
            if not restarted.wait_until_ready():
                raise RuntimeError('restarted embedding model did not become ready')
            warm_hits = restarted.query(
                queries[0]['text'], top_k=top_k, rerank=False).hits
            summary['restart_warm_index_ok'] = bool(warm_hits)
            summary['restart_hit_count'] = len(warm_hits)
        finally:
            restarted.close()
    print(json.dumps({'summary': summary}, sort_keys=True))
    return summary


def main(argv=None):
    """CLI entry point for movement-free integration replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode_dir', type=Path)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument(
        '--embodied-agent-root',
        type=Path,
        default=Path(os.environ.get('EMBODIED_AGENT_ROOT', '/opt/EmbodiedAgent')),
    )
    parser.add_argument('--model', default='siglip_base')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--top-k', default=3, type=int)
    parser.add_argument('--rerank', action='store_true')
    parser.add_argument('--verify-restart', action='store_true')
    args = parser.parse_args(argv)
    replay_episode(
        episode_dir=args.episode_dir.resolve(),
        data_dir=args.data_dir.resolve(),
        embodied_agent_root=args.embodied_agent_root.resolve(),
        model=args.model,
        device=args.device,
        top_k=args.top_k,
        rerank=args.rerank,
        verify_restart=args.verify_restart,
    )


if __name__ == '__main__':
    main()
