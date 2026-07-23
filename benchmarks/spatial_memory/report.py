#!/usr/bin/env python3
"""Render benchmark ``results.json`` as browsable HTML and concise Markdown."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import statistics
from pathlib import Path
from urllib.parse import quote


METRICS = [
    ("static_recall_at_k", "Static Recall@K", "rate"),
    ("static_coverage_at_k", "Static visible-view coverage@K", "rate"),
    ("static_mrr", "Static MRR", "rate"),
    ("static_location_error_m_top1", "Static location error Top-1", "meters"),
    ("static_location_success_at_0_5m", "Static location success @ 0.5 m", "rate"),
    ("live_current_recall_at_k", "Current-location Recall@K", "rate"),
    ("live_current_coverage_at_k", "Current visible-view coverage@K", "rate"),
    ("live_latest_visible_frame_recall_at_k", "Latest visible frame Recall@K", "rate"),
    ("live_latest_visible_frame_top1_accuracy", "Latest visible frame Top-1", "rate"),
    ("live_current_top1_accuracy", "Current Top-1", "rate"),
    ("live_stale_top1_rate", "Stale Top-1 rate", "rate"),
    ("live_stale_fraction_at_k", "Stale fraction@K", "rate"),
    ("live_location_error_m_top1", "Current location error Top-1", "meters"),
    ("live_location_success_at_0_5m", "Current location success @ 0.5 m", "rate"),
    ("live_stale_location_top1_rate", "Stale-location Top-1 rate", "rate"),
    ("update_lag_frames_at_k", "Update lag@K", "frames"),
    ("update_lag_frames_top1", "Update lag Top-1", "frames"),
    ("location_update_lag_frames_at_0_5m", "Location update lag @ 0.5 m", "frames"),
    ("history_recall_at_k", "History Recall@K", "rate"),
    ("history_coverage_at_k", "History visible-view coverage@K", "rate"),
    ("history_mrr", "History MRR", "rate"),
    ("history_location_error_m_top1", "History location error Top-1", "meters"),
    ("history_location_success_at_0_5m", "History location success @ 0.5 m", "rate"),
    ("absent_top1_score", "Absent Top-1 score", "score"),
    ("absent_top1_confidence", "Absent Top-1 confidence", "rate"),
    ("absent_false_positive_rate_at_0_5", "Absent false-positive rate @ 0.5", "rate"),
    ("vlm_faiss_recall_pool_hit_rate", "FAISS recall-pool hit rate", "rate"),
    ("vlm_faiss_recall_pool_coverage", "FAISS recall-pool coverage", "rate"),
    ("vlm_valid_response_rate", "VLM valid-response rate", "rate"),
    ("vlm_fallback_rate", "VLM fallback rate", "rate"),
    ("ingest_latency_ms_p50", "Ingest latency p50", "ms"),
    ("ingest_latency_ms_p95", "Ingest latency p95", "ms"),
    ("query_latency_ms_p50", "Query latency p50", "ms"),
    ("query_latency_ms_p95", "Query latency p95", "ms"),
]

PRIMARY_METRICS = [
    "static_recall_at_k",
    "live_current_recall_at_k",
    "live_latest_visible_frame_recall_at_k",
    "live_stale_top1_rate",
    "live_location_success_at_0_5m",
    "live_stale_location_top1_rate",
    "update_lag_frames_at_k",
    "history_recall_at_k",
]

TRACK_LABELS = {
    "static": "Static",
    "live_current": "Live current",
    "history": "History",
    "absent": "Absent",
}


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _format_metric(value, kind: str) -> str:
    if value is None:
        return "N/A"
    number = float(value)
    if not math.isfinite(number):
        return "N/A"
    if kind == "rate":
        return f"{number * 100.0:.1f}%"
    if kind == "frames":
        return f"{number:.1f} frames"
    if kind == "ms":
        return f"{number:.1f} ms"
    if kind == "meters":
        return f"{number:.3f} m"
    return f"{number:.4f}"


def _format_bytes(value) -> str:
    if value is None:
        return "N/A"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return "N/A"


def _nested(payload: dict, *keys):
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _git_label(value) -> str:
    if not isinstance(value, dict) or not value.get("commit"):
        return "N/A"
    label = str(value["commit"])[:12]
    if value.get("dirty"):
        count = value.get("dirty_entry_count")
        label += f" + dirty ({int(count)})" if count is not None else " + dirty"
    return label


def _run_metadata_rows(report: dict) -> list[tuple[str, str]]:
    metadata = report.get("run_metadata") or {}
    if not metadata:
        return []
    gpu_devices = _nested(metadata, "resources", "gpu", "devices") or []
    gpu_name = ", ".join(str(item.get("name", "unknown")) for item in gpu_devices)
    gpu_peak = sum(int(item.get("peak_allocated_bytes", 0)) for item in gpu_devices)
    rows = [
        ("Run status", str(metadata.get("status", "unknown"))),
        ("Wall time", f"{float(metadata.get('wall_time_seconds', 0.0)):.1f} s"),
        ("Process peak RAM", _format_bytes(_nested(metadata, "resources", "process_peak_rss_bytes"))),
        ("GPU", gpu_name or "CPU / unavailable"),
        ("Peak allocated VRAM", _format_bytes(gpu_peak) if gpu_devices else "N/A"),
        ("Dataset payload", _format_bytes(_nested(metadata, "resources", "storage", "payload_bytes"))),
        ("Adapter state", _format_bytes(_nested(metadata, "resources", "storage", "adapter_state_bytes"))),
        ("Container image", str(_nested(metadata, "runtime", "container_image") or "N/A")),
        ("HAMS commit", _git_label(_nested(metadata, "software", "hams"))),
        ("EmbodiedAgent commit", _git_label(_nested(metadata, "software", "embodied_agent"))),
    ]
    vlm = _nested(metadata, "adapter", "vlm")
    if isinstance(vlm, dict):
        rows.extend([
            ("VLM logical calls / API attempts", f"{int(vlm.get('logical_calls', 0))} / {int(vlm.get('api_attempts', 0))}"),
            ("VLM retries / API errors", f"{int(vlm.get('retry_count', 0))} / {int(vlm.get('api_errors', 0))}"),
            ("VLM parse failures", f"{int(vlm.get('parse_failures', 0))}"),
            ("VLM valid / failed calls", f"{int(vlm.get('successful_calls', 0))} / {int(vlm.get('failed_calls', 0))}"),
            ("Input / output tokens", f"{int(vlm.get('prompt_tokens', 0)):,} / {int(vlm.get('output_tokens_including_thoughts', 0)):,}"),
            ("Total tokens", f"{int(vlm.get('total_tokens', 0)):,}"),
            ("Estimated standard cost", (
                f"${float(vlm['estimated_standard_cost_usd']):.4f}"
                if vlm.get("estimated_standard_cost_usd") is not None else "N/A"
            )),
        ])
        if vlm.get("error_types"):
            rows.append((
                "VLM error types",
                ", ".join(
                    f"{name} × {int(count)}"
                    for name, count in sorted(vlm["error_types"].items())
                ),
            ))
        if vlm.get("last_error"):
            rows.append(("Last VLM error", str(vlm["last_error"])[:240]))
    return rows


def _run_metadata_html(report: dict) -> str:
    rows = _run_metadata_rows(report)
    if not rows:
        return ""
    body = "".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
        for label, value in rows
    )
    pricing = _nested(report, "run_metadata", "adapter", "vlm", "pricing_assumption")
    pricing_note = ""
    if isinstance(pricing, dict):
        source = html.escape(str(pricing.get("source", "")), quote=True)
        pricing_note = (
            '<p class="meta-note">Cost uses the recorded standard paid-tier list-price '
            f'assumption verified {html.escape(str(pricing.get("verified_on", "unknown")))}. '
            f'<a href="{source}">Pricing source</a>; free-tier actual charge may be zero.</p>'
        )
    return (
        '<h2>Run metadata</h2><div class="run-metadata">'
        f'<table class="metric-table"><tbody>{body}</tbody></table>{pricing_note}</div>'
    )


def _metric_definition(metric_name: str) -> tuple[str, str]:
    for name, label, kind in METRICS:
        if name == metric_name:
            return label, kind
    return metric_name.replace("_", " ").title(), "score"


def _relative_url(path: Path, output_dir: Path) -> str:
    relative = os.path.relpath(path.resolve(), start=output_dir.resolve())
    return quote(relative.replace(os.sep, "/"), safe="/._-")


def _markdown_link(path: Path, output_dir: Path) -> str:
    return os.path.relpath(path.resolve(), start=output_dir.resolve()).replace(os.sep, "/")


def _episode_manifest_map(manifest: dict) -> dict[str, dict]:
    return {item["episode_id"]: item for item in manifest.get("episodes", [])}


def _augment_legacy_metrics(report: dict) -> None:
    """Derive newer coverage metrics when rendering an older results.json."""
    summary_keys = (
        "static_coverage_at_k",
        "live_current_coverage_at_k",
        "live_latest_visible_frame_recall_at_k",
        "history_coverage_at_k",
    )
    for episode in report.get("episodes", []):
        by_track: dict[str, list[dict]] = {}
        for query_result in episode.get("queries", []):
            metrics = query_result.setdefault("metrics", {})
            relevant = set(query_result.get("relevant_observation_ids", []))
            ranked = {
                item["observation_id"] for item in query_result.get("candidates", [])
            }
            if relevant:
                metrics.setdefault(
                    "relevant_coverage_at_k", len(relevant & ranked) / len(relevant)
                )
                checkpoint_id = f"obs_{int(query_result['checkpoint_frame']):06d}"
                if checkpoint_id in relevant:
                    metrics.setdefault(
                        "checkpoint_recall_at_k", float(checkpoint_id in ranked)
                    )
            by_track.setdefault(query_result["track"], []).append(metrics)

        def mean(track: str, key: str) -> float | None:
            values = [
                item[key] for item in by_track.get(track, []) if item.get(key) is not None
            ]
            return statistics.fmean(values) if values else None

        summary = episode.setdefault("summary", {})
        derived = {
            "static_coverage_at_k": mean("static", "relevant_coverage_at_k"),
            "live_current_coverage_at_k": mean(
                "live_current", "relevant_coverage_at_k"
            ),
            "live_latest_visible_frame_recall_at_k": mean(
                "live_current", "checkpoint_recall_at_k"
            ),
            "history_coverage_at_k": mean("history", "relevant_coverage_at_k"),
        }
        for name, value in derived.items():
            summary.setdefault(name, value)

    aggregate = report.setdefault("aggregate", {})
    for name in summary_keys:
        values = [
            episode["summary"][name]
            for episode in report.get("episodes", [])
            if episode.get("summary", {}).get(name) is not None
        ]
        aggregate.setdefault(name, statistics.fmean(values) if values else None)


def _observation_map(dataset_dir: Path, episode_id: str) -> dict[str, dict]:
    observations = _read_jsonl(
        dataset_dir / "episodes" / episode_id / "observations.jsonl"
    )
    return {item["observation_id"]: item for item in observations}


def _oracle_timeline(dataset_dir: Path, episode_id: str) -> dict | None:
    path = dataset_dir / "episodes" / episode_id / "oracle" / "episode.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as file:
        oracle = json.load(file)

    timeline = {}
    for lap, pose_key, surface_key in (
        (1, "target_pose_a_xyzw", "surface_a"),
        (2, "target_pose_b_xyzw", "surface_b"),
    ):
        frames = [item for item in oracle.get("frames", []) if item.get("lap") == lap]
        if not frames:
            continue
        pose = oracle.get(pose_key, [])
        timeline[lap] = {
            "surface": oracle.get(surface_key, {}).get("name", "?"),
            "first_frame": min(int(item["frame_idx"]) for item in frames),
            "last_frame": max(int(item["frame_idx"]) for item in frames),
            "visible_frames": [
                int(item["frame_idx"]) for item in frames if item.get("target_visible")
            ],
            "target_xyz": [float(value) for value in pose[:3]] if len(pose) >= 3 else None,
        }
    return timeline if len(timeline) == 2 else None


def _frame_label(observation_id: str, observation: dict | None) -> str:
    if observation is not None and "frame_idx" in observation:
        return f"Frame {int(observation['frame_idx']):06d}"
    try:
        return f"Frame {int(observation_id.rsplit('_', maxsplit=1)[1]):06d}"
    except (IndexError, ValueError):
        return observation_id


def _relevance_label(track: str, relevant: bool, stale: bool) -> str:
    if stale:
        return '<span class="tag stale-tag">stale location A</span>'
    if not relevant:
        return ""
    text = {
        "static": "correct location A",
        "live_current": "current location B",
        "history": "historical location A",
    }.get(track, "relevant")
    return f'<span class="tag relevant-tag">{html.escape(text)}</span>'


def _candidate_gallery(
    query_result: dict,
    observations: dict[str, dict],
    episode_dir: Path,
    output_dir: Path,
) -> str:
    relevant = set(query_result.get("relevant_observation_ids", []))
    stale = set(query_result.get("stale_observation_ids", []))
    track = query_result.get("track", "")
    figures = []
    for candidate in query_result.get("candidates", []):
        observation_id = candidate["observation_id"]
        observation = observations.get(observation_id)
        if observation is None:
            image_html = '<div class="missing-image">missing image</div>'
        else:
            image_path = episode_dir / observation["image_path"]
            image_url = html.escape(_relative_url(image_path, output_dir), quote=True)
            image_html = (
                f'<a href="{image_url}"><img src="{image_url}" '
                f'alt="{html.escape(observation_id, quote=True)}"></a>'
            )
        classes = ["candidate"]
        is_relevant = observation_id in relevant
        is_stale = observation_id in stale
        if observation_id in relevant:
            classes.append("relevant")
        elif observation_id in stale:
            classes.append("stale")
        label = _relevance_label(track, is_relevant, is_stale)
        frame_label = _frame_label(observation_id, observation)
        metadata = candidate.get("metadata") or {}
        score_label = str(metadata.get("score_label", "score"))
        rerank_details = ""
        if metadata.get("faiss_rank") is not None:
            rerank_details += (
                f'<br><span class="rerank-detail">FAISS Rank '
                f'{int(metadata["faiss_rank"])} · cosine '
                f'{float(metadata.get("faiss_score", 0.0)):.4f}</span>'
            )
        if metadata.get("rerank_reasoning"):
            rerank_details += (
                f'<span class="rerank-reason">'
                f'{html.escape(str(metadata["rerank_reasoning"]))}</span>'
            )
        predicted_xyz = metadata.get("predicted_world_xyz")
        if isinstance(predicted_xyz, list) and len(predicted_xyz) == 3:
            rerank_details += (
                '<br><span class="rerank-detail">predicted xyz '
                f'({float(predicted_xyz[0]):.3f}, '
                f'{float(predicted_xyz[1]):.3f}, '
                f'{float(predicted_xyz[2]):.3f}) m</span>'
            )
        if metadata.get("selection_mode"):
            rerank_details += (
                '<br><span class="rerank-detail">selection: '
                f'{html.escape(str(metadata["selection_mode"]))}</span>'
            )
        figures.append(
            f'<figure class="{" ".join(classes)}">'
            f"{image_html}<figcaption><strong>Rank {int(candidate['rank'])}</strong><br>"
            f'<span class="frame-label">{html.escape(frame_label)}</span><br>'
            f'<span class="observation-id">{html.escape(observation_id)}</span><br>'
            f"{html.escape(score_label)} {float(candidate['score']):.4f}"
            f"{rerank_details}<br>{label}</figcaption></figure>"
        )
    return '<div class="candidate-gallery">' + "".join(figures) + "</div>"


def _metric_table_html(summary: dict) -> str:
    rows = []
    for name, label, kind in METRICS:
        if name not in summary:
            continue
        rows.append(
            f"<tr><th>{html.escape(label)}</th>"
            f"<td>{html.escape(_format_metric(summary.get(name), kind))}</td></tr>"
        )
    return '<table class="metric-table"><tbody>' + "".join(rows) + "</tbody></table>"


def _query_html(
    query_result: dict,
    observations: dict[str, dict],
    episode_dir: Path,
    output_dir: Path,
    top_k: int,
) -> str:
    metrics = query_result["metrics"]
    recall = metrics.get("recall_at_k")
    recall_text = "N/A" if recall is None else ("hit" if recall == 1.0 else "miss")
    recall_class = "neutral" if recall is None else ("hit" if recall == 1.0 else "miss")
    stale = metrics.get("stale_top1", 0.0)
    stale_html = ""
    if query_result.get("track") == "live_current":
        stale_text = "stale A at Top-1" if stale == 1.0 else "Top-1 is not stale A"
        stale_class = "miss" if stale == 1.0 else "neutral"
        stale_html = f'<span class="result {stale_class}">{stale_text}</span>'
    track = TRACK_LABELS.get(query_result["track"], query_result["track"])
    diagnostics = query_result.get("adapter_diagnostics") or {}
    rerank_html = ""
    if diagnostics.get("rerank_attempted") is True:
        pool_metrics = query_result.get("faiss_recall_pool_metrics") or {}
        pool_recall = pool_metrics.get("recall_at_k")
        pool_html = ""
        if pool_recall is not None:
            pool_class = "hit" if pool_recall == 1.0 else "miss"
            pool_text = "hit" if pool_recall == 1.0 else "miss"
            pool_html = (
                f'<span class="result {pool_class}">FAISS pool: {pool_text}</span>'
            )
        if diagnostics.get("rerank_valid") is True:
            rerank_html = (
                '<span class="result hit">VLM rerank valid · '
                f'{int(diagnostics.get("recall_count", 0))} candidates</span>'
                f"{pool_html}"
            )
        else:
            rerank_html = (
                '<span class="result miss">VLM invalid · FAISS fallback</span>'
                f"{pool_html}"
            )
    map_html = ""
    if diagnostics.get("map_native") is True:
        selection_mode = str(diagnostics.get("selection_mode", "map query"))
        selection_class = (
            "miss" if "fallback" in selection_mode else "neutral"
        )
        map_html = (
            f'<span class="result {selection_class}">'
            f'{html.escape(selection_mode)} · '
            f'{int(diagnostics.get("voxel_count", 0))} voxels · '
            f'{int(diagnostics.get("component_count", 0))} components</span>'
        )
    return (
        '<article class="query">'
        '<div class="query-heading">'
        f'<span class="track">{html.escape(track)}</span>'
        f'<code>{html.escape(query_result["query_id"])}</code>'
        f'<span>queried after Frame {int(query_result["checkpoint_frame"]):06d}</span>'
        f'<span>{float(query_result["query_latency_ms"]):.1f} ms</span>'
        "</div>"
        f'<p class="query-text">{html.escape(query_result["text"])}</p>'
        '<div class="query-outcome">'
        f'<span class="result {recall_class}">Recall@{top_k}: {recall_text}</span>'
        f"{stale_html}"
        f"{rerank_html}"
        f"{map_html}"
        "</div>"
        f"{_candidate_gallery(query_result, observations, episode_dir, output_dir)}"
        "</article>"
    )


def _episode_html(
    episode_result: dict,
    episode_meta: dict,
    dataset_dir: Path,
    output_dir: Path,
    top_k: int,
    open_by_default: bool,
) -> str:
    episode_id = episode_result["episode_id"]
    episode_dir = dataset_dir / "episodes" / episode_id
    observations = _observation_map(dataset_dir, episode_id)
    contact_sheet = episode_dir / "contact_sheet.jpg"
    contact_url = html.escape(_relative_url(contact_sheet, output_dir), quote=True)
    query_sections = "".join(
        _query_html(query, observations, episode_dir, output_dir, top_k)
        for query in episode_result["queries"]
    )
    open_attr = " open" if open_by_default else ""
    target = episode_meta.get("target_language", episode_meta.get("object_group", "unknown"))
    surface_a = episode_meta.get("surface_a", "?")
    surface_b = episode_meta.get("surface_b", "?")
    visible_a = episode_meta.get("visible_frames_lap1", "?")
    visible_b = episode_meta.get("visible_frames_lap2", "?")
    timeline = _oracle_timeline(dataset_dir, episode_id)
    timeline_html = ""
    if timeline is not None:
        stops = []
        for lap, location in ((1, "A"), (2, "B")):
            item = timeline[lap]
            visible_text = ", ".join(
                f"Frame {frame:06d}" for frame in item["visible_frames"]
            ) or "none"
            xyz = item["target_xyz"]
            xyz_text = (
                f"({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) m"
                if xyz is not None else "unknown"
            )
            stops.append(
                '<div class="timeline-stop">'
                f'<strong>Lap {lap}: location {location}</strong>'
                f'<span>{html.escape(str(item["surface"]))}</span>'
                f'<span>Frames {item["first_frame"]:06d}–{item["last_frame"]:06d}</span>'
                f'<span>Target visible in: {html.escape(visible_text)}</span>'
                f'<code>target xyz = {html.escape(xyz_text)}</code>'
                "</div>"
            )
        timeline_html = (
            '<div class="ground-truth"><h3>Ground-truth timeline</h3>'
            '<p>There are exactly two physical target locations. Multiple visible frames '
            "within one lap are different camera views of the same target pose.</p>"
            '<div class="timeline">'
            f'{stops[0]}<div class="relocation-arrow">one A→B relocation</div>{stops[1]}'
            "</div></div>"
        )
    return (
        f'<details class="episode"{open_attr}>'
        "<summary>"
        f'<span class="episode-title">{html.escape(episode_id)}</span>'
        f'<span>{html.escape(str(target))}</span>'
        f'<span>{html.escape(str(surface_a))} &rarr; {html.escape(str(surface_b))}</span>'
        "</summary>"
        '<div class="episode-body">'
        '<div class="episode-overview">'
        '<div class="scene-card">'
        f'<a href="{contact_url}"><img src="{contact_url}" alt="{html.escape(episode_id)} contact sheet"></a>'
        f'<p><strong>Target:</strong> {html.escape(str(target))}<br>'
        f'<strong>Relocation:</strong> {html.escape(str(surface_a))} &rarr; {html.escape(str(surface_b))}<br>'
        f'<strong>Visible frames:</strong> lap 1 = {visible_a}, lap 2 = {visible_b}<br>'
        f'<strong>Frames / queries:</strong> {episode_result["frame_count"]} / {episode_result["query_count"]}</p>'
        "</div>"
        f"{_metric_table_html(episode_result['summary'])}"
        "</div>"
        f"{timeline_html}"
        f'<h3>Query timeline and Top-{top_k}</h3>{query_sections}'
        "</div></details>"
    )


def _html_document(
    report: dict,
    manifest: dict,
    dataset_dir: Path,
    output_dir: Path,
) -> str:
    aggregate = report["aggregate"]
    definitions = {name: (label, kind) for name, label, kind in METRICS}
    cards = []
    for name in PRIMARY_METRICS:
        label, kind = definitions[name]
        cards.append(
            '<div class="metric-card">'
            f'<span>{html.escape(label)}</span>'
            f'<strong>{html.escape(_format_metric(aggregate.get(name), kind))}</strong>'
            "</div>"
        )
    episode_meta = _episode_manifest_map(manifest)
    episodes = "".join(
        _episode_html(
            episode,
            episode_meta.get(episode["episode_id"], {}),
            dataset_dir,
            output_dir,
            int(report["top_k"]),
            index == 0,
        )
        for index, episode in enumerate(report["episodes"])
    )
    css = """
    :root { color-scheme: light; --ink:#17212b; --muted:#66717d; --line:#d9e0e7;
      --panel:#fff; --soft:#f4f7fa; --blue:#155eef; --green:#067647; --red:#b42318;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }
    * { box-sizing:border-box; } body { margin:0; color:var(--ink); background:#eef2f6; }
    main { max-width:1500px; margin:auto; padding:28px; } h1 { margin:0 0 8px; }
    h2 { margin-top:34px; } h3 { margin:24px 0 12px; } p { line-height:1.5; }
    .subtitle { color:var(--muted); margin:0 0 16px; } .meta { display:flex; gap:8px;
      flex-wrap:wrap; } .meta span,.track,.result,.tag { border-radius:999px; padding:4px 9px;
      background:#e8eef5; font-size:12px; } .metric-grid { display:grid;
      grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:22px 0; }
    .metric-card { background:var(--panel); border:1px solid var(--line); border-radius:12px;
      padding:16px; box-shadow:0 2px 8px #1822300d; } .metric-card span { color:var(--muted);
      font-size:13px; display:block; } .metric-card strong { font-size:26px; display:block;
      margin-top:6px; } .note { background:#fff8e7; border:1px solid #f0d58a; padding:12px 15px;
      border-radius:10px; } .metric-table { width:100%; border-collapse:collapse; background:white; }
    .metric-table th,.metric-table td { border-bottom:1px solid var(--line); padding:8px 10px;
      text-align:left; } .metric-table th { color:var(--muted); font-weight:500; }
    .metric-table td { font-variant-numeric:tabular-nums; font-weight:650; }
    .aggregate { max-width:720px; border:1px solid var(--line); border-radius:10px; overflow:hidden; }
    .run-metadata { max-width:900px; border:1px solid var(--line); border-radius:10px;
      overflow:hidden; background:white; } .meta-note { color:var(--muted); padding:0 12px 10px;
      font-size:12px; }
    .episode { background:var(--panel); border:1px solid var(--line); border-radius:12px;
      margin:14px 0; overflow:hidden; } .episode summary { cursor:pointer; display:grid;
      grid-template-columns:minmax(210px,1fr) minmax(100px,.5fr) minmax(300px,1.5fr);
      gap:16px; align-items:center; padding:16px 20px; background:#f9fbfc; }
    .episode-title { font-weight:750; } .episode-body { padding:20px; }
    .episode-overview { display:grid; grid-template-columns:minmax(430px,1.5fr) minmax(330px,1fr);
      gap:20px; align-items:start; } .scene-card img { width:100%; max-height:300px;
      object-fit:contain; background:#111; border-radius:8px; }
    .ground-truth { margin:22px 0; padding:15px; border:1px solid #b2ccff;
      background:#f5f8ff; border-radius:10px; } .ground-truth h3 { margin:0 0 5px; }
    .ground-truth p { color:var(--muted); margin:0 0 12px; }
    .timeline { display:grid; grid-template-columns:1fr auto 1fr; gap:12px; align-items:center; }
    .timeline-stop { display:flex; flex-direction:column; gap:4px; background:white;
      border:1px solid var(--line); border-radius:8px; padding:12px; }
    .timeline-stop span,.timeline-stop code { font-size:12px; } .relocation-arrow {
      color:var(--blue); font-weight:700; font-size:12px; text-align:center; }
    .query { border-top:1px solid var(--line); padding:16px 0; } .query-heading { display:flex;
      align-items:center; gap:9px; flex-wrap:wrap; color:var(--muted); font-size:13px; }
    .query-heading code { color:var(--ink); } .query-text { font-weight:650; margin:10px 0 7px; }
    .query-outcome { display:flex; gap:7px; margin-bottom:10px; } .result.hit { color:var(--green);
      background:#dcfae6; } .result.miss { color:var(--red); background:#fee4e2; }
    .candidate-gallery { display:flex; gap:10px; overflow-x:auto; padding:2px 1px 9px; }
    .candidate { min-width:142px; width:142px; margin:0; padding:7px; border:2px solid transparent;
      border-radius:9px; background:var(--soft); } .candidate.relevant { border-color:#32d583; }
    .candidate.stale { border-color:#f97066; } .candidate img,.missing-image { width:124px;
      height:124px; object-fit:cover; border-radius:5px; display:block; background:#d0d5dd; }
    .candidate figcaption { font-size:11px; line-height:1.45; margin-top:6px; }
    .candidate .frame-label { font-weight:750; font-size:12px; } .observation-id { color:var(--muted); }
    .rerank-detail { color:var(--muted); } .rerank-reason { display:block; margin-top:5px;
      padding-top:5px; border-top:1px solid var(--line); color:#344054; }
    .tag { padding:2px 5px; font-size:9px; display:inline-block; } .relevant-tag { color:var(--green);
      background:#dcfae6; } .stale-tag { color:var(--red); background:#fee4e2; }
    footer { color:var(--muted); margin:35px 0 10px; font-size:12px; }
    @media (max-width:800px) { main { padding:14px; } .episode summary,
      .episode-overview,.timeline { grid-template-columns:1fr; } }
    """
    dataset_name = dataset_dir.name
    image_size = manifest.get("capture", {}).get("image_size")
    resolution_html = (
        f'<span>images: {int(image_size)}×{int(image_size)}</span>'
        if isinstance(image_size, (int, float)) else ""
    )
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Spatial-memory benchmark — {html.escape(dataset_name)}</title>"
        f"<style>{css}</style></head><body><main>"
        "<h1>Spatial-memory benchmark report</h1>"
        f'<p class="subtitle">Two-pass RoboCasa object relocation: surface A &rarr; surface B.</p>'
        '<div class="meta">'
        f'<span>dataset: {html.escape(dataset_name)}</span>'
        f'<span>adapter: {html.escape(str(report["adapter"]))}</span>'
        f'<span>episodes: {int(aggregate["episode_count"])}</span>'
        f'<span>Top-K: {int(report["top_k"])}</span>'
        f"{resolution_html}"
        f'<span>created: {html.escape(str(report["created_at"]))}</span>'
        "</div>"
        f'<div class="metric-grid">{"".join(cards)}</div>'
        '<p class="note"><strong>How to read this:</strong> current/relevant candidates are green; '
        "old-location stale candidates are red. Update lag starts at the first lap-2 frame where "
        "the relocated object is actually visible. <strong>Current-location Recall@K</strong> needs "
        "any correct view at B; <strong>visible-view coverage@K</strong> measures how many correct "
        "views were returned; <strong>Latest visible frame Recall@K</strong> checks the just-ingested "
        "visible frame itself. Lower stale rate and update lag are better.</p>"
        '<p class="note"><strong>Map-native metrics:</strong> adapters such as VLMaps also return '
        "a predicted world xyz. Location error and success @ 0.5 m use that coordinate; image-only "
        "adapters show N/A rather than treating camera pose as object position. A top-similarity "
        "fallback is displayed explicitly when no voxel wins VLMaps category classification.</p>"
        f'<h2>Aggregate metrics</h2><div class="aggregate">{_metric_table_html(aggregate)}</div>'
        f'{_run_metadata_html(report)}'
        f"<h2>Episodes</h2>{episodes}"
        '<footer>Generated from results.json. Oracle visibility labels were used only by the evaluator, '
        "not supplied to the memory adapter.</footer></main></body></html>"
    )


def _markdown_document(
    report: dict,
    manifest: dict,
    dataset_dir: Path,
    output_dir: Path,
) -> str:
    image_size = manifest.get("capture", {}).get("image_size")
    lines = [
        "# Spatial-memory benchmark report",
        "",
        f"- Dataset: `{dataset_dir.name}`",
        f"- Adapter: `{report['adapter']}`",
        f"- Episodes: {report['aggregate']['episode_count']}",
        f"- Top-K: {report['top_k']}",
        *(
            [f"- Image resolution: {int(image_size)}×{int(image_size)}"]
            if isinstance(image_size, (int, float)) else []
        ),
        f"- Created: {report['created_at']}",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name, label, kind in METRICS:
        if name in report["aggregate"]:
            lines.append(f"| {label} | {_format_metric(report['aggregate'].get(name), kind)} |")
    metadata_rows = _run_metadata_rows(report)
    if metadata_rows:
        lines.extend([
            "",
            "## Run metadata",
            "",
            "| Field | Value |",
            "|---|---:|",
            *(f"| {label} | {value} |" for label, value in metadata_rows),
        ])
        pricing = _nested(
            report, "run_metadata", "adapter", "vlm", "pricing_assumption"
        )
        if isinstance(pricing, dict):
            lines.extend([
                "",
                "Cost is an estimate using the recorded standard paid-tier list price "
                f"verified {pricing.get('verified_on', 'unknown')}; free-tier actual "
                "charge may be zero.",
            ])
    lines.extend([
        "",
        "> Update lag begins at the first lap-2 frame where the relocated object is visible. "
        "The memory adapter never receives oracle visibility labels.",
        "",
        "## Episodes",
        "",
    ])
    episode_meta = _episode_manifest_map(manifest)
    for episode in report["episodes"]:
        episode_id = episode["episode_id"]
        meta = episode_meta.get(episode_id, {})
        episode_dir = dataset_dir / "episodes" / episode_id
        contact = _markdown_link(episode_dir / "contact_sheet.jpg", output_dir)
        target = meta.get("target_language", meta.get("object_group", "unknown"))
        timeline = _oracle_timeline(dataset_dir, episode_id)
        lines.extend([
            f"### {episode_id}",
            "",
            f"Target: **{target}**  ",
            f"Relocation: `{meta.get('surface_a', '?')}` → `{meta.get('surface_b', '?')}`  ",
            f"Visible frames: lap 1 = {meta.get('visible_frames_lap1', '?')}, "
            f"lap 2 = {meta.get('visible_frames_lap2', '?')}  ",
            f"[Open contact sheet]({contact})",
            "",
        ])
        if timeline is not None:
            lines.extend([
                "There are exactly two physical target locations; visible frames "
                "within one lap are different camera views of the same pose:",
                "",
            ])
            for lap, location in ((1, "A"), (2, "B")):
                item = timeline[lap]
                xyz = item["target_xyz"]
                xyz_text = (
                    f"({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) m"
                    if xyz is not None else "unknown"
                )
                visible_text = ", ".join(
                    f"Frame {frame:06d}" for frame in item["visible_frames"]
                ) or "none"
                lines.append(
                    f"- Lap {lap}, location {location}, `{item['surface']}`: "
                    f"Frames {item['first_frame']:06d}–{item['last_frame']:06d}; "
                    f"target xyz `{xyz_text}`; visible in {visible_text}."
                )
            lines.extend(["", "| Metric | Value |", "|---|---:|"])
        else:
            lines.extend(["| Metric | Value |", "|---|---:|"])
        for name, label, kind in METRICS:
            if name in episode["summary"]:
                lines.append(
                    f"| {label} | {_format_metric(episode['summary'].get(name), kind)} |"
                )
        lines.extend([
            "",
            "| Track | Frame | Query | Top-1 | Recall@K | Stale Top-1 | Rerank |",
            "|---|---:|---|---|---:|---:|---|",
        ])
        for query_result in episode["queries"]:
            candidates = query_result.get("candidates", [])
            top1 = candidates[0]["observation_id"] if candidates else "none"
            recall = query_result["metrics"].get("recall_at_k")
            recall_text = "N/A" if recall is None else ("hit" if recall == 1.0 else "miss")
            query_text = str(query_result["text"]).replace("|", "\\|")
            diagnostics = query_result.get("adapter_diagnostics") or {}
            if diagnostics.get("rerank_attempted") is True:
                rerank_text = (
                    "valid" if diagnostics.get("rerank_valid") is True else "FAISS fallback"
                )
            else:
                rerank_text = "—"
            lines.append(
                f"| {TRACK_LABELS.get(query_result['track'], query_result['track'])} "
                f"| {query_result['checkpoint_frame']} | {query_text} | `{top1}` "
                f"| {recall_text} | {int(query_result['metrics'].get('stale_top1', 0.0))} "
                f"| {rerank_text} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_reports(
    report: dict,
    manifest: dict,
    dataset_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    _augment_legacy_metrics(report)
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()
    html_path = output_dir / "report.html"
    markdown_path = output_dir / "summary.md"
    html_path.write_text(
        _html_document(report, manifest, dataset_dir, output_dir), encoding="utf-8"
    )
    markdown_path.write_text(
        _markdown_document(report, manifest, dataset_dir, output_dir), encoding="utf-8"
    )
    return html_path, markdown_path


def render_existing(results_path: Path, dataset_override: Path | None = None) -> tuple[Path, Path]:
    results_path = Path(results_path).resolve()
    with open(results_path, encoding="utf-8") as file:
        report = json.load(file)
    dataset_dir = (
        Path(dataset_override).resolve()
        if dataset_override is not None
        else Path(report["dataset"]).resolve()
    )
    with open(dataset_dir / "benchmark_manifest.json", encoding="utf-8") as file:
        manifest = json.load(file)
    return write_reports(report, manifest, dataset_dir, results_path.parent)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a readable spatial-memory report")
    parser.add_argument("--results", required=True)
    parser.add_argument("--dataset", default=None, help="override dataset path in results.json")
    args = parser.parse_args()
    html_path, markdown_path = render_existing(
        Path(args.results), Path(args.dataset) if args.dataset else None
    )
    print(f"[memory-report] HTML: {html_path}")
    print(f"[memory-report] Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
