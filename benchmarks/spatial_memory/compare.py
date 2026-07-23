#!/usr/bin/env python3
"""Compare several benchmark ``results.json`` files on one frozen dataset."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path


METHOD_LABELS = {
    "latest_only": "Latest-only",
    "embodied_agent": "SigLIP + FAISS",
    "embodied_agent_recency": "SigLIP + FAISS + recency",
    "embodied_agent_vlm": "SigLIP + FAISS + Gemini",
    "vlmaps": "VLMaps (LSeg 3D map)",
}

COMPARISON_METRICS = [
    ("static_recall_at_k", "Static Recall@K", "rate", "higher"),
    ("static_mrr", "Static MRR", "rate", "higher"),
    ("static_location_error_m_top1", "Static location error Top-1", "meters", "lower"),
    ("static_location_success_at_0_5m", "Static location success @ 0.5 m", "rate", "higher"),
    ("live_current_recall_at_k", "Current-location Recall@K", "rate", "higher"),
    ("live_latest_visible_frame_recall_at_k", "Latest-frame Recall@K", "rate", "higher"),
    ("live_latest_visible_frame_top1_accuracy", "Latest-frame Top-1", "rate", "higher"),
    ("live_current_top1_accuracy", "Current Top-1", "rate", "higher"),
    ("live_stale_top1_rate", "Stale Top-1 rate", "rate", "lower"),
    ("live_stale_fraction_at_k", "Stale fraction@K", "rate", "lower"),
    ("live_location_error_m_top1", "Current location error Top-1", "meters", "lower"),
    ("live_location_success_at_0_5m", "Current location success @ 0.5 m", "rate", "higher"),
    ("live_stale_location_top1_rate", "Stale-location Top-1 rate", "rate", "lower"),
    ("location_update_lag_frames_at_0_5m", "Location update lag @ 0.5 m", "frames", "lower"),
    ("update_lag_frames_at_k", "Update lag@K", "frames", "lower"),
    ("history_recall_at_k", "History Recall@K", "rate", "higher"),
    ("history_mrr", "History MRR", "rate", "higher"),
    ("history_location_error_m_top1", "History location error Top-1", "meters", "lower"),
    ("history_location_success_at_0_5m", "History location success @ 0.5 m", "rate", "higher"),
    ("absent_top1_score", "Absent Top-1 raw score*", "score", "lower"),
    ("absent_top1_confidence", "Absent Top-1 confidence", "rate", "lower"),
    ("absent_false_positive_rate_at_0_5", "Absent false-positive rate @ 0.5", "rate", "lower"),
    ("vlm_valid_response_rate", "VLM valid-response rate", "rate", "higher"),
    ("vlm_fallback_rate", "VLM fallback rate", "rate", "lower"),
    ("ingest_latency_ms_p50", "Ingest latency p50", "ms", "lower"),
    ("query_latency_ms_p50", "Query latency p50", "ms", "lower"),
]

RUN_METADATA_FIELDS = [
    (("wall_time_seconds",), "Wall time", "seconds"),
    (("resources", "process_peak_rss_bytes"), "Process peak RAM", "bytes"),
    (("resources", "storage", "adapter_state_bytes"), "Adapter state", "bytes"),
    (("adapter", "vlm", "logical_calls"), "VLM logical calls", "integer"),
    (("adapter", "vlm", "api_attempts"), "VLM API attempts", "integer"),
    (("adapter", "vlm", "api_errors"), "VLM API errors", "integer"),
    (("adapter", "vlm", "total_tokens"), "VLM total tokens", "integer"),
    (("adapter", "vlm", "estimated_standard_cost_usd"), "Estimated standard cost", "usd"),
]


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as file:
        report = json.load(file)
    report["_results_path"] = str(path.resolve())
    return report


def _episode_signature(report: dict) -> list[tuple]:
    return [
        (
            episode["episode_id"],
            tuple(
                (
                    query["query_id"],
                    query["track"],
                    query["text"],
                    int(query["checkpoint_frame"]),
                    tuple(query.get("relevant_observation_ids", [])),
                    tuple(query.get("stale_observation_ids", [])),
                )
                for query in episode["queries"]
            ),
        )
        for episode in report["episodes"]
    ]


def validate_comparable(reports: list[dict]) -> None:
    if len(reports) < 2:
        raise ValueError("comparison requires at least two result files")
    reference = reports[0]
    signature = _episode_signature(reference)
    for report in reports[1:]:
        if report.get("benchmark_id") != reference.get("benchmark_id"):
            raise ValueError("results use different benchmark contracts")
        if int(report.get("top_k", -1)) != int(reference.get("top_k", -1)):
            raise ValueError("results use different Top-K values")
        if _episode_signature(report) != signature:
            raise ValueError(
                "results do not contain the same episodes, queries, and oracle labels"
            )


def _metric_stats(report: dict, metric: str, kind: str) -> dict | None:
    values = [
        float(episode["summary"][metric])
        for episode in report["episodes"]
        if episode.get("summary", {}).get(metric) is not None
    ]
    if not values:
        return None
    mean = statistics.fmean(values)
    low = high = None
    if len(values) > 1:
        margin = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
        low, high = mean - margin, mean + margin
        if kind == "rate":
            low, high = max(0.0, low), min(1.0, high)
        elif kind in {"frames", "ms", "meters"}:
            low = max(0.0, low)
    return {"mean": mean, "count": len(values), "ci95_low": low, "ci95_high": high}


def _format_number(value: float, kind: str) -> str:
    if kind == "rate":
        return f"{value * 100.0:.1f}%"
    if kind == "frames":
        return f"{value:.2f}"
    if kind == "ms":
        return f"{value:.1f} ms"
    if kind == "meters":
        return f"{value:.3f} m"
    return f"{value:.4f}"


def _format_stats(stats: dict | None, kind: str, include_ci: bool = True) -> str:
    if stats is None:
        return "N/A"
    text = _format_number(float(stats["mean"]), kind)
    if include_ci and stats.get("ci95_low") is not None:
        text += (
            " ["
            + _format_number(float(stats["ci95_low"]), kind)
            + ", "
            + _format_number(float(stats["ci95_high"]), kind)
            + "]"
        )
    return text


def _metadata_value(metadata: dict, path: tuple[str, ...]):
    current = metadata
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _format_run_value(value, kind: str) -> str:
    if value is None:
        return "N/A"
    if kind == "seconds":
        return f"{float(value):.1f} s"
    if kind == "bytes":
        size = float(value)
        for unit in ("B", "KiB", "MiB", "GiB"):
            if size < 1024.0 or unit == "GiB":
                return f"{size:.1f} {unit}"
            size /= 1024.0
    if kind == "integer":
        return f"{int(value):,}"
    if kind == "usd":
        return f"${float(value):.4f}"
    return str(value)


def _method_key(report: dict) -> str:
    return str(report.get("adapter_spec") or report.get("adapter"))


def _display_label(report: dict) -> str:
    key = _method_key(report)
    return METHOD_LABELS.get(key, str(report.get("adapter", key)))


def _select_latest_for_adapters(dataset_dir: Path, adapters: list[str]) -> list[Path]:
    candidates: dict[str, list[tuple[str, Path]]] = {adapter: [] for adapter in adapters}
    for path in dataset_dir.glob("reports/*/*/results.json"):
        try:
            report = _load(path)
        except (OSError, ValueError, KeyError):
            continue
        spec = _method_key(report)
        if spec in candidates:
            candidates[spec].append((str(report.get("created_at", "")), path))
    missing = [adapter for adapter, items in candidates.items() if not items]
    if missing:
        raise FileNotFoundError(
            "no result found for adapter(s): " + ", ".join(missing)
        )
    return [max(candidates[adapter])[1] for adapter in adapters]


def _relative_path(path: Path, start: Path) -> str:
    return os.path.relpath(path.resolve(), start=start.resolve()).replace(os.sep, "/")


def build_comparison(
    reports: list[dict],
    dataset_dir: Path,
    labels: list[str] | None = None,
) -> dict:
    validate_comparable(reports)
    if labels is not None and len(labels) != len(reports):
        raise ValueError("custom labels must match the number of result files")
    methods = []
    for index, report in enumerate(reports):
        metrics = {
            name: _metric_stats(report, name, kind)
            for name, _label, kind, _direction in COMPARISON_METRICS
        }
        methods.append({
            "label": labels[index] if labels is not None else _display_label(report),
            "adapter": report.get("adapter"),
            "adapter_spec": report.get("adapter_spec"),
            "adapter_kwargs": report.get("adapter_kwargs", {}),
            "created_at": report.get("created_at"),
            "results_path": report["_results_path"],
            "metrics": metrics,
            "run_metadata": report.get("run_metadata", {}),
        })
    return {
        "schema_version": 1,
        "benchmark_id": reports[0]["benchmark_id"],
        "dataset": str(dataset_dir.resolve()),
        "top_k": int(reports[0]["top_k"]),
        "episode_count": len(reports[0]["episodes"]),
        "episode_ids": [episode["episode_id"] for episode in reports[0]["episodes"]],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "methods": methods,
    }


def _write_markdown(comparison: dict, output_dir: Path) -> Path:
    path = output_dir / "comparison.md"
    methods = comparison["methods"]
    lines = [
        "# Spatial-memory method comparison",
        "",
        f"- Dataset: `{Path(comparison['dataset']).name}`",
        f"- Episodes: {comparison['episode_count']}",
        f"- Top-K: {comparison['top_k']}",
        "- Values are mean [95% CI] across episodes.",
        "",
        "| Metric | " + " | ".join(method["label"] for method in methods) + " |",
        "|---|" + "---:|" * len(methods),
    ]
    for name, label, kind, direction in COMPARISON_METRICS:
        cells = [
            _format_stats(method["metrics"].get(name), kind) for method in methods
        ]
        lines.append(f"| {label} ({direction}) | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "\* Absent Top-1 raw scores use method-specific scales (for example cosine "
        "similarity, recency score, or VLM confidence) and must not be compared across "
        "different score types.",
        "",
        "## Run metadata",
        "",
        "| Field | " + " | ".join(method["label"] for method in methods) + " |",
        "|---|" + "---:|" * len(methods),
    ])
    for path_parts, label, kind in RUN_METADATA_FIELDS:
        cells = [
            _format_run_value(
                _metadata_value(method["run_metadata"], path_parts), kind
            )
            for method in methods
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## Runs",
        "",
    ])
    for method in methods:
        result_path = _relative_path(Path(method["results_path"]), output_dir)
        lines.append(
            f"- **{method['label']}**: [`results.json`]({result_path}); "
            f"adapter `{method['adapter_spec']}`; kwargs `{json.dumps(method['adapter_kwargs'], sort_keys=True)}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_html(comparison: dict, output_dir: Path) -> Path:
    path = output_dir / "comparison.html"
    methods = comparison["methods"]
    header = "".join(f"<th>{html.escape(method['label'])}</th>" for method in methods)
    rows = []
    for name, label, kind, direction in COMPARISON_METRICS:
        cells = "".join(
            f"<td>{html.escape(_format_stats(method['metrics'].get(name), kind))}</td>"
            for method in methods
        )
        rows.append(
            f"<tr><th>{html.escape(label)}<small>{html.escape(direction)} is better</small></th>"
            f"{cells}</tr>"
        )
    run_rows = []
    for path_parts, label, kind in RUN_METADATA_FIELDS:
        cells = "".join(
            f"<td>{html.escape(_format_run_value(_metadata_value(method['run_metadata'], path_parts), kind))}</td>"
            for method in methods
        )
        run_rows.append(f"<tr><th>{html.escape(label)}</th>{cells}</tr>")
    run_cards = []
    for method in methods:
        result_path = _relative_path(Path(method["results_path"]), output_dir)
        run_cards.append(
            '<article class="run"><strong>' + html.escape(method["label"]) + "</strong>"
            f'<a href="{html.escape(result_path, quote=True)}">results.json</a>'
            f'<code>{html.escape(str(method["adapter_spec"]))}</code></article>'
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spatial-memory method comparison</title>
<style>
:root {{ color-scheme:light; font-family:Inter,ui-sans-serif,system-ui,sans-serif;
  color:#17212b; background:#eef2f6; }} body {{ margin:0; }} main {{ max-width:1400px;
  margin:auto; padding:28px; }} h1 {{ margin-bottom:6px; }} .subtitle {{ color:#66717d; }}
.table-wrap {{ overflow:auto; background:white; border:1px solid #d9e0e7;
  border-radius:12px; }} table {{ border-collapse:collapse; width:100%; min-width:850px; }}
th,td {{ border-bottom:1px solid #e4e9ee; padding:10px 12px; text-align:right; }}
th:first-child {{ text-align:left; position:sticky; left:0; background:white; }}
thead th {{ background:#f7f9fb; }} small {{ display:block; color:#66717d;
  font-weight:400; }} .note {{ padding:12px 15px; border:1px solid #f0d58a;
  background:#fff8e7; border-radius:10px; }} .runs {{ display:grid;
  grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:10px; }} .run {{
  display:flex; flex-direction:column; gap:7px; padding:14px; background:white;
  border:1px solid #d9e0e7; border-radius:10px; }} code {{ color:#344054; }}
</style></head><body><main>
<h1>Spatial-memory method comparison</h1>
<p class="subtitle">{comparison['episode_count']} identical episodes · Top-{comparison['top_k']} · mean [95% CI]</p>
<div class="table-wrap"><table><thead><tr><th>Metric</th>{header}</tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p class="note"><strong>* Absent score warning:</strong> raw absent scores use method-specific
scales (cosine similarity, recency, or VLM confidence) and are not comparable across
different score types.</p>
<h2>Run metadata</h2>
<div class="table-wrap"><table><thead><tr><th>Field</th>{header}</tr></thead>
<tbody>{''.join(run_rows)}</tbody></table></div>
<h2>Runs</h2><div class="runs">{''.join(run_cards)}</div>
</main></body></html>"""
    path.write_text(document, encoding="utf-8")
    return path


def compare(
    dataset_dir: Path,
    result_paths: list[Path],
    output_dir: Path | None = None,
    labels: list[str] | None = None,
) -> Path:
    dataset_dir = Path(dataset_dir).resolve()
    reports = [_load(Path(path).resolve()) for path in result_paths]
    comparison = build_comparison(reports, dataset_dir, labels=labels)
    if output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = dataset_dir / "comparisons" / timestamp
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"comparison output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    with open(output_dir / "comparison.json", "w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)
        file.write("\n")
    markdown = _write_markdown(comparison, output_dir)
    html_path = _write_html(comparison, output_dir)
    print(f"[memory-compare] HTML: {html_path}")
    print(f"[memory-compare] Markdown: {markdown}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare spatial-memory benchmark runs")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", action="append", default=[])
    parser.add_argument("--adapter", action="append", default=[])
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="optional display label repeated once per result",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if bool(args.results) == bool(args.adapter):
        raise SystemExit("provide either repeated --results or repeated --adapter")
    dataset_dir = Path(args.dataset).resolve()
    result_paths = (
        [Path(path) for path in args.results]
        if args.results
        else _select_latest_for_adapters(dataset_dir, args.adapter)
    )
    if args.label and len(args.label) != len(result_paths):
        raise SystemExit("repeat --label exactly once per selected result")
    compare(
        dataset_dir=dataset_dir,
        result_paths=result_paths,
        output_dir=Path(args.output) if args.output else None,
        labels=args.label or None,
    )


if __name__ == "__main__":
    main()
