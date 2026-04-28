from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

from mlb_props_lab.feature_registry import (
    family_summary,
    features_by_family,
    load_registry,
    validate_registry,
)


def slugify(value: str) -> str:
    safe = []
    for char in value.lower():
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        elif char in {" ", "/", ":"}:
            safe.append("-")
    return "".join(safe).strip("-") or "run"


def make_run_id(now: datetime | None = None) -> str:
    current = now or datetime.now(UTC)
    return current.strftime("%Y%m%dT%H%M%SZ")


def generate_feature_report(
    issue: str,
    output_dir: str | Path = "artifacts/reports",
    run_id: str | None = None,
    registry_path: str | Path | None = None,
) -> Path:
    registry = load_registry(registry_path)
    validation = validate_registry(registry)
    if not validation.ok:
        joined = "\n".join(validation.errors)
        raise ValueError(f"feature registry is invalid:\n{joined}")

    run = run_id or make_run_id()
    report_root = Path(output_dir) / issue / run
    visuals_root = report_root / "visuals"
    visuals_root.mkdir(parents=True, exist_ok=True)

    summary = family_summary(registry)
    grouped = features_by_family(registry)
    visuals = []
    statcast_manifest = _latest_statcast_manifest(Path(output_dir) / issue)

    overview_visual = visuals_root / "feature_family_coverage.svg"
    _write_family_coverage_svg(overview_visual, summary)
    visuals.append(overview_visual.relative_to(report_root).as_posix())

    for family, features in grouped.items():
        visual_path = visuals_root / f"{family}.svg"
        _write_family_detail_svg(visual_path, family, features)
        visuals.append(visual_path.relative_to(report_root).as_posix())

    if statcast_manifest:
        statcast_root = Path(statcast_manifest["_manifest_path"]).parent
        for visual in statcast_manifest.get("visuals", []):
            visual_path = statcast_root / visual
            visuals.append(_relative_path(visual_path, report_root))

    report_md = report_root / "report.md"
    report_md.write_text(
        _render_markdown_report(registry, summary, visuals, statcast_manifest),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "2026-04-28.1",
        "report_type": "feature_registry",
        "issue": issue,
        "run_id": run,
        "generated_at": datetime.now(UTC).isoformat(),
        "registry_schema_version": registry["schema_version"],
        "feature_count": validation.feature_count,
        "family_count": validation.family_count,
        "required_feature_count": validation.required_feature_count,
        "families": summary,
        "visuals": visuals,
        "report": "report.md",
        "sources": registry["sources"],
    }
    if statcast_manifest:
        manifest["materialized_feature_run"] = _statcast_manifest_summary(
            statcast_manifest,
            report_root,
        )
    (report_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_root


def _render_markdown_report(
    registry: dict[str, Any],
    summary: dict[str, dict[str, int]],
    visuals: list[str],
    statcast_manifest: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# Source-Backed Feature Registry Report",
        "",
        f"Registry schema: `{registry['schema_version']}`",
        f"Feature count: `{len(registry['features'])}`",
        "",
        "## Visuals",
        "",
    ]
    lines.extend(f"- `{visual}`" for visual in visuals)
    lines.extend(["", "## Family Summary", ""])
    lines.append("| Family | V1 required | V1 optional | Later | Total |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for family, counts in summary.items():
        lines.append(
            f"| {family} | {counts['v1_required']} | {counts['v1_optional']} | "
            f"{counts['later']} | {counts['total']} |"
        )

    lines.extend(["", "## Registered Features", ""])
    for feature in registry["features"]:
        lines.extend(
            [
                f"### {feature['id']}",
                "",
                f"- Family: `{feature['family']}`",
                f"- Status: `{feature['status']}`",
                f"- Formula: {feature['formula']}",
                f"- Lookback: {feature['lookback_window']}",
                f"- Timestamp cutoff: {feature['timestamp_cutoff']}",
                f"- Missing policy: {feature['missing_policy']}",
                f"- Leakage risk: {feature['leakage_risk']}",
                f"- Required visual: {feature['required_visual']}",
                "",
            ]
        )
    if statcast_manifest:
        statcast_root = Path(statcast_manifest["_manifest_path"]).parent
        feature_matrix = _relative_path(
            statcast_root / statcast_manifest["feature_matrix"],
            statcast_root,
        )
        coverage = _relative_path(statcast_root / statcast_manifest["coverage"], statcast_root)
        skipped_rows = _relative_path(
            statcast_root / statcast_manifest["skipped_rows"],
            statcast_root,
        )
        lines.extend(
            [
                "",
                "## Materialized Statcast Feature Evidence",
                "",
                f"Run: `{statcast_manifest['run_id']}`",
                f"Targets: `{statcast_manifest['target_count']}`",
                f"Feature matrix: `{feature_matrix}`",
                f"Coverage: `{coverage}`",
                f"Skipped rows: `{skipped_rows}`",
                "",
                "Materialized registered feature IDs:",
                "",
            ]
        )
        lines.extend(
            f"- `{feature_id}`" for feature_id in statcast_manifest["materialized_feature_ids"]
        )
        if statcast_manifest.get("gap_feature_ids"):
            lines.extend(["", "Reported source gaps:", ""])
            lines.extend(f"- `{feature_id}`" for feature_id in statcast_manifest["gap_feature_ids"])
        if statcast_manifest.get("limitations"):
            lines.extend(["", "Limitations:", ""])
            lines.extend(f"- {limitation}" for limitation in statcast_manifest["limitations"])
    return "\n".join(lines).rstrip() + "\n"


def _write_family_coverage_svg(path: Path, summary: dict[str, dict[str, int]]) -> None:
    row_height = 34
    width = 920
    height = 70 + row_height * len(summary)
    max_total = max((counts["total"] for counts in summary.values()), default=1)
    rows = []
    y = 52
    for family, counts in summary.items():
        bar_width = int(520 * counts["total"] / max_total)
        required_width = int(bar_width * counts["v1_required"] / max(counts["total"], 1))
        optional_width = int(bar_width * counts["v1_optional"] / max(counts["total"], 1))
        later_width = max(bar_width - required_width - optional_width, 0)
        rows.append(f'<text x="24" y="{y + 18}" class="label">{escape(family)}</text>')
        rows.append(
            f'<rect x="310" y="{y}" width="{required_width}" height="20" '
            'class="required"/>'
        )
        rows.append(
            f'<rect x="{310 + required_width}" y="{y}" width="{optional_width}" '
            'height="20" class="optional"/>'
        )
        rows.append(
            f'<rect x="{310 + required_width + optional_width}" y="{y}" width="{later_width}" '
            'height="20" class="later"/>'
        )
        rows.append(f'<text x="850" y="{y + 16}" class="count">{counts["total"]}</text>')
        y += row_height

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg"
  width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 22px system-ui, sans-serif; fill: #17202a; }}
    .label {{ font: 13px system-ui, sans-serif; fill: #243447; }}
    .count {{ font: 12px system-ui, sans-serif; fill: #243447; }}
    .required {{ fill: #2364aa; }}
    .optional {{ fill: #3da35d; }}
    .later {{ fill: #f2a541; }}
    .axis {{ stroke: #ccd6dd; stroke-width: 1; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="24" y="30" class="title">Feature Family Coverage</text>
  <line x1="310" y1="40" x2="830" y2="40" class="axis"/>
  {"".join(rows)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _write_family_detail_svg(path: Path, family: str, features: list[dict[str, Any]]) -> None:
    width = 920
    row_height = 28
    height = max(110, 72 + row_height * len(features))
    rows = []
    y = 56
    colors = {"v1_required": "#2364aa", "v1_optional": "#3da35d", "later": "#f2a541"}
    for feature in features:
        status = feature["status"]
        rows.append(f'<circle cx="32" cy="{y - 5}" r="6" fill="{colors[status]}"/>')
        rows.append(f'<text x="48" y="{y}" class="feature">{escape(feature["id"])}</text>')
        rows.append(f'<text x="360" y="{y}" class="status">{escape(status)}</text>')
        rows.append(
            f'<text x="520" y="{y}" class="visual">'
            f'{escape(feature["required_visual"])}</text>'
        )
        y += row_height

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg"
  width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 20px system-ui, sans-serif; fill: #17202a; }}
    .feature {{ font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; fill: #243447; }}
    .status {{ font: 12px system-ui, sans-serif; fill: #243447; }}
    .visual {{ font: 12px system-ui, sans-serif; fill: #566573; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="24" y="30" class="title">{escape(family)}</text>
  {"".join(rows)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _latest_statcast_manifest(issue_root: Path) -> dict[str, Any] | None:
    candidates = sorted(issue_root.glob("*/statcast_feature_manifest.json"))
    manifests = []
    for path in candidates:
        with path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)
        manifest["_manifest_path"] = path
        manifests.append(manifest)
    if not manifests:
        return None
    return max(manifests, key=lambda manifest: manifest.get("generated_at", ""))


def _statcast_manifest_summary(manifest: dict[str, Any], report_root: Path) -> dict[str, Any]:
    statcast_root = Path(manifest["_manifest_path"]).parent
    return {
        "run_id": manifest["run_id"],
        "manifest": _relative_path(Path(manifest["_manifest_path"]), report_root),
        "feature_matrix": _relative_path(statcast_root / manifest["feature_matrix"], report_root),
        "coverage": _relative_path(statcast_root / manifest["coverage"], report_root),
        "skipped_rows": _relative_path(statcast_root / manifest["skipped_rows"], report_root),
        "target_count": manifest["target_count"],
        "materialized_feature_ids": manifest["materialized_feature_ids"],
        "gap_feature_ids": manifest.get("gap_feature_ids", []),
        "limitations": manifest.get("limitations", []),
    }


def _relative_path(target: Path, base: Path) -> str:
    return os.path.relpath(target, start=base).replace(os.sep, "/")
