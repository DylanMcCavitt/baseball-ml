from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from importlib import resources
from pathlib import Path
from typing import Any

PREGAME_FIELDS = (
    "pregame_game_pk",
    "pregame_game_date",
    "pregame_game_time_utc",
    "pregame_as_of",
    "pregame_pitcher_mlb_id",
    "pregame_pitcher_name",
    "pregame_pitcher_hand",
    "pregame_team_id",
    "pregame_team_abbr",
    "pregame_opponent_team_id",
    "pregame_opponent_team_abbr",
    "pregame_home_away",
)

START_METADATA_FIELDS = (
    "start_id",
    "start_started_game",
    "start_source_updated_at",
)

COMPLETION_FIELDS = (
    "completion_game_status",
    "completion_completed_at",
)

TARGET_FIELDS = (
    "target_final_strikeouts",
    "target_batters_faced",
    "target_pitches",
    "target_outs_recorded",
    "target_innings_pitched",
    "target_available_at",
)

REQUIRED_TARGET_SOURCE_FIELDS = (
    "final_strikeouts",
    "batters_faced",
    "pitches",
    "outs_recorded",
)


@dataclass(frozen=True)
class PitcherTargetRows:
    accepted_rows: list[dict[str, Any]]
    rejected_rows: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass(frozen=True)
class PitcherTargetBuild:
    report_root: Path
    manifest_path: Path
    target_table_path: Path
    audit_path: Path
    rejected_rows_path: Path
    visual_paths: tuple[Path, ...]


def default_start_sample_path() -> Path:
    path = resources.files("mlb_props_lab.resources").joinpath("pitcher_start_targets_sample.csv")
    return Path(str(path))


def default_identity_sample_path() -> Path:
    path = resources.files("mlb_props_lab.resources").joinpath("pitcher_identities_sample.csv")
    return Path(str(path))


def build_pitcher_start_target_artifacts(
    issue: str,
    output_dir: str | Path = "artifacts/reports",
    run_id: str | None = None,
    starts_path: str | Path | None = None,
    identities_path: str | Path | None = None,
) -> PitcherTargetBuild:
    starts_input = Path(starts_path) if starts_path else default_start_sample_path()
    identities_input = Path(identities_path) if identities_path else default_identity_sample_path()
    starts = _read_csv(starts_input)
    identities = _read_csv(identities_input)
    target_rows = build_pitcher_start_targets(starts, identities)

    run = run_id or _make_run_id()
    report_root = Path(output_dir) / issue / run
    visuals_root = report_root / "visuals"
    report_root.mkdir(parents=True, exist_ok=True)
    visuals_root.mkdir(parents=True, exist_ok=True)

    target_table_path = report_root / "pitcher_start_targets.csv"
    audit_path = report_root / "pitcher_start_target_audit.csv"
    rejected_rows_path = report_root / "pitcher_start_rejected_rows.csv"
    _write_csv(target_table_path, target_rows.accepted_rows)
    _write_csv(audit_path, [_audit_row(target_rows.summary)])
    _write_csv(rejected_rows_path, target_rows.rejected_rows)

    visual_paths = (visuals_root / "pitcher_start_target_quality.svg",)
    _write_target_quality_svg(visual_paths[0], target_rows.summary)

    generated_at = datetime.now(UTC).isoformat()
    manifest_path = report_root / "pitcher_start_target_manifest.json"
    manifest = {
        "schema_version": "2026-04-28.1",
        "report_type": "pitcher_start_targets",
        "issue": issue,
        "run_id": run,
        "generated_at": generated_at,
        "start_input": str(starts_input),
        "identity_input": str(identities_input),
        "target_table": target_table_path.name,
        "audit": audit_path.name,
        "rejected_rows": rejected_rows_path.name,
        "visuals": [path.relative_to(report_root).as_posix() for path in visual_paths],
        "summary": target_rows.summary,
        "field_groups": {
            "pregame_fields": list(PREGAME_FIELDS),
            "start_metadata_fields": list(START_METADATA_FIELDS),
            "completion_fields": list(COMPLETION_FIELDS),
            "target_fields": list(TARGET_FIELDS),
        },
        "timestamp_rules": [
            "Pregame fields are valid only at pregame_as_of, which must be before game_time_utc.",
            (
                "Target and completion fields are post-game labels and must not feed "
                "feature generation."
            ),
            "target_available_at is set to completion_completed_at for accepted fixture rows.",
        ],
        "limitations": [
            "Fixture-backed sample only; live MLB Stats API ingestion is out of scope.",
            (
                "Rows with missing outcomes, duplicate pitcher starts, unresolved identities, "
                "or invalid timestamps are rejected and reported."
            ),
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return PitcherTargetBuild(
        report_root=report_root,
        manifest_path=manifest_path,
        target_table_path=target_table_path,
        audit_path=audit_path,
        rejected_rows_path=rejected_rows_path,
        visual_paths=visual_paths,
    )


def build_pitcher_start_targets(
    start_rows: list[dict[str, str]],
    identity_rows: list[dict[str, str]],
) -> PitcherTargetRows:
    identities = {
        _clean(row.get("pitcher_mlb_id", "")): row
        for row in identity_rows
        if _clean(row.get("pitcher_mlb_id", ""))
    }
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    seen_starts: set[tuple[str, str]] = set()

    for index, source_row in enumerate(start_rows, start=2):
        row = {key: _clean(value) for key, value in source_row.items()}
        reasons: list[str] = []
        game_pk = row.get("game_pk", "")
        pitcher_id = row.get("pitcher_mlb_id", "")
        start_key = (game_pk, pitcher_id)

        if not game_pk or not pitcher_id:
            reasons.append("missing_start_key")
        elif start_key in seen_starts:
            reasons.append("duplicate_pitcher_start")
        else:
            seen_starts.add(start_key)

        identity = identities.get(pitcher_id)
        if not identity:
            reasons.append("unresolved_pitcher_identity")

        game_time = _parse_datetime(row.get("game_time_utc", ""))
        pregame_as_of = _parse_datetime(row.get("pregame_as_of", ""))
        completed_at = _parse_datetime(row.get("game_completed_at", ""))
        if not game_time:
            reasons.append("missing_game_time_utc")
        if not pregame_as_of:
            reasons.append("missing_pregame_as_of")
        if not completed_at:
            reasons.append("missing_completion_at")
        if game_time and pregame_as_of and pregame_as_of >= game_time:
            reasons.append("timestamp_invalid_pregame_as_of")
        if game_time and completed_at and completed_at < game_time:
            reasons.append("timestamp_invalid_completion_at")

        if _bool(row.get("started_game", "")) is not True:
            reasons.append("not_pitcher_start")

        missing_targets = [
            field for field in REQUIRED_TARGET_SOURCE_FIELDS if not row.get(field, "")
        ]
        if missing_targets:
            reasons.append("missing_target_fields")

        if reasons:
            for reason in reasons:
                reason_counts[reason] += 1
            rejected_rows.append(_rejected_row(index, row, reasons, missing_targets))
            continue

        assert identity is not None
        assert completed_at is not None
        accepted_rows.append(_accepted_row(row, identity, completed_at))

    summary = {
        "raw_row_count": len(start_rows),
        "accepted_row_count": len(accepted_rows),
        "rejected_row_count": len(rejected_rows),
        "missing_target_count": reason_counts["missing_target_fields"],
        "duplicate_start_count": reason_counts["duplicate_pitcher_start"],
        "unresolved_identity_count": reason_counts["unresolved_pitcher_identity"],
        "timestamp_rejection_count": (
            reason_counts["timestamp_invalid_pregame_as_of"]
            + reason_counts["timestamp_invalid_completion_at"]
            + reason_counts["missing_game_time_utc"]
            + reason_counts["missing_pregame_as_of"]
            + reason_counts["missing_completion_at"]
        ),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
    }
    return PitcherTargetRows(
        accepted_rows=accepted_rows,
        rejected_rows=rejected_rows,
        summary=summary,
    )


def _accepted_row(
    row: dict[str, str],
    identity: dict[str, str],
    completed_at: datetime,
) -> dict[str, Any]:
    game_pk = row["game_pk"]
    pitcher_id = row["pitcher_mlb_id"]
    hand = row.get("p_throws") or _clean(identity.get("pitcher_hand", ""))
    pitcher_name = _clean(identity.get("pitcher_name", "")) or row.get("pitcher_name", "")
    outs = _int(row["outs_recorded"])
    return {
        "start_id": f"{game_pk}:{pitcher_id}",
        "pregame_game_pk": game_pk,
        "pregame_game_date": row.get("game_date", ""),
        "pregame_game_time_utc": _iso_z(_parse_datetime(row["game_time_utc"])),
        "pregame_as_of": _iso_z(_parse_datetime(row["pregame_as_of"])),
        "pregame_pitcher_mlb_id": pitcher_id,
        "pregame_pitcher_name": pitcher_name,
        "pregame_pitcher_hand": hand,
        "pregame_team_id": row.get("team_id", ""),
        "pregame_team_abbr": row.get("team_abbr", ""),
        "pregame_opponent_team_id": row.get("opponent_team_id", ""),
        "pregame_opponent_team_abbr": row.get("opponent_team_abbr", ""),
        "pregame_home_away": "home" if _bool(row.get("is_home", "")) else "away",
        "start_started_game": "yes",
        "start_source_updated_at": row.get("source_updated_at", ""),
        "completion_game_status": row.get("game_status", ""),
        "completion_completed_at": _iso_z(completed_at),
        "target_final_strikeouts": _int(row["final_strikeouts"]),
        "target_batters_faced": _int(row["batters_faced"]),
        "target_pitches": _int(row["pitches"]),
        "target_outs_recorded": outs,
        "target_innings_pitched": _innings_label(outs),
        "target_available_at": _iso_z(completed_at),
    }


def _rejected_row(
    source_row_number: int,
    row: dict[str, str],
    reasons: list[str],
    missing_targets: list[str],
) -> dict[str, Any]:
    return {
        "source_row_number": source_row_number,
        "game_pk": row.get("game_pk", ""),
        "pitcher_mlb_id": row.get("pitcher_mlb_id", ""),
        "pitcher_name": row.get("pitcher_name", ""),
        "reasons": "|".join(reasons),
        "missing_target_fields": "|".join(missing_targets),
    }


def _audit_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_row_count": summary["raw_row_count"],
        "accepted_row_count": summary["accepted_row_count"],
        "rejected_row_count": summary["rejected_row_count"],
        "missing_target_count": summary["missing_target_count"],
        "duplicate_start_count": summary["duplicate_start_count"],
        "unresolved_identity_count": summary["unresolved_identity_count"],
        "timestamp_rejection_count": summary["timestamp_rejection_count"],
        "rejection_reason_counts": summary["rejection_reason_counts"],
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _parse_datetime(value: str) -> datetime | None:
    cleaned = _clean(value)
    if not cleaned:
        return None
    parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _iso_z(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _bool(value: str) -> bool | None:
    normalized = _clean(value).lower()
    if normalized in {"1", "true", "yes", "y", "home"}:
        return True
    if normalized in {"0", "false", "no", "n", "away"}:
        return False
    return None


def _int(value: str) -> int:
    return int(_clean(value))


def _innings_label(outs: int) -> str:
    return f"{outs // 3}.{outs % 3}"


def _clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _csv_value(value: Any) -> str | int | float:
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return value


def _make_run_id(now: datetime | None = None) -> str:
    current = now or datetime.now(UTC)
    return current.strftime("%Y%m%dT%H%M%SZ")


def _write_target_quality_svg(path: Path, summary: dict[str, Any]) -> None:
    bars = [
        ("Accepted", summary["accepted_row_count"], "#2364aa"),
        ("Missing targets", summary["missing_target_count"], "#c44536"),
        ("Duplicate starts", summary["duplicate_start_count"], "#f2a541"),
        ("Rejected rows", summary["rejected_row_count"], "#6c757d"),
    ]
    width = 900
    height = 250
    max_value = max((value for _, value, _ in bars), default=1)
    rows = []
    y = 68
    for label, value, color in bars:
        bar_width = int(560 * value / max(max_value, 1))
        rows.append(f'<text x="32" y="{y + 16}" class="label">{escape(label)}</text>')
        rows.append(f'<rect x="220" y="{y}" width="560" height="22" class="track"/>')
        rows.append(
            f'<rect x="220" y="{y}" width="{bar_width}" height="22" fill="{color}"/>'
        )
        rows.append(f'<text x="800" y="{y + 17}" class="count">{value}</text>')
        y += 38

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg"
  width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 22px system-ui, sans-serif; fill: #17202a; }}
    .label {{ font: 13px system-ui, sans-serif; fill: #243447; }}
    .count {{ font: 12px system-ui, sans-serif; fill: #243447; }}
    .track {{ fill: #edf1f4; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="32" y="34" class="title">Pitcher Start Target Quality</text>
  {"".join(rows)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")
