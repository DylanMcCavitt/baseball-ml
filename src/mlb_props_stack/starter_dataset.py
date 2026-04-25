"""Standalone starter-game strikeout dataset build for projection rebuilds."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from io import StringIO
import json
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode

from .ingest.mlb_stats_api import utc_now
from .ingest.statcast_features import StatcastSearchClient
from .ingest.statcast_ingest import (
    DEFAULT_MAX_FETCH_WORKERS,
    STATCAST_SEARCH_CSV_ENDPOINT,
    STRIKEOUT_EVENTS,
    _fetch_csv_texts_concurrently,
)
from .modeling import (
    StarterStrikeoutTrainingRow,
    _fetch_starter_outcome,
    _is_missing_outcome_error,
    _load_feature_rows_for_date,
    _requested_dates,
    _unique_timestamp_run_id,
)

DATASET_VERSION = "starter_game_strikeout_training_dataset_v1"
SHORT_START_PLATE_APPEARANCE_THRESHOLD = 12
DEFAULT_DIRECT_CHUNK_DAYS = 3
STATCAST_CSV_CAP_WARNING_THRESHOLD = 25000
REGULAR_SEASON_MONTHS = frozenset({3, 4, 5, 6, 7, 8, 9, 10})


@dataclass(frozen=True)
class StarterGameDatasetBuildResult:
    """Filesystem output summary for one starter-game dataset build."""

    start_date: date
    end_date: date
    run_id: str
    source_mode: str
    requested_date_count: int
    source_date_count: int
    row_count: int
    missing_target_count: int
    excluded_start_count: int
    season_count: int
    dataset_path: Path
    coverage_report_path: Path
    coverage_report_markdown_path: Path
    missing_targets_path: Path
    source_manifest_path: Path
    schema_drift_report_path: Path
    timestamp_policy_path: Path
    reproducibility_notes_path: Path


@dataclass(frozen=True)
class _DirectStatcastPitchRow:
    source_pull_id: str
    source_row_number: int
    source_url: str
    raw_path: Path
    captured_at: datetime
    game_date: str
    game_pk: int
    at_bat_number: int
    pitch_number: int
    inning: int | None
    inning_topbot: str | None
    pitcher_id: int
    pitcher_name: str | None
    batter_id: int | None
    events: str | None
    home_team_abbreviation: str | None
    away_team_abbreviation: str | None
    p_throws: str | None
    is_plate_appearance_final_pitch: bool
    is_strikeout_event: bool


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(_json_ready(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    if not rendered or rendered.lower() == "null":
        return None
    return rendered


def _optional_int(value: Any) -> int | None:
    rendered = _optional_text(value)
    if rendered is None:
        return None
    return int(float(rendered))


def _season_environment(official_date: str) -> dict[str, Any]:
    season = int(official_date[:4])
    return {
        "season": season,
        "month": official_date[:7],
        "pitch_clock_era": "pitch_clock" if season >= 2023 else "pre_pitch_clock",
        "league_k_environment": "modern_high_k_environment_2019_plus",
    }


def _role_fields(*, plate_appearance_count: int) -> dict[str, Any]:
    if plate_appearance_count < SHORT_START_PLATE_APPEARANCE_THRESHOLD:
        edge_case = "short_start_review"
    else:
        edge_case = "standard_starter"
    return {
        "starter_role_status": "probable_starter_feature_row_with_same_game_outcome",
        "starter_role_edge_case": edge_case,
        "starter_plate_appearance_count": plate_appearance_count,
        "starter_role_review_threshold_pa": SHORT_START_PLATE_APPEARANCE_THRESHOLD,
    }


def _dataset_row(
    *,
    row: StarterStrikeoutTrainingRow,
    outcome: Any,
) -> dict[str, Any]:
    payload = asdict(row)
    payload["starter_strikeouts"] = outcome.starter_strikeouts
    payload.update(_season_environment(row.official_date))
    payload.update(_role_fields(plate_appearance_count=outcome.plate_appearance_count))
    payload["outcome_id"] = outcome.outcome_id
    payload["outcome_captured_at"] = outcome.captured_at
    payload["outcome_source_url"] = outcome.source_url
    payload["outcome_raw_path"] = outcome.raw_path
    payload["outcome_pitch_row_count"] = outcome.pitch_row_count
    payload["source_references"] = {
        "pitcher_feature_row_id": row.pitcher_feature_row_id,
        "lineup_feature_row_id": row.lineup_feature_row_id,
        "game_context_feature_row_id": row.game_context_feature_row_id,
        "lineup_snapshot_id": row.lineup_snapshot_id,
        "features_as_of": row.features_as_of,
        "outcome_captured_at": outcome.captured_at,
        "outcome_raw_path": outcome.raw_path,
    }
    payload["timestamp_policy_status"] = (
        "ok" if row.features_as_of <= outcome.captured_at else "violation"
    )
    return payload


def _direct_dataset_row(
    *,
    starter: _DirectStatcastPitchRow,
    team_abbreviation: str,
    opponent_team_abbreviation: str,
    home_away: str,
    pitcher_rows: list[_DirectStatcastPitchRow],
) -> dict[str, Any]:
    final_pitch_rows = [row for row in pitcher_rows if row.is_plate_appearance_final_pitch]
    starter_strikeouts = sum(1 for row in final_pitch_rows if row.is_strikeout_event)
    payload: dict[str, Any] = {
        "training_row_id": (
            f"starter-training:{starter.game_date}:{starter.game_pk}:{starter.pitcher_id}"
        ),
        "official_date": starter.game_date,
        "game_pk": starter.game_pk,
        "pitcher_id": starter.pitcher_id,
        "pitcher_name": starter.pitcher_name,
        "team_abbreviation": team_abbreviation,
        "opponent_team_abbreviation": opponent_team_abbreviation,
        "home_away": home_away,
        "pitcher_hand": starter.p_throws,
        "starter_strikeouts": starter_strikeouts,
        "features_as_of": None,
        "pregame_reference_status": "not_applicable_target_foundation",
        "target_source_status": "postgame_statcast_pitch_log_target_only",
        "starter_role_status": "inferred_from_first_pitch_for_fielding_team",
        "starter_plate_appearance_count": len(final_pitch_rows),
        "starter_role_review_threshold_pa": SHORT_START_PLATE_APPEARANCE_THRESHOLD,
        "starter_role_edge_case": (
            "short_start_review"
            if len(final_pitch_rows) < SHORT_START_PLATE_APPEARANCE_THRESHOLD
            else "standard_starter"
        ),
        "outcome_id": f"starter-outcome:{starter.game_date}:{starter.game_pk}:{starter.pitcher_id}",
        "outcome_captured_at": starter.captured_at,
        "outcome_source_url": starter.source_url,
        "outcome_raw_path": starter.raw_path,
        "outcome_pitch_row_count": len(pitcher_rows),
        "source_references": {
            "source_pull_id": starter.source_pull_id,
            "source_url": starter.source_url,
            "raw_path": starter.raw_path,
            "captured_at": starter.captured_at,
            "starter_inference": "first_pitch_for_fielding_team_in_game",
            "target_label": "count_final_pitch_strikeout_events_for_inferred_starter",
        },
        "timestamp_policy_status": "target_only_no_pregame_features",
    }
    payload.update(_season_environment(starter.game_date))
    return payload


def _missing_target_row(
    *,
    row: StarterStrikeoutTrainingRow,
    reason: str,
) -> dict[str, Any]:
    return {
        "official_date": row.official_date,
        "season": int(row.official_date[:4]),
        "game_pk": row.game_pk,
        "pitcher_id": row.pitcher_id,
        "pitcher_name": row.pitcher_name,
        "team_abbreviation": row.team_abbreviation,
        "opponent_team_abbreviation": row.opponent_team_abbreviation,
        "home_away": row.home_away,
        "features_as_of": row.features_as_of,
        "reason": reason,
    }


def _direct_missing_target_row(
    *,
    starter: _DirectStatcastPitchRow,
    team_abbreviation: str,
    opponent_team_abbreviation: str,
    home_away: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "official_date": starter.game_date,
        "season": int(starter.game_date[:4]),
        "game_pk": starter.game_pk,
        "pitcher_id": starter.pitcher_id,
        "pitcher_name": starter.pitcher_name,
        "team_abbreviation": team_abbreviation,
        "opponent_team_abbreviation": opponent_team_abbreviation,
        "home_away": home_away,
        "features_as_of": None,
        "reason": reason,
    }


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))


def _count_by_team(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row["team_abbreviation"])] += 1
    return dict(sorted(counts.items()))


def _min_max_iso(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = sorted(row[key] for row in rows if row.get(key) is not None)
    return {
        "min": values[0] if values else None,
        "max": values[-1] if values else None,
    }


def _schema_drift_report(rows: list[dict[str, Any]], *, requested_dates: list[date]) -> dict[str, Any]:
    field_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for key, value in row.items():
            if value is not None:
                field_counts[key] += 1
    row_count = len(rows)
    fields = []
    for field in sorted({key for row in rows for key in row}):
        non_null_count = field_counts[field]
        fields.append(
            {
                "field": field,
                "non_null_count": non_null_count,
                "coverage": round(non_null_count / row_count, 6) if row_count else 0.0,
            }
        )
    return {
        "schema_report_version": "starter_game_dataset_schema_drift_v1",
        "requested_date_count": len(requested_dates),
        "row_count": row_count,
        "fields": fields,
    }


def _date_chunks(start_date: date, end_date: date, *, chunk_days: int) -> list[tuple[date, date]]:
    if chunk_days < 1:
        raise ValueError("chunk_days must be at least 1")
    chunks: list[tuple[date, date]] = []
    cursor = start_date
    while cursor <= end_date:
        if cursor.month not in REGULAR_SEASON_MONTHS:
            cursor += timedelta(days=1)
            continue
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end_date)
        while chunk_end.month not in REGULAR_SEASON_MONTHS and chunk_end >= cursor:
            chunk_end -= timedelta(days=1)
        if chunk_end >= cursor:
            chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _build_statcast_pitch_log_csv_url(*, start_date: date, end_date: date) -> str:
    query = [
        ("all", "true"),
        ("hfPT", ""),
        ("hfAB", ""),
        ("hfBBT", ""),
        ("hfPR", ""),
        ("hfZ", ""),
        ("stadium", ""),
        ("hfBBL", ""),
        ("hfNewZones", ""),
        ("hfGT", "R|"),
        ("hfC", ""),
        ("hfSea", ""),
        ("hfSit", ""),
        ("player_type", "pitcher"),
        ("hfOuts", ""),
        ("opponent", ""),
        ("pitcher_throws", ""),
        ("batter_stands", ""),
        ("hfSA", ""),
        ("game_date_gt", start_date.isoformat()),
        ("game_date_lt", end_date.isoformat()),
        ("team", ""),
        ("position", ""),
        ("hfRO", ""),
        ("home_road", ""),
        ("hfFlag", ""),
        ("metric_1", ""),
        ("hfInn", ""),
        ("min_pitches", "0"),
        ("min_results", "0"),
        ("group_by", "name"),
        ("sort_col", "pitches"),
        ("player_event_sort", "api_p_release_speed"),
        ("sort_order", "desc"),
        ("min_pas", "0"),
        ("type", "details"),
    ]
    return f"{STATCAST_SEARCH_CSV_ENDPOINT}?{urlencode(query, doseq=True)}"


def _parse_direct_statcast_pitch_rows(
    csv_text: str,
    *,
    source_pull_id: str,
    source_url: str,
    raw_path: Path,
    captured_at: datetime,
) -> list[_DirectStatcastPitchRow]:
    reader = csv.DictReader(StringIO(csv_text.lstrip("\ufeff")))
    if not reader.fieldnames or "game_date" not in reader.fieldnames:
        return []

    rows: list[_DirectStatcastPitchRow] = []
    for row_number, row in enumerate(reader, start=2):
        game_date = _optional_text(row.get("game_date"))
        game_pk = _optional_int(row.get("game_pk"))
        at_bat_number = _optional_int(row.get("at_bat_number"))
        pitch_number = _optional_int(row.get("pitch_number"))
        pitcher_id = _optional_int(row.get("pitcher"))
        if (
            game_date is None
            or game_pk is None
            or at_bat_number is None
            or pitch_number is None
            or pitcher_id is None
        ):
            continue
        events = _optional_text(row.get("events"))
        events_key = events.lower() if events else None
        rows.append(
            _DirectStatcastPitchRow(
                source_pull_id=source_pull_id,
                source_row_number=row_number,
                source_url=source_url,
                raw_path=raw_path,
                captured_at=captured_at,
                game_date=game_date,
                game_pk=game_pk,
                at_bat_number=at_bat_number,
                pitch_number=pitch_number,
                inning=_optional_int(row.get("inning")),
                inning_topbot=_optional_text(row.get("inning_topbot")),
                pitcher_id=pitcher_id,
                pitcher_name=_optional_text(row.get("player_name")),
                batter_id=_optional_int(row.get("batter")),
                events=events,
                home_team_abbreviation=_optional_text(row.get("home_team")),
                away_team_abbreviation=_optional_text(row.get("away_team")),
                p_throws=_optional_text(row.get("p_throws")),
                is_plate_appearance_final_pitch=events_key is not None,
                is_strikeout_event=events_key in STRIKEOUT_EVENTS,
            )
        )
    return rows


def _fielding_context(row: _DirectStatcastPitchRow) -> tuple[str, str, str] | None:
    if row.inning_topbot == "Top":
        if row.home_team_abbreviation is None or row.away_team_abbreviation is None:
            return None
        return row.home_team_abbreviation, row.away_team_abbreviation, "home"
    if row.inning_topbot == "Bot":
        if row.home_team_abbreviation is None or row.away_team_abbreviation is None:
            return None
        return row.away_team_abbreviation, row.home_team_abbreviation, "away"
    return None


def _pitch_sort_key(row: _DirectStatcastPitchRow) -> tuple[int, int, int]:
    inning = row.inning if row.inning is not None else 99
    return inning, row.at_bat_number, row.pitch_number


def _direct_rows_from_pitch_rows(
    pitch_rows: list[_DirectStatcastPitchRow],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    rows_by_game: dict[tuple[str, int], list[_DirectStatcastPitchRow]] = defaultdict(list)
    source_dates: set[str] = set()
    for row in pitch_rows:
        rows_by_game[(row.game_date, row.game_pk)].append(row)
        source_dates.add(row.game_date)

    dataset_rows: list[dict[str, Any]] = []
    missing_targets: list[dict[str, Any]] = []
    for game_rows in rows_by_game.values():
        rows_by_team: dict[tuple[str, str, str], list[_DirectStatcastPitchRow]] = defaultdict(list)
        for row in game_rows:
            context = _fielding_context(row)
            if context is None:
                continue
            rows_by_team[context].append(row)

        for (team_abbreviation, opponent_team_abbreviation, home_away), team_rows in rows_by_team.items():
            starter = min(team_rows, key=_pitch_sort_key)
            starter_rows = [row for row in game_rows if row.pitcher_id == starter.pitcher_id]
            final_pitch_rows = [
                row for row in starter_rows if row.is_plate_appearance_final_pitch
            ]
            if not final_pitch_rows:
                missing_targets.append(
                    _direct_missing_target_row(
                        starter=starter,
                        team_abbreviation=team_abbreviation,
                        opponent_team_abbreviation=opponent_team_abbreviation,
                        home_away=home_away,
                        reason="starter_has_no_final_pitch_rows",
                    )
                )
                continue
            dataset_rows.append(
                _direct_dataset_row(
                    starter=starter,
                    team_abbreviation=team_abbreviation,
                    opponent_team_abbreviation=opponent_team_abbreviation,
                    home_away=home_away,
                    pitcher_rows=starter_rows,
                )
            )
    return dataset_rows, missing_targets, source_dates


def _build_direct_statcast_dataset_rows(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path,
    normalized_root: Path,
    run_id: str,
    client: StatcastSearchClient,
    now: Callable[[], datetime],
    chunk_days: int,
    max_fetch_workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], set[date]]:
    chunks = _date_chunks(start_date, end_date, chunk_days=chunk_days)
    chunk_specs = [
        (
            chunk_start,
            chunk_end,
            _build_statcast_pitch_log_csv_url(start_date=chunk_start, end_date=chunk_end),
            now().astimezone(UTC),
        )
        for chunk_start, chunk_end in chunks
    ]
    dataset_rows: list[dict[str, Any]] = []
    missing_targets: list[dict[str, Any]] = []
    source_manifest_rows: list[dict[str, Any]] = []
    source_dates: set[date] = set()
    raw_root = (
        output_dir
        / "raw"
        / "statcast_search_starter_dataset"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
        / f"run={run_id}"
    )
    batch_size = max(1, max_fetch_workers * 4)
    for batch_start in range(0, len(chunk_specs), batch_size):
        batch_specs = chunk_specs[batch_start : batch_start + batch_size]
        csv_texts = _fetch_csv_texts_concurrently(
            client=client,
            source_urls=[spec[2] for spec in batch_specs],
            max_workers=max_fetch_workers,
        )
        for (chunk_start, chunk_end, source_url, captured_at), csv_text in zip(
            batch_specs,
            csv_texts,
            strict=True,
        ):
            chunk_id = f"{chunk_start.isoformat()}_{chunk_end.isoformat()}"
            raw_path = raw_root / f"chunk={chunk_id}.csv"
            _write_text(raw_path, csv_text)
            source_pull_id = f"starter-dataset-statcast:{chunk_id}:{run_id}"
            pitch_rows = _parse_direct_statcast_pitch_rows(
                csv_text,
                source_pull_id=source_pull_id,
                source_url=source_url,
                raw_path=raw_path,
                captured_at=captured_at,
            )
            chunk_dataset_rows, chunk_missing_targets, chunk_source_dates = _direct_rows_from_pitch_rows(
                pitch_rows
            )
            dataset_rows.extend(chunk_dataset_rows)
            missing_targets.extend(chunk_missing_targets)
            source_dates.update(date.fromisoformat(value) for value in chunk_source_dates)
            source_manifest_rows.append(
                {
                    "source_pull_id": source_pull_id,
                    "chunk_start_date": chunk_start,
                    "chunk_end_date": chunk_end,
                    "captured_at": captured_at,
                    "source_url": source_url,
                    "raw_path": raw_path,
                    "pitch_row_count": len(pitch_rows),
                    "dataset_row_count": len(chunk_dataset_rows),
                    "missing_target_count": len(chunk_missing_targets),
                    "cap_warning": len(pitch_rows) >= STATCAST_CSV_CAP_WARNING_THRESHOLD,
                }
            )
    return dataset_rows, missing_targets, source_manifest_rows, source_dates


def _coverage_report(
    *,
    start_date: date,
    end_date: date,
    run_id: str,
    source_mode: str,
    requested_dates: list[date],
    source_dates: list[date],
    rows: list[dict[str, Any]],
    missing_targets: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    timestamp_violations: list[dict[str, Any]],
    source_manifest_rows: list[dict[str, Any]],
    dataset_path: Path,
    source_manifest_path: Path,
    schema_drift_report_path: Path,
) -> dict[str, Any]:
    seasons = sorted({int(row["season"]) for row in rows})
    source_date_set = set(source_dates)
    cap_warning_chunks = [
        row for row in source_manifest_rows if row.get("cap_warning")
    ]
    return {
        "coverage_report_version": "starter_game_strikeout_coverage_v1",
        "dataset_version": DATASET_VERSION,
        "run_id": run_id,
        "source_mode": source_mode,
        "date_window": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "requested_date_count": len(requested_dates),
            "source_date_count": len(source_dates),
            "missing_source_dates": [
                day.isoformat() for day in requested_dates if day not in source_date_set
            ],
        },
        "coverage_status": {
            "season_count": len(seasons),
            "seasons": seasons,
            "preferred_5_to_7_seasons_achieved": len(seasons) >= 5,
            "minimum_4_full_plus_current_shape_achieved": len(seasons) >= 5,
            "note": (
                "This report counts seasons represented in the built artifact. "
                "Full-season completeness must be judged from row counts by month and missing targets."
            ),
        },
        "row_counts": {
            "source_starter_rows": len(rows) + len(missing_targets) + len(duplicate_rows),
            "dataset_rows": len(rows),
            "missing_targets": len(missing_targets),
            "excluded_starts": len(missing_targets) + len(duplicate_rows),
            "duplicate_source_rows": len(duplicate_rows),
            "timestamp_violations": len(timestamp_violations),
        },
        "source_chunks": {
            "chunk_count": len(source_manifest_rows),
            "cap_warning_count": len(cap_warning_chunks),
            "cap_warning_threshold_rows": STATCAST_CSV_CAP_WARNING_THRESHOLD,
        },
        "row_counts_by_season": _count_by(rows, "season"),
        "row_counts_by_month": _count_by(rows, "month"),
        "row_counts_by_team": _count_by_team(rows),
        "missing_targets_by_season": _count_by(missing_targets, "season"),
        "starter_role_edge_cases": _count_by(rows, "starter_role_edge_case"),
        "source_freshness": {
            "features_as_of": _min_max_iso(rows, "features_as_of"),
            "outcome_captured_at": _min_max_iso(rows, "outcome_captured_at"),
        },
        "timestamp_policy": {
            "status": "ok" if not timestamp_violations else "violations",
            "features_as_of_must_be_lte_outcome_captured_at": True,
            "violations": timestamp_violations,
        },
        "artifacts": {
            "dataset_path": dataset_path,
            "source_manifest_path": source_manifest_path,
            "schema_drift_report_path": schema_drift_report_path,
        },
    }


def _render_coverage_markdown(report: dict[str, Any]) -> str:
    rows = report["row_counts"]
    status = report["coverage_status"]
    lines = [
        "# Starter Game Strikeout Dataset Coverage",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Source mode: `{report['source_mode']}`",
        (
            "- Date window: "
            f"`{report['date_window']['start_date']}` -> `{report['date_window']['end_date']}`"
        ),
        f"- Dataset rows: `{rows['dataset_rows']}`",
        f"- Missing targets: `{rows['missing_targets']}`",
        f"- Excluded starts: `{rows['excluded_starts']}`",
        f"- Timestamp violations: `{rows['timestamp_violations']}`",
        f"- Seasons represented: `{status['season_count']}`",
        f"- Preferred 5-7 season coverage achieved: `{status['preferred_5_to_7_seasons_achieved']}`",
        "",
        "## Rows By Season",
        "",
    ]
    for season, count in report["row_counts_by_season"].items():
        lines.append(f"- `{season}`: `{count}`")
    lines.extend(["", "## Rows By Team", ""])
    for team, count in report["row_counts_by_team"].items():
        lines.append(f"- `{team}`: `{count}`")
    lines.extend(
        [
            "",
            "## Timestamp Policy",
            "",
            "Same-game Statcast pitch-log pulls define target labels only. "
            "When pregame feature references are present, `features_as_of` must be less than "
            "or equal to `outcome_captured_at`; outcome fields must not become model features.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_timestamp_policy() -> str:
    return "\n".join(
        [
            "# Starter Game Dataset Timestamp Policy",
            "",
            "- One row represents one MLB starter-game keyed by official date, game id, and pitcher id.",
            "- Direct Statcast pitch-log mode derives target labels and starter identity only; it does not create model features.",
            "- Feature-run mode keeps pregame feature artifacts and their original `features_as_of` timestamp.",
            "- Same-game Statcast outcome pulls define only the target `starter_strikeouts`.",
            "- Outcome fields must not be fed back into pregame feature generation.",
            "- The dataset build records a timestamp violation when a non-null `features_as_of > outcome_captured_at`.",
            "",
        ]
    )


def _render_reproducibility_notes(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str,
    run_id: str,
    source_mode: str,
    chunk_days: int,
    max_fetch_workers: int,
) -> str:
    return "\n".join(
        [
            "# Starter Game Dataset Reproducibility",
            "",
            f"- Run ID: `{run_id}`",
            "- Command:",
            "",
            "```bash",
            "uv run python -m mlb_props_stack build-starter-strikeout-dataset "
            f"--start-date {start_date.isoformat()} --end-date {end_date.isoformat()} "
            f"--output-dir {output_dir} "
            f"--chunk-days {chunk_days} --max-fetch-workers {max_fetch_workers}",
            "```",
            "",
            f"- Source mode: `{source_mode}`",
            f"- Direct Statcast chunk days: `{chunk_days}`",
            f"- Direct Statcast max fetch workers: `{max_fetch_workers}`",
            "- If normalized Statcast feature runs are present, the command uses them as pregame feature references.",
            "- If no feature runs are present, it falls back to direct regular-season Statcast pitch-log chunks and derives actual starters from the first pitcher used by each fielding team.",
            "- Missing source dates are recorded in the coverage report instead of silently treated as covered.",
            "- Missing same-game targets are recorded in `missing_targets.jsonl` and excluded from the dataset.",
            "",
        ]
    )


def build_starter_strikeout_dataset(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str = "data",
    client: StatcastSearchClient | None = None,
    now: Callable[[], datetime] = utc_now,
    chunk_days: int = DEFAULT_DIRECT_CHUNK_DAYS,
    max_fetch_workers: int = DEFAULT_MAX_FETCH_WORKERS,
) -> StarterGameDatasetBuildResult:
    """Build a standalone starter-game strikeout training dataset artifact."""
    if client is None:
        client = StatcastSearchClient()
    output_root = Path(output_dir)
    requested_dates = [
        day for day in _requested_dates(start_date, end_date)
        if day.month in REGULAR_SEASON_MONTHS
    ]
    run_root = (
        output_root
        / "normalized"
        / "starter_strikeout_training_dataset"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
    )
    run_id = _unique_timestamp_run_id(now().astimezone(UTC), run_root)
    normalized_root = run_root / f"run={run_id}"
    dataset_path = normalized_root / "starter_game_training_dataset.jsonl"
    coverage_report_path = normalized_root / "coverage_report.json"
    coverage_report_markdown_path = normalized_root / "coverage_report.md"
    missing_targets_path = normalized_root / "missing_targets.jsonl"
    source_manifest_path = normalized_root / "source_manifest.jsonl"
    schema_drift_report_path = normalized_root / "schema_drift_report.json"
    timestamp_policy_path = normalized_root / "timestamp_policy.md"
    reproducibility_notes_path = normalized_root / "reproducibility_notes.md"

    feature_dates: list[date] = []
    feature_rows: list[StarterStrikeoutTrainingRow] = []
    for target_date in requested_dates:
        date_root = output_root / "normalized" / "statcast_search" / f"date={target_date.isoformat()}"
        if not date_root.exists():
            continue
        loaded_rows = _load_feature_rows_for_date(target_date=target_date, output_dir=output_root)
        if loaded_rows:
            feature_dates.append(target_date)
            feature_rows.extend(loaded_rows)

    source_mode = "feature_runs" if feature_rows else "direct_statcast_pitch_log"
    rows: list[dict[str, Any]]
    missing_targets: list[dict[str, Any]]
    source_manifest_rows: list[dict[str, Any]]
    source_dates: set[date]
    duplicate_rows: list[dict[str, Any]] = []
    if feature_rows:
        rows = []
        missing_targets = []
        source_manifest_rows = []
        source_dates = set(feature_dates)
        seen_keys: set[tuple[str, int, int]] = set()
        for row in sorted(feature_rows, key=lambda item: (item.official_date, item.game_pk, item.pitcher_id)):
            key = (row.official_date, row.game_pk, row.pitcher_id)
            if key in seen_keys:
                duplicate_rows.append(_missing_target_row(row=row, reason="duplicate_feature_row"))
                continue
            seen_keys.add(key)
            try:
                outcome = _fetch_starter_outcome(
                    row=row,
                    output_dir=output_root,
                    client=client,
                    now=now,
                )
            except ValueError as error:
                if _is_missing_outcome_error(error):
                    missing_targets.append(_missing_target_row(row=row, reason="missing_same_game_statcast_outcome"))
                    continue
                raise
            rows.append(_dataset_row(row=row, outcome=outcome))
    else:
        rows, missing_targets, source_manifest_rows, source_dates = _build_direct_statcast_dataset_rows(
            start_date=start_date,
            end_date=end_date,
            output_dir=output_root,
            normalized_root=normalized_root,
            run_id=run_id,
            client=client,
            now=now,
            chunk_days=chunk_days,
            max_fetch_workers=max_fetch_workers,
        )

    deduped_rows: list[dict[str, Any]] = []
    seen_dataset_keys: set[tuple[str, int, int]] = set()
    for row in sorted(rows, key=lambda item: (item["official_date"], item["game_pk"], item["pitcher_id"])):
        key = (str(row["official_date"]), int(row["game_pk"]), int(row["pitcher_id"]))
        if key in seen_dataset_keys:
            duplicate_rows.append(
                {
                    "official_date": row["official_date"],
                    "season": row["season"],
                    "game_pk": row["game_pk"],
                    "pitcher_id": row["pitcher_id"],
                    "pitcher_name": row.get("pitcher_name"),
                    "team_abbreviation": row.get("team_abbreviation"),
                    "opponent_team_abbreviation": row.get("opponent_team_abbreviation"),
                    "home_away": row.get("home_away"),
                    "features_as_of": row.get("features_as_of"),
                    "reason": "duplicate_dataset_row",
                }
            )
            continue
        seen_dataset_keys.add(key)
        deduped_rows.append(row)
    rows = deduped_rows

    timestamp_violations = [
        {
            "training_row_id": row["training_row_id"],
            "official_date": row["official_date"],
            "game_pk": row["game_pk"],
            "pitcher_id": row["pitcher_id"],
            "features_as_of": row["features_as_of"],
            "outcome_captured_at": row["outcome_captured_at"],
        }
        for row in rows
        if row["timestamp_policy_status"] == "violation"
    ]

    schema_report = _schema_drift_report(rows, requested_dates=requested_dates)
    coverage = _coverage_report(
        start_date=start_date,
        end_date=end_date,
        run_id=run_id,
        source_mode=source_mode,
        requested_dates=requested_dates,
        source_dates=sorted(source_dates),
        rows=rows,
        missing_targets=missing_targets,
        duplicate_rows=duplicate_rows,
        timestamp_violations=timestamp_violations,
        source_manifest_rows=source_manifest_rows,
        dataset_path=dataset_path,
        source_manifest_path=source_manifest_path,
        schema_drift_report_path=schema_drift_report_path,
    )

    _write_jsonl(dataset_path, rows)
    _write_jsonl(missing_targets_path, missing_targets + duplicate_rows)
    _write_jsonl(source_manifest_path, source_manifest_rows)
    _write_json(schema_drift_report_path, schema_report)
    _write_json(coverage_report_path, coverage)
    _write_text(coverage_report_markdown_path, _render_coverage_markdown(coverage))
    _write_text(timestamp_policy_path, _render_timestamp_policy())
    _write_text(
        reproducibility_notes_path,
        _render_reproducibility_notes(
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            run_id=run_id,
            source_mode=source_mode,
            chunk_days=chunk_days,
            max_fetch_workers=max_fetch_workers,
        ),
    )

    seasons = {int(row["season"]) for row in rows}
    return StarterGameDatasetBuildResult(
        start_date=start_date,
        end_date=end_date,
        run_id=run_id,
        source_mode=source_mode,
        requested_date_count=len(requested_dates),
        source_date_count=len(source_dates),
        row_count=len(rows),
        missing_target_count=len(missing_targets),
        excluded_start_count=len(missing_targets) + len(duplicate_rows),
        season_count=len(seasons),
        dataset_path=dataset_path,
        coverage_report_path=coverage_report_path,
        coverage_report_markdown_path=coverage_report_markdown_path,
        missing_targets_path=missing_targets_path,
        source_manifest_path=source_manifest_path,
        schema_drift_report_path=schema_drift_report_path,
        timestamp_policy_path=timestamp_policy_path,
        reproducibility_notes_path=reproducibility_notes_path,
    )
