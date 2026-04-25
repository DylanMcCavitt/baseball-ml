"""Workload, leash, and role-context features for the projection rebuild."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .ingest.mlb_stats_api import utc_now
from .ingest.statcast_ingest import StatcastPitchRecord, normalize_statcast_csv_text
from .modeling import _unique_timestamp_run_id

FEATURE_SET_VERSION = "workload_leash_features_v1"
RECENT_FORM_DAYS = 15
REST_BUCKETS = (
    "no_prior_start",
    "short_rest",
    "standard_rest",
    "extra_rest",
    "long_layoff",
)


@dataclass(frozen=True)
class WorkloadLeashFeatureBuildResult:
    """Filesystem output summary for one workload/leash feature build."""

    start_date: date
    end_date: date
    run_id: str
    dataset_row_count: int
    feature_row_count: int
    pitch_row_count: int
    pitcher_count: int
    feature_path: Path
    feature_report_path: Path
    feature_report_markdown_path: Path
    reproducibility_notes_path: Path


@dataclass(frozen=True)
class _StartSummary:
    game_date: date
    game_pk: int
    pitcher_id: int
    team_abbreviation: str | None
    pitch_count: int
    batters_faced: int


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


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(_json_ready(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 6)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)


def _variance(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return round(sum((value - mean) ** 2 for value in values) / (len(values) - 1), 6)


def _correlation(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return None
    return round(numerator / (denom_x * denom_y), 6)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _latest_run_dir(root: Path) -> Path:
    candidates = sorted(path for path in root.glob("run=*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {root}")
    return candidates[-1]


def _resolve_dataset_run_dir(
    *,
    output_dir: Path,
    start_date: date,
    end_date: date,
    dataset_run_dir: Path | None,
) -> Path:
    if dataset_run_dir is not None:
        return dataset_run_dir
    root = (
        output_dir
        / "normalized"
        / "starter_strikeout_training_dataset"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
    )
    return _latest_run_dir(root)


def _load_pitch_records_from_manifest(
    source_manifest_path: Path, *, latest_needed_date: date
) -> list[StatcastPitchRecord]:
    records: list[StatcastPitchRecord] = []
    for index, manifest_row in enumerate(_load_jsonl(source_manifest_path), start=1):
        chunk_start = manifest_row.get("chunk_start_date")
        if chunk_start is not None and date.fromisoformat(str(chunk_start)) > latest_needed_date:
            continue
        raw_path_text = manifest_row.get("raw_path")
        if raw_path_text is None:
            continue
        raw_path = Path(str(raw_path_text))
        if not raw_path.is_absolute():
            raw_path = source_manifest_path.parent / raw_path
        if not raw_path.exists():
            continue
        pull_id = str(manifest_row.get("source_pull_id") or f"manifest-pull-{index}")
        records.extend(
            normalize_statcast_csv_text(
                raw_path.read_text(encoding="utf-8").lstrip("\ufeff"),
                pull_id=pull_id,
            )
        )
    return records


def _row_date(row: StatcastPitchRecord) -> date:
    return date.fromisoformat(row.game_date)


def _plate_appearance_count(rows: list[StatcastPitchRecord]) -> int:
    return sum(1 for row in rows if row.is_plate_appearance_final_pitch)


def _pitcher_start_index(rows: list[StatcastPitchRecord]) -> dict[int, list[_StartSummary]]:
    rows_by_game_team: dict[tuple[date, int, str], list[StatcastPitchRecord]] = defaultdict(list)
    for row in rows:
        if row.fielding_team_abbreviation is None:
            continue
        rows_by_game_team[(_row_date(row), row.game_pk, row.fielding_team_abbreviation)].append(row)

    grouped: dict[tuple[int, date, int], list[StatcastPitchRecord]] = defaultdict(list)
    for (game_date, game_pk, _team), team_rows in rows_by_game_team.items():
        ordered = sorted(team_rows, key=lambda row: (row.at_bat_number, row.pitch_number))
        if not ordered:
            continue
        starter_pitcher_id = ordered[0].pitcher_id
        for row in ordered:
            if row.pitcher_id == starter_pitcher_id:
                grouped[(starter_pitcher_id, game_date, game_pk)].append(row)
    indexed: dict[int, list[_StartSummary]] = defaultdict(list)
    for (pitcher_id, game_date, game_pk), start_rows in grouped.items():
        ordered = sorted(start_rows, key=lambda row: (row.at_bat_number, row.pitch_number))
        team = next((row.fielding_team_abbreviation for row in ordered if row.fielding_team_abbreviation), None)
        indexed[pitcher_id].append(
            _StartSummary(
                game_date=game_date,
                game_pk=game_pk,
                pitcher_id=pitcher_id,
                team_abbreviation=team,
                pitch_count=len(ordered),
                batters_faced=_plate_appearance_count(ordered),
            )
        )
    return {
        pitcher_id: sorted(starts, key=lambda row: (row.game_date, row.game_pk))
        for pitcher_id, starts in indexed.items()
    }


def _dataset_start_index(
    dataset_rows: list[dict[str, Any]],
    pitcher_starts: dict[int, list[_StartSummary]],
) -> dict[str, list[_StartSummary]]:
    by_key = {
        (summary.game_pk, summary.pitcher_id): summary
        for summaries in pitcher_starts.values()
        for summary in summaries
    }
    indexed: dict[str, list[_StartSummary]] = defaultdict(list)
    for row in dataset_rows:
        summary = by_key.get((int(row["game_pk"]), int(row["pitcher_id"])))
        if summary is None:
            continue
        team = str(row.get("team_abbreviation") or summary.team_abbreviation or "")
        if team:
            indexed[team].append(summary)
    return {
        team: sorted(starts, key=lambda row: (row.game_date, row.game_pk))
        for team, starts in indexed.items()
    }


def _prior_starts(starts: list[_StartSummary], *, target_date: date) -> list[_StartSummary]:
    return [start for start in starts if start.game_date < target_date]


def _recent_starts(
    starts: list[_StartSummary], *, target_date: date, days: int
) -> list[_StartSummary]:
    earliest = target_date - timedelta(days=days)
    return [start for start in starts if earliest <= start.game_date < target_date]


def _last_game_date(starts: list[_StartSummary]) -> date | None:
    if not starts:
        return None
    return max(start.game_date for start in starts)


def _rest_context(*, target_date: date, prior_starts: list[_StartSummary]) -> dict[str, Any]:
    last_game_date = _last_game_date(prior_starts)
    if last_game_date is None:
        rest_days = None
        bucket = "no_prior_start"
    else:
        rest_days = max(0, (target_date - last_game_date).days)
        if rest_days <= 3:
            bucket = "short_rest"
        elif rest_days <= 5:
            bucket = "standard_rest"
        elif rest_days <= 14:
            bucket = "extra_rest"
        else:
            bucket = "long_layoff"
    return {
        "last_prior_game_date": last_game_date,
        "rest_days_capped": None if rest_days is None else min(rest_days, 14),
        "rest_bucket": bucket,
        "short_rest_flag": bucket == "short_rest",
        "standard_rest_flag": bucket == "standard_rest",
        "extra_rest_flag": bucket == "extra_rest",
        "long_layoff_flag": bucket == "long_layoff",
        "unknown_or_no_prior_rest_flag": bucket == "no_prior_start",
        "long_layoff_unknown_flag": bucket == "long_layoff",
        "injury_context_status": (
            "unknown_no_prior_start"
            if bucket == "no_prior_start"
            else "unknown_long_layoff"
            if bucket == "long_layoff"
            else "not_source_backed"
        ),
        "il_return_flag": False,
        "rehab_return_flag": False,
        "role_change_source_status": "not_source_backed",
    }


def _distribution(starts: list[_StartSummary]) -> dict[str, Any]:
    pitch_counts = [float(start.pitch_count) for start in starts]
    batter_counts = [float(start.batters_faced) for start in starts]
    return {
        "start_count": len(starts),
        "pitch_count_mean": _mean(pitch_counts),
        "pitch_count_p25": _percentile(pitch_counts, 0.25),
        "pitch_count_p50": _percentile(pitch_counts, 0.50),
        "pitch_count_p75": _percentile(pitch_counts, 0.75),
        "batters_faced_mean": _mean(batter_counts),
        "batters_faced_p25": _percentile(batter_counts, 0.25),
        "batters_faced_p50": _percentile(batter_counts, 0.50),
        "batters_faced_p75": _percentile(batter_counts, 0.75),
        "reached_18_batters_rate": _safe_rate(
            sum(1 for start in starts if start.batters_faced >= 18), len(starts)
        ),
        "reached_22_batters_rate": _safe_rate(
            sum(1 for start in starts if start.batters_faced >= 22), len(starts)
        ),
        "reached_27_batters_rate": _safe_rate(
            sum(1 for start in starts if start.batters_faced >= 27), len(starts)
        ),
    }


def _blend_expected_opportunity(
    *,
    pitcher_recent: dict[str, Any],
    pitcher_season: dict[str, Any],
    team_season: dict[str, Any],
) -> tuple[float | None, float | None, str]:
    pitch_values = [
        float(value)
        for value in (
            pitcher_recent["pitch_count_mean"],
            pitcher_season["pitch_count_mean"],
            team_season["pitch_count_mean"],
        )
        if isinstance(value, (int, float))
    ]
    batter_values = [
        float(value)
        for value in (
            pitcher_recent["batters_faced_mean"],
            pitcher_season["batters_faced_mean"],
            team_season["batters_faced_mean"],
        )
        if isinstance(value, (int, float))
    ]
    source = "missing_prior_workload_history"
    if pitcher_recent["start_count"]:
        source = "pitcher_recent_plus_season_team_context"
    elif pitcher_season["start_count"]:
        source = "pitcher_season_plus_team_context"
    elif team_season["start_count"]:
        source = "team_prior_starter_context"
    return _mean(pitch_values), _mean(batter_values), source


def _role_context(prior_starts: list[_StartSummary]) -> dict[str, Any]:
    prior_short_starts = [
        start for start in prior_starts if start.batters_faced > 0 and start.batters_faced < 10
    ]
    opener_or_bulk = bool(prior_short_starts)
    source = "prior_starter_short_workload_pattern" if prior_short_starts else "no_source_evidence"
    return {
        "opener_or_bulk_role_flag": opener_or_bulk,
        "opener_or_bulk_role_source": source,
        "prior_short_start_count": len(prior_short_starts),
        "prior_short_start_rate": _safe_rate(len(prior_short_starts), len(prior_starts)),
    }


def _feature_row(
    *,
    starter_row: dict[str, Any],
    pitcher_starts: dict[int, list[_StartSummary]],
    team_starts: dict[str, list[_StartSummary]],
) -> dict[str, Any]:
    target_date = date.fromisoformat(str(starter_row["official_date"]))
    pitcher_id = int(starter_row["pitcher_id"])
    team = str(starter_row.get("team_abbreviation") or "")
    prior_pitcher_starts = _prior_starts(
        pitcher_starts.get(pitcher_id, []), target_date=target_date
    )
    season_pitcher_starts = [
        start for start in prior_pitcher_starts if start.game_date.year == target_date.year
    ]
    recent_pitcher_starts = _recent_starts(
        prior_pitcher_starts, target_date=target_date, days=RECENT_FORM_DAYS
    )
    team_prior_starts = _prior_starts(team_starts.get(team, []), target_date=target_date)
    season_team_starts = [
        start for start in team_prior_starts if start.game_date.year == target_date.year
    ]
    last3_pitcher_starts = sorted(
        prior_pitcher_starts, key=lambda row: (row.game_date, row.game_pk), reverse=True
    )[:3]
    career = _distribution(prior_pitcher_starts)
    season = _distribution(season_pitcher_starts)
    recent = _distribution(recent_pitcher_starts)
    last3 = _distribution(last3_pitcher_starts)
    team_season = _distribution(season_team_starts)
    expected_pitch_count, expected_batters_faced, expected_source = _blend_expected_opportunity(
        pitcher_recent=last3,
        pitcher_season=season,
        team_season=team_season,
    )
    payload: dict[str, Any] = {
        "feature_row_id": (
            "workload-leash-feature:"
            f"{starter_row['official_date']}:{starter_row['game_pk']}:{starter_row['pitcher_id']}"
        ),
        "training_row_id": starter_row["training_row_id"],
        "official_date": starter_row["official_date"],
        "season": int(starter_row["season"]),
        "game_pk": starter_row["game_pk"],
        "pitcher_id": pitcher_id,
        "pitcher_name": starter_row.get("pitcher_name"),
        "team_abbreviation": starter_row.get("team_abbreviation"),
        "opponent_team_abbreviation": starter_row.get("opponent_team_abbreviation"),
        "features_as_of": f"{target_date.isoformat()}T00:00:00Z",
        "feature_status": "ok" if prior_pitcher_starts else "missing_prior_workload_history",
        "leakage_policy_status": "ok_prior_games_only",
        "prior_start_count": career["start_count"],
        "season_start_count": season["start_count"],
        "recent_15d_start_count": recent["start_count"],
        "last_3_start_count": last3["start_count"],
        "recent_15d_pitch_count_mean": recent["pitch_count_mean"],
        "recent_15d_batters_faced_mean": recent["batters_faced_mean"],
        "last_3_starts_pitch_count_mean": last3["pitch_count_mean"],
        "last_3_starts_batters_faced_mean": last3["batters_faced_mean"],
        "season_pitch_count_mean": season["pitch_count_mean"],
        "season_pitch_count_p25": season["pitch_count_p25"],
        "season_pitch_count_p50": season["pitch_count_p50"],
        "season_pitch_count_p75": season["pitch_count_p75"],
        "season_batters_faced_mean": season["batters_faced_mean"],
        "season_batters_faced_p25": season["batters_faced_p25"],
        "season_batters_faced_p50": season["batters_faced_p50"],
        "season_batters_faced_p75": season["batters_faced_p75"],
        "season_reached_18_batters_rate": season["reached_18_batters_rate"],
        "season_reached_22_batters_rate": season["reached_22_batters_rate"],
        "season_reached_27_batters_rate": season["reached_27_batters_rate"],
        "career_pitch_count_mean": career["pitch_count_mean"],
        "career_batters_faced_mean": career["batters_faced_mean"],
        "career_reached_18_batters_rate": career["reached_18_batters_rate"],
        "career_reached_22_batters_rate": career["reached_22_batters_rate"],
        "career_reached_27_batters_rate": career["reached_27_batters_rate"],
        "team_prior_start_count": team_season["start_count"],
        "team_season_pitch_count_mean": team_season["pitch_count_mean"],
        "team_season_batters_faced_mean": team_season["batters_faced_mean"],
        "team_season_reached_22_batters_rate": team_season["reached_22_batters_rate"],
        "expected_leash_pitch_count": expected_pitch_count,
        "expected_leash_batters_faced": expected_batters_faced,
        "expected_opportunity_source": expected_source,
        "feature_group": "expected_opportunity",
        "feature_usage": "opportunity_volume_not_strikeout_skill",
    }
    payload.update(_rest_context(target_date=target_date, prior_starts=prior_pitcher_starts))
    payload.update(_role_context(prior_pitcher_starts))
    return payload


def _numeric_feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    excluded = {"season", "game_pk", "pitcher_id", "starter_strikeouts"}
    for row in rows:
        for key, value in row.items():
            if key in excluded or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                names.add(key)
    return sorted(names)


def _feature_report(
    *,
    start_date: date,
    end_date: date,
    run_id: str,
    dataset_run_dir: Path,
    pitch_row_count: int,
    rows: list[dict[str, Any]],
    targets_by_training_row_id: dict[str, int | float],
    feature_path: Path,
) -> dict[str, Any]:
    numeric_names = _numeric_feature_names(rows)
    field_coverage: list[dict[str, Any]] = []
    for name in sorted({key for row in rows for key in row}):
        non_null_count = sum(1 for row in rows if row.get(name) is not None)
        values = [
            float(row[name])
            for row in rows
            if name in numeric_names and isinstance(row.get(name), (int, float))
        ]
        field_coverage.append(
            {
                "field": name,
                "non_null_count": non_null_count,
                "missing_count": len(rows) - non_null_count,
                "coverage": round(non_null_count / len(rows), 6) if rows else 0.0,
                "variance": _variance(values),
            }
        )
    correlations_by_season: dict[str, list[dict[str, Any]]] = {}
    for season in sorted({int(row["season"]) for row in rows}):
        season_rows = [row for row in rows if int(row["season"]) == season]
        correlations: list[dict[str, Any]] = []
        for name in numeric_names:
            pairs = [
                (
                    float(row[name]),
                    float(targets_by_training_row_id[str(row["training_row_id"])]),
                )
                for row in season_rows
                if isinstance(row.get(name), (int, float))
                and isinstance(
                    targets_by_training_row_id.get(str(row["training_row_id"])),
                    (int, float),
                )
            ]
            coefficient = _correlation(pairs)
            if coefficient is not None:
                correlations.append(
                    {
                        "feature": name,
                        "correlation": coefficient,
                        "sample_size": len(pairs),
                    }
                )
        correlations_by_season[str(season)] = sorted(
            correlations,
            key=lambda row: abs(float(row["correlation"])),
            reverse=True,
        )[:10]
    rest_counts = Counter(str(row["rest_bucket"]) for row in rows)
    role_counts = Counter(str(row["opener_or_bulk_role_source"]) for row in rows)
    return {
        "feature_report_version": FEATURE_SET_VERSION,
        "run_id": run_id,
        "date_window": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        "dataset_run_dir": dataset_run_dir,
        "row_counts": {
            "dataset_rows": len(rows),
            "feature_rows": len(rows),
            "pitch_rows": pitch_row_count,
            "pitchers": len({row["pitcher_id"] for row in rows}),
            "long_layoff_rows": rest_counts["long_layoff"],
            "opener_or_bulk_rows": sum(1 for row in rows if row["opener_or_bulk_role_flag"]),
        },
        "rest_policy": {
            "raw_rest_days_primary_driver": False,
            "rest_days_capped_max": 14,
            "buckets": list(REST_BUCKETS),
            "long_layoff_has_unbounded_positive_numeric_feature": False,
        },
        "role_context": {
            "source_counts": dict(role_counts),
            "injury_return_source": "not_source_backed_in_v1",
            "long_layoff_is_not_injury_label": True,
        },
        "leakage_policy": {
            "status": "ok"
            if all(row["leakage_policy_status"] == "ok_prior_games_only" for row in rows)
            else "violations",
            "rule": "Only Statcast rows from games before the starter official date are workload inputs.",
        },
        "field_coverage": field_coverage,
        "top_correlations_by_season": correlations_by_season,
        "artifacts": {
            "workload_leash_features_path": feature_path,
        },
    }


def _render_feature_report_markdown(report: dict[str, Any]) -> str:
    rows = report["row_counts"]
    lines = [
        "# Workload And Leash Feature Report",
        "",
        f"- Run ID: `{report['run_id']}`",
        (
            "- Date window: "
            f"`{report['date_window']['start_date']}` -> `{report['date_window']['end_date']}`"
        ),
        f"- Feature rows: `{rows['feature_rows']}`",
        f"- Pitch rows read: `{rows['pitch_rows']}`",
        f"- Pitchers: `{rows['pitchers']}`",
        f"- Long layoff rows: `{rows['long_layoff_rows']}`",
        f"- Opener/bulk role rows: `{rows['opener_or_bulk_rows']}`",
        f"- Leakage policy: `{report['leakage_policy']['status']}`",
        f"- Raw rest-days primary driver: `{report['rest_policy']['raw_rest_days_primary_driver']}`",
        "",
        "## Top Correlations By Season",
        "",
    ]
    for season, correlations in report["top_correlations_by_season"].items():
        lines.append(f"### {season}")
        if not correlations:
            lines.append("- No non-constant numeric feature correlations available.")
            continue
        for row in correlations[:5]:
            lines.append(
                f"- `{row['feature']}`: `{row['correlation']}` "
                f"(n=`{row['sample_size']}`)"
            )
    lines.extend(
        [
            "",
            "## Feature Policy",
            "",
            "These fields describe expected opportunity volume: pitch counts, "
            "batters faced, team leash tendency, times-through-order thresholds, "
            "rest buckets, and source-backed role context. They are separate from "
            "pitcher strikeout skill and opponent matchup features.",
            "",
            "Raw continuous `rest_days` is not emitted. Long layoffs remain "
            "`unknown_long_layoff` unless a later source explicitly labels IL, "
            "rehab, or role-change context.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_reproducibility_notes(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str,
    dataset_run_dir: Path | None,
    run_id: str,
) -> str:
    command = (
        "uv run python -m mlb_props_stack build-workload-leash-features "
        f"--start-date {start_date.isoformat()} --end-date {end_date.isoformat()} "
        f"--output-dir {output_dir}"
    )
    if dataset_run_dir is not None:
        command += f" --dataset-run-dir {dataset_run_dir}"
    return "\n".join(
        [
            "# Workload Leash Feature Reproducibility",
            "",
            f"- Run ID: `{run_id}`",
            f"- Feature set version: `{FEATURE_SET_VERSION}`",
            "",
            "## Command",
            "",
            "```bash",
            command,
            "```",
            "",
            "## Timestamp Rule",
            "",
            "Pitch counts, batters faced, rest buckets, team leash tendency, and "
            "times-through-order features use only games before each starter "
            "official date. Same-game target strikeouts are used only for report "
            "correlations.",
            "",
        ]
    )


def build_workload_leash_features(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str = "data",
    dataset_run_dir: Path | str | None = None,
    now: Any = utc_now,
) -> WorkloadLeashFeatureBuildResult:
    """Build expected workload, leash, rest, and role-context features."""
    output_root = Path(output_dir)
    explicit_dataset_dir = Path(dataset_run_dir) if dataset_run_dir is not None else None
    resolved_dataset_run_dir = _resolve_dataset_run_dir(
        output_dir=output_root,
        start_date=start_date,
        end_date=end_date,
        dataset_run_dir=explicit_dataset_dir,
    )
    dataset_rows = _load_jsonl(resolved_dataset_run_dir / "starter_game_training_dataset.jsonl")
    if not dataset_rows:
        raise FileNotFoundError(
            "No starter-game dataset rows found at "
            f"{resolved_dataset_run_dir / 'starter_game_training_dataset.jsonl'}"
        )
    pitch_rows = _load_pitch_records_from_manifest(
        resolved_dataset_run_dir / "source_manifest.jsonl",
        latest_needed_date=end_date,
    )
    pitcher_starts = _pitcher_start_index(pitch_rows)
    team_starts = _dataset_start_index(dataset_rows, pitcher_starts)
    feature_rows: list[dict[str, Any]] = []
    targets_by_training_row_id = {
        str(row["training_row_id"]): row["starter_strikeouts"]
        for row in dataset_rows
        if "starter_strikeouts" in row
    }
    for starter_row in sorted(
        dataset_rows,
        key=lambda row: (row["official_date"], row["game_pk"], row["pitcher_id"]),
    ):
        target_date = date.fromisoformat(str(starter_row["official_date"]))
        if target_date < start_date or target_date > end_date:
            continue
        feature_rows.append(
            _feature_row(
                starter_row=starter_row,
                pitcher_starts=pitcher_starts,
                team_starts=team_starts,
            )
        )

    run_root = (
        output_root
        / "normalized"
        / "workload_leash_features"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
    )
    run_id = _unique_timestamp_run_id(now().astimezone(UTC), run_root)
    run_dir = run_root / f"run={run_id}"
    feature_path = run_dir / "workload_leash_features.jsonl"
    feature_report_path = run_dir / "feature_report.json"
    feature_report_markdown_path = run_dir / "feature_report.md"
    reproducibility_notes_path = run_dir / "reproducibility_notes.md"
    report = _feature_report(
        start_date=start_date,
        end_date=end_date,
        run_id=run_id,
        dataset_run_dir=resolved_dataset_run_dir,
        pitch_row_count=len(pitch_rows),
        rows=feature_rows,
        targets_by_training_row_id=targets_by_training_row_id,
        feature_path=feature_path,
    )
    _write_jsonl(feature_path, feature_rows)
    _write_json(feature_report_path, report)
    _write_text(feature_report_markdown_path, _render_feature_report_markdown(report))
    _write_text(
        reproducibility_notes_path,
        _render_reproducibility_notes(
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            dataset_run_dir=explicit_dataset_dir,
            run_id=run_id,
        ),
    )
    return WorkloadLeashFeatureBuildResult(
        start_date=start_date,
        end_date=end_date,
        run_id=run_id,
        dataset_row_count=len(dataset_rows),
        feature_row_count=len(feature_rows),
        pitch_row_count=len(pitch_rows),
        pitcher_count=len({row["pitcher_id"] for row in feature_rows}),
        feature_path=feature_path,
        feature_report_path=feature_report_path,
        feature_report_markdown_path=feature_report_markdown_path,
        reproducibility_notes_path=reproducibility_notes_path,
    )
