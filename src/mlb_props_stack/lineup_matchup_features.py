"""Batter-by-batter lineup matchup features for the projection rebuild."""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .ingest.mlb_stats_api import utc_now
from .ingest.statcast_ingest import (
    StatcastPitchRecord,
    normalize_statcast_csv_text,
)
from .modeling import _unique_timestamp_run_id

FEATURE_SET_VERSION = "lineup_matchup_features_v1"
PRIOR_PLATE_APPEARANCE_WEIGHT = 120
RECENT_FORM_DAYS = 15
MAX_PROJECTED_LINEUP_SIZE = 9


@dataclass(frozen=True)
class LineupMatchupFeatureBuildResult:
    """Filesystem output summary for one lineup matchup feature build."""

    start_date: date
    end_date: date
    run_id: str
    dataset_row_count: int
    feature_row_count: int
    batter_feature_row_count: int
    pitch_row_count: int
    feature_path: Path
    batter_feature_path: Path
    feature_report_path: Path
    feature_report_markdown_path: Path
    reproducibility_notes_path: Path


@dataclass(frozen=True)
class _DatedPitchRows:
    rows: tuple[StatcastPitchRecord, ...]
    dates: tuple[date, ...]


_EMPTY_DATED_PITCH_ROWS = _DatedPitchRows(rows=(), dates=())


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


def _weighted_mean(values: list[tuple[float, float]]) -> float | None:
    if not values:
        return None
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return None
    return round(sum(value * weight for value, weight in values) / total_weight, 6)


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


def _batting_order_for_game(
    rows: list[StatcastPitchRecord],
    *,
    game_date: date,
    game_pk: int,
    team_abbreviation: str,
) -> tuple[int, ...]:
    seen: set[int] = set()
    player_ids: list[int] = []
    candidates = sorted(
        (
            row
            for row in rows
            if row.game_pk == game_pk
            and _row_date(row) == game_date
            and row.batting_team_abbreviation == team_abbreviation
        ),
        key=lambda row: (row.at_bat_number, row.pitch_number),
    )
    for row in candidates:
        if row.batter_id in seen:
            continue
        seen.add(row.batter_id)
        player_ids.append(row.batter_id)
        if len(player_ids) >= MAX_PROJECTED_LINEUP_SIZE:
            break
    return tuple(player_ids)


def _projected_lineup_from_prior_game(
    *,
    target_date: date,
    team_abbreviation: str,
    team_lineups: dict[str, list[tuple[date, int, tuple[int, ...]]]],
) -> tuple[int, ...]:
    for game_date, _game_pk, lineup in reversed(team_lineups.get(team_abbreviation, [])):
        if game_date < target_date and lineup:
            return lineup
    return ()


def _lineup_from_starter_row(starter_row: dict[str, Any]) -> tuple[tuple[int, ...], str, bool, str | None]:
    raw_ids = starter_row.get("lineup_player_ids")
    lineup_ids: tuple[int, ...] = ()
    if isinstance(raw_ids, list):
        lineup_ids = tuple(int(player_id) for player_id in raw_ids[:MAX_PROJECTED_LINEUP_SIZE])
    lineup_status = str(starter_row.get("lineup_status") or "")
    if lineup_ids and lineup_status == "confirmed":
        return lineup_ids, "confirmed", True, starter_row.get("lineup_snapshot_id")
    if lineup_ids:
        return lineup_ids, "projected_snapshot", False, starter_row.get("lineup_snapshot_id")
    return (), "no_projection", False, None


def _prior_rows(index: _DatedPitchRows, *, target_date: date) -> tuple[StatcastPitchRecord, ...]:
    return index.rows[: bisect_left(index.dates, target_date)]


def _prior_rows_since(
    index: _DatedPitchRows,
    *,
    start_date: date,
    target_date: date,
) -> tuple[StatcastPitchRecord, ...]:
    start_index = bisect_left(index.dates, start_date)
    end_index = bisect_left(index.dates, target_date)
    return index.rows[start_index:end_index]


def _pitcher_hand_from_history(
    pitcher_rows: tuple[StatcastPitchRecord, ...],
) -> str | None:
    for row in reversed(pitcher_rows):
        if not row.p_throws:
            continue
        return row.p_throws
    return None


def _plate_appearance_rows(rows: list[StatcastPitchRecord]) -> list[StatcastPitchRecord]:
    return [row for row in rows if row.is_plate_appearance_final_pitch]


def _counts(rows: list[StatcastPitchRecord], *, pitcher_hand: str | None = None) -> dict[str, int]:
    counts = {
        "pitch_count": 0,
        "plate_appearance_count": 0,
        "strikeout_count": 0,
        "whiff_count": 0,
        "csw_count": 0,
        "swing_count": 0,
        "contact_count": 0,
        "out_of_zone_count": 0,
        "chase_count": 0,
    }
    for row in rows:
        if pitcher_hand is not None and row.p_throws != pitcher_hand:
            continue
        counts["pitch_count"] += 1
        if row.is_plate_appearance_final_pitch:
            counts["plate_appearance_count"] += 1
            if row.is_strikeout_event:
                counts["strikeout_count"] += 1
        if row.is_whiff:
            counts["whiff_count"] += 1
        if row.is_whiff or row.is_called_strike:
            counts["csw_count"] += 1
        if row.is_swing:
            counts["swing_count"] += 1
            if row.is_contact:
                counts["contact_count"] += 1
        if row.is_out_of_zone is True:
            counts["out_of_zone_count"] += 1
            if row.is_chase_swing is True:
                counts["chase_count"] += 1
    return counts


def _metric_bundle(
    rows: list[StatcastPitchRecord],
    *,
    pitcher_hand: str | None,
    prior_k_rate: float | None,
) -> dict[str, Any]:
    all_counts = _counts(rows)
    hand_counts = _counts(rows, pitcher_hand=pitcher_hand)
    k_rate = _safe_rate(all_counts["strikeout_count"], all_counts["plate_appearance_count"])
    k_rate_vs_hand = _safe_rate(
        hand_counts["strikeout_count"], hand_counts["plate_appearance_count"]
    )
    shrunk_source = k_rate_vs_hand if k_rate_vs_hand is not None else k_rate
    attempts = (
        hand_counts["plate_appearance_count"]
        if k_rate_vs_hand is not None
        else all_counts["plate_appearance_count"]
    )
    if shrunk_source is None and prior_k_rate is None:
        shrunk = None
    elif shrunk_source is None:
        shrunk = prior_k_rate
    else:
        prior_successes = 0.0 if prior_k_rate is None else prior_k_rate * PRIOR_PLATE_APPEARANCE_WEIGHT
        shrunk = round(
            ((shrunk_source * attempts) + prior_successes)
            / (attempts + (0 if prior_k_rate is None else PRIOR_PLATE_APPEARANCE_WEIGHT)),
            6,
        )
    return {
        "pitch_count": all_counts["pitch_count"],
        "plate_appearance_count": all_counts["plate_appearance_count"],
        "k_rate": k_rate,
        "k_rate_vs_pitcher_hand": k_rate_vs_hand,
        "k_rate_vs_pitcher_hand_shrunk": shrunk,
        "contact_rate": _safe_rate(all_counts["contact_count"], all_counts["swing_count"]),
        "chase_rate": _safe_rate(all_counts["chase_count"], all_counts["out_of_zone_count"]),
        "whiff_rate": _safe_rate(all_counts["whiff_count"], all_counts["swing_count"]),
        "csw_rate": _safe_rate(all_counts["csw_count"], all_counts["pitch_count"]),
    }


def _pitch_type_weakness(rows: list[StatcastPitchRecord]) -> dict[str, dict[str, Any]]:
    by_type: dict[str, list[StatcastPitchRecord]] = defaultdict(list)
    for row in rows:
        if row.pitch_type:
            by_type[row.pitch_type].append(row)
    payload: dict[str, dict[str, Any]] = {}
    for pitch_type, pitch_rows in sorted(by_type.items()):
        counts = _counts(pitch_rows)
        payload[pitch_type] = {
            "pitch_count": counts["pitch_count"],
            "whiff_rate": _safe_rate(counts["whiff_count"], counts["swing_count"]),
            "csw_rate": _safe_rate(counts["csw_count"], counts["pitch_count"]),
            "contact_rate": _safe_rate(counts["contact_count"], counts["swing_count"]),
        }
    return payload


def _pitcher_pitch_type_usage(
    pitcher_rows: tuple[StatcastPitchRecord, ...],
) -> dict[str, float]:
    counts: Counter[str] = Counter(
        row.pitch_type
        for row in pitcher_rows
        if row.pitch_type
    )
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {
        pitch_type: round(count / total, 6)
        for pitch_type, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    }


def _batting_order_weight(*, slot_index: int, lineup_size: int) -> float:
    return float(lineup_size - slot_index)


def _league_k_priors_by_date(
    *,
    rows: list[StatcastPitchRecord],
    target_dates: list[date],
) -> dict[date, float | None]:
    sorted_rows = sorted(rows, key=lambda row: (row.game_date, row.game_pk))
    priors: dict[date, float | None] = {}
    cursor = 0
    plate_appearances = 0
    strikeouts = 0
    for target_date in sorted(set(target_dates)):
        while cursor < len(sorted_rows) and _row_date(sorted_rows[cursor]) < target_date:
            row = sorted_rows[cursor]
            if row.is_plate_appearance_final_pitch:
                plate_appearances += 1
                strikeouts += 1 if row.is_strikeout_event else 0
            cursor += 1
        priors[target_date] = _safe_rate(strikeouts, plate_appearances)
    return priors


def _batter_index(
    rows: list[StatcastPitchRecord],
) -> dict[int, _DatedPitchRows]:
    indexed: dict[int, list[StatcastPitchRecord]] = defaultdict(list)
    for row in rows:
        indexed[row.batter_id].append(row)
    result: dict[int, _DatedPitchRows] = {}
    for batter_id, batter_rows in indexed.items():
        sorted_rows = tuple(sorted(
            batter_rows,
            key=lambda row: (row.game_date, row.game_pk, row.at_bat_number, row.pitch_number),
        ))
        result[batter_id] = _DatedPitchRows(
            rows=sorted_rows,
            dates=tuple(_row_date(row) for row in sorted_rows),
        )
    return result


def _pitcher_index(
    rows: list[StatcastPitchRecord],
) -> dict[int, _DatedPitchRows]:
    indexed: dict[int, list[StatcastPitchRecord]] = defaultdict(list)
    for row in rows:
        indexed[row.pitcher_id].append(row)
    result: dict[int, _DatedPitchRows] = {}
    for pitcher_id, pitcher_rows in indexed.items():
        sorted_rows = tuple(sorted(
            pitcher_rows,
            key=lambda row: (row.game_date, row.game_pk, row.at_bat_number, row.pitch_number),
        ))
        result[pitcher_id] = _DatedPitchRows(
            rows=sorted_rows,
            dates=tuple(_row_date(row) for row in sorted_rows),
        )
    return result


def _team_lineup_index(
    rows: list[StatcastPitchRecord],
) -> dict[str, list[tuple[date, int, tuple[int, ...]]]]:
    grouped: dict[tuple[str, date, int], list[StatcastPitchRecord]] = defaultdict(list)
    for row in rows:
        if row.batting_team_abbreviation is None:
            continue
        grouped[(row.batting_team_abbreviation, _row_date(row), row.game_pk)].append(row)
    indexed: dict[str, list[tuple[date, int, tuple[int, ...]]]] = defaultdict(list)
    for (team_abbreviation, game_date, game_pk), game_rows in grouped.items():
        seen: set[int] = set()
        lineup: list[int] = []
        for row in sorted(game_rows, key=lambda row: (row.at_bat_number, row.pitch_number)):
            if row.batter_id in seen:
                continue
            seen.add(row.batter_id)
            lineup.append(row.batter_id)
            if len(lineup) >= MAX_PROJECTED_LINEUP_SIZE:
                break
        indexed[team_abbreviation].append((game_date, game_pk, tuple(lineup)))
    return {
        team_abbreviation: sorted(lineups, key=lambda row: (row[0], row[1]))
        for team_abbreviation, lineups in indexed.items()
    }


def _batter_feature_row(
    *,
    starter_row: dict[str, Any],
    batter_id: int,
    slot_index: int,
    lineup_size: int,
    batter_rows: list[StatcastPitchRecord],
    recent_rows: list[StatcastPitchRecord],
    pitcher_hand: str | None,
    pitcher_pitch_type_usage: dict[str, float],
    prior_k_rate: float | None,
) -> dict[str, Any]:
    season = int(starter_row["season"])
    season_rows = [row for row in batter_rows if _row_date(row).year == season]
    career = _metric_bundle(batter_rows, pitcher_hand=pitcher_hand, prior_k_rate=prior_k_rate)
    season_bundle = _metric_bundle(
        season_rows, pitcher_hand=pitcher_hand, prior_k_rate=career["k_rate_vs_pitcher_hand_shrunk"]
    )
    recent = _metric_bundle(
        recent_rows, pitcher_hand=pitcher_hand, prior_k_rate=season_bundle["k_rate_vs_pitcher_hand_shrunk"]
    )
    pitch_type_weakness = _pitch_type_weakness(season_rows or batter_rows)
    matchup_values: list[tuple[float, float]] = []
    for pitch_type, usage in pitcher_pitch_type_usage.items():
        weakness = pitch_type_weakness.get(pitch_type, {})
        value = weakness.get("whiff_rate") or weakness.get("csw_rate")
        if value is not None:
            matchup_values.append((float(value), usage))
    return {
        "batter_feature_row_id": (
            f"batter-matchup-feature:{starter_row['official_date']}:"
            f"{starter_row['game_pk']}:{starter_row['pitcher_id']}:{slot_index + 1}:{batter_id}"
        ),
        "lineup_feature_row_id": (
            f"lineup-matchup-feature:{starter_row['official_date']}:"
            f"{starter_row['game_pk']}:{starter_row['pitcher_id']}"
        ),
        "training_row_id": starter_row["training_row_id"],
        "official_date": starter_row["official_date"],
        "game_pk": starter_row["game_pk"],
        "pitcher_id": starter_row["pitcher_id"],
        "pitcher_hand": pitcher_hand,
        "opponent_team_abbreviation": starter_row.get("opponent_team_abbreviation"),
        "batting_order_slot": slot_index + 1,
        "batting_order_weight": _batting_order_weight(slot_index=slot_index, lineup_size=lineup_size),
        "batter_id": batter_id,
        "batter_stand": next((row.stand for row in reversed(batter_rows) if row.stand), None),
        "prior_pitch_count": career["pitch_count"],
        "prior_plate_appearance_count": career["plate_appearance_count"],
        "career_k_rate": career["k_rate"],
        "career_k_rate_vs_pitcher_hand": career["k_rate_vs_pitcher_hand"],
        "career_k_rate_vs_pitcher_hand_shrunk": career["k_rate_vs_pitcher_hand_shrunk"],
        "career_contact_rate": career["contact_rate"],
        "career_chase_rate": career["chase_rate"],
        "career_whiff_rate": career["whiff_rate"],
        "career_csw_rate": career["csw_rate"],
        "season_plate_appearance_count": season_bundle["plate_appearance_count"],
        "season_k_rate_vs_pitcher_hand_shrunk": season_bundle["k_rate_vs_pitcher_hand_shrunk"],
        "recent_15d_plate_appearance_count": recent["plate_appearance_count"],
        "recent_15d_k_rate_vs_pitcher_hand_shrunk": recent["k_rate_vs_pitcher_hand_shrunk"],
        "pitch_type_weakness": pitch_type_weakness,
        "arsenal_weighted_pitch_type_weakness": _weighted_mean(matchup_values),
        "history_status": "ok" if batter_rows else "missing_batter_history",
    }


def _lineup_feature_row(
    *,
    starter_row: dict[str, Any],
    batter_rows_by_id: dict[int, _DatedPitchRows],
    pitcher_rows_by_id: dict[int, _DatedPitchRows],
    team_lineups: dict[str, list[tuple[date, int, tuple[int, ...]]]],
    prior_k_rate: float | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target_date = date.fromisoformat(str(starter_row["official_date"]))
    lineup_ids, lineup_status, lineup_is_confirmed, lineup_snapshot_id = _lineup_from_starter_row(starter_row)
    if not lineup_ids:
        projected = _projected_lineup_from_prior_game(
            target_date=target_date,
            team_abbreviation=str(starter_row.get("opponent_team_abbreviation") or ""),
            team_lineups=team_lineups,
        )
        if projected:
            lineup_ids = projected
            lineup_status = "projected_from_prior_team_game"
    pitcher_id = int(starter_row["pitcher_id"])
    pitcher_rows = _prior_rows(
        pitcher_rows_by_id.get(pitcher_id, _EMPTY_DATED_PITCH_ROWS),
        target_date=target_date,
    )
    pitcher_hand = starter_row.get("pitcher_hand") or _pitcher_hand_from_history(
        pitcher_rows
    )
    pitcher_usage = _pitcher_pitch_type_usage(pitcher_rows)
    batter_feature_rows: list[dict[str, Any]] = []
    lineup_size = len(lineup_ids)
    recent_cutoff = target_date - timedelta(days=RECENT_FORM_DAYS)
    for slot_index, batter_id in enumerate(lineup_ids):
        batter_index = batter_rows_by_id.get(batter_id, _EMPTY_DATED_PITCH_ROWS)
        batter_rows = _prior_rows(
            batter_index,
            target_date=target_date,
        )
        recent_rows = _prior_rows_since(
            batter_index,
            start_date=recent_cutoff,
            target_date=target_date,
        )
        batter_feature_rows.append(
            _batter_feature_row(
                starter_row=starter_row,
                batter_id=batter_id,
                slot_index=slot_index,
                lineup_size=lineup_size,
                batter_rows=batter_rows,
                recent_rows=recent_rows,
                pitcher_hand=pitcher_hand,
                pitcher_pitch_type_usage=pitcher_usage,
                prior_k_rate=prior_k_rate,
            )
        )
    weighted_k = [
        (row["career_k_rate_vs_pitcher_hand_shrunk"], row["batting_order_weight"])
        for row in batter_feature_rows
        if row["career_k_rate_vs_pitcher_hand_shrunk"] is not None
    ]
    weighted_contact = [
        (row["career_contact_rate"], row["batting_order_weight"])
        for row in batter_feature_rows
        if row["career_contact_rate"] is not None
    ]
    weighted_chase = [
        (row["career_chase_rate"], row["batting_order_weight"])
        for row in batter_feature_rows
        if row["career_chase_rate"] is not None
    ]
    weighted_whiff = [
        (row["career_whiff_rate"], row["batting_order_weight"])
        for row in batter_feature_rows
        if row["career_whiff_rate"] is not None
    ]
    weighted_csw = [
        (row["career_csw_rate"], row["batting_order_weight"])
        for row in batter_feature_rows
        if row["career_csw_rate"] is not None
    ]
    weighted_arsenal = [
        (row["arsenal_weighted_pitch_type_weakness"], row["batting_order_weight"])
        for row in batter_feature_rows
        if row["arsenal_weighted_pitch_type_weakness"] is not None
    ]
    available_count = sum(1 for row in batter_feature_rows if row["history_status"] == "ok")
    incomplete_count = max(0, lineup_size - available_count)
    feature_status = "ok"
    if lineup_status == "no_projection":
        feature_status = "missing_lineup_projection"
    elif incomplete_count:
        feature_status = "incomplete_batter_history"
    lineup_row = {
        "feature_row_id": (
            f"lineup-matchup-feature:{starter_row['official_date']}:"
            f"{starter_row['game_pk']}:{starter_row['pitcher_id']}"
        ),
        "training_row_id": starter_row["training_row_id"],
        "official_date": starter_row["official_date"],
        "season": int(starter_row["season"]),
        "game_pk": starter_row["game_pk"],
        "pitcher_id": pitcher_id,
        "pitcher_name": starter_row.get("pitcher_name"),
        "pitcher_hand": pitcher_hand,
        "team_abbreviation": starter_row.get("team_abbreviation"),
        "opponent_team_abbreviation": starter_row.get("opponent_team_abbreviation"),
        "features_as_of": f"{target_date.isoformat()}T00:00:00Z",
        "leakage_policy_status": "ok_prior_games_only",
        "lineup_status": lineup_status,
        "lineup_is_confirmed": lineup_is_confirmed,
        "lineup_snapshot_id": lineup_snapshot_id,
        "lineup_size": lineup_size,
        "lineup_player_ids": list(lineup_ids),
        "available_batter_feature_count": available_count,
        "incomplete_batter_history_count": incomplete_count,
        "feature_status": feature_status,
        "projected_lineup_k_rate_vs_pitcher_hand_weighted": _weighted_mean(weighted_k),
        "projected_lineup_contact_rate_weighted": _weighted_mean(weighted_contact),
        "projected_lineup_chase_rate_weighted": _weighted_mean(weighted_chase),
        "projected_lineup_whiff_rate_weighted": _weighted_mean(weighted_whiff),
        "projected_lineup_csw_rate_weighted": _weighted_mean(weighted_csw),
        "arsenal_weighted_lineup_pitch_type_weakness": _weighted_mean(weighted_arsenal),
        "pitcher_pitch_type_usage": pitcher_usage,
        "identity_prior_context": {
            "league_k_rate_prior": prior_k_rate,
            "prior_plate_appearance_weight": PRIOR_PLATE_APPEARANCE_WEIGHT,
            "usage": "sample_size_regression_not_batter_id_model_feature",
        },
    }
    return lineup_row, batter_feature_rows


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
    batter_rows: list[dict[str, Any]],
    targets_by_training_row_id: dict[str, int | float],
    feature_path: Path,
    batter_feature_path: Path,
) -> dict[str, Any]:
    status_counts = Counter(str(row["lineup_status"]) for row in rows)
    feature_status_counts = Counter(str(row["feature_status"]) for row in rows)
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
            "batter_feature_rows": len(batter_rows),
            "pitch_rows": pitch_row_count,
            "confirmed_lineups": status_counts["confirmed"],
            "projected_lineups": (
                status_counts["projected_snapshot"]
                + status_counts["projected_from_prior_team_game"]
            ),
            "missing_lineup_projection": feature_status_counts["missing_lineup_projection"],
            "incomplete_batter_history": feature_status_counts["incomplete_batter_history"],
        },
        "missingness": {
            "no_confirmed_lineup": len(rows) - status_counts["confirmed"],
            "no_projection": status_counts["no_projection"],
            "incomplete_batter_history": feature_status_counts["incomplete_batter_history"],
        },
        "leakage_policy": {
            "status": "ok" if all(row["leakage_policy_status"] == "ok_prior_games_only" for row in rows) else "violations",
            "rule": "Only batter and pitcher Statcast rows with game_date before the starter official_date are eligible.",
        },
        "field_coverage": field_coverage,
        "top_correlations_by_season": correlations_by_season,
        "artifacts": {
            "lineup_matchup_features_path": feature_path,
            "batter_matchup_features_path": batter_feature_path,
        },
    }


def _render_feature_report_markdown(report: dict[str, Any]) -> str:
    rows = report["row_counts"]
    missingness = report["missingness"]
    lines = [
        "# Lineup Matchup Feature Report",
        "",
        f"- Run ID: `{report['run_id']}`",
        (
            "- Date window: "
            f"`{report['date_window']['start_date']}` -> `{report['date_window']['end_date']}`"
        ),
        f"- Feature rows: `{rows['feature_rows']}`",
        f"- Batter feature rows: `{rows['batter_feature_rows']}`",
        f"- Pitch rows read: `{rows['pitch_rows']}`",
        f"- Confirmed lineups: `{rows['confirmed_lineups']}`",
        f"- Projected lineups: `{rows['projected_lineups']}`",
        f"- No confirmed lineup: `{missingness['no_confirmed_lineup']}`",
        f"- No projection: `{missingness['no_projection']}`",
        f"- Incomplete batter history rows: `{missingness['incomplete_batter_history']}`",
        f"- Leakage policy: `{report['leakage_policy']['status']}`",
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
            "## Timestamp Policy",
            "",
            "Confirmed lineup snapshots are used only when carried by the starter-game "
            "dataset as pregame references. Otherwise the builder projects the opposing "
            "lineup from that team's most recent prior game in the preserved Statcast "
            "history. Same-game batting orders are never used as pregame lineup features.",
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
        "uv run python -m mlb_props_stack build-lineup-matchup-features "
        f"--start-date {start_date.isoformat()} --end-date {end_date.isoformat()} "
        f"--output-dir {output_dir}"
    )
    if dataset_run_dir is not None:
        command += f" --dataset-run-dir {dataset_run_dir}"
    return "\n".join(
        [
            "# Lineup Matchup Feature Reproducibility",
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
            "Batter history, pitch-type weakness, pitcher arsenal, and projected "
            "lineup fallback inputs are restricted to games before each starter "
            "official date. Missing confirmed lineups, missing projections, and "
            "incomplete batter history remain explicit status fields and are not "
            "converted to zero-valued features.",
            "",
        ]
    )


def build_lineup_matchup_features(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str = "data",
    dataset_run_dir: Path | str | None = None,
    now: Any = utc_now,
) -> LineupMatchupFeatureBuildResult:
    """Build batter-by-batter and aggregate lineup matchup features."""
    output_root = Path(output_dir)
    explicit_dataset_dir = Path(dataset_run_dir) if dataset_run_dir is not None else None
    resolved_dataset_run_dir = _resolve_dataset_run_dir(
        output_dir=output_root,
        start_date=start_date,
        end_date=end_date,
        dataset_run_dir=explicit_dataset_dir,
    )
    dataset_rows = _load_jsonl(resolved_dataset_run_dir / "starter_game_training_dataset.jsonl")
    pitch_rows = _load_pitch_records_from_manifest(
        resolved_dataset_run_dir / "source_manifest.jsonl",
        latest_needed_date=end_date,
    )
    batter_rows_by_id = _batter_index(pitch_rows)
    pitcher_rows_by_id = _pitcher_index(pitch_rows)
    team_lineups = _team_lineup_index(pitch_rows)
    target_dates = [
        date.fromisoformat(str(row["official_date"]))
        for row in dataset_rows
        if start_date <= date.fromisoformat(str(row["official_date"])) <= end_date
    ]
    league_priors = _league_k_priors_by_date(rows=pitch_rows, target_dates=target_dates)
    lineup_rows: list[dict[str, Any]] = []
    batter_rows: list[dict[str, Any]] = []
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
        lineup_row, row_batter_rows = _lineup_feature_row(
            starter_row=starter_row,
            batter_rows_by_id=batter_rows_by_id,
            pitcher_rows_by_id=pitcher_rows_by_id,
            team_lineups=team_lineups,
            prior_k_rate=league_priors[target_date],
        )
        lineup_rows.append(lineup_row)
        batter_rows.extend(row_batter_rows)
    run_root = (
        output_root
        / "normalized"
        / "lineup_matchup_features"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
    )
    run_id = _unique_timestamp_run_id(now().astimezone(UTC), run_root)
    run_dir = (
        run_root
        / f"run={run_id}"
    )
    feature_path = run_dir / "lineup_matchup_features.jsonl"
    batter_feature_path = run_dir / "batter_matchup_features.jsonl"
    feature_report_path = run_dir / "feature_report.json"
    feature_report_markdown_path = run_dir / "feature_report.md"
    reproducibility_notes_path = run_dir / "reproducibility_notes.md"
    report = _feature_report(
        start_date=start_date,
        end_date=end_date,
        run_id=run_id,
        dataset_run_dir=resolved_dataset_run_dir,
        pitch_row_count=len(pitch_rows),
        rows=lineup_rows,
        batter_rows=batter_rows,
        targets_by_training_row_id=targets_by_training_row_id,
        feature_path=feature_path,
        batter_feature_path=batter_feature_path,
    )
    _write_jsonl(feature_path, lineup_rows)
    _write_jsonl(batter_feature_path, batter_rows)
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
    return LineupMatchupFeatureBuildResult(
        start_date=start_date,
        end_date=end_date,
        run_id=run_id,
        dataset_row_count=len(dataset_rows),
        feature_row_count=len(lineup_rows),
        batter_feature_row_count=len(batter_rows),
        pitch_row_count=len(pitch_rows),
        feature_path=feature_path,
        batter_feature_path=batter_feature_path,
        feature_report_path=feature_report_path,
        feature_report_markdown_path=feature_report_markdown_path,
        reproducibility_notes_path=reproducibility_notes_path,
    )
