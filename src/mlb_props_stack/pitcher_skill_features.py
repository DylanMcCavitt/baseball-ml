"""Pitcher skill and pitch-arsenal features for the projection rebuild."""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from io import StringIO
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .ingest.mlb_stats_api import utc_now
from .ingest.statcast_ingest import (
    CALLED_STRIKE_DESCRIPTIONS,
    CONTACT_DESCRIPTIONS,
    STRIKEOUT_EVENTS,
    SWING_DESCRIPTIONS,
    WHIFF_DESCRIPTIONS,
)
from .modeling import _unique_timestamp_run_id

FEATURE_SET_VERSION = "pitcher_skill_arsenal_features_v1"
PRIOR_PLATE_APPEARANCE_WEIGHT = 120
PRIOR_PITCH_WEIGHT = 350
RECENT_FORM_DAYS = 15
REST_BUCKETS = (
    "no_prior_start",
    "short_rest",
    "standard_rest",
    "extra_rest",
    "long_layoff",
)


@dataclass(frozen=True)
class PitcherSkillFeatureBuildResult:
    """Filesystem output summary for one pitcher skill feature build."""

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
class _PitchRow:
    game_date: date
    game_pk: int
    pitcher_id: int
    pitch_type: str | None
    release_speed: float | None
    release_spin_rate: float | None
    release_extension: float | None
    pfx_x: float | None
    pfx_z: float | None
    description: str | None
    events: str | None
    strikes: int | None
    is_final_pitch: bool
    is_strikeout: bool
    is_walk: bool
    is_whiff: bool
    is_called_strike: bool
    is_swing: bool
    is_contact: bool
    is_strike: bool


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


def _optional_float(value: Any) -> float | None:
    rendered = _optional_text(value)
    if rendered is None:
        return None
    return float(rendered)


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


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


def _parse_pitch_rows(csv_text: str) -> list[_PitchRow]:
    reader = csv.DictReader(StringIO(csv_text.lstrip("\ufeff")))
    if not reader.fieldnames or "game_date" not in reader.fieldnames:
        return []
    rows: list[_PitchRow] = []
    for raw in reader:
        game_date_text = _optional_text(raw.get("game_date"))
        game_pk = _optional_int(raw.get("game_pk"))
        pitcher_id = _optional_int(raw.get("pitcher"))
        if game_date_text is None or game_pk is None or pitcher_id is None:
            continue
        description = _optional_text(raw.get("description"))
        description_key = description.lower() if description else None
        events = _optional_text(raw.get("events"))
        events_key = events.lower() if events else None
        is_whiff = description_key in WHIFF_DESCRIPTIONS
        is_called_strike = description_key in CALLED_STRIKE_DESCRIPTIONS
        is_contact = description_key in CONTACT_DESCRIPTIONS
        is_swing = description_key in SWING_DESCRIPTIONS
        rows.append(
            _PitchRow(
                game_date=date.fromisoformat(game_date_text),
                game_pk=game_pk,
                pitcher_id=pitcher_id,
                pitch_type=_optional_text(raw.get("pitch_type")),
                release_speed=_optional_float(raw.get("release_speed")),
                release_spin_rate=_optional_float(
                    raw.get("release_spin_rate") or raw.get("release_spin")
                ),
                release_extension=_optional_float(raw.get("release_extension")),
                pfx_x=_optional_float(raw.get("pfx_x")),
                pfx_z=_optional_float(raw.get("pfx_z")),
                description=description,
                events=events,
                strikes=_optional_int(raw.get("strikes")),
                is_final_pitch=events_key is not None,
                is_strikeout=events_key in STRIKEOUT_EVENTS,
                is_walk=events_key in {"walk", "intent_walk"},
                is_whiff=is_whiff,
                is_called_strike=is_called_strike,
                is_swing=is_swing,
                is_contact=is_contact,
                is_strike=is_whiff or is_called_strike or is_contact,
            )
        )
    return rows


def _load_pitch_rows_from_manifest(source_manifest_path: Path) -> list[_PitchRow]:
    pitch_rows: list[_PitchRow] = []
    for manifest_row in _load_jsonl(source_manifest_path):
        raw_path_text = manifest_row.get("raw_path")
        if raw_path_text is None:
            continue
        raw_path = Path(str(raw_path_text))
        if not raw_path.is_absolute():
            raw_path = source_manifest_path.parent / raw_path
        if not raw_path.exists():
            continue
        pitch_rows.extend(_parse_pitch_rows(raw_path.read_text(encoding="utf-8")))
    return pitch_rows


def _pitcher_index(pitch_rows: list[_PitchRow]) -> dict[int, list[_PitchRow]]:
    rows_by_pitcher: dict[int, list[_PitchRow]] = defaultdict(list)
    for row in pitch_rows:
        rows_by_pitcher[row.pitcher_id].append(row)
    return {
        pitcher_id: sorted(rows, key=lambda row: (row.game_date, row.game_pk))
        for pitcher_id, rows in rows_by_pitcher.items()
    }


def _pitcher_date_index(
    rows_by_pitcher: dict[int, list[_PitchRow]],
) -> dict[int, list[date]]:
    return {
        pitcher_id: [row.game_date for row in rows]
        for pitcher_id, rows in rows_by_pitcher.items()
    }


def _prior_rows(
    *,
    pitcher_rows: list[_PitchRow],
    pitcher_dates: list[date],
    target_date: date,
) -> list[_PitchRow]:
    return pitcher_rows[: bisect_left(pitcher_dates, target_date)]


def _rows_since(rows: list[_PitchRow], *, target_date: date, days: int) -> list[_PitchRow]:
    earliest = target_date - timedelta(days=days)
    return [row for row in rows if row.game_date >= earliest]


def _last_starts(rows: list[_PitchRow], *, count: int) -> list[_PitchRow]:
    by_start: dict[tuple[date, int], list[_PitchRow]] = defaultdict(list)
    for row in rows:
        by_start[(row.game_date, row.game_pk)].append(row)
    selected_keys = sorted(by_start, reverse=True)[:count]
    selected: list[_PitchRow] = []
    for key in selected_keys:
        selected.extend(by_start[key])
    return selected


def _last_game_date(rows: list[_PitchRow]) -> date | None:
    if not rows:
        return None
    return max(row.game_date for row in rows)


def _rest_context(*, target_date: date, prior_rows: list[_PitchRow]) -> dict[str, Any]:
    last_game_date = _last_game_date(prior_rows)
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
    payload = {
        "last_prior_game_date": last_game_date,
        "rest_days_capped": None if rest_days is None else min(rest_days, 14),
        "rest_bucket": bucket,
        "short_rest_flag": bucket == "short_rest",
        "standard_rest_flag": bucket == "standard_rest",
        "extra_rest_flag": bucket == "extra_rest",
        "long_layoff_flag": bucket == "long_layoff",
        "unknown_or_no_prior_rest_flag": bucket == "no_prior_start",
    }
    return payload


def _pitch_type_rates(rows: list[_PitchRow]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    counts: Counter[str] = Counter(row.pitch_type for row in rows if row.pitch_type)
    whiffs: Counter[str] = Counter(row.pitch_type for row in rows if row.pitch_type and row.is_whiff)
    csw: Counter[str] = Counter(
        row.pitch_type
        for row in rows
        if row.pitch_type and (row.is_whiff or row.is_called_strike)
    )
    total = sum(counts.values())
    usage = {
        pitch_type: round(count / total, 6)
        for pitch_type, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    } if total else {}
    whiff_rates = {
        pitch_type: round(whiffs[pitch_type] / count, 6)
        for pitch_type, count in sorted(counts.items())
    }
    csw_rates = {
        pitch_type: round(csw[pitch_type] / count, 6)
        for pitch_type, count in sorted(counts.items())
    }
    return usage, whiff_rates, csw_rates


def _rates(rows: list[_PitchRow]) -> dict[str, Any]:
    final_pitch_rows = [row for row in rows if row.is_final_pitch]
    two_strike_rows = [row for row in rows if row.strikes is not None and row.strikes >= 2]
    strikeouts = sum(1 for row in final_pitch_rows if row.is_strikeout)
    walks = sum(1 for row in final_pitch_rows if row.is_walk)
    pitches = len(rows)
    plate_appearances = len(final_pitch_rows)
    return {
        "pitch_count": pitches,
        "plate_appearance_count": plate_appearances,
        "strikeout_count": strikeouts,
        "walk_count": walks,
        "k_rate": _safe_rate(strikeouts, plate_appearances),
        "walk_rate": _safe_rate(walks, plate_appearances),
        "k_minus_bb_rate": _safe_rate(strikeouts - walks, plate_appearances),
        "strike_rate": _safe_rate(sum(1 for row in rows if row.is_strike), pitches),
        "csw_rate": _safe_rate(
            sum(1 for row in rows if row.is_whiff or row.is_called_strike),
            pitches,
        ),
        "swstr_rate": _safe_rate(sum(1 for row in rows if row.is_whiff), pitches),
        "whiff_rate": _safe_rate(
            sum(1 for row in rows if row.is_whiff),
            sum(1 for row in rows if row.is_swing),
        ),
        "called_strike_rate": _safe_rate(
            sum(1 for row in rows if row.is_called_strike),
            pitches,
        ),
        "putaway_rate": _safe_rate(
            sum(1 for row in two_strike_rows if row.is_final_pitch and row.is_strikeout),
            len(two_strike_rows),
        ),
    }


def _shrink_rate(
    *,
    successes: int,
    attempts: int,
    prior_rate: float | None,
    prior_weight: int,
) -> float | None:
    if prior_rate is None and attempts == 0:
        return None
    if prior_rate is None:
        return _safe_rate(successes, attempts)
    return round((successes + (prior_rate * prior_weight)) / (attempts + prior_weight), 6)


def _metric_delta(recent: list[_PitchRow], career: list[_PitchRow], attr: str) -> tuple[float | None, float | None, float | None]:
    recent_values = [value for row in recent if (value := getattr(row, attr)) is not None]
    career_values = [value for row in career if (value := getattr(row, attr)) is not None]
    recent_mean = _mean(recent_values)
    career_mean = _mean(career_values)
    if recent_mean is None or career_mean is None:
        return recent_mean, career_mean, None
    return recent_mean, career_mean, round(recent_mean - career_mean, 6)


def _league_priors(rows: list[_PitchRow]) -> dict[str, float | None]:
    rates = _rates(rows)
    return {
        "k_rate": rates["k_rate"],
        "walk_rate": rates["walk_rate"],
        "csw_rate": rates["csw_rate"],
        "swstr_rate": rates["swstr_rate"],
    }


def _league_priors_from_counts(counts: dict[str, int]) -> dict[str, float | None]:
    return {
        "k_rate": _safe_rate(counts["strikeouts"], counts["plate_appearances"]),
        "walk_rate": _safe_rate(counts["walks"], counts["plate_appearances"]),
        "csw_rate": _safe_rate(counts["csw"], counts["pitches"]),
        "swstr_rate": _safe_rate(counts["whiffs"], counts["pitches"]),
    }


def _league_priors_by_date(
    *,
    pitch_rows: list[_PitchRow],
    target_dates: list[date],
) -> dict[date, dict[str, float | None]]:
    sorted_pitch_rows = sorted(pitch_rows, key=lambda row: row.game_date)
    priors: dict[date, dict[str, float | None]] = {}
    cursor = 0
    counts = {
        "pitches": 0,
        "plate_appearances": 0,
        "strikeouts": 0,
        "walks": 0,
        "csw": 0,
        "whiffs": 0,
    }
    for target_date in sorted(set(target_dates)):
        while cursor < len(sorted_pitch_rows) and sorted_pitch_rows[cursor].game_date < target_date:
            row = sorted_pitch_rows[cursor]
            counts["pitches"] += 1
            counts["csw"] += 1 if row.is_whiff or row.is_called_strike else 0
            counts["whiffs"] += 1 if row.is_whiff else 0
            if row.is_final_pitch:
                counts["plate_appearances"] += 1
                counts["strikeouts"] += 1 if row.is_strikeout else 0
                counts["walks"] += 1 if row.is_walk else 0
            cursor += 1
        priors[target_date] = _league_priors_from_counts(counts)
    return priors


def _feature_row(
    *,
    starter_row: dict[str, Any],
    prior_rows: list[_PitchRow],
    league_priors: dict[str, float | None],
) -> dict[str, Any]:
    target_date = date.fromisoformat(str(starter_row["official_date"]))
    season_rows = [row for row in prior_rows if row.game_date.year == target_date.year]
    recent_rows = _rows_since(prior_rows, target_date=target_date, days=RECENT_FORM_DAYS)
    last3_rows = _last_starts(prior_rows, count=3)
    career = _rates(prior_rows)
    season = _rates(season_rows)
    recent = _rates(recent_rows)
    last3 = _rates(last3_rows)
    usage, whiff_by_type, csw_by_type = _pitch_type_rates(season_rows or prior_rows)
    recent_speed, career_speed, speed_delta = _metric_delta(recent_rows, prior_rows, "release_speed")
    recent_spin, career_spin, spin_delta = _metric_delta(recent_rows, prior_rows, "release_spin_rate")
    recent_extension, career_extension, extension_delta = _metric_delta(
        recent_rows, prior_rows, "release_extension"
    )
    recent_pfx_x, career_pfx_x, pfx_x_delta = _metric_delta(recent_rows, prior_rows, "pfx_x")
    recent_pfx_z, career_pfx_z, pfx_z_delta = _metric_delta(recent_rows, prior_rows, "pfx_z")
    career_k_shrunk = _shrink_rate(
        successes=career["strikeout_count"],
        attempts=career["plate_appearance_count"],
        prior_rate=league_priors["k_rate"],
        prior_weight=PRIOR_PLATE_APPEARANCE_WEIGHT,
    )
    season_k_shrunk = _shrink_rate(
        successes=season["strikeout_count"],
        attempts=season["plate_appearance_count"],
        prior_rate=career_k_shrunk or league_priors["k_rate"],
        prior_weight=PRIOR_PLATE_APPEARANCE_WEIGHT,
    )
    swstr_shrunk = _shrink_rate(
        successes=sum(1 for row in prior_rows if row.is_whiff),
        attempts=len(prior_rows),
        prior_rate=league_priors["swstr_rate"],
        prior_weight=PRIOR_PITCH_WEIGHT,
    )
    rest_context = _rest_context(target_date=target_date, prior_rows=prior_rows)
    feature_status = "ok" if prior_rows else "missing_prior_pitch_history"
    payload: dict[str, Any] = {
        "feature_row_id": (
            "pitcher-skill-feature:"
            f"{starter_row['official_date']}:{starter_row['game_pk']}:{starter_row['pitcher_id']}"
        ),
        "training_row_id": starter_row["training_row_id"],
        "official_date": starter_row["official_date"],
        "season": int(starter_row["season"]),
        "game_pk": starter_row["game_pk"],
        "pitcher_id": starter_row["pitcher_id"],
        "pitcher_name": starter_row.get("pitcher_name"),
        "team_abbreviation": starter_row.get("team_abbreviation"),
        "opponent_team_abbreviation": starter_row.get("opponent_team_abbreviation"),
        "features_as_of": f"{target_date.isoformat()}T00:00:00Z",
        "feature_status": feature_status,
        "leakage_policy_status": "ok_prior_games_only",
        "prior_pitch_count": career["pitch_count"],
        "prior_plate_appearance_count": career["plate_appearance_count"],
        "career_k_rate": career["k_rate"],
        "career_k_rate_shrunk": career_k_shrunk,
        "career_walk_rate": career["walk_rate"],
        "career_k_minus_bb_rate": career["k_minus_bb_rate"],
        "career_strike_rate": career["strike_rate"],
        "career_csw_rate": career["csw_rate"],
        "career_swstr_rate": career["swstr_rate"],
        "career_whiff_rate": career["whiff_rate"],
        "career_called_strike_rate": career["called_strike_rate"],
        "career_putaway_rate": career["putaway_rate"],
        "season_pitch_count": season["pitch_count"],
        "season_plate_appearance_count": season["plate_appearance_count"],
        "season_k_rate": season["k_rate"],
        "season_k_rate_shrunk": season_k_shrunk,
        "season_walk_rate": season["walk_rate"],
        "season_k_minus_bb_rate": season["k_minus_bb_rate"],
        "season_csw_rate": season["csw_rate"],
        "season_swstr_rate": season["swstr_rate"],
        "recent_15d_pitch_count": recent["pitch_count"],
        "recent_15d_plate_appearance_count": recent["plate_appearance_count"],
        "recent_15d_k_rate": recent["k_rate"],
        "recent_15d_csw_rate": recent["csw_rate"],
        "recent_15d_swstr_rate": recent["swstr_rate"],
        "last_3_starts_pitch_count": last3["pitch_count"],
        "last_3_starts_plate_appearance_count": last3["plate_appearance_count"],
        "last_3_starts_k_rate": last3["k_rate"],
        "last_3_starts_csw_rate": last3["csw_rate"],
        "pitch_type_usage": usage,
        "pitch_type_whiff_rate": whiff_by_type,
        "pitch_type_csw_rate": csw_by_type,
        "average_release_speed": career_speed,
        "recent_release_speed": recent_speed,
        "release_speed_delta_vs_pitcher_baseline": speed_delta,
        "average_release_spin_rate": career_spin,
        "recent_release_spin_rate": recent_spin,
        "release_spin_rate_delta_vs_pitcher_baseline": spin_delta,
        "average_release_extension": career_extension,
        "recent_release_extension": recent_extension,
        "release_extension_delta_vs_pitcher_baseline": extension_delta,
        "average_horizontal_movement": career_pfx_x,
        "recent_horizontal_movement": recent_pfx_x,
        "horizontal_movement_delta_vs_pitcher_baseline": pfx_x_delta,
        "average_vertical_movement": career_pfx_z,
        "recent_vertical_movement": recent_pfx_z,
        "vertical_movement_delta_vs_pitcher_baseline": pfx_z_delta,
        "identity_prior_context": {
            "career_k_rate_shrunk": career_k_shrunk,
            "season_k_rate_shrunk": season_k_shrunk,
            "swstr_rate_shrunk": swstr_shrunk,
            "prior_plate_appearance_weight": PRIOR_PLATE_APPEARANCE_WEIGHT,
            "prior_pitch_weight": PRIOR_PITCH_WEIGHT,
            "usage": "shrinkage_context_not_pitcher_id_model_feature",
        },
    }
    payload.update(rest_context)
    return payload


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
        },
        "rest_policy": {
            "raw_rest_days_primary_driver": False,
            "rest_days_capped_max": 14,
            "buckets": list(REST_BUCKETS),
            "long_layoff_has_unbounded_positive_numeric_feature": False,
        },
        "leakage_policy": {
            "status": "ok" if all(row["leakage_policy_status"] == "ok_prior_games_only" for row in rows) else "violations",
            "rule": "Only Statcast rows with game_date before the starter official_date are eligible.",
        },
        "field_coverage": field_coverage,
        "top_correlations_by_season": correlations_by_season,
        "artifacts": {
            "pitcher_skill_features_path": feature_path,
        },
    }


def _render_feature_report_markdown(report: dict[str, Any]) -> str:
    rows = report["row_counts"]
    lines = [
        "# Pitcher Skill And Arsenal Feature Report",
        "",
        f"- Run ID: `{report['run_id']}`",
        (
            "- Date window: "
            f"`{report['date_window']['start_date']}` -> `{report['date_window']['end_date']}`"
        ),
        f"- Feature rows: `{rows['feature_rows']}`",
        f"- Pitch rows read: `{rows['pitch_rows']}`",
        f"- Pitchers: `{rows['pitchers']}`",
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
            "## Rest Policy",
            "",
            "Raw continuous rest days are not emitted as an unbounded model driver. "
            "`rest_days_capped` is capped at 14 and paired with explicit rest buckets, "
            "including `long_layoff` and `no_prior_start`.",
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
        "uv run python -m mlb_props_stack build-pitcher-skill-features "
        f"--start-date {start_date.isoformat()} --end-date {end_date.isoformat()} "
        f"--output-dir {output_dir}"
    )
    if dataset_run_dir is not None:
        command += f" --dataset-run-dir {dataset_run_dir}"
    return "\n".join(
        [
            "# Pitcher Skill Feature Reproducibility",
            "",
            f"- Run ID: `{run_id}`",
            "- Command:",
            "",
            "```bash",
            command,
            "```",
            "",
            "- Features use only raw Statcast pitch rows with `game_date` before the starter-game official date.",
            "- Same-game `starter_strikeouts` is used only for feature-report correlations.",
            "- Rest context is bucketed and capped instead of exposing raw continuous rest as a primary model driver.",
            "",
        ]
    )


def build_pitcher_skill_features(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str = "data",
    dataset_run_dir: Path | str | None = None,
    now: Any = utc_now,
) -> PitcherSkillFeatureBuildResult:
    """Build pitcher-centered skill and arsenal features from AGE-287 artifacts."""
    output_root = Path(output_dir)
    resolved_dataset_run_dir = _resolve_dataset_run_dir(
        output_dir=output_root,
        start_date=start_date,
        end_date=end_date,
        dataset_run_dir=None if dataset_run_dir is None else Path(dataset_run_dir),
    )
    dataset_path = resolved_dataset_run_dir / "starter_game_training_dataset.jsonl"
    source_manifest_path = resolved_dataset_run_dir / "source_manifest.jsonl"
    dataset_rows = _load_jsonl(dataset_path)
    if not dataset_rows:
        raise FileNotFoundError(f"No starter-game dataset rows found at {dataset_path}")
    pitch_rows = _load_pitch_rows_from_manifest(source_manifest_path)
    pitcher_rows = _pitcher_index(pitch_rows)
    pitcher_dates = _pitcher_date_index(pitcher_rows)
    target_dates = [
        date.fromisoformat(str(row["official_date"]))
        for row in dataset_rows
        if start_date <= date.fromisoformat(str(row["official_date"])) <= end_date
    ]
    league_priors_for_date = _league_priors_by_date(
        pitch_rows=pitch_rows,
        target_dates=target_dates,
    )
    run_root = (
        output_root
        / "normalized"
        / "pitcher_skill_features"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
    )
    run_id = _unique_timestamp_run_id(now().astimezone(UTC), run_root)
    normalized_root = run_root / f"run={run_id}"
    feature_path = normalized_root / "pitcher_skill_features.jsonl"
    feature_report_path = normalized_root / "feature_report.json"
    feature_report_markdown_path = normalized_root / "feature_report.md"
    reproducibility_notes_path = normalized_root / "reproducibility_notes.md"

    feature_rows: list[dict[str, Any]] = []
    targets_by_training_row_id: dict[str, int | float] = {}
    for starter_row in sorted(
        dataset_rows,
        key=lambda row: (row["official_date"], row["game_pk"], row["pitcher_id"]),
    ):
        row_date = date.fromisoformat(str(starter_row["official_date"]))
        if row_date < start_date or row_date > end_date:
            continue
        league_priors = league_priors_for_date[row_date]
        pitcher_id = int(starter_row["pitcher_id"])
        prior = _prior_rows(
            pitcher_rows=pitcher_rows.get(pitcher_id, []),
            pitcher_dates=pitcher_dates.get(pitcher_id, []),
            target_date=row_date,
        )
        feature = _feature_row(
            starter_row=starter_row,
            prior_rows=prior,
            league_priors=league_priors,
        )
        target = starter_row.get("starter_strikeouts")
        if isinstance(target, (int, float)):
            targets_by_training_row_id[str(starter_row["training_row_id"])] = target
        feature_rows.append(feature)

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
            dataset_run_dir=None if dataset_run_dir is None else Path(dataset_run_dir),
            run_id=run_id,
        ),
    )
    return PitcherSkillFeatureBuildResult(
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
