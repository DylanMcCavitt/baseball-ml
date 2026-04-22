"""Starter strikeout baseline training on top of AGE-146 feature artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, date, datetime, timedelta
import json
from math import exp, floor, log, sqrt
from pathlib import Path
from typing import Any, Callable, Iterable

from .ingest.mlb_stats_api import format_utc_timestamp, parse_api_datetime, utc_now
from .ingest.statcast_features import (
    StatcastSearchClient,
    build_statcast_search_csv_url,
    normalize_statcast_csv_text,
)

MODEL_VERSION = "starter-strikeout-baseline-v1"
BENCHMARK_NAME = "pitcher_k_rate_x_expected_leash_batters_faced"
COUNT_DISTRIBUTION_NAME = "negative_binomial_global_dispersion_v1"
COUNT_DISTRIBUTION_FIT_METHOD = "method_of_moments"
RIDGE_ALPHA = 1.0
OUTCOME_QUERY_PADDING_DAYS = 1
DISTRIBUTION_TAIL_TOLERANCE = 1e-9
MIN_DISTRIBUTION_MEAN = 1e-6
MIN_DISPERSION_ALPHA = 1e-6
MIN_PROBABILITY = 1e-12
MAX_DISTRIBUTION_SUPPORT = 250
BASE_NUMERIC_FEATURES = (
    "pitch_sample_size",
    "plate_appearance_sample_size",
    "pitcher_k_rate",
    "swinging_strike_rate",
    "csw_rate",
    "average_release_speed",
    "release_speed_delta_vs_baseline",
    "average_release_extension",
    "release_extension_delta_vs_baseline",
    "recent_batters_faced",
    "recent_pitch_count",
    "rest_days",
    "last_start_pitch_count",
    "last_start_batters_faced",
    "projected_lineup_k_rate",
    "projected_lineup_k_rate_vs_pitcher_hand",
    "projected_lineup_chase_rate",
    "projected_lineup_contact_rate",
    "lineup_size",
    "available_batter_feature_count",
    "lineup_continuity_count",
    "lineup_continuity_ratio",
    "expected_leash_pitch_count",
    "expected_leash_batters_faced",
    "lineup_is_confirmed",
)
CATEGORICAL_FEATURES = (
    "pitcher_feature_status",
    "lineup_status",
    "pitcher_hand",
    "home_away",
    "day_night",
    "double_header",
    "park_factor_status",
    "weather_status",
)
PROHIBITED_MODEL_FEATURE_FIELDS = frozenset(
    {
        "starter_strikeouts",
        "naive_benchmark_mean",
        "official_date",
        "game_pk",
        "pitcher_id",
        "pitcher_name",
        "team_abbreviation",
        "opponent_team_abbreviation",
        "pitcher_feature_row_id",
        "lineup_feature_row_id",
        "game_context_feature_row_id",
        "lineup_snapshot_id",
        "features_as_of",
        "training_row_id",
    }
)


@dataclass(frozen=True)
class StarterStrikeoutOutcomeRecord:
    """Observed starter strikeout total pulled from same-day Statcast rows."""

    outcome_id: str
    official_date: str
    game_pk: int
    pitcher_id: int
    pitcher_name: str | None
    captured_at: datetime
    source_url: str
    raw_path: Path
    pitch_row_count: int
    plate_appearance_count: int
    starter_strikeouts: int


@dataclass(frozen=True)
class StarterStrikeoutTrainingRow:
    """One date-keyed training row assembled from AGE-146 feature tables."""

    training_row_id: str
    official_date: str
    game_pk: int
    pitcher_id: int
    pitcher_name: str | None
    team_abbreviation: str
    opponent_team_abbreviation: str
    pitcher_feature_row_id: str
    lineup_feature_row_id: str
    game_context_feature_row_id: str
    lineup_snapshot_id: str | None
    features_as_of: datetime
    pitcher_feature_status: str
    lineup_status: str
    lineup_is_confirmed: float
    pitcher_hand: str | None
    home_away: str
    day_night: str
    double_header: str
    park_factor_status: str
    weather_status: str
    pitch_sample_size: int
    plate_appearance_sample_size: int
    pitcher_k_rate: float | None
    swinging_strike_rate: float | None
    csw_rate: float | None
    average_release_speed: float | None
    release_speed_delta_vs_baseline: float | None
    average_release_extension: float | None
    release_extension_delta_vs_baseline: float | None
    recent_batters_faced: int
    recent_pitch_count: int
    rest_days: int | None
    last_start_pitch_count: int | None
    last_start_batters_faced: int | None
    projected_lineup_k_rate: float | None
    projected_lineup_k_rate_vs_pitcher_hand: float | None
    projected_lineup_chase_rate: float | None
    projected_lineup_contact_rate: float | None
    lineup_size: int
    available_batter_feature_count: int
    lineup_continuity_count: int | None
    lineup_continuity_ratio: float | None
    expected_leash_pitch_count: float | None
    expected_leash_batters_faced: float | None
    pitch_type_usage: dict[str, float]
    naive_benchmark_mean: float
    starter_strikeouts: int


@dataclass(frozen=True)
class StarterStrikeoutBaselineTrainingResult:
    """Filesystem output summary for one starter strikeout baseline run."""

    start_date: date
    end_date: date
    run_id: str
    row_count: int
    outcome_count: int
    dispersion_alpha: float
    dataset_path: Path
    outcomes_path: Path
    date_splits_path: Path
    model_path: Path
    evaluation_path: Path
    ladder_probabilities_path: Path


@dataclass(frozen=True)
class _FeatureVectorizer:
    numeric_features: tuple[str, ...]
    numeric_means: dict[str, float]
    numeric_stds: dict[str, float]
    categorical_levels: dict[str, tuple[str, ...]]
    encoded_feature_names: tuple[str, ...]


@dataclass(frozen=True)
class _LinearModel:
    intercept: float
    coefficients: tuple[float, ...]
    vectorizer: _FeatureVectorizer


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, datetime):
        return format_utc_timestamp(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(_json_ready(payload), indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_ready(row), sort_keys=True))
            handle.write("\n")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _latest_feature_run_dir(root: Path) -> Path:
    run_dirs = sorted(
        (path for path in root.glob("run=*") if path.is_dir()),
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"No Statcast feature runs found under {root}. "
            "Run `uv run python -m mlb_props_stack ingest-statcast-features --date ...` first."
        )
    for run_dir in run_dirs:
        required_paths = (
            run_dir / "pitcher_daily_features.jsonl",
            run_dir / "lineup_daily_features.jsonl",
            run_dir / "game_context_features.jsonl",
        )
        if all(path.exists() for path in required_paths):
            return run_dir
    raise FileNotFoundError(f"Latest Statcast feature runs under {root} were incomplete.")


def _requested_dates(start_date: date, end_date: date) -> list[date]:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    dates: list[date] = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def _feature_key(row: dict[str, Any]) -> tuple[str, int, int]:
    pitcher_id = row.get("pitcher_id")
    if pitcher_id is None:
        raise ValueError("pitcher_id is required for starter strikeout training rows")
    return row["official_date"], int(row["game_pk"]), int(pitcher_id)


def _load_feature_rows_for_date(*, target_date: date, output_dir: Path) -> list[StarterStrikeoutTrainingRow]:
    run_dir = _latest_feature_run_dir(
        output_dir / "normalized" / "statcast_search" / f"date={target_date.isoformat()}"
    )
    pitcher_rows = _load_jsonl_rows(run_dir / "pitcher_daily_features.jsonl")
    lineup_rows = {
        _feature_key(row): row
        for row in _load_jsonl_rows(run_dir / "lineup_daily_features.jsonl")
        if row.get("pitcher_id") is not None
    }
    game_context_rows = {
        _feature_key(row): row
        for row in _load_jsonl_rows(run_dir / "game_context_features.jsonl")
        if row.get("pitcher_id") is not None
    }

    training_rows: list[StarterStrikeoutTrainingRow] = []
    for pitcher_row in pitcher_rows:
        pitcher_id = pitcher_row.get("pitcher_id")
        if pitcher_id is None:
            continue
        key = _feature_key(pitcher_row)
        lineup_row = lineup_rows.get(key)
        game_context_row = game_context_rows.get(key)
        if lineup_row is None or game_context_row is None:
            raise ValueError(
                "Expected matching lineup and game-context feature rows for "
                f"{pitcher_row['official_date']} game {pitcher_row['game_pk']} pitcher {pitcher_id}."
            )

        features_as_of = max(
            parse_api_datetime(pitcher_row["features_as_of"]),
            parse_api_datetime(lineup_row["features_as_of"]),
            parse_api_datetime(game_context_row["features_as_of"]),
        )
        naive_benchmark_mean = _naive_benchmark_mean(
            pitcher_k_rate=_as_optional_float(pitcher_row.get("pitcher_k_rate")),
            expected_leash_batters_faced=_as_optional_float(
                game_context_row.get("expected_leash_batters_faced")
            ),
            last_start_batters_faced=_as_optional_float(pitcher_row.get("last_start_batters_faced")),
        )
        training_rows.append(
            StarterStrikeoutTrainingRow(
                training_row_id=(
                    f"starter-training:{pitcher_row['official_date']}:{pitcher_row['game_pk']}:{pitcher_id}"
                ),
                official_date=str(pitcher_row["official_date"]),
                game_pk=int(pitcher_row["game_pk"]),
                pitcher_id=int(pitcher_id),
                pitcher_name=pitcher_row.get("pitcher_name"),
                team_abbreviation=str(pitcher_row["team_abbreviation"]),
                opponent_team_abbreviation=str(pitcher_row["opponent_team_abbreviation"]),
                pitcher_feature_row_id=str(pitcher_row["feature_row_id"]),
                lineup_feature_row_id=str(lineup_row["feature_row_id"]),
                game_context_feature_row_id=str(game_context_row["feature_row_id"]),
                lineup_snapshot_id=lineup_row.get("lineup_snapshot_id"),
                features_as_of=features_as_of,
                pitcher_feature_status=str(pitcher_row["feature_status"]),
                lineup_status=str(lineup_row["lineup_status"]),
                lineup_is_confirmed=1.0 if lineup_row["lineup_is_confirmed"] else 0.0,
                pitcher_hand=_as_optional_text(lineup_row.get("pitcher_hand")),
                home_away=str(game_context_row["home_away"]),
                day_night=str(game_context_row["day_night"]),
                double_header=str(game_context_row["double_header"]),
                park_factor_status=str(game_context_row["park_factor_status"]),
                weather_status=str(game_context_row["weather_status"]),
                pitch_sample_size=int(pitcher_row["pitch_sample_size"]),
                plate_appearance_sample_size=int(pitcher_row["plate_appearance_sample_size"]),
                pitcher_k_rate=_as_optional_float(pitcher_row.get("pitcher_k_rate")),
                swinging_strike_rate=_as_optional_float(pitcher_row.get("swinging_strike_rate")),
                csw_rate=_as_optional_float(pitcher_row.get("csw_rate")),
                average_release_speed=_as_optional_float(pitcher_row.get("average_release_speed")),
                release_speed_delta_vs_baseline=_as_optional_float(
                    pitcher_row.get("release_speed_delta_vs_baseline")
                ),
                average_release_extension=_as_optional_float(
                    pitcher_row.get("average_release_extension")
                ),
                release_extension_delta_vs_baseline=_as_optional_float(
                    pitcher_row.get("release_extension_delta_vs_baseline")
                ),
                recent_batters_faced=int(pitcher_row["recent_batters_faced"]),
                recent_pitch_count=int(pitcher_row["recent_pitch_count"]),
                rest_days=_as_optional_int(pitcher_row.get("rest_days")),
                last_start_pitch_count=_as_optional_int(pitcher_row.get("last_start_pitch_count")),
                last_start_batters_faced=_as_optional_int(
                    pitcher_row.get("last_start_batters_faced")
                ),
                projected_lineup_k_rate=_as_optional_float(lineup_row.get("projected_lineup_k_rate")),
                projected_lineup_k_rate_vs_pitcher_hand=_as_optional_float(
                    lineup_row.get("projected_lineup_k_rate_vs_pitcher_hand")
                ),
                projected_lineup_chase_rate=_as_optional_float(
                    lineup_row.get("projected_lineup_chase_rate")
                ),
                projected_lineup_contact_rate=_as_optional_float(
                    lineup_row.get("projected_lineup_contact_rate")
                ),
                lineup_size=int(lineup_row["lineup_size"]),
                available_batter_feature_count=int(lineup_row["available_batter_feature_count"]),
                lineup_continuity_count=_as_optional_int(
                    lineup_row.get("lineup_continuity_count")
                ),
                lineup_continuity_ratio=_as_optional_float(
                    lineup_row.get("lineup_continuity_ratio")
                ),
                expected_leash_pitch_count=_as_optional_float(
                    game_context_row.get("expected_leash_pitch_count")
                ),
                expected_leash_batters_faced=_as_optional_float(
                    game_context_row.get("expected_leash_batters_faced")
                ),
                pitch_type_usage={
                    str(key): float(value)
                    for key, value in (pitcher_row.get("pitch_type_usage") or {}).items()
                },
                naive_benchmark_mean=naive_benchmark_mean,
                starter_strikeouts=0,
            )
        )
    return training_rows


def _as_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _naive_benchmark_mean(
    *,
    pitcher_k_rate: float | None,
    expected_leash_batters_faced: float | None,
    last_start_batters_faced: float | None,
) -> float:
    k_rate = pitcher_k_rate or 0.0
    batter_volume = expected_leash_batters_faced
    if batter_volume is None:
        batter_volume = last_start_batters_faced
    if batter_volume is None:
        batter_volume = 0.0
    return round(max(0.0, k_rate * batter_volume), 6)


def _outcome_source_url(target_date: date, pitcher_id: int) -> str:
    # Baseball Savant uses gt/lt-style date filters; pad the request and then
    # filter the normalized rows back to the exact starter game locally.
    return build_statcast_search_csv_url(
        player_type="pitcher",
        player_id=pitcher_id,
        start_date=target_date - timedelta(days=OUTCOME_QUERY_PADDING_DAYS),
        end_date=target_date + timedelta(days=OUTCOME_QUERY_PADDING_DAYS),
    )


def _fetch_starter_outcome(
    *,
    row: StarterStrikeoutTrainingRow,
    output_dir: Path,
    client: StatcastSearchClient,
    now: Callable[[], datetime],
) -> StarterStrikeoutOutcomeRecord:
    official_date = date.fromisoformat(row.official_date)
    captured_at = now().astimezone(UTC)
    source_url = _outcome_source_url(official_date, row.pitcher_id)
    csv_text = client.fetch_csv(source_url)
    raw_path = (
        output_dir
        / "raw"
        / "statcast_search_outcomes"
        / f"date={row.official_date}"
        / f"player_id={row.pitcher_id}"
        / f"captured_at={captured_at.strftime('%Y%m%dT%H%M%SZ')}.csv"
    )
    _write_text(raw_path, csv_text)
    pull_id = f"starter-outcome:{row.pitcher_id}:{row.official_date}:{captured_at.strftime('%Y%m%dT%H%M%SZ')}"
    normalized_rows = normalize_statcast_csv_text(csv_text, pull_id=pull_id)
    matching_rows = [
        pitch_row
        for pitch_row in normalized_rows
        if pitch_row.game_pk == row.game_pk
        and pitch_row.pitcher_id == row.pitcher_id
        and pitch_row.game_date == row.official_date
    ]
    if not matching_rows:
        raise ValueError(
            "Could not derive a same-game starter outcome for "
            f"{row.pitcher_name or row.pitcher_id} on {row.official_date}."
        )
    final_pitch_rows = [pitch_row for pitch_row in matching_rows if pitch_row.is_plate_appearance_final_pitch]
    return StarterStrikeoutOutcomeRecord(
        outcome_id=f"starter-outcome:{row.official_date}:{row.game_pk}:{row.pitcher_id}",
        official_date=row.official_date,
        game_pk=row.game_pk,
        pitcher_id=row.pitcher_id,
        pitcher_name=row.pitcher_name,
        captured_at=captured_at,
        source_url=source_url,
        raw_path=raw_path,
        pitch_row_count=len(matching_rows),
        plate_appearance_count=len(final_pitch_rows),
        starter_strikeouts=sum(1 for pitch_row in final_pitch_rows if pitch_row.is_strikeout_event),
    )


def _attach_outcomes(
    *,
    rows: list[StarterStrikeoutTrainingRow],
    output_dir: Path,
    client: StatcastSearchClient,
    now: Callable[[], datetime],
) -> tuple[list[StarterStrikeoutTrainingRow], list[StarterStrikeoutOutcomeRecord]]:
    rows_with_outcomes: list[StarterStrikeoutTrainingRow] = []
    outcome_records: list[StarterStrikeoutOutcomeRecord] = []
    for row in rows:
        outcome = _fetch_starter_outcome(
            row=row,
            output_dir=output_dir,
            client=client,
            now=now,
        )
        outcome_records.append(outcome)
        rows_with_outcomes.append(
            StarterStrikeoutTrainingRow(
                **{
                    **asdict(row),
                    "starter_strikeouts": outcome.starter_strikeouts,
                }
            )
        )
    return rows_with_outcomes, outcome_records


def _split_dates(dates: list[str]) -> dict[str, list[str]]:
    unique_dates = sorted(set(dates))
    if len(unique_dates) < 3:
        raise ValueError("Need at least 3 distinct official dates for date-based train/validation/test splits.")
    test_count = max(1, len(unique_dates) // 5)
    validation_count = max(1, len(unique_dates) // 5)
    train_count = len(unique_dates) - validation_count - test_count
    if train_count < 1:
        train_count = 1
        validation_count = 1
        test_count = len(unique_dates) - 2
    return {
        "train": unique_dates[:train_count],
        "validation": unique_dates[train_count : train_count + validation_count],
        "test": unique_dates[train_count + validation_count :],
    }


def _feature_numeric_value(row: StarterStrikeoutTrainingRow, feature_name: str) -> float | None:
    if feature_name.startswith("pitch_type_usage:"):
        return row.pitch_type_usage.get(feature_name.removeprefix("pitch_type_usage:"), 0.0)
    value = getattr(row, feature_name)
    if value is None:
        return None
    return float(value)


def _feature_categorical_value(row: StarterStrikeoutTrainingRow, field_name: str) -> str:
    value = getattr(row, field_name)
    if value is None:
        return "missing"
    return str(value)


def _build_vectorizer(train_rows: list[StarterStrikeoutTrainingRow]) -> _FeatureVectorizer:
    pitch_type_features = tuple(
        f"pitch_type_usage:{pitch_type}"
        for pitch_type in sorted(
            {
                pitch_type
                for row in train_rows
                for pitch_type, value in row.pitch_type_usage.items()
                if value > 0.0
            }
        )
    )
    numeric_features = tuple((*BASE_NUMERIC_FEATURES, *pitch_type_features))
    numeric_means: dict[str, float] = {}
    numeric_stds: dict[str, float] = {}
    for feature_name in numeric_features:
        values = [
            value
            for row in train_rows
            if (value := _feature_numeric_value(row, feature_name)) is not None
        ]
        if not values:
            numeric_means[feature_name] = 0.0
            numeric_stds[feature_name] = 1.0
            continue
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        std_value = sqrt(variance) or 1.0
        numeric_means[feature_name] = mean_value
        numeric_stds[feature_name] = std_value

    categorical_levels = {
        field_name: tuple(
            sorted({_feature_categorical_value(row, field_name) for row in train_rows})
        )
        for field_name in CATEGORICAL_FEATURES
    }
    encoded_feature_names = tuple(numeric_features) + tuple(
        f"{field_name}={level}"
        for field_name in CATEGORICAL_FEATURES
        for level in categorical_levels[field_name]
    )
    leaked_fields = PROHIBITED_MODEL_FEATURE_FIELDS & set(encoded_feature_names)
    if leaked_fields:
        raise ValueError(f"Training matrix included prohibited fields: {sorted(leaked_fields)}")
    return _FeatureVectorizer(
        numeric_features=numeric_features,
        numeric_means=numeric_means,
        numeric_stds=numeric_stds,
        categorical_levels=categorical_levels,
        encoded_feature_names=encoded_feature_names,
    )


def _encode_row(row: StarterStrikeoutTrainingRow, vectorizer: _FeatureVectorizer) -> list[float]:
    encoded: list[float] = []
    for feature_name in vectorizer.numeric_features:
        raw_value = _feature_numeric_value(row, feature_name)
        mean_value = vectorizer.numeric_means[feature_name]
        std_value = vectorizer.numeric_stds[feature_name]
        value = mean_value if raw_value is None else raw_value
        encoded.append((value - mean_value) / std_value if std_value else 0.0)
    for field_name in CATEGORICAL_FEATURES:
        row_value = _feature_categorical_value(row, field_name)
        for level in vectorizer.categorical_levels[field_name]:
            encoded.append(1.0 if row_value == level else 0.0)
    return encoded


def _fit_ridge_regression(train_rows: list[StarterStrikeoutTrainingRow]) -> _LinearModel:
    vectorizer = _build_vectorizer(train_rows)
    encoded_rows = [_encode_row(row, vectorizer) for row in train_rows]
    feature_count = len(vectorizer.encoded_feature_names)
    matrix_size = feature_count + 1
    xtx = [[0.0 for _ in range(matrix_size)] for _ in range(matrix_size)]
    xty = [0.0 for _ in range(matrix_size)]

    for row, encoded in zip(train_rows, encoded_rows):
        design_row = [1.0, *encoded]
        target = float(row.starter_strikeouts)
        for index_i in range(matrix_size):
            xty[index_i] += design_row[index_i] * target
            for index_j in range(matrix_size):
                xtx[index_i][index_j] += design_row[index_i] * design_row[index_j]

    for diagonal_index in range(1, matrix_size):
        xtx[diagonal_index][diagonal_index] += RIDGE_ALPHA

    coefficients = _solve_linear_system(xtx, xty)
    return _LinearModel(
        intercept=coefficients[0],
        coefficients=tuple(coefficients[1:]),
        vectorizer=vectorizer,
    )


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    size = len(augmented)
    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda row_index: abs(augmented[row_index][pivot_index]))
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            raise ValueError("Could not fit ridge baseline because the design matrix was singular.")
        augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]
        pivot_value = augmented[pivot_index][pivot_index]
        for column_index in range(pivot_index, size + 1):
            augmented[pivot_index][column_index] /= pivot_value
        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            if factor == 0.0:
                continue
            for column_index in range(pivot_index, size + 1):
                augmented[row_index][column_index] -= factor * augmented[pivot_index][column_index]
    return [augmented[row_index][size] for row_index in range(size)]


def _predict_mean(row: StarterStrikeoutTrainingRow, model: _LinearModel) -> float:
    encoded = _encode_row(row, model.vectorizer)
    mean_prediction = model.intercept + sum(
        coefficient * feature_value
        for coefficient, feature_value in zip(model.coefficients, encoded)
    )
    return max(0.0, mean_prediction)


def _normalized_count_distribution(
    mean: float,
    dispersion_alpha: float,
    *,
    minimum_count: int = 0,
    tail_tolerance: float = DISTRIBUTION_TAIL_TOLERANCE,
) -> list[float]:
    if mean < 0.0:
        raise ValueError("mean must be >= 0.0")
    if dispersion_alpha < 0.0:
        raise ValueError("dispersion_alpha must be >= 0.0")
    if minimum_count < 0:
        raise ValueError("minimum_count must be >= 0")

    if mean == 0.0:
        return [1.0, *([0.0] * minimum_count)]

    effective_mean = max(MIN_DISTRIBUTION_MEAN, mean)
    effective_dispersion_alpha = (
        0.0 if dispersion_alpha <= MIN_DISPERSION_ALPHA else dispersion_alpha
    )
    probabilities: list[float]
    cumulative: float
    if effective_dispersion_alpha == 0.0:
        probabilities = [exp(-effective_mean)]
        cumulative = probabilities[0]
        while cumulative < 1.0 - tail_tolerance or len(probabilities) - 1 < minimum_count:
            count = len(probabilities)
            probabilities.append(probabilities[-1] * effective_mean / count)
            cumulative += probabilities[-1]
            if len(probabilities) > MAX_DISTRIBUTION_SUPPORT:
                raise ValueError("Poisson support exceeded the configured maximum support.")
    else:
        size = 1.0 / effective_dispersion_alpha
        failure_probability = effective_mean / (size + effective_mean)
        success_probability = size / (size + effective_mean)
        probabilities = [success_probability**size]
        cumulative = probabilities[0]
        while cumulative < 1.0 - tail_tolerance or len(probabilities) - 1 < minimum_count:
            count = len(probabilities)
            previous_probability = probabilities[-1]
            probabilities.append(
                previous_probability
                * ((count - 1 + size) / count)
                * failure_probability
            )
            cumulative += probabilities[-1]
            if len(probabilities) > MAX_DISTRIBUTION_SUPPORT:
                raise ValueError(
                    "Negative-binomial support exceeded the configured maximum support."
                )

    total_probability = sum(probabilities)
    if total_probability <= 0.0:
        raise ValueError("Count distribution total probability must be positive.")
    return [probability / total_probability for probability in probabilities]


def starter_strikeout_line_probability(
    *,
    mean: float,
    line: float,
    dispersion_alpha: float,
    tail_tolerance: float = DISTRIBUTION_TAIL_TOLERANCE,
) -> tuple[float, float]:
    """Return over/under probabilities for one strikeout line."""
    if line < 0.0:
        raise ValueError("line must be >= 0.0")
    threshold = int(floor(line)) + 1
    probabilities = _normalized_count_distribution(
        mean,
        dispersion_alpha,
        minimum_count=threshold,
        tail_tolerance=tail_tolerance,
    )
    under_probability = sum(probabilities[:threshold])
    over_probability = max(0.0, 1.0 - under_probability)
    return over_probability, under_probability


def starter_strikeout_ladder_probabilities(
    *,
    mean: float,
    dispersion_alpha: float,
    tail_tolerance: float = DISTRIBUTION_TAIL_TOLERANCE,
) -> list[dict[str, float]]:
    """Return ladder probabilities for half-strikeout lines."""
    probabilities = _normalized_count_distribution(
        mean,
        dispersion_alpha,
        tail_tolerance=tail_tolerance,
    )
    ladder_rows: list[dict[str, float]] = []
    cumulative_probability = 0.0
    for strikeout_count, exact_probability in enumerate(probabilities):
        cumulative_probability += exact_probability
        ladder_rows.append(
            {
                "line": round(strikeout_count + 0.5, 6),
                "over_probability": round(max(0.0, 1.0 - cumulative_probability), 6),
                "under_probability": round(cumulative_probability, 6),
            }
        )
    return ladder_rows


def _fit_negative_binomial_dispersion_alpha(
    rows: list[StarterStrikeoutTrainingRow],
    mean_predictions: list[float],
) -> float:
    if len(rows) != len(mean_predictions):
        raise ValueError("rows and mean_predictions must be aligned")
    denominator = sum(max(MIN_DISTRIBUTION_MEAN, mean_prediction) ** 2 for mean_prediction in mean_predictions)
    if denominator == 0.0:
        return 0.0
    numerator = sum(
        (float(row.starter_strikeouts) - max(MIN_DISTRIBUTION_MEAN, mean_prediction)) ** 2
        - max(MIN_DISTRIBUTION_MEAN, mean_prediction)
        for row, mean_prediction in zip(rows, mean_predictions)
    )
    fitted_alpha = max(0.0, numerator / denominator)
    return 0.0 if fitted_alpha <= MIN_DISPERSION_ALPHA else fitted_alpha


def _ranked_probability_score(actual_count: int, probabilities: list[float]) -> float:
    cumulative_probability = 0.0
    score = 0.0
    for strikeout_count, probability in enumerate(probabilities):
        cumulative_probability += probability
        observed_cdf = 1.0 if strikeout_count >= actual_count else 0.0
        difference = cumulative_probability - observed_cdf
        score += difference * difference
    return score / max(1, len(probabilities) - 1)


def _distribution_metrics(
    rows: list[StarterStrikeoutTrainingRow],
    mean_predictions: list[float],
    *,
    dispersion_alpha: float,
) -> dict[str, Any]:
    if len(rows) != len(mean_predictions):
        raise ValueError("rows and mean_predictions must be aligned")
    negative_log_likelihoods: list[float] = []
    ranked_probability_scores: list[float] = []
    for row, mean_prediction in zip(rows, mean_predictions):
        probabilities = _normalized_count_distribution(
            mean_prediction,
            dispersion_alpha,
            minimum_count=row.starter_strikeouts,
        )
        exact_probability = probabilities[row.starter_strikeouts]
        negative_log_likelihoods.append(-log(max(exact_probability, MIN_PROBABILITY)))
        ranked_probability_scores.append(
            _ranked_probability_score(row.starter_strikeouts, probabilities)
        )
    return {
        "mean_negative_log_likelihood": round(
            sum(negative_log_likelihoods) / len(negative_log_likelihoods),
            6,
        ),
        "mean_ranked_probability_score": round(
            sum(ranked_probability_scores) / len(ranked_probability_scores),
            6,
        ),
    }


def _rmse(actuals: list[float], predictions: list[float]) -> float:
    return round(sqrt(sum((actual - prediction) ** 2 for actual, prediction in zip(actuals, predictions)) / len(actuals)), 6)


def _mae(actuals: list[float], predictions: list[float]) -> float:
    return round(sum(abs(actual - prediction) for actual, prediction in zip(actuals, predictions)) / len(actuals), 6)


def _pearson_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((left_value - left_mean) * (right_value - right_mean) for left_value, right_value in zip(left, right))
    left_variance = sum((left_value - left_mean) ** 2 for left_value in left)
    right_variance = sum((right_value - right_mean) ** 2 for right_value in right)
    denominator = sqrt(left_variance * right_variance)
    if denominator == 0.0:
        return None
    return numerator / denominator


def _average_ranks(values: list[float]) -> list[float]:
    sorted_indices = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0 for _ in values]
    position = 1
    cursor = 0
    while cursor < len(sorted_indices):
        end = cursor + 1
        while end < len(sorted_indices) and values[sorted_indices[end]] == values[sorted_indices[cursor]]:
            end += 1
        average_rank = (position + (position + end - cursor - 1)) / 2.0
        for index in sorted_indices[cursor:end]:
            ranks[index] = average_rank
        position += end - cursor
        cursor = end
    return ranks


def _spearman_rank_correlation(actuals: list[float], predictions: list[float]) -> float | None:
    if len(actuals) < 2:
        return None
    return _pearson_correlation(_average_ranks(actuals), _average_ranks(predictions))


def _metrics(actuals: list[float], predictions: list[float]) -> dict[str, Any]:
    return {
        "rmse": _rmse(actuals, predictions),
        "mae": _mae(actuals, predictions),
        "spearman_rank_correlation": (
            None
            if (value := _spearman_rank_correlation(actuals, predictions)) is None
            else round(value, 6)
        ),
    }


def _rows_for_dates(rows: list[StarterStrikeoutTrainingRow], dates: list[str]) -> list[StarterStrikeoutTrainingRow]:
    date_set = set(dates)
    return [row for row in rows if row.official_date in date_set]


def _feature_importance(model: _LinearModel) -> list[dict[str, float | str]]:
    importances = [
        {
            "feature": feature_name,
            "coefficient": round(coefficient, 6),
            "absolute_importance": round(abs(coefficient), 6),
        }
        for feature_name, coefficient in zip(
            model.vectorizer.encoded_feature_names,
            model.coefficients,
        )
    ]
    return sorted(importances, key=lambda item: (-float(item["absolute_importance"]), str(item["feature"])))


def _model_artifact(model: _LinearModel, *, dispersion_alpha: float) -> dict[str, Any]:
    return {
        "model_version": MODEL_VERSION,
        "ridge_alpha": RIDGE_ALPHA,
        "count_distribution": {
            "name": COUNT_DISTRIBUTION_NAME,
            "fit_method": COUNT_DISTRIBUTION_FIT_METHOD,
            "dispersion_alpha": round(dispersion_alpha, 6),
            "variance_formula": "variance = mean + alpha * mean^2",
        },
        "intercept": round(model.intercept, 6),
        "encoded_feature_names": list(model.vectorizer.encoded_feature_names),
        "coefficients": {
            feature_name: round(coefficient, 6)
            for feature_name, coefficient in zip(
                model.vectorizer.encoded_feature_names,
                model.coefficients,
            )
        },
        "numeric_feature_stats": {
            feature_name: {
                "mean": round(model.vectorizer.numeric_means[feature_name], 6),
                "std": round(model.vectorizer.numeric_stds[feature_name], 6),
            }
            for feature_name in model.vectorizer.numeric_features
        },
        "categorical_feature_levels": {
            feature_name: list(levels)
            for feature_name, levels in model.vectorizer.categorical_levels.items()
        },
    }


def train_starter_strikeout_baseline(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str = "data",
    client: StatcastSearchClient | None = None,
    now: Callable[[], datetime] = utc_now,
) -> StarterStrikeoutBaselineTrainingResult:
    """Train the first deterministic starter strikeout baseline model."""
    if client is None:
        client = StatcastSearchClient()

    output_root = Path(output_dir)
    feature_rows: list[StarterStrikeoutTrainingRow] = []
    for target_date in _requested_dates(start_date, end_date):
        date_root = output_root / "normalized" / "statcast_search" / f"date={target_date.isoformat()}"
        if not date_root.exists():
            continue
        feature_rows.extend(
            _load_feature_rows_for_date(
                target_date=target_date,
                output_dir=output_root,
            )
        )
    if not feature_rows:
        raise FileNotFoundError(
            "No AGE-146 Statcast feature runs were found inside the requested date range."
        )

    rows_with_outcomes, outcome_records = _attach_outcomes(
        rows=sorted(feature_rows, key=lambda row: (row.official_date, row.game_pk, row.pitcher_id)),
        output_dir=output_root,
        client=client,
        now=now,
    )
    date_splits = _split_dates([row.official_date for row in rows_with_outcomes])
    train_rows = _rows_for_dates(rows_with_outcomes, date_splits["train"])
    validation_rows = _rows_for_dates(rows_with_outcomes, date_splits["validation"])
    test_rows = _rows_for_dates(rows_with_outcomes, date_splits["test"])
    held_out_rows = [*validation_rows, *test_rows]
    if not train_rows or not validation_rows or not test_rows:
        raise ValueError("Date splits must leave at least one row in train, validation, and test.")

    model = _fit_ridge_regression(train_rows)
    train_model_predictions = [_predict_mean(row, model) for row in train_rows]
    dispersion_alpha = _fit_negative_binomial_dispersion_alpha(
        train_rows,
        train_model_predictions,
    )
    evaluation_rows = {
        "train": train_rows,
        "validation": validation_rows,
        "test": test_rows,
        "held_out": held_out_rows,
    }
    evaluation: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "benchmark_name": BENCHMARK_NAME,
        "date_splits": date_splits,
        "row_counts": {split_name: len(split_rows) for split_name, split_rows in evaluation_rows.items()},
        "feature_importance": _feature_importance(model),
        "feature_schema": {
            "encoded_feature_names": list(model.vectorizer.encoded_feature_names),
            "prohibited_fields_checked": sorted(PROHIBITED_MODEL_FEATURE_FIELDS),
        },
        "benchmark": {},
        "model": {},
        "count_distribution": {
            "name": COUNT_DISTRIBUTION_NAME,
            "fit_method": COUNT_DISTRIBUTION_FIT_METHOD,
            "dispersion_alpha": round(dispersion_alpha, 6),
            "variance_formula": "variance = mean + alpha * mean^2",
            "poisson": {},
            "negative_binomial": {},
        },
    }
    for split_name, split_rows in evaluation_rows.items():
        actuals = [float(row.starter_strikeouts) for row in split_rows]
        benchmark_predictions = [row.naive_benchmark_mean for row in split_rows]
        model_predictions = [_predict_mean(row, model) for row in split_rows]
        evaluation["benchmark"][split_name] = _metrics(actuals, benchmark_predictions)
        evaluation["model"][split_name] = _metrics(actuals, model_predictions)
        evaluation["count_distribution"]["poisson"][split_name] = _distribution_metrics(
            split_rows,
            model_predictions,
            dispersion_alpha=0.0,
        )
        evaluation["count_distribution"]["negative_binomial"][split_name] = _distribution_metrics(
            split_rows,
            model_predictions,
            dispersion_alpha=dispersion_alpha,
        )
    evaluation["held_out_beats_benchmark"] = {
        "rmse": evaluation["model"]["held_out"]["rmse"] < evaluation["benchmark"]["held_out"]["rmse"],
        "mae": evaluation["model"]["held_out"]["mae"] <= evaluation["benchmark"]["held_out"]["mae"],
    }
    evaluation["count_distribution"]["held_out_beats_poisson"] = {
        "mean_negative_log_likelihood": (
            evaluation["count_distribution"]["negative_binomial"]["held_out"][
                "mean_negative_log_likelihood"
            ]
            < evaluation["count_distribution"]["poisson"]["held_out"][
                "mean_negative_log_likelihood"
            ]
        ),
        "mean_ranked_probability_score": (
            evaluation["count_distribution"]["negative_binomial"]["held_out"][
                "mean_ranked_probability_score"
            ]
            <= evaluation["count_distribution"]["poisson"]["held_out"][
                "mean_ranked_probability_score"
            ]
        ),
    }

    run_id = now().astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    normalized_root = (
        output_root
        / "normalized"
        / "starter_strikeout_baseline"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
        / f"run={run_id}"
    )
    dataset_path = normalized_root / "training_dataset.jsonl"
    outcomes_path = normalized_root / "starter_outcomes.jsonl"
    date_splits_path = normalized_root / "date_splits.json"
    model_path = normalized_root / "baseline_model.json"
    evaluation_path = normalized_root / "evaluation.json"
    ladder_probabilities_path = normalized_root / "ladder_probabilities.jsonl"
    _write_jsonl(dataset_path, rows_with_outcomes)
    _write_jsonl(outcomes_path, outcome_records)
    _write_json(date_splits_path, date_splits)
    _write_json(model_path, _model_artifact(model, dispersion_alpha=dispersion_alpha))
    _write_json(evaluation_path, evaluation)
    split_by_date = {
        split_date: split_name
        for split_name, split_dates in date_splits.items()
        for split_date in split_dates
    }
    _write_jsonl(
        ladder_probabilities_path,
        [
            {
                "training_row_id": row.training_row_id,
                "official_date": row.official_date,
                "game_pk": row.game_pk,
                "pitcher_id": row.pitcher_id,
                "pitcher_name": row.pitcher_name,
                "split": split_by_date[row.official_date],
                "actual_strikeouts": row.starter_strikeouts,
                "naive_benchmark_mean": round(row.naive_benchmark_mean, 6),
                "model_mean": round(_predict_mean(row, model), 6),
                "count_distribution": {
                    "name": COUNT_DISTRIBUTION_NAME,
                    "dispersion_alpha": round(dispersion_alpha, 6),
                },
                "ladder_probabilities": starter_strikeout_ladder_probabilities(
                    mean=_predict_mean(row, model),
                    dispersion_alpha=dispersion_alpha,
                ),
            }
            for row in rows_with_outcomes
        ],
    )
    return StarterStrikeoutBaselineTrainingResult(
        start_date=start_date,
        end_date=end_date,
        run_id=run_id,
        row_count=len(rows_with_outcomes),
        outcome_count=len(outcome_records),
        dispersion_alpha=round(dispersion_alpha, 6),
        dataset_path=dataset_path,
        outcomes_path=outcomes_path,
        date_splits_path=date_splits_path,
        model_path=model_path,
        evaluation_path=evaluation_path,
        ladder_probabilities_path=ladder_probabilities_path,
    )
