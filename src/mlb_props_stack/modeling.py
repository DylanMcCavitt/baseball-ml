"""Starter strikeout baseline training on top of AGE-146 feature artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, date, datetime, timedelta
import json
from math import exp, floor, log, sqrt
from pathlib import Path
from shlex import quote
from typing import Any, Callable, Iterable

from .config import StackConfig
from .ingest.mlb_stats_api import format_utc_timestamp, parse_api_datetime, utc_now
from .ingest.statcast_features import (
    StatcastSearchClient,
    build_statcast_search_csv_url,
    normalize_statcast_csv_text,
)
from .tracking import (
    TrackingConfig,
    log_run_artifact,
    log_run_metrics,
    log_run_params,
    start_experiment_run,
)

MODEL_VERSION = "starter-strikeout-baseline-v1"
BENCHMARK_NAME = "pitcher_k_rate_x_expected_leash_batters_faced"
COUNT_DISTRIBUTION_NAME = "negative_binomial_global_dispersion_v1"
COUNT_DISTRIBUTION_FIT_METHOD = "method_of_moments"
PROBABILITY_CALIBRATOR_NAME = "isotonic_ladder_probability_calibrator_v1"
PROBABILITY_CALIBRATOR_SOURCE = "out_of_fold_ladder_events"
RIDGE_ALPHA = 10.0
OUTCOME_QUERY_PADDING_DAYS = 1
DISTRIBUTION_TAIL_TOLERANCE = 1e-9
MIN_DISTRIBUTION_MEAN = 1e-6
MIN_DISPERSION_ALPHA = 1e-6
MIN_PROBABILITY = 1e-12
MAX_DISTRIBUTION_SUPPORT = 250
OOF_MIN_TRAIN_DATES = 2
RELIABILITY_BIN_COUNT = 10
OPTIONAL_FEATURE_MIN_COVERAGE = 0.75
MIN_FEATURE_VARIANCE = 1e-9
CORE_NUMERIC_FEATURES = (
    "pitch_sample_size",
    "plate_appearance_sample_size",
    "pitcher_k_rate",
    "swinging_strike_rate",
    "csw_rate",
    "recent_batters_faced",
    "recent_pitch_count",
    "rest_days",
    "last_start_batters_faced",
    "expected_leash_batters_faced",
    "lineup_is_confirmed",
)
OPTIONAL_NUMERIC_FEATURES = (
    "pitcher_k_rate_vs_rhh",
    "pitcher_k_rate_vs_lhh",
    "pitcher_whiff_rate_vs_rhh",
    "pitcher_whiff_rate_vs_lhh",
    "projected_lineup_k_rate",
    "projected_lineup_k_rate_vs_pitcher_hand",
    "lineup_k_rate_vs_rhp",
    "lineup_k_rate_vs_lhp",
    "projected_lineup_chase_rate",
    "projected_lineup_contact_rate",
    "lineup_continuity_ratio",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ()
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
    pitcher_k_rate_vs_rhh: float | None
    pitcher_k_rate_vs_lhh: float | None
    swinging_strike_rate: float | None
    pitcher_whiff_rate_vs_rhh: float | None
    pitcher_whiff_rate_vs_lhh: float | None
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
    lineup_k_rate_vs_rhp: float | None
    lineup_k_rate_vs_lhp: float | None
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
    mlflow_run_id: str
    mlflow_experiment_name: str
    row_count: int
    outcome_count: int
    dispersion_alpha: float
    dataset_path: Path
    outcomes_path: Path
    date_splits_path: Path
    model_path: Path
    evaluation_path: Path
    ladder_probabilities_path: Path
    probability_calibrator_path: Path
    raw_vs_calibrated_path: Path
    calibration_summary_path: Path
    evaluation_summary_path: Path
    evaluation_summary_markdown_path: Path
    reproducibility_notes_path: Path
    held_out_status: str
    held_out_model_rmse: float | None
    held_out_benchmark_rmse: float | None
    held_out_model_mae: float | None
    held_out_benchmark_mae: float | None
    previous_run_id: str | None


@dataclass(frozen=True)
class StarterStrikeoutInferenceResult:
    """Filesystem output summary for one target-date inference run."""

    target_date: date
    run_id: str
    source_model_run_id: str
    source_model_path: Path
    feature_run_dir: Path
    model_path: Path
    ladder_probabilities_path: Path
    projection_count: int


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


@dataclass(frozen=True)
class _ProbabilityCalibratorBucket:
    raw_probability_min: float
    raw_probability_max: float
    calibrated_probability: float
    sample_count: int
    positive_count: int


@dataclass(frozen=True)
class _ProbabilityCalibrator:
    name: str
    source: str
    configured_min_sample: int
    sample_count: int
    fitted_from_date: str | None
    fitted_through_date: str | None
    is_identity: bool
    reason: str | None
    sample_warning: str | None
    buckets: tuple[_ProbabilityCalibratorBucket, ...]


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _path_run_id(run_dir: Path) -> str:
    return run_dir.name.split("=", 1)[-1]


def _clip_probability(probability: float) -> float:
    if probability <= 0.0:
        return MIN_PROBABILITY
    if probability >= 1.0:
        return 1.0 - MIN_PROBABILITY
    return probability


def _identity_probability_calibrator(
    *,
    configured_min_sample: int,
    reason: str,
    sample_count: int,
    fitted_from_date: str | None,
    fitted_through_date: str | None,
) -> _ProbabilityCalibrator:
    sample_warning = None
    if sample_count < configured_min_sample:
        sample_warning = (
            f"sample_count {sample_count} is below configured minimum "
            f"{configured_min_sample}; using identity calibration."
        )
    return _ProbabilityCalibrator(
        name=PROBABILITY_CALIBRATOR_NAME,
        source=PROBABILITY_CALIBRATOR_SOURCE,
        configured_min_sample=configured_min_sample,
        sample_count=sample_count,
        fitted_from_date=fitted_from_date,
        fitted_through_date=fitted_through_date,
        is_identity=True,
        reason=reason,
        sample_warning=sample_warning,
        buckets=(),
    )


def _coerce_probability_calibrator(
    calibrator: _ProbabilityCalibrator | dict[str, Any],
) -> _ProbabilityCalibrator:
    if isinstance(calibrator, _ProbabilityCalibrator):
        return calibrator
    buckets = tuple(
        _ProbabilityCalibratorBucket(
            raw_probability_min=float(bucket["raw_probability_min"]),
            raw_probability_max=float(bucket["raw_probability_max"]),
            calibrated_probability=float(bucket["calibrated_probability"]),
            sample_count=int(bucket["sample_count"]),
            positive_count=int(bucket["positive_count"]),
        )
        for bucket in calibrator.get("buckets", [])
    )
    return _ProbabilityCalibrator(
        name=str(calibrator["name"]),
        source=str(calibrator["source"]),
        configured_min_sample=int(calibrator["configured_min_sample"]),
        sample_count=int(calibrator["sample_count"]),
        fitted_from_date=calibrator.get("fitted_from_date"),
        fitted_through_date=calibrator.get("fitted_through_date"),
        is_identity=bool(calibrator["is_identity"]),
        reason=calibrator.get("reason"),
        sample_warning=calibrator.get("sample_warning"),
        buckets=buckets,
    )


def _fit_probability_calibrator(
    *,
    probabilities: list[float],
    outcomes: list[int],
    configured_min_sample: int,
    fitted_from_date: str | None,
    fitted_through_date: str | None,
) -> _ProbabilityCalibrator:
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must be aligned")
    if not probabilities:
        return _identity_probability_calibrator(
            configured_min_sample=configured_min_sample,
            reason="no_prior_out_of_fold_probability_rows",
            sample_count=0,
            fitted_from_date=fitted_from_date,
            fitted_through_date=fitted_through_date,
        )

    blocks: list[dict[str, float | int]] = []
    for raw_probability, outcome in sorted(zip(probabilities, outcomes), key=lambda item: item[0]):
        clipped_probability = _clip_probability(raw_probability)
        if blocks and clipped_probability == blocks[-1]["raw_probability_min"]:
            blocks[-1]["sample_count"] = int(blocks[-1]["sample_count"]) + 1
            blocks[-1]["positive_count"] = int(blocks[-1]["positive_count"]) + outcome
            continue
        blocks.append(
            {
                "raw_probability_min": clipped_probability,
                "raw_probability_max": clipped_probability,
                "sample_count": 1,
                "positive_count": outcome,
            }
        )

    pooled_blocks: list[dict[str, float | int]] = []
    for block in blocks:
        pooled_blocks.append(block)
        while len(pooled_blocks) >= 2:
            previous = pooled_blocks[-2]
            current = pooled_blocks[-1]
            previous_mean = int(previous["positive_count"]) / int(previous["sample_count"])
            current_mean = int(current["positive_count"]) / int(current["sample_count"])
            if previous_mean <= current_mean:
                break
            pooled_blocks[-2] = {
                "raw_probability_min": previous["raw_probability_min"],
                "raw_probability_max": current["raw_probability_max"],
                "sample_count": int(previous["sample_count"]) + int(current["sample_count"]),
                "positive_count": int(previous["positive_count"])
                + int(current["positive_count"]),
            }
            pooled_blocks.pop()

    buckets = tuple(
        _ProbabilityCalibratorBucket(
            raw_probability_min=float(block["raw_probability_min"]),
            raw_probability_max=float(block["raw_probability_max"]),
            calibrated_probability=round(
                int(block["positive_count"]) / int(block["sample_count"]),
                6,
            ),
            sample_count=int(block["sample_count"]),
            positive_count=int(block["positive_count"]),
        )
        for block in pooled_blocks
    )
    sample_warning = None
    if len(probabilities) < configured_min_sample:
        sample_warning = (
            f"sample_count {len(probabilities)} is below configured minimum "
            f"{configured_min_sample}; retain calibration as a bootstrap diagnostic."
        )
    return _ProbabilityCalibrator(
        name=PROBABILITY_CALIBRATOR_NAME,
        source=PROBABILITY_CALIBRATOR_SOURCE,
        configured_min_sample=configured_min_sample,
        sample_count=len(probabilities),
        fitted_from_date=fitted_from_date,
        fitted_through_date=fitted_through_date,
        is_identity=False,
        reason=None,
        sample_warning=sample_warning,
        buckets=buckets,
    )


def _apply_probability_calibrator(
    probability: float,
    calibrator: _ProbabilityCalibrator | dict[str, Any],
) -> float:
    typed_calibrator = _coerce_probability_calibrator(calibrator)
    clipped_probability = _clip_probability(probability)
    if typed_calibrator.is_identity or not typed_calibrator.buckets:
        return clipped_probability
    for bucket in typed_calibrator.buckets:
        if clipped_probability <= bucket.raw_probability_max:
            return bucket.calibrated_probability
    return typed_calibrator.buckets[-1].calibrated_probability


def calibrate_starter_strikeout_ladder_probabilities(
    ladder_probabilities: list[dict[str, float]],
    calibrator: _ProbabilityCalibrator | dict[str, Any],
) -> list[dict[str, float]]:
    """Apply a probability calibrator to a raw strikeout ladder."""
    typed_calibrator = _coerce_probability_calibrator(calibrator)
    calibrated_rows: list[dict[str, float]] = []
    previous_over_probability = 1.0
    for row in ladder_probabilities:
        calibrated_over_probability = min(
            previous_over_probability,
            _apply_probability_calibrator(row["over_probability"], typed_calibrator),
        )
        calibrated_under_probability = 1.0 - calibrated_over_probability
        calibrated_rows.append(
            {
                "line": round(float(row["line"]), 6),
                "over_probability": round(calibrated_over_probability, 6),
                "under_probability": round(calibrated_under_probability, 6),
            }
        )
        previous_over_probability = calibrated_over_probability
    return calibrated_rows


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
                pitcher_k_rate_vs_rhh=_as_optional_float(
                    pitcher_row.get("pitcher_k_rate_vs_rhh")
                ),
                pitcher_k_rate_vs_lhh=_as_optional_float(
                    pitcher_row.get("pitcher_k_rate_vs_lhh")
                ),
                swinging_strike_rate=_as_optional_float(pitcher_row.get("swinging_strike_rate")),
                pitcher_whiff_rate_vs_rhh=_as_optional_float(
                    pitcher_row.get("pitcher_whiff_rate_vs_rhh")
                ),
                pitcher_whiff_rate_vs_lhh=_as_optional_float(
                    pitcher_row.get("pitcher_whiff_rate_vs_lhh")
                ),
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
                lineup_k_rate_vs_rhp=_as_optional_float(
                    lineup_row.get("lineup_k_rate_vs_rhp")
                ),
                lineup_k_rate_vs_lhp=_as_optional_float(
                    lineup_row.get("lineup_k_rate_vs_lhp")
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


def _is_missing_outcome_error(error: ValueError) -> bool:
    return "Could not derive a same-game starter outcome" in str(error)


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
        try:
            outcome = _fetch_starter_outcome(
                row=row,
                output_dir=output_dir,
                client=client,
                now=now,
            )
        except ValueError as error:
            if _is_missing_outcome_error(error):
                continue
            raise
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


def _selected_numeric_features(train_rows: list[StarterStrikeoutTrainingRow]) -> tuple[str, ...]:
    selected_features = list(CORE_NUMERIC_FEATURES)
    for feature_name in OPTIONAL_NUMERIC_FEATURES:
        values = [
            value
            for row in train_rows
            if (value := _feature_numeric_value(row, feature_name)) is not None
        ]
        if len(values) / len(train_rows) < OPTIONAL_FEATURE_MIN_COVERAGE:
            continue
        if max(values) - min(values) <= MIN_FEATURE_VARIANCE:
            continue
        selected_features.append(feature_name)
    return tuple(selected_features)


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
    numeric_features = tuple((*_selected_numeric_features(train_rows), *pitch_type_features))
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


def _coerce_feature_vectorizer(payload: dict[str, Any]) -> _FeatureVectorizer:
    numeric_feature_stats = payload["numeric_feature_stats"]
    categorical_feature_levels = payload["categorical_feature_levels"]
    return _FeatureVectorizer(
        numeric_features=tuple(str(feature_name) for feature_name in numeric_feature_stats),
        numeric_means={
            str(feature_name): float(stats["mean"])
            for feature_name, stats in numeric_feature_stats.items()
        },
        numeric_stds={
            str(feature_name): float(stats["std"])
            for feature_name, stats in numeric_feature_stats.items()
        },
        categorical_levels={
            str(feature_name): tuple(str(level) for level in levels)
            for feature_name, levels in categorical_feature_levels.items()
        },
        encoded_feature_names=tuple(
            str(feature_name) for feature_name in payload["encoded_feature_names"]
        ),
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


def _coerce_linear_model(payload: dict[str, Any]) -> _LinearModel:
    vectorizer = _coerce_feature_vectorizer(payload)
    coefficient_map = payload["coefficients"]
    return _LinearModel(
        intercept=float(payload["intercept"]),
        coefficients=tuple(
            float(coefficient_map[feature_name])
            for feature_name in vectorizer.encoded_feature_names
        ),
        vectorizer=vectorizer,
    )


def _run_dir_end_date(run_dir: Path) -> date | None:
    label = run_dir.parent.name
    if label.startswith("start=") and "_end=" in label:
        try:
            return date.fromisoformat(label.split("_end=", 1)[1])
        except ValueError:
            pass
    dataset_path = run_dir / "training_dataset.jsonl"
    if not dataset_path.exists():
        return None
    available_dates = [
        date.fromisoformat(str(row["official_date"]))
        for row in _load_jsonl_rows(dataset_path)
        if row.get("official_date")
    ]
    if not available_dates:
        return None
    return max(available_dates)


def _latest_honest_model_run_before_date(output_root: Path, *, target_date: date) -> Path:
    model_root = output_root / "normalized" / "starter_strikeout_baseline"
    candidate_runs: list[tuple[date, Path]] = []
    for run_dir in model_root.rglob("run=*"):
        if not run_dir.is_dir():
            continue
        if not run_dir.joinpath("baseline_model.json").exists():
            continue
        if not run_dir.joinpath("ladder_probabilities.jsonl").exists():
            continue
        run_end_date = _run_dir_end_date(run_dir)
        if run_end_date is None or run_end_date >= target_date:
            continue
        candidate_runs.append((run_end_date, run_dir))
    if not candidate_runs:
        raise FileNotFoundError(
            "No starter strikeout baseline run ends before "
            f"{target_date.isoformat()}. Train a historical baseline first."
        )
    candidate_runs.sort(key=lambda item: (item[0], str(item[1])))
    return candidate_runs[-1][1]


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


def _observed_over_probability(actual_strikeouts: int, line: float) -> int:
    return int(actual_strikeouts >= (int(floor(line)) + 1))


def _ladder_probability_event_rows(
    rows: list[StarterStrikeoutTrainingRow],
    mean_predictions: list[float],
    *,
    dispersion_alpha: float,
    split_by_date: dict[str, str],
    model_train_dates: list[str],
) -> list[dict[str, Any]]:
    if len(rows) != len(mean_predictions):
        raise ValueError("rows and mean_predictions must be aligned")
    event_rows: list[dict[str, Any]] = []
    fitted_from_date = model_train_dates[0] if model_train_dates else None
    fitted_through_date = model_train_dates[-1] if model_train_dates else None
    for row, mean_prediction in zip(rows, mean_predictions):
        raw_ladder = starter_strikeout_ladder_probabilities(
            mean=mean_prediction,
            dispersion_alpha=dispersion_alpha,
        )
        for ladder_row in raw_ladder:
            event_rows.append(
                {
                    "training_row_id": row.training_row_id,
                    "official_date": row.official_date,
                    "split": split_by_date[row.official_date],
                    "game_pk": row.game_pk,
                    "pitcher_id": row.pitcher_id,
                    "pitcher_name": row.pitcher_name,
                    "actual_strikeouts": row.starter_strikeouts,
                    "model_mean": round(mean_prediction, 6),
                    "model_train_from_date": fitted_from_date,
                    "model_train_through_date": fitted_through_date,
                    "count_distribution": {
                        "name": COUNT_DISTRIBUTION_NAME,
                        "dispersion_alpha": round(dispersion_alpha, 6),
                    },
                    "line": ladder_row["line"],
                    "observed_over": _observed_over_probability(
                        row.starter_strikeouts,
                        ladder_row["line"],
                    ),
                    "raw_over_probability": ladder_row["over_probability"],
                    "raw_under_probability": ladder_row["under_probability"],
                }
            )
    return event_rows


def _out_of_fold_probability_rows(
    rows: list[StarterStrikeoutTrainingRow],
    *,
    date_splits: dict[str, list[str]],
) -> list[dict[str, Any]]:
    unique_dates = sorted({row.official_date for row in rows})
    split_by_date = {
        split_date: split_name
        for split_name, split_dates in date_splits.items()
        for split_date in split_dates
    }
    oof_rows: list[dict[str, Any]] = []
    for prediction_index in range(OOF_MIN_TRAIN_DATES, len(unique_dates)):
        prediction_date = unique_dates[prediction_index]
        prior_dates = unique_dates[:prediction_index]
        prior_rows = _rows_for_dates(rows, prior_dates)
        prediction_rows = _rows_for_dates(rows, [prediction_date])
        if not prior_rows or not prediction_rows:
            continue
        model = _fit_ridge_regression(prior_rows)
        prior_predictions = [_predict_mean(row, model) for row in prior_rows]
        dispersion_alpha = _fit_negative_binomial_dispersion_alpha(
            prior_rows,
            prior_predictions,
        )
        prediction_mean_predictions = [_predict_mean(row, model) for row in prediction_rows]
        oof_rows.extend(
            _ladder_probability_event_rows(
                prediction_rows,
                prediction_mean_predictions,
                dispersion_alpha=dispersion_alpha,
                split_by_date=split_by_date,
                model_train_dates=prior_dates,
            )
        )
    return oof_rows


def _mean_brier_score(outcomes: list[int], probabilities: list[float]) -> float:
    return round(
        sum((probability - outcome) ** 2 for outcome, probability in zip(outcomes, probabilities))
        / len(probabilities),
        6,
    )


def _mean_log_loss(outcomes: list[int], probabilities: list[float]) -> float:
    losses = []
    for outcome, probability in zip(outcomes, probabilities):
        clipped_probability = _clip_probability(probability)
        if outcome:
            losses.append(-log(clipped_probability))
        else:
            losses.append(-log(1.0 - clipped_probability))
    return round(sum(losses) / len(losses), 6)


def _reliability_bins(
    rows: list[dict[str, Any]],
    *,
    probability_key: str,
) -> list[dict[str, Any]]:
    binned_rows: list[dict[str, Any]] = []
    for bucket_index in range(RELIABILITY_BIN_COUNT):
        lower_bound = bucket_index / RELIABILITY_BIN_COUNT
        upper_bound = (bucket_index + 1) / RELIABILITY_BIN_COUNT
        bucket_rows = [
            row
            for row in rows
            if (
                lower_bound <= float(row[probability_key]) < upper_bound
                or (
                    bucket_index == RELIABILITY_BIN_COUNT - 1
                    and float(row[probability_key]) == 1.0
                )
            )
        ]
        if not bucket_rows:
            continue
        sample_count = len(bucket_rows)
        mean_predicted_probability = sum(float(row[probability_key]) for row in bucket_rows) / sample_count
        observed_rate = sum(int(row["observed_over"]) for row in bucket_rows) / sample_count
        binned_rows.append(
            {
                "bin_start": round(lower_bound, 6),
                "bin_end": round(upper_bound, 6),
                "sample_count": sample_count,
                "mean_predicted_probability": round(mean_predicted_probability, 6),
                "observed_rate": round(observed_rate, 6),
            }
        )
    return binned_rows


def _expected_calibration_error(reliability_bins: list[dict[str, Any]]) -> float:
    total_count = sum(int(bucket["sample_count"]) for bucket in reliability_bins)
    if total_count == 0:
        return 0.0
    weighted_error = sum(
        abs(
            float(bucket["mean_predicted_probability"]) - float(bucket["observed_rate"])
        )
        * int(bucket["sample_count"])
        for bucket in reliability_bins
    )
    return round(weighted_error / total_count, 6)


def _probability_metrics_for_rows(
    rows: list[dict[str, Any]],
    *,
    probability_key: str,
) -> dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "mean_brier_score": None,
            "mean_log_loss": None,
            "expected_calibration_error": None,
        }
    outcomes = [int(row["observed_over"]) for row in rows]
    probabilities = [float(row[probability_key]) for row in rows]
    reliability_bins = _reliability_bins(rows, probability_key=probability_key)
    return {
        "sample_count": len(rows),
        "mean_brier_score": _mean_brier_score(outcomes, probabilities),
        "mean_log_loss": _mean_log_loss(outcomes, probabilities),
        "expected_calibration_error": _expected_calibration_error(reliability_bins),
    }


def _calibration_improvement_summary(
    raw_metrics: dict[str, Any],
    calibrated_metrics: dict[str, Any],
    *,
    sample_warning: str | None,
) -> dict[str, Any]:
    improvement = {
        "mean_brier_score": (
            calibrated_metrics["mean_brier_score"] is not None
            and raw_metrics["mean_brier_score"] is not None
            and calibrated_metrics["mean_brier_score"] <= raw_metrics["mean_brier_score"]
        ),
        "mean_log_loss": (
            calibrated_metrics["mean_log_loss"] is not None
            and raw_metrics["mean_log_loss"] is not None
            and calibrated_metrics["mean_log_loss"] <= raw_metrics["mean_log_loss"]
        ),
        "expected_calibration_error": (
            calibrated_metrics["expected_calibration_error"] is not None
            and raw_metrics["expected_calibration_error"] is not None
            and calibrated_metrics["expected_calibration_error"]
            <= raw_metrics["expected_calibration_error"]
        ),
    }
    explanation = None
    if not all(improvement.values()):
        explanation = (
            "Calibrated probabilities did not beat raw probabilities on every bootstrap "
            "reliability metric for this split; keep the raw-vs-calibrated table for inspection."
        )
    if sample_warning:
        explanation = (
            sample_warning
            if explanation is None
            else f"{sample_warning} {explanation}"
        )
    return {
        "improved": improvement,
        "explanation": explanation,
    }


def _evaluation_probability_rows(
    rows: list[dict[str, Any]],
    *,
    split_name: str,
    calibrator: _ProbabilityCalibrator,
    calibration_training_splits: list[str],
) -> list[dict[str, Any]]:
    evaluation_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["split"] != split_name:
            continue
        calibrated_over_probability = _apply_probability_calibrator(
            float(row["raw_over_probability"]),
            calibrator,
        )
        evaluation_rows.append(
            {
                **row,
                "calibrated_over_probability": round(calibrated_over_probability, 6),
                "calibrated_under_probability": round(1.0 - calibrated_over_probability, 6),
                "calibration_method": calibrator.name,
                "calibration_training_splits": calibration_training_splits,
                "calibration_sample_count": calibrator.sample_count,
                "calibration_fit_from_date": calibrator.fitted_from_date,
                "calibration_fit_through_date": calibrator.fitted_through_date,
                "calibration_is_identity": calibrator.is_identity,
            }
        )
    return evaluation_rows


def _calibration_summary(
    rows: list[dict[str, Any]],
    *,
    calibration_splits: list[str],
    calibrator: _ProbabilityCalibrator,
) -> dict[str, Any]:
    raw_metrics = _probability_metrics_for_rows(rows, probability_key="raw_over_probability")
    calibrated_metrics = _probability_metrics_for_rows(
        rows,
        probability_key="calibrated_over_probability",
    )
    return {
        "calibration_training_splits": calibration_splits,
        "calibrator": _json_ready(calibrator),
        "raw": {
            **raw_metrics,
            "reliability_bins": _reliability_bins(rows, probability_key="raw_over_probability"),
        },
        "calibrated": {
            **calibrated_metrics,
            "reliability_bins": _reliability_bins(
                rows,
                probability_key="calibrated_over_probability",
            ),
        },
        "improvement": _calibration_improvement_summary(
            raw_metrics,
            calibrated_metrics,
            sample_warning=calibrator.sample_warning,
        ),
    }


def _training_ladder_artifact_rows(
    rows: list[StarterStrikeoutTrainingRow],
    *,
    model: _LinearModel,
    dispersion_alpha: float,
    split_by_date: dict[str, str],
    probability_calibrator: _ProbabilityCalibrator,
) -> list[dict[str, Any]]:
    artifact_rows: list[dict[str, Any]] = []
    for row in rows:
        model_mean = _predict_mean(row, model)
        raw_ladder = starter_strikeout_ladder_probabilities(
            mean=model_mean,
            dispersion_alpha=dispersion_alpha,
        )
        artifact_rows.append(
            {
                "training_row_id": row.training_row_id,
                "official_date": row.official_date,
                "game_pk": row.game_pk,
                "pitcher_id": row.pitcher_id,
                "pitcher_name": row.pitcher_name,
                "split": split_by_date[row.official_date],
                "feature_row_id": row.training_row_id,
                "lineup_snapshot_id": row.lineup_snapshot_id,
                # Historical edge builds use the feature cutoff as the
                # conservative projection timestamp until a dedicated
                # pregame inference runner writes its own snapshot.
                "features_as_of": row.features_as_of,
                "projection_generated_at": row.features_as_of,
                "actual_strikeouts": row.starter_strikeouts,
                "naive_benchmark_mean": round(row.naive_benchmark_mean, 6),
                "model_mean": round(model_mean, 6),
                "count_distribution": {
                    "name": COUNT_DISTRIBUTION_NAME,
                    "dispersion_alpha": round(dispersion_alpha, 6),
                },
                "probability_calibration": {
                    "name": probability_calibrator.name,
                    "sample_count": probability_calibrator.sample_count,
                    "is_identity": probability_calibrator.is_identity,
                },
                "ladder_probabilities": raw_ladder,
                "calibrated_ladder_probabilities": calibrate_starter_strikeout_ladder_probabilities(
                    raw_ladder,
                    probability_calibrator,
                ),
            }
        )
    return artifact_rows


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


def _metric_value(payload: dict[str, Any], *path: str) -> float | None:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None
    return round(float(current), 6)


def _comparison_entry(
    *,
    current: float | None,
    previous: float | None,
    lower_is_better: bool,
) -> dict[str, Any] | None:
    if current is None or previous is None:
        return None
    delta = round(current - previous, 6)
    if abs(delta) < 1e-9:
        status = "unchanged"
    elif (delta < 0.0 and lower_is_better) or (delta > 0.0 and not lower_is_better):
        status = "improved"
    else:
        status = "worsened"
    return {
        "current": current,
        "previous": previous,
        "delta": delta,
        "status": status,
    }


def _held_out_status(beats_benchmark: dict[str, bool]) -> str:
    if all(bool(value) for value in beats_benchmark.values()):
        return "beating_benchmark"
    if any(bool(value) for value in beats_benchmark.values()):
        return "mixed_vs_benchmark"
    return "underperforming_benchmark"


def _latest_previous_training_run(
    output_root: Path,
    *,
    start_date: date,
    end_date: date,
) -> Path | None:
    run_root = (
        output_root
        / "normalized"
        / "starter_strikeout_baseline"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
    )
    if not run_root.exists():
        return None
    run_dirs = sorted(
        path
        for path in run_root.glob("run=*")
        if path.is_dir() and path.joinpath("evaluation.json").exists()
    )
    if not run_dirs:
        return None
    return run_dirs[-1]


def _evaluation_summary_payload(
    *,
    start_date: date,
    end_date: date,
    run_id: str,
    mlflow_run_id: str,
    mlflow_experiment_name: str,
    tracking_uri: str,
    evaluation: dict[str, Any],
    evaluation_path: Path,
    reproducibility_notes_path: Path,
    rerun_command: str,
    previous_run_dir: Path | None,
) -> dict[str, Any]:
    held_out_status = _held_out_status(evaluation["held_out_beats_benchmark"])
    held_out_probability_metrics = (
        evaluation["probability_calibration"]["honest_held_out"]["held_out"]
    )
    summary: dict[str, Any] = {
        "summary_version": "starter_strikeout_baseline_evaluation_v1",
        "model_version": evaluation["model_version"],
        "benchmark_name": evaluation["benchmark_name"],
        "run_id": run_id,
        "mlflow_run_id": mlflow_run_id,
        "mlflow_experiment_name": mlflow_experiment_name,
        "tracking_uri": tracking_uri,
        "date_window": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        "evaluation_path": str(evaluation_path),
        "reproducibility": {
            "notes_path": str(reproducibility_notes_path),
            "rerun_command": rerun_command,
        },
        "row_counts": evaluation["row_counts"],
        "date_splits": evaluation["date_splits"],
        "held_out_performance": {
            "status": held_out_status,
            "benchmark": evaluation["benchmark"]["held_out"],
            "model": evaluation["model"]["held_out"],
            "beats_benchmark": evaluation["held_out_beats_benchmark"],
        },
        "held_out_probability_calibration": {
            "raw": {
                "mean_brier_score": _metric_value(
                    held_out_probability_metrics,
                    "raw",
                    "mean_brier_score",
                ),
                "mean_log_loss": _metric_value(
                    held_out_probability_metrics,
                    "raw",
                    "mean_log_loss",
                ),
                "expected_calibration_error": _metric_value(
                    held_out_probability_metrics,
                    "raw",
                    "expected_calibration_error",
                ),
            },
            "calibrated": {
                "mean_brier_score": _metric_value(
                    held_out_probability_metrics,
                    "calibrated",
                    "mean_brier_score",
                ),
                "mean_log_loss": _metric_value(
                    held_out_probability_metrics,
                    "calibrated",
                    "mean_log_loss",
                ),
                "expected_calibration_error": _metric_value(
                    held_out_probability_metrics,
                    "calibrated",
                    "expected_calibration_error",
                ),
            },
            "improves_raw": evaluation["probability_calibration"]["held_out_improves_raw"],
        },
        "held_out_count_distribution": {
            "dispersion_alpha": _metric_value(
                evaluation,
                "count_distribution",
                "dispersion_alpha",
            ),
            "poisson": evaluation["count_distribution"]["poisson"]["held_out"],
            "negative_binomial": evaluation["count_distribution"]["negative_binomial"][
                "held_out"
            ],
            "beats_poisson": evaluation["count_distribution"]["held_out_beats_poisson"],
        },
        "top_feature_importance": evaluation["feature_importance"][:10],
        "previous_run_comparison": None,
    }

    if previous_run_dir is None:
        return summary

    previous_evaluation_path = previous_run_dir / "evaluation.json"
    previous_evaluation = _load_json(previous_evaluation_path)
    summary["previous_run_comparison"] = {
        "previous_run_id": _path_run_id(previous_run_dir),
        "evaluation_path": str(previous_evaluation_path),
        "held_out_model": {
            "rmse": _comparison_entry(
                current=_metric_value(evaluation, "model", "held_out", "rmse"),
                previous=_metric_value(
                    previous_evaluation,
                    "model",
                    "held_out",
                    "rmse",
                ),
                lower_is_better=True,
            ),
            "mae": _comparison_entry(
                current=_metric_value(evaluation, "model", "held_out", "mae"),
                previous=_metric_value(
                    previous_evaluation,
                    "model",
                    "held_out",
                    "mae",
                ),
                lower_is_better=True,
            ),
            "spearman_rank_correlation": _comparison_entry(
                current=_metric_value(
                    evaluation,
                    "model",
                    "held_out",
                    "spearman_rank_correlation",
                ),
                previous=_metric_value(
                    previous_evaluation,
                    "model",
                    "held_out",
                    "spearman_rank_correlation",
                ),
                lower_is_better=False,
            ),
        },
        "held_out_probability_calibration": {
            "mean_brier_score": _comparison_entry(
                current=_metric_value(
                    held_out_probability_metrics,
                    "calibrated",
                    "mean_brier_score",
                ),
                previous=_metric_value(
                    previous_evaluation,
                    "probability_calibration",
                    "honest_held_out",
                    "held_out",
                    "calibrated",
                    "mean_brier_score",
                ),
                lower_is_better=True,
            ),
            "mean_log_loss": _comparison_entry(
                current=_metric_value(
                    held_out_probability_metrics,
                    "calibrated",
                    "mean_log_loss",
                ),
                previous=_metric_value(
                    previous_evaluation,
                    "probability_calibration",
                    "honest_held_out",
                    "held_out",
                    "calibrated",
                    "mean_log_loss",
                ),
                lower_is_better=True,
            ),
            "expected_calibration_error": _comparison_entry(
                current=_metric_value(
                    held_out_probability_metrics,
                    "calibrated",
                    "expected_calibration_error",
                ),
                previous=_metric_value(
                    previous_evaluation,
                    "probability_calibration",
                    "honest_held_out",
                    "held_out",
                    "calibrated",
                    "expected_calibration_error",
                ),
                lower_is_better=True,
            ),
        },
    }
    return summary


def _render_evaluation_summary_markdown(summary: dict[str, Any]) -> str:
    def _format_value(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    row_counts = summary["row_counts"]
    held_out = summary["held_out_performance"]
    calibration = summary["held_out_probability_calibration"]
    count_distribution = summary["held_out_count_distribution"]

    lines = [
        "# Starter Strikeout Baseline Evaluation Summary",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- MLflow run ID: `{summary['mlflow_run_id']}`",
        f"- MLflow experiment: `{summary['mlflow_experiment_name']}`",
        f"- Tracking URI: `{summary['tracking_uri']}`",
        (
            "- Date window: "
            f"`{summary['date_window']['start_date']}` -> `{summary['date_window']['end_date']}`"
        ),
        f"- Reproducibility notes: `{summary['reproducibility']['notes_path']}`",
        (
            "- Row counts: "
            f"train={row_counts['train']}, validation={row_counts['validation']}, "
            f"test={row_counts['test']}, held_out={row_counts['held_out']}"
        ),
        f"- Held-out status: `{held_out['status']}`",
        "",
        "## Held-Out Performance",
        "",
        "| Metric | Benchmark | Model | Model Beats Benchmark |",
        "| --- | ---: | ---: | --- |",
        (
            f"| RMSE | {_format_value(held_out['benchmark']['rmse'])} | "
            f"{_format_value(held_out['model']['rmse'])} | "
            f"{_format_value(held_out['beats_benchmark']['rmse'])} |"
        ),
        (
            f"| MAE | {_format_value(held_out['benchmark']['mae'])} | "
            f"{_format_value(held_out['model']['mae'])} | "
            f"{_format_value(held_out['beats_benchmark']['mae'])} |"
        ),
        (
            "| Spearman | "
            f"{_format_value(held_out['benchmark']['spearman_rank_correlation'])} | "
            f"{_format_value(held_out['model']['spearman_rank_correlation'])} | n/a |"
        ),
        "",
        "## Held-Out Probability Calibration",
        "",
        "| Metric | Raw | Calibrated | Improved |",
        "| --- | ---: | ---: | --- |",
        (
            f"| Mean Brier Score | {_format_value(calibration['raw']['mean_brier_score'])} | "
            f"{_format_value(calibration['calibrated']['mean_brier_score'])} | "
            f"{_format_value(calibration['improves_raw']['mean_brier_score'])} |"
        ),
        (
            f"| Mean Log Loss | {_format_value(calibration['raw']['mean_log_loss'])} | "
            f"{_format_value(calibration['calibrated']['mean_log_loss'])} | "
            f"{_format_value(calibration['improves_raw']['mean_log_loss'])} |"
        ),
        (
            "| Expected Calibration Error | "
            f"{_format_value(calibration['raw']['expected_calibration_error'])} | "
            f"{_format_value(calibration['calibrated']['expected_calibration_error'])} | "
            f"{_format_value(calibration['improves_raw']['expected_calibration_error'])} |"
        ),
        "",
        "## Held-Out Count Distribution",
        "",
        (
            f"- Dispersion alpha: "
            f"`{_format_value(count_distribution['dispersion_alpha'])}`"
        ),
        (
            "- Negative binomial beats Poisson on held-out metrics: "
            f"`{count_distribution['beats_poisson']}`"
        ),
        "",
        "## Top Feature Importance",
        "",
        "| Feature | Coefficient | Absolute Importance |",
        "| --- | ---: | ---: |",
    ]
    for feature in summary["top_feature_importance"]:
        lines.append(
            f"| `{feature['feature']}` | {_format_value(feature['coefficient'])} | "
            f"{_format_value(feature['absolute_importance'])} |"
        )

    lines.extend(["", "## Comparison To Previous Run", ""])
    previous_run = summary.get("previous_run_comparison")
    if previous_run is None:
        lines.append("No previous run was found for this exact date window.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            f"- Previous run ID: `{previous_run['previous_run_id']}`",
            f"- Previous evaluation: `{previous_run['evaluation_path']}`",
            "",
            "### Held-Out Model Metric Deltas",
            "",
            "| Metric | Previous | Current | Delta | Status |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for label, item in (
        ("RMSE", previous_run["held_out_model"]["rmse"]),
        ("MAE", previous_run["held_out_model"]["mae"]),
        ("Spearman", previous_run["held_out_model"]["spearman_rank_correlation"]),
    ):
        if item is None:
            lines.append(f"| {label} | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {label} | {_format_value(item['previous'])} | {_format_value(item['current'])} | "
            f"{_format_value(item['delta'])} | {item['status']} |"
        )

    lines.extend(
        [
            "",
            "### Held-Out Calibrated Probability Deltas",
            "",
            "| Metric | Previous | Current | Delta | Status |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for label, item in (
        (
            "Mean Brier Score",
            previous_run["held_out_probability_calibration"]["mean_brier_score"],
        ),
        (
            "Mean Log Loss",
            previous_run["held_out_probability_calibration"]["mean_log_loss"],
        ),
        (
            "Expected Calibration Error",
            previous_run["held_out_probability_calibration"]["expected_calibration_error"],
        ),
    ):
        if item is None:
            lines.append(f"| {label} | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {label} | {_format_value(item['previous'])} | {_format_value(item['current'])} | "
            f"{_format_value(item['delta'])} | {item['status']} |"
        )
    return "\n".join(lines) + "\n"


def _model_artifact(
    model: _LinearModel,
    *,
    dispersion_alpha: float,
    probability_calibrator: _ProbabilityCalibrator | dict[str, Any] | None = None,
    tracking: dict[str, Any] | None = None,
) -> dict[str, Any]:
    calibrator_payload = None
    if probability_calibrator is not None:
        calibrator_payload = _json_ready(_coerce_probability_calibrator(probability_calibrator))
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
        "probability_calibration": calibrator_payload,
        "tracking": tracking,
    }


def _training_rerun_command(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str,
) -> str:
    return (
        "uv run python -m mlb_props_stack train-starter-strikeout-baseline "
        f"--start-date {start_date.isoformat()} "
        f"--end-date {end_date.isoformat()} "
        f"--output-dir {quote(str(output_dir))}"
    )


def _render_training_reproducibility_notes(
    *,
    start_date: date,
    end_date: date,
    run_id: str,
    normalized_root: Path,
    tracking_uri: str,
    mlflow_experiment_name: str,
    mlflow_run_id: str,
    rerun_command: str,
) -> str:
    return "\n".join(
        [
            "# Training Reproducibility Notes",
            "",
            f"- Local run ID: `{run_id}`",
            f"- MLflow run ID: `{mlflow_run_id}`",
            f"- MLflow experiment: `{mlflow_experiment_name}`",
            f"- Tracking URI: `{tracking_uri}`",
            f"- Local run directory: `{normalized_root}`",
            (
                "- Date window: "
                f"`{start_date.isoformat()}` -> `{end_date.isoformat()}`"
            ),
            f"- CLI rerun command: `{rerun_command}`",
            (
                "- Inputs: saved AGE-146 Statcast feature rows already written under "
                "`data/normalized/statcast_search/date=...` inside the requested window."
            ),
            (
                "- Honest rules: date splits stay chronological and same-game outcome rows "
                "are fetched without using post-game features."
            ),
            "",
        ]
    )


def _training_tracking_params(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str,
    row_count: int,
    outcome_count: int,
    evaluation: dict[str, Any],
    held_out_status: str,
    min_sample_for_calibration: int,
) -> dict[str, Any]:
    return {
        "pipeline": "train_starter_strikeout_baseline",
        "model_version": MODEL_VERSION,
        "benchmark_name": BENCHMARK_NAME,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "output_dir": str(output_dir),
        "row_count": row_count,
        "outcome_count": outcome_count,
        "train_date_count": len(evaluation["date_splits"]["train"]),
        "validation_date_count": len(evaluation["date_splits"]["validation"]),
        "test_date_count": len(evaluation["date_splits"]["test"]),
        "ridge_alpha": RIDGE_ALPHA,
        "min_sample_for_calibration": min_sample_for_calibration,
        "held_out_status": held_out_status,
    }


def _training_tracking_metrics(
    *,
    evaluation: dict[str, Any],
    dispersion_alpha: float,
) -> dict[str, float | int | None]:
    held_out_probability_metrics = evaluation["probability_calibration"]["honest_held_out"]["held_out"]
    return {
        "training_rows": evaluation["row_counts"]["train"],
        "validation_rows": evaluation["row_counts"]["validation"],
        "test_rows": evaluation["row_counts"]["test"],
        "held_out_rows": evaluation["row_counts"]["held_out"],
        "dispersion_alpha": round(dispersion_alpha, 6),
        "held_out_model_rmse": _metric_value(evaluation, "model", "held_out", "rmse"),
        "held_out_benchmark_rmse": _metric_value(evaluation, "benchmark", "held_out", "rmse"),
        "held_out_model_mae": _metric_value(evaluation, "model", "held_out", "mae"),
        "held_out_benchmark_mae": _metric_value(evaluation, "benchmark", "held_out", "mae"),
        "held_out_model_spearman": _metric_value(
            evaluation,
            "model",
            "held_out",
            "spearman_rank_correlation",
        ),
        "held_out_benchmark_spearman": _metric_value(
            evaluation,
            "benchmark",
            "held_out",
            "spearman_rank_correlation",
        ),
        "held_out_negative_binomial_mean_negative_log_likelihood": _metric_value(
            evaluation,
            "count_distribution",
            "negative_binomial",
            "held_out",
            "mean_negative_log_likelihood",
        ),
        "held_out_poisson_mean_negative_log_likelihood": _metric_value(
            evaluation,
            "count_distribution",
            "poisson",
            "held_out",
            "mean_negative_log_likelihood",
        ),
        "held_out_raw_mean_brier_score": _metric_value(
            held_out_probability_metrics,
            "raw",
            "mean_brier_score",
        ),
        "held_out_calibrated_mean_brier_score": _metric_value(
            held_out_probability_metrics,
            "calibrated",
            "mean_brier_score",
        ),
        "held_out_raw_mean_log_loss": _metric_value(
            held_out_probability_metrics,
            "raw",
            "mean_log_loss",
        ),
        "held_out_calibrated_mean_log_loss": _metric_value(
            held_out_probability_metrics,
            "calibrated",
            "mean_log_loss",
        ),
        "held_out_raw_expected_calibration_error": _metric_value(
            held_out_probability_metrics,
            "raw",
            "expected_calibration_error",
        ),
        "held_out_calibrated_expected_calibration_error": _metric_value(
            held_out_probability_metrics,
            "calibrated",
            "expected_calibration_error",
        ),
    }


def generate_starter_strikeout_inference_for_date(
    *,
    target_date: date,
    output_dir: Path | str = "data",
    source_model_run_dir: Path | str | None = None,
    now: Callable[[], datetime] = utc_now,
) -> StarterStrikeoutInferenceResult:
    """Score one target date from the latest non-leaky saved baseline model."""
    output_root = Path(output_dir)
    resolved_source_model_run_dir = (
        Path(source_model_run_dir)
        if source_model_run_dir is not None
        else _latest_honest_model_run_before_date(output_root, target_date=target_date)
    )
    source_model_path = resolved_source_model_run_dir / "baseline_model.json"
    source_model_artifact = _load_json(source_model_path)
    probability_calibration = source_model_artifact.get("probability_calibration")
    if probability_calibration is None:
        raise ValueError(
            "baseline_model.json is missing probability_calibration metadata."
        )

    feature_run_dir = _latest_feature_run_dir(
        output_root / "normalized" / "statcast_search" / f"date={target_date.isoformat()}"
    )
    inference_rows = sorted(
        _load_feature_rows_for_date(target_date=target_date, output_dir=output_root),
        key=lambda row: (row.official_date, row.game_pk, row.pitcher_id),
    )
    if not inference_rows:
        raise FileNotFoundError(
            "No AGE-146 Statcast feature rows were found for "
            f"{target_date.isoformat()}."
        )

    model = _coerce_linear_model(source_model_artifact)
    dispersion_alpha = float(source_model_artifact["count_distribution"]["dispersion_alpha"])
    calibrator = _coerce_probability_calibrator(probability_calibration)
    source_date_splits_path = resolved_source_model_run_dir / "date_splits.json"
    source_date_splits = (
        _load_json(source_date_splits_path) if source_date_splits_path.exists() else {}
    )
    train_dates = sorted(str(value) for value in source_date_splits.get("train", []))
    inference_generated_at = now().astimezone(UTC)
    run_id = inference_generated_at.strftime("%Y%m%dT%H%M%SZ")

    ladder_rows: list[dict[str, Any]] = []
    for row in inference_rows:
        model_mean = _predict_mean(row, model)
        raw_ladder = starter_strikeout_ladder_probabilities(
            mean=model_mean,
            dispersion_alpha=dispersion_alpha,
        )
        ladder_rows.append(
            {
                "training_row_id": row.training_row_id,
                "official_date": row.official_date,
                "game_pk": row.game_pk,
                "pitcher_id": row.pitcher_id,
                "pitcher_name": row.pitcher_name,
                "split": "inference",
                "feature_row_id": row.training_row_id,
                "lineup_snapshot_id": row.lineup_snapshot_id,
                "features_as_of": row.features_as_of,
                "projection_generated_at": row.features_as_of,
                "inference_generated_at": inference_generated_at,
                "model_train_from_date": train_dates[0] if train_dates else None,
                "model_train_through_date": train_dates[-1] if train_dates else None,
                "naive_benchmark_mean": round(row.naive_benchmark_mean, 6),
                "model_mean": round(model_mean, 6),
                "count_distribution": {
                    "name": COUNT_DISTRIBUTION_NAME,
                    "dispersion_alpha": round(dispersion_alpha, 6),
                },
                "probability_calibration": {
                    "name": calibrator.name,
                    "sample_count": calibrator.sample_count,
                    "fit_from_date": calibrator.fitted_from_date,
                    "fit_through_date": calibrator.fitted_through_date,
                    "is_identity": calibrator.is_identity,
                },
                "ladder_probabilities": raw_ladder,
                "calibrated_ladder_probabilities": (
                    calibrate_starter_strikeout_ladder_probabilities(
                        raw_ladder,
                        calibrator,
                    )
                ),
            }
        )

    normalized_root = (
        output_root
        / "normalized"
        / "starter_strikeout_inference"
        / f"date={target_date.isoformat()}"
        / f"run={run_id}"
    )
    model_path = normalized_root / "baseline_model.json"
    ladder_probabilities_path = normalized_root / "ladder_probabilities.jsonl"
    inference_manifest_path = normalized_root / "inference_manifest.json"
    _write_json(
        model_path,
        {
            **source_model_artifact,
            "source_model_run_id": _path_run_id(resolved_source_model_run_dir),
            "source_model_path": source_model_path,
            "target_date": target_date,
            "feature_run_dir": feature_run_dir,
            "inference_generated_at": inference_generated_at,
            "model_train_from_date": train_dates[0] if train_dates else None,
            "model_train_through_date": train_dates[-1] if train_dates else None,
        },
    )
    _write_jsonl(ladder_probabilities_path, ladder_rows)
    _write_json(
        inference_manifest_path,
        {
            "target_date": target_date,
            "run_id": run_id,
            "source_model_run_id": _path_run_id(resolved_source_model_run_dir),
            "source_model_path": source_model_path,
            "feature_run_dir": feature_run_dir,
            "projection_count": len(ladder_rows),
            "inference_generated_at": inference_generated_at,
        },
    )
    return StarterStrikeoutInferenceResult(
        target_date=target_date,
        run_id=run_id,
        source_model_run_id=_path_run_id(resolved_source_model_run_dir),
        source_model_path=source_model_path,
        feature_run_dir=feature_run_dir,
        model_path=model_path,
        ladder_probabilities_path=ladder_probabilities_path,
        projection_count=len(ladder_rows),
    )


def train_starter_strikeout_baseline(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str = "data",
    client: StatcastSearchClient | None = None,
    now: Callable[[], datetime] = utc_now,
    tracking_config: TrackingConfig | None = None,
) -> StarterStrikeoutBaselineTrainingResult:
    """Train the first deterministic starter strikeout baseline model."""
    if client is None:
        client = StatcastSearchClient()

    config = StackConfig()
    tracking = tracking_config or TrackingConfig()
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
    split_by_date = {
        split_date: split_name
        for split_name, split_dates in date_splits.items()
        for split_date in split_dates
    }
    oof_probability_rows = _out_of_fold_probability_rows(
        rows_with_outcomes,
        date_splits=date_splits,
    )
    production_calibrator = _fit_probability_calibrator(
        probabilities=[
            float(row["raw_over_probability"])
            for row in oof_probability_rows
        ],
        outcomes=[int(row["observed_over"]) for row in oof_probability_rows],
        configured_min_sample=config.min_sample_for_calibration,
        fitted_from_date=(
            min((str(row["official_date"]) for row in oof_probability_rows), default=None)
        ),
        fitted_through_date=(
            max((str(row["official_date"]) for row in oof_probability_rows), default=None)
        ),
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
        "probability_calibration": {
            "name": PROBABILITY_CALIBRATOR_NAME,
            "source": PROBABILITY_CALIBRATOR_SOURCE,
            "configured_min_sample": config.min_sample_for_calibration,
            "production_calibrator": _json_ready(production_calibrator),
            "honest_held_out": {},
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

    validation_calibrator_rows = [
        row
        for row in oof_probability_rows
        if row["split"] == "train"
    ]
    validation_calibrator = _fit_probability_calibrator(
        probabilities=[
            float(row["raw_over_probability"])
            for row in validation_calibrator_rows
        ],
        outcomes=[int(row["observed_over"]) for row in validation_calibrator_rows],
        configured_min_sample=config.min_sample_for_calibration,
        fitted_from_date=(
            min((str(row["official_date"]) for row in validation_calibrator_rows), default=None)
        ),
        fitted_through_date=(
            max((str(row["official_date"]) for row in validation_calibrator_rows), default=None)
        ),
    )
    validation_probability_rows = _evaluation_probability_rows(
        oof_probability_rows,
        split_name="validation",
        calibrator=validation_calibrator,
        calibration_training_splits=["train"],
    )
    evaluation["probability_calibration"]["honest_held_out"]["validation"] = _calibration_summary(
        validation_probability_rows,
        calibration_splits=["train"],
        calibrator=validation_calibrator,
    )

    test_calibrator_rows = [
        row
        for row in oof_probability_rows
        if row["split"] in {"train", "validation"}
    ]
    test_calibrator = _fit_probability_calibrator(
        probabilities=[
            float(row["raw_over_probability"])
            for row in test_calibrator_rows
        ],
        outcomes=[int(row["observed_over"]) for row in test_calibrator_rows],
        configured_min_sample=config.min_sample_for_calibration,
        fitted_from_date=(
            min((str(row["official_date"]) for row in test_calibrator_rows), default=None)
        ),
        fitted_through_date=(
            max((str(row["official_date"]) for row in test_calibrator_rows), default=None)
        ),
    )
    test_probability_rows = _evaluation_probability_rows(
        oof_probability_rows,
        split_name="test",
        calibrator=test_calibrator,
        calibration_training_splits=["train", "validation"],
    )
    evaluation["probability_calibration"]["honest_held_out"]["test"] = _calibration_summary(
        test_probability_rows,
        calibration_splits=["train", "validation"],
        calibrator=test_calibrator,
    )

    honest_held_out_probability_rows = [
        *validation_probability_rows,
        *test_probability_rows,
    ]
    evaluation["probability_calibration"]["honest_held_out"]["held_out"] = {
        "raw": {
            **_probability_metrics_for_rows(
                honest_held_out_probability_rows,
                probability_key="raw_over_probability",
            ),
            "reliability_bins": _reliability_bins(
                honest_held_out_probability_rows,
                probability_key="raw_over_probability",
            ),
        },
        "calibrated": {
            **_probability_metrics_for_rows(
                honest_held_out_probability_rows,
                probability_key="calibrated_over_probability",
            ),
            "reliability_bins": _reliability_bins(
                honest_held_out_probability_rows,
                probability_key="calibrated_over_probability",
            ),
        },
        "row_count": len(honest_held_out_probability_rows),
    }
    evaluation["probability_calibration"]["held_out_improves_raw"] = {
        "mean_brier_score": (
            evaluation["probability_calibration"]["honest_held_out"]["held_out"]["calibrated"][
                "mean_brier_score"
            ]
            is not None
            and evaluation["probability_calibration"]["honest_held_out"]["held_out"]["raw"][
                "mean_brier_score"
            ]
            is not None
            and evaluation["probability_calibration"]["honest_held_out"]["held_out"]["calibrated"][
                "mean_brier_score"
            ]
            <= evaluation["probability_calibration"]["honest_held_out"]["held_out"]["raw"][
                "mean_brier_score"
            ]
        ),
        "mean_log_loss": (
            evaluation["probability_calibration"]["honest_held_out"]["held_out"]["calibrated"][
                "mean_log_loss"
            ]
            is not None
            and evaluation["probability_calibration"]["honest_held_out"]["held_out"]["raw"][
                "mean_log_loss"
            ]
            is not None
            and evaluation["probability_calibration"]["honest_held_out"]["held_out"]["calibrated"][
                "mean_log_loss"
            ]
            <= evaluation["probability_calibration"]["honest_held_out"]["held_out"]["raw"][
                "mean_log_loss"
            ]
        ),
        "expected_calibration_error": (
            evaluation["probability_calibration"]["honest_held_out"]["held_out"]["calibrated"][
                "expected_calibration_error"
            ]
            is not None
            and evaluation["probability_calibration"]["honest_held_out"]["held_out"]["raw"][
                "expected_calibration_error"
            ]
            is not None
            and evaluation["probability_calibration"]["honest_held_out"]["held_out"]["calibrated"][
                "expected_calibration_error"
            ]
            <= evaluation["probability_calibration"]["honest_held_out"]["held_out"]["raw"][
                "expected_calibration_error"
            ]
        ),
    }
    if not all(evaluation["probability_calibration"]["held_out_improves_raw"].values()):
        evaluation["probability_calibration"]["held_out_explanation"] = (
            "Held-out calibrated probabilities are stored alongside raw probabilities so the next "
            "pricing issue can inspect where reliability improved and where it did not."
        )

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
    probability_calibrator_path = normalized_root / "probability_calibrator.json"
    raw_vs_calibrated_path = normalized_root / "raw_vs_calibrated_probabilities.jsonl"
    calibration_summary_path = normalized_root / "calibration_summary.json"
    evaluation_summary_path = normalized_root / "evaluation_summary.json"
    evaluation_summary_markdown_path = normalized_root / "evaluation_summary.md"
    reproducibility_notes_path = normalized_root / "reproducibility_notes.md"
    previous_run_dir = _latest_previous_training_run(
        output_root,
        start_date=start_date,
        end_date=end_date,
    )
    rerun_command = _training_rerun_command(
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
    )
    run_name = (
        f"starter-strikeout-baseline-{start_date.isoformat()}-"
        f"{end_date.isoformat()}-{run_id}"
    )
    with start_experiment_run(
        experiment_name=tracking.training_experiment_name,
        run_name=run_name,
        tags={
            "run_kind": "training",
            "pipeline": "train_starter_strikeout_baseline",
            "model_version": MODEL_VERSION,
        },
        config=tracking,
    ) as tracking_run:
        tracking_payload = {
            "tracking_uri": tracking_run.tracking_uri,
            "mlflow_experiment_name": tracking_run.experiment_name,
            "mlflow_run_id": tracking_run.run_id,
        }
        evaluation["tracking"] = tracking_payload
        evaluation["reproducibility"] = {
            "notes_path": reproducibility_notes_path,
            "rerun_command": rerun_command,
        }
        evaluation_summary = _evaluation_summary_payload(
            start_date=start_date,
            end_date=end_date,
            run_id=run_id,
            mlflow_run_id=tracking_run.run_id,
            mlflow_experiment_name=tracking_run.experiment_name,
            tracking_uri=tracking_run.tracking_uri,
            evaluation=evaluation,
            evaluation_path=evaluation_path,
            reproducibility_notes_path=reproducibility_notes_path,
            rerun_command=rerun_command,
            previous_run_dir=previous_run_dir,
        )
        _write_jsonl(dataset_path, rows_with_outcomes)
        _write_jsonl(outcomes_path, outcome_records)
        _write_json(date_splits_path, date_splits)
        _write_json(
            model_path,
            _model_artifact(
                model,
                dispersion_alpha=dispersion_alpha,
                probability_calibrator=production_calibrator,
                tracking=tracking_payload,
            ),
        )
        _write_json(probability_calibrator_path, production_calibrator)
        _write_json(evaluation_path, evaluation)
        _write_json(calibration_summary_path, evaluation["probability_calibration"])
        _write_json(evaluation_summary_path, evaluation_summary)
        _write_text(
            evaluation_summary_markdown_path,
            _render_evaluation_summary_markdown(evaluation_summary),
        )
        _write_text(
            reproducibility_notes_path,
            _render_training_reproducibility_notes(
                start_date=start_date,
                end_date=end_date,
                run_id=run_id,
                normalized_root=normalized_root,
                tracking_uri=tracking_run.tracking_uri,
                mlflow_experiment_name=tracking_run.experiment_name,
                mlflow_run_id=tracking_run.run_id,
                rerun_command=rerun_command,
            ),
        )
        _write_jsonl(raw_vs_calibrated_path, honest_held_out_probability_rows)
        _write_jsonl(
            ladder_probabilities_path,
            _training_ladder_artifact_rows(
                rows_with_outcomes,
                model=model,
                dispersion_alpha=dispersion_alpha,
                split_by_date=split_by_date,
                probability_calibrator=production_calibrator,
            ),
        )
        log_run_params(
            _training_tracking_params(
                start_date=start_date,
                end_date=end_date,
                output_dir=output_dir,
                row_count=len(rows_with_outcomes),
                outcome_count=len(outcome_records),
                evaluation=evaluation,
                held_out_status=str(evaluation_summary["held_out_performance"]["status"]),
                min_sample_for_calibration=config.min_sample_for_calibration,
            )
        )
        log_run_metrics(
            _training_tracking_metrics(
                evaluation=evaluation,
                dispersion_alpha=dispersion_alpha,
            )
        )
        log_run_artifact(normalized_root)
    return StarterStrikeoutBaselineTrainingResult(
        start_date=start_date,
        end_date=end_date,
        run_id=run_id,
        mlflow_run_id=tracking_run.run_id,
        mlflow_experiment_name=tracking_run.experiment_name,
        row_count=len(rows_with_outcomes),
        outcome_count=len(outcome_records),
        dispersion_alpha=round(dispersion_alpha, 6),
        dataset_path=dataset_path,
        outcomes_path=outcomes_path,
        date_splits_path=date_splits_path,
        model_path=model_path,
        evaluation_path=evaluation_path,
        ladder_probabilities_path=ladder_probabilities_path,
        probability_calibrator_path=probability_calibrator_path,
        raw_vs_calibrated_path=raw_vs_calibrated_path,
        calibration_summary_path=calibration_summary_path,
        evaluation_summary_path=evaluation_summary_path,
        evaluation_summary_markdown_path=evaluation_summary_markdown_path,
        reproducibility_notes_path=reproducibility_notes_path,
        held_out_status=str(evaluation_summary["held_out_performance"]["status"]),
        held_out_model_rmse=_metric_value(evaluation, "model", "held_out", "rmse"),
        held_out_benchmark_rmse=_metric_value(evaluation, "benchmark", "held_out", "rmse"),
        held_out_model_mae=_metric_value(evaluation, "model", "held_out", "mae"),
        held_out_benchmark_mae=_metric_value(evaluation, "benchmark", "held_out", "mae"),
        previous_run_id=(
            None
            if evaluation_summary["previous_run_comparison"] is None
            else str(evaluation_summary["previous_run_comparison"]["previous_run_id"])
        ),
    )
