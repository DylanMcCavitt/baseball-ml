"""Candidate starter strikeout model-family comparison.

This module intentionally stays dependency-light. The issue calls for serious
candidate families, but the repo has not approved heavy modeling dependencies
yet, so the tree model is a deterministic boosted-stump ensemble and the neural
challenger is reported only when the tabular artifact shape can support it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import json
from math import exp, floor, isfinite, log, sqrt
from pathlib import Path
from shlex import quote
from typing import Any, Callable

from .ingest.mlb_stats_api import utc_now
from .modeling import (
    _normalized_count_distribution,
    _solve_linear_system,
    _split_dates,
)

MODEL_FAMILY_REPORT_VERSION = "candidate_strikeout_models_v1"
SELECTED_MODEL_VERSION = "starter-strikeout-candidate-v1"
COMMON_PROP_LINES = (2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5)
RIDGE_ALPHA = 8.0
MIN_PROBABILITY = 1e-12
MIN_MEAN = 1e-6
MIN_DISPERSION_ALPHA = 1e-6
BOOSTED_STUMP_COUNT = 40
BOOSTED_STUMP_LEARNING_RATE = 0.08


@dataclass(frozen=True)
class CandidateStrikeoutModelTrainingResult:
    """Filesystem output summary for one candidate-family training run."""

    start_date: date
    end_date: date
    run_id: str
    selected_candidate: str
    row_count: int
    report_path: Path
    report_markdown_path: Path
    selected_model_path: Path
    model_outputs_path: Path
    reproducibility_notes_path: Path


@dataclass(frozen=True)
class _FeatureSpec:
    name: str
    group: str
    source: str
    field: str


@dataclass(frozen=True)
class _Vectorizer:
    feature_specs: tuple[_FeatureSpec, ...]
    means: dict[str, float]
    stds: dict[str, float]
    encoded_feature_names: tuple[str, ...]


@dataclass(frozen=True)
class _LinearModel:
    intercept: float
    coefficients: tuple[float, ...]
    vectorizer: _Vectorizer
    target_transform: str


@dataclass(frozen=True)
class _BoostedStump:
    feature_index: int
    feature_name: str
    threshold: float
    left_value: float
    right_value: float
    learning_rate: float


@dataclass(frozen=True)
class _BoostedStumpModel:
    base_value: float
    stumps: tuple[_BoostedStump, ...]
    vectorizer: _Vectorizer


@dataclass(frozen=True)
class _CandidateModel:
    name: str
    family: str
    distribution: str
    model: _LinearModel | _BoostedStumpModel | tuple[str, ...] | None
    dispersion_alpha: float
    expected_batters_faced: float
    status: str = "trained"
    reason: str | None = None


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(_json_ready(payload), indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(_json_ready(row), sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _unique_timestamp_run_id(base_time: datetime, run_root: Path) -> str:
    candidate_time = base_time.astimezone(UTC)
    while True:
        run_id = candidate_time.strftime("%Y%m%dT%H%M%SZ")
        if not run_root.joinpath(f"run={run_id}").exists():
            return run_id
        candidate_time += timedelta(seconds=1)


def _latest_run_dir(root: Path, artifact_name: str) -> Path | None:
    if not root.exists():
        return None
    candidates = sorted(
        path.parent
        for path in root.glob(f"**/{artifact_name}")
        if path.is_file() and path.parent.name.startswith("run=")
    )
    return candidates[-1] if candidates else None


def _resolve_run_dir(
    explicit_run_dir: Path | str | None,
    *,
    output_root: Path,
    family: str,
    start_date: date,
    end_date: date,
    artifact_name: str,
) -> Path | None:
    if explicit_run_dir is not None:
        run_dir = Path(explicit_run_dir)
        if not run_dir.joinpath(artifact_name).exists():
            raise FileNotFoundError(f"{run_dir} is missing {artifact_name}.")
        return run_dir
    exact_root = (
        output_root
        / "normalized"
        / family
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
    )
    exact = _latest_run_dir(exact_root, artifact_name)
    if exact is not None:
        return exact
    return _latest_run_dir(output_root / "normalized" / family, artifact_name)


def _row_key(row: dict[str, Any]) -> str:
    value = row.get("training_row_id")
    if value:
        return str(value)
    return "starter-training:{official_date}:{game_pk}:{pitcher_id}".format(
        official_date=row["official_date"],
        game_pk=row["game_pk"],
        pitcher_id=row["pitcher_id"],
    )


def _index_feature_rows(
    run_dir: Path | None,
    artifact_name: str,
    *,
    start_date: date,
    end_date: date,
) -> dict[str, dict[str, Any]]:
    if run_dir is None:
        return {}
    rows = {}
    for row in _load_jsonl(run_dir / artifact_name):
        row_date = date.fromisoformat(str(row["official_date"]))
        if start_date <= row_date <= end_date:
            rows[_row_key(row)] = row
    return rows


def _feature_specs() -> tuple[_FeatureSpec, ...]:
    return (
        _FeatureSpec("pitcher_career_k_rate_shrunk", "pitcher_skill", "pitcher", "career_k_rate_shrunk"),
        _FeatureSpec("pitcher_season_k_rate_shrunk", "pitcher_skill", "pitcher", "season_k_rate_shrunk"),
        _FeatureSpec("pitcher_last_3_starts_k_rate", "pitcher_skill", "pitcher", "last_3_starts_k_rate"),
        _FeatureSpec("pitcher_recent_15d_k_rate", "pitcher_skill", "pitcher", "recent_15d_k_rate"),
        _FeatureSpec("pitcher_career_csw_rate", "pitcher_skill", "pitcher", "career_csw_rate"),
        _FeatureSpec("pitcher_career_swstr_rate", "pitcher_skill", "pitcher", "career_swstr_rate"),
        _FeatureSpec("pitcher_career_whiff_rate", "pitcher_skill", "pitcher", "career_whiff_rate"),
        _FeatureSpec("pitcher_prior_pa_log", "pitcher_skill", "pitcher", "prior_plate_appearance_count"),
        _FeatureSpec("pitcher_average_release_speed", "pitcher_skill", "pitcher", "average_release_speed"),
        _FeatureSpec("pitcher_release_extension", "pitcher_skill", "pitcher", "average_release_extension"),
        _FeatureSpec("matchup_lineup_k_rate", "matchup", "lineup", "projected_lineup_k_rate_vs_pitcher_hand_weighted"),
        _FeatureSpec("matchup_lineup_whiff_rate", "matchup", "lineup", "projected_lineup_whiff_rate_weighted"),
        _FeatureSpec("matchup_lineup_csw_rate", "matchup", "lineup", "projected_lineup_csw_rate_weighted"),
        _FeatureSpec("matchup_lineup_contact_rate", "matchup", "lineup", "projected_lineup_contact_rate_weighted"),
        _FeatureSpec("matchup_arsenal_weakness", "matchup", "lineup", "arsenal_weighted_lineup_pitch_type_weakness"),
        _FeatureSpec("matchup_available_batters", "matchup", "lineup", "available_batter_feature_count"),
        _FeatureSpec("workload_expected_bf", "workload", "workload", "expected_leash_batters_faced"),
        _FeatureSpec("workload_expected_pitches", "workload", "workload", "expected_leash_pitch_count"),
        _FeatureSpec("workload_recent_15d_bf", "workload", "workload", "recent_15d_batters_faced_mean"),
        _FeatureSpec("workload_recent_15d_pitches", "workload", "workload", "recent_15d_pitch_count_mean"),
        _FeatureSpec("workload_last_3_bf", "workload", "workload", "last_3_starts_batters_faced_mean"),
        _FeatureSpec("workload_season_bf", "workload", "workload", "season_batters_faced_mean"),
        _FeatureSpec("workload_team_bf", "workload", "workload", "team_season_batters_faced_mean"),
        _FeatureSpec("workload_reached_22_rate", "workload", "workload", "season_reached_22_batters_rate"),
        _FeatureSpec("context_home", "context", "starter", "home_away=home"),
        _FeatureSpec("context_pitcher_right_handed", "context", "starter", "pitcher_hand=R"),
        _FeatureSpec("context_pitch_clock_era", "context", "starter", "pitch_clock_era=pitch_clock"),
    )


def _source_row(row: dict[str, Any], source: str) -> dict[str, Any]:
    value = row.get(source)
    return value if isinstance(value, dict) else {}


def _raw_feature_value(row: dict[str, Any], spec: _FeatureSpec) -> float | None:
    source_row = _source_row(row, spec.source)
    if "=" in spec.field:
        field_name, expected = spec.field.split("=", 1)
        value = source_row.get(field_name)
        if value is None:
            return None
        return 1.0 if str(value) == expected else 0.0
    value = source_row.get(spec.field)
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and isfinite(float(value)):
        if spec.field in {"prior_plate_appearance_count"}:
            return log(float(value) + 1.0)
        return float(value)
    return None


def _build_vectorizer(rows: list[dict[str, Any]]) -> _Vectorizer:
    specs = _feature_specs()
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    active_specs: list[_FeatureSpec] = []
    for spec in specs:
        values = [
            value
            for row in rows
            if (value := _raw_feature_value(row, spec)) is not None
        ]
        if not values:
            continue
        value_range = max(values) - min(values)
        if value_range <= 1e-12:
            continue
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        means[spec.name] = mean_value
        stds[spec.name] = sqrt(variance) or 1.0
        active_specs.append(spec)
    if not active_specs:
        raise ValueError("No non-constant candidate model features were available.")
    return _Vectorizer(
        feature_specs=tuple(active_specs),
        means=means,
        stds=stds,
        encoded_feature_names=tuple(spec.name for spec in active_specs),
    )


def _encode_row(row: dict[str, Any], vectorizer: _Vectorizer) -> list[float]:
    encoded: list[float] = []
    for spec in vectorizer.feature_specs:
        raw_value = _raw_feature_value(row, spec)
        mean_value = vectorizer.means[spec.name]
        std_value = vectorizer.stds[spec.name]
        value = mean_value if raw_value is None else raw_value
        encoded.append((value - mean_value) / std_value if std_value else 0.0)
    return encoded


def _target(row: dict[str, Any]) -> float:
    return float(_source_row(row, "starter")["starter_strikeouts"])


def _pa_count(row: dict[str, Any]) -> float:
    value = _source_row(row, "starter").get("starter_plate_appearance_count")
    if isinstance(value, (int, float)) and float(value) > 0.0:
        return float(value)
    return _expected_batters_faced(row, fallback=22.0)


def _expected_batters_faced(row: dict[str, Any], fallback: float) -> float:
    workload = _source_row(row, "workload")
    for field_name in (
        "expected_leash_batters_faced",
        "season_batters_faced_mean",
        "team_season_batters_faced_mean",
        "career_batters_faced_mean",
    ):
        value = workload.get(field_name)
        if isinstance(value, (int, float)) and isfinite(float(value)) and float(value) > 0.0:
            return float(value)
    return fallback


def _fit_linear_model(
    train_rows: list[dict[str, Any]],
    *,
    target_transform: str,
) -> _LinearModel:
    vectorizer = _build_vectorizer(train_rows)
    encoded_rows = [_encode_row(row, vectorizer) for row in train_rows]
    feature_count = len(vectorizer.encoded_feature_names)
    matrix_size = feature_count + 1
    xtx = [[0.0 for _ in range(matrix_size)] for _ in range(matrix_size)]
    xty = [0.0 for _ in range(matrix_size)]
    for row, encoded in zip(train_rows, encoded_rows):
        if target_transform == "log_count":
            target_value = log(_target(row) + 0.35)
        elif target_transform == "logit_k_per_pa":
            successes = _target(row)
            attempts = max(1.0, _pa_count(row))
            probability = min(1.0 - 1e-6, max(1e-6, (successes + 0.5) / (attempts + 1.0)))
            target_value = log(probability / (1.0 - probability))
        else:
            target_value = _target(row)
        design_row = [1.0, *encoded]
        for index_i in range(matrix_size):
            xty[index_i] += design_row[index_i] * target_value
            for index_j in range(matrix_size):
                xtx[index_i][index_j] += design_row[index_i] * design_row[index_j]
    for diagonal_index in range(1, matrix_size):
        xtx[diagonal_index][diagonal_index] += RIDGE_ALPHA
    coefficients = _solve_linear_system(xtx, xty)
    return _LinearModel(
        intercept=coefficients[0],
        coefficients=tuple(coefficients[1:]),
        vectorizer=vectorizer,
        target_transform=target_transform,
    )


def _predict_linear_value(row: dict[str, Any], model: _LinearModel) -> float:
    encoded = _encode_row(row, model.vectorizer)
    return model.intercept + sum(
        coefficient * value
        for coefficient, value in zip(model.coefficients, encoded)
    )


def _predict_linear_mean(
    row: dict[str, Any],
    model: _LinearModel,
    *,
    expected_batters_faced: float,
) -> float:
    raw_prediction = _predict_linear_value(row, model)
    if model.target_transform == "log_count":
        return max(MIN_MEAN, exp(raw_prediction) - 0.35)
    if model.target_transform == "logit_k_per_pa":
        probability = 1.0 / (1.0 + exp(-max(-30.0, min(30.0, raw_prediction))))
        return max(MIN_MEAN, probability * _expected_batters_faced(row, fallback=expected_batters_faced))
    return max(MIN_MEAN, raw_prediction)


def _fit_boosted_stumps(train_rows: list[dict[str, Any]]) -> _BoostedStumpModel:
    vectorizer = _build_vectorizer(train_rows)
    encoded_rows = [_encode_row(row, vectorizer) for row in train_rows]
    actuals = [_target(row) for row in train_rows]
    base_value = sum(actuals) / len(actuals)
    predictions = [base_value for _ in actuals]
    stumps: list[_BoostedStump] = []
    feature_count = len(vectorizer.encoded_feature_names)
    for _ in range(min(BOOSTED_STUMP_COUNT, feature_count * 3)):
        residuals = [actual - prediction for actual, prediction in zip(actuals, predictions)]
        best: tuple[float, int, float, float, float] | None = None
        for feature_index in range(feature_count):
            values = [encoded[feature_index] for encoded in encoded_rows]
            threshold = sorted(values)[len(values) // 2]
            left = [residual for value, residual in zip(values, residuals) if value <= threshold]
            right = [residual for value, residual in zip(values, residuals) if value > threshold]
            if not left or not right:
                continue
            left_value = sum(left) / len(left)
            right_value = sum(right) / len(right)
            squared_error = 0.0
            for value, residual in zip(values, residuals):
                fitted = left_value if value <= threshold else right_value
                squared_error += (residual - fitted) ** 2
            if best is None or squared_error < best[0]:
                best = (squared_error, feature_index, threshold, left_value, right_value)
        if best is None:
            break
        _, feature_index, threshold, left_value, right_value = best
        stump = _BoostedStump(
            feature_index=feature_index,
            feature_name=vectorizer.encoded_feature_names[feature_index],
            threshold=threshold,
            left_value=left_value,
            right_value=right_value,
            learning_rate=BOOSTED_STUMP_LEARNING_RATE,
        )
        stumps.append(stump)
        for index, encoded in enumerate(encoded_rows):
            adjustment = left_value if encoded[feature_index] <= threshold else right_value
            predictions[index] += BOOSTED_STUMP_LEARNING_RATE * adjustment
    return _BoostedStumpModel(
        base_value=base_value,
        stumps=tuple(stumps),
        vectorizer=vectorizer,
    )


def _predict_boosted_stump_mean(row: dict[str, Any], model: _BoostedStumpModel) -> float:
    encoded = _encode_row(row, model.vectorizer)
    prediction = model.base_value
    for stump in model.stumps:
        adjustment = (
            stump.left_value
            if encoded[stump.feature_index] <= stump.threshold
            else stump.right_value
        )
        prediction += stump.learning_rate * adjustment
    return max(MIN_MEAN, prediction)


def _poisson_distribution(mean: float, *, minimum_count: int = 0) -> list[float]:
    return _normalized_count_distribution(
        max(MIN_MEAN, mean),
        0.0,
        minimum_count=minimum_count,
    )


def _negative_binomial_distribution(
    mean: float,
    dispersion_alpha: float,
    *,
    minimum_count: int = 0,
) -> list[float]:
    return _normalized_count_distribution(
        max(MIN_MEAN, mean),
        max(0.0, dispersion_alpha),
        minimum_count=minimum_count,
    )


def _binomial_distribution(mean: float, expected_batters_faced: float, *, minimum_count: int = 0) -> list[float]:
    attempts = max(1, int(round(expected_batters_faced)))
    probability = max(1e-6, min(1.0 - 1e-6, mean / attempts))
    probabilities: list[float] = []
    for count in range(attempts + 1):
        coefficient = 1
        for factor in range(1, count + 1):
            coefficient = coefficient * (attempts - factor + 1) // factor
        probabilities.append(
            coefficient
            * (probability**count)
            * ((1.0 - probability) ** (attempts - count))
        )
    while len(probabilities) <= minimum_count:
        probabilities.append(0.0)
    total = sum(probabilities)
    return [value / total for value in probabilities]


def strikeout_line_probabilities(probabilities: list[float], line: float) -> dict[str, float]:
    """Return over/under probabilities for any strikeout line from a count distribution."""
    if not probabilities:
        raise ValueError("probabilities must not be empty.")
    if not isfinite(line) or line < 0.0:
        raise ValueError("line must be a finite non-negative value.")
    threshold = int(floor(line)) + 1
    if threshold >= len(probabilities):
        under_probability = 1.0
    else:
        under_probability = sum(probabilities[:threshold])
    over_probability = max(0.0, 1.0 - under_probability)
    return {
        "line": line,
        "over_probability": over_probability,
        "under_probability": under_probability,
    }


def _line_probability(probabilities: list[float], line: float) -> tuple[float, float]:
    line_probabilities = strikeout_line_probabilities(probabilities, line)
    over_probability = line_probabilities["over_probability"]
    under_probability = line_probabilities["under_probability"]
    return over_probability, under_probability


def _distribution_for_candidate(
    row: dict[str, Any],
    candidate: _CandidateModel,
) -> tuple[float, list[float]]:
    if candidate.status != "trained":
        raise ValueError(f"Candidate {candidate.name} was not trained.")
    if isinstance(candidate.model, _LinearModel):
        mean = _predict_linear_mean(
            row,
            candidate.model,
            expected_batters_faced=candidate.expected_batters_faced,
        )
    elif isinstance(candidate.model, _BoostedStumpModel):
        mean = _predict_boosted_stump_mean(row, candidate.model)
    elif isinstance(candidate.model, tuple):
        means = [
            _distribution_for_candidate(row, _CANDIDATE_REGISTRY[name])[0]
            for name in candidate.model
        ]
        mean = sum(means) / len(means)
    else:
        raise ValueError(f"Candidate {candidate.name} has no trained model.")
    minimum_count = int(_target(row))
    if candidate.distribution == "poisson":
        probabilities = _poisson_distribution(mean, minimum_count=minimum_count)
    elif candidate.distribution == "binomial":
        probabilities = _binomial_distribution(
            mean,
            _expected_batters_faced(row, fallback=candidate.expected_batters_faced),
            minimum_count=minimum_count,
        )
    else:
        probabilities = _negative_binomial_distribution(
            mean,
            candidate.dispersion_alpha,
            minimum_count=minimum_count,
        )
    return mean, probabilities


_CANDIDATE_REGISTRY: dict[str, _CandidateModel] = {}


def _fit_dispersion(rows: list[dict[str, Any]], means: list[float]) -> float:
    denominator = sum(max(MIN_MEAN, mean) ** 2 for mean in means)
    if denominator <= 0.0:
        return 0.0
    numerator = sum(
        (_target(row) - max(MIN_MEAN, mean)) ** 2 - max(MIN_MEAN, mean)
        for row, mean in zip(rows, means)
    )
    fitted = max(0.0, numerator / denominator)
    return 0.0 if fitted <= MIN_DISPERSION_ALPHA else fitted


def _ranked_probability_score(actual_count: int, probabilities: list[float]) -> float:
    cumulative_probability = 0.0
    score = 0.0
    for count, probability in enumerate(probabilities):
        cumulative_probability += probability
        observed_cdf = 1.0 if count >= actual_count else 0.0
        score += (cumulative_probability - observed_cdf) ** 2
    return score / max(1, len(probabilities) - 1)


def _count_interval(probabilities: list[float], lower: float, upper: float) -> tuple[int, int]:
    cumulative = 0.0
    lower_count = 0
    upper_count = len(probabilities) - 1
    for count, probability in enumerate(probabilities):
        cumulative += probability
        if cumulative >= lower:
            lower_count = count
            break
    cumulative = 0.0
    for count, probability in enumerate(probabilities):
        cumulative += probability
        if cumulative >= upper:
            upper_count = count
            break
    return lower_count, upper_count


def _probability_metrics(rows: list[dict[str, Any]], predictions: list[tuple[float, list[float]]]) -> dict[str, Any]:
    line_metrics: dict[str, dict[str, Any]] = {}
    all_line_events: list[dict[str, float]] = []
    for line in COMMON_PROP_LINES:
        brier_values: list[float] = []
        log_losses: list[float] = []
        for row, (_, probabilities) in zip(rows, predictions):
            over_probability, _ = _line_probability(probabilities, line)
            observed = 1.0 if _target(row) > line else 0.0
            brier_values.append((over_probability - observed) ** 2)
            log_losses.append(
                -(
                    observed * log(max(MIN_PROBABILITY, over_probability))
                    + (1.0 - observed) * log(max(MIN_PROBABILITY, 1.0 - over_probability))
                )
            )
            all_line_events.append(
                {
                    "line": line,
                    "probability": over_probability,
                    "observed": observed,
                }
            )
        line_metrics[f"{line:.1f}"] = {
            "brier_score": round(sum(brier_values) / len(brier_values), 6),
            "log_loss": round(sum(log_losses) / len(log_losses), 6),
            "event_count": len(brier_values),
        }
    return {
        "by_common_prop_line": line_metrics,
        "overall": {
            "mean_brier_score": round(
                sum(item["brier_score"] for item in line_metrics.values()) / len(line_metrics),
                6,
            ),
            "mean_log_loss": round(
                sum(item["log_loss"] for item in line_metrics.values()) / len(line_metrics),
                6,
            ),
            "event_count": len(all_line_events),
        },
        "calibration_curve": _calibration_curve(all_line_events),
    }


def _calibration_curve(events: list[dict[str, float]], bin_count: int = 10) -> list[dict[str, Any]]:
    bins: list[dict[str, Any]] = []
    for bin_index in range(bin_count):
        lower = bin_index / bin_count
        upper = (bin_index + 1) / bin_count
        rows = [
            event
            for event in events
            if lower <= event["probability"] <= upper
            and (bin_index == bin_count - 1 or event["probability"] < upper)
        ]
        if not rows:
            bins.append(
                {
                    "bin": bin_index,
                    "probability_min": round(lower, 6),
                    "probability_max": round(upper, 6),
                    "count": 0,
                    "mean_probability": None,
                    "observed_rate": None,
                }
            )
            continue
        bins.append(
            {
                "bin": bin_index,
                "probability_min": round(lower, 6),
                "probability_max": round(upper, 6),
                "count": len(rows),
                "mean_probability": round(
                    sum(row["probability"] for row in rows) / len(rows),
                    6,
                ),
                "observed_rate": round(
                    sum(row["observed"] for row in rows) / len(rows),
                    6,
                ),
            }
        )
    return bins


def _candidate_metrics(
    rows: list[dict[str, Any]],
    candidate: _CandidateModel,
) -> tuple[dict[str, Any], list[tuple[float, list[float]]]]:
    predictions = [_distribution_for_candidate(row, candidate) for row in rows]
    actuals = [_target(row) for row in rows]
    means = [prediction[0] for prediction in predictions]
    errors = [mean - actual for mean, actual in zip(means, actuals)]
    nll_values: list[float] = []
    rps_values: list[float] = []
    exact_hit_count = 0
    interval_hit_count = 0
    predictive_sds: list[float] = []
    for row, (mean, probabilities) in zip(rows, predictions):
        actual = int(_target(row))
        exact_probability = probabilities[actual] if actual < len(probabilities) else 0.0
        nll_values.append(-log(max(MIN_PROBABILITY, exact_probability)))
        rps_values.append(_ranked_probability_score(actual, probabilities))
        mode_count = max(range(len(probabilities)), key=lambda index: probabilities[index])
        exact_hit_count += 1 if mode_count == actual else 0
        lower_count, upper_count = _count_interval(probabilities, 0.10, 0.90)
        interval_hit_count += 1 if lower_count <= actual <= upper_count else 0
        variance = sum(((count - mean) ** 2) * probability for count, probability in enumerate(probabilities))
        predictive_sds.append(sqrt(max(0.0, variance)))
    probability_metrics = _probability_metrics(rows, predictions)
    return (
        {
            "row_count": len(rows),
            "mae": round(sum(abs(error) for error in errors) / len(errors), 6),
            "rmse": round(sqrt(sum(error * error for error in errors) / len(errors)), 6),
            "mean_negative_log_likelihood": round(sum(nll_values) / len(nll_values), 6),
            "mean_ranked_probability_score": round(sum(rps_values) / len(rps_values), 6),
            "exact_mode_accuracy": round(exact_hit_count / len(rows), 6),
            "central_80_interval_coverage": round(interval_hit_count / len(rows), 6),
            "mean_predictive_sd": round(sum(predictive_sds) / len(predictive_sds), 6),
            "probability_metrics": probability_metrics,
        },
        predictions,
    )


def _feature_group_contributions(candidate: _CandidateModel) -> list[dict[str, Any]]:
    group_values = {group: 0.0 for group in ("pitcher_skill", "matchup", "workload", "context")}
    if isinstance(candidate.model, _LinearModel):
        spec_by_name = {
            spec.name: spec
            for spec in candidate.model.vectorizer.feature_specs
        }
        for feature_name, coefficient in zip(
            candidate.model.vectorizer.encoded_feature_names,
            candidate.model.coefficients,
        ):
            spec = spec_by_name[feature_name]
            group_values[spec.group] += abs(coefficient)
    elif isinstance(candidate.model, _BoostedStumpModel):
        spec_by_name = {
            spec.name: spec
            for spec in candidate.model.vectorizer.feature_specs
        }
        for stump in candidate.model.stumps:
            spec = spec_by_name[stump.feature_name]
            group_values[spec.group] += abs(stump.left_value - stump.right_value) * stump.learning_rate
    total = sum(group_values.values())
    return [
        {
            "feature_group": group,
            "absolute_contribution": round(value, 6),
            "share": None if total == 0.0 else round(value / total, 6),
        }
        for group, value in sorted(group_values.items())
    ]


def _candidate_artifact(candidate: _CandidateModel) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": candidate.name,
        "family": candidate.family,
        "distribution": candidate.distribution,
        "dispersion_alpha": round(candidate.dispersion_alpha, 6),
        "expected_batters_faced_fallback": round(candidate.expected_batters_faced, 6),
        "status": candidate.status,
        "reason": candidate.reason,
    }
    if isinstance(candidate.model, _LinearModel):
        payload["model"] = {
            "type": "ridge_linear",
            "target_transform": candidate.model.target_transform,
            "intercept": round(candidate.model.intercept, 10),
            "coefficients": {
                feature_name: round(coefficient, 10)
                for feature_name, coefficient in zip(
                    candidate.model.vectorizer.encoded_feature_names,
                    candidate.model.coefficients,
                )
            },
            "feature_stats": {
                spec.name: {
                    "group": spec.group,
                    "source": spec.source,
                    "field": spec.field,
                    "mean": round(candidate.model.vectorizer.means[spec.name], 10),
                    "std": round(candidate.model.vectorizer.stds[spec.name], 10),
                }
                for spec in candidate.model.vectorizer.feature_specs
            },
        }
    elif isinstance(candidate.model, _BoostedStumpModel):
        payload["model"] = {
            "type": "boosted_decision_stumps",
            "base_value": round(candidate.model.base_value, 10),
            "stumps": [
                {
                    "feature_name": stump.feature_name,
                    "feature_index": stump.feature_index,
                    "threshold": round(stump.threshold, 10),
                    "left_value": round(stump.left_value, 10),
                    "right_value": round(stump.right_value, 10),
                    "learning_rate": stump.learning_rate,
                }
                for stump in candidate.model.stumps
            ],
            "feature_stats": {
                spec.name: {
                    "group": spec.group,
                    "source": spec.source,
                    "field": spec.field,
                    "mean": round(candidate.model.vectorizer.means[spec.name], 10),
                    "std": round(candidate.model.vectorizer.stds[spec.name], 10),
                }
                for spec in candidate.model.vectorizer.feature_specs
            },
        }
    elif isinstance(candidate.model, tuple):
        payload["model"] = {
            "type": "mean_blend",
            "members": list(candidate.model),
        }
    return payload


def _select_candidate(report_candidates: dict[str, dict[str, Any]]) -> str:
    trained = [
        (name, payload)
        for name, payload in report_candidates.items()
        if payload.get("status") == "trained"
    ]
    if not trained:
        raise ValueError("No trained candidate models were available for selection.")
    return min(
        trained,
        key=lambda item: (
            item[1]["splits"]["validation"]["probability_metrics"]["overall"]["mean_log_loss"],
            item[1]["splits"]["validation"]["mean_negative_log_likelihood"],
            item[1]["splits"]["validation"]["rmse"],
            item[0],
        ),
    )[0]


def _build_joined_rows(
    *,
    start_date: date,
    end_date: date,
    output_root: Path,
    dataset_run_dir: Path | str | None,
    pitcher_skill_run_dir: Path | str | None,
    lineup_matchup_run_dir: Path | str | None,
    workload_leash_run_dir: Path | str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    starter_run = _resolve_run_dir(
        dataset_run_dir,
        output_root=output_root,
        family="starter_strikeout_training_dataset",
        start_date=start_date,
        end_date=end_date,
        artifact_name="starter_game_training_dataset.jsonl",
    )
    if starter_run is None:
        raise FileNotFoundError("No starter-game training dataset run was found.")
    pitcher_run = _resolve_run_dir(
        pitcher_skill_run_dir,
        output_root=output_root,
        family="pitcher_skill_features",
        start_date=start_date,
        end_date=end_date,
        artifact_name="pitcher_skill_features.jsonl",
    )
    lineup_run = _resolve_run_dir(
        lineup_matchup_run_dir,
        output_root=output_root,
        family="lineup_matchup_features",
        start_date=start_date,
        end_date=end_date,
        artifact_name="lineup_matchup_features.jsonl",
    )
    workload_run = _resolve_run_dir(
        workload_leash_run_dir,
        output_root=output_root,
        family="workload_leash_features",
        start_date=start_date,
        end_date=end_date,
        artifact_name="workload_leash_features.jsonl",
    )
    pitcher_rows = _index_feature_rows(
        pitcher_run,
        "pitcher_skill_features.jsonl",
        start_date=start_date,
        end_date=end_date,
    )
    lineup_rows = _index_feature_rows(
        lineup_run,
        "lineup_matchup_features.jsonl",
        start_date=start_date,
        end_date=end_date,
    )
    workload_rows = _index_feature_rows(
        workload_run,
        "workload_leash_features.jsonl",
        start_date=start_date,
        end_date=end_date,
    )
    rows: list[dict[str, Any]] = []
    for starter in _load_jsonl(starter_run / "starter_game_training_dataset.jsonl"):
        row_date = date.fromisoformat(str(starter["official_date"]))
        if not start_date <= row_date <= end_date:
            continue
        if starter.get("starter_strikeouts") is None:
            continue
        key = _row_key(starter)
        rows.append(
            {
                "training_row_id": key,
                "official_date": starter["official_date"],
                "starter": starter,
                "pitcher": pitcher_rows.get(key, {}),
                "lineup": lineup_rows.get(key, {}),
                "workload": workload_rows.get(key, {}),
            }
        )
    if not rows:
        raise ValueError("No starter rows with strikeout targets were available.")
    source_summary = {
        "starter_dataset_run_dir": starter_run,
        "pitcher_skill_run_dir": pitcher_run,
        "lineup_matchup_run_dir": lineup_run,
        "workload_leash_run_dir": workload_run,
        "joined_rows": len(rows),
        "pitcher_skill_matches": sum(1 for row in rows if row["pitcher"]),
        "lineup_matchup_matches": sum(1 for row in rows if row["lineup"]),
        "workload_leash_matches": sum(1 for row in rows if row["workload"]),
    }
    return sorted(rows, key=lambda row: (row["official_date"], row["training_row_id"])), source_summary


def _rows_for_dates(rows: list[dict[str, Any]], dates: list[str]) -> list[dict[str, Any]]:
    date_set = set(dates)
    return [row for row in rows if str(row["official_date"]) in date_set]


def _train_candidates(train_rows: list[dict[str, Any]]) -> list[_CandidateModel]:
    expected_bf_values = [_pa_count(row) for row in train_rows]
    expected_bf_fallback = sum(expected_bf_values) / len(expected_bf_values)

    poisson_model = _fit_linear_model(train_rows, target_transform="log_count")
    poisson_candidate = _CandidateModel(
        name="poisson_glm_count_baseline",
        family="count_baseline_poisson_glm",
        distribution="poisson",
        model=poisson_model,
        dispersion_alpha=0.0,
        expected_batters_faced=expected_bf_fallback,
    )
    nb_means = [
        _predict_linear_mean(row, poisson_model, expected_batters_faced=expected_bf_fallback)
        for row in train_rows
    ]
    negative_binomial_candidate = _CandidateModel(
        name="negative_binomial_glm_count_baseline",
        family="count_baseline_negative_binomial_glm",
        distribution="negative_binomial",
        model=poisson_model,
        dispersion_alpha=_fit_dispersion(train_rows, nb_means),
        expected_batters_faced=expected_bf_fallback,
    )
    pa_model = _fit_linear_model(train_rows, target_transform="logit_k_per_pa")
    pa_candidate = _CandidateModel(
        name="plate_appearance_logistic_rate",
        family="plate_appearance_logistic_probability",
        distribution="binomial",
        model=pa_model,
        dispersion_alpha=0.0,
        expected_batters_faced=expected_bf_fallback,
    )
    stump_model = _fit_boosted_stumps(train_rows)
    stump_means = [_predict_boosted_stump_mean(row, stump_model) for row in train_rows]
    tree_candidate = _CandidateModel(
        name="boosted_stump_tree_ensemble",
        family="tree_ensemble_repo_approved_equivalent",
        distribution="negative_binomial",
        model=stump_model,
        dispersion_alpha=_fit_dispersion(train_rows, stump_means),
        expected_batters_faced=expected_bf_fallback,
    )
    neural_candidate = _CandidateModel(
        name="neural_sequence_challenger",
        family="neural_sequence_challenger",
        distribution="negative_binomial",
        model=None,
        dispersion_alpha=0.0,
        expected_batters_faced=expected_bf_fallback,
        status="skipped",
        reason=(
            "Skipped because the current AGE-287/288/289/290 artifacts are "
            "tabular aggregate rows, not pitch or batter sequences; training a "
            "neural challenger here would add dependency and architecture scope "
            "without sequence-shaped inputs."
        ),
    )
    return [
        poisson_candidate,
        negative_binomial_candidate,
        pa_candidate,
        tree_candidate,
        neural_candidate,
    ]


def _add_ensemble_candidate(
    candidates: list[_CandidateModel],
    validation_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
) -> list[_CandidateModel]:
    global _CANDIDATE_REGISTRY
    _CANDIDATE_REGISTRY = {candidate.name: candidate for candidate in candidates}
    trained_names = [candidate.name for candidate in candidates if candidate.status == "trained"]
    if len(trained_names) < 2:
        return candidates
    ranked = sorted(
        trained_names,
        key=lambda name: _candidate_metrics(
            validation_rows,
            _CANDIDATE_REGISTRY[name],
        )[0]["probability_metrics"]["overall"]["mean_log_loss"],
    )
    members = tuple(ranked[:2])
    train_means: list[float] = []
    for row in train_rows:
        member_means = [
            _distribution_for_candidate(row, _CANDIDATE_REGISTRY[name])[0]
            for name in members
        ]
        train_means.append(sum(member_means) / len(member_means))
    ensemble = _CandidateModel(
        name="validation_top_two_mean_blend",
        family="ensemble_blend_candidate",
        distribution="negative_binomial",
        model=members,
        dispersion_alpha=_fit_dispersion(train_rows, train_means),
        expected_batters_faced=candidates[0].expected_batters_faced,
    )
    candidates = [*candidates, ensemble]
    _CANDIDATE_REGISTRY = {candidate.name: candidate for candidate in candidates}
    return candidates


def _model_output_rows(
    rows: list[dict[str, Any]],
    candidate: _CandidateModel,
    *,
    split_by_date: dict[str, str],
    generated_at: datetime,
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        mean, probabilities = _distribution_for_candidate(row, candidate)
        lower_count, upper_count = _count_interval(probabilities, 0.10, 0.90)
        variance = sum(((count - mean) ** 2) * probability for count, probability in enumerate(probabilities))
        ladder = []
        for line in COMMON_PROP_LINES:
            line_probabilities = strikeout_line_probabilities(probabilities, line)
            ladder.append(
                {
                    "line": line,
                    "over_probability": round(line_probabilities["over_probability"], 6),
                    "under_probability": round(line_probabilities["under_probability"], 6),
                }
            )
        output_rows.append(
            {
                "training_row_id": row["training_row_id"],
                "official_date": row["official_date"],
                "split": split_by_date[str(row["official_date"])],
                "game_pk": _source_row(row, "starter").get("game_pk"),
                "pitcher_id": _source_row(row, "starter").get("pitcher_id"),
                "pitcher_name": _source_row(row, "starter").get("pitcher_name"),
                "actual_strikeouts": int(_target(row)),
                "selected_candidate": candidate.name,
                "point_projection": round(mean, 6),
                "probability_distribution": [
                    {
                        "strikeouts": count,
                        "probability": round(probability, 8),
                    }
                    for count, probability in enumerate(probabilities)
                ],
                "over_under_probabilities": ladder,
                "line_probability_contract": {
                    "supports_arbitrary_lines": True,
                    "rule": "over_probability = P(strikeouts > line); under_probability = P(strikeouts <= floor(line))",
                    "source": "probability_distribution",
                },
                "confidence": {
                    "predictive_sd": round(sqrt(max(0.0, variance)), 6),
                    "central_80_interval": [lower_count, upper_count],
                    "distribution": candidate.distribution,
                    "dispersion_alpha": round(candidate.dispersion_alpha, 6),
                },
            }
        )
    return output_rows


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Candidate Strikeout Model Comparison",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Window: `{report['date_window']['start_date']}` to `{report['date_window']['end_date']}`",
        f"- Selected candidate: `{report['selection']['selected_candidate']}`",
        f"- Selection metric: `{report['selection']['primary_metric']}`",
        "",
        "## Candidate Metrics",
        "",
        "| Candidate | Status | Validation Log Loss | Validation RMSE | Held-Out Log Loss | Held-Out RMSE |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, candidate in sorted(report["candidates"].items()):
        if candidate["status"] != "trained":
            lines.append(f"| `{name}` | {candidate['status']} | n/a | n/a | n/a | n/a |")
            continue
        validation = candidate["splits"]["validation"]
        held_out = candidate["splits"]["held_out"]
        lines.append(
            "| `{name}` | trained | {validation_log_loss:.6f} | {validation_rmse:.6f} | "
            "{held_log_loss:.6f} | {held_rmse:.6f} |".format(
                name=name,
                validation_log_loss=validation["probability_metrics"]["overall"]["mean_log_loss"],
                validation_rmse=validation["rmse"],
                held_log_loss=held_out["probability_metrics"]["overall"]["mean_log_loss"],
                held_rmse=held_out["rmse"],
            )
        )
    lines.extend(
        [
            "",
            "## Selection Rationale",
            "",
            report["selection"]["rationale"],
            "",
            "## Feature Group Contributions",
            "",
            "| Feature Group | Share |",
            "| --- | ---: |",
        ]
    )
    selected = report["candidates"][report["selection"]["selected_candidate"]]
    for contribution in selected["feature_group_contributions"]:
        share = contribution["share"]
        lines.append(
            f"| {contribution['feature_group']} | {'n/a' if share is None else f'{share:.3f}'} |"
        )
    lines.extend(
        [
            "",
            "## Neural Challenger",
            "",
            report["candidates"]["neural_sequence_challenger"]["reason"],
            "",
        ]
    )
    return "\n".join(lines)


def _rerun_command(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str,
    dataset_run_dir: Path | str | None,
    pitcher_skill_run_dir: Path | str | None,
    lineup_matchup_run_dir: Path | str | None,
    workload_leash_run_dir: Path | str | None,
) -> str:
    parts = [
        "uv",
        "run",
        "python",
        "-m",
        "mlb_props_stack",
        "train-candidate-strikeout-models",
        "--start-date",
        start_date.isoformat(),
        "--end-date",
        end_date.isoformat(),
        "--output-dir",
        str(output_dir),
    ]
    optional = (
        ("--dataset-run-dir", dataset_run_dir),
        ("--pitcher-skill-run-dir", pitcher_skill_run_dir),
        ("--lineup-matchup-run-dir", lineup_matchup_run_dir),
        ("--workload-leash-run-dir", workload_leash_run_dir),
    )
    for flag, value in optional:
        if value is not None:
            parts.extend([flag, str(value)])
    return " ".join(quote(part) for part in parts)


def train_candidate_strikeout_models(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str = "data",
    dataset_run_dir: Path | str | None = None,
    pitcher_skill_run_dir: Path | str | None = None,
    lineup_matchup_run_dir: Path | str | None = None,
    workload_leash_run_dir: Path | str | None = None,
    now: Callable[[], datetime] = utc_now,
) -> CandidateStrikeoutModelTrainingResult:
    """Train and compare candidate starter strikeout model families."""
    output_root = Path(output_dir)
    rows, source_summary = _build_joined_rows(
        start_date=start_date,
        end_date=end_date,
        output_root=output_root,
        dataset_run_dir=dataset_run_dir,
        pitcher_skill_run_dir=pitcher_skill_run_dir,
        lineup_matchup_run_dir=lineup_matchup_run_dir,
        workload_leash_run_dir=workload_leash_run_dir,
    )
    date_splits = _split_dates([str(row["official_date"]) for row in rows])
    split_by_date = {
        split_date: split_name
        for split_name, split_dates in date_splits.items()
        for split_date in split_dates
    }
    train_rows = _rows_for_dates(rows, date_splits["train"])
    validation_rows = _rows_for_dates(rows, date_splits["validation"])
    test_rows = _rows_for_dates(rows, date_splits["test"])
    held_out_rows = [*validation_rows, *test_rows]
    if not train_rows or not validation_rows or not test_rows:
        raise ValueError("Date splits must leave train, validation, and test rows.")

    candidates = _train_candidates(train_rows)
    candidates = _add_ensemble_candidate(candidates, validation_rows, train_rows)
    report_candidates: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if candidate.status != "trained":
            report_candidates[candidate.name] = _candidate_artifact(candidate)
            continue
        split_metrics: dict[str, Any] = {}
        for split_name, split_rows in (
            ("train", train_rows),
            ("validation", validation_rows),
            ("test", test_rows),
            ("held_out", held_out_rows),
        ):
            split_metrics[split_name], _ = _candidate_metrics(split_rows, candidate)
        report_candidates[candidate.name] = {
            **_candidate_artifact(candidate),
            "splits": split_metrics,
            "feature_group_contributions": _feature_group_contributions(candidate),
        }

    selected_candidate_name = _select_candidate(report_candidates)
    selected_candidate = _CANDIDATE_REGISTRY[selected_candidate_name]
    generated_at = now().astimezone(UTC)
    run_root = (
        output_root
        / "normalized"
        / "candidate_strikeout_models"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
    )
    run_id = _unique_timestamp_run_id(generated_at, run_root)
    normalized_root = run_root / f"run={run_id}"
    report_path = normalized_root / "model_comparison.json"
    report_markdown_path = normalized_root / "model_comparison.md"
    selected_model_path = normalized_root / "selected_model.json"
    model_outputs_path = normalized_root / "model_outputs.jsonl"
    reproducibility_notes_path = normalized_root / "reproducibility_notes.md"
    rerun_command = _rerun_command(
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        dataset_run_dir=dataset_run_dir,
        pitcher_skill_run_dir=pitcher_skill_run_dir,
        lineup_matchup_run_dir=lineup_matchup_run_dir,
        workload_leash_run_dir=workload_leash_run_dir,
    )
    validation_metrics = report_candidates[selected_candidate_name]["splits"]["validation"]
    runner_up = [
        name
        for name, payload in sorted(
            report_candidates.items(),
            key=lambda item: (
                item[1].get("splits", {})
                .get("validation", {})
                .get("probability_metrics", {})
                .get("overall", {})
                .get("mean_log_loss", float("inf")),
                item[0],
            ),
        )
        if name != selected_candidate_name and payload.get("status") == "trained"
    ]
    report = {
        "report_version": MODEL_FAMILY_REPORT_VERSION,
        "selected_model_version": SELECTED_MODEL_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "date_window": {
            "start_date": start_date,
            "end_date": end_date,
        },
        "date_splits": date_splits,
        "row_counts": {
            "all": len(rows),
            "train": len(train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
            "held_out": len(held_out_rows),
        },
        "source_artifacts": source_summary,
        "candidate_families_required": [
            "count baseline: Poisson and negative-binomial GLM",
            "plate-appearance logistic probability by expected batters faced",
            "tree ensemble: boosted stump repo-approved equivalent",
            "neural/sequence challenger when sequence-shaped data exists",
            "ensemble/blend candidate",
        ],
        "selection": {
            "selected_candidate": selected_candidate_name,
            "runner_up_candidate": runner_up[0] if runner_up else None,
            "primary_metric": "validation common-line mean log loss",
            "rationale": (
                f"{selected_candidate_name} had the best validation common-line "
                "mean log loss among trained candidates, with validation RMSE "
                f"{validation_metrics['rmse']:.6f}. Test and held-out metrics are "
                "reported for audit but were not used to choose the candidate."
            ),
        },
        "candidates": report_candidates,
        "reproducibility": {
            "rerun_command": rerun_command,
            "notes_path": reproducibility_notes_path,
        },
        "scope_guardrails": {
            "time_aware_training_only": True,
            "betting_decisions_included": False,
            "selection_uses_validation_not_preference": True,
            "neural_net_requires_validation_win_before_selection": True,
        },
        "output_contract": {
            "supports_arbitrary_strikeout_lines": True,
            "line_probability_function": "mlb_props_stack.candidate_models.strikeout_line_probabilities",
            "common_prop_lines_reported": list(COMMON_PROP_LINES),
        },
    }
    selected_model = {
        "model_version": SELECTED_MODEL_VERSION,
        "run_id": run_id,
        "selected_candidate": selected_candidate_name,
        "selected_candidate_artifact": _candidate_artifact(selected_candidate),
        "feature_group_contributions": report_candidates[selected_candidate_name][
            "feature_group_contributions"
        ],
        "source_artifacts": source_summary,
        "date_splits": date_splits,
        "line_probability_contract": report["output_contract"],
        "model_comparison_path": report_path,
        "model_outputs_path": model_outputs_path,
    }
    _write_json(report_path, report)
    _write_text(report_markdown_path, _render_markdown(report))
    _write_json(selected_model_path, selected_model)
    _write_jsonl(
        model_outputs_path,
        _model_output_rows(
            rows,
            selected_candidate,
            split_by_date=split_by_date,
            generated_at=generated_at,
        ),
    )
    _write_text(
        reproducibility_notes_path,
        "\n".join(
            [
                "# Candidate Strikeout Models Reproducibility",
                "",
                f"- Run ID: `{run_id}`",
                f"- Generated at: `{generated_at.isoformat().replace('+00:00', 'Z')}`",
                f"- Rerun command: `{rerun_command}`",
                "",
                "The command trains candidate projection models only. It does not "
                "price sportsbook lines, size wagers, or mark outputs as betting-ready.",
                "",
            ]
        ),
    )
    return CandidateStrikeoutModelTrainingResult(
        start_date=start_date,
        end_date=end_date,
        run_id=run_id,
        selected_candidate=selected_candidate_name,
        row_count=len(rows),
        report_path=report_path,
        report_markdown_path=report_markdown_path,
        selected_model_path=selected_model_path,
        model_outputs_path=model_outputs_path,
        reproducibility_notes_path=reproducibility_notes_path,
    )
