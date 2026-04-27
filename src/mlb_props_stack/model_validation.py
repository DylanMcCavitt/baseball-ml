"""Model-only walk-forward validation for starter strikeout projections."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from math import floor, isfinite, log, sqrt
from pathlib import Path
from shlex import quote
from statistics import median
from typing import Any, Callable

from .candidate_models import (
    COMMON_PROP_LINES,
    MIN_PROBABILITY,
    _add_ensemble_candidate,
    _build_joined_rows,
    _candidate_metrics,
    _count_interval,
    _distribution_for_candidate,
    _feature_group_contributions,
    _source_row,
    _target,
    _train_candidates,
    _unique_timestamp_run_id,
    _write_json,
    _write_jsonl,
    _write_text,
    strikeout_line_probabilities,
)
from .ingest.mlb_stats_api import utc_now

VALIDATION_REPORT_VERSION = "model_only_walk_forward_validation_v1"
DEFAULT_FIRST_VALIDATION_SEASON = 2023
DEFAULT_MAX_TRAINING_SEASONS = 6
RECENCY_WINDOWS = (3, 5)


@dataclass(frozen=True)
class ModelOnlyWalkForwardValidationResult:
    """Filesystem output summary for one model-only validation run."""

    start_date: date
    end_date: date
    run_id: str
    split_count: int
    prediction_count: int
    recommendation: str
    report_path: Path
    report_markdown_path: Path
    predictions_path: Path
    reproducibility_notes_path: Path


def _season(row: dict[str, Any]) -> int:
    starter = _source_row(row, "starter")
    value = starter.get("season")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return date.fromisoformat(str(row["official_date"])).year


def _starter_pa(row: dict[str, Any]) -> float:
    value = _source_row(row, "starter").get("starter_plate_appearance_count")
    if isinstance(value, (int, float)) and isfinite(float(value)) and float(value) > 0.0:
        return float(value)
    return 0.0


def _attach_prior_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pitcher_history: dict[str, dict[str, Any]] = {}
    enriched: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (item["official_date"], item["training_row_id"])):
        starter = _source_row(row, "starter")
        pitcher_id = str(starter.get("pitcher_id") or "")
        row_date = date.fromisoformat(str(row["official_date"]))
        history = pitcher_history.setdefault(
            pitcher_id,
            {
                "strikeouts": 0.0,
                "plate_appearances": 0.0,
                "starts": 0,
                "last_date": None,
            },
        )
        prior_pa = float(history["plate_appearances"])
        prior_k_rate = None if prior_pa <= 0.0 else float(history["strikeouts"]) / prior_pa
        last_date = history["last_date"]
        rest_days = None if last_date is None else max(0, (row_date - last_date).days)
        row["validation_prior_context"] = {
            "prior_pitcher_k_per_pa": prior_k_rate,
            "prior_start_count": int(history["starts"]),
            "prior_plate_appearances": prior_pa,
            "rest_days": rest_days,
        }
        enriched.append(row)
        history["strikeouts"] = float(history["strikeouts"]) + _target(row)
        history["plate_appearances"] = prior_pa + _starter_pa(row)
        history["starts"] = int(history["starts"]) + 1
        history["last_date"] = row_date
    return sorted(enriched, key=lambda item: (item["official_date"], item["training_row_id"]))


def _split_rows_for_validation(
    rows: list[dict[str, Any]],
    *,
    validation_season: int,
    max_training_seasons: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    first_training_season = max(
        min(_season(row) for row in rows),
        validation_season - max_training_seasons,
    )
    train_rows = [
        row
        for row in rows
        if first_training_season <= _season(row) < validation_season
    ]
    validation_rows = [
        row
        for row in rows
        if _season(row) == validation_season
    ]
    return train_rows, validation_rows, first_training_season


def _validation_seasons(
    rows: list[dict[str, Any]],
    *,
    first_validation_season: int,
    max_training_seasons: int,
) -> list[int]:
    seasons = sorted({_season(row) for row in rows})
    selected: list[int] = []
    for validation_season in seasons:
        if validation_season < first_validation_season:
            continue
        train_rows, validation_rows, _ = _split_rows_for_validation(
            rows,
            validation_season=validation_season,
            max_training_seasons=max_training_seasons,
        )
        if train_rows and validation_rows:
            selected.append(validation_season)
    return selected


def _select_candidate_from_training_window(
    train_rows: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    selection_season = max(_season(row) for row in train_rows)
    selection_rows = [
        row
        for row in train_rows
        if _season(row) == selection_season
    ]
    candidates = _train_candidates(train_rows)
    candidates = _add_ensemble_candidate(candidates, selection_rows, train_rows)
    trained_candidates = [
        candidate for candidate in candidates if candidate.status == "trained"
    ]
    selection_metrics = {
        candidate.name: _candidate_metrics(selection_rows, candidate)[0]
        for candidate in trained_candidates
    }
    selected = min(
        trained_candidates,
        key=lambda candidate: (
            selection_metrics[candidate.name]["probability_metrics"]["overall"]["mean_log_loss"],
            selection_metrics[candidate.name]["mean_negative_log_likelihood"],
            selection_metrics[candidate.name]["rmse"],
            candidate.name,
        ),
    )
    diagnostics = {
        "selection_basis": "latest training season only; validation season was held out",
        "selection_season": selection_season,
        "selection_row_count": len(selection_rows),
        "selected_candidate": selected.name,
        "candidate_count": len(trained_candidates),
        "feature_group_contributions": _feature_group_contributions(selected),
    }
    return selected, diagnostics


def _line_bucket(line: float) -> str:
    if line <= 4.5:
        return "low_line_2.5_to_4.5"
    if line <= 6.5:
        return "middle_line_5.5_to_6.5"
    return "high_line_7.5_plus"


def _confidence_bucket(confidence: float) -> str:
    lower = min(0.95, max(0.50, floor(confidence * 10.0) / 10.0))
    upper = min(1.00, lower + 0.10)
    return f"{lower:.1f}_to_{upper:.1f}"


def _pitcher_tier(row: dict[str, Any]) -> str:
    pitcher = _source_row(row, "pitcher")
    for field in ("season_k_rate_shrunk", "career_k_rate_shrunk"):
        value = pitcher.get(field)
        if isinstance(value, (int, float)) and isfinite(float(value)):
            k_rate = float(value)
            if k_rate >= 0.27:
                return "high_k_pitcher"
            if k_rate >= 0.22:
                return "middle_k_pitcher"
            return "low_k_pitcher"
    prior = row.get("validation_prior_context")
    if isinstance(prior, dict):
        value = prior.get("prior_pitcher_k_per_pa")
        starts = prior.get("prior_start_count")
        if (
            isinstance(value, (int, float))
            and isfinite(float(value))
            and isinstance(starts, int)
            and starts >= 3
        ):
            k_rate = float(value)
            if k_rate >= 0.27:
                return "high_k_pitcher_prior_history"
            if k_rate >= 0.22:
                return "middle_k_pitcher_prior_history"
            return "low_k_pitcher_prior_history"
    return "unknown_pitcher_tier"


def _handedness_bucket(row: dict[str, Any]) -> str:
    hand = _source_row(row, "starter").get("pitcher_hand") or _source_row(row, "lineup").get("pitcher_hand")
    if hand in {"L", "R"}:
        return f"pitcher_hand_{hand}"
    return "pitcher_hand_unknown"


def _workload_bucket(row: dict[str, Any]) -> str:
    workload = _source_row(row, "workload")
    value = workload.get("expected_leash_batters_faced")
    if not isinstance(value, (int, float)) or not isfinite(float(value)):
        value = _source_row(row, "starter").get("starter_plate_appearance_count")
    if not isinstance(value, (int, float)) or not isfinite(float(value)):
        return "unknown_workload"
    batters = float(value)
    if batters < 21.0:
        return "short_workload_under_21_bf"
    if batters <= 24.0:
        return "standard_workload_21_to_24_bf"
    return "deep_workload_over_24_bf"


def _rest_bucket(row: dict[str, Any]) -> str:
    value = _source_row(row, "workload").get("rest_bucket")
    if value:
        return str(value)
    prior = row.get("validation_prior_context")
    if not isinstance(prior, dict):
        return "unknown_rest_or_no_workload_feature"
    rest_days = prior.get("rest_days")
    if rest_days is None:
        return "no_prior_start"
    if not isinstance(rest_days, (int, float)):
        return "unknown_rest_or_no_workload_feature"
    if rest_days <= 3:
        return "short_rest_prior_history"
    if rest_days <= 5:
        return "standard_rest_prior_history"
    if rest_days <= 14:
        return "extra_rest_prior_history"
    return "long_layoff_prior_history"


def _validation_context_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pitcher_tier_context": (
            "Uses pitcher-skill feature tiers when feature artifacts are joined; "
            "otherwise uses each pitcher's prior-only starter-game strikeouts per "
            "plate appearance with at least three prior starts."
        ),
        "rest_layoff_context": (
            "Uses workload/leash rest buckets when joined; otherwise derives "
            "prior-only rest days from the pitcher's starter-game history."
        ),
        "unknown_pitcher_tier_rows": sum(
            1 for record in records if record["pitcher_tier"] == "unknown_pitcher_tier"
        ),
        "unknown_rest_rows": sum(
            1
            for record in records
            if record["rest_layoff_bucket"] == "unknown_rest_or_no_workload_feature"
        ),
    }


def _rule_environment(row: dict[str, Any]) -> str:
    value = _source_row(row, "starter").get("pitch_clock_era")
    return str(value) if value else "unknown_rule_environment"


def _prediction_record(
    *,
    row: dict[str, Any],
    split_id: str,
    validation_season: int,
    selected_candidate: str,
    mean: float,
    probabilities: list[float],
) -> dict[str, Any]:
    actual = int(_target(row))
    lower_count, upper_count = _count_interval(probabilities, 0.10, 0.90)
    variance = sum(
        ((count - mean) ** 2) * probability
        for count, probability in enumerate(probabilities)
    )
    common_lines = []
    for line in COMMON_PROP_LINES:
        line_probabilities = strikeout_line_probabilities(probabilities, line)
        over_probability = line_probabilities["over_probability"]
        under_probability = line_probabilities["under_probability"]
        selected_side_probability = max(over_probability, under_probability)
        selected_side = "over" if over_probability >= under_probability else "under"
        if selected_side == "over":
            selected_side_hit = actual > line
        else:
            selected_side_hit = actual <= line
        common_lines.append(
            {
                "line": line,
                "line_bucket": _line_bucket(line),
                "over_probability": round(over_probability, 6),
                "under_probability": round(under_probability, 6),
                "selected_side": selected_side,
                "selected_side_probability": round(selected_side_probability, 6),
                "selected_side_hit": selected_side_hit,
                "confidence_bucket": _confidence_bucket(selected_side_probability),
            }
        )
    exact_probability = probabilities[actual] if actual < len(probabilities) else 0.0
    return {
        "split_id": split_id,
        "validation_season": validation_season,
        "training_row_id": row["training_row_id"],
        "official_date": row["official_date"],
        "game_pk": _source_row(row, "starter").get("game_pk"),
        "pitcher_id": _source_row(row, "starter").get("pitcher_id"),
        "pitcher_name": _source_row(row, "starter").get("pitcher_name"),
        "selected_candidate": selected_candidate,
        "actual_strikeouts": actual,
        "point_projection": round(mean, 6),
        "projection_error": round(mean - actual, 6),
        "absolute_error": round(abs(mean - actual), 6),
        "exact_count_probability": round(exact_probability, 8),
        "exact_count_log_loss": round(-log(max(MIN_PROBABILITY, exact_probability)), 6),
        "predictive_sd": round(sqrt(max(0.0, variance)), 6),
        "central_80_interval": [lower_count, upper_count],
        "pitcher_tier": _pitcher_tier(row),
        "handedness_matchup_bucket": _handedness_bucket(row),
        "workload_bucket": _workload_bucket(row),
        "rest_layoff_bucket": _rest_bucket(row),
        "rule_environment": _rule_environment(row),
        "line_probabilities": common_lines,
    }


def _metrics_from_prediction_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "row_count": 0,
            "mae": None,
            "rmse": None,
            "mean_count_log_loss": None,
            "probability_metrics": {},
        }
    errors = [float(record["projection_error"]) for record in records]
    line_events: list[dict[str, Any]] = []
    for record in records:
        actual = float(record["actual_strikeouts"])
        for event in record["line_probabilities"]:
            observed_over = 1.0 if actual > float(event["line"]) else 0.0
            over_probability = float(event["over_probability"])
            selected_probability = float(event["selected_side_probability"])
            line_events.append(
                {
                    "line": float(event["line"]),
                    "line_bucket": event["line_bucket"],
                    "confidence_bucket": event["confidence_bucket"],
                    "over_probability": over_probability,
                    "observed_over": observed_over,
                    "selected_side_probability": selected_probability,
                    "selected_side_hit": 1.0 if event["selected_side_hit"] else 0.0,
                }
            )
    return {
        "row_count": len(records),
        "mae": round(sum(abs(error) for error in errors) / len(errors), 6),
        "rmse": round(sqrt(sum(error * error for error in errors) / len(errors)), 6),
        "mean_bias": round(sum(errors) / len(errors), 6),
        "mean_count_log_loss": round(
            sum(float(record["exact_count_log_loss"]) for record in records) / len(records),
            6,
        ),
        "common_line_probability_metrics": _line_event_summary(line_events),
        "calibration_by_line_bucket": _line_event_bucket_summary(line_events, "line_bucket"),
        "calibration_by_confidence_bucket": _line_event_bucket_summary(line_events, "confidence_bucket"),
    }


def _line_event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_line: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_line.setdefault(f"{float(event['line']):.1f}", []).append(event)
    line_metrics = {
        line: _line_event_bucket_metrics(rows)
        for line, rows in sorted(by_line.items())
    }
    if not events:
        return {"overall": {"event_count": 0}, "by_common_prop_line": line_metrics}
    return {
        "overall": _line_event_bucket_metrics(events),
        "by_common_prop_line": line_metrics,
    }


def _line_event_bucket_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {
            "event_count": 0,
            "mean_brier_score": None,
            "mean_log_loss": None,
            "mean_probability": None,
            "observed_rate": None,
            "absolute_calibration_error": None,
        }
    brier_values = [
        (float(event["over_probability"]) - float(event["observed_over"])) ** 2
        for event in events
    ]
    log_losses = [
        -(
            float(event["observed_over"]) * log(max(MIN_PROBABILITY, float(event["over_probability"])))
            + (1.0 - float(event["observed_over"]))
            * log(max(MIN_PROBABILITY, 1.0 - float(event["over_probability"])))
        )
        for event in events
    ]
    mean_probability = sum(float(event["selected_side_probability"]) for event in events) / len(events)
    observed_rate = sum(float(event["selected_side_hit"]) for event in events) / len(events)
    return {
        "event_count": len(events),
        "mean_brier_score": round(sum(brier_values) / len(brier_values), 6),
        "mean_log_loss": round(sum(log_losses) / len(log_losses), 6),
        "mean_selected_side_probability": round(mean_probability, 6),
        "selected_side_hit_rate": round(observed_rate, 6),
        "absolute_calibration_error": round(abs(mean_probability - observed_rate), 6),
    }


def _line_event_bucket_summary(events: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault(str(event[key]), []).append(event)
    return [
        {
            key: bucket,
            **_line_event_bucket_metrics(rows),
        }
        for bucket, rows in sorted(grouped.items())
    ]


def _bias_summary(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record[key]), []).append(record)
    summary = []
    for bucket, rows in sorted(grouped.items()):
        errors = [float(row["projection_error"]) for row in rows]
        actuals = [float(row["actual_strikeouts"]) for row in rows]
        projections = [float(row["point_projection"]) for row in rows]
        summary.append(
            {
                key: bucket,
                "row_count": len(rows),
                "mean_projection": round(sum(projections) / len(projections), 6),
                "mean_actual": round(sum(actuals) / len(actuals), 6),
                "mean_bias": round(sum(errors) / len(errors), 6),
                "mae": round(sum(abs(error) for error in errors) / len(errors), 6),
                "rmse": round(sqrt(sum(error * error for error in errors) / len(errors)), 6),
                "overprojected_share": round(
                    sum(1 for error in errors if error > 0.0) / len(errors),
                    6,
                ),
            }
        )
    return summary


def _aggregate_bias_summaries(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "by_season": _bias_summary(records, "validation_season"),
        "by_pitcher_tier": _bias_summary(records, "pitcher_tier"),
        "by_handedness_matchup": _bias_summary(records, "handedness_matchup_bucket"),
        "by_workload_bucket": _bias_summary(records, "workload_bucket"),
        "by_rest_layoff_bucket": _bias_summary(records, "rest_layoff_bucket"),
        "by_rule_environment": _bias_summary(records, "rule_environment"),
    }


def _evaluate_policy(
    *,
    validation_season: int,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    split_id: str,
    policy_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected_candidate, diagnostics = _select_candidate_from_training_window(train_rows)
    records = [
        _prediction_record(
            row=row,
            split_id=split_id,
            validation_season=validation_season,
            selected_candidate=selected_candidate.name,
            mean=mean,
            probabilities=probabilities,
        )
        for row in validation_rows
        for mean, probabilities in [_distribution_for_candidate(row, selected_candidate)]
    ]
    metrics = _metrics_from_prediction_records(records)
    split = {
        "split_id": split_id,
        "policy": policy_name,
        "train_start_season": min(_season(row) for row in train_rows),
        "train_end_season": max(_season(row) for row in train_rows),
        "validation_season": validation_season,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "selected_candidate": selected_candidate.name,
        "selection": diagnostics,
        "metrics": metrics,
        "bias": _aggregate_bias_summaries(records),
    }
    return split, records


def _recency_policy_splits(
    *,
    rows: list[dict[str, Any]],
    validation_season: int,
    validation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    policies = []
    for window in RECENCY_WINDOWS:
        first_season = validation_season - window
        train_rows = [
            row
            for row in rows
            if first_season <= _season(row) < validation_season
        ]
        if not train_rows:
            continue
        split, _ = _evaluate_policy(
            validation_season=validation_season,
            train_rows=train_rows,
            validation_rows=validation_rows,
            split_id=f"validate_{validation_season}_recent_{window}_season_train",
            policy_name=f"recent_{window}_season_train",
        )
        policies.append(split)
    return policies


def _threshold_recommendations(metrics: dict[str, Any]) -> dict[str, Any]:
    confidence_rows = [
        row
        for row in metrics["calibration_by_confidence_bucket"]
        if row["event_count"] > 0 and row["absolute_calibration_error"] is not None
    ]
    line_rows = [
        row
        for row in metrics["calibration_by_line_bucket"]
        if row["event_count"] > 0 and row["absolute_calibration_error"] is not None
    ]
    if not confidence_rows:
        return {
            "status": "blocked_no_scoreable_confidence_buckets",
            "source": "walk-forward model-only calibration buckets",
            "recommendations": [],
        }
    median_error = median(float(row["absolute_calibration_error"]) for row in confidence_rows)
    median_count = median(int(row["event_count"]) for row in confidence_rows)
    qualified_confidence = [
        row
        for row in confidence_rows
        if int(row["event_count"]) >= median_count
        and float(row["absolute_calibration_error"]) <= median_error
    ]
    excluded_lines = [
        row
        for row in line_rows
        if float(row["absolute_calibration_error"]) > median_error
    ]
    status = "thresholds_observed_from_calibration"
    if not qualified_confidence:
        status = "blocked_no_stable_confidence_bucket"
    return {
        "status": status,
        "source": "observed walk-forward line and confidence calibration; no ROI, CLV, or guessed wagering threshold used",
        "observed_median_confidence_bucket_error": round(median_error, 6),
        "observed_median_confidence_bucket_events": median_count,
        "candidate_confidence_buckets_for_later_approval": qualified_confidence,
        "line_buckets_to_exclude_or_discount": excluded_lines,
    }


def _go_no_go(report: dict[str, Any]) -> dict[str, Any]:
    aggregate = report["headline_metrics"]
    threshold_status = report["proposed_later_wager_approval_thresholds"]["status"]
    source_artifacts = report["source_artifacts"]
    missing_feature_layers = [
        name
        for name in (
            "pitcher_skill_matches",
            "lineup_matchup_matches",
            "workload_leash_matches",
        )
        if int(source_artifacts.get(name) or 0) == 0
    ]
    season_bias_rows = report["bias_and_stability"]["by_season"]
    unstable_seasons = [
        row for row in season_bias_rows if abs(float(row["mean_bias"])) > float(aggregate["mae"])
    ]
    usable = (
        aggregate["row_count"] > 0
        and threshold_status == "thresholds_observed_from_calibration"
        and not missing_feature_layers
        and not unstable_seasons
    )
    if usable:
        return {
            "recommendation": "conditional_go_for_betting_layer_rebuild",
            "reason": (
                "Model-only walk-forward calibration produced observed confidence "
                "buckets that can seed later approval gates, with no season bias "
                "larger than aggregate MAE."
            ),
        }
    return {
        "recommendation": "no_go_betting_layer_still_blocked",
        "reason": (
            "Betting-layer rebuild stays blocked until model-only validation has "
            "feature-layer coverage, stable observed calibration buckets, and no "
            "season-level bias larger than aggregate MAE."
        ),
        "blocking_details": {
            "missing_feature_layers": missing_feature_layers,
            "threshold_status": threshold_status,
            "unstable_season_bias_count": len(unstable_seasons),
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    recommendation = report["go_no_go_recommendation"]
    lines = [
        "# Model-Only Walk-Forward Validation",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Window: `{report['date_window']['start_date']}` to `{report['date_window']['end_date']}`",
        f"- Headline split policy: `{report['validation_design']['headline_policy']}`",
        f"- Recommendation: `{recommendation['recommendation']}`",
        f"- Reason: {recommendation['reason']}",
        "",
        "## Headline Metrics",
        "",
        "| Rows | MAE | RMSE | Mean Bias | Count Log Loss | Common-Line Log Loss | Common-Line Brier |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    headline = report["headline_metrics"]
    line_overall = headline["common_line_probability_metrics"]["overall"]
    lines.append(
        "| {rows} | {mae:.6f} | {rmse:.6f} | {bias:.6f} | {count_log_loss:.6f} | {line_log_loss:.6f} | {brier:.6f} |".format(
            rows=headline["row_count"],
            mae=headline["mae"],
            rmse=headline["rmse"],
            bias=headline["mean_bias"],
            count_log_loss=headline["mean_count_log_loss"],
            line_log_loss=line_overall["mean_log_loss"],
            brier=line_overall["mean_brier_score"],
        )
    )
    lines.extend(
        [
            "",
            "## Walk-Forward Splits",
            "",
            "| Split | Train Seasons | Validation Season | Rows | Selected Candidate | MAE | RMSE | Line Log Loss |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for split in report["walk_forward_splits"]:
        metrics = split["metrics"]
        lines.append(
            "| `{split_id}` | {train_start}-{train_end} | {validation_season} | {rows} | `{candidate}` | {mae:.6f} | {rmse:.6f} | {line_log_loss:.6f} |".format(
                split_id=split["split_id"],
                train_start=split["train_start_season"],
                train_end=split["train_end_season"],
                validation_season=split["validation_season"],
                rows=metrics["row_count"],
                candidate=split["selected_candidate"],
                mae=metrics["mae"],
                rmse=metrics["rmse"],
                line_log_loss=metrics["common_line_probability_metrics"]["overall"]["mean_log_loss"],
            )
        )
    lines.extend(
        [
            "",
            "## Threshold Proposal",
            "",
            f"- Status: `{report['proposed_later_wager_approval_thresholds']['status']}`",
            f"- Source: {report['proposed_later_wager_approval_thresholds']['source']}",
            "",
            "No wagering, CLV, ROI, or stake-sizing metrics were used in this report.",
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
    first_validation_season: int,
    max_training_seasons: int,
) -> str:
    parts = [
        "uv",
        "run",
        "python",
        "-m",
        "mlb_props_stack",
        "validate-model-only-strikeouts",
        "--start-date",
        start_date.isoformat(),
        "--end-date",
        end_date.isoformat(),
        "--output-dir",
        str(output_dir),
        "--first-validation-season",
        str(first_validation_season),
        "--max-training-seasons",
        str(max_training_seasons),
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


def validate_model_only_strikeouts_walk_forward(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str = "data",
    dataset_run_dir: Path | str | None = None,
    pitcher_skill_run_dir: Path | str | None = None,
    lineup_matchup_run_dir: Path | str | None = None,
    workload_leash_run_dir: Path | str | None = None,
    first_validation_season: int = DEFAULT_FIRST_VALIDATION_SEASON,
    max_training_seasons: int = DEFAULT_MAX_TRAINING_SEASONS,
    now: Callable[[], datetime] = utc_now,
) -> ModelOnlyWalkForwardValidationResult:
    """Validate candidate strikeout models with rolling season walk-forward splits."""
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
    rows = _attach_prior_context(rows)
    seasons = _validation_seasons(
        rows,
        first_validation_season=first_validation_season,
        max_training_seasons=max_training_seasons,
    )
    if not seasons:
        raise ValueError(
            "No walk-forward validation seasons were available after applying "
            "the requested first validation season and training-window policy."
        )
    headline_splits: list[dict[str, Any]] = []
    recency_splits: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    for validation_season in seasons:
        train_rows, validation_rows, first_training_season = _split_rows_for_validation(
            rows,
            validation_season=validation_season,
            max_training_seasons=max_training_seasons,
        )
        split_id = (
            f"train_{first_training_season}_{validation_season - 1}"
            f"_validate_{validation_season}"
        )
        split, records = _evaluate_policy(
            validation_season=validation_season,
            train_rows=train_rows,
            validation_rows=validation_rows,
            split_id=split_id,
            policy_name="rolling_expanding_prior_seasons_with_cap",
        )
        headline_splits.append(split)
        prediction_records.extend(records)
        recency_splits.extend(
            _recency_policy_splits(
                rows=rows,
                validation_season=validation_season,
                validation_rows=validation_rows,
            )
        )

    generated_at = now().astimezone(UTC)
    run_root = (
        output_root
        / "normalized"
        / "model_only_walk_forward_validation"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
    )
    run_id = _unique_timestamp_run_id(generated_at, run_root)
    normalized_root = run_root / f"run={run_id}"
    report_path = normalized_root / "validation_report.json"
    report_markdown_path = normalized_root / "validation_report.md"
    predictions_path = normalized_root / "validation_predictions.jsonl"
    reproducibility_notes_path = normalized_root / "reproducibility_notes.md"
    headline_metrics = _metrics_from_prediction_records(prediction_records)
    threshold_recommendations = _threshold_recommendations(headline_metrics)
    report = {
        "report_version": VALIDATION_REPORT_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "date_window": {
            "start_date": start_date,
            "end_date": end_date,
        },
        "source_artifacts": source_summary,
        "validation_design": {
            "random_splits_used": False,
            "headline_policy": "rolling walk-forward by validation season",
            "first_validation_season": first_validation_season,
            "max_training_seasons": max_training_seasons,
            "common_prop_lines": list(COMMON_PROP_LINES),
            "selection_uses_validation_season": False,
        },
        "row_counts": {
            "joined_rows": len(rows),
            "prediction_rows": len(prediction_records),
            "walk_forward_splits": len(headline_splits),
        },
        "headline_metrics": headline_metrics,
        "walk_forward_splits": headline_splits,
        "season_weighting_and_recency_sensitivity": {
            "headline_weighting": "each validation row weighted equally; split and season rows are reported separately",
            "recency_train_window_splits": recency_splits,
        },
        "bias_and_stability": _aggregate_bias_summaries(prediction_records),
        "validation_context": _validation_context_summary(prediction_records),
        "proposed_later_wager_approval_thresholds": threshold_recommendations,
        "scope_guardrails": {
            "model_only_projection_validation": True,
            "betting_decisions_included": False,
            "clv_or_roi_headline_metrics_included": False,
            "wager_approval_thresholds_are_observed_from_calibration": True,
        },
        "reproducibility": {
            "rerun_command": _rerun_command(
                start_date=start_date,
                end_date=end_date,
                output_dir=output_dir,
                dataset_run_dir=dataset_run_dir,
                pitcher_skill_run_dir=pitcher_skill_run_dir,
                lineup_matchup_run_dir=lineup_matchup_run_dir,
                workload_leash_run_dir=workload_leash_run_dir,
                first_validation_season=first_validation_season,
                max_training_seasons=max_training_seasons,
            ),
            "notes_path": reproducibility_notes_path,
        },
    }
    report["go_no_go_recommendation"] = _go_no_go(report)
    _write_json(report_path, report)
    _write_text(report_markdown_path, _render_markdown(report))
    _write_jsonl(predictions_path, prediction_records)
    _write_text(
        reproducibility_notes_path,
        "\n".join(
            [
                "# Model-Only Walk-Forward Validation Reproducibility",
                "",
                f"- Run ID: `{run_id}`",
                f"- Generated at: `{generated_at.isoformat().replace('+00:00', 'Z')}`",
                f"- Rerun command: `{report['reproducibility']['rerun_command']}`",
                "",
                "The command validates projection accuracy only. It does not "
                "price sportsbook lines, compute CLV or ROI, size wagers, or "
                "unblock betting-layer work unless the report recommendation says so.",
                "",
            ]
        ),
    )
    return ModelOnlyWalkForwardValidationResult(
        start_date=start_date,
        end_date=end_date,
        run_id=run_id,
        split_count=len(headline_splits),
        prediction_count=len(prediction_records),
        recommendation=report["go_no_go_recommendation"]["recommendation"],
        report_path=report_path,
        report_markdown_path=report_markdown_path,
        predictions_path=predictions_path,
        reproducibility_notes_path=reproducibility_notes_path,
    )
