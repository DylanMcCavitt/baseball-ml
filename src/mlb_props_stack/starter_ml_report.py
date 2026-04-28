"""Vertical starter strikeout ML report.

This report is intentionally a thin orchestration layer over the existing
candidate-model code. It does not build new features or introduce a new model
family; it joins whatever timestamp-valid artifacts already exist, trains on
past dates, scores held-out dates, and writes a readable research-only report.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from math import isfinite, sqrt
from pathlib import Path
from shlex import quote
from typing import Any, Callable

from . import candidate_models as cm
from .ingest.mlb_stats_api import utc_now
from .modeling import _split_dates

STARTER_ML_REPORT_VERSION = "starter_strikeout_ml_report_v1"


@dataclass(frozen=True)
class StarterStrikeoutMLReportResult:
    """Filesystem output summary for one starter strikeout ML report."""

    start_date: date
    end_date: date
    run_id: str
    selected_candidate: str
    row_count: int
    held_out_rmse: float
    held_out_mae: float
    report_path: Path
    report_markdown_path: Path
    predictions_path: Path
    reproducibility_notes_path: Path


def _split_windows(
    rows: list[dict[str, Any]],
    date_splits: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    windows: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "validation", "test"):
        split_rows = cm._rows_for_dates(rows, date_splits[split_name])
        split_dates = [str(row["official_date"]) for row in split_rows]
        windows[split_name] = {
            "start_date": min(split_dates) if split_dates else None,
            "end_date": max(split_dates) if split_dates else None,
            "date_count": len(set(split_dates)),
            "row_count": len(split_rows),
        }
    return windows


def _spearman_rank_correlation(actuals: list[float], predictions: list[float]) -> float | None:
    if len(actuals) < 2 or len(actuals) != len(predictions):
        return None
    actual_ranks = _average_ranks(actuals)
    prediction_ranks = _average_ranks(predictions)
    actual_mean = sum(actual_ranks) / len(actual_ranks)
    prediction_mean = sum(prediction_ranks) / len(prediction_ranks)
    numerator = sum(
        (actual_rank - actual_mean) * (prediction_rank - prediction_mean)
        for actual_rank, prediction_rank in zip(actual_ranks, prediction_ranks)
    )
    actual_denominator = sqrt(
        sum((actual_rank - actual_mean) ** 2 for actual_rank in actual_ranks)
    )
    prediction_denominator = sqrt(
        sum((prediction_rank - prediction_mean) ** 2 for prediction_rank in prediction_ranks)
    )
    denominator = actual_denominator * prediction_denominator
    if denominator <= 0.0:
        return None
    return round(numerator / denominator, 6)


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    index = 0
    while index < len(indexed):
        next_index = index + 1
        while next_index < len(indexed) and indexed[next_index][1] == indexed[index][1]:
            next_index += 1
        average_rank = (index + 1 + next_index) / 2.0
        for original_index, _ in indexed[index:next_index]:
            ranks[original_index] = average_rank
        index = next_index
    return ranks


def _prediction_rows(
    rows: list[dict[str, Any]],
    *,
    split_by_date: dict[str, str],
    candidate: Any,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for row in rows:
        mean, probabilities = cm._distribution_for_candidate(row, candidate)
        actual = int(cm._target(row))
        error = mean - actual
        starter = cm._source_row(row, "starter")
        lineup = cm._source_row(row, "lineup")
        pitcher = cm._source_row(row, "pitcher")
        workload = cm._source_row(row, "workload")
        features_as_of = (
            starter.get("features_as_of")
            or pitcher.get("features_as_of")
            or lineup.get("features_as_of")
            or workload.get("features_as_of")
        )
        lineup_snapshot_id = (
            starter.get("lineup_snapshot_id") or lineup.get("lineup_snapshot_id")
        )
        common_lines = []
        for line in cm.COMMON_PROP_LINES:
            line_probabilities = cm.strikeout_line_probabilities(probabilities, line)
            observed_over = actual > line
            common_lines.append(
                {
                    "line": line,
                    "over_probability": round(line_probabilities["over_probability"], 6),
                    "under_probability": round(line_probabilities["under_probability"], 6),
                    "observed_over": observed_over,
                }
            )
        exact_probability = probabilities[actual] if actual < len(probabilities) else 0.0
        predictions.append(
            {
                "training_row_id": row["training_row_id"],
                "feature_row_id": row["training_row_id"],
                "official_date": row["official_date"],
                "split": split_by_date[str(row["official_date"])],
                "game_pk": starter.get("game_pk"),
                "pitcher_id": starter.get("pitcher_id"),
                "pitcher_name": starter.get("pitcher_name"),
                "lineup_snapshot_id": lineup_snapshot_id,
                "features_as_of": features_as_of,
                "projection_generated_at": features_as_of,
                "model_input_refs": {
                    "pitcher_feature_row_id": (
                        starter.get("pitcher_feature_row_id")
                        or pitcher.get("feature_row_id")
                    ),
                    "lineup_feature_row_id": (
                        starter.get("lineup_feature_row_id")
                        or lineup.get("feature_row_id")
                    ),
                    "game_context_feature_row_id": (
                        starter.get("game_context_feature_row_id")
                        or workload.get("feature_row_id")
                    ),
                },
                "team_abbreviation": starter.get("team_abbreviation"),
                "opponent_team_abbreviation": starter.get("opponent_team_abbreviation"),
                "home_away": starter.get("home_away"),
                "pitcher_hand": starter.get("pitcher_hand"),
                "season": starter.get("season") or date.fromisoformat(str(row["official_date"])).year,
                "rule_environment": starter.get("pitch_clock_era") or "unknown_rule_environment",
                "actual_strikeouts": actual,
                "point_projection": round(mean, 6),
                "projection_error": round(error, 6),
                "absolute_error": round(abs(error), 6),
                "exact_count_probability": round(exact_probability, 8),
                "common_line_probabilities": common_lines,
            }
        )
    return predictions


def _held_out_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    held_out = [
        row for row in predictions if row["split"] in {"validation", "test"}
    ]
    if not held_out:
        return {
            "row_count": 0,
            "rmse": None,
            "mae": None,
            "mean_bias": None,
            "spearman_rank_correlation": None,
        }
    errors = [float(row["projection_error"]) for row in held_out]
    actuals = [float(row["actual_strikeouts"]) for row in held_out]
    projections = [float(row["point_projection"]) for row in held_out]
    return {
        "row_count": len(held_out),
        "rmse": round(sqrt(sum(error * error for error in errors) / len(errors)), 6),
        "mae": round(sum(abs(error) for error in errors) / len(errors), 6),
        "mean_bias": round(sum(errors) / len(errors), 6),
        "spearman_rank_correlation": _spearman_rank_correlation(actuals, projections),
    }


def _bias_summary(predictions: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    held_out = [
        row for row in predictions if row["split"] in {"validation", "test"}
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in held_out:
        grouped.setdefault(str(row.get(key) or "unknown"), []).append(row)
    rows: list[dict[str, Any]] = []
    for bucket, bucket_rows in sorted(grouped.items()):
        errors = [float(row["projection_error"]) for row in bucket_rows]
        actuals = [float(row["actual_strikeouts"]) for row in bucket_rows]
        projections = [float(row["point_projection"]) for row in bucket_rows]
        rows.append(
            {
                key: bucket,
                "row_count": len(bucket_rows),
                "mean_projection": round(sum(projections) / len(projections), 6),
                "mean_actual": round(sum(actuals) / len(actuals), 6),
                "mean_bias": round(sum(errors) / len(errors), 6),
                "mae": round(sum(abs(error) for error in errors) / len(errors), 6),
                "rmse": round(sqrt(sum(error * error for error in errors) / len(errors)), 6),
            }
        )
    return rows


def _prediction_examples(predictions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    held_out = [
        row for row in predictions if row["split"] in {"validation", "test"}
    ]

    def _example(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "official_date": row["official_date"],
            "split": row["split"],
            "pitcher_name": row["pitcher_name"],
            "pitcher_id": row["pitcher_id"],
            "actual_strikeouts": row["actual_strikeouts"],
            "point_projection": row["point_projection"],
            "projection_error": row["projection_error"],
            "absolute_error": row["absolute_error"],
        }

    return {
        "best_predictions": [
            _example(row)
            for row in sorted(held_out, key=lambda item: (item["absolute_error"], item["official_date"]))[:10]
        ],
        "worst_predictions": [
            _example(row)
            for row in sorted(
                held_out,
                key=lambda item: (-float(item["absolute_error"]), item["official_date"]),
            )[:10]
        ],
    }


def _selected_feature_columns(candidate: Any) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}

    def _collect(model_candidate: Any) -> None:
        model = getattr(model_candidate, "model", None)
        if isinstance(model, tuple):
            for member_name in model:
                _collect(cm._CANDIDATE_REGISTRY[member_name])
            return
        vectorizer = getattr(model, "vectorizer", None)
        if vectorizer is None:
            return
        for spec in vectorizer.feature_specs:
            selected[spec.name] = {
                "feature": spec.name,
                "family": spec.group,
                "source": spec.source,
                "field": spec.field,
            }

    _collect(candidate)
    return [selected[name] for name in sorted(selected)]


def _excluded_feature_families(
    *,
    source_summary: dict[str, Any],
    selected_feature_columns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_families = {str(row["family"]) for row in selected_feature_columns}
    availability = {
        "pitcher_skill": int(source_summary.get("pitcher_skill_matches") or 0),
        "matchup": int(source_summary.get("lineup_matchup_matches") or 0),
        "workload": int(source_summary.get("workload_leash_matches") or 0),
    }
    rows: list[dict[str, Any]] = []
    for family, match_count in availability.items():
        if match_count == 0:
            rows.append(
                {
                    "family": family,
                    "status": "missing_artifact_or_no_joined_rows",
                    "reason": "No matching existing feature artifact rows were available; no feature builder was run.",
                }
            )
        elif family not in selected_families:
            rows.append(
                {
                    "family": family,
                    "status": "not_selected",
                    "reason": "Existing rows were available but the trained selected model did not use a non-constant column from this family.",
                }
            )
    if "context" not in selected_families:
        rows.append(
            {
                "family": "context",
                "status": "not_selected",
                "reason": "No non-constant starter context column was selected for the final model.",
            }
        )
    rows.extend(
        [
            {
                "family": "same_game_outcomes",
                "status": "excluded_by_guardrail",
                "reason": "Starter strikeouts are used only as the target label, not as a feature.",
            },
            {
                "family": "sportsbook_market_or_betting",
                "status": "excluded_by_scope",
                "reason": "This is a research-only ML report; no odds, CLV, ROI, edge, approval, or stake-sizing inputs are used.",
            },
        ]
    )
    return rows


def _timestamp_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    feature_timestamp_rows = 0
    feature_timestamp_after_official_date_rows = 0
    for row in rows:
        starter = cm._source_row(row, "starter")
        status = str(starter.get("timestamp_policy_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        feature_as_of = starter.get("features_as_of")
        if not feature_as_of:
            continue
        feature_timestamp_rows += 1
        try:
            feature_dt = datetime.fromisoformat(str(feature_as_of).replace("Z", "+00:00"))
        except ValueError:
            continue
        official_date = date.fromisoformat(str(row["official_date"]))
        if feature_dt.date() > official_date:
            feature_timestamp_after_official_date_rows += 1
    return {
        "status": (
            "ok"
            if feature_timestamp_after_official_date_rows == 0
            else "review_feature_timestamps"
        ),
        "date_ordered_train_validation_test_split": True,
        "random_splits_used": False,
        "same_game_outcomes_used_as_features": False,
        "target_label": "starter_strikeouts",
        "target_label_usage": "same-game outcome is target-only",
        "timestamp_policy_status_counts": status_counts,
        "feature_timestamp_rows": feature_timestamp_rows,
        "feature_timestamp_after_official_date_rows": feature_timestamp_after_official_date_rows,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    held_out = report["held_out_metrics"]
    probability = report["probability_quality"]["common_line_over_under"]
    lines = [
        "# Starter Strikeout ML Report",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Window: `{report['date_window']['start_date']}` to `{report['date_window']['end_date']}`",
        f"- Selected candidate: `{report['selected_model']['selected_candidate']}`",
        f"- Research-only: `{str(report['scope_guardrails']['research_only_not_betting_ready']).lower()}`",
        "",
        "## Rows And Splits",
        "",
        "| Split | Dates | Rows | Window |",
        "| --- | ---: | ---: | --- |",
    ]
    for split_name, split in report["split_windows"].items():
        lines.append(
            "| {split_name} | {date_count} | {row_count} | {start_date} to {end_date} |".format(
                split_name=split_name,
                date_count=split["date_count"],
                row_count=split["row_count"],
                start_date=split["start_date"],
                end_date=split["end_date"],
            )
        )
    lines.extend(
        [
            "",
            "## Held-Out Metrics",
            "",
            "| Rows | RMSE | MAE | Mean Bias | Spearman |",
            "| ---: | ---: | ---: | ---: | ---: |",
            "| {rows} | {rmse:.6f} | {mae:.6f} | {bias:.6f} | {spearman} |".format(
                rows=held_out["row_count"],
                rmse=held_out["rmse"],
                mae=held_out["mae"],
                bias=held_out["mean_bias"],
                spearman=(
                    "n/a"
                    if held_out["spearman_rank_correlation"] is None
                    else f"{held_out['spearman_rank_correlation']:.6f}"
                ),
            ),
            "",
            "## Probability Quality",
            "",
            "| Events | Common-Line Log Loss | Common-Line Brier | Count NLL | Ranked Probability Score |",
            "| ---: | ---: | ---: | ---: | ---: |",
            "| {events} | {line_log_loss:.6f} | {brier:.6f} | {count_nll:.6f} | {rps:.6f} |".format(
                events=probability["overall"]["event_count"],
                line_log_loss=probability["overall"]["mean_log_loss"],
                brier=probability["overall"]["mean_brier_score"],
                count_nll=report["probability_quality"]["count_distribution"]["mean_negative_log_likelihood"],
                rps=report["probability_quality"]["count_distribution"]["mean_ranked_probability_score"],
            ),
            "",
            "## Selected Feature Columns",
            "",
        ]
    )
    for column in report["feature_contract"]["selected_feature_columns"][:40]:
        lines.append(
            f"- `{column['feature']}` ({column['family']} from {column['source']}.{column['field']})"
        )
    if len(report["feature_contract"]["selected_feature_columns"]) > 40:
        lines.append("- ...")
    lines.extend(
        [
            "",
            "## Excluded Or Missing Feature Families",
            "",
        ]
    )
    for item in report["feature_contract"]["excluded_feature_families"]:
        lines.append(f"- `{item['family']}`: {item['status']} - {item['reason']}")
    lines.extend(
        [
            "",
            "## Leakage And Timestamp Status",
            "",
            f"- Status: `{report['leakage_and_timestamp_status']['status']}`",
            "- Date-ordered split: `true`",
            "- Same-game outcomes as features: `false`",
            "",
            "## Best Held-Out Examples",
            "",
        ]
    )
    for row in report["prediction_examples"]["best_predictions"][:5]:
        lines.append(
            "- {date} {pitcher}: actual={actual} predicted={predicted:.3f} abs_error={error:.3f}".format(
                date=row["official_date"],
                pitcher=row["pitcher_name"] or row["pitcher_id"],
                actual=row["actual_strikeouts"],
                predicted=row["point_projection"],
                error=row["absolute_error"],
            )
        )
    lines.extend(["", "## Worst Held-Out Examples", ""])
    for row in report["prediction_examples"]["worst_predictions"][:5]:
        lines.append(
            "- {date} {pitcher}: actual={actual} predicted={predicted:.3f} abs_error={error:.3f}".format(
                date=row["official_date"],
                pitcher=row["pitcher_name"] or row["pitcher_id"],
                actual=row["actual_strikeouts"],
                predicted=row["point_projection"],
                error=row["absolute_error"],
            )
        )
    lines.extend(
        [
            "",
            "No betting decisions, CLV, ROI, approval gates, or stake sizing are included.",
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
        "build-starter-strikeout-ml-report",
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


def build_starter_strikeout_ml_report(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str = "data",
    dataset_run_dir: Path | str | None = None,
    pitcher_skill_run_dir: Path | str | None = None,
    lineup_matchup_run_dir: Path | str | None = None,
    workload_leash_run_dir: Path | str | None = None,
    now: Callable[[], datetime] = utc_now,
) -> StarterStrikeoutMLReportResult:
    """Build a human-readable starter strikeout ML report from existing artifacts."""
    output_root = Path(output_dir)
    rows, source_summary = cm._build_joined_rows(
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
    train_rows = cm._rows_for_dates(rows, date_splits["train"])
    validation_rows = cm._rows_for_dates(rows, date_splits["validation"])
    test_rows = cm._rows_for_dates(rows, date_splits["test"])
    held_out_rows = [*validation_rows, *test_rows]
    if not train_rows or not validation_rows or not test_rows:
        raise ValueError("Date splits must leave train, validation, and test rows.")

    candidates = cm._train_candidates(train_rows)
    candidates = cm._add_ensemble_candidate(candidates, validation_rows, train_rows)
    report_candidates: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if candidate.status != "trained":
            report_candidates[candidate.name] = cm._candidate_artifact(candidate)
            continue
        split_metrics: dict[str, Any] = {}
        for split_name, split_rows in (
            ("train", train_rows),
            ("validation", validation_rows),
            ("test", test_rows),
            ("held_out", held_out_rows),
        ):
            split_metrics[split_name], _ = cm._candidate_metrics(split_rows, candidate)
        report_candidates[candidate.name] = {
            **cm._candidate_artifact(candidate),
            "splits": split_metrics,
            "feature_group_contributions": cm._feature_group_contributions(candidate),
        }
    selected_candidate_name = cm._select_candidate(report_candidates)
    selected_candidate = cm._CANDIDATE_REGISTRY[selected_candidate_name]
    predictions = _prediction_rows(
        rows,
        split_by_date=split_by_date,
        candidate=selected_candidate,
    )
    selected_feature_columns = _selected_feature_columns(selected_candidate)
    held_out_metrics = _held_out_metrics(predictions)
    generated_at = now().astimezone(UTC)
    run_root = (
        output_root
        / "normalized"
        / "starter_strikeout_ml_report"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
    )
    run_id = cm._unique_timestamp_run_id(generated_at, run_root)
    normalized_root = run_root / f"run={run_id}"
    report_path = normalized_root / "starter_strikeout_ml_report.json"
    report_markdown_path = normalized_root / "starter_strikeout_ml_report.md"
    predictions_path = normalized_root / "starter_strikeout_ml_predictions.jsonl"
    reproducibility_notes_path = normalized_root / "reproducibility_notes.md"
    selected_metrics = report_candidates[selected_candidate_name]
    selected_held_out = selected_metrics["splits"]["held_out"]
    common_line_metrics = selected_held_out["probability_metrics"]
    report = {
        "report_version": STARTER_ML_REPORT_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "date_window": {
            "start_date": start_date,
            "end_date": end_date,
        },
        "row_counts": {
            "joined_rows": len(rows),
            "prediction_rows": len(predictions),
            "held_out_rows": held_out_metrics["row_count"],
        },
        "split_windows": _split_windows(rows, date_splits),
        "date_splits": date_splits,
        "source_artifacts": source_summary,
        "selected_model": {
            "selected_candidate": selected_candidate_name,
            "selection_metric": "validation common-line mean log loss",
            "candidate_metrics": selected_metrics,
        },
        "feature_contract": {
            "selected_feature_columns": selected_feature_columns,
            "excluded_feature_families": _excluded_feature_families(
                source_summary=source_summary,
                selected_feature_columns=selected_feature_columns,
            ),
        },
        "leakage_and_timestamp_status": _timestamp_status(rows),
        "held_out_metrics": held_out_metrics,
        "bias_slices": {
            "by_split": _bias_summary(predictions, "split"),
            "by_season": _bias_summary(predictions, "season"),
            "by_home_away": _bias_summary(predictions, "home_away"),
            "by_pitcher_hand": _bias_summary(predictions, "pitcher_hand"),
            "by_rule_environment": _bias_summary(predictions, "rule_environment"),
        },
        "probability_quality": {
            "count_distribution": {
                "available": True,
                "mean_negative_log_likelihood": selected_held_out["mean_negative_log_likelihood"],
                "mean_ranked_probability_score": selected_held_out["mean_ranked_probability_score"],
                "exact_mode_accuracy": selected_held_out["exact_mode_accuracy"],
                "central_80_interval_coverage": selected_held_out["central_80_interval_coverage"],
            },
            "common_line_over_under": common_line_metrics,
        },
        "prediction_examples": _prediction_examples(predictions),
        "scope_guardrails": {
            "research_only_not_betting_ready": True,
            "new_model_families_added": False,
            "new_feature_families_added": False,
            "feature_builders_run": False,
            "betting_decisions_included": False,
            "clv_or_roi_metrics_included": False,
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
            ),
            "notes_path": reproducibility_notes_path,
        },
    }
    cm._write_json(report_path, report)
    cm._write_text(report_markdown_path, _render_markdown(report))
    cm._write_jsonl(predictions_path, predictions)
    cm._write_text(
        reproducibility_notes_path,
        "\n".join(
            [
                "# Starter Strikeout ML Report Reproducibility",
                "",
                f"- Run ID: `{run_id}`",
                f"- Generated at: `{generated_at.isoformat().replace('+00:00', 'Z')}`",
                f"- Rerun command: `{report['reproducibility']['rerun_command']}`",
                "",
                "This command reads existing starter-game and feature artifacts, "
                "trains on date-ordered past rows, and scores held-out rows. It "
                "does not build feature artifacts, price sportsbook lines, compute "
                "CLV or ROI, or mark outputs as betting-ready.",
                "",
            ]
        ),
    )
    rmse = held_out_metrics["rmse"]
    mae = held_out_metrics["mae"]
    if not isinstance(rmse, (int, float)) or not isfinite(float(rmse)):
        raise ValueError("Held-out RMSE was not available.")
    if not isinstance(mae, (int, float)) or not isfinite(float(mae)):
        raise ValueError("Held-out MAE was not available.")
    return StarterStrikeoutMLReportResult(
        start_date=start_date,
        end_date=end_date,
        run_id=run_id,
        selected_candidate=selected_candidate_name,
        row_count=len(rows),
        held_out_rmse=float(rmse),
        held_out_mae=float(mae),
        report_path=report_path,
        report_markdown_path=report_markdown_path,
        predictions_path=predictions_path,
        reproducibility_notes_path=reproducibility_notes_path,
    )
