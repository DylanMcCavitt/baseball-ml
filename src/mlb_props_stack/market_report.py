"""Sportsbook market report over starter strikeout ML predictions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import json
from pathlib import Path
from shlex import quote
from typing import Any, Callable

from .backtest import WalkForwardBacktestResult, build_walk_forward_backtest
from .ingest.mlb_stats_api import utc_now
from .tracking import TrackingConfig

MARKET_REPORT_VERSION = "starter_strikeout_market_report_v1"


@dataclass(frozen=True)
class StarterStrikeoutMarketReportResult:
    """Filesystem output summary for one sportsbook market report."""

    start_date: date
    end_date: date
    run_id: str
    report_path: Path
    report_markdown_path: Path
    adapted_model_run_dir: Path
    backtest_result: WalkForwardBacktestResult
    scoreable_row_count: int
    skipped_row_count: int


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.isoformat()
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
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_ready(row), sort_keys=True))
            handle.write("\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_backtest_rows(path: Path) -> list[dict[str, Any]]:
    return _load_jsonl(path)


def _path_run_id(run_dir: Path) -> str:
    return run_dir.name.split("=", 1)[-1]


def _unique_timestamp_run_id(base_time: datetime, run_root: Path) -> str:
    candidate_time = base_time.astimezone(UTC)
    while True:
        run_id = candidate_time.strftime("%Y%m%dT%H%M%SZ")
        if not run_root.joinpath(f"run={run_id}").exists():
            return run_id
        candidate_time += timedelta(seconds=1)


def _resolve_ml_report_run_dir(
    output_root: Path,
    *,
    start_date: date,
    end_date: date,
    ml_report_run_dir: Path | str | None,
    predictions_path: Path | str | None,
) -> Path:
    if ml_report_run_dir is not None:
        run_dir = Path(ml_report_run_dir)
        if not run_dir.joinpath("starter_strikeout_ml_predictions.jsonl").exists():
            raise FileNotFoundError(
                f"{run_dir} is missing starter_strikeout_ml_predictions.jsonl."
            )
        return run_dir
    if predictions_path is not None:
        path = Path(predictions_path)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")
        return path.parent
    root = (
        output_root
        / "normalized"
        / "starter_strikeout_ml_report"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
    )
    candidates = sorted(
        path
        for path in root.glob("run=*")
        if path.is_dir() and path.joinpath("starter_strikeout_ml_predictions.jsonl").exists()
    )
    if not candidates:
        raise FileNotFoundError(
            "No starter strikeout ML report run with predictions was found for "
            f"{start_date.isoformat()} -> {end_date.isoformat()}."
        )
    return candidates[-1]


def _prediction_line_probability(
    row: dict[str, Any],
    *,
    line: float,
) -> dict[str, Any] | None:
    for item in row.get("common_line_probabilities") or []:
        if round(float(item["line"]), 6) == round(line, 6):
            return item
    return None


def _adapt_predictions_to_backtest_model_run(
    *,
    predictions_path: Path,
    adapted_model_run_dir: Path,
    start_date: date,
    end_date: date,
    model_version: str,
) -> dict[str, Any]:
    source_rows = [
        row
        for row in _load_jsonl(predictions_path)
        if start_date <= date.fromisoformat(str(row["official_date"])) <= end_date
    ]
    if not source_rows:
        raise ValueError("No AGE-311 prediction rows cover the requested market window.")

    split_dates: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    training_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    train_dates = sorted(
        str(row["official_date"]) for row in source_rows if row.get("split") == "train"
    )
    model_train_from_date = train_dates[0] if train_dates else None
    model_train_through_date = train_dates[-1] if train_dates else None

    for row in source_rows:
        split = str(row.get("split") or "unknown")
        if split in split_dates:
            split_dates[split].add(str(row["official_date"]))
        training_row_id = str(row["training_row_id"])
        game_pk = row.get("game_pk")
        pitcher_id = row.get("pitcher_id")
        training_rows.append(
            {
                "training_row_id": training_row_id,
                "official_date": row["official_date"],
                "game_pk": game_pk,
                "pitcher_id": pitcher_id,
                "lineup_snapshot_id": row.get("lineup_snapshot_id"),
                "features_as_of": row.get("features_as_of"),
                "projection_generated_at": row.get("projection_generated_at"),
                "model_input_refs": row.get("model_input_refs") or {},
            }
        )
        outcome_rows.append(
            {
                "outcome_id": (
                    "starter-outcome:{official_date}:{game_pk}:{pitcher_id}".format(
                        official_date=row["official_date"],
                        game_pk=game_pk,
                        pitcher_id=pitcher_id,
                    )
                ),
                "official_date": row["official_date"],
                "game_pk": game_pk,
                "pitcher_id": pitcher_id,
                "starter_strikeouts": row.get("actual_strikeouts"),
            }
        )
        for probability in row.get("common_line_probabilities") or []:
            probability_rows.append(
                {
                    "training_row_id": training_row_id,
                    "official_date": row["official_date"],
                    "split": split,
                    "game_pk": game_pk,
                    "pitcher_id": pitcher_id,
                    "line": float(probability["line"]),
                    "model_mean": row.get("point_projection"),
                    "count_distribution": {
                        "name": "starter_strikeout_ml_report_common_line_probabilities",
                        "source_predictions_path": str(predictions_path),
                    },
                    "raw_over_probability": probability["over_probability"],
                    "raw_under_probability": probability["under_probability"],
                    "calibrated_over_probability": probability["over_probability"],
                    "calibrated_under_probability": probability["under_probability"],
                    "projection_generated_at": row.get("projection_generated_at"),
                    "model_train_from_date": model_train_from_date,
                    "model_train_through_date": model_train_through_date,
                    "calibration_method": "identity_from_starter_strikeout_ml_report",
                    "calibration_training_splits": [],
                    "calibration_sample_count": 0,
                    "calibration_fit_from_date": None,
                    "calibration_fit_through_date": None,
                    "calibration_is_identity": True,
                }
            )

    _write_json(
        adapted_model_run_dir / "baseline_model.json",
        {
            "model_version": model_version,
            "source": "starter_strikeout_ml_report",
            "source_predictions_path": predictions_path,
        },
    )
    _write_json(
        adapted_model_run_dir / "date_splits.json",
        {key: sorted(value) for key, value in split_dates.items()},
    )
    _write_jsonl(adapted_model_run_dir / "training_dataset.jsonl", training_rows)
    _write_jsonl(adapted_model_run_dir / "starter_outcomes.jsonl", outcome_rows)
    _write_jsonl(
        adapted_model_run_dir / "raw_vs_calibrated_probabilities.jsonl",
        probability_rows,
    )
    return {
        "source_prediction_rows": len(source_rows),
        "adapted_training_rows": len(training_rows),
        "adapted_probability_rows": len(probability_rows),
        "date_splits": {key: sorted(value) for key, value in split_dates.items()},
    }


def _count_by(
    rows: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = key_fn(row)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _book_line_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["sportsbook"]), float(row["line"])), []).append(row)
    coverage: list[dict[str, Any]] = []
    for (sportsbook, line), group_rows in sorted(grouped.items()):
        coverage.append(
            {
                "sportsbook": sportsbook,
                "line": line,
                "snapshot_groups": len(group_rows),
                "scoreable_rows": sum(
                    1
                    for row in group_rows
                    if row["evaluation_status"] in {"actionable", "below_threshold"}
                ),
                "actionable_rows": sum(
                    1 for row in group_rows if row["evaluation_status"] == "actionable"
                ),
                "below_threshold_rows": sum(
                    1
                    for row in group_rows
                    if row["evaluation_status"] == "below_threshold"
                ),
                "skipped_rows": sum(
                    1
                    for row in group_rows
                    if row["evaluation_status"] not in {"actionable", "below_threshold"}
                ),
            }
        )
    return coverage


def _calibration_by_line_bucket(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scoreable = [
        row for row in rows if row["evaluation_status"] in {"actionable", "below_threshold"}
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in scoreable:
        grouped.setdefault(f"{float(row['line']):.1f}", []).append(row)
    calibration: list[dict[str, Any]] = []
    for bucket, bucket_rows in sorted(grouped.items(), key=lambda item: float(item[0])):
        predicted = [float(row["selected_model_probability"]) for row in bucket_rows]
        actual = [1.0 if row["settlement_status"] == "win" else 0.0 for row in bucket_rows]
        avg_predicted = sum(predicted) / len(predicted)
        observed_rate = sum(actual) / len(actual)
        calibration.append(
            {
                "line_bucket": bucket,
                "row_count": len(bucket_rows),
                "average_selected_model_probability": round(avg_predicted, 6),
                "observed_win_rate": round(observed_rate, 6),
                "absolute_calibration_error": round(abs(avg_predicted - observed_rate), 6),
            }
        )
    return calibration


def _examples(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    scoreable = [
        row for row in rows if row["evaluation_status"] in {"actionable", "below_threshold"}
    ]
    skipped = [
        row for row in rows if row["evaluation_status"] not in {"actionable", "below_threshold"}
    ]

    def _scoreable_example(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "official_date": row["official_date"],
            "sportsbook": row["sportsbook"],
            "pitcher": row["player_name"],
            "line": row["line"],
            "side": row.get("selected_side"),
            "model_probability": row.get("selected_model_probability"),
            "market_probability": row.get("selected_market_probability"),
            "edge_pct": row.get("edge_pct"),
            "evaluation_status": row["evaluation_status"],
            "settlement_status": row.get("settlement_status"),
            "clv_probability_delta": row.get("clv_probability_delta"),
        }

    def _skipped_example(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "official_date": row["official_date"],
            "sportsbook": row["sportsbook"],
            "pitcher": row["player_name"],
            "line": row["line"],
            "evaluation_status": row["evaluation_status"],
            "reason": row.get("reason"),
            "line_snapshot_id": row.get("line_snapshot_id"),
            "latest_observed_line_snapshot_id": row.get(
                "latest_observed_line_snapshot_id"
            ),
        }

    return {
        "largest_edges": [
            _scoreable_example(row)
            for row in sorted(
                scoreable,
                key=lambda item: -float(item.get("edge_pct") or 0.0),
            )[:10]
        ],
        "skipped_rows": [_skipped_example(row) for row in skipped[:10]],
    }


def _render_markdown(report: dict[str, Any]) -> str:
    rows = report["row_counts"]
    clv = report["clv"]
    roi = report["roi"]
    lines = [
        "# Starter Strikeout Sportsbook Market Report",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Window: `{report['date_window']['start_date']}` to `{report['date_window']['end_date']}`",
        f"- Scoreable rows: `{rows['scoreable_rows']}`",
        f"- Skipped rows: `{rows['skipped_rows']}`",
        f"- Actionable rows: `{rows['actionable_rows']}`",
        f"- Below-threshold rows: `{rows['below_threshold_rows']}`",
        "",
        "## CLV And ROI",
        "",
        "| CLV Sample | Median CLV Prob Delta | Placed Bets | ROI | Profit Units |",
        "| ---: | ---: | ---: | ---: | ---: |",
        "| {clv_sample} | {clv_median} | {placed} | {roi_value} | {profit} |".format(
            clv_sample=clv["sample_count"],
            clv_median=clv["median_probability_delta"],
            placed=roi["placed_bets"],
            roi_value=roi["roi"],
            profit=roi["total_profit_units"],
        ),
        "",
        "## Join And Skip Reasons",
        "",
    ]
    if report["join_failure_reasons"]:
        for reason, count in report["join_failure_reasons"].items():
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- No skipped join failures in this run.")
    lines.extend(["", "## Book And Line Coverage", ""])
    if report["book_line_coverage"]:
        lines.extend(
            [
                "| Sportsbook | Line | Groups | Scoreable | Skipped |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in report["book_line_coverage"][:30]:
            lines.append(
                "| {sportsbook} | {line:.1f} | {groups} | {scoreable} | {skipped} |".format(
                    sportsbook=row["sportsbook"],
                    line=row["line"],
                    groups=row["snapshot_groups"],
                    scoreable=row["scoreable_rows"],
                    skipped=row["skipped_rows"],
                )
            )
    else:
        lines.append("No sportsbook line snapshot groups were available.")
    lines.extend(["", "## Calibration By Line", ""])
    if report["calibration_by_line_bucket"]:
        lines.extend(
            [
                "| Line | Rows | Avg Model Prob | Observed Win Rate | Abs Error |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in report["calibration_by_line_bucket"]:
            lines.append(
                "| {line} | {count} | {pred} | {actual} | {error} |".format(
                    line=row["line_bucket"],
                    count=row["row_count"],
                    pred=row["average_selected_model_probability"],
                    actual=row["observed_win_rate"],
                    error=row["absolute_calibration_error"],
                )
            )
    else:
        lines.append("No scoreable rows were available for line-bucket calibration.")
    lines.extend(["", "## Examples", ""])
    for row in report["examples"]["largest_edges"][:5]:
        lines.append(
            "- {date} {book} {pitcher} {side} {line}: edge={edge} result={result} clv={clv_delta}".format(
                date=row["official_date"],
                book=row["sportsbook"],
                pitcher=row["pitcher"],
                side=row["side"],
                line=row["line"],
                edge=row["edge_pct"],
                result=row["settlement_status"],
                clv_delta=row["clv_probability_delta"],
            )
        )
    if report["examples"]["skipped_rows"]:
        lines.extend(["", "## Skipped Examples", ""])
        for row in report["examples"]["skipped_rows"][:5]:
            lines.append(
                "- {date} {book} {pitcher} {line}: {status} - {reason}".format(
                    date=row["official_date"],
                    book=row["sportsbook"],
                    pitcher=row["pitcher"],
                    line=row["line"],
                    status=row["evaluation_status"],
                    reason=row["reason"],
                )
            )
    lines.extend(
        [
            "",
            "Rows remain research-only. The report preserves skipped rows and does not fabricate uncertain sportsbook joins.",
            "",
        ]
    )
    return "\n".join(lines)


def _rerun_command(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str,
    ml_report_run_dir: Path,
    odds_input_dir: Path | str | None,
    cutoff_minutes_before_first_pitch: int,
) -> str:
    parts = [
        "uv",
        "run",
        "python",
        "-m",
        "mlb_props_stack",
        "build-starter-strikeout-market-report",
        "--start-date",
        start_date.isoformat(),
        "--end-date",
        end_date.isoformat(),
        "--output-dir",
        str(output_dir),
        "--ml-report-run-dir",
        str(ml_report_run_dir),
        "--cutoff-minutes-before-first-pitch",
        str(cutoff_minutes_before_first_pitch),
    ]
    if odds_input_dir is not None:
        parts.extend(["--odds-input-dir", str(odds_input_dir)])
    return " ".join(quote(part) for part in parts)


def build_starter_strikeout_market_report(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str = "data",
    ml_report_run_dir: Path | str | None = None,
    predictions_path: Path | str | None = None,
    odds_input_dir: Path | str | None = None,
    cutoff_minutes_before_first_pitch: int = 30,
    now: Callable[[], datetime] = utc_now,
    tracking_config: TrackingConfig | None = None,
) -> StarterStrikeoutMarketReportResult:
    """Build a sportsbook report from AGE-311 starter strikeout ML predictions."""
    output_root = Path(output_dir)
    generated_at = now().astimezone(UTC)
    report_run_root = (
        output_root
        / "normalized"
        / "starter_strikeout_market_report"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
    )
    run_id = _unique_timestamp_run_id(generated_at, report_run_root)
    report_run_dir = report_run_root / f"run={run_id}"
    resolved_ml_report_run_dir = _resolve_ml_report_run_dir(
        output_root,
        start_date=start_date,
        end_date=end_date,
        ml_report_run_dir=ml_report_run_dir,
        predictions_path=predictions_path,
    )
    resolved_predictions_path = (
        Path(predictions_path)
        if predictions_path is not None
        else resolved_ml_report_run_dir / "starter_strikeout_ml_predictions.jsonl"
    )
    adapted_model_run_dir = report_run_dir / "adapted_model_run"
    model_version = (
        f"{MARKET_REPORT_VERSION}:{_path_run_id(resolved_ml_report_run_dir)}"
    )
    adapter_summary = _adapt_predictions_to_backtest_model_run(
        predictions_path=resolved_predictions_path,
        adapted_model_run_dir=adapted_model_run_dir,
        start_date=start_date,
        end_date=end_date,
        model_version=model_version,
    )
    backtest_result = build_walk_forward_backtest(
        start_date=start_date,
        end_date=end_date,
        output_dir=output_root,
        model_run_dir=adapted_model_run_dir,
        odds_input_dir=odds_input_dir,
        cutoff_minutes_before_first_pitch=cutoff_minutes_before_first_pitch,
        now=lambda: generated_at,
        tracking_config=tracking_config,
    )
    backtest_rows = _load_backtest_rows(backtest_result.backtest_bets_path)
    summary_rows = _load_jsonl(backtest_result.backtest_runs_path)
    summary = summary_rows[0] if summary_rows else {}
    scoreable_rows = [
        row
        for row in backtest_rows
        if row["evaluation_status"] in {"actionable", "below_threshold"}
    ]
    skipped_rows = [
        row
        for row in backtest_rows
        if row["evaluation_status"] not in {"actionable", "below_threshold"}
    ]
    report_path = report_run_dir / "starter_strikeout_market_report.json"
    report_markdown_path = report_run_dir / "starter_strikeout_market_report.md"
    report = {
        "report_version": MARKET_REPORT_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "date_window": {"start_date": start_date, "end_date": end_date},
        "row_counts": {
            "snapshot_groups": backtest_result.snapshot_group_count,
            "scoreable_rows": len(scoreable_rows),
            "actionable_rows": backtest_result.actionable_bet_count,
            "below_threshold_rows": backtest_result.below_threshold_count,
            "skipped_rows": len(skipped_rows),
        },
        "status_counts": _count_by(backtest_rows, lambda row: str(row["evaluation_status"])),
        "join_failure_reasons": backtest_result.skip_reason_counts,
        "book_line_coverage": _book_line_coverage(backtest_rows),
        "clv": (summary.get("clv_summary") or {}),
        "roi": (summary.get("roi_summary") or {}),
        "calibration_by_line_bucket": _calibration_by_line_bucket(backtest_rows),
        "examples": _examples(backtest_rows),
        "source_artifacts": {
            "ml_report_run_dir": resolved_ml_report_run_dir,
            "ml_predictions_path": resolved_predictions_path,
            "adapted_model_run_dir": adapted_model_run_dir,
            "odds_input_dir": Path(odds_input_dir) if odds_input_dir is not None else output_root,
            "backtest_bets_path": backtest_result.backtest_bets_path,
            "join_audit_path": backtest_result.join_audit_path,
            "backtest_runs_path": backtest_result.backtest_runs_path,
            "clv_summary_path": backtest_result.clv_summary_path,
            "roi_summary_path": backtest_result.roi_summary_path,
        },
        "adapter_summary": adapter_summary,
        "timing_guardrails": {
            "required_order": "features_as_of <= projection_generated_at <= line captured_at",
            "enforced_by": "build_walk_forward_backtest timestamp_invalid_projection skip path",
        },
        "scope_guardrails": {
            "research_only_not_betting_ready": True,
            "dashboard_work_included": False,
            "forced_joins_allowed": False,
            "wager_approval_added": False,
        },
        "reproducibility": {
            "rerun_command": _rerun_command(
                start_date=start_date,
                end_date=end_date,
                output_dir=output_dir,
                ml_report_run_dir=resolved_ml_report_run_dir,
                odds_input_dir=odds_input_dir,
                cutoff_minutes_before_first_pitch=cutoff_minutes_before_first_pitch,
            )
        },
    }
    _write_json(report_path, report)
    _write_text(report_markdown_path, _render_markdown(report))
    return StarterStrikeoutMarketReportResult(
        start_date=start_date,
        end_date=end_date,
        run_id=run_id,
        report_path=report_path,
        report_markdown_path=report_markdown_path,
        adapted_model_run_dir=adapted_model_run_dir,
        backtest_result=backtest_result,
        scoreable_row_count=len(scoreable_rows),
        skipped_row_count=len(skipped_rows),
    )
