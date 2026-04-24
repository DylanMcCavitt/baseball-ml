"""Backtest policy and walk-forward evaluation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
import json
from pathlib import Path
from shlex import quote
from statistics import median
from typing import Any, Callable, Optional

from .config import StackConfig
from .edge import analyze_projection
from .markets import PropLine, PropProjection, ProjectionInputRef
from .pricing import american_to_decimal, devig_two_way
from .tracking import (
    TrackingConfig,
    log_run_artifact,
    log_run_metrics,
    log_run_params,
    start_experiment_run,
)

EDGE_BUCKETS = (
    (0.0, 0.02, "0_to_2_pct"),
    (0.02, 0.05, "2_to_5_pct"),
    (0.05, 0.08, "5_to_8_pct"),
    (0.08, None, "8_pct_plus"),
)


@dataclass(frozen=True)
class BacktestPolicy:
    """Rules that keep historical evaluation honest."""

    use_walk_forward_only: bool = True
    require_prelock_lines: bool = True
    require_pregame_feature_timestamps: bool = True
    require_projection_input_refs: bool = True
    preserve_rejected_props: bool = True
    report_clv: bool = True
    report_roi: bool = True
    report_edge_buckets: bool = True
    report_line_movement: bool = True

    def __post_init__(self) -> None:
        if not any(
            (
                self.report_clv,
                self.report_roi,
                self.report_edge_buckets,
                self.report_line_movement,
            )
        ):
            raise ValueError("at least one reporting output must be enabled")


BACKTEST_CHECKLIST = [
    "No feature may use information after the prop capture timestamp.",
    "Model training windows must end before the evaluated day begins.",
    "Each evaluated projection should point to explicit lineup and feature-row inputs.",
    "Closing-line value should be tracked separately from realized ROI.",
    "Backtests must include vig and any book-specific settlement quirks.",
    "Rejected bets should be preserved so threshold changes can be audited.",
]


@dataclass(frozen=True)
class WalkForwardBacktestResult:
    """Filesystem output summary for one walk-forward backtest run."""

    start_date: date
    end_date: date
    run_id: str
    mlflow_run_id: str
    mlflow_experiment_name: str
    model_version: str
    model_run_id: str
    cutoff_minutes_before_first_pitch: int
    backtest_bets_path: Path
    bet_reporting_path: Path
    backtest_runs_path: Path
    join_audit_path: Path
    clv_summary_path: Path
    roi_summary_path: Path
    edge_bucket_summary_path: Path
    reproducibility_notes_path: Path
    snapshot_group_count: int
    actionable_bet_count: int
    below_threshold_count: int
    skipped_count: int
    skip_reason_counts: dict[str, int]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.isoformat()
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _path_run_id(run_dir: Path) -> str:
    return run_dir.name.split("=", 1)[-1]


def _unique_timestamp_run_id(base_time: datetime, run_root: Path) -> str:
    candidate_time = base_time.astimezone(UTC)
    while True:
        run_id = candidate_time.strftime("%Y%m%dT%H%M%SZ")
        if not run_root.joinpath(f"run={run_id}").exists():
            return run_id
        candidate_time += timedelta(seconds=1)


def _requested_dates(start_date: date, end_date: date) -> list[date]:
    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date")
    day_count = (end_date - start_date).days + 1
    return [start_date + timedelta(days=offset) for offset in range(day_count)]


def _find_latest_model_run_covering_dates(
    output_root: Path,
    *,
    target_dates: list[date],
) -> Path:
    model_root = output_root / "normalized" / "starter_strikeout_baseline"
    run_dirs = sorted(
        path
        for path in model_root.rglob("run=*")
        if path.is_dir() and path.joinpath("training_dataset.jsonl").exists()
    )
    requested_dates = {target_date.isoformat() for target_date in target_dates}
    for run_dir in reversed(run_dirs):
        dataset_rows = _load_jsonl_rows(run_dir / "training_dataset.jsonl")
        available_dates = {str(row["official_date"]) for row in dataset_rows}
        if requested_dates.issubset(available_dates):
            return run_dir
    requested_label = ", ".join(sorted(requested_dates))
    raise FileNotFoundError(
        "No starter strikeout baseline run contains every requested date: "
        f"{requested_label}."
    )


def _line_group_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, float]:
    return (
        str(row["official_date"]),
        str(row["sportsbook"]),
        str(row["event_id"]),
        str(row["player_id"]),
        str(row["market"]),
        round(float(row["line"]), 6),
    )


def _snapshot_sort_key(row: dict[str, Any]) -> tuple[datetime, str]:
    return (
        _parse_datetime(str(row["captured_at"])),
        str(row["line_snapshot_id"]),
    )


def _load_snapshot_groups_for_dates(
    output_root: Path,
    *,
    target_dates: list[date],
) -> dict[tuple[str, str, str, str, str, float], list[dict[str, Any]]]:
    grouped_rows: dict[tuple[str, str, str, str, str, float], list[dict[str, Any]]] = {}
    for target_date in target_dates:
        odds_root = output_root / "normalized" / "the_odds_api" / f"date={target_date.isoformat()}"
        run_dirs = sorted(path for path in odds_root.glob("run=*") if path.is_dir())
        for run_dir in run_dirs:
            path = run_dir / "prop_line_snapshots.jsonl"
            if not path.exists():
                continue
            run_id = _path_run_id(run_dir)
            for row in _load_jsonl_rows(path):
                enriched_row = {**row, "_odds_run_id": run_id}
                grouped_rows.setdefault(_line_group_key(enriched_row), []).append(enriched_row)
    return grouped_rows


def _prop_line_from_snapshot_row(row: dict[str, Any]) -> PropLine:
    return PropLine(
        line_snapshot_id=str(row["line_snapshot_id"]),
        sportsbook=str(row["sportsbook"]),
        event_id=str(row["event_id"]),
        player_id=str(row["player_id"]),
        player_name=str(row["player_name"]),
        market=str(row["market"]),
        line=float(row["line"]),
        over_odds=int(row["over_odds"]),
        under_odds=int(row["under_odds"]),
        captured_at=_parse_datetime(str(row["captured_at"])),
    )


def _selected_market_probability(row: dict[str, Any], *, side: str) -> float:
    over_probability, under_probability = devig_two_way(
        int(row["over_odds"]),
        int(row["under_odds"]),
    )
    return over_probability if side == "over" else under_probability


def _settlement_status(*, actual_strikeouts: int, line: float, side: str) -> str:
    if actual_strikeouts > line:
        return "win" if side == "over" else "loss"
    if actual_strikeouts < line:
        return "win" if side == "under" else "loss"
    return "push"


def _profit_units_for_bet(*, stake_fraction: float, odds: int, settlement_status: str) -> float:
    if settlement_status == "win":
        return round(stake_fraction * (american_to_decimal(odds) - 1.0), 6)
    if settlement_status == "loss":
        return round(-stake_fraction, 6)
    return 0.0


def _edge_bucket_label(edge_pct: float) -> str:
    for lower_bound, upper_bound, label in EDGE_BUCKETS:
        if edge_pct < lower_bound:
            continue
        if upper_bound is None or edge_pct < upper_bound:
            return label
    return EDGE_BUCKETS[-1][2]


def _minutes_until(
    *,
    later_time: datetime | None,
    earlier_time: datetime | None,
) -> float | None:
    if later_time is None or earlier_time is None:
        return None
    return round((later_time - earlier_time).total_seconds() / 60.0, 6)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(median(values), 6)


def _ratio_or_none(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return round(numerator / denominator, 6)


def _clv_outcome_label(clv_probability_delta: float | None) -> str:
    if clv_probability_delta is None:
        return "no_closing_line"
    if clv_probability_delta > 0.0:
        return "beat_closing_line"
    if clv_probability_delta < 0.0:
        return "missed_closing_line"
    return "tied_closing_line"


def _build_bet_reporting_rows(
    *,
    backtest_rows: list[dict[str, Any]],
    backtest_run_id: str,
) -> list[dict[str, Any]]:
    reporting_rows: list[dict[str, Any]] = []
    for row in backtest_rows:
        probability_calibration = row.get("probability_calibration") or {}
        count_distribution = row.get("count_distribution") or {}
        clv_probability_delta = (
            float(row["clv_probability_delta"])
            if row.get("clv_probability_delta") is not None
            else None
        )
        edge_pct = float(row["edge_pct"]) if row.get("edge_pct") is not None else None
        expected_value_pct = (
            float(row["expected_value_pct"])
            if row.get("expected_value_pct") is not None
            else None
        )
        selected_model_probability = (
            float(row["selected_model_probability"])
            if row.get("selected_model_probability") is not None
            else None
        )
        selected_market_probability = (
            float(row["selected_market_probability"])
            if row.get("selected_market_probability") is not None
            else None
        )
        settlement_status = row.get("settlement_status")
        clv_outcome = _clv_outcome_label(clv_probability_delta)

        reporting_rows.append(
            {
                "backtest_run_id": backtest_run_id,
                "backtest_entry_id": str(row["backtest_entry_id"]),
                "official_date": str(row["official_date"]),
                "evaluation_status": str(row["evaluation_status"]),
                "bet_placed": bool(row.get("bet_placed")),
                "paper_result": (
                    str(settlement_status) if settlement_status is not None else "not_placed"
                ),
                "paper_win": (
                    settlement_status == "win"
                    if settlement_status is not None and bool(row.get("bet_placed"))
                    else None
                ),
                "beat_closing_line": (
                    clv_probability_delta > 0.0 if clv_probability_delta is not None else None
                ),
                "clv_outcome": clv_outcome,
                "same_line_close_available": row.get("closing_line_snapshot_id") is not None,
                "model_version": str(row["model_version"]),
                "model_run_id": str(row["model_run_id"]),
                "sportsbook": str(row["sportsbook"]),
                "sportsbook_title": str(row["sportsbook_title"]),
                "event_id": str(row["event_id"]),
                "game_pk": row.get("game_pk"),
                "player_id": str(row["player_id"]),
                "pitcher_mlb_id": row.get("pitcher_mlb_id"),
                "player_name": str(row["player_name"]),
                "market": str(row["market"]),
                "line": row.get("line"),
                "selected_side": row.get("selected_side"),
                "selected_odds": row.get("selected_odds"),
                "over_odds": row.get("over_odds"),
                "under_odds": row.get("under_odds"),
                "fair_odds": row.get("fair_odds"),
                "edge_pct": edge_pct,
                "edge_bucket": _edge_bucket_label(edge_pct) if edge_pct is not None else None,
                "expected_value_pct": expected_value_pct,
                "stake_fraction": row.get("stake_fraction"),
                "profit_units": row.get("profit_units"),
                "return_on_stake": row.get("return_on_stake"),
                "decision_snapshot_captured_at": row.get("decision_snapshot_captured_at"),
                "closing_snapshot_captured_at": row.get("closing_snapshot_captured_at"),
                "commence_time": row.get("commence_time"),
                "decision_minutes_before_first_pitch": _minutes_until(
                    later_time=row.get("commence_time"),
                    earlier_time=row.get("decision_snapshot_captured_at"),
                ),
                "closing_minutes_before_first_pitch": _minutes_until(
                    later_time=row.get("commence_time"),
                    earlier_time=row.get("closing_snapshot_captured_at"),
                ),
                "line_snapshot_id": row.get("line_snapshot_id"),
                "closing_line_snapshot_id": row.get("closing_line_snapshot_id"),
                "feature_row_id": row.get("feature_row_id"),
                "lineup_snapshot_id": row.get("lineup_snapshot_id"),
                "outcome_id": row.get("outcome_id"),
                "actual_strikeouts": row.get("actual_strikeouts"),
                "settlement_status": settlement_status,
                "selected_model_probability": selected_model_probability,
                "selected_market_probability": selected_market_probability,
                "model_minus_market_probability": (
                    round(selected_model_probability - selected_market_probability, 6)
                    if selected_model_probability is not None
                    and selected_market_probability is not None
                    else None
                ),
                "closing_selected_market_probability": row.get(
                    "closing_selected_market_probability"
                ),
                "clv_probability_delta": clv_probability_delta,
                "scatter_model_probability": selected_model_probability,
                "scatter_market_probability": selected_market_probability,
                "model_mean": row.get("model_mean"),
                "raw_model_over_probability": row.get("raw_model_over_probability"),
                "raw_model_under_probability": row.get("raw_model_under_probability"),
                "model_over_probability": row.get("model_over_probability"),
                "model_under_probability": row.get("model_under_probability"),
                "market_over_probability": row.get("market_over_probability"),
                "market_under_probability": row.get("market_under_probability"),
                "calibration_name": probability_calibration.get("name"),
                "calibration_sample_count": probability_calibration.get("sample_count"),
                "calibration_fit_from_date": probability_calibration.get("fit_from_date"),
                "calibration_fit_through_date": probability_calibration.get(
                    "fit_through_date"
                ),
                "calibration_is_identity": probability_calibration.get("is_identity"),
                "count_distribution_name": count_distribution.get("name"),
                "count_distribution_dispersion_alpha": count_distribution.get(
                    "dispersion_alpha"
                ),
                "reason": str(row["reason"]),
            }
        )

    reporting_rows.sort(
        key=lambda row: (
            str(row["official_date"]),
            0 if bool(row["bet_placed"]) else 1,
            -float(row.get("edge_pct") or 0.0),
            str(row["backtest_entry_id"]),
        )
    )
    return reporting_rows


def _build_clv_summary_row(
    *,
    rows: list[dict[str, Any]],
    backtest_run_id: str,
    summary_scope: str,
    official_date: str | None,
) -> dict[str, Any]:
    clv_values = [
        float(row["clv_probability_delta"])
        for row in rows
        if row.get("clv_probability_delta") is not None
    ]
    return {
        "backtest_run_id": backtest_run_id,
        "summary_scope": summary_scope,
        "official_date": official_date,
        "placed_bets": len(rows),
        "sample_count": len(clv_values),
        "missing_close_count": len(rows) - len(clv_values),
        "beat_closing_line_count": sum(
            1 for row in rows if row.get("clv_outcome") == "beat_closing_line"
        ),
        "tied_closing_line_count": sum(
            1 for row in rows if row.get("clv_outcome") == "tied_closing_line"
        ),
        "missed_closing_line_count": sum(
            1 for row in rows if row.get("clv_outcome") == "missed_closing_line"
        ),
        "paper_wins": sum(1 for row in rows if row.get("settlement_status") == "win"),
        "paper_losses": sum(1 for row in rows if row.get("settlement_status") == "loss"),
        "paper_pushes": sum(1 for row in rows if row.get("settlement_status") == "push"),
        "win_and_beat_closing_line_count": sum(
            1
            for row in rows
            if row.get("settlement_status") == "win"
            and row.get("clv_outcome") == "beat_closing_line"
        ),
        "win_without_beating_closing_line_count": sum(
            1
            for row in rows
            if row.get("settlement_status") == "win"
            and row.get("clv_outcome") != "beat_closing_line"
        ),
        "loss_but_beat_closing_line_count": sum(
            1
            for row in rows
            if row.get("settlement_status") == "loss"
            and row.get("clv_outcome") == "beat_closing_line"
        ),
        "mean_probability_delta": _mean_or_none(clv_values),
        "median_probability_delta": _median_or_none(clv_values),
    }


def _build_clv_summary_rows(
    *,
    reporting_rows: list[dict[str, Any]],
    backtest_run_id: str,
) -> list[dict[str, Any]]:
    placed_rows = [row for row in reporting_rows if bool(row["bet_placed"])]
    dated_rows = sorted({str(row["official_date"]) for row in placed_rows})
    summary_rows: list[dict[str, Any]] = []
    cumulative_rows: list[dict[str, Any]] = []
    for official_date in dated_rows:
        date_rows = [row for row in placed_rows if row["official_date"] == official_date]
        cumulative_rows.extend(date_rows)
        summary_row = _build_clv_summary_row(
            rows=date_rows,
            backtest_run_id=backtest_run_id,
            summary_scope="date",
            official_date=official_date,
        )
        cumulative_clv_values = [
            float(row["clv_probability_delta"])
            for row in cumulative_rows
            if row.get("clv_probability_delta") is not None
        ]
        summary_row["cumulative_placed_bets"] = len(cumulative_rows)
        summary_row["cumulative_sample_count"] = len(cumulative_clv_values)
        summary_row["cumulative_mean_probability_delta"] = _mean_or_none(
            cumulative_clv_values
        )
        summary_row["cumulative_beat_closing_line_count"] = sum(
            1
            for row in cumulative_rows
            if row.get("clv_outcome") == "beat_closing_line"
        )
        summary_rows.append(summary_row)

    summary_rows.append(
        _build_clv_summary_row(
            rows=placed_rows,
            backtest_run_id=backtest_run_id,
            summary_scope="overall",
            official_date=None,
        )
    )
    return summary_rows


def _build_roi_summary_row(
    *,
    rows: list[dict[str, Any]],
    backtest_run_id: str,
    summary_scope: str,
    official_date: str | None,
) -> dict[str, Any]:
    total_stake = round(sum(float(row["stake_fraction"]) for row in rows), 6)
    total_profit = round(sum(float(row["profit_units"]) for row in rows), 6)
    edge_values = [
        float(row["edge_pct"]) for row in rows if row.get("edge_pct") is not None
    ]
    expected_value_values = [
        float(row["expected_value_pct"])
        for row in rows
        if row.get("expected_value_pct") is not None
    ]
    realized_return_values = [
        float(row["return_on_stake"])
        for row in rows
        if row.get("return_on_stake") is not None
    ]
    return {
        "backtest_run_id": backtest_run_id,
        "summary_scope": summary_scope,
        "official_date": official_date,
        "placed_bets": len(rows),
        "wins": sum(1 for row in rows if row.get("settlement_status") == "win"),
        "losses": sum(1 for row in rows if row.get("settlement_status") == "loss"),
        "pushes": sum(1 for row in rows if row.get("settlement_status") == "push"),
        "beat_closing_line_count": sum(
            1 for row in rows if row.get("clv_outcome") == "beat_closing_line"
        ),
        "total_stake_units": total_stake,
        "total_profit_units": total_profit,
        "roi": _ratio_or_none(total_profit, total_stake),
        "mean_edge_pct": _mean_or_none(edge_values),
        "mean_expected_value_pct": _mean_or_none(expected_value_values),
        "mean_return_on_stake": _mean_or_none(realized_return_values),
    }


def _build_roi_summary_rows(
    *,
    reporting_rows: list[dict[str, Any]],
    backtest_run_id: str,
) -> list[dict[str, Any]]:
    placed_rows = [row for row in reporting_rows if bool(row["bet_placed"])]
    dated_rows = sorted({str(row["official_date"]) for row in placed_rows})
    summary_rows: list[dict[str, Any]] = []
    cumulative_rows: list[dict[str, Any]] = []
    for official_date in dated_rows:
        date_rows = [row for row in placed_rows if row["official_date"] == official_date]
        cumulative_rows.extend(date_rows)
        summary_row = _build_roi_summary_row(
            rows=date_rows,
            backtest_run_id=backtest_run_id,
            summary_scope="date",
            official_date=official_date,
        )
        cumulative_stake = round(
            sum(float(row["stake_fraction"]) for row in cumulative_rows),
            6,
        )
        cumulative_profit = round(
            sum(float(row["profit_units"]) for row in cumulative_rows),
            6,
        )
        summary_row["cumulative_placed_bets"] = len(cumulative_rows)
        summary_row["cumulative_total_stake_units"] = cumulative_stake
        summary_row["cumulative_total_profit_units"] = cumulative_profit
        summary_row["cumulative_roi"] = _ratio_or_none(cumulative_profit, cumulative_stake)
        summary_rows.append(summary_row)

    summary_rows.append(
        _build_roi_summary_row(
            rows=placed_rows,
            backtest_run_id=backtest_run_id,
            summary_scope="overall",
            official_date=None,
        )
    )
    return summary_rows


def _build_edge_bucket_summary_rows(
    *,
    reporting_rows: list[dict[str, Any]],
    backtest_run_id: str,
) -> list[dict[str, Any]]:
    placed_rows = [row for row in reporting_rows if bool(row["bet_placed"])]
    summary_rows: list[dict[str, Any]] = []
    for _, _, label in EDGE_BUCKETS:
        bucket_rows = [row for row in placed_rows if row.get("edge_bucket") == label]
        total_stake = round(sum(float(row["stake_fraction"]) for row in bucket_rows), 6)
        total_profit = round(sum(float(row["profit_units"]) for row in bucket_rows), 6)
        clv_values = [
            float(row["clv_probability_delta"])
            for row in bucket_rows
            if row.get("clv_probability_delta") is not None
        ]
        edge_values = [
            float(row["edge_pct"]) for row in bucket_rows if row.get("edge_pct") is not None
        ]
        expected_value_values = [
            float(row["expected_value_pct"])
            for row in bucket_rows
            if row.get("expected_value_pct") is not None
        ]
        summary_rows.append(
            {
                "backtest_run_id": backtest_run_id,
                "edge_bucket": label,
                "bet_count": len(bucket_rows),
                "wins": sum(
                    1 for row in bucket_rows if row.get("settlement_status") == "win"
                ),
                "losses": sum(
                    1 for row in bucket_rows if row.get("settlement_status") == "loss"
                ),
                "pushes": sum(
                    1 for row in bucket_rows if row.get("settlement_status") == "push"
                ),
                "sample_count": len(clv_values),
                "beat_closing_line_count": sum(
                    1
                    for row in bucket_rows
                    if row.get("clv_outcome") == "beat_closing_line"
                ),
                "total_stake_units": total_stake,
                "total_profit_units": total_profit,
                "roi": _ratio_or_none(total_profit, total_stake),
                "mean_edge_pct": _mean_or_none(edge_values),
                "mean_expected_value_pct": _mean_or_none(expected_value_values),
                "mean_probability_delta": _mean_or_none(clv_values),
            }
        )
    return summary_rows


def _base_row_from_snapshot_group(
    representative_row: dict[str, Any],
    *,
    model_version: str,
    model_run_id: str,
    commence_time: datetime,
    cutoff_time: datetime,
    snapshot_count: int,
    latest_observed_row: dict[str, Any],
    selected_row: dict[str, Any] | None,
    closing_row: dict[str, Any] | None,
) -> dict[str, Any]:
    line_value = round(float(representative_row["line"]), 6)
    return {
        "backtest_entry_id": (
            f"{representative_row['official_date']}|{representative_row['sportsbook']}|"
            f"{representative_row['event_id']}|{representative_row['player_id']}|"
            f"{representative_row['market']}|{line_value}|{model_version}"
        ),
        "official_date": str(representative_row["official_date"]),
        "model_version": model_version,
        "model_run_id": model_run_id,
        "sportsbook": str(representative_row["sportsbook"]),
        "sportsbook_title": str(
            representative_row.get("sportsbook_title") or representative_row["sportsbook"]
        ),
        "event_id": str(representative_row["event_id"]),
        "game_pk": representative_row.get("game_pk"),
        "odds_matchup_key": representative_row.get("odds_matchup_key"),
        "match_status": representative_row.get("match_status"),
        "player_id": str(representative_row["player_id"]),
        "pitcher_mlb_id": representative_row.get("pitcher_mlb_id"),
        "player_name": str(representative_row["player_name"]),
        "market": str(representative_row["market"]),
        "line": line_value,
        "commence_time": commence_time,
        "decision_cutoff_time": cutoff_time,
        "snapshot_count": snapshot_count,
        "latest_observed_line_snapshot_id": str(latest_observed_row["line_snapshot_id"]),
        "latest_observed_snapshot_captured_at": _parse_datetime(
            str(latest_observed_row["captured_at"])
        ),
        "latest_observed_odds_run_id": str(latest_observed_row["_odds_run_id"]),
        "line_snapshot_id": (
            str(selected_row["line_snapshot_id"]) if selected_row is not None else None
        ),
        "decision_snapshot_captured_at": (
            _parse_datetime(str(selected_row["captured_at"]))
            if selected_row is not None
            else None
        ),
        "decision_odds_run_id": (
            str(selected_row["_odds_run_id"]) if selected_row is not None else None
        ),
        "closing_line_snapshot_id": (
            str(closing_row["line_snapshot_id"]) if closing_row is not None else None
        ),
        "closing_snapshot_captured_at": (
            _parse_datetime(str(closing_row["captured_at"]))
            if closing_row is not None
            else None
        ),
        "closing_odds_run_id": (
            str(closing_row["_odds_run_id"]) if closing_row is not None else None
        ),
    }


def _audit_row(
    base_row: dict[str, Any],
    *,
    audit_status: str,
    reason: str,
    eligible_snapshot_count: int,
    late_snapshot_count: int,
    selected_snapshot_before_cutoff: bool,
    selected_snapshot_is_latest_before_cutoff: bool,
    features_as_of: datetime | None = None,
    projection_generated_at: datetime | None = None,
    data_split: str | None = None,
    model_train_from_date: str | None = None,
    model_train_through_date: str | None = None,
    calibration_fit_through_date: str | None = None,
    feature_row_id: str | None = None,
    lineup_snapshot_id: str | None = None,
    outcome_id: str | None = None,
    outcome_available: bool = False,
) -> dict[str, Any]:
    official_date = str(base_row["official_date"])
    training_window_ok = (
        model_train_through_date is not None and model_train_through_date < official_date
    )
    calibration_window_ok = (
        calibration_fit_through_date is None or calibration_fit_through_date < official_date
    )
    return {
        "audit_id": f"{base_row['backtest_entry_id']}|audit",
        "audit_status": audit_status,
        "reason": reason,
        "official_date": official_date,
        "line_snapshot_id": base_row["line_snapshot_id"],
        "latest_observed_line_snapshot_id": base_row["latest_observed_line_snapshot_id"],
        "decision_snapshot_captured_at": base_row["decision_snapshot_captured_at"],
        "latest_observed_snapshot_captured_at": base_row[
            "latest_observed_snapshot_captured_at"
        ],
        "commence_time": base_row["commence_time"],
        "decision_cutoff_time": base_row["decision_cutoff_time"],
        "eligible_snapshot_count": eligible_snapshot_count,
        "late_snapshot_count": late_snapshot_count,
        "selected_snapshot_before_cutoff": selected_snapshot_before_cutoff,
        "selected_snapshot_is_latest_before_cutoff": selected_snapshot_is_latest_before_cutoff,
        "feature_row_id": feature_row_id,
        "lineup_snapshot_id": lineup_snapshot_id,
        "features_as_of": features_as_of,
        "projection_generated_at": projection_generated_at,
        "features_before_cutoff": (
            features_as_of is not None and features_as_of <= base_row["decision_cutoff_time"]
        ),
        "projection_before_cutoff": (
            projection_generated_at is not None
            and projection_generated_at <= base_row["decision_cutoff_time"]
        ),
        "data_split": data_split,
        "model_train_from_date": model_train_from_date,
        "model_train_through_date": model_train_through_date,
        "training_window_before_evaluated_date": training_window_ok,
        "calibration_fit_through_date": calibration_fit_through_date,
        "calibration_window_before_evaluated_date": calibration_window_ok,
        "outcome_id": outcome_id,
        "outcome_available": outcome_available,
    }


def _line_probability_lookup_key(
    *,
    official_date: str,
    game_pk: int,
    pitcher_id: int,
    line: float,
) -> tuple[str, int, int, float]:
    return (
        official_date,
        game_pk,
        pitcher_id,
        round(line, 6),
    )


def _increment_reason_count(
    reason_counts: dict[str, int],
    *,
    reason_key: str,
) -> None:
    reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1


def _missing_snapshot_join_status(
    selected_row: dict[str, Any],
) -> tuple[str, str]:
    match_status = str(selected_row.get("match_status") or "")
    player_id = str(selected_row.get("player_id") or "")
    if match_status and match_status != "matched":
        return (
            "unmatched_event_mapping",
            (
                "Selected line snapshot still belongs to an unmatched Odds API event, "
                "so no honest MLB game join exists for backtest scoring."
            ),
        )
    if selected_row.get("game_pk") is None:
        return (
            "missing_game_mapping",
            (
                "Selected line snapshot is missing the mapped MLB game identifier "
                "required for walk-forward joins."
            ),
        )
    if selected_row.get("pitcher_mlb_id") is None:
        if player_id.startswith("odds-player:"):
            return (
                "unresolved_pitcher_identity",
                (
                    "Selected line snapshot could not resolve the sportsbook pitcher "
                    "name to an MLB probable-starter id, so no honest held-out "
                    "probability row can be joined."
                ),
            )
        return (
            "missing_pitcher_mapping",
            (
                "Selected line snapshot is missing the resolved MLB pitcher "
                "identifier required for walk-forward joins."
            ),
        )
    return (
        "missing_join_keys",
        "Selected line snapshot is missing mapped game or pitcher identifiers.",
    )


def build_walk_forward_backtest(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str = "data",
    model_run_dir: Path | str | None = None,
    cutoff_minutes_before_first_pitch: int = 30,
    now: Callable[[], datetime] | None = None,
    tracking_config: TrackingConfig | None = None,
) -> WalkForwardBacktestResult:
    """Build timestamp-safe walk-forward backtest artifacts for one date range."""
    if cutoff_minutes_before_first_pitch <= 0:
        raise ValueError("cutoff_minutes_before_first_pitch must be positive")
    if now is None:
        now = lambda: datetime.now(tz=UTC)

    policy = BacktestPolicy()
    config = StackConfig()
    tracking = tracking_config or TrackingConfig()
    output_root = Path(output_dir)
    target_dates = _requested_dates(start_date, end_date)
    resolved_model_run_dir = (
        Path(model_run_dir)
        if model_run_dir is not None
        else _find_latest_model_run_covering_dates(output_root, target_dates=target_dates)
    )

    training_dataset_path = resolved_model_run_dir / "training_dataset.jsonl"
    outcomes_path = resolved_model_run_dir / "starter_outcomes.jsonl"
    date_splits_path = resolved_model_run_dir / "date_splits.json"
    model_path = resolved_model_run_dir / "baseline_model.json"
    raw_vs_calibrated_path = resolved_model_run_dir / "raw_vs_calibrated_probabilities.jsonl"

    model_artifact = _load_json(model_path)
    model_version = str(model_artifact["model_version"])
    model_run_id = _path_run_id(resolved_model_run_dir)

    training_rows = _load_jsonl_rows(training_dataset_path)
    training_lookup = {
        (
            str(row["official_date"]),
            int(row["game_pk"]),
            int(row["pitcher_id"]),
        ): row
        for row in training_rows
    }
    split_payload = _load_json(date_splits_path)
    split_by_date = {
        split_date: split_name
        for split_name, split_dates in split_payload.items()
        for split_date in split_dates
    }
    honest_probability_rows = _load_jsonl_rows(raw_vs_calibrated_path)
    honest_probability_lookup = {
        _line_probability_lookup_key(
            official_date=str(row["official_date"]),
            game_pk=int(row["game_pk"]),
            pitcher_id=int(row["pitcher_id"]),
            line=float(row["line"]),
        ): row
        for row in honest_probability_rows
    }
    outcome_lookup = {
        (
            str(row["official_date"]),
            int(row["game_pk"]),
            int(row["pitcher_id"]),
        ): row
        for row in _load_jsonl_rows(outcomes_path)
    }
    snapshot_groups = _load_snapshot_groups_for_dates(output_root, target_dates=target_dates)

    backtest_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    actionable_bet_count = 0
    below_threshold_count = 0
    skipped_count = 0
    skip_reason_counts: dict[str, int] = {}

    for group_key in sorted(snapshot_groups):
        group_rows = sorted(snapshot_groups[group_key], key=_snapshot_sort_key)
        representative_row = group_rows[0]
        latest_observed_row = group_rows[-1]
        commence_time = _parse_datetime(str(latest_observed_row["commence_time"]))
        cutoff_time = commence_time - timedelta(
            minutes=cutoff_minutes_before_first_pitch
        )
        eligible_rows = [
            row
            for row in group_rows
            if _parse_datetime(str(row["captured_at"])) <= cutoff_time
        ]
        closing_rows = [
            row
            for row in group_rows
            if _parse_datetime(str(row["captured_at"])) <= commence_time
        ]
        selected_row = eligible_rows[-1] if eligible_rows else None
        closing_row = closing_rows[-1] if closing_rows else None
        base_row = _base_row_from_snapshot_group(
            representative_row,
            model_version=model_version,
            model_run_id=model_run_id,
            commence_time=commence_time,
            cutoff_time=cutoff_time,
            snapshot_count=len(group_rows),
            latest_observed_row=latest_observed_row,
            selected_row=selected_row,
            closing_row=closing_row,
        )
        late_snapshot_count = max(0, len(group_rows) - len(eligible_rows))

        if selected_row is None:
            skipped_count += 1
            evaluation_status = "late_snapshot_after_cutoff"
            _increment_reason_count(
                skip_reason_counts,
                reason_key=evaluation_status,
            )
            backtest_rows.append(
                {
                    **base_row,
                    "evaluation_status": evaluation_status,
                    "bet_placed": False,
                    "reason": (
                        "No snapshot for this exact line was available on or before the "
                        "configured backtest cutoff."
                    ),
                }
            )
            audit_rows.append(
                _audit_row(
                    base_row,
                    audit_status="late_snapshot_after_cutoff",
                    reason=(
                        "All snapshots for this exact line arrived after the backtest cutoff."
                    ),
                    eligible_snapshot_count=0,
                    late_snapshot_count=late_snapshot_count,
                    selected_snapshot_before_cutoff=False,
                    selected_snapshot_is_latest_before_cutoff=False,
                )
            )
            continue

        selected_game_pk = selected_row.get("game_pk")
        selected_pitcher_id = selected_row.get("pitcher_mlb_id")
        if selected_game_pk is None or selected_pitcher_id is None:
            skipped_count += 1
            evaluation_status, reason = _missing_snapshot_join_status(selected_row)
            _increment_reason_count(
                skip_reason_counts,
                reason_key=evaluation_status,
            )
            backtest_rows.append(
                {
                    **base_row,
                    "evaluation_status": evaluation_status,
                    "bet_placed": False,
                    "reason": reason,
                }
            )
            audit_rows.append(
                _audit_row(
                    base_row,
                    audit_status=evaluation_status,
                    reason=reason,
                    eligible_snapshot_count=len(eligible_rows),
                    late_snapshot_count=late_snapshot_count,
                    selected_snapshot_before_cutoff=True,
                    selected_snapshot_is_latest_before_cutoff=True,
                )
            )
            continue

        training_key = (
            str(selected_row["official_date"]),
            int(selected_game_pk),
            int(selected_pitcher_id),
        )
        training_row = training_lookup.get(training_key)
        if training_row is None:
            skipped_count += 1
            evaluation_status = "missing_projection"
            _increment_reason_count(
                skip_reason_counts,
                reason_key=evaluation_status,
            )
            reason = "No feature-backed training row was found for the selected line snapshot."
            backtest_rows.append(
                {
                    **base_row,
                    "evaluation_status": evaluation_status,
                    "bet_placed": False,
                    "reason": reason,
                }
            )
            audit_rows.append(
                _audit_row(
                    base_row,
                    audit_status=evaluation_status,
                    reason=reason,
                    eligible_snapshot_count=len(eligible_rows),
                    late_snapshot_count=late_snapshot_count,
                    selected_snapshot_before_cutoff=True,
                    selected_snapshot_is_latest_before_cutoff=True,
                )
            )
            continue

        split_name = split_by_date.get(str(selected_row["official_date"]))
        if split_name is None:
            skipped_count += 1
            evaluation_status = "missing_projection"
            _increment_reason_count(
                skip_reason_counts,
                reason_key=evaluation_status,
            )
            reason = "The selected line snapshot is not covered by the model date splits."
            backtest_rows.append(
                {
                    **base_row,
                    "evaluation_status": evaluation_status,
                    "bet_placed": False,
                    "feature_row_id": str(training_row["training_row_id"]),
                    "lineup_snapshot_id": training_row.get("lineup_snapshot_id"),
                    "reason": reason,
                }
            )
            audit_rows.append(
                _audit_row(
                    base_row,
                    audit_status=evaluation_status,
                    reason=reason,
                    eligible_snapshot_count=len(eligible_rows),
                    late_snapshot_count=late_snapshot_count,
                    selected_snapshot_before_cutoff=True,
                    selected_snapshot_is_latest_before_cutoff=True,
                    feature_row_id=str(training_row["training_row_id"]),
                    lineup_snapshot_id=(
                        str(training_row["lineup_snapshot_id"])
                        if training_row.get("lineup_snapshot_id") is not None
                        else None
                    ),
                )
            )
            continue

        if split_name == "train":
            skipped_count += 1
            evaluation_status = "train_split_projection"
            _increment_reason_count(
                skip_reason_counts,
                reason_key=evaluation_status,
            )
            reason = (
                "The selected line only maps to a training-split row and is not honest "
                "for walk-forward evaluation."
            )
            backtest_rows.append(
                {
                    **base_row,
                    "evaluation_status": evaluation_status,
                    "bet_placed": False,
                    "data_split": split_name,
                    "feature_row_id": str(training_row["training_row_id"]),
                    "lineup_snapshot_id": training_row.get("lineup_snapshot_id"),
                    "features_as_of": _parse_datetime(str(training_row["features_as_of"])),
                    "projection_generated_at": _parse_datetime(
                        str(training_row["features_as_of"])
                    ),
                    "reason": reason,
                }
            )
            audit_rows.append(
                _audit_row(
                    base_row,
                    audit_status=evaluation_status,
                    reason=reason,
                    eligible_snapshot_count=len(eligible_rows),
                    late_snapshot_count=late_snapshot_count,
                    selected_snapshot_before_cutoff=True,
                    selected_snapshot_is_latest_before_cutoff=True,
                    features_as_of=_parse_datetime(str(training_row["features_as_of"])),
                    projection_generated_at=_parse_datetime(
                        str(training_row["features_as_of"])
                    ),
                    data_split=split_name,
                    feature_row_id=str(training_row["training_row_id"]),
                    lineup_snapshot_id=(
                        str(training_row["lineup_snapshot_id"])
                        if training_row.get("lineup_snapshot_id") is not None
                        else None
                    ),
                )
            )
            continue

        probability_row = honest_probability_lookup.get(
            _line_probability_lookup_key(
                official_date=str(selected_row["official_date"]),
                game_pk=int(selected_game_pk),
                pitcher_id=int(selected_pitcher_id),
                line=float(selected_row["line"]),
            )
        )
        if probability_row is None:
            skipped_count += 1
            evaluation_status = "missing_line_probability"
            _increment_reason_count(
                skip_reason_counts,
                reason_key=evaluation_status,
            )
            reason = "No honest held-out probability row was found for this exact line."
            backtest_rows.append(
                {
                    **base_row,
                    "evaluation_status": evaluation_status,
                    "bet_placed": False,
                    "data_split": split_name,
                    "feature_row_id": str(training_row["training_row_id"]),
                    "lineup_snapshot_id": training_row.get("lineup_snapshot_id"),
                    "reason": reason,
                }
            )
            audit_rows.append(
                _audit_row(
                    base_row,
                    audit_status=evaluation_status,
                    reason=reason,
                    eligible_snapshot_count=len(eligible_rows),
                    late_snapshot_count=late_snapshot_count,
                    selected_snapshot_before_cutoff=True,
                    selected_snapshot_is_latest_before_cutoff=True,
                    data_split=split_name,
                    feature_row_id=str(training_row["training_row_id"]),
                    lineup_snapshot_id=(
                        str(training_row["lineup_snapshot_id"])
                        if training_row.get("lineup_snapshot_id") is not None
                        else None
                    ),
                )
            )
            continue

        lineup_snapshot_id = training_row.get("lineup_snapshot_id")
        if lineup_snapshot_id is None:
            skipped_count += 1
            evaluation_status = "missing_lineup_snapshot_id"
            _increment_reason_count(
                skip_reason_counts,
                reason_key=evaluation_status,
            )
            reason = "The matched held-out projection does not carry a lineup snapshot reference."
            backtest_rows.append(
                {
                    **base_row,
                    "evaluation_status": evaluation_status,
                    "bet_placed": False,
                    "data_split": split_name,
                    "feature_row_id": str(training_row["training_row_id"]),
                    "reason": reason,
                }
            )
            audit_rows.append(
                _audit_row(
                    base_row,
                    audit_status=evaluation_status,
                    reason=reason,
                    eligible_snapshot_count=len(eligible_rows),
                    late_snapshot_count=late_snapshot_count,
                    selected_snapshot_before_cutoff=True,
                    selected_snapshot_is_latest_before_cutoff=True,
                    data_split=split_name,
                    feature_row_id=str(training_row["training_row_id"]),
                )
            )
            continue

        outcome_row = outcome_lookup.get(training_key)
        if outcome_row is None:
            skipped_count += 1
            evaluation_status = "missing_result"
            _increment_reason_count(
                skip_reason_counts,
                reason_key=evaluation_status,
            )
            reason = "No same-game outcome record was found for the selected projection."
            backtest_rows.append(
                {
                    **base_row,
                    "evaluation_status": evaluation_status,
                    "bet_placed": False,
                    "data_split": split_name,
                    "feature_row_id": str(training_row["training_row_id"]),
                    "lineup_snapshot_id": str(lineup_snapshot_id),
                    "reason": reason,
                }
            )
            audit_rows.append(
                _audit_row(
                    base_row,
                    audit_status=evaluation_status,
                    reason=reason,
                    eligible_snapshot_count=len(eligible_rows),
                    late_snapshot_count=late_snapshot_count,
                    selected_snapshot_before_cutoff=True,
                    selected_snapshot_is_latest_before_cutoff=True,
                    data_split=split_name,
                    feature_row_id=str(training_row["training_row_id"]),
                    lineup_snapshot_id=str(lineup_snapshot_id),
                )
            )
            continue

        features_as_of = _parse_datetime(str(training_row["features_as_of"]))
        projection_generated_at = _parse_datetime(str(training_row["features_as_of"]))
        try:
            analysis = analyze_projection(
                _prop_line_from_snapshot_row(selected_row),
                PropProjection(
                    event_id=str(selected_row["event_id"]),
                    player_id=str(selected_row["player_id"]),
                    market=str(selected_row["market"]),
                    line=float(selected_row["line"]),
                    mean=float(probability_row["model_mean"]),
                    over_probability=float(probability_row["calibrated_over_probability"]),
                    under_probability=float(probability_row["calibrated_under_probability"]),
                    model_version=model_version,
                    input_ref=ProjectionInputRef(
                        lineup_snapshot_id=str(lineup_snapshot_id),
                        feature_row_id=str(training_row["training_row_id"]),
                        features_as_of=features_as_of,
                    ),
                    generated_at=projection_generated_at,
                ),
                config=config,
            )
        except (TypeError, ValueError) as error:
            skipped_count += 1
            evaluation_status = "invalid_projection"
            _increment_reason_count(
                skip_reason_counts,
                reason_key=evaluation_status,
            )
            reason = str(error)
            backtest_rows.append(
                {
                    **base_row,
                    "evaluation_status": evaluation_status,
                    "bet_placed": False,
                    "data_split": split_name,
                    "feature_row_id": str(training_row["training_row_id"]),
                    "lineup_snapshot_id": str(lineup_snapshot_id),
                    "features_as_of": features_as_of,
                    "projection_generated_at": projection_generated_at,
                    "reason": reason,
                }
            )
            audit_rows.append(
                _audit_row(
                    base_row,
                    audit_status=evaluation_status,
                    reason=reason,
                    eligible_snapshot_count=len(eligible_rows),
                    late_snapshot_count=late_snapshot_count,
                    selected_snapshot_before_cutoff=True,
                    selected_snapshot_is_latest_before_cutoff=True,
                    features_as_of=features_as_of,
                    projection_generated_at=projection_generated_at,
                    data_split=split_name,
                    model_train_from_date=probability_row.get("model_train_from_date"),
                    model_train_through_date=probability_row.get("model_train_through_date"),
                    calibration_fit_through_date=probability_row.get(
                        "calibration_fit_through_date"
                    ),
                    feature_row_id=str(training_row["training_row_id"]),
                    lineup_snapshot_id=str(lineup_snapshot_id),
                    outcome_id=str(outcome_row["outcome_id"]),
                    outcome_available=True,
                )
            )
            continue

        settlement_status = _settlement_status(
            actual_strikeouts=int(outcome_row["starter_strikeouts"]),
            line=float(selected_row["line"]),
            side=str(analysis["side"]),
        )
        bet_placed = bool(analysis["clears_min_edge"])
        profit_units = (
            _profit_units_for_bet(
                stake_fraction=float(analysis["stake_fraction"]),
                odds=int(analysis["selected_odds"]),
                settlement_status=settlement_status,
            )
            if bet_placed
            else 0.0
        )
        if bet_placed:
            actionable_bet_count += 1
            evaluation_status = "actionable"
        else:
            below_threshold_count += 1
            evaluation_status = "below_threshold"

        closing_selected_market_probability = (
            _selected_market_probability(closing_row, side=str(analysis["side"]))
            if closing_row is not None
            else None
        )
        decision_selected_market_probability = float(analysis["selected_market_probability"])
        clv_probability_delta = (
            round(
                closing_selected_market_probability - decision_selected_market_probability,
                6,
            )
            if closing_selected_market_probability is not None
            else None
        )

        backtest_rows.append(
            {
                **base_row,
                "evaluation_status": evaluation_status,
                "bet_placed": bet_placed,
                "data_split": split_name,
                "feature_row_id": str(training_row["training_row_id"]),
                "lineup_snapshot_id": str(lineup_snapshot_id),
                "features_as_of": features_as_of,
                "projection_generated_at": projection_generated_at,
                "model_train_from_date": probability_row.get("model_train_from_date"),
                "model_train_through_date": probability_row.get("model_train_through_date"),
                "model_mean": round(float(probability_row["model_mean"]), 6),
                "count_distribution": dict(probability_row["count_distribution"]),
                "probability_calibration": {
                    "name": str(probability_row["calibration_method"]),
                    "sample_count": int(probability_row["calibration_sample_count"]),
                    "fit_from_date": probability_row.get("calibration_fit_from_date"),
                    "fit_through_date": probability_row.get("calibration_fit_through_date"),
                    "is_identity": bool(probability_row["calibration_is_identity"]),
                    "training_splits": list(
                        probability_row["calibration_training_splits"]
                    ),
                },
                "raw_model_over_probability": round(
                    float(probability_row["raw_over_probability"]),
                    6,
                ),
                "raw_model_under_probability": round(
                    float(probability_row["raw_under_probability"]),
                    6,
                ),
                "model_over_probability": round(
                    float(probability_row["calibrated_over_probability"]),
                    6,
                ),
                "model_under_probability": round(
                    float(probability_row["calibrated_under_probability"]),
                    6,
                ),
                "market_over_probability": round(
                    float(analysis["market_over_probability"]),
                    6,
                ),
                "market_under_probability": round(
                    float(analysis["market_under_probability"]),
                    6,
                ),
                "selected_side": str(analysis["side"]),
                "selected_model_probability": round(
                    float(analysis["selected_model_probability"]),
                    6,
                ),
                "selected_market_probability": round(
                    decision_selected_market_probability,
                    6,
                ),
                "selected_odds": int(analysis["selected_odds"]),
                "over_odds": int(selected_row["over_odds"]),
                "under_odds": int(selected_row["under_odds"]),
                "edge_pct": round(float(analysis["edge_pct"]), 6),
                "expected_value_pct": round(float(analysis["expected_value_pct"]), 6),
                "uncapped_stake_fraction": round(
                    float(analysis["uncapped_stake_fraction"]),
                    6,
                ),
                "stake_fraction": round(float(analysis["stake_fraction"]), 6),
                "fair_odds": int(analysis["fair_odds"]),
                "clears_min_edge": bool(analysis["clears_min_edge"]),
                "closing_over_odds": (
                    int(closing_row["over_odds"]) if closing_row is not None else None
                ),
                "closing_under_odds": (
                    int(closing_row["under_odds"]) if closing_row is not None else None
                ),
                "closing_selected_market_probability": (
                    round(closing_selected_market_probability, 6)
                    if closing_selected_market_probability is not None
                    else None
                ),
                "clv_probability_delta": clv_probability_delta,
                "outcome_id": str(outcome_row["outcome_id"]),
                "actual_strikeouts": int(outcome_row["starter_strikeouts"]),
                "settlement_status": settlement_status,
                "profit_units": profit_units,
                "return_on_stake": (
                    round(
                        profit_units / float(analysis["stake_fraction"]),
                        6,
                    )
                    if bet_placed and float(analysis["stake_fraction"]) > 0.0
                    else None
                ),
                "reason": str(analysis["reason"]),
            }
        )
        audit_rows.append(
            _audit_row(
                base_row,
                audit_status="ok",
                reason="Selected snapshot, projection refs, and outcome all passed cutoff checks.",
                eligible_snapshot_count=len(eligible_rows),
                late_snapshot_count=late_snapshot_count,
                selected_snapshot_before_cutoff=True,
                selected_snapshot_is_latest_before_cutoff=True,
                features_as_of=features_as_of,
                projection_generated_at=projection_generated_at,
                data_split=split_name,
                model_train_from_date=probability_row.get("model_train_from_date"),
                model_train_through_date=probability_row.get("model_train_through_date"),
                calibration_fit_through_date=probability_row.get(
                    "calibration_fit_through_date"
                ),
                feature_row_id=str(training_row["training_row_id"]),
                lineup_snapshot_id=str(lineup_snapshot_id),
                outcome_id=str(outcome_row["outcome_id"]),
                outcome_available=True,
            )
        )

    backtest_rows.sort(
        key=lambda row: (
            0 if row["evaluation_status"] == "actionable" else 1,
            -float(row.get("edge_pct") or 0.0),
            str(row["backtest_entry_id"]),
        )
    )
    audit_rows.sort(key=lambda row: str(row["audit_id"]))

    sorted_skip_reason_counts = dict(sorted(skip_reason_counts.items()))
    run_root = (
        output_root
        / "normalized"
        / "walk_forward_backtest"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
    )
    run_id = _unique_timestamp_run_id(now().astimezone(UTC), run_root)
    normalized_root = run_root / f"run={run_id}"
    backtest_bets_path = normalized_root / "backtest_bets.jsonl"
    bet_reporting_path = normalized_root / "bet_reporting.jsonl"
    backtest_runs_path = normalized_root / "backtest_runs.jsonl"
    join_audit_path = normalized_root / "join_audit.jsonl"
    clv_summary_path = normalized_root / "clv_summary.jsonl"
    roi_summary_path = normalized_root / "roi_summary.jsonl"
    edge_bucket_summary_path = normalized_root / "edge_bucket_summary.jsonl"
    reproducibility_notes_path = normalized_root / "reproducibility_notes.md"

    placed_bets = [row for row in backtest_rows if bool(row.get("bet_placed"))]
    total_stake = round(sum(float(row["stake_fraction"]) for row in placed_bets), 6)
    total_profit = round(sum(float(row["profit_units"]) for row in placed_bets), 6)
    reporting_rows = _build_bet_reporting_rows(
        backtest_rows=backtest_rows,
        backtest_run_id=run_id,
    )
    clv_summary_rows = _build_clv_summary_rows(
        reporting_rows=reporting_rows,
        backtest_run_id=run_id,
    )
    roi_summary_rows = _build_roi_summary_rows(
        reporting_rows=reporting_rows,
        backtest_run_id=run_id,
    )
    edge_bucket_summary_rows = _build_edge_bucket_summary_rows(
        reporting_rows=reporting_rows,
        backtest_run_id=run_id,
    )
    overall_clv_summary = next(
        row for row in clv_summary_rows if row["summary_scope"] == "overall"
    )
    overall_roi_summary = next(
        row for row in roi_summary_rows if row["summary_scope"] == "overall"
    )
    rerun_command = (
        "uv run python -m mlb_props_stack build-walk-forward-backtest "
        f"--start-date {start_date.isoformat()} "
        f"--end-date {end_date.isoformat()} "
        f"--output-dir {quote(str(output_dir))} "
        f"--model-run-dir {quote(str(resolved_model_run_dir))} "
        "--cutoff-minutes-before-first-pitch "
        f"{cutoff_minutes_before_first_pitch}"
    )
    run_name = (
        f"walk-forward-backtest-{start_date.isoformat()}-"
        f"{end_date.isoformat()}-{run_id}"
    )

    with start_experiment_run(
        experiment_name=tracking.backtest_experiment_name,
        run_name=run_name,
        tags={
            "run_kind": "backtest",
            "pipeline": "build_walk_forward_backtest",
            "model_version": model_version,
            "model_run_id": model_run_id,
        },
        config=tracking,
    ) as tracking_run:
        summary_row = {
            "backtest_run_id": run_id,
            "mlflow_run_id": tracking_run.run_id,
            "mlflow_experiment_name": tracking_run.experiment_name,
            "tracking_uri": tracking_run.tracking_uri,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "evaluated_dates": [target_date.isoformat() for target_date in target_dates],
            "model_version": model_version,
            "model_run_id": model_run_id,
            "cutoff_minutes_before_first_pitch": cutoff_minutes_before_first_pitch,
            "policy": asdict(policy),
            "row_counts": {
                "snapshot_groups": len(snapshot_groups),
                "actionable": actionable_bet_count,
                "below_threshold": below_threshold_count,
                "skipped": skipped_count,
            },
            "skip_reason_counts": sorted_skip_reason_counts,
            "bet_outcomes": {
                "placed_bets": len(placed_bets),
                "wins": sum(1 for row in placed_bets if row["settlement_status"] == "win"),
                "losses": sum(1 for row in placed_bets if row["settlement_status"] == "loss"),
                "pushes": sum(1 for row in placed_bets if row["settlement_status"] == "push"),
                "total_stake_units": total_stake,
                "total_profit_units": total_profit,
                "roi": round(total_profit / total_stake, 6) if total_stake > 0.0 else None,
            },
            "clv_summary": (
                {
                    "sample_count": overall_clv_summary["sample_count"],
                    "mean_probability_delta": overall_clv_summary["mean_probability_delta"],
                    "median_probability_delta": overall_clv_summary["median_probability_delta"],
                    "beat_closing_line_count": overall_clv_summary[
                        "beat_closing_line_count"
                    ],
                    "missing_close_count": overall_clv_summary["missing_close_count"],
                }
                if policy.report_clv
                else {
                    "sample_count": 0,
                    "mean_probability_delta": None,
                    "median_probability_delta": None,
                    "beat_closing_line_count": 0,
                    "missing_close_count": len(placed_bets),
                }
            ),
            "roi_summary": (
                {
                    "placed_bets": overall_roi_summary["placed_bets"],
                    "wins": overall_roi_summary["wins"],
                    "losses": overall_roi_summary["losses"],
                    "pushes": overall_roi_summary["pushes"],
                    "total_stake_units": overall_roi_summary["total_stake_units"],
                    "total_profit_units": overall_roi_summary["total_profit_units"],
                    "roi": overall_roi_summary["roi"],
                }
                if policy.report_roi
                else {
                    "placed_bets": 0,
                    "wins": 0,
                    "losses": 0,
                    "pushes": 0,
                    "total_stake_units": 0.0,
                    "total_profit_units": 0.0,
                    "roi": None,
                }
            ),
            "edge_bucket_summary": edge_bucket_summary_rows,
            "source_artifacts": {
                "training_dataset_path": training_dataset_path,
                "raw_vs_calibrated_path": raw_vs_calibrated_path,
                "outcomes_path": outcomes_path,
            },
            "reporting_artifacts": {
                "bet_reporting_path": bet_reporting_path,
                "clv_summary_path": clv_summary_path,
                "roi_summary_path": roi_summary_path,
                "edge_bucket_summary_path": edge_bucket_summary_path,
            },
            "reproducibility": {
                "notes_path": str(reproducibility_notes_path),
                "rerun_command": rerun_command,
            },
        }

        _write_jsonl(backtest_bets_path, backtest_rows)
        _write_jsonl(bet_reporting_path, reporting_rows)
        _write_jsonl(backtest_runs_path, [summary_row])
        _write_jsonl(join_audit_path, audit_rows)
        _write_jsonl(clv_summary_path, clv_summary_rows)
        _write_jsonl(roi_summary_path, roi_summary_rows)
        _write_jsonl(edge_bucket_summary_path, edge_bucket_summary_rows)
        _write_text(
            reproducibility_notes_path,
            "\n".join(
                [
                    "# Walk-Forward Backtest Reproducibility Notes",
                    "",
                    f"- Local backtest run ID: `{run_id}`",
                    f"- MLflow run ID: `{tracking_run.run_id}`",
                    f"- MLflow experiment: `{tracking_run.experiment_name}`",
                    f"- Tracking URI: `{tracking_run.tracking_uri}`",
                    f"- Local run directory: `{normalized_root}`",
                    (
                        "- Date window: "
                        f"`{start_date.isoformat()}` -> `{end_date.isoformat()}`"
                    ),
                    f"- Source model run ID: `{model_run_id}`",
                    f"- Source model run directory: `{resolved_model_run_dir}`",
                    (
                        "- Cutoff minutes before first pitch: "
                        f"`{cutoff_minutes_before_first_pitch}`"
                    ),
                    f"- CLI rerun command: `{rerun_command}`",
                    (
                        "- Honest rules: the rerun pins the exact model run directory so the "
                        "same saved historical probabilities and timestamps are replayed."
                    ),
                    "",
                ]
            ),
        )
        log_run_params(
            {
                "pipeline": "build_walk_forward_backtest",
                "model_version": model_version,
                "model_run_id": model_run_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "output_dir": str(output_dir),
                "resolved_model_run_dir": str(resolved_model_run_dir),
                "cutoff_minutes_before_first_pitch": cutoff_minutes_before_first_pitch,
            }
        )
        log_run_metrics(
            {
                "snapshot_groups": len(snapshot_groups),
                "actionable_bets": actionable_bet_count,
                "below_threshold": below_threshold_count,
                "skipped": skipped_count,
                "placed_bets": summary_row["bet_outcomes"]["placed_bets"],
                "wins": summary_row["bet_outcomes"]["wins"],
                "losses": summary_row["bet_outcomes"]["losses"],
                "pushes": summary_row["bet_outcomes"]["pushes"],
                "total_stake_units": summary_row["bet_outcomes"]["total_stake_units"],
                "total_profit_units": summary_row["bet_outcomes"]["total_profit_units"],
                "roi": summary_row["bet_outcomes"]["roi"],
                "clv_sample_count": summary_row["clv_summary"]["sample_count"],
                "clv_mean_probability_delta": summary_row["clv_summary"][
                    "mean_probability_delta"
                ],
                "clv_beat_closing_line_count": summary_row["clv_summary"][
                    "beat_closing_line_count"
                ],
            }
        )
        log_run_artifact(normalized_root)

    return WalkForwardBacktestResult(
        start_date=start_date,
        end_date=end_date,
        run_id=run_id,
        mlflow_run_id=tracking_run.run_id,
        mlflow_experiment_name=tracking_run.experiment_name,
        model_version=model_version,
        model_run_id=model_run_id,
        cutoff_minutes_before_first_pitch=cutoff_minutes_before_first_pitch,
        backtest_bets_path=backtest_bets_path,
        bet_reporting_path=bet_reporting_path,
        backtest_runs_path=backtest_runs_path,
        join_audit_path=join_audit_path,
        clv_summary_path=clv_summary_path,
        roi_summary_path=roi_summary_path,
        edge_bucket_summary_path=edge_bucket_summary_path,
        reproducibility_notes_path=reproducibility_notes_path,
        snapshot_group_count=len(snapshot_groups),
        actionable_bet_count=actionable_bet_count,
        below_threshold_count=below_threshold_count,
        skipped_count=skipped_count,
        skip_reason_counts=sorted_skip_reason_counts,
    )
