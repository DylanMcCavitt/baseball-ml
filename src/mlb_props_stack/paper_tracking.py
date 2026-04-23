"""Daily candidate sheet and paper-tracking workflow helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
import json
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from .config import StackConfig
from .edge import build_edge_candidates_for_date
from .ingest.mlb_stats_api import utc_now
from .modeling import generate_starter_strikeout_inference_for_date
from .pricing import american_to_decimal, devig_two_way
from .wager_approval import WagerApprovalSettings, annotate_wager_approval_rows


@dataclass(frozen=True)
class DailyCandidateWorkflowResult:
    """Filesystem output summary for one daily candidate workflow run."""

    target_date: date
    run_id: str
    inference_run_id: str
    edge_candidate_run_id: str
    daily_candidates_path: Path
    paper_results_path: Path
    scored_candidate_count: int
    actionable_candidate_count: int
    settled_result_count: int
    pending_result_count: int
    approved_wager_count: int | None = None


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
    if isinstance(value, date):
        return value.isoformat()
    return value


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_ready(row), sort_keys=True))
            handle.write("\n")


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _path_run_id(run_dir: Path) -> str:
    return run_dir.name.split("=", 1)[-1]


def _resolve_target_date(
    target_date: date | None,
    *,
    now: Callable[[], datetime],
) -> date:
    if target_date is not None:
        return target_date
    config = StackConfig()
    current_time = now().astimezone(ZoneInfo(config.timezone))
    return current_time.date()


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


def _latest_daily_candidate_run_dirs(
    output_root: Path,
    *,
    through_date: date,
) -> dict[str, Path]:
    daily_root = output_root / "normalized" / "daily_candidates"
    if not daily_root.exists():
        return {}
    latest_runs: dict[str, Path] = {}
    for date_dir in daily_root.glob("date=*"):
        official_date = date_dir.name.split("=", 1)[-1]
        if date.fromisoformat(official_date) > through_date:
            continue
        run_dirs = sorted(path for path in date_dir.glob("run=*") if path.is_dir())
        if run_dirs:
            latest_runs[official_date] = run_dirs[-1]
    return latest_runs


def list_available_daily_candidate_dates(
    *,
    output_dir: Path | str = "data",
) -> list[str]:
    output_root = Path(output_dir)
    return sorted(
        _latest_daily_candidate_run_dirs(
            output_root,
            through_date=date.max,
        ).keys()
    )


def _load_latest_daily_candidates_for_date(
    *,
    output_root: Path,
    target_date: date,
) -> list[dict[str, Any]]:
    run_dir = _latest_daily_candidate_run_dirs(
        output_root,
        through_date=target_date,
    ).get(target_date.isoformat())
    if run_dir is None:
        raise FileNotFoundError(
            "No daily candidate run was found for "
            f"{target_date.isoformat()}."
        )
    return _load_jsonl_rows(run_dir / "daily_candidates.jsonl")


def _load_latest_paper_results_for_date(
    *,
    output_root: Path,
    target_date: date,
) -> list[dict[str, Any]]:
    paper_root = (
        output_root
        / "normalized"
        / "paper_results"
        / f"date={target_date.isoformat()}"
    )
    run_dirs = sorted(path for path in paper_root.glob("run=*") if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(
            "No paper results run was found for "
            f"{target_date.isoformat()}."
        )
    return _load_jsonl_rows(run_dirs[-1] / "paper_results.jsonl")


def load_latest_daily_candidates(
    *,
    output_dir: Path | str = "data",
    target_date: date | None = None,
) -> list[dict[str, Any]]:
    output_root = Path(output_dir)
    available_dates = list_available_daily_candidate_dates(output_dir=output_root)
    if not available_dates:
        return []
    resolved_target_date = (
        target_date if target_date is not None else date.fromisoformat(available_dates[-1])
    )
    return _load_latest_daily_candidates_for_date(
        output_root=output_root,
        target_date=resolved_target_date,
    )


def load_latest_paper_results(
    *,
    output_dir: Path | str = "data",
    target_date: date | None = None,
) -> list[dict[str, Any]]:
    output_root = Path(output_dir)
    if target_date is not None:
        try:
            return _load_latest_paper_results_for_date(
                output_root=output_root,
                target_date=target_date,
            )
        except FileNotFoundError:
            return []

    paper_root = output_root / "normalized" / "paper_results"
    if not paper_root.exists():
        return []
    date_dirs = sorted(path for path in paper_root.glob("date=*") if path.is_dir())
    if not date_dirs:
        return []
    latest_date = date.fromisoformat(date_dirs[-1].name.split("=", 1)[-1])
    return _load_latest_paper_results_for_date(
        output_root=output_root,
        target_date=latest_date,
    )


def summarize_paper_results_by_date(
    rows: list[dict[str, Any]],
    *,
    max_dates: int = 7,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["official_date"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for official_date in sorted(grouped.keys(), reverse=True)[:max_dates]:
        date_rows = grouped[official_date]
        settled_rows = [
            row
            for row in date_rows
            if row.get("settlement_status") in {"win", "loss", "push"}
        ]
        total_stake = round(
            sum(float(row["stake_fraction"]) for row in settled_rows),
            6,
        )
        total_profit = round(
            sum(float(row["profit_units"]) for row in settled_rows),
            6,
        )
        summary_rows.append(
            {
                "official_date": official_date,
                "actionable_candidates": len(date_rows),
                "settled_bets": len(settled_rows),
                "pending_bets": len(date_rows) - len(settled_rows),
                "wins": sum(
                    1 for row in settled_rows if row.get("settlement_status") == "win"
                ),
                "losses": sum(
                    1 for row in settled_rows if row.get("settlement_status") == "loss"
                ),
                "pushes": sum(
                    1 for row in settled_rows if row.get("settlement_status") == "push"
                ),
                "beat_closing_line_count": sum(
                    1 for row in settled_rows if row.get("clv_outcome") == "beat_closing_line"
                ),
                "total_stake_units": total_stake,
                "total_profit_units": total_profit,
                "roi": _ratio_or_none(total_profit, total_stake),
            }
        )
    return summary_rows


def _load_latest_outcomes_for_date(
    output_root: Path,
    *,
    official_date: str,
) -> dict[tuple[int, int], dict[str, Any]]:
    outcome_lookup: dict[tuple[int, int], dict[str, Any]] = {}
    model_root = output_root / "normalized" / "starter_strikeout_baseline"
    for run_dir in sorted(path for path in model_root.rglob("run=*") if path.is_dir()):
        outcomes_path = run_dir / "starter_outcomes.jsonl"
        if not outcomes_path.exists():
            continue
        for row in _load_jsonl_rows(outcomes_path):
            if str(row.get("official_date")) != official_date:
                continue
            outcome_lookup[(int(row["game_pk"]), int(row["pitcher_id"]))] = row
    return outcome_lookup


def _load_snapshot_rows_for_date(
    output_root: Path,
    *,
    official_date: str,
) -> list[dict[str, Any]]:
    odds_root = output_root / "normalized" / "the_odds_api" / f"date={official_date}"
    if not odds_root.exists():
        return []
    snapshot_rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in odds_root.glob("run=*") if path.is_dir()):
        snapshots_path = run_dir / "prop_line_snapshots.jsonl"
        if not snapshots_path.exists():
            continue
        run_id = _path_run_id(run_dir)
        snapshot_rows.extend(
            {**row, "_odds_run_id": run_id}
            for row in _load_jsonl_rows(snapshots_path)
        )
    return snapshot_rows


def _find_closing_snapshot(
    snapshot_rows: list[dict[str, Any]],
    *,
    candidate_row: dict[str, Any],
) -> dict[str, Any] | None:
    decision_captured_at = _parse_datetime(str(candidate_row["captured_at"]))
    matching_rows = [
        row
        for row in snapshot_rows
        if str(row["sportsbook"]) == str(candidate_row["sportsbook"])
        and str(row["event_id"]) == str(candidate_row["event_id"])
        and str(row["player_id"]) == str(candidate_row["player_id"])
        and str(row["market"]) == str(candidate_row["market"])
        and round(float(row["line"]), 6) == round(float(candidate_row["line"]), 6)
    ]
    if decision_captured_at is None:
        return None
    later_rows = [
        row
        for row in matching_rows
        if (captured_at := _parse_datetime(str(row["captured_at"]))) is not None
        and captured_at >= decision_captured_at
    ]
    if not later_rows:
        return None
    return max(
        later_rows,
        key=lambda row: (
            _parse_datetime(str(row["captured_at"])),
            str(row["line_snapshot_id"]),
        ),
    )


def _build_daily_candidate_rows(
    edge_rows: list[dict[str, Any]],
    *,
    run_id: str,
    inference_run_id: str,
    edge_candidate_run_id: str,
    approval_settings: WagerApprovalSettings | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    if approval_settings is None:
        approval_settings = WagerApprovalSettings()
    scored_rows = [
        row
        for row in edge_rows
        if row.get("evaluation_status") in {"actionable", "below_threshold"}
    ]
    daily_candidate_rows: list[dict[str, Any]] = []
    actionable_rank = 1
    for slate_rank, row in enumerate(scored_rows, start=1):
        evaluation_status = str(row["evaluation_status"])
        daily_candidate_rows.append(
            {
                **row,
                "daily_candidate_id": str(row["candidate_id"]),
                "daily_candidate_run_id": run_id,
                "inference_run_id": inference_run_id,
                "edge_candidate_run_id": edge_candidate_run_id,
                "slate_rank": slate_rank,
                "actionable_rank": (
                    actionable_rank if evaluation_status == "actionable" else None
                ),
                "bet_placed": False,
            }
        )
        if evaluation_status == "actionable":
            actionable_rank += 1
    approved_rows = annotate_wager_approval_rows(
        daily_candidate_rows,
        settings=approval_settings,
        now=now,
    )
    approved_rank = 1
    for row in approved_rows:
        wager_approved = bool(row["wager_approved"])
        row["approved_rank"] = approved_rank if wager_approved else None
        row["bet_placed"] = wager_approved
        if wager_approved:
            approved_rank += 1
    return approved_rows


def _paper_row_is_placed(row: dict[str, Any]) -> bool:
    if "wager_approved" in row:
        return bool(row["wager_approved"])
    return bool(row.get("bet_placed"))


def _build_paper_result_rows(
    output_root: Path,
    *,
    through_date: date,
    run_id: str,
) -> list[dict[str, Any]]:
    latest_daily_runs = _latest_daily_candidate_run_dirs(
        output_root,
        through_date=through_date,
    )
    paper_result_rows: list[dict[str, Any]] = []
    outcome_cache: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    snapshot_cache: dict[str, list[dict[str, Any]]] = {}
    for official_date in sorted(latest_daily_runs.keys()):
        daily_candidate_rows = _load_jsonl_rows(
            latest_daily_runs[official_date] / "daily_candidates.jsonl"
        )
        actionable_rows = [row for row in daily_candidate_rows if _paper_row_is_placed(row)]
        if official_date not in outcome_cache:
            outcome_cache[official_date] = _load_latest_outcomes_for_date(
                output_root,
                official_date=official_date,
            )
        if official_date not in snapshot_cache:
            snapshot_cache[official_date] = _load_snapshot_rows_for_date(
                output_root,
                official_date=official_date,
            )
        outcomes = outcome_cache[official_date]
        snapshot_rows = snapshot_cache[official_date]

        for row in actionable_rows:
            outcome_row = None
            game_pk = row.get("game_pk")
            pitcher_mlb_id = row.get("pitcher_mlb_id")
            if game_pk is not None and pitcher_mlb_id is not None:
                outcome_row = outcomes.get((int(game_pk), int(pitcher_mlb_id)))
            closing_row = _find_closing_snapshot(snapshot_rows, candidate_row=row)
            closing_selected_market_probability = (
                round(
                    _selected_market_probability(
                        closing_row,
                        side=str(row["selected_side"]),
                    ),
                    6,
                )
                if closing_row is not None
                else None
            )
            selected_market_probability = float(row["selected_market_probability"])
            clv_probability_delta = (
                round(
                    closing_selected_market_probability - selected_market_probability,
                    6,
                )
                if closing_selected_market_probability is not None
                else None
            )
            settlement_status = None
            profit_units = None
            return_on_stake = None
            actual_strikeouts = None
            if outcome_row is not None:
                actual_strikeouts = int(outcome_row["starter_strikeouts"])
                settlement_status = _settlement_status(
                    actual_strikeouts=actual_strikeouts,
                    line=float(row["line"]),
                    side=str(row["selected_side"]),
                )
                profit_units = _profit_units_for_bet(
                    stake_fraction=float(row["stake_fraction"]),
                    odds=int(row["selected_odds"]),
                    settlement_status=settlement_status,
                )
                return_on_stake = (
                    round(profit_units / float(row["stake_fraction"]), 6)
                    if float(row["stake_fraction"]) > 0.0
                    else None
                )

            paper_result_rows.append(
                {
                    "paper_result_id": f"{row['daily_candidate_id']}|paper",
                    "paper_results_run_id": run_id,
                    "daily_candidate_id": str(row["daily_candidate_id"]),
                    "daily_candidate_run_id": str(row["daily_candidate_run_id"]),
                    "official_date": str(row["official_date"]),
                    "slate_rank": int(row["slate_rank"]),
                    "actionable_rank": row.get("actionable_rank"),
                    "approved_rank": row.get("approved_rank"),
                    "wager_approved": bool(row.get("wager_approved", row.get("bet_placed"))),
                    "wager_gate_status": row.get("wager_gate_status"),
                    "wager_blocked_reason": row.get("wager_blocked_reason"),
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
                    "line": float(row["line"]),
                    "selected_side": str(row["selected_side"]),
                    "selected_odds": int(row["selected_odds"]),
                    "fair_odds": int(row["fair_odds"]),
                    "edge_pct": float(row["edge_pct"]),
                    "expected_value_pct": float(row["expected_value_pct"]),
                    "stake_fraction": float(row["stake_fraction"]),
                    "line_snapshot_id": str(row["line_snapshot_id"]),
                    "decision_snapshot_captured_at": _parse_datetime(str(row["captured_at"])),
                    "closing_line_snapshot_id": (
                        str(closing_row["line_snapshot_id"]) if closing_row is not None else None
                    ),
                    "closing_snapshot_captured_at": (
                        _parse_datetime(str(closing_row["captured_at"]))
                        if closing_row is not None
                        else None
                    ),
                    "same_line_close_available": closing_row is not None,
                    "selected_model_probability": float(row["selected_model_probability"]),
                    "selected_market_probability": selected_market_probability,
                    "closing_selected_market_probability": closing_selected_market_probability,
                    "clv_probability_delta": clv_probability_delta,
                    "beat_closing_line": (
                        clv_probability_delta > 0.0
                        if clv_probability_delta is not None
                        else None
                    ),
                    "clv_outcome": _clv_outcome_label(clv_probability_delta),
                    "paper_result": (
                        settlement_status if settlement_status is not None else "pending"
                    ),
                    "paper_win": (
                        settlement_status == "win"
                        if settlement_status is not None
                        else None
                    ),
                    "settlement_status": settlement_status,
                    "actual_strikeouts": actual_strikeouts,
                    "profit_units": profit_units,
                    "return_on_stake": return_on_stake,
                }
            )

    paper_result_rows.sort(
        key=lambda row: (
            str(row["official_date"]),
            int(row.get("approved_rank") or row.get("actionable_rank") or 0),
            str(row["paper_result_id"]),
        )
    )
    return paper_result_rows


def build_daily_candidate_workflow(
    *,
    target_date: date | None = None,
    output_dir: Path | str = "data",
    source_model_run_dir: Path | str | None = None,
    now: Callable[[], datetime] = utc_now,
) -> DailyCandidateWorkflowResult:
    """Build one daily candidate sheet and refresh cumulative paper results."""
    resolved_target_date = _resolve_target_date(target_date, now=now)
    output_root = Path(output_dir)
    inference_result = generate_starter_strikeout_inference_for_date(
        target_date=resolved_target_date,
        output_dir=output_root,
        source_model_run_dir=source_model_run_dir,
        now=now,
    )
    edge_result = build_edge_candidates_for_date(
        target_date=resolved_target_date,
        output_dir=output_root,
        model_run_dir=inference_result.model_path.parent,
    )
    edge_rows = _load_jsonl_rows(edge_result.edge_candidates_path)
    run_id = edge_result.run_id
    daily_candidate_rows = _build_daily_candidate_rows(
        edge_rows,
        run_id=run_id,
        inference_run_id=inference_result.run_id,
        edge_candidate_run_id=edge_result.run_id,
        now=now(),
    )
    daily_candidates_path = (
        output_root
        / "normalized"
        / "daily_candidates"
        / f"date={resolved_target_date.isoformat()}"
        / f"run={run_id}"
        / "daily_candidates.jsonl"
    )
    _write_jsonl(daily_candidates_path, daily_candidate_rows)

    paper_result_rows = _build_paper_result_rows(
        output_root,
        through_date=resolved_target_date,
        run_id=run_id,
    )
    paper_results_path = (
        output_root
        / "normalized"
        / "paper_results"
        / f"date={resolved_target_date.isoformat()}"
        / f"run={run_id}"
        / "paper_results.jsonl"
    )
    _write_jsonl(paper_results_path, paper_result_rows)

    settled_result_count = sum(
        1 for row in paper_result_rows if row.get("settlement_status") is not None
    )
    pending_result_count = sum(
        1 for row in paper_result_rows if row.get("settlement_status") is None
    )
    actionable_candidate_count = sum(
        1 for row in daily_candidate_rows if row.get("evaluation_status") == "actionable"
    )
    approved_wager_count = sum(
        1 for row in daily_candidate_rows if bool(row.get("wager_approved"))
    )
    return DailyCandidateWorkflowResult(
        target_date=resolved_target_date,
        run_id=run_id,
        inference_run_id=inference_result.run_id,
        edge_candidate_run_id=edge_result.run_id,
        daily_candidates_path=daily_candidates_path,
        paper_results_path=paper_results_path,
        scored_candidate_count=len(daily_candidate_rows),
        actionable_candidate_count=actionable_candidate_count,
        settled_result_count=settled_result_count,
        pending_result_count=pending_result_count,
        approved_wager_count=approved_wager_count,
    )
