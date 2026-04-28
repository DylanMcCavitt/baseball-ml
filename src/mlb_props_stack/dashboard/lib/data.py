"""Artifact-backed data loaders for the Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from math import exp, lgamma, log
import json
from pathlib import Path
from typing import Any

import pandas as pd

from mlb_props_stack.data_alignment import collect_artifact_counts_for_date
from mlb_props_stack.modeling import (
    CORE_NUMERIC_FEATURES,
    MIN_FEATURE_VARIANCE,
    OPTIONAL_FEATURE_MIN_COVERAGE,
    OPTIONAL_NUMERIC_FEATURES,
)
from mlb_props_stack.paper_tracking import (
    list_available_daily_candidate_dates,
    load_latest_daily_candidates,
    load_latest_paper_results,
    summarize_paper_results_by_date,
)
from mlb_props_stack.pricing import (
    american_to_implied_probability,
    fractional_kelly,
)
from mlb_props_stack.tracking import TrackingConfig
from mlb_props_stack.wager_approval import (
    WagerApprovalSettings,
    annotate_wager_approval_rows,
)

from .mlflow_io import registered_versions_by_run_id, search_runs


@dataclass(frozen=True)
class DashboardSettings(WagerApprovalSettings):
    """Persisted dashboard controls that affect server-side row evaluation."""

    devig_method: str = "shin"
    calibration_method: str = "isotonic"
    seed: int = 42
    decision_cutoff_minutes: int = 90
    refresh_cadence_minutes: int = 5
    active_run_id: str | None = None
    active_model_label: str | None = None


OPTIONAL_FEATURE_FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "key": "split_features",
        "label": "Split Features",
        "features": (
            "pitcher_k_rate_vs_rhh",
            "pitcher_k_rate_vs_lhh",
            "pitcher_whiff_rate_vs_rhh",
            "pitcher_whiff_rate_vs_lhh",
        ),
        "target_artifact": "pitcher_daily_features.jsonl",
        "status_fields": (),
        "aliases": {},
    },
    {
        "key": "lineup_aggregate_features",
        "label": "Lineup Aggregates",
        "features": (
            "projected_lineup_k_rate",
            "projected_lineup_k_rate_vs_pitcher_hand",
            "lineup_k_rate_vs_rhp",
            "lineup_k_rate_vs_lhp",
            "projected_lineup_chase_rate",
            "projected_lineup_contact_rate",
            "lineup_continuity_ratio",
        ),
        "target_artifact": "lineup_daily_features.jsonl",
        "status_fields": ("lineup_status",),
        "aliases": {},
    },
    {
        "key": "park_factors",
        "label": "Park Factors",
        "features": (
            "park_k_factor",
            "park_k_factor_vs_rhh",
            "park_k_factor_vs_lhh",
        ),
        "target_artifact": "game_context_features.jsonl",
        "status_fields": ("park_factor_status",),
        "aliases": {"park_k_factor": ("park_factor",)},
    },
    {
        "key": "weather",
        "label": "Weather",
        "features": (
            "weather_temperature_f",
            "weather_wind_speed_mph",
            "weather_humidity_pct",
        ),
        "target_artifact": "game_context_features.jsonl",
        "status_fields": ("weather_status",),
        "aliases": {"weather_wind_speed_mph": ("weather_wind_mph",)},
    },
    {
        "key": "umpire",
        "label": "Umpire",
        "features": (
            "ump_called_strike_rate_30d",
            "ump_k_per_9_delta_vs_league_30d",
        ),
        "target_artifact": "game_context_features.jsonl",
        "status_fields": ("umpire_status",),
        "aliases": {},
    },
)


def parse_datetime(value: str | None) -> datetime | None:
    """Parse an ISO timestamp if present."""
    if value is None or value == "":
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def format_timestamp(value: datetime | None) -> str:
    """Return a compact ET-friendly timestamp label."""
    if value is None:
        return "n/a"
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def format_number(value: float | int | None, *, digits: int = 2, suffix: str = "") -> str:
    """Format a scalar for UI rendering."""
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value}{suffix}"
    return f"{value:.{digits}f}{suffix}"


def format_pct(value: float | None, *, digits: int = 1) -> str:
    """Format a probability or rate."""
    if value is None:
        return "n/a"
    return f"{value:.{digits}%}"


@lru_cache(maxsize=None)
def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _run_id_from_dir(run_dir: Path) -> str:
    return run_dir.name.split("=", 1)[-1]


def _parse_run_id(run_id: str | None) -> datetime | None:
    if not run_id:
        return None
    try:
        return datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def _latest_run_dirs(root: Path, artifact_name: str) -> list[Path]:
    if not root.exists():
        return []
    run_dirs = [
        path
        for path in root.rglob("run=*")
        if path.is_dir() and path.joinpath(artifact_name).exists()
    ]
    return sorted(run_dirs)


def _available_edge_dates(output_root: Path) -> list[str]:
    edge_root = output_root / "normalized" / "edge_candidates"
    if not edge_root.exists():
        return []
    dates: list[str] = []
    for date_dir in sorted(edge_root.glob("date=*")):
        run_dirs = sorted(path for path in date_dir.glob("run=*") if path.is_dir())
        if run_dirs:
            dates.append(date_dir.name.split("=", 1)[-1])
    return dates


def _available_backtest_dates(output_root: Path) -> list[str]:
    run_dir = _latest_backtest_run_dir(output_root)
    if run_dir is None:
        return []
    reporting_path = run_dir / "bet_reporting.jsonl"
    if reporting_path.exists():
        return sorted(
            {
                str(row["official_date"])
                for row in _load_jsonl(str(reporting_path))
                if row.get("official_date") is not None
            }
        )
    backtest_bets_path = run_dir / "backtest_bets.jsonl"
    if backtest_bets_path.exists():
        return sorted(
            {
                str(row["official_date"])
                for row in _load_jsonl(str(backtest_bets_path))
                if row.get("official_date") is not None
            }
        )
    summary = load_backtest_summary(output_root)
    if summary is None:
        return []
    return sorted(str(value) for value in summary.get("evaluated_dates", []) if value is not None)


def list_available_board_dates(*, output_root: Path) -> list[str]:
    """Return target dates that can populate the board."""
    dates = set(list_available_daily_candidate_dates(output_dir=output_root))
    dates.update(_available_edge_dates(output_root))
    dates.update(_available_backtest_dates(output_root))
    return sorted(dates)


def _load_edge_candidates(output_root: Path, *, target_date: date) -> list[dict[str, Any]]:
    edge_root = (
        output_root / "normalized" / "edge_candidates" / f"date={target_date.isoformat()}"
    )
    run_dirs = sorted(path for path in edge_root.glob("run=*") if path.is_dir())
    if not run_dirs:
        return []
    return _load_jsonl(str(run_dirs[-1] / "edge_candidates.jsonl"))


def _load_backtest_board_rows(
    output_root: Path,
    *,
    target_date: date,
) -> list[dict[str, Any]]:
    run_dir = _latest_backtest_run_dir(output_root)
    if run_dir is None:
        return []

    rows: list[dict[str, Any]] = []
    reporting_path = run_dir / "bet_reporting.jsonl"
    if reporting_path.exists():
        rows = _load_jsonl(str(reporting_path))
    else:
        backtest_bets_path = run_dir / "backtest_bets.jsonl"
        if backtest_bets_path.exists():
            rows = _load_jsonl(str(backtest_bets_path))

    target_date_str = target_date.isoformat()
    return [
        row
        for row in rows
        if str(row.get("official_date")) == target_date_str
        and str(row.get("evaluation_status")) in {"actionable", "below_threshold"}
        and row.get("selected_side") is not None
    ]


def _latest_model_runs(output_root: Path) -> list[Path]:
    return _latest_run_dirs(
        output_root / "normalized" / "starter_strikeout_baseline",
        "evaluation_summary.json",
    )


def _latest_distribution_model_runs(output_root: Path) -> list[Path]:
    return _latest_run_dirs(
        output_root / "normalized" / "candidate_strikeout_models",
        "model_outputs.jsonl",
    )


def _model_run_lookup(output_root: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for run_dir in _latest_model_runs(output_root):
        lookup[_run_id_from_dir(run_dir)] = run_dir
    return lookup


def _distribution_model_run_lookup(output_root: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for run_dir in _latest_distribution_model_runs(output_root):
        lookup[_run_id_from_dir(run_dir)] = run_dir
    return lookup


def _ladder_run_lookup(output_root: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    roots = [
        output_root / "normalized" / "starter_strikeout_inference",
        output_root / "normalized" / "starter_strikeout_baseline",
    ]
    for root in roots:
        for run_dir in _latest_run_dirs(root, "ladder_probabilities.jsonl"):
            lookup[_run_id_from_dir(run_dir)] = run_dir
    return lookup


def _latest_ladder_run_dirs(output_root: Path) -> list[Path]:
    roots = [
        output_root / "normalized" / "starter_strikeout_inference",
        output_root / "normalized" / "starter_strikeout_baseline",
    ]
    return sorted(
        (
            run_dir
            for root in roots
            for run_dir in _latest_run_dirs(root, "ladder_probabilities.jsonl")
        ),
        key=_run_id_from_dir,
    )


def _latest_stage_gate_report(output_root: Path) -> dict[str, Any] | None:
    reports = sorted(
        path
        for path in (output_root / "normalized" / "stage_gates").rglob(
            "stage_gate_report.json"
        )
        if path.is_file()
    )
    if not reports:
        return None
    return _load_json(str(reports[-1]))


def stack_readiness_status(output_root: Path) -> str:
    """Return the latest saved stage-gate status, defaulting to research_only."""
    report = _latest_stage_gate_report(output_root)
    if report is None:
        return "research_only"
    return str(report.get("status") or "research_only")


def latest_model_run_dir(
    output_root: Path,
    *,
    run_id: str | None = None,
) -> Path | None:
    """Resolve the current model run directory."""
    lookup = _model_run_lookup(output_root)
    if run_id is not None and run_id in lookup:
        return lookup[run_id]
    run_dirs = sorted(lookup.values())
    return run_dirs[-1] if run_dirs else None


def load_latest_model_summary(
    output_root: Path,
    *,
    run_id: str | None = None,
) -> dict[str, Any] | None:
    """Load the latest model evaluation summary."""
    run_dir = latest_model_run_dir(output_root, run_id=run_id)
    if run_dir is None:
        return None
    return _load_json(str(run_dir / "evaluation_summary.json"))


def _latest_backtest_run_dir(output_root: Path) -> Path | None:
    run_dirs = _latest_run_dirs(
        output_root / "normalized" / "walk_forward_backtest",
        "backtest_runs.jsonl",
    )
    return run_dirs[-1] if run_dirs else None


def load_backtest_summary(output_root: Path) -> dict[str, Any] | None:
    """Load the latest walk-forward backtest summary row."""
    run_dir = _latest_backtest_run_dir(output_root)
    if run_dir is None:
        return None
    rows = _load_jsonl(str(run_dir / "backtest_runs.jsonl"))
    return rows[-1] if rows else None


def _lookup_feature_metadata(output_root: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
    for run_dir in _latest_run_dirs(
        output_root / "normalized" / "starter_strikeout_baseline",
        "training_dataset.jsonl",
    ):
        for row in _load_jsonl(str(run_dir / "training_dataset.jsonl")):
            game_pk = row.get("game_pk")
            pitcher_id = row.get("pitcher_id")
            official_date = row.get("official_date")
            if game_pk is None or pitcher_id is None or official_date is None:
                continue
            lookup[(str(official_date), int(game_pk), int(pitcher_id))] = row
    return lookup


def _lookup_line_snapshot_metadata(output_root: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for run_dir in _latest_run_dirs(
        output_root / "normalized" / "the_odds_api",
        "prop_line_snapshots.jsonl",
    ):
        for row in _load_jsonl(str(run_dir / "prop_line_snapshots.jsonl")):
            line_snapshot_id = row.get("line_snapshot_id")
            if line_snapshot_id:
                lookup[str(line_snapshot_id)] = row
    return lookup


def _sportsbook_key_from_snapshot_id(line_snapshot_id: str | None) -> str:
    if not line_snapshot_id:
        return ""
    parts = str(line_snapshot_id).split(":")
    if len(parts) >= 2 and parts[0] == "prop-line":
        return parts[1]
    return ""


def _short_identifier(value: Any, *, length: int = 8) -> str:
    if value is None or value == "":
        return "n/a"
    return str(value)[:length]


def line_snapshot_display_id(line_snapshot_id: str | None) -> str:
    """Return a compact line-snapshot label for dashboard provenance."""
    if not line_snapshot_id:
        return "n/a"
    parts = str(line_snapshot_id).split(":")
    if len(parts) >= 2 and parts[0] == "prop-line":
        return parts[-1]
    return _short_identifier(line_snapshot_id, length=12)


def _line_label(row: dict[str, Any] | pd.Series) -> str:
    side = str(row.get("side") or row.get("selected_side") or "").upper()
    line = row.get("line")
    try:
        line_label = f"{float(line):.1f}"
    except (TypeError, ValueError):
        line_label = "n/a"
    sportsbook = str(row.get("sportsbook") or row.get("sportsbook_title") or "unknown book")
    return f"{sportsbook} {side} {line_label}".strip()


def _compact_join(values: list[str], *, limit: int = 4) -> str:
    clean_values = []
    for value in values:
        if not value or value in clean_values:
            continue
        clean_values.append(value)
    if len(clean_values) <= limit:
        return ", ".join(clean_values)
    remaining = len(clean_values) - limit
    return f"{', '.join(clean_values[:limit])}, +{remaining} more"


def _ensure_board_source_rows(
    output_root: Path,
    *,
    target_date: date,
) -> tuple[list[dict[str, Any]], str | None]:
    daily_rows = load_latest_daily_candidates(
        output_dir=output_root,
        target_date=target_date,
    )
    if daily_rows:
        return daily_rows, "daily_candidates"
    edge_rows = _load_edge_candidates(output_root, target_date=target_date)
    if edge_rows:
        return edge_rows, "edge_candidates"
    backtest_rows = _load_backtest_board_rows(output_root, target_date=target_date)
    if backtest_rows:
        return backtest_rows, "walk_forward_backtest"
    return [], None


def devig(prices: list[int] | tuple[int, int], method: str) -> tuple[float, float]:
    """Remove hold from two-way American prices."""
    if len(prices) != 2:
        raise ValueError("dashboard devig only supports two-way markets")
    raw = [american_to_implied_probability(int(price)) for price in prices]
    if method == "multiplicative":
        total = sum(raw)
        values = [probability / total for probability in raw]
    elif method == "power":
        lower = 0.01
        upper = 25.0
        for _ in range(60):
            midpoint = (lower + upper) / 2.0
            total = sum(probability**midpoint for probability in raw)
            if total > 1.0:
                lower = midpoint
            else:
                upper = midpoint
        exponent = (lower + upper) / 2.0
        values = [probability**exponent for probability in raw]
    elif method == "shin":
        margin = sum(raw) - 1.0
        values = [probability - (margin / 2.0) for probability in raw]
    else:
        raise ValueError(f"unsupported devig method: {method}")
    clipped = [max(1e-6, probability) for probability in values]
    total = sum(clipped)
    return clipped[0] / total, clipped[1] / total


def _side_probability(row: dict[str, Any], *, side: str, field_prefix: str) -> float | None:
    if side == "over":
        value = row.get(f"{field_prefix}_over_probability")
    else:
        value = row.get(f"{field_prefix}_under_probability")
    return float(value) if value is not None else None


def _raw_hold(over_odds: int | None, under_odds: int | None) -> float | None:
    if over_odds is None or under_odds is None:
        return None
    return (
        american_to_implied_probability(int(over_odds))
        + american_to_implied_probability(int(under_odds))
        - 1.0
    )


def _short_reason(reason: str | None) -> str:
    if not reason:
        return "artifact-backed board row"
    lowered = str(reason).lower()
    if "clears minimum edge threshold" in lowered:
        return "clears edge gate"
    if "does not clear minimum edge threshold" in lowered:
        return "below edge gate"
    if "missing" in lowered:
        return str(reason).replace("_", " ")
    return str(reason)


def _normalize_board_dataframe(
    rows: list[dict[str, Any]],
    *,
    source: str,
    output_root: Path,
    settings: DashboardSettings,
) -> pd.DataFrame:
    feature_lookup = _lookup_feature_metadata(output_root)
    line_lookup = _lookup_line_snapshot_metadata(output_root)
    if not rows:
        return pd.DataFrame(
            columns=[
                "pitcher_id",
                "pitcher",
                "team",
                "opp",
                "hand",
                "line",
                "side",
                "p_model",
                "p_market",
                "american",
                "edge",
                "kelly_units",
                "conf",
                "cleared",
                "notes",
                "official_date",
                "game_pk",
                "pitcher_mlb_id",
                "model_run_id",
                "model_version",
                "commence_time",
                "features_as_of",
                "sportsbook",
                "sportsbook_key",
                "source_event_id",
                "market_last_update",
                "line_captured_at",
                "provenance",
                "line_row_count",
                "hidden_line_row_count",
                "sportsbook_count",
                "line_group_summary",
                "approval_status",
                "approval_reason",
                "research_readiness_status",
                "probability_distribution",
                "feature_group_contributions",
            ]
        )

    normalized_rows: list[dict[str, Any]] = []
    readiness_status = stack_readiness_status(output_root)
    for row in rows:
        line_snapshot_id = str(row.get("line_snapshot_id") or "")
        line_snapshot = line_lookup.get(line_snapshot_id) or {}
        game_pk = row.get("game_pk")
        pitcher_mlb_id = row.get("pitcher_mlb_id")
        feature_row = (
            feature_lookup.get((str(row.get("official_date")), int(game_pk), int(pitcher_mlb_id)))
            if game_pk is not None and pitcher_mlb_id is not None
            else None
        )
        over_odds = row.get("over_odds")
        under_odds = row.get("under_odds")
        selected_side = str(row.get("selected_side") or "over")
        if source == "edge_candidates" and row.get("market_over_probability") is not None:
            market_over, market_under = (
                float(row.get("market_over_probability") or 0.5),
                float(row.get("market_under_probability") or 0.5),
            )
        else:
            market_over, market_under = (
                devig([int(over_odds), int(under_odds)], settings.devig_method)
                if over_odds is not None and under_odds is not None
                else (
                    float(row.get("market_over_probability") or 0.5),
                    float(row.get("market_under_probability") or 0.5),
                )
            )
        if selected_side == "under":
            market_probability = market_under
        else:
            market_probability = market_over
        model_probability = (
            _side_probability(row, side=selected_side, field_prefix="model")
            or float(row.get("selected_model_probability") or 0.0)
        )
        selected_odds = row.get("selected_odds")
        if selected_odds is None:
            if selected_side == "under":
                selected_odds = under_odds
            else:
                selected_odds = over_odds
        edge_probability = (
            float(row["edge_pct"])
            if source == "edge_candidates" and row.get("edge_pct") is not None
            else (
                round(model_probability - market_probability, 6)
                if model_probability is not None
                else None
            )
        )
        expected_value_pct = float(row.get("expected_value_pct") or 0.0)
        kelly_fraction = (
            float(row.get("stake_fraction") or 0.0)
            if source == "edge_candidates"
            else (
                fractional_kelly(
                    model_probability,
                    int(selected_odds),
                    fraction=settings.kelly_fraction,
                )
                if model_probability not in {None, 0.0}
                and selected_odds is not None
                else 0.0
            )
        )
        kelly_units = min(
            round(kelly_fraction * settings.bankroll_units, 4),
            settings.max_stake_units,
        )
        raw_hold = _raw_hold(
            int(over_odds) if over_odds is not None else None,
            int(under_odds) if under_odds is not None else None,
        )
        notes = [_short_reason(row.get("reason"))]
        approval_reason = row.get("approval_reason")
        if source == "edge_candidates" and approval_reason:
            notes.append(str(approval_reason).replace("_", " "))
        if source == "edge_candidates":
            notes.append(str(row.get("research_readiness_status") or readiness_status))
        lineup_status = feature_row.get("lineup_status") if feature_row else None
        if lineup_status:
            notes.append(str(lineup_status).replace("_", " "))
        if source == "edge_candidates" and row.get("evaluation_status") != "actionable":
            notes.append("not yet promoted to daily candidates")
        if source == "walk_forward_backtest":
            settlement_status = row.get("settlement_status")
            if settlement_status:
                notes.append(str(settlement_status).replace("_", " "))
            clv_outcome = row.get("clv_outcome")
            if clv_outcome:
                notes.append(str(clv_outcome).replace("_", " "))

        captured_at_value = row.get("captured_at") or row.get("decision_snapshot_captured_at")
        commence_time_value = line_snapshot.get("commence_time") or row.get("commence_time")
        sportsbook = str(
            line_snapshot.get("sportsbook_title")
            or row.get("sportsbook_title")
            or line_snapshot.get("sportsbook")
            or row.get("sportsbook")
            or "n/a"
        )
        sportsbook_key = str(
            line_snapshot.get("sportsbook_key")
            or row.get("sportsbook_key")
            or _sportsbook_key_from_snapshot_id(line_snapshot_id)
        )
        source_event_id = str(
            line_snapshot.get("event_id")
            or row.get("event_id")
            or row.get("source_event_id")
            or ""
        )
        market_last_update = (
            parse_datetime(str(line_snapshot.get("market_last_update")))
            if line_snapshot.get("market_last_update")
            else None
        )
        line_captured_at = (
            parse_datetime(str(line_snapshot.get("captured_at") or captured_at_value))
            if line_snapshot.get("captured_at") or captured_at_value
            else None
        )
        provenance_parts = [
            sportsbook,
            f"event {_short_identifier(source_event_id)}",
            f"snapshot {line_snapshot_display_id(line_snapshot_id)}",
        ]
        if market_last_update is not None:
            provenance_parts.append(f"updated {format_timestamp(market_last_update)}")

        normalized_rows.append(
            {
                "source": source,
                "pitcher_id": str(row.get("player_id") or row.get("pitcher_mlb_id") or ""),
                "pitcher_mlb_id": row.get("pitcher_mlb_id"),
                "pitcher": str(row.get("player_name") or "Unknown Pitcher"),
                "team": (
                    str(feature_row.get("team_abbreviation"))
                    if feature_row and feature_row.get("team_abbreviation")
                    else "n/a"
                ),
                "opp": (
                    str(feature_row.get("opponent_team_abbreviation"))
                    if feature_row and feature_row.get("opponent_team_abbreviation")
                    else "n/a"
                ),
                "hand": (
                    str(feature_row.get("pitcher_hand"))
                    if feature_row and feature_row.get("pitcher_hand")
                    else "?"
                ),
                "line": float(row.get("line") or 0.0),
                "side": selected_side,
                "p_model": model_probability,
                "p_market": market_probability,
                "model_over_probability": row.get("model_over_probability"),
                "model_under_probability": row.get("model_under_probability"),
                "market_over_probability": row.get("market_over_probability"),
                "market_under_probability": row.get("market_under_probability"),
                "american": int(selected_odds) if selected_odds is not None else None,
                "edge": edge_probability,
                "kelly_units": kelly_units,
                "expected_units": round(kelly_units * expected_value_pct, 6),
                "conf": model_probability,
                "note": " · ".join(notes),
                "notes": notes,
                "raw_hold": raw_hold,
                "evaluation_status": str(row.get("evaluation_status") or ""),
                "captured_at": parse_datetime(str(captured_at_value)) if captured_at_value else None,
                "features_as_of": parse_datetime(str(row.get("features_as_of"))) if row.get("features_as_of") else None,
                "commence_time": parse_datetime(str(commence_time_value)) if commence_time_value else None,
                "game_pk": row.get("game_pk"),
                "official_date": str(row.get("official_date") or ""),
                "model_run_id": str(row.get("model_run_id") or ""),
                "model_version": str(row.get("model_version") or ""),
                "line_snapshot_id": line_snapshot_id,
                "sportsbook": sportsbook,
                "sportsbook_key": sportsbook_key,
                "source_event_id": source_event_id,
                "market_last_update": market_last_update,
                "line_captured_at": line_captured_at,
                "provenance": " · ".join(part for part in provenance_parts if part),
                "line_row_count": 1,
                "hidden_line_row_count": 0,
                "sportsbook_count": 1 if sportsbook != "n/a" else 0,
                "line_group_summary": _line_label(
                    {
                        "sportsbook": sportsbook,
                        "side": selected_side,
                        "line": row.get("line"),
                    }
                ),
                "lineup_status": str(lineup_status or ""),
                "pitcher_status": (
                    "confirmed"
                    if lineup_status == "confirmed"
                    else "probable"
                    if row.get("pitcher_mlb_id") is not None
                    else "unknown"
                ),
                "approval_status": str(row.get("approval_status") or ""),
                "approval_allowed": bool(row.get("approval_allowed"))
                if "approval_allowed" in row
                else None,
                "approval_reason": str(approval_reason or ""),
                "validation_recommendation": str(row.get("validation_recommendation") or ""),
                "validation_threshold_status": str(row.get("validation_threshold_status") or ""),
                "validation_min_edge_pct": row.get("validation_min_edge_pct"),
                "validation_min_confidence": row.get("validation_min_confidence"),
                "model_projection": row.get("model_projection") or row.get("model_mean"),
                "model_confidence_bucket": str(row.get("model_confidence_bucket") or ""),
                "probability_distribution": row.get("probability_distribution")
                or row.get("count_distribution")
                or [],
                "feature_group_contributions": row.get("feature_group_contributions") or [],
                "correlation_group_key": str(row.get("correlation_group_key") or ""),
                "correlation_group_size": int(row.get("correlation_group_size") or 1),
                "correlation_group_rank": int(row.get("correlation_group_rank") or 1),
                "research_readiness_status": str(
                    row.get("research_readiness_status") or readiness_status
                ),
            }
        )

    if source == "edge_candidates":
        for row in normalized_rows:
            approved = (
                row["approval_status"] == "approved"
                and row["approval_allowed"] is True
            )
            row["wager_approved"] = approved
            row["wager_gate_status"] = "approved" if approved else "blocked"
            row["wager_blocked_reason"] = (
                "approved" if approved else row["approval_reason"] or "rejected by rebuilt wager gates"
            )
            row["wager_gate_notes"] = [] if approved else [row["wager_blocked_reason"]]
            row["wager_gate_details"] = {
                "model_validation": {
                    "recommendation": row["validation_recommendation"],
                    "threshold_status": row["validation_threshold_status"],
                    "passed": approved,
                },
                "correlation_group": {
                    "group": row["correlation_group_key"],
                    "rank": row["correlation_group_rank"],
                    "size": row["correlation_group_size"],
                    "passed": row["correlation_group_rank"] <= 1,
                },
                "research_readiness": {
                    "status": row["research_readiness_status"],
                    "live_discussion_eligible": (
                        row["research_readiness_status"]
                        != "research_only"
                    ),
                },
            }
            row["cleared"] = approved
            row["cleared_edge_gate"] = bool(row.get("edge") is not None and row["edge"] >= 0.0)
            row["cleared_conf_gate"] = row["approval_status"] == "approved"
            row["cleared_vig_gate"] = row["approval_status"] == "approved"
            row["cleared_stake_gate"] = row["approval_status"] == "approved"
            row["cleared_status_gate"] = row["pitcher_status"] in {"probable", "confirmed"}
            row["cleared_model_age_gate"] = True
            row["cleared_correlation_gate"] = row["correlation_group_rank"] <= 1
            row["cleared_exposure_gate"] = approved
            row["model_age_days"] = None
            row["daily_exposure_before_units"] = 0.0
            row["daily_exposure_after_units"] = row["kelly_units"] if approved else 0.0
        dataframe = pd.DataFrame(normalized_rows)
        if dataframe.empty:
            return dataframe
        return dataframe.sort_values(
            ["cleared", "edge", "pitcher"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    approved_rows = annotate_wager_approval_rows(
        normalized_rows,
        settings=settings,
    )
    for row in approved_rows:
        row_notes = list(row["notes"])
        for gate_note in row["wager_gate_notes"]:
            if gate_note not in row_notes:
                row_notes.append(gate_note)
        row["notes"] = row_notes
        row["note"] = " · ".join(row_notes)
        row["cleared"] = bool(row["wager_approved"])

    dataframe = pd.DataFrame(approved_rows)
    if dataframe.empty:
        return dataframe
    dataframe = dataframe.sort_values(
        ["cleared", "edge", "pitcher"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return dataframe


def group_board_by_pitcher(board: pd.DataFrame) -> pd.DataFrame:
    """Collapse visible line rows to the current best row for each pitcher."""
    if board.empty or "pitcher_id" not in board.columns:
        return board.copy()

    grouped_rows: list[pd.Series] = []
    for _, group in board.groupby("pitcher_id", sort=False):
        selected = group.iloc[0].copy()
        line_labels = [_line_label(row) for _, row in group.iterrows()]
        sportsbook_values = [
            str(value)
            for value in group.get("sportsbook", pd.Series(dtype=object)).dropna().tolist()
            if str(value) and str(value) != "n/a"
        ]
        line_row_count = int(len(group))
        hidden_line_row_count = max(0, line_row_count - 1)
        selected["line_row_count"] = line_row_count
        selected["hidden_line_row_count"] = hidden_line_row_count
        selected["sportsbook_count"] = len(set(sportsbook_values))
        selected["line_group_summary"] = _compact_join(line_labels)
        if hidden_line_row_count:
            notes = list(selected.get("notes") or [])
            grouping_note = (
                f"grouped view hides {hidden_line_row_count} book/line rows: "
                f"{selected['line_group_summary']}"
            )
            if grouping_note not in notes:
                notes.append(grouping_note)
            selected["notes"] = notes
            selected["note"] = " · ".join(str(note) for note in notes)
        grouped_rows.append(selected)

    return pd.DataFrame(grouped_rows).reset_index(drop=True)


def load_board_dataframe(
    output_root: Path,
    *,
    target_date: date,
    settings: DashboardSettings,
) -> tuple[pd.DataFrame, str | None]:
    """Load the current board data frame and its source."""
    rows, source = _ensure_board_source_rows(output_root, target_date=target_date)
    return _normalize_board_dataframe(
        rows,
        source=source or "none",
        output_root=output_root,
        settings=settings,
    ), source


def current_slate_metrics(board: pd.DataFrame) -> dict[str, Any]:
    """Summarize the board KPI strip."""
    actionable = board[board["cleared"]] if not board.empty else board
    return {
        "plays_cleared": int(len(actionable)),
        "total_stake_units": float(actionable["kelly_units"].sum()) if not actionable.empty else 0.0,
        "expected_units": float(actionable["expected_units"].sum()) if not actionable.empty else 0.0,
        "avg_edge": float(actionable["edge"].mean()) if not actionable.empty else None,
        "model_name": (
            str(actionable.iloc[0]["model_version"])
            if not actionable.empty and actionable.iloc[0]["model_version"]
            else (str(board.iloc[0]["model_version"]) if not board.empty else "n/a")
        ),
    }


def ticker_context(board: pd.DataFrame, *, settings: DashboardSettings) -> dict[str, str]:
    """Return the shell ticker values."""
    if board.empty:
        now = datetime.now().astimezone()
        return {
            "slate_date": "NO SLATE",
            "game_count": "0 GAMES",
            "first_pitch": "FIRST PITCH n/a",
            "live_label": "NO LIVE ODDS",
            "model": settings.active_model_label or settings.active_run_id or "MODEL n/a",
            "bankroll": f"BANK {settings.bankroll_units:.1f}u",
            "clock": now.strftime("%H:%M:%S %Z"),
        }
    official_date = str(board.iloc[0]["official_date"])
    game_count = int(board["game_pk"].dropna().nunique()) if "game_pk" in board else 0
    commence_values = [value for value in board["commence_time"] if pd.notna(value)]
    first_pitch = min(commence_values) if commence_values else None
    if first_pitch is not None and hasattr(first_pitch, "to_pydatetime"):
        first_pitch = first_pitch.to_pydatetime()
    source_label = str(board.iloc[0]["source"]) if "source" in board.columns else ""
    readiness_status = (
        str(board.iloc[0]["research_readiness_status"])
        if "research_readiness_status" in board.columns
        and board.iloc[0]["research_readiness_status"]
        else ""
    )
    if source_label == "walk_forward_backtest":
        live_label = "HIST REPLAY"
    elif readiness_status == "research_only":
        live_label = "RESEARCH ONLY"
    else:
        live_label = "LIVE ODDS"
    return {
        "slate_date": official_date,
        "game_count": f"{game_count} GAMES" if game_count else "GAMES n/a",
        "first_pitch": (
            f"FIRST PITCH {first_pitch.astimezone().strftime('%H:%M %Z')}"
            if first_pitch is not None
            else "FIRST PITCH n/a"
        ),
        "live_label": live_label,
        "model": (
            f"MODEL {board.iloc[0]['model_version']}"
            if board.iloc[0]["model_version"]
            else "MODEL n/a"
        ),
        "bankroll": f"BANK {settings.bankroll_units:.1f}u",
        "clock": datetime.now().astimezone().strftime("%H:%M:%S %Z"),
    }


def load_paper_results_dataframe(output_root: Path, *, target_date: date | None = None) -> pd.DataFrame:
    """Load the latest paper results."""
    rows = load_latest_paper_results(output_dir=output_root, target_date=target_date)
    return pd.DataFrame(rows)


def paper_summary_dataframe(paper_results: pd.DataFrame) -> pd.DataFrame:
    """Return grouped recent paper results."""
    if paper_results.empty:
        return pd.DataFrame()
    rows = summarize_paper_results_by_date(paper_results.to_dict("records"), max_dates=7)
    return pd.DataFrame(rows)


def paper_performance_metrics(paper_results: pd.DataFrame) -> dict[str, Any]:
    """Summarize paper results for the KPI strip."""
    if paper_results.empty:
        return {
            "settled_bets": 0,
            "pending_bets": 0,
            "total_profit_units": 0.0,
            "roi": None,
        }
    settled = paper_results[
        paper_results["settlement_status"].isin(["win", "loss", "push"])
    ]
    total_stake = float(settled["stake_fraction"].sum()) if not settled.empty else 0.0
    total_profit = float(settled["profit_units"].sum()) if not settled.empty else 0.0
    return {
        "settled_bets": int(len(settled)),
        "pending_bets": int(len(paper_results) - len(settled)),
        "total_profit_units": total_profit,
        "roi": (total_profit / total_stake) if total_stake > 0.0 else None,
    }


def latest_pitcher_row(board: pd.DataFrame, *, pitcher_id: str | None) -> pd.Series | None:
    """Resolve the selected board row."""
    if board.empty:
        return None
    if pitcher_id:
        match = board[board["pitcher_id"] == pitcher_id]
        if not match.empty:
            return match.iloc[0]
    cleared = board[board["cleared"]]
    if not cleared.empty:
        return cleared.iloc[0]
    return board.iloc[0]


def _load_ladder_rows(output_root: Path, run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "ladder_probabilities.jsonl"
    if not path.exists():
        return []
    return _load_jsonl(str(path))


def _load_distribution_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "model_outputs.jsonl"
    if not path.exists():
        return []
    return _load_jsonl(str(path))


def _find_pitcher_distribution_row(
    output_root: Path,
    *,
    official_date: str,
    pitcher_mlb_id: int | None,
    model_run_id: str | None = None,
) -> dict[str, Any] | None:
    if pitcher_mlb_id is None:
        return None
    if model_run_id is not None:
        resolved_run_dir = _distribution_model_run_lookup(output_root).get(model_run_id)
        run_dirs = [resolved_run_dir] if resolved_run_dir is not None else []
    else:
        run_dirs = list(reversed(_latest_distribution_model_runs(output_root)))
    for run_dir in run_dirs:
        for row in _load_distribution_rows(run_dir):
            if (
                str(row.get("official_date")) == official_date
                and int(row.get("pitcher_id") or -1) == int(pitcher_mlb_id)
            ):
                return row
    return None


def _find_pitcher_ladder_row(
    output_root: Path,
    *,
    official_date: str,
    pitcher_mlb_id: int | None,
    model_run_id: str | None = None,
) -> tuple[dict[str, Any] | None, Path | None]:
    if pitcher_mlb_id is None:
        return None, None
    run_dirs: list[Path]
    if model_run_id is not None:
        resolved_run_dir = _ladder_run_lookup(output_root).get(model_run_id)
        run_dirs = [resolved_run_dir] if resolved_run_dir is not None else []
    else:
        run_dirs = list(reversed(_latest_ladder_run_dirs(output_root)))
    for run_dir in run_dirs:
        for row in _load_ladder_rows(output_root, run_dir):
            if (
                str(row.get("official_date")) == official_date
                and int(row.get("pitcher_id") or -1) == int(pitcher_mlb_id)
            ):
                return row, run_dir
    return None, None


def _poisson_pmf(mean: float, k: int) -> float:
    return exp(-mean + (k * (0.0 if mean == 0 else log(mean))) - lgamma(k + 1))


def _negative_binomial_pmf(mean: float, alpha: float, k: int) -> float:
    if alpha <= 0.0:
        return _poisson_pmf(mean, k)
    r = 1.0 / alpha
    p = r / (r + mean)
    return exp(
        lgamma(k + r)
        - lgamma(r)
        - lgamma(k + 1)
        + (r * log(p))
        + (k * log(1.0 - p))
    )


def get_pmf(
    output_root: Path,
    *,
    official_date: str,
    pitcher_mlb_id: int | None,
    line: float,
    model_run_id: str | None = None,
) -> tuple[list[dict[str, float | str]], dict[str, Any] | None]:
    """Return discrete PMF rows for the selected pitcher/date."""
    distribution_row = _find_pitcher_distribution_row(
        output_root,
        official_date=official_date,
        pitcher_mlb_id=pitcher_mlb_id,
        model_run_id=model_run_id,
    )
    if distribution_row is not None:
        raw_rows = list(distribution_row.get("probability_distribution") or [])
        pmf_rows = [
            {
                "k": int(item["strikeouts"]),
                "label": str(item["strikeouts"]),
                "p": float(item["probability"]),
                "color": "#2ecc71" if int(item["strikeouts"]) > line else "#6b7280",
            }
            for item in raw_rows
            if item.get("strikeouts") is not None and item.get("probability") is not None
        ]
        total = sum(float(row["p"]) for row in pmf_rows)
        if total > 0:
            for row in pmf_rows:
                row["p"] = float(row["p"]) / total
        return pmf_rows, {
            **distribution_row,
            "model_mean": distribution_row.get("point_projection"),
        }

    ladder_row, _ = _find_pitcher_ladder_row(
        output_root,
        official_date=official_date,
        pitcher_mlb_id=pitcher_mlb_id,
        model_run_id=model_run_id,
    )
    if ladder_row is None:
        return [], None
    mean = float(ladder_row.get("model_mean") or 0.0)
    alpha = float((ladder_row.get("count_distribution") or {}).get("dispersion_alpha") or 0.0)
    max_k = max(int(line + 8), int(mean + 10), 12)
    pmf_rows: list[dict[str, float | str]] = []
    for strikeouts in range(0, max_k + 1):
        probability = _negative_binomial_pmf(mean, alpha, strikeouts)
        pmf_rows.append(
            {
                "k": strikeouts,
                "label": str(strikeouts),
                "p": probability,
                "color": "#2ecc71" if strikeouts > line else "#6b7280",
            }
        )
    total = sum(float(row["p"]) for row in pmf_rows)
    if total > 0:
        for row in pmf_rows:
            row["p"] = float(row["p"]) / total
    return pmf_rows, ladder_row


def get_recent_form(
    output_root: Path,
    *,
    pitcher_mlb_id: int | None,
    n: int = 5,
) -> pd.DataFrame:
    """Return the latest historical starts for one pitcher."""
    if pitcher_mlb_id is None:
        return pd.DataFrame(
            columns=["date", "opp", "ip", "k", "bb", "er", "pit", "line", "res"]
        )
    feature_lookup = _lookup_feature_metadata(output_root)
    outcome_rows: list[dict[str, Any]] = []
    for run_dir in _latest_run_dirs(
        output_root / "normalized" / "starter_strikeout_baseline",
        "starter_outcomes.jsonl",
    ):
        outcome_rows.extend(_load_jsonl(str(run_dir / "starter_outcomes.jsonl")))
    rows: list[dict[str, Any]] = []
    for outcome_row in outcome_rows:
        if int(outcome_row.get("pitcher_id") or -1) != int(pitcher_mlb_id):
            continue
        key = (
            str(outcome_row.get("official_date")),
            int(outcome_row.get("game_pk") or -1),
            int(pitcher_mlb_id),
        )
        feature_row = feature_lookup.get(key, {})
        rows.append(
            {
                "date": str(outcome_row.get("official_date") or ""),
                "opp": str(feature_row.get("opponent_team_abbreviation") or "n/a"),
                "ip": None,
                "k": int(outcome_row.get("starter_strikeouts") or 0),
                "bb": None,
                "er": None,
                "pit": int(outcome_row.get("pitch_row_count") or 0),
                "line": None,
                "res": "",
            }
        )
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return dataframe
    return dataframe.sort_values("date", ascending=False).head(n).reset_index(drop=True)


def _latest_backtest_artifact(output_root: Path, name: str) -> list[dict[str, Any]]:
    run_dir = _latest_backtest_run_dir(output_root)
    if run_dir is None:
        return []
    path = run_dir / name
    if not path.exists():
        return []
    return _load_jsonl(str(path))


def backtest_kpis(
    output_root: Path,
    *,
    model_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the current backtest KPI values."""
    summary = load_backtest_summary(output_root)
    roi_summary_rows = _latest_backtest_artifact(output_root, "roi_summary.jsonl")
    clv_summary_rows = _latest_backtest_artifact(output_root, "clv_summary.jsonl")
    overall_roi = next(
        (row for row in roi_summary_rows if row.get("summary_scope") == "overall"),
        None,
    )
    overall_clv = next(
        (row for row in clv_summary_rows if row.get("summary_scope") == "overall"),
        None,
    )
    series = backtest_series(output_root)
    cumulative = float(series["cum_pnl_units"].iloc[-1]) if not series.empty else 0.0
    max_drawdown = None
    if not series.empty:
        running_max = series["cum_pnl_units"].cummax()
        drawdown = series["cum_pnl_units"] - running_max
        max_drawdown = float(drawdown.min())
    held_out = (model_summary or {}).get("held_out_probability_calibration") or {}
    calibrated = held_out.get("calibrated") or {}
    return {
        "bets": int((summary or {}).get("bet_outcomes", {}).get("placed_bets") or 0),
        "cum_pnl": cumulative,
        "roi": (overall_roi or {}).get("roi"),
        "clv": (overall_clv or {}).get("mean_probability_delta"),
        "brier": calibrated.get("mean_brier_score"),
        "log_loss": calibrated.get("mean_log_loss"),
        "max_drawdown": max_drawdown,
    }


def backtest_series(output_root: Path) -> pd.DataFrame:
    """Return date-level backtest rows suitable for charts."""
    roi_rows = [
        row
        for row in _latest_backtest_artifact(output_root, "roi_summary.jsonl")
        if row.get("summary_scope") == "date"
    ]
    clv_rows = {
        str(row["official_date"]): row
        for row in _latest_backtest_artifact(output_root, "clv_summary.jsonl")
        if row.get("summary_scope") == "date"
    }
    if not roi_rows:
        return pd.DataFrame(columns=["date", "daily_pnl", "cum_pnl_units", "clv_units", "roi"])
    rows: list[dict[str, Any]] = []
    cumulative_pnl = 0.0
    cumulative_clv = 0.0
    for roi_row in sorted(roi_rows, key=lambda row: str(row["official_date"])):
        official_date = str(roi_row["official_date"])
        daily_pnl = float(roi_row.get("total_profit_units") or 0.0)
        cumulative_pnl += daily_pnl
        clv_row = clv_rows.get(official_date)
        clv_delta = 0.0
        if clv_row is not None and clv_row.get("mean_probability_delta") is not None:
            clv_delta = float(clv_row["mean_probability_delta"]) * float(clv_row.get("sample_count") or 0.0)
        cumulative_clv += clv_delta
        rows.append(
            {
                "date": official_date,
                "daily_pnl": daily_pnl,
                "cum_pnl_units": cumulative_pnl,
                "clv_units": cumulative_clv,
                "roi": float(roi_row.get("roi") or 0.0),
            }
        )
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return dataframe
    daily_returns = dataframe["daily_pnl"].rolling(30, min_periods=1).sum()
    stake_rows = [
        float(row.get("total_stake_units") or 0.0)
        for row in sorted(roi_rows, key=lambda row: str(row["official_date"]))
    ]
    stake_series = pd.Series(stake_rows).rolling(30, min_periods=1).sum()
    dataframe["rolling_30d_roi"] = [
        (float(daily_returns.iloc[index]) / float(stake_series.iloc[index]))
        if float(stake_series.iloc[index]) > 0.0
        else 0.0
        for index in range(len(dataframe))
    ]
    return dataframe


def get_backtest(
    output_root: Path,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Return the date-level backtest series."""
    dataframe = backtest_series(output_root)
    if dataframe.empty:
        return dataframe
    if start is not None:
        dataframe = dataframe[dataframe["date"] >= start]
    if end is not None:
        dataframe = dataframe[dataframe["date"] <= end]
    return dataframe.reset_index(drop=True)


def backtest_scatter(output_root: Path) -> list[dict[str, Any]]:
    """Return backtest scatter points."""
    rows = _latest_backtest_artifact(output_root, "bet_reporting.jsonl")
    return [
        {
            "market_probability": float(row["scatter_market_probability"]),
            "model_probability": float(row["scatter_model_probability"]),
            "label": str(row.get("player_name") or row.get("backtest_entry_id") or ""),
        }
        for row in rows
        if row.get("scatter_market_probability") is not None
        and row.get("scatter_model_probability") is not None
    ]


def get_calibration(output_root: Path) -> pd.DataFrame:
    """Return calibrated reliability bins from the latest model run."""
    run_dir = latest_model_run_dir(output_root)
    if run_dir is None:
        return pd.DataFrame(columns=["pred_bin", "actual_rate", "n"])
    path = run_dir / "calibration_summary.json"
    if not path.exists():
        return pd.DataFrame(columns=["pred_bin", "actual_rate", "n"])
    summary = _load_json(str(path))
    bins = (
        summary.get("honest_held_out", {})
        .get("held_out", {})
        .get("calibrated", {})
        .get("reliability_bins", [])
    )
    rows = [
        {
            "pred_bin": float(row["mean_predicted_probability"]),
            "actual_rate": float(row["observed_rate"]),
            "n": int(row["sample_count"]),
        }
        for row in bins
    ]
    return pd.DataFrame(rows)


def _latest_inference_run_dir(
    output_root: Path,
    *,
    target_date: date | None = None,
    run_id: str | None = None,
) -> Path | None:
    inference_root = output_root / "normalized" / "starter_strikeout_inference"
    if not inference_root.exists():
        return None
    if run_id is not None:
        candidates = [
            path
            for path in inference_root.rglob(f"run={run_id}")
            if path.is_dir() and path.joinpath("baseline_model.json").exists()
        ]
        return sorted(candidates)[-1] if candidates else None
    if target_date is None:
        run_dirs = _latest_run_dirs(inference_root, "baseline_model.json")
        return run_dirs[-1] if run_dirs else None
    date_root = inference_root / f"date={target_date.isoformat()}"
    run_dirs = [
        path
        for path in sorted(date_root.glob("run=*"))
        if path.is_dir() and path.joinpath("baseline_model.json").exists()
    ]
    return run_dirs[-1] if run_dirs else None


def _resolve_active_model_run_dir(
    output_root: Path,
    *,
    target_date: date | None = None,
    run_id: str | None = None,
) -> tuple[Path | None, str | None]:
    if run_id is not None:
        inference_run_dir = _latest_inference_run_dir(output_root, run_id=run_id)
        if inference_run_dir is not None:
            return inference_run_dir, "inference"
        baseline_run_dir = _model_run_lookup(output_root).get(run_id)
        if baseline_run_dir is not None:
            return baseline_run_dir, "baseline"

    inference_run_dir = _latest_inference_run_dir(output_root, target_date=target_date)
    if inference_run_dir is not None:
        return inference_run_dir, "inference"

    baseline_run_dir = latest_model_run_dir(output_root, run_id=run_id)
    if baseline_run_dir is not None:
        return baseline_run_dir, "baseline"
    return None, None


def _load_baseline_model_payload(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    path = run_dir / "baseline_model.json"
    if not path.exists():
        return {}
    return _load_json(str(path))


def _source_model_run_dir_for_active(
    output_root: Path,
    *,
    active_run_dir: Path,
    active_run_kind: str | None,
    model_payload: dict[str, Any],
) -> Path | None:
    if active_run_kind == "baseline":
        return active_run_dir

    source_run_id = model_payload.get("source_model_run_id")
    if source_run_id is not None:
        source_run_dir = _model_run_lookup(output_root).get(str(source_run_id))
        if source_run_dir is not None:
            return source_run_dir

    source_model_path = model_payload.get("source_model_path")
    if source_model_path:
        source_run_dir = Path(str(source_model_path)).parent
        if source_run_dir.joinpath("training_dataset.jsonl").exists():
            return source_run_dir
    return None


def _latest_statcast_feature_run_dir(
    output_root: Path,
    *,
    target_date: date,
    artifact_name: str,
) -> Path | None:
    date_root = (
        output_root
        / "normalized"
        / "statcast_search"
        / f"date={target_date.isoformat()}"
    )
    if not date_root.exists():
        return None
    run_dirs = [
        path
        for path in sorted(date_root.glob("run=*"))
        if path.is_dir() and path.joinpath(artifact_name).exists()
    ]
    return run_dirs[-1] if run_dirs else None


def _load_target_feature_rows(
    output_root: Path,
    *,
    target_date: date | None,
    artifact_name: str,
) -> tuple[list[dict[str, Any]], Path | None]:
    if target_date is None:
        return [], None
    run_dir = _latest_statcast_feature_run_dir(
        output_root,
        target_date=target_date,
        artifact_name=artifact_name,
    )
    if run_dir is None:
        return [], None
    return _load_jsonl(str(run_dir / artifact_name)), run_dir


def _has_feature_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def _feature_numeric_values(rows: list[dict[str, Any]], feature_name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(feature_name)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _feature_coverage_stats(
    rows: list[dict[str, Any]],
    *,
    feature_name: str,
    aliases: tuple[str, ...] = (),
) -> dict[str, Any]:
    row_count = len(rows)
    non_null_count = sum(1 for row in rows if _has_feature_value(row.get(feature_name)))
    alias_fields_present = sorted(
        alias
        for alias in aliases
        if any(alias in row for row in rows)
    )
    numeric_values = _feature_numeric_values(rows, feature_name)
    variance_range = (
        max(numeric_values) - min(numeric_values)
        if len(numeric_values) >= 2
        else (0.0 if len(numeric_values) == 1 else None)
    )
    return {
        "feature": feature_name,
        "field_present": any(feature_name in row for row in rows),
        "non_null_count": non_null_count,
        "row_count": row_count,
        "coverage": (non_null_count / row_count) if row_count else None,
        "variance_range": variance_range,
        "alias_fields_present": alias_fields_present,
    }


def _coverage_label(non_null_count: int, row_count: int) -> str:
    if row_count <= 0:
        return "n/a"
    return f"{non_null_count}/{row_count} ({non_null_count / row_count:.0%})"


def _family_missing_source_reason(
    family_key: str,
    *,
    status_values: set[str],
    target_date: date | None,
    output_root: Path,
) -> str | None:
    missing_statuses = sorted(
        status for status in status_values if "missing" in status.lower()
    )
    if family_key == "weather" and target_date is not None:
        counts = collect_artifact_counts_for_date(output_root, target_date=target_date)
        usable_weather = counts.weather_ok_snapshots + counts.weather_roof_closed_snapshots
        if counts.weather_snapshots == 0:
            return "target-date weather source artifact missing"
        if usable_weather == 0:
            return "target-date weather source has 0 usable snapshots"
    if family_key == "umpire" and target_date is not None:
        counts = collect_artifact_counts_for_date(output_root, target_date=target_date)
        if counts.umpire_assignments == 0:
            return "target-date umpire source artifact missing"
        if counts.umpire_ok_snapshots == 0:
            return "target-date umpire source has 0 usable snapshots"
    if missing_statuses:
        return ", ".join(missing_statuses)
    return None


def _status_values_for_fields(
    rows: list[dict[str, Any]],
    status_fields: tuple[str, ...],
) -> set[str]:
    values: set[str] = set()
    for row in rows:
        for field_name in status_fields:
            value = row.get(field_name)
            if value is not None and str(value):
                values.add(str(value))
    return values


def _max_coverage(stats: list[dict[str, Any]]) -> tuple[int, int, float | None]:
    if not stats:
        return 0, 0, None
    row_count = max((int(row["row_count"]) for row in stats), default=0)
    non_null_count = max((int(row["non_null_count"]) for row in stats), default=0)
    return non_null_count, row_count, (non_null_count / row_count) if row_count else None


def _optional_family_diagnostics(
    *,
    family: dict[str, Any],
    encoded_features: set[str],
    selection_rows: list[dict[str, Any]],
    all_training_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    target_date: date | None,
    output_root: Path,
) -> dict[str, Any]:
    features = tuple(str(feature) for feature in family["features"])
    aliases = {
        str(feature_name): tuple(str(alias) for alias in alias_names)
        for feature_name, alias_names in dict(family.get("aliases") or {}).items()
    }
    active_features = sorted(feature for feature in features if feature in encoded_features)
    selection_stats = [
        _feature_coverage_stats(
            selection_rows,
            feature_name=feature,
            aliases=aliases.get(feature, ()),
        )
        for feature in features
    ]
    all_training_stats = [
        _feature_coverage_stats(
            all_training_rows,
            feature_name=feature,
            aliases=aliases.get(feature, ()),
        )
        for feature in features
    ]
    target_stats = [
        _feature_coverage_stats(
            target_rows,
            feature_name=feature,
            aliases=aliases.get(feature, ()),
        )
        for feature in features
    ]
    selection_non_null, selection_row_count, selection_coverage = _max_coverage(selection_stats)
    all_non_null, all_row_count, _ = _max_coverage(all_training_stats)
    target_non_null, target_row_count, target_coverage = _max_coverage(target_stats)
    variance_ranges = [
        float(row["variance_range"])
        for row in selection_stats
        if row["variance_range"] is not None
        and float(row["coverage"] or 0.0) >= OPTIONAL_FEATURE_MIN_COVERAGE
    ]
    max_variance = max(variance_ranges) if variance_ranges else None
    status_fields = tuple(str(field) for field in family.get("status_fields", ()))
    status_values = _status_values_for_fields(target_rows, status_fields)
    status_values.update(_status_values_for_fields(all_training_rows, status_fields))
    missing_source_reason = _family_missing_source_reason(
        str(family["key"]),
        status_values=status_values,
        target_date=target_date,
        output_root=output_root,
    )

    if active_features:
        status = "active"
        reason = "selected in active encoded model schema"
    elif missing_source_reason:
        status = "missing_source"
        reason = missing_source_reason
    elif selection_coverage is None:
        status = "missing_source"
        reason = "source training rows unavailable"
    elif selection_coverage < OPTIONAL_FEATURE_MIN_COVERAGE:
        status = "excluded_below_coverage"
        reason = (
            f"source train coverage {_coverage_label(selection_non_null, selection_row_count)} "
            f"is below {OPTIONAL_FEATURE_MIN_COVERAGE:.0%}"
        )
    elif max_variance is not None and max_variance <= MIN_FEATURE_VARIANCE:
        status = "excluded_low_variance"
        reason = "source train values pass coverage but have no usable variance"
    else:
        status = "excluded_by_selection"
        reason = "not present in active encoded model schema"

    schema_notes: list[str] = []
    for stat in [*target_stats, *all_training_stats]:
        for alias in stat["alias_fields_present"]:
            schema_notes.append(f"{alias} present; expected {stat['feature']}")
    schema_notes = sorted(set(schema_notes))

    return {
        "key": str(family["key"]),
        "label": str(family["label"]),
        "status": status,
        "reason": reason,
        "active_features": active_features,
        "inactive_features": [feature for feature in features if feature not in active_features],
        "source_train_non_null_count": selection_non_null,
        "source_train_row_count": selection_row_count,
        "source_all_non_null_count": all_non_null,
        "source_all_row_count": all_row_count,
        "target_non_null_count": target_non_null,
        "target_row_count": target_row_count,
        "source_train_coverage": selection_coverage,
        "target_coverage": target_coverage,
        "source_train_coverage_label": _coverage_label(selection_non_null, selection_row_count),
        "source_all_coverage_label": _coverage_label(all_non_null, all_row_count),
        "target_coverage_label": _coverage_label(target_non_null, target_row_count),
        "status_values": sorted(status_values),
        "schema_notes": schema_notes,
        "feature_stats": {
            "source_train": selection_stats,
            "source_all": all_training_stats,
            "target": target_stats,
        },
    }


def _training_selection_rows(
    run_dir: Path | None,
    all_training_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if run_dir is None:
        return []
    summary_path = run_dir / "evaluation_summary.json"
    if not summary_path.exists():
        return all_training_rows
    train_dates = set(
        str(value)
        for value in (_load_json(str(summary_path)).get("date_splits") or {}).get("train", [])
    )
    if not train_dates:
        return all_training_rows
    return [
        row
        for row in all_training_rows
        if str(row.get("official_date")) in train_dates
    ]


def get_optional_feature_diagnostics(
    output_root: Path,
    *,
    target_date: date | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Return active-schema and optional-feature coverage diagnostics."""
    active_run_dir, active_run_kind = _resolve_active_model_run_dir(
        output_root,
        target_date=target_date,
        run_id=run_id,
    )
    if active_run_dir is None:
        return {
            "active_model_run_id": None,
            "active_run_kind": None,
            "source_model_run_id": None,
            "target_date": target_date.isoformat() if target_date else None,
            "encoded_feature_count": 0,
            "active_core_features": [],
            "active_optional_features": [],
            "active_other_features": [],
            "family_rows": [],
            "warnings": ["No active model artifact was found."],
        }

    model_payload = _load_baseline_model_payload(active_run_dir)
    encoded_features = [str(feature) for feature in model_payload.get("encoded_feature_names", [])]
    encoded_feature_set = set(encoded_features)
    optional_feature_set = set(OPTIONAL_NUMERIC_FEATURES)
    core_feature_set = set(CORE_NUMERIC_FEATURES)
    active_optional = sorted(feature for feature in encoded_features if feature in optional_feature_set)
    active_core = sorted(feature for feature in encoded_features if feature in core_feature_set)
    active_other = sorted(
        feature
        for feature in encoded_features
        if feature not in optional_feature_set and feature not in core_feature_set
    )

    source_run_dir = _source_model_run_dir_for_active(
        output_root,
        active_run_dir=active_run_dir,
        active_run_kind=active_run_kind,
        model_payload=model_payload,
    )
    source_model_run_id = (
        str(model_payload.get("source_model_run_id"))
        if model_payload.get("source_model_run_id") is not None
        else (_run_id_from_dir(source_run_dir) if source_run_dir is not None else None)
    )
    all_training_rows = (
        _load_jsonl(str(source_run_dir / "training_dataset.jsonl"))
        if source_run_dir is not None
        and source_run_dir.joinpath("training_dataset.jsonl").exists()
        else []
    )
    selection_rows = _training_selection_rows(source_run_dir, all_training_rows)

    target_rows_by_artifact: dict[str, list[dict[str, Any]]] = {}
    target_run_dirs: dict[str, str] = {}
    for family in OPTIONAL_FEATURE_FAMILIES:
        artifact_name = str(family["target_artifact"])
        if artifact_name in target_rows_by_artifact:
            continue
        rows, target_run_dir = _load_target_feature_rows(
            output_root,
            target_date=target_date,
            artifact_name=artifact_name,
        )
        target_rows_by_artifact[artifact_name] = rows
        if target_run_dir is not None:
            target_run_dirs[artifact_name] = str(target_run_dir)

    family_rows = [
        _optional_family_diagnostics(
            family=family,
            encoded_features=encoded_feature_set,
            selection_rows=selection_rows,
            all_training_rows=all_training_rows,
            target_rows=target_rows_by_artifact.get(str(family["target_artifact"]), []),
            target_date=target_date,
            output_root=output_root,
        )
        for family in OPTIONAL_FEATURE_FAMILIES
    ]

    return {
        "active_model_run_id": _run_id_from_dir(active_run_dir),
        "active_run_kind": active_run_kind,
        "source_model_run_id": source_model_run_id,
        "target_date": target_date.isoformat() if target_date else None,
        "model_version": str(model_payload.get("model_version") or ""),
        "active_model_path": str(active_run_dir / "baseline_model.json"),
        "source_model_path": (
            str(source_run_dir / "baseline_model.json")
            if source_run_dir is not None
            and source_run_dir.joinpath("baseline_model.json").exists()
            else str(model_payload.get("source_model_path") or "")
        ),
        "source_training_row_count": len(all_training_rows),
        "source_selection_row_count": len(selection_rows),
        "encoded_feature_count": len(encoded_features),
        "active_core_features": active_core,
        "active_optional_features": active_optional,
        "active_other_features": active_other,
        "optional_feature_count": len(active_optional),
        "family_rows": family_rows,
        "target_feature_run_dirs": target_run_dirs,
        "selection_thresholds": {
            "optional_feature_min_coverage": OPTIONAL_FEATURE_MIN_COVERAGE,
            "min_feature_variance": MIN_FEATURE_VARIANCE,
        },
        "warnings": [],
    }


def _feature_values_for_run(
    output_root: Path,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    run_dir = latest_model_run_dir(output_root, run_id=run_id)
    if run_dir is None:
        return {}
    dataset_path = run_dir / "training_dataset.jsonl"
    if not dataset_path.exists():
        return {}
    dataset = _load_jsonl(str(dataset_path))
    if not dataset:
        return {}
    latest_row = max(dataset, key=lambda row: (str(row.get("official_date")), str(row.get("training_row_id"))))
    return latest_row


def get_feature_importance(
    output_root: Path,
    *,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Return feature-importance rows for the features screen."""
    distribution_run_dir = (
        _distribution_model_run_lookup(output_root).get(run_id)
        if run_id is not None
        else (_latest_distribution_model_runs(output_root)[-1] if _latest_distribution_model_runs(output_root) else None)
    )
    if distribution_run_dir is not None:
        selected_model_path = distribution_run_dir / "selected_model.json"
        if selected_model_path.exists():
            selected_model = _load_json(str(selected_model_path))
            contributions = selected_model.get("feature_group_contributions") or []
            return pd.DataFrame(
                [
                    {
                        "name": str(row.get("feature_group") or row.get("group") or ""),
                        "importance": float(
                            row.get("absolute_contribution")
                            or row.get("share")
                            or 0.0
                        ),
                        "direction": "+",
                        "last_value": (
                            None
                            if row.get("share") is None
                            else f"{float(row['share']):.1%}"
                        ),
                    }
                    for row in contributions
                ]
            )

    summary = load_latest_model_summary(output_root, run_id=run_id)
    run_dir = latest_model_run_dir(output_root, run_id=run_id)
    if summary is None or run_dir is None:
        return pd.DataFrame(columns=["name", "importance", "direction", "last_value"])
    evaluation = _load_json(str(run_dir / "evaluation.json")) if (run_dir / "evaluation.json").exists() else {}
    feature_rows = evaluation.get("feature_importance") or summary.get("top_feature_importance") or []
    latest_values = _feature_values_for_run(output_root, run_id=run_id)
    normalized: list[dict[str, Any]] = []
    for row in feature_rows:
        feature_name = str(row.get("feature") or row.get("name") or "")
        coefficient = float(row.get("coefficient") or row.get("importance") or 0.0)
        importance = float(row.get("absolute_importance") or row.get("importance") or abs(coefficient))
        normalized.append(
            {
                "name": feature_name,
                "importance": importance,
                "direction": "+" if coefficient >= 0.0 else "-",
                "last_value": latest_values.get(feature_name),
            }
        )
    return pd.DataFrame(normalized)


def _psi(train_values: list[float], recent_values: list[float], *, bucket_count: int = 5) -> float | None:
    if len(train_values) < bucket_count or len(recent_values) < bucket_count:
        return None
    quantiles = sorted(pd.Series(train_values).quantile([index / bucket_count for index in range(1, bucket_count)]).tolist())
    boundaries = [-float("inf"), *quantiles, float("inf")]
    psi_value = 0.0
    epsilon = 1e-6
    for index in range(len(boundaries) - 1):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        train_count = sum(1 for value in train_values if lower < value <= upper)
        recent_count = sum(1 for value in recent_values if lower < value <= upper)
        train_share = max(train_count / len(train_values), epsilon)
        recent_share = max(recent_count / len(recent_values), epsilon)
        psi_value += (recent_share - train_share) * log(recent_share / train_share)
    return round(psi_value, 6)


def feature_drift_rows(output_root: Path) -> list[dict[str, Any]]:
    """Return PSI rows for recent features vs training."""
    run_dir = latest_model_run_dir(output_root)
    if run_dir is None:
        return []
    dataset_path = run_dir / "training_dataset.jsonl"
    if not dataset_path.exists():
        return []
    dataset = _load_jsonl(str(dataset_path))
    if not dataset:
        return []
    date_splits = _load_json(str(run_dir / "evaluation_summary.json")).get("date_splits", {})
    train_dates = set(date_splits.get("train", []))
    latest_date = max(str(row["official_date"]) for row in dataset)
    latest_cutoff = date.fromisoformat(latest_date) - timedelta(days=6)
    importance = get_feature_importance(output_root)
    rows: list[dict[str, Any]] = []
    for feature_name in list(importance["name"])[:5]:
        train_values = [
            float(row[feature_name])
            for row in dataset
            if feature_name in row
            and row.get(feature_name) is not None
            and str(row.get("official_date")) in train_dates
            and isinstance(row.get(feature_name), (int, float))
        ]
        recent_values = [
            float(row[feature_name])
            for row in dataset
            if feature_name in row
            and row.get(feature_name) is not None
            and date.fromisoformat(str(row.get("official_date"))) >= latest_cutoff
            and isinstance(row.get(feature_name), (int, float))
        ]
        psi_value = _psi(train_values, recent_values)
        rows.append(
            {
                "name": feature_name,
                "psi": psi_value,
                "status": "warn" if psi_value is not None and psi_value > 0.25 else "ok",
            }
        )
    return rows


def leakage_checks(board: pd.DataFrame, *, settings: DashboardSettings) -> list[dict[str, Any]]:
    """Return leakage check statuses derived from current rows."""
    has_board_rows = not board.empty
    decision_cutoff_checks = []
    for _, row in board.iterrows():
        features_as_of = row.get("features_as_of")
        commence_time = row.get("commence_time")
        if features_as_of is None or commence_time is None:
            continue
        decision_cutoff_checks.append(
            features_as_of <= commence_time - timedelta(minutes=settings.decision_cutoff_minutes)
        )
    return [
        {
            "label": "no post-decision features in pipeline",
            "ok": True,
        },
        {
            "label": f"all timestamps <= first pitch - {settings.decision_cutoff_minutes}m",
            "ok": all(decision_cutoff_checks) if decision_cutoff_checks else has_board_rows,
        },
        {
            "label": "closing-line features gated to eval only",
            "ok": True,
        },
    ]


def registry_rows(
    output_root: Path,
    *,
    settings: DashboardSettings,
    stage_overrides: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return registry rows merged from evaluation artifacts and MLflow metadata."""
    tracking = TrackingConfig()
    model_summaries: list[dict[str, Any]] = []
    for run_dir in _latest_model_runs(output_root):
        summary = _load_json(str(run_dir / "evaluation_summary.json"))
        summary["_run_dir"] = str(run_dir)
        model_summaries.append(summary)
    mlflow_runs = search_runs(
        tracking_uri=tracking.tracking_uri,
        experiment_names=[tracking.training_experiment_name, tracking.backtest_experiment_name],
    )
    mlflow_lookup = {str(row["run_id"]): row for row in mlflow_runs}
    registered_lookup = registered_versions_by_run_id(tracking_uri=tracking.tracking_uri)
    backtest_runs = _latest_backtest_artifact(output_root, "backtest_runs.jsonl")
    backtest_by_model_run = {
        str(row["model_run_id"]): row
        for row in backtest_runs
        if row.get("model_run_id") is not None
    }
    ordered = sorted(
        model_summaries,
        key=lambda row: (
            _parse_run_id(str(row.get("run_id")) or "") or datetime.min.replace(tzinfo=UTC),
            str(row.get("run_id")),
        ),
        reverse=True,
    )
    rows: list[dict[str, Any]] = []
    overrides = stage_overrides or {}
    for index, summary in enumerate(ordered):
        run_id = str(summary.get("run_id") or "")
        mlflow_run_id = str(summary.get("mlflow_run_id") or "")
        backtest_row = backtest_by_model_run.get(run_id, {})
        registry_row = registered_lookup.get(mlflow_run_id) or {}
        stage = overrides.get(run_id)
        if stage is None:
            stage = registry_row.get("stage")
        if stage is None:
            if settings.active_run_id == run_id or index == 0:
                stage = "Production"
            elif index == 1:
                stage = "Staging"
            else:
                stage = "Archived"
        rows.append(
            {
                "run_id": run_id,
                "mlflow_run_id": mlflow_run_id,
                "name": str(summary.get("model_version") or run_id),
                "stage": stage,
                "brier": (summary.get("held_out_probability_calibration", {}).get("calibrated", {}).get("mean_brier_score")),
                "logloss": (summary.get("held_out_probability_calibration", {}).get("calibrated", {}).get("mean_log_loss")),
                "clv": (backtest_row.get("clv_summary", {}) or {}).get("mean_probability_delta"),
                "roi": (backtest_row.get("roi_summary", {}) or {}).get("roi"),
                "n": int(summary.get("row_counts", {}).get("held_out") or 0),
                "created": _parse_run_id(run_id),
                "model_name": registry_row.get("model_name"),
                "version": registry_row.get("version"),
                "tracking_row": mlflow_lookup.get(mlflow_run_id),
            }
        )
    return rows


def backtest_date_range(output_root: Path) -> tuple[str | None, str | None]:
    """Return the latest backtest date window."""
    summary = load_backtest_summary(output_root)
    if summary is None:
        return None, None
    return str(summary.get("start_date")), str(summary.get("end_date"))
