"""Shared final wager approval gates for daily sheets and dashboard rows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .pricing import american_to_implied_probability, fractional_kelly


@dataclass(frozen=True)
class WagerApprovalSettings:
    """Runtime controls for final wager eligibility."""

    edge_min: float = 0.03
    confidence_min: float = 0.55
    max_hold: float = 0.055
    kelly_fraction: float = 0.25
    max_stake_units: float = 2.0
    max_daily_exposure_units: float = 15.0
    bankroll_units: float = 200.0
    retrain_window_days: int = 120


GATE_ORDER: tuple[tuple[str, str], ...] = (
    ("cleared_edge_gate", "below edge threshold"),
    ("cleared_conf_gate", "below confidence floor"),
    ("cleared_vig_gate", "hold above max"),
    ("cleared_stake_gate", "stake above cap"),
    ("cleared_status_gate", "pitcher status unresolved"),
    ("cleared_model_age_gate", "model outside retrain window"),
    ("cleared_correlation_gate", "correlated same-slate play"),
    ("cleared_exposure_gate", "daily exposure exhausted"),
)


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _coerce_float(row.get(key))
        if value is not None:
            return value
    return None


def _first_int(row: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _coerce_int(row.get(key))
        if value is not None:
            return value
    return None


def _parse_run_id(run_id: str | None) -> datetime | None:
    if not run_id:
        return None
    try:
        return datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def model_run_age_days(
    model_run_id: str | None,
    *,
    now: datetime | None = None,
) -> float | None:
    """Return age in days for a timestamp-shaped model run id."""

    trained_at = _parse_run_id(model_run_id)
    if trained_at is None:
        return None
    resolved_now = now.astimezone(UTC) if now is not None else datetime.now(tz=UTC)
    return max(0.0, (resolved_now - trained_at).total_seconds() / 86400.0)


def raw_book_hold(over_odds: int | None, under_odds: int | None) -> float | None:
    """Return raw two-way book hold when both sides are available."""

    if over_odds is None or under_odds is None:
        return None
    return (
        american_to_implied_probability(int(over_odds))
        + american_to_implied_probability(int(under_odds))
        - 1.0
    )


def _row_hold(row: dict[str, Any]) -> float | None:
    existing = _coerce_float(row.get("raw_hold"))
    if existing is not None:
        return existing
    return raw_book_hold(
        _first_int(row, "over_odds"),
        _first_int(row, "under_odds"),
    )


def _row_confidence(row: dict[str, Any]) -> float | None:
    return _first_float(row, "conf", "selected_model_probability")


def _row_edge(row: dict[str, Any]) -> float | None:
    return _first_float(row, "edge", "edge_pct")


def _row_selected_odds(row: dict[str, Any]) -> int | None:
    return _first_int(row, "american", "selected_odds")


def _row_pitcher_status(row: dict[str, Any]) -> str:
    status = row.get("pitcher_status")
    if status:
        return str(status)
    lineup_status = row.get("lineup_status")
    if lineup_status == "confirmed":
        return "confirmed"
    if row.get("pitcher_mlb_id") is not None:
        return "probable"
    return "unknown"


def _row_pitcher_key(row: dict[str, Any]) -> str:
    return str(row.get("pitcher_id") or row.get("player_id") or row.get("pitcher_mlb_id") or "")


def _row_stake_units(
    row: dict[str, Any],
    *,
    settings: WagerApprovalSettings,
) -> float:
    probability = _row_confidence(row)
    selected_odds = _row_selected_odds(row)
    if probability is None or selected_odds is None:
        return 0.0
    stake_fraction = fractional_kelly(
        probability,
        selected_odds,
        fraction=settings.kelly_fraction,
    )
    return min(round(stake_fraction * settings.bankroll_units, 4), settings.max_stake_units)


def _gate_details_for_row(
    row: dict[str, Any],
    *,
    settings: WagerApprovalSettings,
) -> dict[str, dict[str, Any]]:
    return {
        "edge": {
            "value": row["edge"],
            "threshold": settings.edge_min,
            "passed": bool(row["cleared_edge_gate"]),
        },
        "hold": {
            "value": row["raw_hold"],
            "threshold": settings.max_hold,
            "passed": bool(row["cleared_vig_gate"]),
        },
        "confidence": {
            "value": row["conf"],
            "threshold": settings.confidence_min,
            "passed": bool(row["cleared_conf_gate"]),
        },
        "stake": {
            "value_units": row["kelly_units"],
            "threshold_units": settings.max_stake_units,
            "passed": bool(row["cleared_stake_gate"]),
        },
        "status": {
            "value": row["pitcher_status"],
            "allowed": ["probable", "confirmed"],
            "passed": bool(row["cleared_status_gate"]),
        },
        "model_age": {
            "value_days": row["model_age_days"],
            "threshold_days": settings.retrain_window_days,
            "passed": bool(row["cleared_model_age_gate"]),
        },
        "correlation": {
            "passed": bool(row["cleared_correlation_gate"]),
            "group": row["wager_correlation_key"],
        },
        "exposure": {
            "stake_units": row["kelly_units"],
            "daily_total_before_units": row["daily_exposure_before_units"],
            "daily_total_after_units": row["daily_exposure_after_units"],
            "threshold_units": settings.max_daily_exposure_units,
            "passed": bool(row["cleared_exposure_gate"]),
        },
    }


def _approval_sort_key(row: dict[str, Any]) -> tuple[float, float, str]:
    row_id = (
        row.get("daily_candidate_id")
        or row.get("candidate_id")
        or row.get("line_snapshot_id")
        or ""
    )
    return (
        -float(row.get("edge") or 0.0),
        -float(row.get("kelly_units") or 0.0),
        str(row_id),
    )


def annotate_wager_approval_rows(
    rows: list[dict[str, Any]],
    *,
    settings: WagerApprovalSettings,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Annotate rows with the shared final wager gates.

    This intentionally covers final board approval only. Per-pitcher and
    per-game Kelly allocation remains the AGE-209 sizing issue; the exposure
    gate here prevents the daily sheet from diverging from the dashboard board.
    """

    resolved_now = now.astimezone(UTC) if now is not None else datetime.now(tz=UTC)
    annotated: list[dict[str, Any]] = []
    for row in rows:
        annotated_row = dict(row)
        edge = _row_edge(annotated_row)
        raw_hold = _row_hold(annotated_row)
        confidence = _row_confidence(annotated_row)
        model_age = model_run_age_days(
            str(annotated_row.get("model_run_id") or ""),
            now=resolved_now,
        )
        pitcher_status = _row_pitcher_status(annotated_row)
        kelly_units = _row_stake_units(annotated_row, settings=settings)
        official_date = str(annotated_row.get("official_date") or "")
        pitcher_key = _row_pitcher_key(annotated_row)

        annotated_row.update(
            {
                "edge": edge,
                "raw_hold": raw_hold,
                "conf": confidence,
                "kelly_units": kelly_units,
                "pitcher_status": pitcher_status,
                "model_age_days": model_age,
                "wager_correlation_key": f"{official_date}|{pitcher_key}",
                "cleared_edge_gate": (edge is not None and edge >= settings.edge_min),
                "cleared_vig_gate": (
                    raw_hold is None or raw_hold <= settings.max_hold
                ),
                "cleared_stake_gate": kelly_units <= settings.max_stake_units,
                "cleared_conf_gate": (
                    confidence is not None and confidence >= settings.confidence_min
                ),
                "cleared_status_gate": pitcher_status in {"probable", "confirmed"},
                "cleared_model_age_gate": (
                    model_age is None or model_age <= settings.retrain_window_days
                ),
                "cleared_correlation_gate": True,
                "cleared_exposure_gate": True,
                "daily_exposure_before_units": 0.0,
                "daily_exposure_after_units": None,
            }
        )
        annotated.append(annotated_row)

    by_correlation_key: dict[str, list[int]] = {}
    for index, row in sorted(
        enumerate(annotated),
        key=lambda item: (
            str(item[1].get("wager_correlation_key") or ""),
            -float(item[1].get("edge") or 0.0),
            float(item[1].get("line") or 0.0),
            str(item[1].get("line_snapshot_id") or ""),
        ),
    ):
        by_correlation_key.setdefault(str(row["wager_correlation_key"]), []).append(index)
    for indexes in by_correlation_key.values():
        for index in indexes[1:]:
            annotated[index]["cleared_correlation_gate"] = False

    cumulative_units = 0.0
    for row in sorted(annotated, key=_approval_sort_key):
        prechecks = (
            bool(row["cleared_edge_gate"])
            and bool(row["cleared_vig_gate"])
            and bool(row["cleared_stake_gate"])
            and bool(row["cleared_conf_gate"])
            and bool(row["cleared_status_gate"])
            and bool(row["cleared_model_age_gate"])
            and bool(row["cleared_correlation_gate"])
        )
        row["daily_exposure_before_units"] = round(cumulative_units, 4)
        if not prechecks:
            row["daily_exposure_after_units"] = round(cumulative_units, 4)
            continue
        next_total = cumulative_units + float(row["kelly_units"])
        if next_total > settings.max_daily_exposure_units:
            row["cleared_exposure_gate"] = False
            row["daily_exposure_after_units"] = round(cumulative_units, 4)
            continue
        cumulative_units = next_total
        row["daily_exposure_after_units"] = round(cumulative_units, 4)

    for row in annotated:
        failed_reasons = [
            reason for field, reason in GATE_ORDER if not bool(row[field])
        ]
        approved = not failed_reasons
        row["wager_gate_details"] = _gate_details_for_row(row, settings=settings)
        row["wager_gate_notes"] = failed_reasons
        row["wager_gate_status"] = "approved" if approved else "blocked"
        row["wager_blocked_reason"] = "approved" if approved else failed_reasons[0]
        row["wager_approved"] = approved
    return annotated
