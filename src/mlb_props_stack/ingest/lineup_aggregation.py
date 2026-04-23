"""Lineup-level aggregation from normalized Statcast pitches.

Consumes :class:`StatcastPitchRecord` rows plus the game's pregame
:class:`LineupSnapshot` and emits one :class:`LineupDailyFeatureRow` per
probable starter describing the opposing batting order.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from .mlb_stats_api import GameRecord, LineupSnapshot, ProbableStarterRecord
from .statcast_ingest import (
    StatcastPitchRecord,
    _batter_rows,
    _history_cutoff,
    _mean,
    _opponent_team,
    _round_optional,
    _safe_rate,
    _sorted_rows,
)


@dataclass(frozen=True)
class LineupDailyFeatureRow:
    """Pregame opponent-lineup feature row for one probable starter."""

    feature_row_id: str
    official_date: str
    game_pk: int
    pitcher_id: int | None
    pitcher_name: str | None
    team_abbreviation: str
    opponent_team_abbreviation: str
    opponent_team_name: str
    lineup_snapshot_id: str | None
    history_start_date: date
    history_end_date: date
    features_as_of: datetime
    lineup_status: str
    lineup_is_confirmed: bool
    lineup_size: int
    available_batter_feature_count: int
    pitcher_hand: str | None
    projected_lineup_k_rate: float | None
    projected_lineup_k_rate_vs_pitcher_hand: float | None
    lineup_k_rate_vs_rhp: float | None
    lineup_k_rate_vs_lhp: float | None
    projected_lineup_chase_rate: float | None
    projected_lineup_contact_rate: float | None
    lineup_continuity_count: int | None
    lineup_continuity_ratio: float | None
    lineup_player_ids: tuple[int, ...]


@dataclass(frozen=True)
class _BatterMetricBundle:
    k_rate: float | None
    k_rate_vs_pitcher_hand: float | None
    k_rate_vs_rhp: float | None
    k_rate_vs_lhp: float | None
    chase_rate: float | None
    contact_rate: float | None


def _batter_k_rate_vs_p_throws(
    final_pitch_rows: list[StatcastPitchRecord], *, p_throws: str
) -> float | None:
    hand_rows = [row for row in final_pitch_rows if row.p_throws == p_throws]
    return _safe_rate(
        sum(1 for row in hand_rows if row.is_strikeout_event),
        len(hand_rows),
    )


def _batter_metric_bundle(
    *,
    batter_rows: list[StatcastPitchRecord],
    pitcher_hand: str | None,
) -> _BatterMetricBundle:
    final_pitch_rows = [row for row in batter_rows if row.is_plate_appearance_final_pitch]
    k_rate = _safe_rate(
        sum(1 for row in final_pitch_rows if row.is_strikeout_event),
        len(final_pitch_rows),
    )
    k_rate_vs_pitcher_hand: float | None = None
    if pitcher_hand is not None:
        k_rate_vs_pitcher_hand = _batter_k_rate_vs_p_throws(
            final_pitch_rows, p_throws=pitcher_hand
        )
    k_rate_vs_rhp = _batter_k_rate_vs_p_throws(final_pitch_rows, p_throws="R")
    k_rate_vs_lhp = _batter_k_rate_vs_p_throws(final_pitch_rows, p_throws="L")
    out_of_zone_rows = [row for row in batter_rows if row.is_out_of_zone is True]
    swing_rows = [row for row in batter_rows if row.is_swing]
    chase_rate = _safe_rate(
        sum(1 for row in out_of_zone_rows if row.is_chase_swing is True),
        len(out_of_zone_rows),
    )
    contact_rate = _safe_rate(
        sum(1 for row in swing_rows if row.is_contact),
        len(swing_rows),
    )
    return _BatterMetricBundle(
        k_rate=k_rate,
        k_rate_vs_pitcher_hand=k_rate_vs_pitcher_hand,
        k_rate_vs_rhp=k_rate_vs_rhp,
        k_rate_vs_lhp=k_rate_vs_lhp,
        chase_rate=chase_rate,
        contact_rate=contact_rate,
    )


def _batting_order_weight(*, slot_index: int, lineup_size: int) -> float:
    """Linearly decreasing weight that models the PA distribution of a lineup.

    Slot 1 gets the largest weight and slot N the smallest, reflecting the fact
    that the top of the order turns over more plate appearances than the bottom.
    Weights are unnormalized; the caller should divide by the sum of consumed
    weights so missing-history batters do not distort the average.
    """
    return float(lineup_size - slot_index)


def _weighted_mean(values: list[tuple[float, float]]) -> float | None:
    """Weighted mean of ``(value, weight)`` pairs. Returns ``None`` when empty."""
    if not values:
        return None
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return None
    weighted_sum = sum(value * weight for value, weight in values)
    return _round_optional(weighted_sum / total_weight)


def _latest_prior_team_lineup_player_ids(
    *,
    team_abbreviation: str,
    all_rows: list[StatcastPitchRecord],
    target_date: date,
) -> tuple[int, ...]:
    prior_rows = [
        row
        for row in all_rows
        if row.batting_team_abbreviation == team_abbreviation
        and date.fromisoformat(row.game_date) < target_date
    ]
    if not prior_rows:
        return ()
    latest_date = max(row.game_date for row in prior_rows)
    return tuple(
        sorted(
            {
                row.batter_id
                for row in prior_rows
                if row.game_date == latest_date
            }
        )
    )


def _build_lineup_daily_feature_row(
    *,
    starter: ProbableStarterRecord,
    game: GameRecord,
    lineup_snapshot: LineupSnapshot | None,
    all_rows: list[StatcastPitchRecord],
    history_start_date: date,
    history_end_date: date,
    pitcher_hand: str | None,
) -> LineupDailyFeatureRow:
    opponent_team_abbreviation, opponent_team_name = _opponent_team(game, starter.team_side)
    base_features_as_of = max(_history_cutoff(date.fromisoformat(starter.official_date)), starter.captured_at)
    lineup_player_ids: tuple[int, ...] = ()
    lineup_status = "missing_pregame_lineup"
    features_as_of = base_features_as_of
    lineup_is_confirmed = False

    if lineup_snapshot is not None:
        lineup_player_ids = lineup_snapshot.batting_order_player_ids
        lineup_status = "confirmed" if lineup_snapshot.is_confirmed else "projected"
        lineup_is_confirmed = lineup_snapshot.is_confirmed
        features_as_of = max(base_features_as_of, lineup_snapshot.captured_at)

    k_rates: list[float] = []
    k_rates_vs_hand: list[float] = []
    chase_rates: list[float] = []
    contact_rates: list[float] = []
    weighted_k_rates_vs_rhp: list[tuple[float, float]] = []
    weighted_k_rates_vs_lhp: list[tuple[float, float]] = []
    available_batter_feature_count = 0
    lineup_size = len(lineup_player_ids)

    for slot_index, batter_id in enumerate(lineup_player_ids):
        batter_rows = _sorted_rows(_batter_rows(all_rows, batter_id=batter_id))
        if not batter_rows:
            continue
        available_batter_feature_count += 1
        bundle = _batter_metric_bundle(
            batter_rows=batter_rows,
            pitcher_hand=pitcher_hand,
        )
        if bundle.k_rate is not None:
            k_rates.append(bundle.k_rate)
        if bundle.k_rate_vs_pitcher_hand is not None:
            k_rates_vs_hand.append(bundle.k_rate_vs_pitcher_hand)
        if bundle.chase_rate is not None:
            chase_rates.append(bundle.chase_rate)
        if bundle.contact_rate is not None:
            contact_rates.append(bundle.contact_rate)
        slot_weight = _batting_order_weight(
            slot_index=slot_index, lineup_size=lineup_size
        )
        if bundle.k_rate_vs_rhp is not None:
            weighted_k_rates_vs_rhp.append((bundle.k_rate_vs_rhp, slot_weight))
        if bundle.k_rate_vs_lhp is not None:
            weighted_k_rates_vs_lhp.append((bundle.k_rate_vs_lhp, slot_weight))

    prior_lineup_ids = _latest_prior_team_lineup_player_ids(
        team_abbreviation=opponent_team_abbreviation,
        all_rows=all_rows,
        target_date=date.fromisoformat(starter.official_date),
    )
    continuity_count = None
    continuity_ratio = None
    if lineup_player_ids:
        continuity_count = len(set(lineup_player_ids) & set(prior_lineup_ids))
        continuity_ratio = _round_optional(continuity_count / len(lineup_player_ids))

    return LineupDailyFeatureRow(
        feature_row_id=(
            f"lineup-feature:{starter.game_pk}:{starter.pitcher_id or 'missing'}:{starter.official_date}"
        ),
        official_date=starter.official_date,
        game_pk=starter.game_pk,
        pitcher_id=starter.pitcher_id,
        pitcher_name=starter.pitcher_name,
        team_abbreviation=starter.team_abbreviation,
        opponent_team_abbreviation=opponent_team_abbreviation,
        opponent_team_name=opponent_team_name,
        lineup_snapshot_id=lineup_snapshot.lineup_snapshot_id if lineup_snapshot is not None else None,
        history_start_date=history_start_date,
        history_end_date=history_end_date,
        features_as_of=features_as_of,
        lineup_status=lineup_status,
        lineup_is_confirmed=lineup_is_confirmed,
        lineup_size=len(lineup_player_ids),
        available_batter_feature_count=available_batter_feature_count,
        pitcher_hand=pitcher_hand,
        projected_lineup_k_rate=_mean(k_rates),
        projected_lineup_k_rate_vs_pitcher_hand=_mean(k_rates_vs_hand),
        lineup_k_rate_vs_rhp=_weighted_mean(weighted_k_rates_vs_rhp),
        lineup_k_rate_vs_lhp=_weighted_mean(weighted_k_rates_vs_lhp),
        projected_lineup_chase_rate=_mean(chase_rates),
        projected_lineup_contact_rate=_mean(contact_rates),
        lineup_continuity_count=continuity_count,
        lineup_continuity_ratio=continuity_ratio,
        lineup_player_ids=lineup_player_ids,
    )
