"""Pitcher-level rolling feature derivation from normalized Statcast pitches.

Consumes :class:`StatcastPitchRecord` rows produced by ``statcast_ingest`` and
emits one :class:`PitcherDailyFeatureRow` per probable starter. Also owns
``_expected_leash`` (shared with ``game_context``) because leash modeling is a
pitcher-centric statistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from .mlb_stats_api import GameRecord, ProbableStarterRecord
from .statcast_ingest import (
    StatcastPitchRecord,
    _count_plate_appearances,
    _history_cutoff,
    _last_game_date,
    _mean,
    _opponent_team,
    _pitch_rows_for_player,
    _pitch_type_usage,
    _plate_appearance_key,
    _round_optional,
    _rows_grouped_by_start,
    _rows_in_recent_window,
    _safe_rate,
    _sorted_rows,
)


@dataclass(frozen=True)
class PitcherDailyFeatureRow:
    """Pregame pitcher feature row for one probable starter."""

    feature_row_id: str
    official_date: str
    game_pk: int
    pitcher_id: int | None
    pitcher_name: str | None
    team_side: str
    team_abbreviation: str
    opponent_team_abbreviation: str
    history_start_date: date
    history_end_date: date
    features_as_of: datetime
    feature_status: str
    pitch_sample_size: int
    plate_appearance_sample_size: int
    pitcher_hand: str | None
    pitcher_k_rate: float | None
    pitcher_k_rate_vs_rhh: float | None
    pitcher_k_rate_vs_lhh: float | None
    swinging_strike_rate: float | None
    pitcher_whiff_rate_vs_rhh: float | None
    pitcher_whiff_rate_vs_lhh: float | None
    csw_rate: float | None
    pitch_type_usage: dict[str, float]
    average_release_speed: float | None
    release_speed_delta_vs_baseline: float | None
    average_release_extension: float | None
    release_extension_delta_vs_baseline: float | None
    recent_batters_faced: int
    recent_pitch_count: int
    rest_days: int | None
    last_start_pitch_count: int | None
    last_start_batters_faced: int | None


def _pitcher_hand(rows: list[StatcastPitchRecord]) -> str | None:
    for row in reversed(_sorted_rows(rows)):
        if row.p_throws:
            return row.p_throws
    return None


def _pitcher_hand_split_rates(
    *,
    pitcher_rows: list[StatcastPitchRecord],
    stand: str,
) -> tuple[float | None, float | None]:
    """Return (k_rate, whiff_rate) for rows where the batter hits from ``stand``.

    ``k_rate`` is measured over plate-appearance-final pitches and ``whiff_rate``
    is measured over every pitch, matching the unsplit equivalents so the new
    columns slot cleanly next to ``pitcher_k_rate`` and ``swinging_strike_rate``.
    """
    hand_rows = [row for row in pitcher_rows if row.stand == stand]
    final_pitch_rows = [row for row in hand_rows if row.is_plate_appearance_final_pitch]
    k_rate = _safe_rate(
        sum(1 for row in final_pitch_rows if row.is_strikeout_event),
        len(final_pitch_rows),
    )
    whiff_rate = _safe_rate(
        sum(1 for row in hand_rows if row.is_whiff),
        len(hand_rows),
    )
    return k_rate, whiff_rate


def _expected_leash(
    rows: list[StatcastPitchRecord],
) -> tuple[float | None, float | None]:
    grouped_starts = _rows_grouped_by_start(rows)
    if not grouped_starts:
        return None, None
    recent_starts = grouped_starts[:3]
    pitch_counts = [float(len(start_rows)) for start_rows in recent_starts]
    batter_counts = [float(_count_plate_appearances(start_rows)) for start_rows in recent_starts]
    return _mean(pitch_counts), _mean(batter_counts)


def _build_pitcher_daily_feature_row(
    *,
    starter: ProbableStarterRecord,
    game: GameRecord,
    all_rows: list[StatcastPitchRecord],
    history_start_date: date,
    history_end_date: date,
) -> PitcherDailyFeatureRow:
    features_as_of = max(_history_cutoff(date.fromisoformat(starter.official_date)), starter.captured_at)
    opponent_team_abbreviation, _ = _opponent_team(game, starter.team_side)

    if starter.pitcher_id is None:
        return PitcherDailyFeatureRow(
            feature_row_id=f"pitcher-feature:{starter.game_pk}:missing:{starter.official_date}",
            official_date=starter.official_date,
            game_pk=starter.game_pk,
            pitcher_id=None,
            pitcher_name=starter.pitcher_name,
            team_side=starter.team_side,
            team_abbreviation=starter.team_abbreviation,
            opponent_team_abbreviation=opponent_team_abbreviation,
            history_start_date=history_start_date,
            history_end_date=history_end_date,
            features_as_of=features_as_of,
            feature_status="missing_pitcher_id",
            pitch_sample_size=0,
            plate_appearance_sample_size=0,
            pitcher_hand=None,
            pitcher_k_rate=None,
            pitcher_k_rate_vs_rhh=None,
            pitcher_k_rate_vs_lhh=None,
            swinging_strike_rate=None,
            pitcher_whiff_rate_vs_rhh=None,
            pitcher_whiff_rate_vs_lhh=None,
            csw_rate=None,
            pitch_type_usage={},
            average_release_speed=None,
            release_speed_delta_vs_baseline=None,
            average_release_extension=None,
            release_extension_delta_vs_baseline=None,
            recent_batters_faced=0,
            recent_pitch_count=0,
            rest_days=None,
            last_start_pitch_count=None,
            last_start_batters_faced=None,
        )

    pitcher_rows = _sorted_rows(_pitch_rows_for_player(all_rows, pitcher_id=starter.pitcher_id))
    recent_rows = _rows_in_recent_window(pitcher_rows, target_date=date.fromisoformat(starter.official_date), days=15)
    final_pitch_rows = [row for row in pitcher_rows if row.is_plate_appearance_final_pitch]
    recent_final_pitch_rows = [row for row in recent_rows if row.is_plate_appearance_final_pitch]
    grouped_starts = _rows_grouped_by_start(pitcher_rows)
    latest_start_rows = grouped_starts[0] if grouped_starts else []
    speed_values = [row.release_speed for row in recent_rows if row.release_speed is not None]
    baseline_speed_values = [row.release_speed for row in pitcher_rows if row.release_speed is not None]
    extension_values = [row.release_extension for row in recent_rows if row.release_extension is not None]
    baseline_extension_values = [row.release_extension for row in pitcher_rows if row.release_extension is not None]
    average_release_speed = _mean(speed_values)
    baseline_release_speed = _mean(baseline_speed_values)
    average_release_extension = _mean(extension_values)
    baseline_release_extension = _mean(baseline_extension_values)

    last_game_date = _last_game_date(pitcher_rows)
    rest_days = None
    if last_game_date is not None:
        rest_days = (date.fromisoformat(starter.official_date) - last_game_date).days

    k_rate_vs_rhh, whiff_rate_vs_rhh = _pitcher_hand_split_rates(
        pitcher_rows=pitcher_rows, stand="R"
    )
    k_rate_vs_lhh, whiff_rate_vs_lhh = _pitcher_hand_split_rates(
        pitcher_rows=pitcher_rows, stand="L"
    )

    return PitcherDailyFeatureRow(
        feature_row_id=f"pitcher-feature:{starter.game_pk}:{starter.pitcher_id}:{starter.official_date}",
        official_date=starter.official_date,
        game_pk=starter.game_pk,
        pitcher_id=starter.pitcher_id,
        pitcher_name=starter.pitcher_name,
        team_side=starter.team_side,
        team_abbreviation=starter.team_abbreviation,
        opponent_team_abbreviation=opponent_team_abbreviation,
        history_start_date=history_start_date,
        history_end_date=history_end_date,
        features_as_of=features_as_of,
        feature_status="ok" if pitcher_rows else "missing_history",
        pitch_sample_size=len(pitcher_rows),
        plate_appearance_sample_size=len(final_pitch_rows),
        pitcher_hand=_pitcher_hand(pitcher_rows),
        pitcher_k_rate=_safe_rate(
            sum(1 for row in final_pitch_rows if row.is_strikeout_event),
            len(final_pitch_rows),
        ),
        pitcher_k_rate_vs_rhh=k_rate_vs_rhh,
        pitcher_k_rate_vs_lhh=k_rate_vs_lhh,
        swinging_strike_rate=_safe_rate(
            sum(1 for row in pitcher_rows if row.is_whiff),
            len(pitcher_rows),
        ),
        pitcher_whiff_rate_vs_rhh=whiff_rate_vs_rhh,
        pitcher_whiff_rate_vs_lhh=whiff_rate_vs_lhh,
        csw_rate=_safe_rate(
            sum(1 for row in pitcher_rows if row.is_whiff or row.is_called_strike),
            len(pitcher_rows),
        ),
        pitch_type_usage=_pitch_type_usage(recent_rows or pitcher_rows),
        average_release_speed=average_release_speed,
        release_speed_delta_vs_baseline=_round_optional(
            None
            if average_release_speed is None or baseline_release_speed is None
            else average_release_speed - baseline_release_speed
        ),
        average_release_extension=average_release_extension,
        release_extension_delta_vs_baseline=_round_optional(
            None
            if average_release_extension is None or baseline_release_extension is None
            else average_release_extension - baseline_release_extension
        ),
        recent_batters_faced=len({_plate_appearance_key(row) for row in recent_final_pitch_rows}),
        recent_pitch_count=len(recent_rows),
        rest_days=rest_days,
        last_start_pitch_count=len(latest_start_rows) if latest_start_rows else None,
        last_start_batters_faced=_count_plate_appearances(latest_start_rows) if latest_start_rows else None,
    )
