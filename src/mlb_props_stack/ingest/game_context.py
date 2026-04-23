"""Game-context feature derivation: venue, weather, umpire, rest, leash.

Joins the per-game pregame context (park factor, weather, umpire) with
pitcher-derived leash estimates into one :class:`GameContextFeatureRow` per
probable starter.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from .mlb_stats_api import GameRecord, LineupSnapshot, ProbableStarterRecord
from .park_factors import (
    PARK_FACTOR_STATUS_MISSING_SOURCE,
    PARK_FACTOR_STATUS_OK,
    ParkKFactorRecord,
    lookup_park_k_factor,
)
from .pitcher_features import _expected_leash
from .statcast_ingest import (
    StatcastPitchRecord,
    _history_cutoff,
    _last_game_date,
    _opponent_team,
    _pitch_rows_for_player,
    _sorted_rows,
)
from .umpire import UMPIRE_STATUS_MISSING_SOURCE, UmpireSnapshotRecord
from .weather import WEATHER_STATUS_MISSING_SOURCE, WeatherSnapshotRecord


@dataclass(frozen=True)
class GameContextFeatureRow:
    """Pregame game-context feature row for one probable starter."""

    feature_row_id: str
    official_date: str
    game_pk: int
    pitcher_id: int | None
    pitcher_name: str | None
    team_abbreviation: str
    opponent_team_abbreviation: str
    home_away: str
    venue_id: int | None
    venue_name: str
    day_night: str
    double_header: str
    features_as_of: datetime
    park_k_factor: float | None
    park_k_factor_vs_rhh: float | None
    park_k_factor_vs_lhh: float | None
    park_factor_status: str
    rest_days: int | None
    weather_status: str
    weather_source: str | None
    weather_temperature_f: float | None
    weather_wind_speed_mph: float | None
    weather_wind_direction_deg: float | None
    weather_humidity_pct: float | None
    weather_captured_at: datetime | None
    roof_type: str | None
    umpire_status: str
    umpire_source: str | None
    umpire_id: int | None
    umpire_name: str | None
    umpire_captured_at: datetime | None
    ump_called_strike_rate_30d: float | None
    ump_k_per_9_delta_vs_league_30d: float | None
    expected_leash_pitch_count: float | None
    expected_leash_batters_faced: float | None


def _build_game_context_feature_row(
    *,
    starter: ProbableStarterRecord,
    game: GameRecord,
    lineup_snapshot: LineupSnapshot | None,
    all_rows: list[StatcastPitchRecord],
    park_k_factor_table: dict[tuple[int, int], ParkKFactorRecord],
    weather_lookup: dict[int, WeatherSnapshotRecord],
    umpire_lookup: dict[int, UmpireSnapshotRecord],
) -> GameContextFeatureRow:
    base_features_as_of = max(game.captured_at, starter.captured_at, _history_cutoff(date.fromisoformat(starter.official_date)))
    if lineup_snapshot is not None:
        base_features_as_of = max(base_features_as_of, lineup_snapshot.captured_at)

    rest_days = None
    expected_leash_pitch_count = None
    expected_leash_batters_faced = None
    if starter.pitcher_id is not None:
        pitcher_rows = _sorted_rows(_pitch_rows_for_player(all_rows, pitcher_id=starter.pitcher_id))
        last_game_date = _last_game_date(pitcher_rows)
        if last_game_date is not None:
            rest_days = (date.fromisoformat(starter.official_date) - last_game_date).days
        expected_leash_pitch_count, expected_leash_batters_faced = _expected_leash(pitcher_rows)

    opponent_team_abbreviation, _ = _opponent_team(game, starter.team_side)
    park_factor_record = lookup_park_k_factor(
        season=date.fromisoformat(starter.official_date).year,
        venue_mlb_id=game.venue_id,
        table=park_k_factor_table,
    )
    if park_factor_record is not None:
        park_k_factor = park_factor_record.park_k_factor
        park_k_factor_vs_rhh = park_factor_record.park_k_factor_vs_rhh
        park_k_factor_vs_lhh = park_factor_record.park_k_factor_vs_lhh
        park_factor_status = PARK_FACTOR_STATUS_OK
    else:
        park_k_factor = None
        park_k_factor_vs_rhh = None
        park_k_factor_vs_lhh = None
        park_factor_status = PARK_FACTOR_STATUS_MISSING_SOURCE

    weather_snapshot = weather_lookup.get(starter.game_pk)
    if weather_snapshot is not None:
        weather_status = weather_snapshot.weather_status
        weather_source = weather_snapshot.weather_source
        weather_temperature_f = weather_snapshot.temperature_f
        weather_wind_speed_mph = weather_snapshot.wind_speed_mph
        weather_wind_direction_deg = weather_snapshot.wind_direction_deg
        weather_humidity_pct = weather_snapshot.humidity_pct
        weather_captured_at = weather_snapshot.captured_at
        roof_type = weather_snapshot.roof_type
    else:
        weather_status = WEATHER_STATUS_MISSING_SOURCE
        weather_source = None
        weather_temperature_f = None
        weather_wind_speed_mph = None
        weather_wind_direction_deg = None
        weather_humidity_pct = None
        weather_captured_at = None
        roof_type = None

    umpire_snapshot = umpire_lookup.get(starter.game_pk)
    if umpire_snapshot is not None:
        umpire_status = umpire_snapshot.umpire_status
        umpire_source = umpire_snapshot.umpire_source
        umpire_id = umpire_snapshot.umpire_id
        umpire_name = umpire_snapshot.umpire_name
        umpire_captured_at = umpire_snapshot.captured_at
        ump_called_strike_rate_30d = umpire_snapshot.ump_called_strike_rate_30d
        ump_k_per_9_delta_vs_league_30d = umpire_snapshot.ump_k_per_9_delta_vs_league_30d
    else:
        umpire_status = UMPIRE_STATUS_MISSING_SOURCE
        umpire_source = None
        umpire_id = None
        umpire_name = None
        umpire_captured_at = None
        ump_called_strike_rate_30d = None
        ump_k_per_9_delta_vs_league_30d = None

    return GameContextFeatureRow(
        feature_row_id=(
            f"game-context:{starter.game_pk}:{starter.pitcher_id or 'missing'}:{starter.official_date}"
        ),
        official_date=starter.official_date,
        game_pk=starter.game_pk,
        pitcher_id=starter.pitcher_id,
        pitcher_name=starter.pitcher_name,
        team_abbreviation=starter.team_abbreviation,
        opponent_team_abbreviation=opponent_team_abbreviation,
        home_away="away" if starter.team_side == "away" else "home",
        venue_id=game.venue_id,
        venue_name=game.venue_name,
        day_night=game.day_night,
        double_header=game.double_header,
        features_as_of=base_features_as_of,
        park_k_factor=park_k_factor,
        park_k_factor_vs_rhh=park_k_factor_vs_rhh,
        park_k_factor_vs_lhh=park_k_factor_vs_lhh,
        park_factor_status=park_factor_status,
        rest_days=rest_days,
        weather_status=weather_status,
        weather_source=weather_source,
        weather_temperature_f=weather_temperature_f,
        weather_wind_speed_mph=weather_wind_speed_mph,
        weather_wind_direction_deg=weather_wind_direction_deg,
        weather_humidity_pct=weather_humidity_pct,
        weather_captured_at=weather_captured_at,
        roof_type=roof_type,
        umpire_status=umpire_status,
        umpire_source=umpire_source,
        umpire_id=umpire_id,
        umpire_name=umpire_name,
        umpire_captured_at=umpire_captured_at,
        ump_called_strike_rate_30d=ump_called_strike_rate_30d,
        ump_k_per_9_delta_vs_league_30d=ump_k_per_9_delta_vs_league_30d,
        expected_leash_pitch_count=expected_leash_pitch_count,
        expected_leash_batters_faced=expected_leash_batters_faced,
    )
