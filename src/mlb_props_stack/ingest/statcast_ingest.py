"""Statcast ingest foundation: CSV fetch, CSV parsing, and shared helpers.

This module owns the pieces that produce :class:`StatcastPitchRecord` objects
from Baseball Savant CSV pulls plus the small set of row-level utilities that
every feature derivation module needs (pitcher, lineup, game-context). Keeping
these helpers here — alongside the ``StatcastPitchRecord`` dataclass they
operate on — gives the feature modules a single foundation to depend on
without circular imports.

The HTTP fetch layer dispatches through ``statcast_features.urlopen`` via a
deferred import so existing tests that monkeypatch
``mlb_props_stack.ingest.statcast_features.urlopen`` keep working after the
refactor. The monkeypatch target stays the call site the tests originally
targeted; only the client implementation moved.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from io import StringIO
from pathlib import Path
import time as time_module
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request

from .mlb_stats_api import GameRecord, LineupSnapshot

STATCAST_SEARCH_CSV_ENDPOINT = "https://baseballsavant.mlb.com/statcast_search/csv"
STATCAST_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; mlb-props-stack/0.1; "
        "+https://github.com/DylanMcCavitt/baseball-ml)"
    ),
}
DEFAULT_MAX_FETCH_ATTEMPTS = 4
DEFAULT_INITIAL_BACKOFF_SECONDS = 1.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0
DEFAULT_MAX_BACKOFF_SECONDS = 30.0
DEFAULT_MAX_FETCH_WORKERS = 4
STRIKEOUT_EVENTS = {"strikeout", "strikeout_double_play"}
WHIFF_DESCRIPTIONS = {
    "missed_bunt",
    "swinging_strike",
    "swinging_strike_blocked",
}
CALLED_STRIKE_DESCRIPTIONS = {"called_strike"}
CONTACT_DESCRIPTIONS = {
    "foul",
    "foul_bunt",
    "foul_tip",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}
SWING_DESCRIPTIONS = CONTACT_DESCRIPTIONS | WHIFF_DESCRIPTIONS


@dataclass(frozen=True)
class StatcastPullRecord:
    """One raw Statcast CSV pull used to build a feature run."""

    pull_id: str
    captured_at: datetime
    player_type: str
    player_id: int
    history_start_date: date
    history_end_date: date
    source_url: str
    raw_path: Path
    row_count: int


@dataclass(frozen=True)
class StatcastPitchRecord:
    """Normalized pitch-level base row derived from one Statcast CSV line."""

    pitch_record_id: str
    source_pull_id: str
    source_row_number: int
    game_date: str
    game_pk: int
    at_bat_number: int
    pitch_number: int
    pitcher_id: int
    batter_id: int
    pitch_type: str | None
    pitch_name: str | None
    release_speed: float | None
    release_spin_rate: float | None
    release_extension: float | None
    plate_x: float | None
    plate_z: float | None
    zone: int | None
    description: str | None
    events: str | None
    stand: str | None
    p_throws: str | None
    balls: int | None
    strikes: int | None
    outs_when_up: int | None
    home_team_abbreviation: str | None
    away_team_abbreviation: str | None
    batting_team_abbreviation: str | None
    fielding_team_abbreviation: str | None
    is_plate_appearance_final_pitch: bool
    is_strikeout_event: bool
    is_whiff: bool
    is_called_strike: bool
    is_swing: bool
    is_contact: bool
    is_out_of_zone: bool | None
    is_chase_swing: bool | None


class StatcastSearchClient:
    """Small stdlib-only HTTP client for Baseball Savant Statcast CSV pulls.

    Retries transient failures (5xx, 429, network errors, timeouts) with
    exponential backoff. 4xx responses other than 429 are surfaced immediately
    so a caller sees a bad request instead of hammering the endpoint.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 60.0,
        max_attempts: int = DEFAULT_MAX_FETCH_ATTEMPTS,
        initial_backoff_seconds: float = DEFAULT_INITIAL_BACKOFF_SECONDS,
        backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
        max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
        sleep: Callable[[float], None] = time_module.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if initial_backoff_seconds < 0:
            raise ValueError("initial_backoff_seconds must be non-negative")
        if backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be at least 1")
        if max_backoff_seconds < 0:
            raise ValueError("max_backoff_seconds must be non-negative")
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.initial_backoff_seconds = initial_backoff_seconds
        self.backoff_multiplier = backoff_multiplier
        self.max_backoff_seconds = max_backoff_seconds
        self._sleep = sleep

    def fetch_csv(self, url: str) -> str:
        request = Request(url, headers=STATCAST_REQUEST_HEADERS)
        last_error: Exception | None = None
        for attempt_index in range(self.max_attempts):
            try:
                with _urlopen(request, timeout=self.timeout_seconds) as response:
                    return response.read().decode("utf-8")
            except HTTPError as error:
                if not _is_retriable_http_error(error):
                    raise
                last_error = error
            except (URLError, TimeoutError) as error:
                last_error = error

            if attempt_index == self.max_attempts - 1:
                break
            self._sleep(self._backoff_delay(attempt_index))

        assert last_error is not None  # loop only exits via break or return
        raise last_error

    def _backoff_delay(self, attempt_index: int) -> float:
        raw_delay = self.initial_backoff_seconds * (
            self.backoff_multiplier**attempt_index
        )
        return min(self.max_backoff_seconds, raw_delay)


def _urlopen(*args: Any, **kwargs: Any) -> Any:
    """Resolve ``urlopen`` via the orchestrator module at call time.

    Tests monkeypatch ``mlb_props_stack.ingest.statcast_features.urlopen`` to
    stub HTTP behaviour. Importing ``statcast_features`` lazily here (and
    doing the attribute lookup at call time rather than at import time) keeps
    that patch effective even though the actual fetch logic now lives in this
    module. The lazy import also sidesteps the circular-import concern with
    the orchestrator.
    """
    from . import statcast_features  # deferred to avoid circular import

    return statcast_features.urlopen(*args, **kwargs)


def _is_retriable_http_error(error: HTTPError) -> bool:
    # 429 Too Many Requests and any 5xx indicate a transient condition where
    # backing off and retrying is appropriate. Everything else (400, 401, 403,
    # 404, ...) reflects a permanent request problem that retries cannot fix.
    return error.code == 429 or error.code >= 500


def _fetch_csv_texts_concurrently(
    *,
    client: StatcastSearchClient,
    source_urls: list[str],
    max_workers: int,
) -> list[str]:
    """Fetch CSV payloads in parallel, preserving the input order of URLs."""
    if not source_urls:
        return []
    worker_count = min(max_workers, len(source_urls))
    if worker_count <= 1:
        return [client.fetch_csv(url) for url in source_urls]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(client.fetch_csv, source_urls))


def build_statcast_search_csv_url(
    *,
    player_type: str,
    player_id: int,
    start_date: date,
    end_date: date,
) -> str:
    """Return a reproducible Statcast Search CSV URL for one player window."""
    if player_type not in {"pitcher", "batter"}:
        raise ValueError("player_type must be 'pitcher' or 'batter'")

    player_lookup_key = "pitchers_lookup[]" if player_type == "pitcher" else "batters_lookup[]"
    query = [
        ("all", "true"),
        ("hfPT", ""),
        ("hfAB", ""),
        ("hfBBT", ""),
        ("hfPR", ""),
        ("hfZ", ""),
        ("stadium", ""),
        ("hfBBL", ""),
        ("hfNewZones", ""),
        ("hfGT", "R|"),
        ("hfC", ""),
        ("hfSea", ""),
        ("hfSit", ""),
        ("player_type", player_type),
        ("hfOuts", ""),
        ("opponent", ""),
        ("pitcher_throws", ""),
        ("batter_stands", ""),
        ("hfSA", ""),
        ("game_date_gt", start_date.isoformat()),
        ("game_date_lt", end_date.isoformat()),
        (player_lookup_key, str(player_id)),
        ("team", ""),
        ("position", ""),
        ("hfRO", ""),
        ("home_road", ""),
        ("hfFlag", ""),
        ("metric_1", ""),
        ("hfInn", ""),
        ("min_pitches", "0"),
        ("min_results", "0"),
        ("group_by", "name"),
        ("sort_col", "pitches"),
        ("player_event_sort", "api_p_release_speed"),
        ("sort_order", "desc"),
        ("min_pas", "0"),
        ("type", "details"),
    ]
    return f"{STATCAST_SEARCH_CSV_ENDPOINT}?{urlencode(query, doseq=True)}"


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return _round_optional(numerator / denominator)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return _round_optional(sum(values) / len(values))


def _history_cutoff(target_date: date) -> datetime:
    return datetime.combine(target_date, time.min, tzinfo=UTC)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    if not rendered or rendered.lower() == "null":
        return None
    return rendered


def _coerce_optional_int(value: Any) -> int | None:
    rendered = _optional_text(value)
    if rendered is None:
        return None
    return int(float(rendered))


def _coerce_optional_float(value: Any) -> float | None:
    rendered = _optional_text(value)
    if rendered is None:
        return None
    return float(rendered)


def _pitch_record_id(
    *,
    game_pk: int,
    at_bat_number: int,
    pitch_number: int,
    pitcher_id: int,
    batter_id: int,
) -> str:
    return f"pitch:{game_pk}:{at_bat_number}:{pitch_number}:{pitcher_id}:{batter_id}"


def _plate_appearance_key(row: StatcastPitchRecord) -> tuple[int, int]:
    return row.game_pk, row.at_bat_number


def _batting_team_abbreviation(
    *,
    inning_topbot: str | None,
    home_team_abbreviation: str | None,
    away_team_abbreviation: str | None,
) -> tuple[str | None, str | None]:
    if inning_topbot == "Top":
        return away_team_abbreviation, home_team_abbreviation
    if inning_topbot == "Bot":
        return home_team_abbreviation, away_team_abbreviation
    return None, None


def _is_out_of_zone(zone: int | None) -> bool | None:
    if zone is None:
        return None
    return zone not in {1, 2, 3, 4, 5, 6, 7, 8, 9}


def normalize_statcast_csv_text(
    csv_text: str,
    *,
    pull_id: str,
) -> list[StatcastPitchRecord]:
    """Normalize one raw Statcast CSV pull into pitch-level base rows."""
    reader = csv.DictReader(StringIO(csv_text))
    if not reader.fieldnames or "game_date" not in reader.fieldnames:
        return []

    records: list[StatcastPitchRecord] = []
    for row_number, row in enumerate(reader, start=2):
        game_date = _optional_text(row.get("game_date"))
        game_pk = _coerce_optional_int(row.get("game_pk"))
        at_bat_number = _coerce_optional_int(row.get("at_bat_number"))
        pitch_number = _coerce_optional_int(row.get("pitch_number"))
        pitcher_id = _coerce_optional_int(row.get("pitcher"))
        batter_id = _coerce_optional_int(row.get("batter"))
        if (
            game_date is None
            or game_pk is None
            or at_bat_number is None
            or pitch_number is None
            or pitcher_id is None
            or batter_id is None
        ):
            continue

        description = _optional_text(row.get("description"))
        description_key = description.lower() if description else None
        events = _optional_text(row.get("events"))
        events_key = events.lower() if events else None
        zone = _coerce_optional_int(row.get("zone"))
        is_out_of_zone = _is_out_of_zone(zone)
        home_team_abbreviation = _optional_text(row.get("home_team"))
        away_team_abbreviation = _optional_text(row.get("away_team"))
        batting_team_abbreviation, fielding_team_abbreviation = _batting_team_abbreviation(
            inning_topbot=_optional_text(row.get("inning_topbot")),
            home_team_abbreviation=home_team_abbreviation,
            away_team_abbreviation=away_team_abbreviation,
        )
        is_whiff = description_key in WHIFF_DESCRIPTIONS
        is_called_strike = description_key in CALLED_STRIKE_DESCRIPTIONS
        is_swing = description_key in SWING_DESCRIPTIONS
        is_contact = description_key in CONTACT_DESCRIPTIONS
        is_plate_appearance_final_pitch = events_key is not None
        is_strikeout_event = events_key in STRIKEOUT_EVENTS
        is_chase_swing = None
        if is_out_of_zone is not None:
            is_chase_swing = is_out_of_zone and is_swing

        records.append(
            StatcastPitchRecord(
                pitch_record_id=_pitch_record_id(
                    game_pk=game_pk,
                    at_bat_number=at_bat_number,
                    pitch_number=pitch_number,
                    pitcher_id=pitcher_id,
                    batter_id=batter_id,
                ),
                source_pull_id=pull_id,
                source_row_number=row_number,
                game_date=game_date,
                game_pk=game_pk,
                at_bat_number=at_bat_number,
                pitch_number=pitch_number,
                pitcher_id=pitcher_id,
                batter_id=batter_id,
                pitch_type=_optional_text(row.get("pitch_type")),
                pitch_name=_optional_text(row.get("pitch_name")),
                release_speed=_coerce_optional_float(row.get("release_speed")),
                release_spin_rate=_coerce_optional_float(
                    row.get("release_spin_rate") or row.get("release_spin")
                ),
                release_extension=_coerce_optional_float(row.get("release_extension")),
                plate_x=_coerce_optional_float(row.get("plate_x")),
                plate_z=_coerce_optional_float(row.get("plate_z")),
                zone=zone,
                description=description,
                events=events,
                stand=_optional_text(row.get("stand")),
                p_throws=_optional_text(row.get("p_throws")),
                balls=_coerce_optional_int(row.get("balls")),
                strikes=_coerce_optional_int(row.get("strikes")),
                outs_when_up=_coerce_optional_int(row.get("outs_when_up")),
                home_team_abbreviation=home_team_abbreviation,
                away_team_abbreviation=away_team_abbreviation,
                batting_team_abbreviation=batting_team_abbreviation,
                fielding_team_abbreviation=fielding_team_abbreviation,
                is_plate_appearance_final_pitch=is_plate_appearance_final_pitch,
                is_strikeout_event=is_strikeout_event,
                is_whiff=is_whiff,
                is_called_strike=is_called_strike,
                is_swing=is_swing,
                is_contact=is_contact,
                is_out_of_zone=is_out_of_zone,
                is_chase_swing=is_chase_swing,
            )
        )

    return records


# ---------------------------------------------------------------------------
# Shared pitch-record utilities used across the feature derivation modules.
# ---------------------------------------------------------------------------


def _sorted_rows(rows: list[StatcastPitchRecord]) -> list[StatcastPitchRecord]:
    return sorted(
        rows,
        key=lambda row: (row.game_date, row.game_pk, row.at_bat_number, row.pitch_number),
    )


def _pitch_rows_for_player(
    rows: list[StatcastPitchRecord], *, pitcher_id: int
) -> list[StatcastPitchRecord]:
    return [row for row in rows if row.pitcher_id == pitcher_id]


def _batter_rows(
    rows: list[StatcastPitchRecord], *, batter_id: int
) -> list[StatcastPitchRecord]:
    return [row for row in rows if row.batter_id == batter_id]


def _count_plate_appearances(rows: list[StatcastPitchRecord]) -> int:
    return len({_plate_appearance_key(row) for row in rows if row.is_plate_appearance_final_pitch})


def _last_game_date(rows: list[StatcastPitchRecord]) -> date | None:
    if not rows:
        return None
    return date.fromisoformat(max(row.game_date for row in rows))


def _pitch_type_usage(rows: list[StatcastPitchRecord]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.pitch_type:
            counts[row.pitch_type] = counts.get(row.pitch_type, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {
        pitch_type: round(count / total, 6)
        for pitch_type, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    }


def _rows_in_recent_window(
    rows: list[StatcastPitchRecord],
    *,
    target_date: date,
    days: int,
) -> list[StatcastPitchRecord]:
    start_date = target_date - timedelta(days=days)
    return [row for row in rows if date.fromisoformat(row.game_date) >= start_date]


def _rows_grouped_by_start(
    rows: list[StatcastPitchRecord],
) -> list[list[StatcastPitchRecord]]:
    groups: dict[tuple[str, int], list[StatcastPitchRecord]] = {}
    for row in rows:
        groups.setdefault((row.game_date, row.game_pk), []).append(row)
    return [
        _sorted_rows(group_rows)
        for _, group_rows in sorted(groups.items(), key=lambda item: item[0], reverse=True)
    ]


# ---------------------------------------------------------------------------
# Team / lineup helpers shared across feature modules.
# ---------------------------------------------------------------------------


def _opponent_team_side(team_side: str) -> str:
    return "home" if team_side == "away" else "away"


def _opponent_team(game: GameRecord, team_side: str) -> tuple[str, str]:
    if team_side == "away":
        return game.home_team_abbreviation, game.home_team_name
    return game.away_team_abbreviation, game.away_team_name


def _select_pregame_lineup_snapshot(
    *,
    game: GameRecord,
    team_side: str,
    lineup_snapshots: tuple[LineupSnapshot, ...],
) -> LineupSnapshot | None:
    candidates = [
        snapshot
        for snapshot in lineup_snapshots
        if snapshot.game_pk == game.game_pk
        and snapshot.team_side == team_side
        and snapshot.captured_at <= game.commence_time
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda snapshot: snapshot.captured_at)
