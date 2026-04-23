"""Statcast-derived pitcher, lineup, and game-context feature ingest."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, date, datetime, time, timedelta
import csv
from io import StringIO
import json
import os
from pathlib import Path
import time as time_module
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from ..config import StackConfig
from .mlb_stats_api import (
    GameRecord,
    LineupEntry,
    LineupSnapshot,
    ProbableStarterRecord,
    format_utc_timestamp,
    parse_api_datetime,
    utc_now,
)
from .park_factors import (
    PARK_FACTOR_STATUS_MISSING_SOURCE,
    PARK_FACTOR_STATUS_OK,
    ParkKFactorRecord,
    load_park_k_factors,
    lookup_park_k_factor,
)
from .weather import (
    WEATHER_STATUS_MISSING_SOURCE,
    WeatherSnapshotRecord,
    load_latest_weather_snapshots_for_date,
)

STATCAST_SEARCH_CSV_ENDPOINT = "https://baseballsavant.mlb.com/statcast_search/csv"
STATCAST_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; mlb-props-stack/0.1; "
        "+https://github.com/DylanMcCavitt/baseball-ml)"
    ),
}
DEFAULT_HISTORY_DAYS = 30
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
    expected_leash_pitch_count: float | None
    expected_leash_batters_faced: float | None


@dataclass(frozen=True)
class StatcastFeatureIngestResult:
    """Filesystem output summary for one Statcast feature build."""

    target_date: date
    history_start_date: date
    history_end_date: date
    run_id: str
    mlb_games_path: Path
    mlb_probable_starters_path: Path
    mlb_lineup_snapshots_path: Path
    pull_manifest_path: Path
    pitch_level_base_path: Path
    pitcher_daily_features_path: Path
    lineup_daily_features_path: Path
    game_context_features_path: Path
    raw_pull_count: int
    pitch_level_record_count: int
    pitcher_feature_count: int
    lineup_feature_count: int
    game_context_feature_count: int


@dataclass(frozen=True)
class _LoadedMLBMetadata:
    games_path: Path
    probable_starters_path: Path
    lineup_snapshots_path: Path
    games: tuple[GameRecord, ...]
    probable_starters: tuple[ProbableStarterRecord, ...]
    lineup_snapshots: tuple[LineupSnapshot, ...]


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
                with urlopen(request, timeout=self.timeout_seconds) as response:
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


def _path_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _history_cutoff(target_date: date) -> datetime:
    return datetime.combine(target_date, time.min, tzinfo=UTC)


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


def _pitch_record_id(*, game_pk: int, at_bat_number: int, pitch_number: int, pitcher_id: int, batter_id: int) -> str:
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


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, datetime):
        return format_utc_timestamp(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(_json_ready(payload), indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(_json_ready(record), sort_keys=True))
                handle.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _latest_complete_run_dir(root: Path) -> Path:
    run_dirs = sorted(
        (path for path in root.glob("run=*") if path.is_dir()),
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(
            "No normalized MLB metadata runs were found. "
            "Run `uv run python -m mlb_props_stack ingest-mlb-metadata --date ...` first."
        )
    for run_dir in run_dirs:
        games_path = run_dir / "games.jsonl"
        probable_starters_path = run_dir / "probable_starters.jsonl"
        if games_path.exists() and probable_starters_path.exists():
            return run_dir
    raise FileNotFoundError(f"Latest MLB metadata runs under {root} were incomplete.")


def _latest_pregame_valid_run_dir(root: Path) -> Path:
    run_dirs = sorted(
        (path for path in root.glob("run=*") if path.is_dir()),
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(
            "No normalized MLB metadata runs were found. "
            "Run `uv run python -m mlb_props_stack ingest-mlb-metadata --date ...` first."
        )

    for run_dir in run_dirs:
        games_path = run_dir / "games.jsonl"
        probable_starters_path = run_dir / "probable_starters.jsonl"
        if not games_path.exists() or not probable_starters_path.exists():
            continue

        games_rows = _load_jsonl_rows(games_path)
        probable_starters_rows = _load_jsonl_rows(probable_starters_path)
        if _run_is_pregame_valid(
            games_rows=games_rows,
            probable_starters_rows=probable_starters_rows,
        ):
            return run_dir

    raise FileNotFoundError(
        "No pregame-valid normalized MLB metadata runs were found. "
        "Run `uv run python -m mlb_props_stack ingest-mlb-metadata --date ...` "
        "before first pitch for the target slate."
    )


def _run_is_pregame_valid(
    *,
    games_rows: list[dict[str, Any]],
    probable_starters_rows: list[dict[str, Any]],
) -> bool:
    try:
        commence_times_by_game_pk: dict[int, datetime] = {}
        for row in games_rows:
            game_pk = int(row["game_pk"])
            commence_time = parse_api_datetime(row["commence_time"])
            captured_at = parse_api_datetime(row["captured_at"])
            if captured_at > commence_time:
                return False
            commence_times_by_game_pk[game_pk] = commence_time

        for row in probable_starters_rows:
            game_pk = int(row["game_pk"])
            commence_time = commence_times_by_game_pk.get(game_pk)
            if commence_time is None:
                return False
            captured_at = parse_api_datetime(row["captured_at"])
            if captured_at > commence_time:
                return False
    except (KeyError, TypeError, ValueError):
        return False

    return True


def _load_latest_mlb_metadata_for_date(
    *,
    target_date: date,
    output_dir: Path | str,
    reference_time: datetime,
) -> _LoadedMLBMetadata:
    normalized_root = (
        Path(output_dir)
        / "normalized"
        / "mlb_stats_api"
        / f"date={target_date.isoformat()}"
    )
    config = StackConfig()
    reference_date = reference_time.astimezone(ZoneInfo(config.timezone)).date()
    try:
        latest_run_dir = _latest_pregame_valid_run_dir(normalized_root)
    except FileNotFoundError:
        if target_date >= reference_date:
            raise
        latest_run_dir = _latest_complete_run_dir(normalized_root)
    games_path = latest_run_dir / "games.jsonl"
    probable_starters_path = latest_run_dir / "probable_starters.jsonl"
    lineup_snapshots_path = latest_run_dir / "lineup_snapshots.jsonl"
    if not games_path.exists() or not probable_starters_path.exists() or not lineup_snapshots_path.exists():
        raise FileNotFoundError(
            f"Expected MLB metadata artifacts in {latest_run_dir}, but they were incomplete."
        )

    games = tuple(
        GameRecord(
            game_pk=row["game_pk"],
            official_date=row["official_date"],
            commence_time=parse_api_datetime(row["commence_time"]),
            captured_at=parse_api_datetime(row["captured_at"]),
            status=row["status"],
            status_code=row["status_code"],
            venue_id=row["venue_id"],
            venue_name=row["venue_name"],
            home_team_id=row["home_team_id"],
            home_team_abbreviation=row["home_team_abbreviation"],
            home_team_name=row["home_team_name"],
            away_team_id=row["away_team_id"],
            away_team_abbreviation=row["away_team_abbreviation"],
            away_team_name=row["away_team_name"],
            game_number=row["game_number"],
            double_header=row["double_header"],
            day_night=row["day_night"],
            odds_matchup_key=row["odds_matchup_key"],
        )
        for row in _load_jsonl_rows(games_path)
    )
    probable_starters = tuple(
        ProbableStarterRecord(
            game_pk=row["game_pk"],
            official_date=row["official_date"],
            captured_at=parse_api_datetime(row["captured_at"]),
            team_side=row["team_side"],
            team_id=row["team_id"],
            team_abbreviation=row["team_abbreviation"],
            team_name=row["team_name"],
            pitcher_id=row["pitcher_id"],
            pitcher_name=row["pitcher_name"],
            pitcher_note=row["pitcher_note"],
            odds_matchup_key=row["odds_matchup_key"],
        )
        for row in _load_jsonl_rows(probable_starters_path)
    )
    lineup_snapshots = tuple(
        LineupSnapshot(
            lineup_snapshot_id=row["lineup_snapshot_id"],
            game_pk=row["game_pk"],
            official_date=row["official_date"],
            captured_at=parse_api_datetime(row["captured_at"]),
            team_side=row["team_side"],
            team_id=row["team_id"],
            team_abbreviation=row["team_abbreviation"],
            team_name=row["team_name"],
            game_state=row["game_state"],
            game_status_code=row["game_status_code"],
            is_confirmed=row["is_confirmed"],
            batting_order_player_ids=tuple(row["batting_order_player_ids"]),
            batter_player_ids=tuple(row["batter_player_ids"]),
            lineup_entries=tuple(
                LineupEntry(
                    lineup_position=entry["lineup_position"],
                    player_id=entry["player_id"],
                    player_name=entry["player_name"],
                    batting_order_code=entry["batting_order_code"],
                    position_abbreviation=entry["position_abbreviation"],
                )
                for entry in row["lineup_entries"]
            ),
            odds_matchup_key=row["odds_matchup_key"],
        )
        for row in _load_jsonl_rows(lineup_snapshots_path)
    )
    return _LoadedMLBMetadata(
        games_path=games_path,
        probable_starters_path=probable_starters_path,
        lineup_snapshots_path=lineup_snapshots_path,
        games=games,
        probable_starters=probable_starters,
        lineup_snapshots=lineup_snapshots,
    )


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


def _sorted_rows(rows: list[StatcastPitchRecord]) -> list[StatcastPitchRecord]:
    return sorted(
        rows,
        key=lambda row: (row.game_date, row.game_pk, row.at_bat_number, row.pitch_number),
    )


def _pitch_rows_for_player(rows: list[StatcastPitchRecord], *, pitcher_id: int) -> list[StatcastPitchRecord]:
    return [row for row in rows if row.pitcher_id == pitcher_id]


def _batter_rows(rows: list[StatcastPitchRecord], *, batter_id: int) -> list[StatcastPitchRecord]:
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


def _rows_grouped_by_start(rows: list[StatcastPitchRecord]) -> list[list[StatcastPitchRecord]]:
    groups: dict[tuple[str, int], list[StatcastPitchRecord]] = {}
    for row in rows:
        groups.setdefault((row.game_date, row.game_pk), []).append(row)
    return [
        _sorted_rows(group_rows)
        for _, group_rows in sorted(groups.items(), key=lambda item: item[0], reverse=True)
    ]


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


def _expected_leash(rows: list[StatcastPitchRecord]) -> tuple[float | None, float | None]:
    grouped_starts = _rows_grouped_by_start(rows)
    if not grouped_starts:
        return None, None
    recent_starts = grouped_starts[:3]
    pitch_counts = [float(len(start_rows)) for start_rows in recent_starts]
    batter_counts = [float(_count_plate_appearances(start_rows)) for start_rows in recent_starts]
    return _mean(pitch_counts), _mean(batter_counts)


def _build_game_context_feature_row(
    *,
    starter: ProbableStarterRecord,
    game: GameRecord,
    lineup_snapshot: LineupSnapshot | None,
    all_rows: list[StatcastPitchRecord],
    park_k_factor_table: dict[tuple[int, int], ParkKFactorRecord],
    weather_lookup: dict[int, WeatherSnapshotRecord],
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
        expected_leash_pitch_count=expected_leash_pitch_count,
        expected_leash_batters_faced=expected_leash_batters_faced,
    )


def ingest_statcast_features_for_date(
    *,
    target_date: date,
    output_dir: Path | str = "data",
    history_days: int = DEFAULT_HISTORY_DAYS,
    client: StatcastSearchClient | None = None,
    now: Callable[[], datetime] = utc_now,
    max_fetch_workers: int = DEFAULT_MAX_FETCH_WORKERS,
) -> StatcastFeatureIngestResult:
    """Fetch Statcast pulls and build normalized feature tables for one slate date."""
    if history_days < 1:
        raise ValueError("history_days must be at least 1")
    if max_fetch_workers < 1:
        raise ValueError("max_fetch_workers must be at least 1")
    if client is None:
        client = StatcastSearchClient()

    run_started_at = now().astimezone(UTC)
    history_end_date = target_date - timedelta(days=1)
    history_start_date = target_date - timedelta(days=history_days)
    mlb_metadata = _load_latest_mlb_metadata_for_date(
        target_date=target_date,
        output_dir=output_dir,
        reference_time=run_started_at,
    )
    games_by_pk = {game.game_pk: game for game in mlb_metadata.games}

    selected_lineups: dict[tuple[int, str], LineupSnapshot | None] = {}
    batter_ids: set[int] = set()
    for starter in mlb_metadata.probable_starters:
        game = games_by_pk.get(starter.game_pk)
        if game is None:
            continue
        opponent_side = _opponent_team_side(starter.team_side)
        selected_lineup = _select_pregame_lineup_snapshot(
            game=game,
            team_side=opponent_side,
            lineup_snapshots=mlb_metadata.lineup_snapshots,
        )
        selected_lineups[(starter.game_pk, opponent_side)] = selected_lineup
        if selected_lineup is not None:
            batter_ids.update(selected_lineup.batting_order_player_ids)

    pitch_records_by_id: dict[str, StatcastPitchRecord] = {}
    pull_records: list[StatcastPullRecord] = []
    output_root = Path(output_dir)
    run_id = _path_timestamp(run_started_at)

    pull_requests = [
        ("pitcher", pitcher_id)
        for pitcher_id in sorted(
            {
                starter.pitcher_id
                for starter in mlb_metadata.probable_starters
                if starter.pitcher_id is not None
            }
        )
    ]
    pull_requests.extend(("batter", batter_id) for batter_id in sorted(batter_ids))

    # Pre-compute one spec per pull serially so the `now()` test seam is
    # consumed in deterministic order even when fetches run in parallel.
    pull_specs: list[tuple[str, int, datetime, str]] = []
    for player_type, player_id in pull_requests:
        captured_at = now().astimezone(UTC)
        source_url = build_statcast_search_csv_url(
            player_type=player_type,
            player_id=player_id,
            start_date=history_start_date,
            end_date=history_end_date,
        )
        pull_specs.append((player_type, player_id, captured_at, source_url))

    csv_texts = _fetch_csv_texts_concurrently(
        client=client,
        source_urls=[spec[3] for spec in pull_specs],
        max_workers=max_fetch_workers,
    )

    for (player_type, player_id, captured_at, source_url), csv_text in zip(
        pull_specs, csv_texts, strict=True
    ):
        raw_path = (
            output_root
            / "raw"
            / "statcast_search"
            / f"date={target_date.isoformat()}"
            / f"player_type={player_type}"
            / f"player_id={player_id}"
            / f"captured_at={_path_timestamp(captured_at)}.csv"
        )
        _write_text(raw_path, csv_text)
        pull_id = f"statcast-pull:{player_type}:{player_id}:{_path_timestamp(captured_at)}"
        normalized_rows = normalize_statcast_csv_text(csv_text, pull_id=pull_id)
        pull_records.append(
            StatcastPullRecord(
                pull_id=pull_id,
                captured_at=captured_at,
                player_type=player_type,
                player_id=player_id,
                history_start_date=history_start_date,
                history_end_date=history_end_date,
                source_url=source_url,
                raw_path=raw_path,
                row_count=len(normalized_rows),
            )
        )
        for row in normalized_rows:
            pitch_records_by_id.setdefault(row.pitch_record_id, row)

    all_pitch_records = _sorted_rows(list(pitch_records_by_id.values()))
    pitcher_feature_rows: list[PitcherDailyFeatureRow] = []
    lineup_feature_rows: list[LineupDailyFeatureRow] = []
    game_context_rows: list[GameContextFeatureRow] = []
    park_k_factor_table = load_park_k_factors()
    weather_lookup = load_latest_weather_snapshots_for_date(
        output_dir=output_dir,
        target_date=target_date,
    )

    for starter in mlb_metadata.probable_starters:
        game = games_by_pk.get(starter.game_pk)
        if game is None:
            continue
        pitcher_row = _build_pitcher_daily_feature_row(
            starter=starter,
            game=game,
            all_rows=all_pitch_records,
            history_start_date=history_start_date,
            history_end_date=history_end_date,
        )
        pitcher_feature_rows.append(pitcher_row)
        lineup_feature_rows.append(
            _build_lineup_daily_feature_row(
                starter=starter,
                game=game,
                lineup_snapshot=selected_lineups.get((starter.game_pk, _opponent_team_side(starter.team_side))),
                all_rows=all_pitch_records,
                history_start_date=history_start_date,
                history_end_date=history_end_date,
                pitcher_hand=pitcher_row.pitcher_hand,
            )
        )
        game_context_rows.append(
            _build_game_context_feature_row(
                starter=starter,
                game=game,
                lineup_snapshot=selected_lineups.get((starter.game_pk, _opponent_team_side(starter.team_side))),
                all_rows=all_pitch_records,
                park_k_factor_table=park_k_factor_table,
                weather_lookup=weather_lookup,
            )
        )

    normalized_root = (
        output_root
        / "normalized"
        / "statcast_search"
        / f"date={target_date.isoformat()}"
        / f"run={run_id}"
    )
    pull_manifest_path = normalized_root / "pull_manifest.jsonl"
    pitch_level_base_path = normalized_root / "pitch_level_base.jsonl"
    pitcher_daily_features_path = normalized_root / "pitcher_daily_features.jsonl"
    lineup_daily_features_path = normalized_root / "lineup_daily_features.jsonl"
    game_context_features_path = normalized_root / "game_context_features.jsonl"
    _write_jsonl(pull_manifest_path, pull_records)
    _write_jsonl(pitch_level_base_path, all_pitch_records)
    _write_jsonl(pitcher_daily_features_path, pitcher_feature_rows)
    _write_jsonl(lineup_daily_features_path, lineup_feature_rows)
    _write_jsonl(game_context_features_path, game_context_rows)

    return StatcastFeatureIngestResult(
        target_date=target_date,
        history_start_date=history_start_date,
        history_end_date=history_end_date,
        run_id=run_id,
        mlb_games_path=mlb_metadata.games_path,
        mlb_probable_starters_path=mlb_metadata.probable_starters_path,
        mlb_lineup_snapshots_path=mlb_metadata.lineup_snapshots_path,
        pull_manifest_path=pull_manifest_path,
        pitch_level_base_path=pitch_level_base_path,
        pitcher_daily_features_path=pitcher_daily_features_path,
        lineup_daily_features_path=lineup_daily_features_path,
        game_context_features_path=game_context_features_path,
        raw_pull_count=len(pull_records),
        pitch_level_record_count=len(all_pitch_records),
        pitcher_feature_count=len(pitcher_feature_rows),
        lineup_feature_count=len(lineup_feature_rows),
        game_context_feature_count=len(game_context_rows),
    )
