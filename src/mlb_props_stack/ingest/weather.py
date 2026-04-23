"""Pregame weather ingest anchored to ``commence_time - 60 minutes``.

The ingest fetches one hourly observation per scheduled game at the
venue's lat/lon from the free, no-API-key
`Open-Meteo Archive API <https://open-meteo.com/en/docs/historical-weather-api>`_.
We chose Open-Meteo because it:

* requires no credentials (so historical backfills stay easy to run),
* exposes the four fields called out by AGE-205 — temperature,
  wind speed, wind direction, and humidity — at hourly granularity,
* returns ``UTC`` timestamps in ISO-8601, which lets us pick the hour
  closest to ``commence_time - 60 minutes`` without timezone juggling, and
* covers the entire MLB history window we intend to backfill.

Fixed-roof stadiums skip the network fetch entirely and emit a sentinel
``weather_status="roof_closed"`` row so downstream features can treat
those games as weather-neutral without losing a slate row. Retractable
roofs are ingested as outdoor but carry the ``roof_type="retractable"``
annotation so the model layer can decide whether to trust the snapshot.

The public entry point is :func:`ingest_weather_for_date`. It reads the
latest normalized MLB metadata run for a date, attaches the static
venue-metadata lookup, and writes raw + normalized weather artifacts
under ``data/{raw,normalized}/weather/...``. Each normalized row
enforces ``captured_at <= commence_time`` as a leakage guardrail.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, date, datetime, timedelta
import csv
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .mlb_stats_api import (
    GameRecord,
    format_utc_timestamp,
    parse_api_datetime,
    utc_now,
)

OPEN_METEO_ARCHIVE_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; mlb-props-stack/0.1; "
        "+https://github.com/DylanMcCavitt/baseball-ml)"
    ),
}

DEFAULT_TARGET_OFFSET_MINUTES = 60
DEFAULT_OPEN_METEO_TIMEOUT_SECONDS = 30.0

WEATHER_SOURCE_OPEN_METEO_ARCHIVE = "open_meteo_archive"

WEATHER_STATUS_OK = "ok"
WEATHER_STATUS_ROOF_CLOSED = "roof_closed"
WEATHER_STATUS_MISSING_VENUE_METADATA = "missing_venue_metadata"
WEATHER_STATUS_MISSING_SOURCE = "missing_weather_source"

WEATHER_STATUSES: tuple[str, ...] = (
    WEATHER_STATUS_OK,
    WEATHER_STATUS_ROOF_CLOSED,
    WEATHER_STATUS_MISSING_VENUE_METADATA,
    WEATHER_STATUS_MISSING_SOURCE,
)

ROOF_TYPE_OPEN = "open"
ROOF_TYPE_RETRACTABLE = "retractable"
ROOF_TYPE_FIXED = "fixed"

ROOF_TYPES: tuple[str, ...] = (
    ROOF_TYPE_OPEN,
    ROOF_TYPE_RETRACTABLE,
    ROOF_TYPE_FIXED,
)

DEFAULT_VENUE_METADATA_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "static"
    / "venues"
    / "venue_metadata.csv"
)


@dataclass(frozen=True)
class VenueMetadata:
    """Static geographic + roof metadata for one MLB venue."""

    venue_mlb_id: int
    venue_name: str
    latitude: float
    longitude: float
    roof_type: str


@dataclass(frozen=True)
class WeatherSnapshotRecord:
    """Normalized pregame weather snapshot for one scheduled game."""

    weather_snapshot_id: str
    official_date: str
    game_pk: int
    venue_id: int | None
    venue_name: str
    roof_type: str | None
    latitude: float | None
    longitude: float | None
    commence_time: datetime
    target_time: datetime
    captured_at: datetime
    weather_source: str | None
    weather_status: str
    temperature_f: float | None
    wind_speed_mph: float | None
    wind_direction_deg: float | None
    humidity_pct: float | None
    error_message: str | None


@dataclass(frozen=True)
class WeatherIngestResult:
    """Filesystem output summary for one pregame weather build."""

    target_date: date
    run_id: str
    mlb_games_path: Path
    weather_snapshots_path: Path
    raw_snapshot_paths: tuple[Path, ...]
    snapshot_count: int
    outdoor_snapshot_count: int
    roof_closed_snapshot_count: int
    missing_venue_metadata_count: int
    missing_source_count: int


class OpenMeteoClient:
    """Small stdlib-only HTTP client for the Open-Meteo archive endpoint."""

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_OPEN_METEO_TIMEOUT_SECONDS,
    ) -> None:
        self.timeout_seconds = timeout_seconds

    def fetch_json(self, url: str) -> dict[str, Any]:
        request = Request(url, headers=OPEN_METEO_REQUEST_HEADERS)
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.load(response)


def _path_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        tmp_path.write_text(
            f"{json.dumps(_json_ready(payload), indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_jsonl(path: Path, records: Iterable[Any]) -> None:
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


def load_venue_metadata(
    path: Path | str = DEFAULT_VENUE_METADATA_PATH,
) -> dict[int, VenueMetadata]:
    """Load the static venue lat/lon + roof CSV keyed by ``venue_mlb_id``.

    Skips rows with unparseable id/coordinates so a partially-edited file
    still loads the valid rows instead of raising. Unknown ``roof_type``
    values are rejected because a typo would silently leak a fixed-roof
    stadium into the outdoor path.
    """

    records: dict[int, VenueMetadata] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                venue_id = int(row["venue_mlb_id"])
                latitude = float(row["latitude"])
                longitude = float(row["longitude"])
            except (KeyError, TypeError, ValueError):
                continue
            roof_type = (row.get("roof_type") or "").strip()
            if roof_type not in ROOF_TYPES:
                raise ValueError(
                    f"unknown roof_type {roof_type!r} in venue_metadata row "
                    f"for venue_mlb_id={venue_id}; expected one of {list(ROOF_TYPES)}"
                )
            venue_name = (row.get("venue_name") or "").strip()
            records[venue_id] = VenueMetadata(
                venue_mlb_id=venue_id,
                venue_name=venue_name,
                latitude=latitude,
                longitude=longitude,
                roof_type=roof_type,
            )
    return records


def lookup_venue_metadata(
    venue_mlb_id: int | None,
    table: dict[int, VenueMetadata] | None = None,
) -> VenueMetadata | None:
    """Return the lat/lon + roof record for a venue id or ``None``."""

    if venue_mlb_id is None:
        return None
    records = table if table is not None else load_venue_metadata()
    return records.get(venue_mlb_id)


def build_open_meteo_archive_url(
    *,
    latitude: float,
    longitude: float,
    target_date: date,
) -> str:
    """Return a deterministic Open-Meteo archive URL for one venue/date."""

    query = [
        ("latitude", f"{latitude:.4f}"),
        ("longitude", f"{longitude:.4f}"),
        ("start_date", target_date.isoformat()),
        ("end_date", target_date.isoformat()),
        (
            "hourly",
            "temperature_2m,wind_speed_10m,wind_direction_10m,relative_humidity_2m",
        ),
        ("temperature_unit", "fahrenheit"),
        ("wind_speed_unit", "mph"),
        ("timezone", "UTC"),
    ]
    return f"{OPEN_METEO_ARCHIVE_ENDPOINT}?{urlencode(query)}"


def _parse_open_meteo_hour(value: str) -> datetime:
    # Open-Meteo returns naive "YYYY-MM-DDTHH:MM" strings when timezone=UTC.
    # Attach a UTC offset so downstream comparisons stay timezone-aware.
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def _nearest_hour_index(hours: list[datetime], target_time: datetime) -> int:
    return min(
        range(len(hours)),
        key=lambda index: (
            abs(hours[index] - target_time),
            hours[index],
        ),
    )


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    # Open-Meteo sometimes returns NaN as a literal null, but harden against
    # a stray string/sentinel by rejecting non-finite values explicitly.
    if coerced != coerced or coerced in (float("inf"), float("-inf")):
        return None
    return coerced


@dataclass(frozen=True)
class _HourlyWeather:
    temperature_f: float | None
    wind_speed_mph: float | None
    wind_direction_deg: float | None
    humidity_pct: float | None


def normalize_open_meteo_payload(
    payload: dict[str, Any],
    *,
    target_time: datetime,
) -> _HourlyWeather:
    """Pick the hourly observation closest to ``target_time``.

    Raises ``ValueError`` when the payload is missing the hourly block or
    the series are length-mismatched — those are server-shape changes we
    want to surface rather than silently ingest as nulls.
    """

    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise ValueError("open-meteo payload is missing the 'hourly' block")
    times = hourly.get("time")
    if not isinstance(times, list) or not times:
        raise ValueError("open-meteo payload has no 'hourly.time' entries")

    series_keys = (
        "temperature_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "relative_humidity_2m",
    )
    for key in series_keys:
        series = hourly.get(key)
        if not isinstance(series, list) or len(series) != len(times):
            raise ValueError(
                f"open-meteo payload series '{key}' length does not match "
                "'hourly.time'"
            )

    hours = [_parse_open_meteo_hour(str(value)) for value in times]
    index = _nearest_hour_index(hours, target_time)
    return _HourlyWeather(
        temperature_f=_coerce_optional_float(hourly["temperature_2m"][index]),
        wind_speed_mph=_coerce_optional_float(hourly["wind_speed_10m"][index]),
        wind_direction_deg=_coerce_optional_float(
            hourly["wind_direction_10m"][index]
        ),
        humidity_pct=_coerce_optional_float(
            hourly["relative_humidity_2m"][index]
        ),
    )


def _latest_mlb_games_path(
    *,
    output_dir: Path | str,
    target_date: date,
) -> Path:
    normalized_root = (
        Path(output_dir)
        / "normalized"
        / "mlb_stats_api"
        / f"date={target_date.isoformat()}"
    )
    run_dirs = sorted(path for path in normalized_root.glob("run=*") if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(
            "No normalized MLB metadata runs were found. "
            "Run `uv run python -m mlb_props_stack ingest-mlb-metadata --date ...` first."
        )
    games_path = run_dirs[-1] / "games.jsonl"
    if not games_path.exists():
        raise FileNotFoundError(
            f"Expected MLB metadata games.jsonl in {run_dirs[-1]}, but it was missing."
        )
    return games_path


def _load_games(games_path: Path) -> tuple[GameRecord, ...]:
    rows: list[GameRecord] = []
    for line in games_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(
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
        )
    return tuple(rows)


def _snapshot_id(*, game_pk: int, captured_at: datetime) -> str:
    return f"weather:{game_pk}:{_path_timestamp(captured_at)}"


def _make_skipped_snapshot(
    *,
    game: GameRecord,
    venue: VenueMetadata | None,
    target_time: datetime,
    captured_at: datetime,
    status: str,
    error_message: str | None,
) -> WeatherSnapshotRecord:
    return WeatherSnapshotRecord(
        weather_snapshot_id=_snapshot_id(
            game_pk=game.game_pk, captured_at=captured_at
        ),
        official_date=game.official_date,
        game_pk=game.game_pk,
        venue_id=game.venue_id,
        venue_name=game.venue_name,
        roof_type=venue.roof_type if venue is not None else None,
        latitude=venue.latitude if venue is not None else None,
        longitude=venue.longitude if venue is not None else None,
        commence_time=game.commence_time,
        target_time=target_time,
        captured_at=captured_at,
        weather_source=None,
        weather_status=status,
        temperature_f=None,
        wind_speed_mph=None,
        wind_direction_deg=None,
        humidity_pct=None,
        error_message=error_message,
    )


def ingest_weather_for_date(
    *,
    target_date: date,
    output_dir: Path | str = "data",
    client: OpenMeteoClient | None = None,
    now: Callable[[], datetime] = utc_now,
    target_offset_minutes: int = DEFAULT_TARGET_OFFSET_MINUTES,
    venue_lookup: dict[int, VenueMetadata] | None = None,
) -> WeatherIngestResult:
    """Fetch one pregame weather snapshot per scheduled game on ``target_date``.

    Parameters
    ----------
    target_date
        MLB official slate date to ingest.
    output_dir
        Filesystem root under which ``raw/weather`` and
        ``normalized/weather`` are written. Defaults to ``data``.
    client
        Optional Open-Meteo HTTP client override for tests.
    now
        Optional UTC clock override for deterministic ``captured_at``
        stamps in tests.
    target_offset_minutes
        Minutes before ``commence_time`` to anchor the observation.
        Defaults to 60 per AGE-205.
    venue_lookup
        Optional pre-loaded venue-metadata table. Defaults to the static
        CSV shipped under ``data/static/venues/``.

    Notes
    -----
    Each normalized row enforces ``captured_at <= commence_time``. Fixed-roof
    stadiums skip the network fetch and emit a sentinel
    ``weather_status="roof_closed"`` row. Venues missing from the
    metadata lookup emit ``weather_status="missing_venue_metadata"`` so
    the slate row is still represented in ``weather_snapshots.jsonl``
    and downstream coverage checks can flag the gap.
    """

    if target_offset_minutes < 0:
        raise ValueError("target_offset_minutes must be non-negative")

    if client is None:
        client = OpenMeteoClient()
    if venue_lookup is None:
        venue_lookup = load_venue_metadata()

    run_started_at = now().astimezone(UTC)
    run_id = _path_timestamp(run_started_at)
    output_root = Path(output_dir)
    games_path = _latest_mlb_games_path(
        output_dir=output_root, target_date=target_date
    )
    games = _load_games(games_path)

    snapshots: list[WeatherSnapshotRecord] = []
    raw_paths: list[Path] = []
    outdoor_count = 0
    roof_closed_count = 0
    missing_venue_count = 0
    missing_source_count = 0
    offset = timedelta(minutes=target_offset_minutes)

    for game in games:
        target_time = game.commence_time - offset
        venue = lookup_venue_metadata(game.venue_id, table=venue_lookup)

        if venue is None:
            captured_at = min(now().astimezone(UTC), game.commence_time)
            snapshots.append(
                _make_skipped_snapshot(
                    game=game,
                    venue=None,
                    target_time=target_time,
                    captured_at=captured_at,
                    status=WEATHER_STATUS_MISSING_VENUE_METADATA,
                    error_message=None,
                )
            )
            missing_venue_count += 1
            continue

        if venue.roof_type == ROOF_TYPE_FIXED:
            captured_at = min(now().astimezone(UTC), game.commence_time)
            snapshots.append(
                _make_skipped_snapshot(
                    game=game,
                    venue=venue,
                    target_time=target_time,
                    captured_at=captured_at,
                    status=WEATHER_STATUS_ROOF_CLOSED,
                    error_message=None,
                )
            )
            roof_closed_count += 1
            continue

        fetch_started_at = now().astimezone(UTC)
        source_url = build_open_meteo_archive_url(
            latitude=venue.latitude,
            longitude=venue.longitude,
            target_date=target_date,
        )
        raw_path = (
            output_root
            / "raw"
            / "weather"
            / f"date={target_date.isoformat()}"
            / f"venue_id={venue.venue_mlb_id}"
            / f"game_pk={game.game_pk}"
            / f"captured_at={_path_timestamp(fetch_started_at)}.json"
        )
        try:
            payload = client.fetch_json(source_url)
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            captured_at = min(fetch_started_at, game.commence_time)
            snapshots.append(
                _make_skipped_snapshot(
                    game=game,
                    venue=venue,
                    target_time=target_time,
                    captured_at=captured_at,
                    status=WEATHER_STATUS_MISSING_SOURCE,
                    error_message=f"{type(exc).__name__}: {exc}",
                )
            )
            missing_source_count += 1
            continue

        _write_json(raw_path, payload)
        raw_paths.append(raw_path)

        try:
            hourly = normalize_open_meteo_payload(payload, target_time=target_time)
        except ValueError as exc:
            captured_at = min(fetch_started_at, game.commence_time)
            snapshots.append(
                _make_skipped_snapshot(
                    game=game,
                    venue=venue,
                    target_time=target_time,
                    captured_at=captured_at,
                    status=WEATHER_STATUS_MISSING_SOURCE,
                    error_message=f"{type(exc).__name__}: {exc}",
                )
            )
            missing_source_count += 1
            continue

        captured_at = min(fetch_started_at, game.commence_time)
        snapshots.append(
            WeatherSnapshotRecord(
                weather_snapshot_id=_snapshot_id(
                    game_pk=game.game_pk, captured_at=captured_at
                ),
                official_date=game.official_date,
                game_pk=game.game_pk,
                venue_id=game.venue_id,
                venue_name=game.venue_name,
                roof_type=venue.roof_type,
                latitude=venue.latitude,
                longitude=venue.longitude,
                commence_time=game.commence_time,
                target_time=target_time,
                captured_at=captured_at,
                weather_source=WEATHER_SOURCE_OPEN_METEO_ARCHIVE,
                weather_status=WEATHER_STATUS_OK,
                temperature_f=hourly.temperature_f,
                wind_speed_mph=hourly.wind_speed_mph,
                wind_direction_deg=hourly.wind_direction_deg,
                humidity_pct=hourly.humidity_pct,
                error_message=None,
            )
        )
        outdoor_count += 1

    for snapshot in snapshots:
        if snapshot.captured_at > snapshot.commence_time:
            raise AssertionError(
                "weather snapshot captured_at must be <= commence_time "
                f"(game_pk={snapshot.game_pk})"
            )

    normalized_root = (
        output_root
        / "normalized"
        / "weather"
        / f"date={target_date.isoformat()}"
        / f"run={run_id}"
    )
    weather_snapshots_path = normalized_root / "weather_snapshots.jsonl"
    _write_jsonl(weather_snapshots_path, snapshots)

    return WeatherIngestResult(
        target_date=target_date,
        run_id=run_id,
        mlb_games_path=games_path,
        weather_snapshots_path=weather_snapshots_path,
        raw_snapshot_paths=tuple(raw_paths),
        snapshot_count=len(snapshots),
        outdoor_snapshot_count=outdoor_count,
        roof_closed_snapshot_count=roof_closed_count,
        missing_venue_metadata_count=missing_venue_count,
        missing_source_count=missing_source_count,
    )


def load_latest_weather_snapshots_for_date(
    *,
    output_dir: Path | str,
    target_date: date,
) -> dict[int, WeatherSnapshotRecord]:
    """Return the most recent pregame weather snapshots for ``target_date``.

    Keys by ``game_pk`` so the statcast feature builder can join one
    snapshot per game. Missing directories return an empty dict so
    downstream code falls back to ``missing_weather_source`` sentinels
    without crashing on fresh slates that haven't been ingested yet.
    """

    normalized_root = (
        Path(output_dir)
        / "normalized"
        / "weather"
        / f"date={target_date.isoformat()}"
    )
    if not normalized_root.exists():
        return {}
    run_dirs = sorted(path for path in normalized_root.glob("run=*") if path.is_dir())
    if not run_dirs:
        return {}
    snapshots_path = run_dirs[-1] / "weather_snapshots.jsonl"
    if not snapshots_path.exists():
        return {}

    records: dict[int, WeatherSnapshotRecord] = {}
    for line in snapshots_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        captured_at = parse_api_datetime(row["captured_at"])
        records[row["game_pk"]] = WeatherSnapshotRecord(
            weather_snapshot_id=row["weather_snapshot_id"],
            official_date=row["official_date"],
            game_pk=row["game_pk"],
            venue_id=row.get("venue_id"),
            venue_name=row.get("venue_name", ""),
            roof_type=row.get("roof_type"),
            latitude=row.get("latitude"),
            longitude=row.get("longitude"),
            commence_time=parse_api_datetime(row["commence_time"]),
            target_time=parse_api_datetime(row["target_time"]),
            captured_at=captured_at,
            weather_source=row.get("weather_source"),
            weather_status=row["weather_status"],
            temperature_f=row.get("temperature_f"),
            wind_speed_mph=row.get("wind_speed_mph"),
            wind_direction_deg=row.get("wind_direction_deg"),
            humidity_pct=row.get("humidity_pct"),
            error_message=row.get("error_message"),
        )
    return records
