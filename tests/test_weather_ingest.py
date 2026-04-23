from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from mlb_props_stack.ingest import (
    ROOF_TYPE_FIXED,
    ROOF_TYPE_OPEN,
    ROOF_TYPE_RETRACTABLE,
    WEATHER_SOURCE_OPEN_METEO_ARCHIVE,
    WEATHER_STATUS_MISSING_SOURCE,
    WEATHER_STATUS_MISSING_VENUE_METADATA,
    WEATHER_STATUS_OK,
    WEATHER_STATUS_ROOF_CLOSED,
    VenueMetadata,
    build_open_meteo_archive_url,
    ingest_weather_for_date,
    load_latest_weather_snapshots_for_date,
    load_venue_metadata,
    lookup_venue_metadata,
    normalize_open_meteo_payload,
)


PROGRESSIVE_VENUE = VenueMetadata(
    venue_mlb_id=5,
    venue_name="Progressive Field",
    latitude=41.4962,
    longitude=-81.6852,
    roof_type=ROOF_TYPE_OPEN,
)
TROPICANA_VENUE = VenueMetadata(
    venue_mlb_id=12,
    venue_name="Tropicana Field",
    latitude=27.76778,
    longitude=-82.6525,
    roof_type=ROOF_TYPE_FIXED,
)
CHASE_VENUE = VenueMetadata(
    venue_mlb_id=15,
    venue_name="Chase Field",
    latitude=33.4453,
    longitude=-112.06669,
    roof_type=ROOF_TYPE_RETRACTABLE,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def _build_game_row(
    *,
    game_pk: int,
    venue_id: int,
    venue_name: str,
    commence_time: str = "2026-04-21T22:10:00Z",
) -> dict:
    return {
        "game_pk": game_pk,
        "official_date": "2026-04-21",
        "commence_time": commence_time,
        "captured_at": "2026-04-21T17:00:00Z",
        "status": "Pre-Game",
        "status_code": "P",
        "venue_id": venue_id,
        "venue_name": venue_name,
        "home_team_id": 114,
        "home_team_abbreviation": "CLE",
        "home_team_name": "Cleveland Guardians",
        "away_team_id": 117,
        "away_team_abbreviation": "HOU",
        "away_team_name": "Houston Astros",
        "game_number": 1,
        "double_header": "N",
        "day_night": "night",
        "odds_matchup_key": "2026-04-21|HOU|CLE|2026-04-21T22:10:00Z",
    }


def seed_mlb_games(output_dir: Path, rows: list[dict]) -> Path:
    games_path = (
        output_dir
        / "normalized"
        / "mlb_stats_api"
        / "date=2026-04-21"
        / "run=20260421T170000Z"
        / "games.jsonl"
    )
    _write_jsonl(games_path, rows)
    return games_path


class _StubOpenMeteoClient:
    def __init__(self, payload: dict | None = None, error: Exception | None = None) -> None:
        self.urls: list[str] = []
        self._payload = payload or _sample_payload()
        self._error = error

    def fetch_json(self, url: str) -> dict:
        self.urls.append(url)
        if self._error is not None:
            raise self._error
        return self._payload


def _sample_payload() -> dict:
    # Open-Meteo returns 24 hourly entries for a single-day request; pick values
    # that let the nearest-hour selector resolve deterministically.
    base = datetime(2026, 4, 21, 0, 0, tzinfo=UTC)
    times = [(base + timedelta(hours=hour)).strftime("%Y-%m-%dT%H:%M") for hour in range(24)]
    temperature = [60.0 + hour for hour in range(24)]
    wind_speed = [5.0 + hour * 0.1 for hour in range(24)]
    wind_direction = [180 + hour for hour in range(24)]
    humidity = [40 + hour for hour in range(24)]
    return {
        "hourly": {
            "time": times,
            "temperature_2m": temperature,
            "wind_speed_10m": wind_speed,
            "wind_direction_10m": wind_direction,
            "relative_humidity_2m": humidity,
        }
    }


def _venue_lookup() -> dict[int, VenueMetadata]:
    return {
        PROGRESSIVE_VENUE.venue_mlb_id: PROGRESSIVE_VENUE,
        TROPICANA_VENUE.venue_mlb_id: TROPICANA_VENUE,
        CHASE_VENUE.venue_mlb_id: CHASE_VENUE,
    }


def test_default_venue_metadata_csv_loads() -> None:
    table = load_venue_metadata()
    assert table, "static venue metadata csv must be non-empty"
    progressive = table[5]
    assert progressive.venue_name == "Progressive Field"
    assert progressive.roof_type == ROOF_TYPE_OPEN

    tropicana = table[12]
    assert tropicana.venue_name == "Tropicana Field"
    assert tropicana.roof_type == ROOF_TYPE_FIXED

    chase = table[15]
    assert chase.venue_name == "Chase Field"
    assert chase.roof_type == ROOF_TYPE_RETRACTABLE


def test_lookup_venue_metadata_returns_none_when_id_unknown() -> None:
    table = _venue_lookup()
    assert lookup_venue_metadata(999999, table=table) is None
    assert lookup_venue_metadata(None, table=table) is None


def test_build_open_meteo_archive_url_includes_required_query_params() -> None:
    url = build_open_meteo_archive_url(
        latitude=41.4962,
        longitude=-81.6852,
        target_date=date(2026, 4, 21),
    )
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    assert parsed.netloc == "archive-api.open-meteo.com"
    assert params["latitude"] == ["41.4962"]
    assert params["longitude"] == ["-81.6852"]
    assert params["start_date"] == ["2026-04-21"]
    assert params["end_date"] == ["2026-04-21"]
    assert params["temperature_unit"] == ["fahrenheit"]
    assert params["wind_speed_unit"] == ["mph"]
    assert params["timezone"] == ["UTC"]
    assert "temperature_2m" in params["hourly"][0]


def test_normalize_open_meteo_payload_picks_hour_closest_to_target() -> None:
    payload = _sample_payload()
    # commence_time=22:10Z → target_time = 21:10Z; nearest hour is 21:00Z → index 21
    hourly = normalize_open_meteo_payload(
        payload, target_time=datetime(2026, 4, 21, 21, 10, tzinfo=UTC)
    )
    assert hourly.temperature_f == 60.0 + 21
    assert hourly.wind_speed_mph == pytest.approx(5.0 + 21 * 0.1)
    assert hourly.wind_direction_deg == 180 + 21
    assert hourly.humidity_pct == 40 + 21


def test_normalize_open_meteo_payload_rejects_missing_hourly_block() -> None:
    with pytest.raises(ValueError):
        normalize_open_meteo_payload({}, target_time=datetime.now(tz=UTC))


def test_normalize_open_meteo_payload_rejects_length_mismatched_series() -> None:
    payload = _sample_payload()
    payload["hourly"]["temperature_2m"] = payload["hourly"]["temperature_2m"][:10]
    with pytest.raises(ValueError):
        normalize_open_meteo_payload(
            payload, target_time=datetime(2026, 4, 21, 21, 10, tzinfo=UTC)
        )


def test_ingest_weather_for_date_outdoor_happy_path(tmp_path: Path) -> None:
    seed_mlb_games(
        tmp_path,
        [_build_game_row(game_pk=824448, venue_id=5, venue_name="Progressive Field")],
    )

    client = _StubOpenMeteoClient()
    fixed_now = datetime(2026, 4, 21, 18, 0, tzinfo=UTC)

    result = ingest_weather_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        client=client,
        now=lambda: fixed_now,
        venue_lookup=_venue_lookup(),
    )

    assert result.snapshot_count == 1
    assert result.outdoor_snapshot_count == 1
    assert result.roof_closed_snapshot_count == 0
    assert result.missing_venue_metadata_count == 0
    assert result.missing_source_count == 0

    snapshots = [
        json.loads(line)
        for line in result.weather_snapshots_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot["game_pk"] == 824448
    assert snapshot["weather_status"] == WEATHER_STATUS_OK
    assert snapshot["weather_source"] == WEATHER_SOURCE_OPEN_METEO_ARCHIVE
    assert snapshot["roof_type"] == ROOF_TYPE_OPEN
    assert snapshot["temperature_f"] == 60.0 + 21
    assert snapshot["wind_speed_mph"] == pytest.approx(5.0 + 21 * 0.1)
    assert snapshot["wind_direction_deg"] == 180 + 21
    assert snapshot["humidity_pct"] == 40 + 21
    assert snapshot["latitude"] == pytest.approx(41.4962)
    assert snapshot["longitude"] == pytest.approx(-81.6852)

    # Raw payload archived under raw/weather/...
    assert len(result.raw_snapshot_paths) == 1
    raw_payload = json.loads(result.raw_snapshot_paths[0].read_text(encoding="utf-8"))
    assert raw_payload["hourly"]["temperature_2m"]

    # One URL hit — outdoor venue only.
    assert len(client.urls) == 1


def test_ingest_weather_for_date_fixed_roof_emits_sentinel(tmp_path: Path) -> None:
    seed_mlb_games(
        tmp_path,
        [_build_game_row(game_pk=111111, venue_id=12, venue_name="Tropicana Field")],
    )

    client = _StubOpenMeteoClient()
    result = ingest_weather_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        client=client,
        now=lambda: datetime(2026, 4, 21, 18, 0, tzinfo=UTC),
        venue_lookup=_venue_lookup(),
    )

    assert result.roof_closed_snapshot_count == 1
    assert result.outdoor_snapshot_count == 0
    assert client.urls == [], "fixed-roof games must not hit the weather source"

    snapshot = json.loads(result.weather_snapshots_path.read_text(encoding="utf-8").splitlines()[0])
    assert snapshot["weather_status"] == WEATHER_STATUS_ROOF_CLOSED
    assert snapshot["weather_source"] is None
    assert snapshot["roof_type"] == ROOF_TYPE_FIXED
    assert snapshot["temperature_f"] is None
    assert snapshot["wind_speed_mph"] is None
    assert snapshot["humidity_pct"] is None


def test_ingest_weather_for_date_retractable_roof_is_treated_as_outdoor(tmp_path: Path) -> None:
    seed_mlb_games(
        tmp_path,
        [_build_game_row(game_pk=222222, venue_id=15, venue_name="Chase Field")],
    )

    client = _StubOpenMeteoClient()
    result = ingest_weather_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        client=client,
        now=lambda: datetime(2026, 4, 21, 18, 0, tzinfo=UTC),
        venue_lookup=_venue_lookup(),
    )

    assert result.outdoor_snapshot_count == 1
    assert result.roof_closed_snapshot_count == 0

    snapshot = json.loads(result.weather_snapshots_path.read_text(encoding="utf-8").splitlines()[0])
    assert snapshot["weather_status"] == WEATHER_STATUS_OK
    assert snapshot["roof_type"] == ROOF_TYPE_RETRACTABLE


def test_ingest_weather_for_date_missing_venue_metadata(tmp_path: Path) -> None:
    seed_mlb_games(
        tmp_path,
        [_build_game_row(game_pk=333333, venue_id=999999, venue_name="Unknown Park")],
    )

    client = _StubOpenMeteoClient()
    result = ingest_weather_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        client=client,
        now=lambda: datetime(2026, 4, 21, 18, 0, tzinfo=UTC),
        venue_lookup=_venue_lookup(),
    )

    assert result.missing_venue_metadata_count == 1
    assert result.outdoor_snapshot_count == 0
    assert client.urls == [], "unknown venues must not hit the weather source"

    snapshot = json.loads(result.weather_snapshots_path.read_text(encoding="utf-8").splitlines()[0])
    assert snapshot["weather_status"] == WEATHER_STATUS_MISSING_VENUE_METADATA
    assert snapshot["roof_type"] is None
    assert snapshot["latitude"] is None
    assert snapshot["longitude"] is None


def test_ingest_weather_for_date_source_error_emits_missing_source(tmp_path: Path) -> None:
    seed_mlb_games(
        tmp_path,
        [_build_game_row(game_pk=444444, venue_id=5, venue_name="Progressive Field")],
    )

    client = _StubOpenMeteoClient(error=TimeoutError("simulated archive timeout"))
    result = ingest_weather_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        client=client,
        now=lambda: datetime(2026, 4, 21, 18, 0, tzinfo=UTC),
        venue_lookup=_venue_lookup(),
    )

    assert result.missing_source_count == 1
    assert result.outdoor_snapshot_count == 0

    snapshot = json.loads(result.weather_snapshots_path.read_text(encoding="utf-8").splitlines()[0])
    assert snapshot["weather_status"] == WEATHER_STATUS_MISSING_SOURCE
    assert "simulated archive timeout" in (snapshot["error_message"] or "")
    assert snapshot["temperature_f"] is None


def test_ingest_weather_for_date_leakage_guard_caps_captured_at(tmp_path: Path) -> None:
    # commence_time earlier than ``now`` → captured_at must be clamped to
    # commence_time so downstream features cannot leak post-pitch data.
    seed_mlb_games(
        tmp_path,
        [
            _build_game_row(
                game_pk=555555,
                venue_id=5,
                venue_name="Progressive Field",
                commence_time="2026-04-21T18:00:00Z",
            )
        ],
    )

    client = _StubOpenMeteoClient()
    post_pitch_now = datetime(2026, 4, 21, 22, 0, tzinfo=UTC)
    result = ingest_weather_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        client=client,
        now=lambda: post_pitch_now,
        venue_lookup=_venue_lookup(),
    )

    snapshot = json.loads(result.weather_snapshots_path.read_text(encoding="utf-8").splitlines()[0])
    captured_at = datetime.fromisoformat(snapshot["captured_at"].replace("Z", "+00:00"))
    commence_time = datetime.fromisoformat(snapshot["commence_time"].replace("Z", "+00:00"))
    assert captured_at <= commence_time


def test_ingest_weather_for_date_rejects_negative_target_offset(tmp_path: Path) -> None:
    seed_mlb_games(
        tmp_path,
        [_build_game_row(game_pk=666666, venue_id=5, venue_name="Progressive Field")],
    )
    with pytest.raises(ValueError):
        ingest_weather_for_date(
            target_date=date(2026, 4, 21),
            output_dir=tmp_path,
            client=_StubOpenMeteoClient(),
            target_offset_minutes=-1,
            venue_lookup=_venue_lookup(),
        )


def test_load_latest_weather_snapshots_for_date_returns_empty_when_missing(
    tmp_path: Path,
) -> None:
    assert (
        load_latest_weather_snapshots_for_date(
            output_dir=tmp_path, target_date=date(2026, 4, 21)
        )
        == {}
    )


def test_load_latest_weather_snapshots_for_date_returns_latest_run(
    tmp_path: Path,
) -> None:
    seed_mlb_games(
        tmp_path,
        [_build_game_row(game_pk=777777, venue_id=5, venue_name="Progressive Field")],
    )
    ingest_weather_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        client=_StubOpenMeteoClient(),
        now=lambda: datetime(2026, 4, 21, 18, 0, tzinfo=UTC),
        venue_lookup=_venue_lookup(),
    )
    table = load_latest_weather_snapshots_for_date(
        output_dir=tmp_path, target_date=date(2026, 4, 21)
    )
    assert 777777 in table
    record = table[777777]
    assert record.weather_status == WEATHER_STATUS_OK
    assert record.roof_type == ROOF_TYPE_OPEN
    assert record.temperature_f is not None
