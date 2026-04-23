from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from mlb_props_stack.ingest import (
    DEFAULT_UMPIRE_HISTORY_DAYS,
    UMPIRE_SOURCE_MLB_STATS_API_FEED_LIVE,
    UMPIRE_STATUS_MISSING_SOURCE,
    UMPIRE_STATUS_OK,
    ingest_umpire_for_date,
    load_latest_umpire_snapshots_for_date,
    normalize_feed_live_officials_payload,
)
from mlb_props_stack.ingest.umpire import (
    _DailyPitchAggregates,
    compute_rolling_umpire_metrics,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_game_row(
    *,
    game_pk: int,
    venue_id: int = 5,
    venue_name: str = "Progressive Field",
    commence_time: str = "2026-04-21T22:10:00Z",
    home_team_abbreviation: str = "CLE",
    away_team_abbreviation: str = "HOU",
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
        "home_team_abbreviation": home_team_abbreviation,
        "home_team_name": "Cleveland Guardians",
        "away_team_id": 117,
        "away_team_abbreviation": away_team_abbreviation,
        "away_team_name": "Houston Astros",
        "game_number": 1,
        "double_header": "N",
        "day_night": "night",
        "odds_matchup_key": (
            f"2026-04-21|{away_team_abbreviation}|{home_team_abbreviation}|"
            f"{commence_time}"
        ),
    }


def _seed_games(output_dir: Path, rows: list[dict]) -> Path:
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


def _seed_persisted_feed_live(
    output_dir: Path,
    *,
    game_pk: int,
    officials: list[dict] | None,
    target_date: date = date(2026, 4, 21),
    captured_at_stamp: str = "20260421T170500Z",
) -> Path:
    payload = {
        "liveData": {
            "boxscore": {
                "officials": officials if officials is not None else [],
            }
        }
    }
    path = (
        output_dir
        / "raw"
        / "mlb_stats_api"
        / f"date={target_date.isoformat()}"
        / "feed_live"
        / f"game_pk={game_pk}"
        / f"captured_at={captured_at_stamp}.json"
    )
    _write_json(path, payload)
    return path


def _home_plate_official(*, umpire_id: int = 427395, umpire_name: str = "Ed Hickox") -> dict:
    return {
        "official": {"id": umpire_id, "fullName": umpire_name, "link": f"/api/v1/people/{umpire_id}"},
        "officialType": "Home Plate",
    }


class _StubMLBStatsAPIClient:
    def __init__(self, payloads_by_game_pk: dict[int, dict] | None = None) -> None:
        self.urls: list[str] = []
        self._payloads = payloads_by_game_pk or {}

    def fetch_json(self, url: str) -> dict:
        self.urls.append(url)
        for game_pk, payload in self._payloads.items():
            if f"/game/{game_pk}/" in url:
                return payload
        return {"liveData": {"boxscore": {"officials": []}}}


def test_normalize_feed_live_officials_payload_extracts_array() -> None:
    payload = {
        "liveData": {
            "boxscore": {
                "officials": [
                    _home_plate_official(),
                    {"official": {"id": 1, "fullName": "First Base"}, "officialType": "First Base"},
                ]
            }
        }
    }
    officials = normalize_feed_live_officials_payload(payload)
    assert isinstance(officials, list)
    assert len(officials) == 2


def test_normalize_feed_live_officials_payload_returns_none_when_missing() -> None:
    assert normalize_feed_live_officials_payload({}) is None
    assert normalize_feed_live_officials_payload({"liveData": {}}) is None
    assert (
        normalize_feed_live_officials_payload({"liveData": {"boxscore": {}}}) is None
    )


def test_ingest_umpire_happy_path_uses_persisted_feed_live(tmp_path: Path) -> None:
    _seed_games(tmp_path, [_build_game_row(game_pk=824448)])
    _seed_persisted_feed_live(
        tmp_path,
        game_pk=824448,
        officials=[_home_plate_official(umpire_id=427395, umpire_name="Ed Hickox")],
    )

    client = _StubMLBStatsAPIClient()
    fixed_now = datetime(2026, 4, 21, 18, 0, tzinfo=UTC)
    result = ingest_umpire_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        client=client,
        now=lambda: fixed_now,
    )

    # No HTTP fetch — the persisted feed/live payload is sufficient.
    assert client.urls == []
    assert result.snapshot_count == 1
    assert result.ok_snapshot_count == 1
    assert result.missing_source_count == 0

    snapshots = [
        json.loads(line)
        for line in result.umpire_snapshots_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot["game_pk"] == 824448
    assert snapshot["umpire_id"] == 427395
    assert snapshot["umpire_name"] == "Ed Hickox"
    assert snapshot["umpire_status"] == UMPIRE_STATUS_OK
    assert snapshot["umpire_source"] == UMPIRE_SOURCE_MLB_STATS_API_FEED_LIVE
    # No prior-day umpire/statcast history is seeded, so the rolling
    # metrics degrade to None rather than imputing.
    assert snapshot["ump_called_strike_rate_30d"] is None
    assert snapshot["ump_k_per_9_delta_vs_league_30d"] is None

    # Raw artifact is written with the extracted officials payload.
    assert len(result.raw_snapshot_paths) == 1
    raw = json.loads(result.raw_snapshot_paths[0].read_text(encoding="utf-8"))
    assert raw["game_pk"] == 824448
    assert raw["umpire_id"] == 427395
    assert raw["umpire_status"] == UMPIRE_STATUS_OK
    assert isinstance(raw["officials"], list) and raw["officials"]


def test_ingest_umpire_falls_back_to_http_when_no_persisted_payload(tmp_path: Path) -> None:
    _seed_games(tmp_path, [_build_game_row(game_pk=824449)])
    client = _StubMLBStatsAPIClient(
        payloads_by_game_pk={
            824449: {
                "liveData": {
                    "boxscore": {
                        "officials": [
                            _home_plate_official(umpire_id=500001, umpire_name="Alfonso Marquez"),
                        ],
                    }
                }
            }
        }
    )
    fixed_now = datetime(2026, 4, 21, 18, 0, tzinfo=UTC)

    result = ingest_umpire_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        client=client,
        now=lambda: fixed_now,
    )

    assert len(client.urls) == 1
    assert "/game/824449/" in client.urls[0]
    assert result.ok_snapshot_count == 1

    snapshot = json.loads(
        result.umpire_snapshots_path.read_text(encoding="utf-8").splitlines()[0]
    )
    assert snapshot["umpire_id"] == 500001
    assert snapshot["umpire_source"] == UMPIRE_SOURCE_MLB_STATS_API_FEED_LIVE


def test_ingest_umpire_missing_home_plate_emits_sentinel(tmp_path: Path) -> None:
    _seed_games(tmp_path, [_build_game_row(game_pk=824450)])
    # Persisted feed/live exists but has no Home Plate entry (assignment
    # has not been published yet).
    _seed_persisted_feed_live(
        tmp_path,
        game_pk=824450,
        officials=[
            {
                "official": {"id": 1, "fullName": "First Base Ump"},
                "officialType": "First Base",
            }
        ],
    )

    result = ingest_umpire_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        client=_StubMLBStatsAPIClient(),
        now=lambda: datetime(2026, 4, 21, 18, 0, tzinfo=UTC),
    )

    assert result.missing_source_count == 1
    assert result.ok_snapshot_count == 0

    snapshot = json.loads(
        result.umpire_snapshots_path.read_text(encoding="utf-8").splitlines()[0]
    )
    assert snapshot["umpire_status"] == UMPIRE_STATUS_MISSING_SOURCE
    assert snapshot["umpire_source"] is None
    assert snapshot["umpire_id"] is None
    assert snapshot["umpire_name"] is None
    assert snapshot["ump_called_strike_rate_30d"] is None
    assert snapshot["ump_k_per_9_delta_vs_league_30d"] is None


def test_ingest_umpire_http_error_emits_sentinel(tmp_path: Path) -> None:
    _seed_games(tmp_path, [_build_game_row(game_pk=824451)])

    class _FailingClient:
        def __init__(self) -> None:
            self.urls: list[str] = []

        def fetch_json(self, url: str) -> dict:
            self.urls.append(url)
            raise TimeoutError("simulated feed/live timeout")

    client = _FailingClient()
    result = ingest_umpire_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        client=client,
        now=lambda: datetime(2026, 4, 21, 18, 0, tzinfo=UTC),
    )

    assert client.urls, "HTTP fallback must be attempted when no persisted feed/live exists"
    assert result.missing_source_count == 1
    snapshot = json.loads(
        result.umpire_snapshots_path.read_text(encoding="utf-8").splitlines()[0]
    )
    assert snapshot["umpire_status"] == UMPIRE_STATUS_MISSING_SOURCE
    assert "simulated feed/live timeout" in (snapshot["error_message"] or "")


def test_ingest_umpire_enforces_captured_at_leakage_guard(tmp_path: Path) -> None:
    # commence_time earlier than ``now`` → captured_at must be clamped to
    # commence_time so downstream features cannot leak post-pitch data.
    _seed_games(
        tmp_path,
        [
            _build_game_row(
                game_pk=824452,
                commence_time="2026-04-21T18:00:00Z",
            )
        ],
    )
    _seed_persisted_feed_live(
        tmp_path,
        game_pk=824452,
        officials=[_home_plate_official(umpire_id=427395, umpire_name="Ed Hickox")],
    )

    post_pitch_now = datetime(2026, 4, 21, 22, 0, tzinfo=UTC)
    result = ingest_umpire_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        client=_StubMLBStatsAPIClient(),
        now=lambda: post_pitch_now,
    )

    snapshot = json.loads(
        result.umpire_snapshots_path.read_text(encoding="utf-8").splitlines()[0]
    )
    captured_at = datetime.fromisoformat(snapshot["captured_at"].replace("Z", "+00:00"))
    commence_time = datetime.fromisoformat(snapshot["commence_time"].replace("Z", "+00:00"))
    assert captured_at <= commence_time


def test_ingest_umpire_rejects_non_positive_history_days(tmp_path: Path) -> None:
    _seed_games(tmp_path, [_build_game_row(game_pk=824453)])
    with pytest.raises(ValueError):
        ingest_umpire_for_date(
            target_date=date(2026, 4, 21),
            output_dir=tmp_path,
            client=_StubMLBStatsAPIClient(),
            history_days=0,
        )


def test_ingest_umpire_joins_rolling_metrics_from_prior_runs(tmp_path: Path) -> None:
    """End-to-end: a prior umpire snapshot + prior pitch_level_base should
    let the current run compute ``ump_called_strike_rate_30d`` and
    ``ump_k_per_9_delta_vs_league_30d`` that are strictly below 1.0 and
    differ from zero.
    """

    target_date = date(2026, 4, 21)
    prior_date = date(2026, 4, 20)
    # Current-day game assigned to the same umpire we saw the day before.
    _seed_games(tmp_path, [_build_game_row(game_pk=900000)])
    _seed_persisted_feed_live(
        tmp_path,
        game_pk=900000,
        officials=[_home_plate_official(umpire_id=427395, umpire_name="Ed Hickox")],
    )

    prior_umpire_run = (
        tmp_path
        / "normalized"
        / "umpire"
        / f"date={prior_date.isoformat()}"
        / "run=20260420T180000Z"
    )
    # The prior umpire call a different game assigned to the same umpire.
    prior_game_pk = 800000
    _write_jsonl(
        prior_umpire_run / "umpire_snapshots.jsonl",
        [
            {
                "umpire_snapshot_id": f"umpire-snapshot:{prior_game_pk}:20260420T180000Z",
                "official_date": prior_date.isoformat(),
                "game_pk": prior_game_pk,
                "commence_time": "2026-04-20T22:10:00Z",
                "captured_at": "2026-04-20T18:00:00Z",
                "umpire_source": UMPIRE_SOURCE_MLB_STATS_API_FEED_LIVE,
                "umpire_status": UMPIRE_STATUS_OK,
                "umpire_id": 427395,
                "umpire_name": "Ed Hickox",
                "error_message": None,
                "history_start_date": (prior_date - timedelta(days=30)).isoformat(),
                "history_end_date": (prior_date - timedelta(days=1)).isoformat(),
                "ump_called_strike_rate_30d": None,
                "ump_k_per_9_delta_vs_league_30d": None,
            }
        ],
    )

    # Seed prior-day pitch_level_base: two games on the same day, one of
    # which is the game the umpire called. Give the umpire's game a
    # markedly higher called-strike rate and slightly higher K rate so the
    # delta is non-zero.
    prior_statcast_run = (
        tmp_path
        / "normalized"
        / "statcast_search"
        / f"date={prior_date.isoformat()}"
        / "run=20260420T200000Z"
    )
    rows: list[dict] = []
    # 20 pitches in umpire's game: 10 called strikes, 5 plate-appearance
    # finals, 2 strikeouts.
    for pitch_index in range(20):
        rows.append(
            {
                "pitch_record_id": f"pitch:{prior_game_pk}:{pitch_index}",
                "game_pk": prior_game_pk,
                "is_called_strike": pitch_index < 10,
                "is_plate_appearance_final_pitch": pitch_index < 5,
                "is_strikeout_event": pitch_index < 2,
            }
        )
    # 20 pitches in a non-umpire game: 2 called strikes, 5 plate-appearance
    # finals, 1 strikeout.
    other_game_pk = 800001
    for pitch_index in range(20):
        rows.append(
            {
                "pitch_record_id": f"pitch:{other_game_pk}:{pitch_index}",
                "game_pk": other_game_pk,
                "is_called_strike": pitch_index < 2,
                "is_plate_appearance_final_pitch": pitch_index < 5,
                "is_strikeout_event": pitch_index < 1,
            }
        )
    _write_jsonl(prior_statcast_run / "pitch_level_base.jsonl", rows)

    result = ingest_umpire_for_date(
        target_date=target_date,
        output_dir=tmp_path,
        client=_StubMLBStatsAPIClient(),
        now=lambda: datetime(2026, 4, 21, 18, 0, tzinfo=UTC),
    )

    snapshot = json.loads(
        result.umpire_snapshots_path.read_text(encoding="utf-8").splitlines()[0]
    )
    assert snapshot["ump_called_strike_rate_30d"] == pytest.approx(10 / 20)
    # Umpire K rate: 2/5 = 0.4; league K rate: 3/10 = 0.3.
    # delta * 38.25 = 0.1 * 38.25 = 3.825.
    assert snapshot["ump_k_per_9_delta_vs_league_30d"] == pytest.approx(3.825)


def test_compute_rolling_umpire_metrics_returns_none_when_no_prior_data() -> None:
    rate, delta = compute_rolling_umpire_metrics(
        umpire_id=1,
        umpire_game_pks=None,
        pitch_aggregates={},
    )
    assert rate is None
    assert delta is None


def test_compute_rolling_umpire_metrics_returns_none_when_umpire_has_no_overlap() -> None:
    aggregates = {
        date(2026, 4, 20): _DailyPitchAggregates(
            total_pitches=10,
            called_strikes=5,
            plate_appearances=3,
            strikeouts=1,
            by_game_pk={},
        )
    }
    rate, delta = compute_rolling_umpire_metrics(
        umpire_id=1,
        umpire_game_pks={date(2026, 4, 20): {999}},
        pitch_aggregates=aggregates,
    )
    assert rate is None
    assert delta is None


def test_compute_rolling_umpire_metrics_returns_rate_when_plate_appearances_missing() -> None:
    # Umpire's games have pitches but no PA finals (e.g. suspended game or
    # stale pitch_level_base). Called-strike rate is still computable;
    # K/9 delta degrades to None rather than dragging rate down with it.
    aggregates = {
        date(2026, 4, 20): _DailyPitchAggregates(
            total_pitches=30,
            called_strikes=12,
            plate_appearances=5,
            strikeouts=2,
            by_game_pk={
                500: (20, 8, 0, 0),
                501: (10, 4, 5, 2),
            },
        )
    }
    rate, delta = compute_rolling_umpire_metrics(
        umpire_id=1,
        umpire_game_pks={date(2026, 4, 20): {500}},
        pitch_aggregates=aggregates,
    )
    assert rate == pytest.approx(8 / 20)
    assert delta is None


def test_load_latest_umpire_snapshots_for_date_returns_empty_when_missing(
    tmp_path: Path,
) -> None:
    assert (
        load_latest_umpire_snapshots_for_date(
            output_dir=tmp_path, target_date=date(2026, 4, 21)
        )
        == {}
    )


def test_load_latest_umpire_snapshots_for_date_returns_latest_run(tmp_path: Path) -> None:
    _seed_games(tmp_path, [_build_game_row(game_pk=900100)])
    _seed_persisted_feed_live(
        tmp_path,
        game_pk=900100,
        officials=[_home_plate_official(umpire_id=427395, umpire_name="Ed Hickox")],
    )
    ingest_umpire_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        client=_StubMLBStatsAPIClient(),
        now=lambda: datetime(2026, 4, 21, 18, 0, tzinfo=UTC),
    )

    table = load_latest_umpire_snapshots_for_date(
        output_dir=tmp_path, target_date=date(2026, 4, 21)
    )
    assert 900100 in table
    record = table[900100]
    assert record.umpire_status == UMPIRE_STATUS_OK
    assert record.umpire_id == 427395
    assert record.umpire_name == "Ed Hickox"


def test_default_umpire_history_days_matches_age_204_spec() -> None:
    assert DEFAULT_UMPIRE_HISTORY_DAYS == 30
