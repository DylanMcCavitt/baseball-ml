from __future__ import annotations

import csv
import json
import threading
from datetime import UTC, date, datetime, timedelta
from io import StringIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

import pytest

from mlb_props_stack.ingest import (
    StatcastFeatureIngestResult,
    build_statcast_search_csv_url,
    ingest_statcast_features_for_date,
)
from mlb_props_stack.ingest.statcast_features import (
    StatcastPitchRecord,
    StatcastSearchClient,
    _pitcher_hand_split_rates,
)


STATCAST_HEADERS = [
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
    "pitch_type",
    "pitch_name",
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "plate_x",
    "plate_z",
    "zone",
    "description",
    "events",
    "stand",
    "p_throws",
    "balls",
    "strikes",
    "outs_when_up",
    "home_team",
    "away_team",
    "inning_topbot",
]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def _csv_text(rows: list[dict[str, object]]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=STATCAST_HEADERS)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def seed_mlb_metadata(output_dir: Path) -> tuple[Path, Path, Path]:
    normalized_root = (
        output_dir
        / "normalized"
        / "mlb_stats_api"
        / "date=2026-04-21"
        / "run=20260421T170000Z"
    )
    games_path = normalized_root / "games.jsonl"
    probable_starters_path = normalized_root / "probable_starters.jsonl"
    lineup_snapshots_path = normalized_root / "lineup_snapshots.jsonl"

    _write_jsonl(
        games_path,
        [
            {
                "game_pk": 824448,
                "official_date": "2026-04-21",
                "commence_time": "2026-04-21T22:10:00Z",
                "captured_at": "2026-04-21T17:00:00Z",
                "status": "Pre-Game",
                "status_code": "P",
                "venue_id": 5,
                "venue_name": "Progressive Field",
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
        ],
    )
    _write_jsonl(
        probable_starters_path,
        [
            {
                "game_pk": 824448,
                "official_date": "2026-04-21",
                "captured_at": "2026-04-21T17:00:00Z",
                "team_side": "away",
                "team_id": 117,
                "team_abbreviation": "HOU",
                "team_name": "Houston Astros",
                "pitcher_id": 680802,
                "pitcher_name": "Ryan Weiss",
                "pitcher_note": None,
                "odds_matchup_key": "2026-04-21|HOU|CLE|2026-04-21T22:10:00Z",
            },
            {
                "game_pk": 824448,
                "official_date": "2026-04-21",
                "captured_at": "2026-04-21T17:00:00Z",
                "team_side": "home",
                "team_id": 114,
                "team_abbreviation": "CLE",
                "team_name": "Cleveland Guardians",
                "pitcher_id": 800048,
                "pitcher_name": "Parker Messick",
                "pitcher_note": "Confirmed starter",
                "odds_matchup_key": "2026-04-21|HOU|CLE|2026-04-21T22:10:00Z",
            },
        ],
    )
    _write_jsonl(
        lineup_snapshots_path,
        [
            {
                "lineup_snapshot_id": "lineup:824448:home:20260421T200000Z",
                "game_pk": 824448,
                "official_date": "2026-04-21",
                "captured_at": "2026-04-21T20:00:00Z",
                "team_side": "home",
                "team_id": 114,
                "team_abbreviation": "CLE",
                "team_name": "Cleveland Guardians",
                "game_state": "Pre-Game",
                "game_status_code": "P",
                "is_confirmed": True,
                "batting_order_player_ids": [
                    680757,
                    800050,
                    608070,
                    700932,
                    671655,
                    682177,
                    682657,
                    595978,
                    677587,
                ],
                "batter_player_ids": [
                    680757,
                    800050,
                    608070,
                    700932,
                    671655,
                    682177,
                    682657,
                    595978,
                    677587,
                ],
                "lineup_entries": [],
                "odds_matchup_key": "2026-04-21|HOU|CLE|2026-04-21T22:10:00Z",
            },
            {
                "lineup_snapshot_id": "lineup:824448:away:20260421T223000Z",
                "game_pk": 824448,
                "official_date": "2026-04-21",
                "captured_at": "2026-04-21T22:30:00Z",
                "team_side": "away",
                "team_id": 117,
                "team_abbreviation": "HOU",
                "team_name": "Houston Astros",
                "game_state": "In Progress",
                "game_status_code": "I",
                "is_confirmed": True,
                "batting_order_player_ids": [700001, 700002, 700003],
                "batter_player_ids": [700001, 700002, 700003],
                "lineup_entries": [],
                "odds_matchup_key": "2026-04-21|HOU|CLE|2026-04-21T22:10:00Z",
            },
        ],
    )
    return games_path, probable_starters_path, lineup_snapshots_path


def seed_postlock_mlb_metadata(output_dir: Path) -> tuple[Path, Path, Path]:
    normalized_root = (
        output_dir
        / "normalized"
        / "mlb_stats_api"
        / "date=2026-04-21"
        / "run=20260421T230000Z"
    )
    games_path = normalized_root / "games.jsonl"
    probable_starters_path = normalized_root / "probable_starters.jsonl"
    lineup_snapshots_path = normalized_root / "lineup_snapshots.jsonl"

    _write_jsonl(
        games_path,
        [
            {
                "game_pk": 824448,
                "official_date": "2026-04-21",
                "commence_time": "2026-04-21T22:10:00Z",
                "captured_at": "2026-04-21T23:00:00Z",
                "status": "In Progress",
                "status_code": "I",
                "venue_id": 5,
                "venue_name": "Progressive Field",
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
        ],
    )
    _write_jsonl(
        probable_starters_path,
        [
            {
                "game_pk": 824448,
                "official_date": "2026-04-21",
                "captured_at": "2026-04-21T23:00:00Z",
                "team_side": "away",
                "team_id": 117,
                "team_abbreviation": "HOU",
                "team_name": "Houston Astros",
                "pitcher_id": 680802,
                "pitcher_name": "Ryan Weiss",
                "pitcher_note": None,
                "odds_matchup_key": "2026-04-21|HOU|CLE|2026-04-21T22:10:00Z",
            },
            {
                "game_pk": 824448,
                "official_date": "2026-04-21",
                "captured_at": "2026-04-21T23:00:00Z",
                "team_side": "home",
                "team_id": 114,
                "team_abbreviation": "CLE",
                "team_name": "Cleveland Guardians",
                "pitcher_id": 800048,
                "pitcher_name": "Parker Messick",
                "pitcher_note": "Confirmed starter",
                "odds_matchup_key": "2026-04-21|HOU|CLE|2026-04-21T22:10:00Z",
            },
        ],
    )
    _write_jsonl(
        lineup_snapshots_path,
        [
            {
                "lineup_snapshot_id": "lineup:824448:home:20260421T230000Z",
                "game_pk": 824448,
                "official_date": "2026-04-21",
                "captured_at": "2026-04-21T23:00:00Z",
                "team_side": "home",
                "team_id": 114,
                "team_abbreviation": "CLE",
                "team_name": "Cleveland Guardians",
                "game_state": "In Progress",
                "game_status_code": "I",
                "is_confirmed": True,
                "batting_order_player_ids": [680757, 800050],
                "batter_player_ids": [680757, 800050],
                "lineup_entries": [],
                "odds_matchup_key": "2026-04-21|HOU|CLE|2026-04-21T22:10:00Z",
            }
        ],
    )
    return games_path, probable_starters_path, lineup_snapshots_path


def statcast_rows_by_player() -> dict[tuple[str, int], list[dict[str, object]]]:
    return {
        ("pitcher", 680802): [
            {
                "game_date": "2026-04-15",
                "game_pk": 823900,
                "at_bat_number": 1,
                "pitch_number": 1,
                "pitcher": 680802,
                "batter": 680757,
                "pitch_type": "FF",
                "pitch_name": "4-Seam Fastball",
                "release_speed": 95.0,
                "release_spin_rate": 2300,
                "release_extension": 6.4,
                "plate_x": 1.3,
                "plate_z": 2.8,
                "zone": 12,
                "description": "swinging_strike",
                "events": "",
                "stand": "L",
                "p_throws": "R",
                "balls": 0,
                "strikes": 0,
                "outs_when_up": 0,
                "home_team": "CLE",
                "away_team": "HOU",
                "inning_topbot": "Bot",
            },
            {
                "game_date": "2026-04-15",
                "game_pk": 823900,
                "at_bat_number": 1,
                "pitch_number": 2,
                "pitcher": 680802,
                "batter": 680757,
                "pitch_type": "FF",
                "pitch_name": "4-Seam Fastball",
                "release_speed": 95.0,
                "release_spin_rate": 2310,
                "release_extension": 6.4,
                "plate_x": 0.0,
                "plate_z": 2.6,
                "zone": 5,
                "description": "called_strike",
                "events": "",
                "stand": "L",
                "p_throws": "R",
                "balls": 0,
                "strikes": 1,
                "outs_when_up": 0,
                "home_team": "CLE",
                "away_team": "HOU",
                "inning_topbot": "Bot",
            },
            {
                "game_date": "2026-04-15",
                "game_pk": 823900,
                "at_bat_number": 1,
                "pitch_number": 3,
                "pitcher": 680802,
                "batter": 680757,
                "pitch_type": "CH",
                "pitch_name": "Changeup",
                "release_speed": 86.0,
                "release_spin_rate": 1800,
                "release_extension": 6.6,
                "plate_x": 0.2,
                "plate_z": 1.9,
                "zone": 8,
                "description": "swinging_strike",
                "events": "strikeout",
                "stand": "L",
                "p_throws": "R",
                "balls": 0,
                "strikes": 2,
                "outs_when_up": 0,
                "home_team": "CLE",
                "away_team": "HOU",
                "inning_topbot": "Bot",
            },
            {
                "game_date": "2026-04-15",
                "game_pk": 823900,
                "at_bat_number": 2,
                "pitch_number": 1,
                "pitcher": 680802,
                "batter": 800050,
                "pitch_type": "FF",
                "pitch_name": "4-Seam Fastball",
                "release_speed": 94.5,
                "release_spin_rate": 2295,
                "release_extension": 6.3,
                "plate_x": 1.5,
                "plate_z": 3.1,
                "zone": 13,
                "description": "ball",
                "events": "",
                "stand": "R",
                "p_throws": "R",
                "balls": 0,
                "strikes": 0,
                "outs_when_up": 1,
                "home_team": "CLE",
                "away_team": "HOU",
                "inning_topbot": "Bot",
            },
            {
                "game_date": "2026-04-15",
                "game_pk": 823900,
                "at_bat_number": 2,
                "pitch_number": 2,
                "pitcher": 680802,
                "batter": 800050,
                "pitch_type": "FF",
                "pitch_name": "4-Seam Fastball",
                "release_speed": 94.8,
                "release_spin_rate": 2288,
                "release_extension": 6.3,
                "plate_x": 1.4,
                "plate_z": 2.2,
                "zone": 13,
                "description": "foul",
                "events": "",
                "stand": "R",
                "p_throws": "R",
                "balls": 1,
                "strikes": 0,
                "outs_when_up": 1,
                "home_team": "CLE",
                "away_team": "HOU",
                "inning_topbot": "Bot",
            },
            {
                "game_date": "2026-04-15",
                "game_pk": 823900,
                "at_bat_number": 2,
                "pitch_number": 3,
                "pitcher": 680802,
                "batter": 800050,
                "pitch_type": "CH",
                "pitch_name": "Changeup",
                "release_speed": 85.7,
                "release_spin_rate": 1798,
                "release_extension": 6.5,
                "plate_x": -0.1,
                "plate_z": 2.1,
                "zone": 6,
                "description": "hit_into_play",
                "events": "single",
                "stand": "R",
                "p_throws": "R",
                "balls": 1,
                "strikes": 1,
                "outs_when_up": 1,
                "home_team": "CLE",
                "away_team": "HOU",
                "inning_topbot": "Bot",
            },
            {
                "game_date": "2026-04-10",
                "game_pk": 823850,
                "at_bat_number": 1,
                "pitch_number": 1,
                "pitcher": 680802,
                "batter": 680757,
                "pitch_type": "FF",
                "pitch_name": "4-Seam Fastball",
                "release_speed": 96.0,
                "release_spin_rate": 2320,
                "release_extension": 6.5,
                "plate_x": 0.0,
                "plate_z": 2.7,
                "zone": 5,
                "description": "called_strike",
                "events": "",
                "stand": "L",
                "p_throws": "R",
                "balls": 0,
                "strikes": 0,
                "outs_when_up": 0,
                "home_team": "CLE",
                "away_team": "HOU",
                "inning_topbot": "Bot",
            },
            {
                "game_date": "2026-04-10",
                "game_pk": 823850,
                "at_bat_number": 1,
                "pitch_number": 2,
                "pitcher": 680802,
                "batter": 680757,
                "pitch_type": "FF",
                "pitch_name": "4-Seam Fastball",
                "release_speed": 95.5,
                "release_spin_rate": 2315,
                "release_extension": 6.5,
                "plate_x": -0.2,
                "plate_z": 2.8,
                "zone": 5,
                "description": "foul",
                "events": "",
                "stand": "L",
                "p_throws": "R",
                "balls": 0,
                "strikes": 1,
                "outs_when_up": 0,
                "home_team": "CLE",
                "away_team": "HOU",
                "inning_topbot": "Bot",
            },
            {
                "game_date": "2026-04-10",
                "game_pk": 823850,
                "at_bat_number": 1,
                "pitch_number": 3,
                "pitcher": 680802,
                "batter": 680757,
                "pitch_type": "CH",
                "pitch_name": "Changeup",
                "release_speed": 86.3,
                "release_spin_rate": 1810,
                "release_extension": 6.6,
                "plate_x": 1.2,
                "plate_z": 1.8,
                "zone": 11,
                "description": "swinging_strike",
                "events": "strikeout",
                "stand": "L",
                "p_throws": "R",
                "balls": 0,
                "strikes": 2,
                "outs_when_up": 0,
                "home_team": "CLE",
                "away_team": "HOU",
                "inning_topbot": "Bot",
            },
        ],
        ("pitcher", 800048): [
            {
                "game_date": "2026-04-16",
                "game_pk": 823950,
                "at_bat_number": 1,
                "pitch_number": 1,
                "pitcher": 800048,
                "batter": 700001,
                "pitch_type": "FF",
                "pitch_name": "4-Seam Fastball",
                "release_speed": 93.0,
                "release_spin_rate": 2250,
                "release_extension": 6.2,
                "plate_x": 0.1,
                "plate_z": 2.5,
                "zone": 5,
                "description": "called_strike",
                "events": "",
                "stand": "R",
                "p_throws": "L",
                "balls": 0,
                "strikes": 0,
                "outs_when_up": 0,
                "home_team": "CLE",
                "away_team": "HOU",
                "inning_topbot": "Top",
            },
            {
                "game_date": "2026-04-16",
                "game_pk": 823950,
                "at_bat_number": 1,
                "pitch_number": 2,
                "pitcher": 800048,
                "batter": 700001,
                "pitch_type": "FF",
                "pitch_name": "4-Seam Fastball",
                "release_speed": 93.1,
                "release_spin_rate": 2255,
                "release_extension": 6.2,
                "plate_x": -0.2,
                "plate_z": 2.9,
                "zone": 5,
                "description": "foul",
                "events": "",
                "stand": "R",
                "p_throws": "L",
                "balls": 0,
                "strikes": 1,
                "outs_when_up": 0,
                "home_team": "CLE",
                "away_team": "HOU",
                "inning_topbot": "Top",
            },
            {
                "game_date": "2026-04-16",
                "game_pk": 823950,
                "at_bat_number": 1,
                "pitch_number": 3,
                "pitcher": 800048,
                "batter": 700001,
                "pitch_type": "FF",
                "pitch_name": "4-Seam Fastball",
                "release_speed": 92.8,
                "release_spin_rate": 2248,
                "release_extension": 6.2,
                "plate_x": 0.3,
                "plate_z": 2.1,
                "zone": 6,
                "description": "hit_into_play",
                "events": "groundout",
                "stand": "R",
                "p_throws": "L",
                "balls": 0,
                "strikes": 2,
                "outs_when_up": 0,
                "home_team": "CLE",
                "away_team": "HOU",
                "inning_topbot": "Top",
            },
            {
                "game_date": "2026-04-09",
                "game_pk": 823840,
                "at_bat_number": 1,
                "pitch_number": 1,
                "pitcher": 800048,
                "batter": 700002,
                "pitch_type": "CH",
                "pitch_name": "Changeup",
                "release_speed": 84.0,
                "release_spin_rate": 1780,
                "release_extension": 6.0,
                "plate_x": 1.1,
                "plate_z": 1.7,
                "zone": 12,
                "description": "swinging_strike",
                "events": "",
                "stand": "L",
                "p_throws": "L",
                "balls": 0,
                "strikes": 0,
                "outs_when_up": 0,
                "home_team": "CLE",
                "away_team": "HOU",
                "inning_topbot": "Top",
            },
            {
                "game_date": "2026-04-09",
                "game_pk": 823840,
                "at_bat_number": 1,
                "pitch_number": 2,
                "pitcher": 800048,
                "batter": 700002,
                "pitch_type": "CH",
                "pitch_name": "Changeup",
                "release_speed": 84.2,
                "release_spin_rate": 1785,
                "release_extension": 6.1,
                "plate_x": 1.2,
                "plate_z": 1.6,
                "zone": 12,
                "description": "swinging_strike",
                "events": "strikeout",
                "stand": "L",
                "p_throws": "L",
                "balls": 0,
                "strikes": 1,
                "outs_when_up": 0,
                "home_team": "CLE",
                "away_team": "HOU",
                "inning_topbot": "Top",
            },
        ],
        ("batter", 680757): [],
        ("batter", 800050): [],
    }


class StubStatcastClient:
    def __init__(self) -> None:
        self.urls: list[str] = []
        self.rows_by_player = statcast_rows_by_player()

    def fetch_csv(self, url: str) -> str:
        self.urls.append(url)
        query = parse_qs(urlparse(url).query)
        player_type = query["player_type"][0]
        lookup_key = "pitchers_lookup[]" if player_type == "pitcher" else "batters_lookup[]"
        player_id = int(query[lookup_key][0])
        if player_type == "batter" and player_id == 680757:
            return _csv_text(self.rows_by_player[("pitcher", 680802)][0:3] + self.rows_by_player[("pitcher", 680802)][6:9])
        if player_type == "batter" and player_id == 800050:
            return _csv_text(self.rows_by_player[("pitcher", 680802)][3:6])
        return _csv_text(self.rows_by_player.get((player_type, player_id), []))


def test_statcast_search_client_sends_user_agent_header(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b"game_date\n"

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr("mlb_props_stack.ingest.statcast_features.urlopen", fake_urlopen)

    client = StatcastSearchClient(timeout_seconds=12.5)
    csv_text = client.fetch_csv("https://example.com/statcast.csv")

    assert csv_text == "game_date\n"
    assert captured["timeout"] == 12.5
    assert dict(captured["request"].header_items())["User-agent"].startswith("Mozilla/5.0")


def test_build_statcast_search_csv_url_uses_player_specific_lookup_key() -> None:
    pitcher_url = build_statcast_search_csv_url(
        player_type="pitcher",
        player_id=680802,
        start_date=date(2026, 3, 22),
        end_date=date(2026, 4, 20),
    )
    batter_url = build_statcast_search_csv_url(
        player_type="batter",
        player_id=680757,
        start_date=date(2026, 3, 22),
        end_date=date(2026, 4, 20),
    )

    assert pitcher_url.startswith("https://baseballsavant.mlb.com/statcast_search/csv?")
    assert "pitchers_lookup%5B%5D=680802" in pitcher_url
    assert "batters_lookup%5B%5D=680757" in batter_url
    assert "game_date_gt=2026-03-22" in pitcher_url
    assert "game_date_lt=2026-04-20" in pitcher_url


def test_ingest_statcast_features_uses_latest_pregame_valid_metadata_run(tmp_path) -> None:
    games_path, _, _ = seed_mlb_metadata(tmp_path)
    seed_postlock_mlb_metadata(tmp_path)
    stub_client = StubStatcastClient()
    timestamps = iter(
        datetime(2026, 4, 21, 18, 0, tzinfo=UTC) + timedelta(minutes=index)
        for index in range(20)
    )

    result = ingest_statcast_features_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        history_days=30,
        client=stub_client,
        now=lambda: next(timestamps),
    )

    assert result.mlb_games_path == games_path
    assert "run=20260421T170000Z" in str(result.mlb_probable_starters_path)


def test_ingest_statcast_features_falls_back_to_latest_complete_historical_metadata_run(
    tmp_path,
) -> None:
    _, probable_starters_path, _ = seed_postlock_mlb_metadata(tmp_path)
    stub_client = StubStatcastClient()
    timestamps = iter(
        datetime(2026, 4, 22, 18, 0, tzinfo=UTC) + timedelta(minutes=index)
        for index in range(20)
    )

    result = ingest_statcast_features_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        history_days=30,
        client=stub_client,
        now=lambda: next(timestamps),
    )

    assert result.mlb_probable_starters_path == probable_starters_path
    assert result.pitcher_feature_count == 2
    for path in (
        result.pitcher_daily_features_path,
        result.lineup_daily_features_path,
        result.game_context_features_path,
    ):
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert {row["features_as_of"] for row in rows} == {"2026-04-21T00:00:00Z"}


def test_ingest_statcast_features_for_date_writes_feature_tables_and_handles_missing_inputs(
    tmp_path,
) -> None:
    seed_mlb_metadata(tmp_path)
    stub_client = StubStatcastClient()
    timestamps = iter(
        [
            datetime(2026, 4, 21, 18, 0, tzinfo=UTC),
            datetime(2026, 4, 21, 18, 1, tzinfo=UTC),
            datetime(2026, 4, 21, 18, 2, tzinfo=UTC),
            datetime(2026, 4, 21, 18, 3, tzinfo=UTC),
            datetime(2026, 4, 21, 18, 4, tzinfo=UTC),
            datetime(2026, 4, 21, 18, 5, tzinfo=UTC),
            datetime(2026, 4, 21, 18, 6, tzinfo=UTC),
            datetime(2026, 4, 21, 18, 7, tzinfo=UTC),
            datetime(2026, 4, 21, 18, 8, tzinfo=UTC),
            datetime(2026, 4, 21, 18, 9, tzinfo=UTC),
            datetime(2026, 4, 21, 18, 10, tzinfo=UTC),
            datetime(2026, 4, 21, 18, 11, tzinfo=UTC),
        ]
    )

    result = ingest_statcast_features_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        history_days=30,
        client=stub_client,
        now=lambda: next(timestamps),
    )

    assert isinstance(result, StatcastFeatureIngestResult)
    assert result.raw_pull_count == 11
    assert result.pitch_level_record_count == 14
    assert result.pitcher_feature_count == 2
    assert result.lineup_feature_count == 2
    assert result.game_context_feature_count == 2
    assert result.pull_manifest_path.exists()
    assert result.pitch_level_base_path.exists()
    assert result.pitcher_daily_features_path.exists()
    assert result.lineup_daily_features_path.exists()
    assert result.game_context_features_path.exists()

    pitch_rows = [
        json.loads(line)
        for line in result.pitch_level_base_path.read_text(encoding="utf-8").splitlines()
    ]
    pitcher_rows = [
        json.loads(line)
        for line in result.pitcher_daily_features_path.read_text(encoding="utf-8").splitlines()
    ]
    lineup_rows = [
        json.loads(line)
        for line in result.lineup_daily_features_path.read_text(encoding="utf-8").splitlines()
    ]
    context_rows = [
        json.loads(line)
        for line in result.game_context_features_path.read_text(encoding="utf-8").splitlines()
    ]

    pitch_by_id = {row["pitch_record_id"]: row for row in pitch_rows}
    traced_pitch = pitch_by_id["pitch:823850:1:1:680802:680757"]
    assert traced_pitch["is_called_strike"] is True
    assert traced_pitch["batting_team_abbreviation"] == "CLE"

    by_pitcher_id = {row["pitcher_id"]: row for row in pitcher_rows}
    ryan = by_pitcher_id[680802]
    parker = by_pitcher_id[800048]
    assert ryan["feature_status"] == "ok"
    assert ryan["pitcher_hand"] == "R"
    assert ryan["pitch_sample_size"] == 9
    assert ryan["plate_appearance_sample_size"] == 3
    assert ryan["pitcher_k_rate"] == 0.666667
    assert ryan["pitcher_k_rate_vs_lhh"] == 1.0
    assert ryan["pitcher_k_rate_vs_rhh"] == 0.0
    assert ryan["swinging_strike_rate"] == 0.333333
    assert ryan["pitcher_whiff_rate_vs_lhh"] == 0.5
    assert ryan["pitcher_whiff_rate_vs_rhh"] == 0.0
    assert ryan["csw_rate"] == 0.555556
    assert ryan["pitch_type_usage"] == {"CH": 0.333333, "FF": 0.666667}
    assert ryan["rest_days"] == 6
    assert ryan["recent_pitch_count"] == 9
    assert ryan["last_start_pitch_count"] == 6
    assert parker["pitcher_hand"] == "L"
    assert parker["rest_days"] == 5
    assert parker["pitcher_k_rate_vs_lhh"] == 1.0
    assert parker["pitcher_k_rate_vs_rhh"] == 0.0
    assert parker["pitcher_whiff_rate_vs_lhh"] == 1.0
    assert parker["pitcher_whiff_rate_vs_rhh"] == 0.0

    lineup_by_pitcher = {row["pitcher_id"]: row for row in lineup_rows}
    ryan_lineup = lineup_by_pitcher[680802]
    parker_lineup = lineup_by_pitcher[800048]
    assert ryan_lineup["lineup_status"] == "confirmed"
    assert ryan_lineup["lineup_snapshot_id"] == "lineup:824448:home:20260421T200000Z"
    assert ryan_lineup["lineup_size"] == 9
    assert ryan_lineup["available_batter_feature_count"] == 2
    assert ryan_lineup["projected_lineup_k_rate"] == 0.5
    assert ryan_lineup["projected_lineup_k_rate_vs_pitcher_hand"] == 0.5
    # Slot 0 (680757) weight=9 with k_rate_vs_rhp=1.0; slot 1 (800050) weight=8
    # with k_rate_vs_rhp=0.0 → weighted mean = 9/17 = 0.529412.
    assert ryan_lineup["lineup_k_rate_vs_rhp"] == 0.529412
    assert ryan_lineup["lineup_k_rate_vs_lhp"] is None
    assert ryan_lineup["projected_lineup_chase_rate"] == 0.75
    assert ryan_lineup["projected_lineup_contact_rate"] == 0.625
    assert ryan_lineup["lineup_continuity_count"] == 2
    assert ryan_lineup["lineup_continuity_ratio"] == 0.222222
    assert parker_lineup["lineup_status"] == "missing_pregame_lineup"
    assert parker_lineup["lineup_snapshot_id"] is None
    assert parker_lineup["lineup_size"] == 0
    assert parker_lineup["lineup_k_rate_vs_rhp"] is None
    assert parker_lineup["lineup_k_rate_vs_lhp"] is None

    context_by_pitcher = {row["pitcher_id"]: row for row in context_rows}
    assert context_by_pitcher[680802]["weather_status"] == "missing_weather_source"
    assert context_by_pitcher[680802]["weather_source"] is None
    assert context_by_pitcher[680802]["weather_temperature_f"] is None
    assert context_by_pitcher[680802]["weather_wind_speed_mph"] is None
    assert context_by_pitcher[680802]["weather_wind_direction_deg"] is None
    assert context_by_pitcher[680802]["weather_humidity_pct"] is None
    assert context_by_pitcher[680802]["weather_captured_at"] is None
    assert context_by_pitcher[680802]["roof_type"] is None
    assert context_by_pitcher[680802]["umpire_status"] == "missing_umpire_source"
    assert context_by_pitcher[680802]["umpire_source"] is None
    assert context_by_pitcher[680802]["umpire_id"] is None
    assert context_by_pitcher[680802]["umpire_name"] is None
    assert context_by_pitcher[680802]["umpire_captured_at"] is None
    assert context_by_pitcher[680802]["ump_called_strike_rate_30d"] is None
    assert context_by_pitcher[680802]["ump_k_per_9_delta_vs_league_30d"] is None
    assert context_by_pitcher[680802]["park_factor_status"] == "ok"
    assert context_by_pitcher[680802]["park_k_factor"] == 1.02
    assert context_by_pitcher[680802]["park_k_factor_vs_rhh"] == 1.02
    assert context_by_pitcher[680802]["park_k_factor_vs_lhh"] == 1.03
    assert context_by_pitcher[680802]["expected_leash_pitch_count"] == 4.5
    assert context_by_pitcher[800048]["expected_leash_batters_faced"] == 1.0

    assert len(stub_client.urls) == 11
    assert "pitchers_lookup%5B%5D=680802" in stub_client.urls[0]
    assert any("batters_lookup%5B%5D=680757" in url for url in stub_client.urls)


def test_ingest_statcast_features_preserves_missing_park_factor_source_for_unknown_venue(
    tmp_path,
) -> None:
    games_path, _, _ = seed_mlb_metadata(tmp_path)
    game_rows = [
        json.loads(line) for line in games_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in game_rows:
        row["venue_id"] = 999999
        row["venue_name"] = "Unknown Park"
    _write_jsonl(games_path, game_rows)

    stub_client = StubStatcastClient()
    timestamps = iter(
        datetime(2026, 4, 21, 18, 0, tzinfo=UTC) + timedelta(minutes=index)
        for index in range(20)
    )

    result = ingest_statcast_features_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        history_days=30,
        client=stub_client,
        now=lambda: next(timestamps),
    )

    context_rows = [
        json.loads(line)
        for line in result.game_context_features_path.read_text(encoding="utf-8").splitlines()
    ]
    assert context_rows
    for row in context_rows:
        assert row["venue_id"] == 999999
        assert row["park_factor_status"] == "missing_park_factor_source"
        assert row["park_k_factor"] is None
        assert row["park_k_factor_vs_rhh"] is None
        assert row["park_k_factor_vs_lhh"] is None


def test_ingest_statcast_features_dedupes_duplicate_pitcher_pulls(tmp_path) -> None:
    _, probable_starters_path, _ = seed_mlb_metadata(tmp_path)
    probable_starter_rows = [
        json.loads(line)
        for line in probable_starters_path.read_text(encoding="utf-8").splitlines()
    ]
    probable_starter_rows.append(dict(probable_starter_rows[0]))
    _write_jsonl(probable_starters_path, probable_starter_rows)

    stub_client = StubStatcastClient()
    timestamps = iter(
        datetime(2026, 4, 21, 18, 0, tzinfo=UTC) + timedelta(minutes=index)
        for index in range(20)
    )

    result = ingest_statcast_features_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        history_days=30,
        client=stub_client,
        now=lambda: next(timestamps),
    )

    assert result.raw_pull_count == 11
    assert sum("pitchers_lookup%5B%5D=680802" in url for url in stub_client.urls) == 1


class _ResponseStub:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._body


def _http_error(code: int) -> HTTPError:
    return HTTPError(url="https://example.com", code=code, msg="stub", hdrs=None, fp=None)


def test_statcast_search_client_retries_transient_http_errors_with_backoff(
    monkeypatch,
) -> None:
    responses: list[object] = [
        _http_error(503),
        URLError("connection reset"),
        _ResponseStub(b"game_date,ok\n"),
    ]

    def fake_urlopen(request, timeout):
        next_response = responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return next_response

    monkeypatch.setattr(
        "mlb_props_stack.ingest.statcast_features.urlopen",
        fake_urlopen,
    )

    sleeps: list[float] = []
    client = StatcastSearchClient(
        max_attempts=3,
        initial_backoff_seconds=0.5,
        backoff_multiplier=2.0,
        max_backoff_seconds=10.0,
        sleep=sleeps.append,
    )

    csv_text = client.fetch_csv("https://example.com/statcast.csv")

    assert csv_text == "game_date,ok\n"
    assert sleeps == [0.5, 1.0]
    assert responses == []


def test_statcast_search_client_raises_after_exhausting_attempts(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise _http_error(502)

    monkeypatch.setattr(
        "mlb_props_stack.ingest.statcast_features.urlopen",
        fake_urlopen,
    )
    sleeps: list[float] = []
    client = StatcastSearchClient(
        max_attempts=3,
        initial_backoff_seconds=0.25,
        backoff_multiplier=3.0,
        max_backoff_seconds=1.0,
        sleep=sleeps.append,
    )

    with pytest.raises(HTTPError) as excinfo:
        client.fetch_csv("https://example.com/statcast.csv")

    assert excinfo.value.code == 502
    # Two sleeps between three attempts; second delay is clamped to max backoff.
    assert sleeps == [0.25, 0.75]


def test_statcast_search_client_does_not_retry_on_client_http_errors(monkeypatch) -> None:
    call_count = {"value": 0}

    def fake_urlopen(request, timeout):
        call_count["value"] += 1
        raise _http_error(404)

    monkeypatch.setattr(
        "mlb_props_stack.ingest.statcast_features.urlopen",
        fake_urlopen,
    )
    sleeps: list[float] = []
    client = StatcastSearchClient(
        max_attempts=5,
        initial_backoff_seconds=0.1,
        sleep=sleeps.append,
    )

    with pytest.raises(HTTPError):
        client.fetch_csv("https://example.com/statcast.csv")

    assert call_count["value"] == 1
    assert sleeps == []


def test_statcast_search_client_rejects_invalid_retry_configuration() -> None:
    with pytest.raises(ValueError):
        StatcastSearchClient(max_attempts=0)
    with pytest.raises(ValueError):
        StatcastSearchClient(initial_backoff_seconds=-1.0)
    with pytest.raises(ValueError):
        StatcastSearchClient(backoff_multiplier=0.5)
    with pytest.raises(ValueError):
        StatcastSearchClient(max_backoff_seconds=-0.1)


class _ConcurrencyProbeClient:
    def __init__(self, rows_by_player: dict[tuple[str, int], list[dict[str, object]]]) -> None:
        self._rows_by_player = rows_by_player
        self._lock = threading.Lock()
        self._in_flight = 0
        self._ready = threading.Event()
        self.urls: list[str] = []
        self.max_in_flight = 0

    def fetch_csv(self, url: str) -> str:
        with self._lock:
            self.urls.append(url)
            self._in_flight += 1
            if self._in_flight > self.max_in_flight:
                self.max_in_flight = self._in_flight
            should_release = self._in_flight >= 2
        if should_release:
            self._ready.set()
        # Each worker waits until at least two fetches are simultaneously in
        # flight; this proves the thread pool actually parallelises work.
        self._ready.wait(timeout=1.0)
        try:
            query = parse_qs(urlparse(url).query)
            player_type = query["player_type"][0]
            lookup_key = (
                "pitchers_lookup[]" if player_type == "pitcher" else "batters_lookup[]"
            )
            player_id = int(query[lookup_key][0])
            rows: list[dict[str, object]] = self._rows_by_player.get(
                (player_type, player_id),
                [],
            )
            return _csv_text(rows)
        finally:
            with self._lock:
                self._in_flight -= 1


def test_ingest_statcast_features_fetches_pulls_in_parallel(tmp_path) -> None:
    seed_mlb_metadata(tmp_path)
    client = _ConcurrencyProbeClient(rows_by_player=statcast_rows_by_player())
    timestamps = iter(
        datetime(2026, 4, 21, 18, 0, tzinfo=UTC) + timedelta(minutes=index)
        for index in range(20)
    )

    result = ingest_statcast_features_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        history_days=30,
        client=client,
        now=lambda: next(timestamps),
        max_fetch_workers=4,
    )

    assert result.raw_pull_count == 11
    assert client.max_in_flight >= 2


def test_ingest_statcast_features_preserves_pull_ordering_under_threaded_fetch(
    tmp_path,
) -> None:
    seed_mlb_metadata(tmp_path)
    stub_client = StubStatcastClient()
    timestamps = iter(
        datetime(2026, 4, 21, 18, 0, tzinfo=UTC) + timedelta(minutes=index)
        for index in range(20)
    )

    result = ingest_statcast_features_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        history_days=30,
        client=stub_client,
        now=lambda: next(timestamps),
        max_fetch_workers=4,
    )

    manifest_rows = [
        json.loads(line)
        for line in result.pull_manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    manifest_player_order = [
        (row["player_type"], row["player_id"]) for row in manifest_rows
    ]
    expected_order = [
        ("pitcher", 680802),
        ("pitcher", 800048),
    ] + [("batter", batter_id) for batter_id in sorted(
        {608070, 671655, 677587, 680757, 682177, 682657, 595978, 700932, 800050}
    )]
    assert manifest_player_order == expected_order
    captured_at_values = [row["captured_at"] for row in manifest_rows]
    assert captured_at_values == sorted(captured_at_values)


def _synthetic_pitch_record(
    *,
    pitch_number: int,
    stand: str,
    is_whiff: bool,
    is_plate_appearance_final_pitch: bool,
    is_strikeout_event: bool,
) -> StatcastPitchRecord:
    return StatcastPitchRecord(
        pitch_record_id=f"pitch:1:1:{pitch_number}:1000:{2000 if stand == 'R' else 2001}",
        source_pull_id="test-pull",
        source_row_number=pitch_number + 1,
        game_date="2026-04-20",
        game_pk=1,
        at_bat_number=1 if stand == "R" else 2,
        pitch_number=pitch_number,
        pitcher_id=1000,
        batter_id=2000 if stand == "R" else 2001,
        pitch_type="FF",
        pitch_name="4-Seam Fastball",
        release_speed=95.0,
        release_spin_rate=2300.0,
        release_extension=6.5,
        plate_x=0.0,
        plate_z=2.5,
        zone=5,
        description="swinging_strike" if is_whiff else "called_strike",
        events="strikeout" if is_strikeout_event else ("foul" if is_plate_appearance_final_pitch else None),
        stand=stand,
        p_throws="R",
        balls=0,
        strikes=2,
        outs_when_up=0,
        home_team_abbreviation="CLE",
        away_team_abbreviation="HOU",
        batting_team_abbreviation="HOU",
        fielding_team_abbreviation="CLE",
        is_plate_appearance_final_pitch=is_plate_appearance_final_pitch,
        is_strikeout_event=is_strikeout_event,
        is_whiff=is_whiff,
        is_called_strike=not is_whiff,
        is_swing=is_whiff,
        is_contact=False,
        is_out_of_zone=False,
        is_chase_swing=False,
    )


def test_pitcher_hand_split_rates_computes_separate_rates_per_batter_stand() -> None:
    rows = [
        # vs L: 4 pitches across 2 PAs, 2 whiffs, 2 Ks
        _synthetic_pitch_record(
            pitch_number=1, stand="L", is_whiff=True,
            is_plate_appearance_final_pitch=False, is_strikeout_event=False,
        ),
        _synthetic_pitch_record(
            pitch_number=2, stand="L", is_whiff=True,
            is_plate_appearance_final_pitch=True, is_strikeout_event=True,
        ),
        _synthetic_pitch_record(
            pitch_number=3, stand="L", is_whiff=False,
            is_plate_appearance_final_pitch=False, is_strikeout_event=False,
        ),
        _synthetic_pitch_record(
            pitch_number=4, stand="L", is_whiff=False,
            is_plate_appearance_final_pitch=True, is_strikeout_event=True,
        ),
        # vs R: 4 pitches across 2 PAs, 1 whiff, 0 Ks
        _synthetic_pitch_record(
            pitch_number=5, stand="R", is_whiff=False,
            is_plate_appearance_final_pitch=False, is_strikeout_event=False,
        ),
        _synthetic_pitch_record(
            pitch_number=6, stand="R", is_whiff=False,
            is_plate_appearance_final_pitch=True, is_strikeout_event=False,
        ),
        _synthetic_pitch_record(
            pitch_number=7, stand="R", is_whiff=True,
            is_plate_appearance_final_pitch=False, is_strikeout_event=False,
        ),
        _synthetic_pitch_record(
            pitch_number=8, stand="R", is_whiff=False,
            is_plate_appearance_final_pitch=True, is_strikeout_event=False,
        ),
    ]

    lhh_k_rate, lhh_whiff_rate = _pitcher_hand_split_rates(pitcher_rows=rows, stand="L")
    rhh_k_rate, rhh_whiff_rate = _pitcher_hand_split_rates(pitcher_rows=rows, stand="R")

    assert lhh_k_rate == 1.0  # 2/2 final pitches were strikeouts
    assert lhh_whiff_rate == 0.5  # 2/4 pitches were whiffs
    assert rhh_k_rate == 0.0  # 0/2 final pitches were strikeouts
    assert rhh_whiff_rate == 0.25  # 1/4 pitches were whiffs


def test_pitcher_hand_split_rates_returns_none_when_no_rows_match_stand() -> None:
    rows = [
        # Pitcher has only faced LHH in the window; vs RHH is a missing split.
        _synthetic_pitch_record(
            pitch_number=1, stand="L", is_whiff=True,
            is_plate_appearance_final_pitch=False, is_strikeout_event=False,
        ),
        _synthetic_pitch_record(
            pitch_number=2, stand="L", is_whiff=True,
            is_plate_appearance_final_pitch=True, is_strikeout_event=True,
        ),
    ]

    rhh_k_rate, rhh_whiff_rate = _pitcher_hand_split_rates(pitcher_rows=rows, stand="R")
    lhh_k_rate, lhh_whiff_rate = _pitcher_hand_split_rates(pitcher_rows=rows, stand="L")

    assert rhh_k_rate is None
    assert rhh_whiff_rate is None
    assert lhh_k_rate == 1.0
    assert lhh_whiff_rate == 1.0


def test_ingest_statcast_features_rejects_invalid_max_fetch_workers(tmp_path) -> None:
    seed_mlb_metadata(tmp_path)
    stub_client = StubStatcastClient()
    timestamps = iter(
        datetime(2026, 4, 21, 18, 0, tzinfo=UTC) + timedelta(minutes=index)
        for index in range(20)
    )

    with pytest.raises(ValueError):
        ingest_statcast_features_for_date(
            target_date=date(2026, 4, 21),
            output_dir=tmp_path,
            history_days=30,
            client=stub_client,
            now=lambda: next(timestamps),
            max_fetch_workers=0,
        )
