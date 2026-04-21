from __future__ import annotations

import json
from datetime import UTC, date, datetime

from mlb_props_stack.ingest import (
    MLBMetadataIngestResult,
    build_odds_matchup_key,
    ingest_mlb_metadata_for_date,
)
from mlb_props_stack.ingest.mlb_stats_api import (
    normalize_feed_live_payload,
    normalize_schedule_payload,
)


def sample_schedule_payload() -> dict:
    return {
        "dates": [
            {
                "date": "2026-04-21",
                "games": [
                    {
                        "gamePk": 824448,
                        "gameDate": "2026-04-21T22:10:00Z",
                        "officialDate": "2026-04-21",
                        "gameNumber": 1,
                        "doubleHeader": "N",
                        "dayNight": "night",
                        "status": {
                            "detailedState": "Pre-Game",
                            "statusCode": "P",
                        },
                        "venue": {"id": 5, "name": "Progressive Field"},
                        "teams": {
                            "away": {
                                "team": {
                                    "id": 117,
                                    "abbreviation": "HOU",
                                    "name": "Houston Astros",
                                },
                                "probablePitcher": {
                                    "id": 680802,
                                    "fullName": "Ryan Weiss",
                                },
                            },
                            "home": {
                                "team": {
                                    "id": 114,
                                    "abbreviation": "CLE",
                                    "name": "Cleveland Guardians",
                                },
                                "probablePitcher": {
                                    "id": 800048,
                                    "fullName": "Parker Messick",
                                    "note": "Confirmed starter",
                                },
                            },
                        },
                    }
                ],
            }
        ]
    }


def sample_feed_live_payload() -> dict:
    return {
        "gamePk": 824448,
        "gameData": {
            "datetime": {
                "dateTime": "2026-04-21T22:10:00Z",
                "officialDate": "2026-04-21",
            },
            "status": {
                "detailedState": "Pre-Game",
                "statusCode": "P",
            },
            "teams": {
                "away": {
                    "id": 117,
                    "abbreviation": "HOU",
                    "name": "Houston Astros",
                },
                "home": {
                    "id": 114,
                    "abbreviation": "CLE",
                    "name": "Cleveland Guardians",
                },
            },
        },
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {
                        "battingOrder": [],
                        "batters": [],
                        "players": {},
                    },
                    "home": {
                        "battingOrder": [
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
                        "batters": [
                            680757,
                            800050,
                            608070,
                            700932,
                            671655,
                            682177,
                            682657,
                            595978,
                            677587,
                            804240,
                        ],
                        "players": {
                            "ID680757": {
                                "person": {"id": 680757, "fullName": "Player One"},
                                "position": {"abbreviation": "CF"},
                                "battingOrder": "100",
                            },
                            "ID800050": {
                                "person": {"id": 800050, "fullName": "Player Two"},
                                "position": {"abbreviation": "SS"},
                                "battingOrder": "200",
                            },
                            "ID608070": {
                                "person": {"id": 608070, "fullName": "Player Three"},
                                "position": {"abbreviation": "3B"},
                                "battingOrder": "300",
                            },
                            "ID700932": {
                                "person": {"id": 700932, "fullName": "Player Four"},
                                "position": {"abbreviation": "1B"},
                                "battingOrder": "400",
                            },
                            "ID671655": {
                                "person": {"id": 671655, "fullName": "Player Five"},
                                "position": {"abbreviation": "RF"},
                                "battingOrder": "500",
                            },
                            "ID682177": {
                                "person": {"id": 682177, "fullName": "Player Six"},
                                "position": {"abbreviation": "DH"},
                                "battingOrder": "600",
                            },
                            "ID682657": {
                                "person": {"id": 682657, "fullName": "Player Seven"},
                                "position": {"abbreviation": "2B"},
                                "battingOrder": "700",
                            },
                            "ID595978": {
                                "person": {"id": 595978, "fullName": "Player Eight"},
                                "position": {"abbreviation": "C"},
                                "battingOrder": "800",
                            },
                            "ID677587": {
                                "person": {"id": 677587, "fullName": "Player Nine"},
                                "position": {"abbreviation": "LF"},
                                "battingOrder": "900",
                            },
                        },
                    },
                }
            }
        },
    }


class StubClient:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def fetch_json(self, url: str) -> dict:
        self.urls.append(url)
        if "schedule" in url:
            return sample_schedule_payload()
        return sample_feed_live_payload()


def test_build_odds_matchup_key_uses_date_teams_and_commence_time() -> None:
    key = build_odds_matchup_key(
        official_date="2026-04-21",
        away_team_abbreviation="hou",
        home_team_abbreviation="cle",
        commence_time=datetime(2026, 4, 21, 22, 10, tzinfo=UTC),
    )

    assert key == "2026-04-21|HOU|CLE|2026-04-21T22:10:00Z"


def test_normalize_schedule_payload_returns_games_and_probable_starters() -> None:
    captured_at = datetime(2026, 4, 21, 18, 0, tzinfo=UTC)

    games, probable_starters = normalize_schedule_payload(
        sample_schedule_payload(),
        captured_at=captured_at,
    )

    assert len(games) == 1
    assert games[0].game_pk == 824448
    assert games[0].venue_name == "Progressive Field"
    assert games[0].odds_matchup_key == "2026-04-21|HOU|CLE|2026-04-21T22:10:00Z"

    assert len(probable_starters) == 2
    assert probable_starters[0].pitcher_name == "Ryan Weiss"
    assert probable_starters[1].pitcher_note == "Confirmed starter"


def test_normalize_feed_live_payload_preserves_captured_at_and_confirmation() -> None:
    captured_at = datetime(2026, 4, 21, 18, 5, tzinfo=UTC)

    snapshots = normalize_feed_live_payload(
        sample_feed_live_payload(),
        captured_at=captured_at,
    )

    assert len(snapshots) == 2
    away_snapshot, home_snapshot = snapshots

    assert away_snapshot.captured_at == captured_at
    assert away_snapshot.game_pk == 824448
    assert away_snapshot.is_confirmed is False
    assert away_snapshot.batting_order_player_ids == ()

    assert home_snapshot.is_confirmed is True
    assert len(home_snapshot.lineup_entries) == 9
    assert home_snapshot.lineup_entries[0].player_name == "Player One"
    assert home_snapshot.lineup_snapshot_id == "lineup:824448:home:20260421T180500Z"


def test_ingest_mlb_metadata_for_date_writes_raw_and_normalized_outputs(
    tmp_path,
) -> None:
    stub_client = StubClient()
    timestamps = iter(
        [
            datetime(2026, 4, 21, 18, 0, tzinfo=UTC),
            datetime(2026, 4, 21, 18, 1, tzinfo=UTC),
            datetime(2026, 4, 21, 18, 2, tzinfo=UTC),
        ]
    )

    result = ingest_mlb_metadata_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        client=stub_client,
        now=lambda: next(timestamps),
    )

    assert isinstance(result, MLBMetadataIngestResult)
    assert result.game_count == 1
    assert result.probable_starter_count == 2
    assert result.lineup_snapshot_count == 2
    assert result.schedule_raw_path.exists()
    assert len(result.feed_live_raw_paths) == 1
    assert result.feed_live_raw_paths[0].exists()
    assert result.games_path.exists()
    assert result.probable_starters_path.exists()
    assert result.lineup_snapshots_path.exists()

    games_rows = [json.loads(line) for line in result.games_path.read_text(encoding="utf-8").splitlines()]
    starter_rows = [json.loads(line) for line in result.probable_starters_path.read_text(encoding="utf-8").splitlines()]
    lineup_rows = [json.loads(line) for line in result.lineup_snapshots_path.read_text(encoding="utf-8").splitlines()]

    assert games_rows[0]["game_pk"] == 824448
    assert games_rows[0]["captured_at"] == "2026-04-21T18:01:00Z"
    assert starter_rows[1]["pitcher_name"] == "Parker Messick"
    assert lineup_rows[0]["captured_at"] == "2026-04-21T18:02:00Z"
    assert lineup_rows[1]["lineup_entries"][0]["player_name"] == "Player One"
    assert "schedule?sportId=1" in stub_client.urls[0]
