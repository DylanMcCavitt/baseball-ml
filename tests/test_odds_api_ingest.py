from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

from mlb_props_stack.ingest import (
    GameRecord,
    OddsAPIIngestResult,
    ingest_odds_api_pitcher_lines_for_date,
)
from mlb_props_stack.ingest.odds_api import (
    normalize_event_odds_payload,
    normalize_events_payload,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def seed_mlb_metadata(output_dir: Path) -> tuple[Path, Path]:
    normalized_root = (
        output_dir
        / "normalized"
        / "mlb_stats_api"
        / "date=2026-04-21"
        / "run=20260421T170000Z"
    )
    games_path = normalized_root / "games.jsonl"
    probable_starters_path = normalized_root / "probable_starters.jsonl"

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
    return games_path, probable_starters_path


def sample_events_payload() -> list[dict]:
    return [
        {
            "id": "odds-event-1",
            "sport_key": "baseball_mlb",
            "sport_title": "MLB",
            "commence_time": "2026-04-21T22:10:00Z",
            "home_team": "Cleveland Guardians",
            "away_team": "Houston Astros",
        },
        {
            "id": "other-event",
            "sport_key": "baseball_mlb",
            "sport_title": "MLB",
            "commence_time": "2026-04-22T01:40:00Z",
            "home_team": "Los Angeles Dodgers",
            "away_team": "San Diego Padres",
        },
    ]


def sample_event_odds_payload() -> dict:
    return {
        "id": "odds-event-1",
        "sport_key": "baseball_mlb",
        "sport_title": "MLB",
        "commence_time": "2026-04-21T22:10:00Z",
        "home_team": "Cleveland Guardians",
        "away_team": "Houston Astros",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "last_update": "2026-04-21T17:54:21Z",
                "markets": [
                    {
                        "key": "pitcher_strikeouts",
                        "last_update": "2026-04-21T17:54:42Z",
                        "outcomes": [
                            {
                                "name": "Over",
                                "description": "Ryan Weiss",
                                "price": -125,
                                "point": 4.5,
                            },
                            {
                                "name": "Under",
                                "description": "Ryan Weiss",
                                "price": 105,
                                "point": 4.5,
                            },
                            {
                                "name": "Over",
                                "description": "Parker Messick",
                                "price": -110,
                                "point": 4.5,
                            },
                            {
                                "name": "Under",
                                "description": "Parker Messick",
                                "price": -110,
                                "point": 4.5,
                            },
                            {
                                "name": "Over",
                                "description": "Incomplete Pitcher",
                                "price": -135,
                                "point": 5.5,
                            },
                        ],
                    }
                ],
            }
        ],
    }


class StubOddsClient:
    def __init__(self) -> None:
        self.api_key = "stub-key"
        self.urls: list[str] = []

    def fetch_json(self, url: str):
        self.urls.append(url)
        if "/events/" in url and "/odds?" in url:
            return sample_event_odds_payload()
        return sample_events_payload()


def test_ingest_odds_api_pitcher_lines_for_date_writes_raw_and_normalized_outputs(
    tmp_path,
) -> None:
    seed_mlb_metadata(tmp_path)
    stub_client = StubOddsClient()
    timestamps = iter(
        [
            datetime(2026, 4, 21, 18, 0, tzinfo=UTC),
            datetime(2026, 4, 21, 18, 1, tzinfo=UTC),
            datetime(2026, 4, 21, 18, 2, tzinfo=UTC),
        ]
    )

    result = ingest_odds_api_pitcher_lines_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        client=stub_client,
        now=lambda: next(timestamps),
    )

    assert isinstance(result, OddsAPIIngestResult)
    assert result.candidate_event_count == 1
    assert result.matched_event_count == 1
    assert result.unmatched_event_count == 0
    assert result.prop_line_count == 2
    assert result.skipped_prop_count == 1
    assert result.events_raw_path.exists()
    assert len(result.event_odds_raw_paths) == 1
    assert result.event_odds_raw_paths[0].exists()
    assert result.event_mappings_path.exists()
    assert result.prop_line_snapshots_path.exists()

    mapping_rows = [
        json.loads(line)
        for line in result.event_mappings_path.read_text(encoding="utf-8").splitlines()
    ]
    prop_rows = [
        json.loads(line)
        for line in result.prop_line_snapshots_path.read_text(encoding="utf-8").splitlines()
    ]

    assert mapping_rows[0]["game_pk"] == 824448
    assert mapping_rows[0]["match_status"] == "matched"
    assert prop_rows[0]["captured_at"] == "2026-04-21T18:02:00Z"
    assert {
        row["player_id"] for row in prop_rows
    } == {"mlb-pitcher:680802", "mlb-pitcher:800048"}
    assert "sports/baseball_mlb/events?" in stub_client.urls[0]
    assert "/events/odds-event-1/odds?" in stub_client.urls[1]


def test_repeated_odds_ingest_runs_keep_prior_snapshots(tmp_path) -> None:
    seed_mlb_metadata(tmp_path)

    first_result = ingest_odds_api_pitcher_lines_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        client=StubOddsClient(),
        now=iter(
            [
                datetime(2026, 4, 21, 18, 0, tzinfo=UTC),
                datetime(2026, 4, 21, 18, 1, tzinfo=UTC),
                datetime(2026, 4, 21, 18, 2, tzinfo=UTC),
            ]
        ).__next__,
    )
    second_result = ingest_odds_api_pitcher_lines_for_date(
        target_date=date(2026, 4, 21),
        output_dir=tmp_path,
        client=StubOddsClient(),
        now=iter(
            [
                datetime(2026, 4, 21, 19, 0, tzinfo=UTC),
                datetime(2026, 4, 21, 19, 1, tzinfo=UTC),
                datetime(2026, 4, 21, 19, 2, tzinfo=UTC),
            ]
        ).__next__,
    )

    run_dirs = sorted(
        (
            tmp_path / "normalized" / "the_odds_api" / "date=2026-04-21"
        ).glob("run=*")
    )
    raw_snapshot_paths = sorted(
        (
            tmp_path
            / "raw"
            / "the_odds_api"
            / "date=2026-04-21"
            / "event_odds"
            / "event_id=odds-event-1"
        ).glob("captured_at=*.json")
    )

    assert first_result.run_id != second_result.run_id
    assert len(run_dirs) == 2
    assert len(raw_snapshot_paths) == 2
    assert run_dirs[0].joinpath("prop_line_snapshots.jsonl").exists()
    assert run_dirs[1].joinpath("prop_line_snapshots.jsonl").exists()


def test_normalize_events_and_props_handle_unmatched_games_with_synthetic_player_ids() -> None:
    mappings = normalize_events_payload(
        [
            {
                "id": "odds-event-2",
                "sport_key": "baseball_mlb",
                "commence_time": "2026-04-21T23:15:00Z",
                "home_team": "Cleveland Guardians",
                "away_team": "Houston Astros",
            }
        ],
        target_date=date(2026, 4, 21),
        captured_at=datetime(2026, 4, 21, 18, 1, tzinfo=UTC),
        games=(
            GameRecord(
                game_pk=824448,
                official_date="2026-04-21",
                commence_time=datetime(2026, 4, 21, 22, 10, tzinfo=UTC),
                captured_at=datetime(2026, 4, 21, 17, 0, tzinfo=UTC),
                status="Pre-Game",
                status_code="P",
                venue_id=5,
                venue_name="Progressive Field",
                home_team_id=114,
                home_team_abbreviation="CLE",
                home_team_name="Cleveland Guardians",
                away_team_id=117,
                away_team_abbreviation="HOU",
                away_team_name="Houston Astros",
                game_number=1,
                double_header="N",
                day_night="night",
                odds_matchup_key="2026-04-21|HOU|CLE|2026-04-21T22:10:00Z",
            ),
        ),
    )

    assert len(mappings) == 1
    assert mappings[0].match_status == "unmatched"
    assert mappings[0].game_pk is None

    snapshots, skipped_groups = normalize_event_odds_payload(
        {
            "id": "odds-event-2",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "title": "FanDuel",
                    "last_update": "2026-04-21T18:05:00Z",
                    "markets": [
                        {
                            "key": "pitcher_strikeouts",
                            "last_update": "2026-04-21T18:05:30Z",
                            "outcomes": [
                                {
                                    "name": "Over",
                                    "description": "Ryan Weiss",
                                    "price": -115,
                                    "point": 5.5,
                                },
                                {
                                    "name": "Under",
                                    "description": "Ryan Weiss",
                                    "price": -105,
                                    "point": 5.5,
                                },
                            ],
                        }
                    ],
                }
            ],
        },
        captured_at=datetime(2026, 4, 21, 18, 6, tzinfo=UTC),
        mapping=mappings[0],
        probable_starter_lookup={},
    )

    assert skipped_groups == 0
    assert len(snapshots) == 1
    assert snapshots[0].player_id == "odds-player:ryan-weiss"
    assert snapshots[0].pitcher_mlb_id is None
    assert snapshots[0].match_status == "unmatched"
