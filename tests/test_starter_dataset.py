from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from urllib.parse import parse_qs, urlparse

from mlb_props_stack.starter_dataset import build_starter_strikeout_dataset
from tests.test_modeling import (
    FakeStatcastClient,
    _build_outcome_csv,
    _csv_text,
    _seed_feature_run,
)


DIRECT_HEADERS = [
    "pitch_type",
    "game_date",
    "player_name",
    "batter",
    "pitcher",
    "events",
    "description",
    "game_type",
    "stand",
    "p_throws",
    "home_team",
    "away_team",
    "balls",
    "strikes",
    "inning",
    "inning_topbot",
    "game_pk",
    "at_bat_number",
    "pitch_number",
]


def _direct_csv_text(rows: list[dict[str, object]]) -> str:
    import csv
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=DIRECT_HEADERS)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


class FakeDirectStatcastClient:
    def __init__(self, responses: dict[tuple[str, str], str]) -> None:
        self.responses = responses

    def fetch_csv(self, url: str) -> str:
        query = parse_qs(urlparse(url).query)
        return self.responses[(query["game_date_gt"][0], query["game_date_lt"][0])]


def _feature_row(
    *,
    official_date: date,
    pitcher_index: int,
    game_pk: int,
    pitcher_id: int,
    home_away: str,
) -> dict[str, object]:
    return {
        "game_pk": game_pk,
        "pitcher_id": pitcher_id,
        "pitcher_name": f"Dataset Pitcher {pitcher_index}",
        "team_side": home_away,
        "team_abbreviation": "CLE" if home_away == "home" else "HOU",
        "opponent_team_abbreviation": "HOU" if home_away == "home" else "CLE",
        "opponent_team_name": "Houston Astros" if home_away == "home" else "Cleveland Guardians",
        "pitcher_feature_row_id": f"pitcher-feature:{game_pk}:{pitcher_id}",
        "lineup_feature_row_id": f"lineup-feature:{game_pk}:{pitcher_id}",
        "game_context_feature_row_id": f"game-context:{game_pk}:{pitcher_id}",
        "lineup_snapshot_id": f"lineup:{game_pk}:{official_date.isoformat()}",
        "pitcher_hand": "R" if pitcher_index == 0 else "L",
        "pitch_sample_size": 440 + pitcher_index,
        "plate_appearance_sample_size": 100 + pitcher_index,
        "pitcher_k_rate": 0.24 + (0.01 * pitcher_index),
        "swinging_strike_rate": 0.12,
        "csw_rate": 0.28,
        "pitch_type_usage": {"FF": 0.55, "SL": 0.45},
        "average_release_speed": 94.0,
        "release_speed_delta_vs_baseline": 0.0,
        "average_release_extension": 6.1,
        "release_extension_delta_vs_baseline": 0.0,
        "recent_batters_faced": 72,
        "recent_pitch_count": 275,
        "rest_days": 5,
        "last_start_pitch_count": 91,
        "last_start_batters_faced": 24,
        "lineup_status": "projected",
        "lineup_is_confirmed": False,
        "lineup_size": 9,
        "available_batter_feature_count": 9,
        "projected_lineup_k_rate": 0.23,
        "projected_lineup_k_rate_vs_pitcher_hand": 0.24,
        "projected_lineup_chase_rate": 0.31,
        "projected_lineup_contact_rate": 0.74,
        "lineup_continuity_count": 6,
        "lineup_continuity_ratio": 0.67,
        "lineup_player_ids": [710000 + slot for slot in range(9)],
        "home_away": home_away,
        "day_night": "night",
        "double_header": "N",
        "expected_leash_pitch_count": 94.0,
        "expected_leash_batters_faced": 24.0,
    }


def test_build_starter_strikeout_dataset_writes_coverage_and_policy_artifacts(tmp_path):
    start_date = date(2022, 4, 7)
    end_date = start_date + timedelta(days=4)
    responses: dict[tuple[str, int], str] = {}

    for date_index in range(5):
        official_date = start_date + timedelta(days=date_index)
        feature_rows = []
        for pitcher_index in range(2):
            pitcher_id = 720000 + (date_index * 10) + pitcher_index
            game_pk = 920000 + (date_index * 10) + pitcher_index
            home_away = "home" if pitcher_index == 0 else "away"
            feature_rows.append(
                _feature_row(
                    official_date=official_date,
                    pitcher_index=pitcher_index,
                    game_pk=game_pk,
                    pitcher_id=pitcher_id,
                    home_away=home_away,
                )
            )
            if date_index == 2 and pitcher_index == 1:
                responses[(official_date.isoformat(), pitcher_id)] = _csv_text([])
                continue
            responses[(official_date.isoformat(), pitcher_id)] = _build_outcome_csv(
                official_date=official_date,
                game_pk=game_pk,
                pitcher_id=pitcher_id,
                strikeout_count=5 + pitcher_index,
                plate_appearance_count=8 if date_index == 0 and pitcher_index == 0 else 24,
                home_team="CLE" if home_away == "home" else "HOU",
                away_team="HOU" if home_away == "home" else "CLE",
                pitcher_hand="R" if pitcher_index == 0 else "L",
            )
        _seed_feature_run(tmp_path, official_date=official_date, feature_rows=feature_rows)

    result = build_starter_strikeout_dataset(
        start_date=start_date,
        end_date=end_date,
        output_dir=tmp_path,
        client=FakeStatcastClient(responses),
        now=iter(
            datetime(2026, 4, 25, 14, 0, tzinfo=UTC) + timedelta(minutes=index)
            for index in range(30)
        ).__next__,
    )

    dataset_rows = [
        json.loads(line)
        for line in result.dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    missing_rows = [
        json.loads(line)
        for line in result.missing_targets_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    coverage = json.loads(result.coverage_report_path.read_text(encoding="utf-8"))
    schema_drift = json.loads(result.schema_drift_report_path.read_text(encoding="utf-8"))
    timestamp_policy = result.timestamp_policy_path.read_text(encoding="utf-8")

    assert result.row_count == 9
    assert result.source_mode == "feature_runs"
    assert result.source_date_count == 5
    assert result.missing_target_count == 1
    assert result.excluded_start_count == 1
    assert result.season_count == 1
    assert len(dataset_rows) == 9
    assert len({(row["official_date"], row["game_pk"], row["pitcher_id"]) for row in dataset_rows}) == 9
    assert dataset_rows[0]["starter_strikeouts"] == 5
    assert dataset_rows[0]["pitch_clock_era"] == "pre_pitch_clock"
    assert dataset_rows[0]["starter_role_edge_case"] == "short_start_review"
    assert dataset_rows[0]["source_references"]["pitcher_feature_row_id"].startswith("pitcher-feature:")
    assert all(row["timestamp_policy_status"] == "ok" for row in dataset_rows)
    assert missing_rows[0]["reason"] == "missing_same_game_statcast_outcome"
    assert coverage["row_counts"]["dataset_rows"] == 9
    assert coverage["row_counts"]["missing_targets"] == 1
    assert coverage["source_mode"] == "feature_runs"
    assert coverage["row_counts_by_team"] == {"CLE": 5, "HOU": 4}
    assert coverage["coverage_status"]["preferred_5_to_7_seasons_achieved"] is False
    assert coverage["source_freshness"]["features_as_of"]["min"] == "2022-04-07T15:45:00Z"
    assert coverage["source_freshness"]["outcome_captured_at"]["max"] == "2026-04-25T14:10:00Z"
    assert schema_drift["row_count"] == 9
    assert any(field["field"] == "starter_strikeouts" for field in schema_drift["fields"])
    assert "Same-game Statcast outcome pulls define only the target" in timestamp_policy


def test_build_starter_strikeout_dataset_records_missing_source_dates(tmp_path):
    start_date = date(2024, 4, 1)
    official_date = start_date + timedelta(days=1)
    pitcher_id = 730001
    game_pk = 930001
    _seed_feature_run(
        tmp_path,
        official_date=official_date,
        feature_rows=[
            _feature_row(
                official_date=official_date,
                pitcher_index=0,
                game_pk=game_pk,
                pitcher_id=pitcher_id,
                home_away="home",
            )
        ],
    )

    result = build_starter_strikeout_dataset(
        start_date=start_date,
        end_date=start_date + timedelta(days=2),
        output_dir=tmp_path,
        client=FakeStatcastClient(
            {
                (official_date.isoformat(), pitcher_id): _build_outcome_csv(
                    official_date=official_date,
                    game_pk=game_pk,
                    pitcher_id=pitcher_id,
                    strikeout_count=7,
                    plate_appearance_count=25,
                    home_team="CLE",
                    away_team="HOU",
                    pitcher_hand="R",
                )
            }
        ),
        now=lambda: datetime(2026, 4, 25, 15, 0, tzinfo=UTC),
    )

    coverage = json.loads(result.coverage_report_path.read_text(encoding="utf-8"))

    assert result.requested_date_count == 3
    assert result.source_date_count == 1
    assert coverage["date_window"]["missing_source_dates"] == ["2024-04-01", "2024-04-03"]


def test_build_starter_strikeout_dataset_falls_back_to_direct_statcast_pitch_logs(tmp_path):
    target_date = date(2024, 4, 1)
    direct_csv = _direct_csv_text(
        [
            {
                "pitch_type": "FF",
                "game_date": target_date.isoformat(),
                "player_name": "Home Starter",
                "batter": 1,
                "pitcher": 800001,
                "events": "strikeout",
                "description": "swinging_strike",
                "game_type": "R",
                "stand": "R",
                "p_throws": "R",
                "home_team": "CLE",
                "away_team": "HOU",
                "balls": 0,
                "strikes": 2,
                "inning": 1,
                "inning_topbot": "Top",
                "game_pk": 940001,
                "at_bat_number": 1,
                "pitch_number": 1,
            },
            {
                "pitch_type": "SL",
                "game_date": target_date.isoformat(),
                "player_name": "Home Starter",
                "batter": 2,
                "pitcher": 800001,
                "events": "field_out",
                "description": "hit_into_play",
                "game_type": "R",
                "stand": "R",
                "p_throws": "R",
                "home_team": "CLE",
                "away_team": "HOU",
                "balls": 0,
                "strikes": 1,
                "inning": 1,
                "inning_topbot": "Top",
                "game_pk": 940001,
                "at_bat_number": 2,
                "pitch_number": 1,
            },
            {
                "pitch_type": "FF",
                "game_date": target_date.isoformat(),
                "player_name": "Away Starter",
                "batter": 3,
                "pitcher": 800002,
                "events": "strikeout",
                "description": "called_strike",
                "game_type": "R",
                "stand": "L",
                "p_throws": "L",
                "home_team": "CLE",
                "away_team": "HOU",
                "balls": 1,
                "strikes": 2,
                "inning": 1,
                "inning_topbot": "Bot",
                "game_pk": 940001,
                "at_bat_number": 3,
                "pitch_number": 1,
            },
        ]
    )

    result = build_starter_strikeout_dataset(
        start_date=target_date,
        end_date=target_date,
        output_dir=tmp_path,
        client=FakeDirectStatcastClient({(target_date.isoformat(), target_date.isoformat()): direct_csv}),
        now=lambda: datetime(2026, 4, 25, 16, 0, tzinfo=UTC),
        max_fetch_workers=1,
    )

    dataset_rows = [
        json.loads(line)
        for line in result.dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    coverage = json.loads(result.coverage_report_path.read_text(encoding="utf-8"))
    manifest_rows = [
        json.loads(line)
        for line in result.source_manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert result.source_mode == "direct_statcast_pitch_log"
    assert result.source_date_count == 1
    assert result.row_count == 2
    assert [row["home_away"] for row in dataset_rows] == ["home", "away"]
    assert [row["starter_strikeouts"] for row in dataset_rows] == [1, 1]
    assert dataset_rows[0]["starter_role_status"] == "inferred_from_first_pitch_for_fielding_team"
    assert dataset_rows[0]["pregame_reference_status"] == "not_applicable_target_foundation"
    assert dataset_rows[0]["target_source_status"] == "postgame_statcast_pitch_log_target_only"
    assert dataset_rows[0]["timestamp_policy_status"] == "target_only_no_pregame_features"
    assert coverage["source_mode"] == "direct_statcast_pitch_log"
    assert coverage["source_chunks"]["chunk_count"] == 1
    assert coverage["source_chunks"]["cap_warning_count"] == 0
    assert manifest_rows[0]["pitch_row_count"] == 3
