from __future__ import annotations

import csv
from datetime import UTC, date, datetime
from io import StringIO
import json
from pathlib import Path

import pytest

from mlb_props_stack.lineup_matchup_features import build_lineup_matchup_features


HEADERS = [
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
    "pitch_type",
    "description",
    "events",
    "stand",
    "p_throws",
    "zone",
    "home_team",
    "away_team",
    "inning_topbot",
]


def _csv_text(rows: list[dict[str, object]]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=HEADERS)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def _pitch(
    *,
    game_date: str,
    game_pk: int,
    at_bat_number: int,
    pitcher: int,
    batter: int,
    pitch_type: str,
    description: str,
    events: str,
    stand: str = "R",
    p_throws: str = "R",
    zone: int = 11,
    inning_topbot: str = "Top",
) -> dict[str, object]:
    return {
        "game_date": game_date,
        "game_pk": game_pk,
        "at_bat_number": at_bat_number,
        "pitch_number": 1,
        "pitcher": pitcher,
        "batter": batter,
        "pitch_type": pitch_type,
        "description": description,
        "events": events,
        "stand": stand,
        "p_throws": p_throws,
        "zone": zone,
        "home_team": "CLE",
        "away_team": "HOU",
        "inning_topbot": inning_topbot,
    }


def _write_dataset_run(
    tmp_path: Path,
    *,
    dataset_rows: list[dict[str, object]],
    pitch_rows: list[dict[str, object]],
) -> Path:
    run_dir = (
        tmp_path
        / "normalized"
        / "starter_strikeout_training_dataset"
        / "start=2024-04-01_end=2024-04-03"
        / "run=20260425T120000Z"
    )
    raw_path = tmp_path / "raw" / "statcast.csv"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(_csv_text(pitch_rows), encoding="utf-8")
    run_dir.mkdir(parents=True)
    (run_dir / "source_manifest.jsonl").write_text(
        json.dumps({"raw_path": str(raw_path)}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "starter_game_training_dataset.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in dataset_rows),
        encoding="utf-8",
    )
    return run_dir


def _starter_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "training_row_id": "starter-training:2024-04-03:10:900",
        "official_date": "2024-04-03",
        "season": 2024,
        "game_pk": 10,
        "pitcher_id": 900,
        "pitcher_name": "Starter",
        "team_abbreviation": "CLE",
        "opponent_team_abbreviation": "HOU",
        "pitcher_hand": "R",
        "starter_strikeouts": 7,
    }
    row.update(overrides)
    return row


def test_lineup_matchup_features_project_prior_lineup_without_same_game_leakage(tmp_path):
    dataset_run_dir = _write_dataset_run(
        tmp_path,
        dataset_rows=[_starter_row()],
        pitch_rows=[
            _pitch(
                game_date="2024-04-01",
                game_pk=1,
                at_bat_number=1,
                pitcher=800,
                batter=101,
                pitch_type="FF",
                description="swinging_strike",
                events="strikeout",
            ),
            _pitch(
                game_date="2024-04-01",
                game_pk=1,
                at_bat_number=2,
                pitcher=801,
                batter=102,
                pitch_type="SL",
                description="hit_into_play",
                events="field_out",
            ),
            _pitch(
                game_date="2024-04-02",
                game_pk=2,
                at_bat_number=1,
                pitcher=900,
                batter=201,
                pitch_type="FF",
                description="swinging_strike",
                events="strikeout",
                inning_topbot="Bot",
            ),
            _pitch(
                game_date="2024-04-03",
                game_pk=10,
                at_bat_number=1,
                pitcher=900,
                batter=999,
                pitch_type="FF",
                description="swinging_strike",
                events="strikeout",
            ),
        ],
    )

    result = build_lineup_matchup_features(
        start_date=date(2024, 4, 1),
        end_date=date(2024, 4, 3),
        output_dir=tmp_path,
        dataset_run_dir=dataset_run_dir,
        now=lambda: datetime(2026, 4, 25, 18, 30, tzinfo=UTC),
    )

    lineup_row = json.loads(result.feature_path.read_text(encoding="utf-8").splitlines()[0])
    batter_rows = [
        json.loads(line)
        for line in result.batter_feature_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = json.loads(result.feature_report_path.read_text(encoding="utf-8"))

    assert lineup_row["lineup_status"] == "projected_from_prior_team_game"
    assert lineup_row["lineup_is_confirmed"] is False
    assert lineup_row["lineup_player_ids"] == [101, 102]
    assert 999 not in lineup_row["lineup_player_ids"]
    assert lineup_row["leakage_policy_status"] == "ok_prior_games_only"
    assert result.batter_feature_row_count == 2
    assert {row["batter_id"] for row in batter_rows} == {101, 102}
    assert report["missingness"]["no_confirmed_lineup"] == 1
    assert report["missingness"]["no_projection"] == 0


def test_lineup_matchup_features_fail_when_dataset_artifact_is_missing(tmp_path):
    missing_run_dir = tmp_path / "missing-dataset-run"

    with pytest.raises(FileNotFoundError, match="No starter-game dataset rows"):
        build_lineup_matchup_features(
            start_date=date(2024, 4, 1),
            end_date=date(2024, 4, 3),
            output_dir=tmp_path,
            dataset_run_dir=missing_run_dir,
            now=lambda: datetime(2026, 4, 25, 18, 30, tzinfo=UTC),
        )


def test_lineup_matchup_features_respect_handedness_and_pitch_type_matchups(tmp_path):
    dataset_run_dir = _write_dataset_run(
        tmp_path,
        dataset_rows=[
            _starter_row(
                lineup_status="confirmed",
                lineup_is_confirmed=True,
                lineup_snapshot_id="lineup:10:confirmed",
                lineup_player_ids=[101, 102],
            )
        ],
        pitch_rows=[
            _pitch(
                game_date="2024-04-01",
                game_pk=1,
                at_bat_number=1,
                pitcher=800,
                batter=101,
                pitch_type="FF",
                description="swinging_strike",
                events="strikeout",
                p_throws="R",
            ),
            _pitch(
                game_date="2024-04-01",
                game_pk=1,
                at_bat_number=2,
                pitcher=801,
                batter=101,
                pitch_type="SL",
                description="hit_into_play",
                events="field_out",
                p_throws="L",
            ),
            _pitch(
                game_date="2024-04-01",
                game_pk=1,
                at_bat_number=3,
                pitcher=800,
                batter=102,
                pitch_type="SL",
                description="swinging_strike",
                events="strikeout",
                p_throws="R",
            ),
            _pitch(
                game_date="2024-04-02",
                game_pk=2,
                at_bat_number=1,
                pitcher=900,
                batter=201,
                pitch_type="FF",
                description="called_strike",
                events="field_out",
                p_throws="R",
            ),
            _pitch(
                game_date="2024-04-02",
                game_pk=2,
                at_bat_number=2,
                pitcher=900,
                batter=202,
                pitch_type="SL",
                description="swinging_strike",
                events="strikeout",
                p_throws="R",
            ),
        ],
    )

    result = build_lineup_matchup_features(
        start_date=date(2024, 4, 1),
        end_date=date(2024, 4, 3),
        output_dir=tmp_path,
        dataset_run_dir=dataset_run_dir,
        now=lambda: datetime(2026, 4, 25, 18, 35, tzinfo=UTC),
    )

    lineup_row = json.loads(result.feature_path.read_text(encoding="utf-8").splitlines()[0])
    batter_rows = [
        json.loads(line)
        for line in result.batter_feature_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    first_batter = next(row for row in batter_rows if row["batter_id"] == 101)
    second_batter = next(row for row in batter_rows if row["batter_id"] == 102)

    assert lineup_row["lineup_status"] == "confirmed"
    assert lineup_row["lineup_is_confirmed"] is True
    assert lineup_row["pitcher_pitch_type_usage"] == {"FF": 0.5, "SL": 0.5}
    assert first_batter["career_k_rate"] == 0.5
    assert first_batter["career_k_rate_vs_pitcher_hand"] == 1.0
    assert first_batter["pitch_type_weakness"]["FF"]["whiff_rate"] == 1.0
    assert first_batter["pitch_type_weakness"]["SL"]["contact_rate"] == 1.0
    assert second_batter["career_k_rate_vs_pitcher_hand"] == 1.0
    assert lineup_row["arsenal_weighted_lineup_pitch_type_weakness"] is not None


def test_lineup_matchup_features_report_missing_projection_and_history(tmp_path):
    dataset_run_dir = _write_dataset_run(
        tmp_path,
        dataset_rows=[
            _starter_row(
                training_row_id="starter-training:2024-04-01:9:901",
                official_date="2024-04-01",
                game_pk=9,
                pitcher_id=901,
                starter_strikeouts=4,
            ),
            _starter_row(
                lineup_status="projected",
                lineup_is_confirmed=False,
                lineup_player_ids=[555],
            )
        ],
        pitch_rows=[
            _pitch(
                game_date="2024-04-02",
                game_pk=2,
                at_bat_number=1,
                pitcher=900,
                batter=201,
                pitch_type="FF",
                description="called_strike",
                events="field_out",
            )
        ],
    )

    result = build_lineup_matchup_features(
        start_date=date(2024, 4, 1),
        end_date=date(2024, 4, 3),
        output_dir=tmp_path,
        dataset_run_dir=dataset_run_dir,
        now=lambda: datetime(2026, 4, 25, 18, 40, tzinfo=UTC),
    )

    lineup_rows = [
        json.loads(line)
        for line in result.feature_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    no_projection_row = next(row for row in lineup_rows if row["game_pk"] == 9)
    incomplete_row = next(row for row in lineup_rows if row["game_pk"] == 10)
    report = json.loads(result.feature_report_path.read_text(encoding="utf-8"))

    assert no_projection_row["lineup_status"] == "no_projection"
    assert no_projection_row["feature_status"] == "missing_lineup_projection"
    assert incomplete_row["lineup_status"] == "projected_snapshot"
    assert incomplete_row["feature_status"] == "incomplete_batter_history"
    assert incomplete_row["available_batter_feature_count"] == 0
    assert incomplete_row["incomplete_batter_history_count"] == 1
    assert report["missingness"]["no_projection"] == 1
    assert report["missingness"]["incomplete_batter_history"] == 1
