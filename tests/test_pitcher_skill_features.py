from __future__ import annotations

import csv
from datetime import UTC, date, datetime
from io import StringIO
import json
from pathlib import Path

from mlb_props_stack.pitcher_skill_features import build_pitcher_skill_features


HEADERS = [
    "game_date",
    "game_pk",
    "pitcher",
    "pitch_type",
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "pfx_x",
    "pfx_z",
    "description",
    "events",
    "strikes",
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
    pitcher: int,
    description: str,
    events: str,
    strikes: int,
    pitch_type: str = "FF",
    release_speed: float = 95.0,
) -> dict[str, object]:
    return {
        "game_date": game_date,
        "game_pk": game_pk,
        "pitcher": pitcher,
        "pitch_type": pitch_type,
        "release_speed": release_speed,
        "release_spin_rate": 2300,
        "release_extension": 6.3,
        "pfx_x": 0.4,
        "pfx_z": 1.2,
        "description": description,
        "events": events,
        "strikes": strikes,
    }


def _write_dataset_run(tmp_path: Path) -> Path:
    run_dir = (
        tmp_path
        / "normalized"
        / "starter_strikeout_training_dataset"
        / "start=2024-03-20_end=2024-04-10"
        / "run=20260425T120000Z"
    )
    raw_path = tmp_path / "raw" / "statcast.csv"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(
        _csv_text(
            [
                _pitch(
                    game_date="2024-03-20",
                    game_pk=1,
                    pitcher=100,
                    description="called_strike",
                    events="field_out",
                    strikes=2,
                    release_speed=92.0,
                ),
                _pitch(
                    game_date="2024-04-01",
                    game_pk=2,
                    pitcher=100,
                    description="swinging_strike",
                    events="strikeout",
                    strikes=2,
                    pitch_type="SL",
                    release_speed=96.0,
                ),
                _pitch(
                    game_date="2024-04-01",
                    game_pk=2,
                    pitcher=100,
                    description="ball",
                    events="walk",
                    strikes=0,
                    pitch_type="CH",
                    release_speed=86.0,
                ),
                _pitch(
                    game_date="2024-04-01",
                    game_pk=3,
                    pitcher=200,
                    description="hit_into_play",
                    events="field_out",
                    strikes=1,
                    release_speed=91.0,
                ),
                _pitch(
                    game_date="2024-04-10",
                    game_pk=4,
                    pitcher=100,
                    description="swinging_strike",
                    events="strikeout",
                    strikes=2,
                    release_speed=99.0,
                ),
            ]
        ),
        encoding="utf-8",
    )
    run_dir.mkdir(parents=True)
    (run_dir / "source_manifest.jsonl").write_text(
        json.dumps({"raw_path": str(raw_path)}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dataset_row = {
        "training_row_id": "starter-training:2024-04-10:4:100",
        "official_date": "2024-04-10",
        "season": 2024,
        "game_pk": 4,
        "pitcher_id": 100,
        "pitcher_name": "Test Starter",
        "team_abbreviation": "CLE",
        "opponent_team_abbreviation": "HOU",
        "starter_strikeouts": 12,
    }
    (run_dir / "starter_game_training_dataset.jsonl").write_text(
        json.dumps(dataset_row, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_pitcher_skill_features_use_only_prior_games_and_bucket_rest(tmp_path):
    dataset_run_dir = _write_dataset_run(tmp_path)

    result = build_pitcher_skill_features(
        start_date=date(2024, 3, 20),
        end_date=date(2024, 4, 10),
        output_dir=tmp_path,
        dataset_run_dir=dataset_run_dir,
        now=lambda: datetime(2026, 4, 25, 17, 0, tzinfo=UTC),
    )

    rows = [
        json.loads(line)
        for line in result.feature_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = json.loads(result.feature_report_path.read_text(encoding="utf-8"))
    row = rows[0]

    assert result.feature_row_count == 1
    assert result.pitch_row_count == 5
    assert row["leakage_policy_status"] == "ok_prior_games_only"
    assert "starter_strikeouts" not in row
    assert row["prior_pitch_count"] == 3
    assert row["career_k_rate"] == 0.333333
    assert row["career_k_rate_shrunk"] != row["career_k_rate"]
    assert row["recent_15d_pitch_count"] == 2
    assert row["recent_15d_k_rate"] == 0.5
    assert row["rest_bucket"] == "extra_rest"
    assert row["rest_days_capped"] == 9
    assert row["pitch_type_usage"] == {"CH": 0.333333, "FF": 0.333333, "SL": 0.333333}
    assert row["release_speed_delta_vs_pitcher_baseline"] == -0.333333
    assert report["leakage_policy"]["status"] == "ok"
    assert report["rest_policy"]["raw_rest_days_primary_driver"] is False
    assert report["rest_policy"]["long_layoff_has_unbounded_positive_numeric_feature"] is False
    assert report["top_correlations_by_season"]["2024"] == []


def test_pitcher_skill_features_cap_long_layoff(tmp_path):
    dataset_run_dir = _write_dataset_run(tmp_path)
    dataset_path = dataset_run_dir / "starter_game_training_dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "training_row_id": "starter-training:2024-04-10:4:100",
                "official_date": "2024-04-10",
                "season": 2024,
                "game_pk": 4,
                "pitcher_id": 100,
                "pitcher_name": "Test Starter",
                "team_abbreviation": "CLE",
                "opponent_team_abbreviation": "HOU",
                "starter_strikeouts": 12,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    raw_path = tmp_path / "raw" / "statcast.csv"
    raw_path.write_text(
        _csv_text(
            [
                _pitch(
                    game_date="2024-03-01",
                    game_pk=1,
                    pitcher=100,
                    description="swinging_strike",
                    events="strikeout",
                    strikes=2,
                )
            ]
        ),
        encoding="utf-8",
    )

    result = build_pitcher_skill_features(
        start_date=date(2024, 4, 10),
        end_date=date(2024, 4, 10),
        output_dir=tmp_path,
        dataset_run_dir=dataset_run_dir,
        now=lambda: datetime(2026, 4, 25, 17, 5, tzinfo=UTC),
    )

    row = json.loads(result.feature_path.read_text(encoding="utf-8").splitlines()[0])

    assert row["rest_bucket"] == "long_layoff"
    assert row["rest_days_capped"] == 14
    assert row["long_layoff_flag"] is True
