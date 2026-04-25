from __future__ import annotations

import csv
from datetime import UTC, date, datetime
from io import StringIO
import json
from pathlib import Path

from mlb_props_stack.workload_leash_features import build_workload_leash_features


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
    events: str = "field_out",
    inning_topbot: str = "Bot",
) -> dict[str, object]:
    return {
        "game_date": game_date,
        "game_pk": game_pk,
        "at_bat_number": at_bat_number,
        "pitch_number": 1,
        "pitcher": pitcher,
        "batter": batter,
        "pitch_type": "FF",
        "description": "hit_into_play",
        "events": events,
        "stand": "R",
        "p_throws": "R",
        "zone": 5,
        "home_team": "CLE",
        "away_team": "HOU",
        "inning_topbot": inning_topbot,
    }


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
        "starter_strikeouts": 6,
    }
    row.update(overrides)
    return row


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
        / "start=2024-03-20_end=2024-04-10"
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


def _pa_rows(
    *, game_date: str, game_pk: int, pitcher: int, plate_appearances: int
) -> list[dict[str, object]]:
    return [
        _pitch(
            game_date=game_date,
            game_pk=game_pk,
            at_bat_number=index + 1,
            pitcher=pitcher,
            batter=1000 + index,
        )
        for index in range(plate_appearances)
    ]


def test_workload_leash_features_use_prior_games_and_expected_opportunity(tmp_path):
    dataset_run_dir = _write_dataset_run(
        tmp_path,
        dataset_rows=[
            _starter_row(
                training_row_id="starter-training:2024-03-20:1:900",
                official_date="2024-03-20",
                game_pk=1,
                starter_strikeouts=4,
            ),
            _starter_row(
                training_row_id="starter-training:2024-04-01:2:900",
                official_date="2024-04-01",
                game_pk=2,
                starter_strikeouts=5,
            ),
            _starter_row(),
        ],
        pitch_rows=[
            *_pa_rows(game_date="2024-03-20", game_pk=1, pitcher=900, plate_appearances=22),
            *_pa_rows(game_date="2024-04-01", game_pk=2, pitcher=900, plate_appearances=24),
            *_pa_rows(game_date="2024-04-03", game_pk=10, pitcher=900, plate_appearances=30),
        ],
    )

    result = build_workload_leash_features(
        start_date=date(2024, 4, 3),
        end_date=date(2024, 4, 3),
        output_dir=tmp_path,
        dataset_run_dir=dataset_run_dir,
        now=lambda: datetime(2026, 4, 25, 19, 0, tzinfo=UTC),
    )

    rows = [
        json.loads(line)
        for line in result.feature_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = json.loads(result.feature_report_path.read_text(encoding="utf-8"))
    row = rows[0]

    assert result.feature_row_count == 1
    assert row["leakage_policy_status"] == "ok_prior_games_only"
    assert "starter_strikeouts" not in row
    assert row["prior_start_count"] == 2
    assert row["last_3_starts_batters_faced_mean"] == 23.0
    assert row["expected_leash_batters_faced"] == 23.0
    assert row["season_reached_22_batters_rate"] == 1.0
    assert row["rest_bucket"] == "short_rest"
    assert row["rest_days_capped"] == 2
    assert row["feature_group"] == "expected_opportunity"
    assert row["feature_usage"] == "opportunity_volume_not_strikeout_skill"
    assert report["leakage_policy"]["status"] == "ok"
    assert report["rest_policy"]["raw_rest_days_primary_driver"] is False


def test_workload_leash_features_classify_long_layoff_without_injury_guess(tmp_path):
    dataset_run_dir = _write_dataset_run(
        tmp_path,
        dataset_rows=[_starter_row(official_date="2024-04-10", game_pk=10)],
        pitch_rows=[
            *_pa_rows(game_date="2024-03-20", game_pk=1, pitcher=900, plate_appearances=21),
            *_pa_rows(game_date="2024-04-10", game_pk=10, pitcher=900, plate_appearances=26),
        ],
    )

    result = build_workload_leash_features(
        start_date=date(2024, 4, 10),
        end_date=date(2024, 4, 10),
        output_dir=tmp_path,
        dataset_run_dir=dataset_run_dir,
        now=lambda: datetime(2026, 4, 25, 19, 5, tzinfo=UTC),
    )

    row = json.loads(result.feature_path.read_text(encoding="utf-8").splitlines()[0])

    assert row["rest_bucket"] == "long_layoff"
    assert row["rest_days_capped"] == 14
    assert row["long_layoff_unknown_flag"] is True
    assert row["injury_context_status"] == "unknown_long_layoff"
    assert row["il_return_flag"] is False
    assert row["rehab_return_flag"] is False


def test_workload_leash_features_detect_prior_opener_bulk_pattern(tmp_path):
    dataset_run_dir = _write_dataset_run(
        tmp_path,
        dataset_rows=[_starter_row()],
        pitch_rows=[
            *_pa_rows(game_date="2024-03-30", game_pk=1, pitcher=900, plate_appearances=5),
            *_pa_rows(game_date="2024-04-01", game_pk=2, pitcher=900, plate_appearances=6),
            *_pa_rows(game_date="2024-04-03", game_pk=10, pitcher=900, plate_appearances=25),
        ],
    )

    result = build_workload_leash_features(
        start_date=date(2024, 4, 3),
        end_date=date(2024, 4, 3),
        output_dir=tmp_path,
        dataset_run_dir=dataset_run_dir,
        now=lambda: datetime(2026, 4, 25, 19, 10, tzinfo=UTC),
    )

    row = json.loads(result.feature_path.read_text(encoding="utf-8").splitlines()[0])

    assert row["opener_or_bulk_role_flag"] is True
    assert row["opener_or_bulk_role_source"] == "prior_starter_short_workload_pattern"
    assert row["prior_short_start_count"] == 2
