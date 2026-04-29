import csv
import json
from pathlib import Path

from mlb_props_lab.targets import (
    PREGAME_FIELDS,
    TARGET_FIELDS,
    build_pitcher_start_target_artifacts,
    build_pitcher_start_targets,
)


def test_pitcher_start_target_parser_joins_identity_and_separates_fields() -> None:
    starts = [
        {
            "game_pk": "7001",
            "game_date": "2024-04-20",
            "game_time_utc": "2024-04-20T23:05:00Z",
            "pregame_as_of": "2024-04-20T22:35:00Z",
            "game_status": "Final",
            "game_completed_at": "2024-04-21T02:02:00Z",
            "pitcher_mlb_id": "111",
            "pitcher_name": "Raw Starter",
            "team_id": "1",
            "team_abbr": "AAA",
            "opponent_team_id": "2",
            "opponent_team_abbr": "BBB",
            "is_home": "yes",
            "p_throws": "",
            "started_game": "yes",
            "source_updated_at": "2024-04-21T02:05:00Z",
            "final_strikeouts": "8",
            "batters_faced": "25",
            "pitches": "99",
            "outs_recorded": "17",
        }
    ]
    identities = [
        {"pitcher_mlb_id": "111", "pitcher_name": "Canonical Starter", "pitcher_hand": "R"}
    ]

    result = build_pitcher_start_targets(starts, identities)

    assert result.summary["accepted_row_count"] == 1
    row = result.accepted_rows[0]
    assert row["pregame_pitcher_name"] == "Canonical Starter"
    assert row["pregame_pitcher_hand"] == "R"
    assert row["pregame_home_away"] == "home"
    assert row["target_final_strikeouts"] == 8
    assert row["target_innings_pitched"] == "5.2"
    assert row["target_available_at"] == row["completion_completed_at"]
    assert set(PREGAME_FIELDS).isdisjoint(TARGET_FIELDS)
    assert all(field.startswith("pregame_") for field in PREGAME_FIELDS)
    assert all(field.startswith("target_") for field in TARGET_FIELDS)


def test_pitcher_start_target_parser_reports_rejections() -> None:
    starts = [
        _start("8001", "111"),
        _start("8001", "111"),
        _start("8002", "111", final_strikeouts=""),
        _start("8003", "999"),
        _start("8004", "111", pregame_as_of="2024-04-20T23:10:00Z"),
    ]
    identities = [{"pitcher_mlb_id": "111", "pitcher_name": "Sample Starter", "pitcher_hand": "R"}]

    result = build_pitcher_start_targets(starts, identities)

    assert result.summary["accepted_row_count"] == 1
    assert result.summary["rejected_row_count"] == 4
    assert result.summary["duplicate_start_count"] == 1
    assert result.summary["missing_target_count"] == 1
    assert result.summary["unresolved_identity_count"] == 1
    assert result.summary["timestamp_rejection_count"] == 1
    reasons = [row["reasons"] for row in result.rejected_rows]
    assert any("duplicate_pitcher_start" in reason for reason in reasons)
    assert any("missing_target_fields" in reason for reason in reasons)
    assert any("unresolved_pitcher_identity" in reason for reason in reasons)
    assert any("timestamp_invalid_pregame_as_of" in reason for reason in reasons)


def test_pitcher_start_target_build_writes_fixture_artifacts(tmp_path: Path) -> None:
    build = build_pitcher_start_target_artifacts(
        issue="AGE-319",
        output_dir=tmp_path,
        run_id="target-sample",
    )

    manifest = json.loads(build.manifest_path.read_text(encoding="utf-8"))
    target_rows = list(csv.DictReader(build.target_table_path.open(encoding="utf-8")))
    rejected_rows = list(csv.DictReader(build.rejected_rows_path.open(encoding="utf-8")))

    assert manifest["report_type"] == "pitcher_start_targets"
    assert manifest["summary"]["raw_row_count"] == 6
    assert manifest["summary"]["accepted_row_count"] == 2
    assert manifest["summary"]["rejected_row_count"] == 4
    assert manifest["summary"]["missing_target_count"] == 1
    assert manifest["summary"]["duplicate_start_count"] == 1
    assert target_rows[0]["pregame_pitcher_hand"] == "R"
    assert target_rows[1]["pregame_pitcher_hand"] == "L"
    assert "target_final_strikeouts" in target_rows[0]
    assert any("duplicate_pitcher_start" in row["reasons"] for row in rejected_rows)
    assert all(path.exists() for path in build.visual_paths)


def _start(
    game_pk: str,
    pitcher_id: str,
    *,
    final_strikeouts: str = "5",
    pregame_as_of: str = "2024-04-20T22:35:00Z",
) -> dict[str, str]:
    return {
        "game_pk": game_pk,
        "game_date": "2024-04-20",
        "game_time_utc": "2024-04-20T23:05:00Z",
        "pregame_as_of": pregame_as_of,
        "game_status": "Final",
        "game_completed_at": "2024-04-21T02:02:00Z",
        "pitcher_mlb_id": pitcher_id,
        "pitcher_name": "Sample Starter",
        "team_id": "1",
        "team_abbr": "AAA",
        "opponent_team_id": "2",
        "opponent_team_abbr": "BBB",
        "is_home": "yes",
        "p_throws": "R",
        "started_game": "yes",
        "source_updated_at": "2024-04-21T02:05:00Z",
        "final_strikeouts": final_strikeouts,
        "batters_faced": "21",
        "pitches": "88",
        "outs_recorded": "15",
    }
