import csv
import json
from pathlib import Path

from mlb_props_lab.feature_registry import load_registry
from mlb_props_lab.statcast_features import (
    MATERIALIZED_STATCAST_FEATURE_IDS,
    build_statcast_feature_artifacts,
    materialize_statcast_features_for_target,
)


def test_statcast_feature_formulas_use_prior_pitch_rows() -> None:
    rows = [
        _pitch("called_strike", "", pitch_type="FF", speed="95.0", spin="2300", stand="L"),
        _pitch("swinging_strike", "", pitch_type="FF", speed="96.0", spin="2320", stand="L"),
        _pitch(
            "swinging_strike",
            "strikeout",
            pitch_type="SL",
            speed="85.0",
            spin="2450",
            stand="L",
        ),
        _pitch("foul", "", pitch_type="FF", speed="94.0", spin="2290", stand="R", at_bat="2"),
        _pitch("ball", "", pitch_type="CH", speed="86.0", spin="1800", stand="R", at_bat="2"),
        _pitch(
            "hit_into_play",
            "field_out",
            pitch_type="CH",
            speed="87.0",
            spin="1810",
            stand="R",
            at_bat="2",
        ),
        _pitch("ball", "", pitch_type="FF", speed="95.0", spin="2310", stand="R", at_bat="3"),
        _pitch("ball", "", pitch_type="FF", speed="95.2", spin="2315", stand="R", at_bat="3"),
        _pitch("ball", "", pitch_type="SL", speed="84.8", spin="2445", stand="R", at_bat="3"),
        _pitch("ball", "walk", pitch_type="SL", speed="85.1", spin="2455", stand="R", at_bat="3"),
    ]
    target = {
        "target_game_pk": "9001",
        "pitcher": "111",
        "pitcher_name": "Sample Starter",
        "target_game_date": "2024-04-10",
        "cutoff_at": "2024-04-10T16:00:00Z",
    }

    result = materialize_statcast_features_for_target(rows, target)
    features = result["features"]

    assert features["pitcher_k_rate_rolling"] == 0.333333
    assert features["pitcher_k9_rolling"] == 13.5
    assert features["pitcher_k_minus_bb_rate"] == 0.0
    assert features["pitcher_batters_faced_rolling"] == 3.0
    assert features["pitcher_pitches_per_start"] == 10.0
    assert features["pitcher_csw_rate"] == 0.3
    assert features["pitcher_whiff_rate"] == 0.5
    assert features["pitch_mix_by_type"] == {"CH": 0.2, "FF": 0.5, "SL": 0.3}
    assert features["release_velocity_by_type"]["FF"] == 95.04
    assert features["release_spin_rate_by_type"]["SL"] == 2450.0
    assert features["pitch_type_whiff_csw_contact"]["FF"]["whiff_rate"] == 0.5
    assert features["pitcher_platoon_k_bb_whiff"]["L"]["k_rate"] == 1.0
    assert features["pitcher_platoon_pitch_mix"]["R"] == {
        "CH": 0.285714,
        "FF": 0.428571,
        "SL": 0.285714,
    }


def test_statcast_feature_cutoff_excludes_same_game_and_future_rows() -> None:
    prior = _pitch(
        "swinging_strike",
        "strikeout",
        speed="95.0",
        available_at="2024-04-01T23:00:00Z",
    )
    same_game = _pitch(
        "swinging_strike",
        "strikeout",
        speed="105.0",
        available_at="2024-04-10T21:00:00Z",
        game_date="2024-04-10",
        game_pk="1002",
    )
    target = {
        "target_game_pk": "9001",
        "pitcher": "111",
        "pitcher_name": "Sample Starter",
        "target_game_date": "2024-04-10",
        "cutoff_at": "2024-04-10T16:00:00Z",
    }

    result = materialize_statcast_features_for_target([prior, same_game], target)

    assert result["source_pitch_count"] == 1
    assert result["skipped_same_or_future_pitch_count"] == 1
    assert result["features"]["release_velocity_by_type"] == {"FF": 95.0}
    assert result["features"]["pitcher_k_rate_rolling"] == 1.0


def test_statcast_build_writes_registered_feature_artifacts(tmp_path: Path) -> None:
    build = build_statcast_feature_artifacts(issue="AGE-317", output_dir=tmp_path, run_id="sample")
    registry_ids = {feature["id"] for feature in load_registry()["features"]}

    manifest = json.loads(build.manifest_path.read_text(encoding="utf-8"))
    feature_rows = list(csv.DictReader(build.feature_matrix_path.open(encoding="utf-8")))
    coverage_rows = list(csv.DictReader(build.coverage_path.open(encoding="utf-8")))

    assert set(manifest["materialized_feature_ids"]) == set(MATERIALIZED_STATCAST_FEATURE_IDS)
    assert set(manifest["materialized_feature_ids"]) <= registry_ids
    assert all(path.exists() for path in build.visual_paths)
    assert len(feature_rows) == 2
    assert "pitcher_k_rate_rolling" in feature_rows[0]
    assert "projected_lineup_handedness_mix" not in feature_rows[0]
    assert any(
        row["feature_id"] == "projected_lineup_handedness_mix"
        and row["materialized"] == "no"
        and "projected lineup" in row["missing_reason"]
        for row in coverage_rows
    )


def _pitch(
    description: str,
    events: str,
    *,
    pitch_type: str = "FF",
    speed: str = "95.0",
    spin: str = "2300",
    stand: str = "L",
    at_bat: str = "1",
    available_at: str = "2024-04-01T23:00:00Z",
    game_date: str = "2024-04-01",
    game_pk: str = "1001",
) -> dict[str, str]:
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "available_at": available_at,
        "pitcher": "111",
        "pitcher_name": "Sample Starter",
        "batter": f"20{at_bat}",
        "stand": stand,
        "p_throws": "R",
        "pitch_type": pitch_type,
        "release_speed": speed,
        "release_spin_rate": spin,
        "release_spin": "",
        "pfx_x": "0.10",
        "pfx_z": "1.20",
        "description": description,
        "type": "S",
        "events": events,
        "at_bat_number": at_bat,
        "pitch_number": "1",
        "inning": "1",
        "outs_when_up": "0",
        "home_team": "AAA",
        "away_team": "BBB",
    }
