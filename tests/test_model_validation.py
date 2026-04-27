from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from mlb_props_stack.model_validation import validate_model_only_strikeouts_walk_forward


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def _seed_walk_forward_window(tmp_path: Path) -> tuple[date, date, Path, Path, Path, Path]:
    start_date = date(2019, 4, 1)
    end_date = date(2024, 4, 4)
    dataset_run_dir = (
        tmp_path
        / "normalized"
        / "starter_strikeout_training_dataset"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
        / "run=20260427T120000Z"
    )
    pitcher_run_dir = (
        tmp_path
        / "normalized"
        / "pitcher_skill_features"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
        / "run=20260427T120100Z"
    )
    lineup_run_dir = (
        tmp_path
        / "normalized"
        / "lineup_matchup_features"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
        / "run=20260427T120200Z"
    )
    workload_run_dir = (
        tmp_path
        / "normalized"
        / "workload_leash_features"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
        / "run=20260427T120300Z"
    )
    dataset_rows: list[dict] = []
    pitcher_rows: list[dict] = []
    lineup_rows: list[dict] = []
    workload_rows: list[dict] = []
    for season in range(2019, 2025):
        for pitcher_index in range(4):
            official_date = date(season, 4, 1) + timedelta(days=pitcher_index)
            pitcher_id = 690000 + pitcher_index
            game_pk = (season * 1000) + pitcher_index
            key = f"starter-training:{official_date.isoformat()}:{game_pk}:{pitcher_id}"
            career_k_rate = 0.18 + (pitcher_index * 0.03)
            matchup_k = 0.20 + (((season + pitcher_index) % 4) * 0.018)
            expected_bf = 20.5 + pitcher_index + (0.25 * (season - 2019))
            strikeouts = round(
                (career_k_rate * expected_bf)
                + (9.0 * (matchup_k - 0.20))
                + (0.15 * (season - 2019))
            )
            strikeouts = max(1, int(strikeouts))
            home_away = "home" if pitcher_index % 2 == 0 else "away"
            pitcher_hand = "R" if pitcher_index % 2 == 0 else "L"
            dataset_rows.append(
                {
                    "training_row_id": key,
                    "official_date": official_date.isoformat(),
                    "season": season,
                    "game_pk": game_pk,
                    "pitcher_id": pitcher_id,
                    "pitcher_name": f"Pitcher {pitcher_index}",
                    "team_abbreviation": "CLE",
                    "opponent_team_abbreviation": "HOU",
                    "home_away": home_away,
                    "pitcher_hand": pitcher_hand,
                    "pitch_clock_era": "pre_pitch_clock" if season < 2023 else "pitch_clock",
                    "starter_plate_appearance_count": int(round(expected_bf)),
                    "starter_strikeouts": strikeouts,
                    "features_as_of": f"{official_date.isoformat()}T00:00:00Z",
                    "timestamp_policy_status": "target_only_no_pregame_features",
                }
            )
            pitcher_rows.append(
                {
                    "training_row_id": key,
                    "feature_row_id": f"pitcher-skill:{key}",
                    "official_date": official_date.isoformat(),
                    "game_pk": game_pk,
                    "pitcher_id": pitcher_id,
                    "features_as_of": f"{official_date.isoformat()}T00:00:00Z",
                    "career_k_rate_shrunk": round(career_k_rate, 6),
                    "season_k_rate_shrunk": round(career_k_rate + (0.003 * (season - 2019)), 6),
                    "last_3_starts_k_rate": round(career_k_rate + 0.01, 6),
                    "recent_15d_k_rate": round(career_k_rate + 0.004, 6),
                    "career_csw_rate": round(0.26 + (pitcher_index * 0.012), 6),
                    "career_swstr_rate": round(0.10 + (pitcher_index * 0.006), 6),
                    "career_whiff_rate": round(0.21 + (pitcher_index * 0.01), 6),
                    "prior_plate_appearance_count": 280 + (pitcher_index * 40) + (season - 2019),
                    "average_release_speed": 91.5 + pitcher_index,
                    "average_release_extension": 5.9 + (pitcher_index * 0.12),
                }
            )
            lineup_rows.append(
                {
                    "training_row_id": key,
                    "feature_row_id": f"lineup-matchup:{key}",
                    "official_date": official_date.isoformat(),
                    "game_pk": game_pk,
                    "pitcher_id": pitcher_id,
                    "features_as_of": f"{official_date.isoformat()}T00:00:00Z",
                    "pitcher_hand": pitcher_hand,
                    "lineup_status": "confirmed" if season >= 2023 else "projected_from_prior_team_game",
                    "projected_lineup_k_rate_vs_pitcher_hand_weighted": round(matchup_k, 6),
                    "projected_lineup_whiff_rate_weighted": round(0.22 + (matchup_k - 0.20), 6),
                    "projected_lineup_csw_rate_weighted": round(0.27 + (matchup_k - 0.20), 6),
                    "projected_lineup_contact_rate_weighted": round(0.75 - (matchup_k - 0.20), 6),
                    "arsenal_weighted_lineup_pitch_type_weakness": round(0.02 + (matchup_k - 0.20), 6),
                    "available_batter_feature_count": 9,
                }
            )
            workload_rows.append(
                {
                    "training_row_id": key,
                    "feature_row_id": f"workload:{key}",
                    "official_date": official_date.isoformat(),
                    "game_pk": game_pk,
                    "pitcher_id": pitcher_id,
                    "features_as_of": f"{official_date.isoformat()}T00:00:00Z",
                    "expected_leash_batters_faced": round(expected_bf, 6),
                    "expected_leash_pitch_count": round(expected_bf * 3.85, 6),
                    "recent_15d_batters_faced_mean": round(expected_bf - 0.8, 6),
                    "recent_15d_pitch_count_mean": round((expected_bf - 0.8) * 3.8, 6),
                    "last_3_starts_batters_faced_mean": round(expected_bf - 0.4, 6),
                    "season_batters_faced_mean": round(expected_bf - 0.1, 6),
                    "team_season_batters_faced_mean": round(expected_bf, 6),
                    "season_reached_22_batters_rate": 0.45 + (0.06 * pitcher_index),
                    "rest_bucket": ("standard_rest" if pitcher_index < 2 else "extra_rest"),
                }
            )
    _write_jsonl(dataset_run_dir / "starter_game_training_dataset.jsonl", dataset_rows)
    _write_jsonl(pitcher_run_dir / "pitcher_skill_features.jsonl", pitcher_rows)
    _write_jsonl(lineup_run_dir / "lineup_matchup_features.jsonl", lineup_rows)
    _write_jsonl(workload_run_dir / "workload_leash_features.jsonl", workload_rows)
    return start_date, end_date, dataset_run_dir, pitcher_run_dir, lineup_run_dir, workload_run_dir


def test_validate_model_only_strikeouts_writes_walk_forward_report(tmp_path) -> None:
    (
        start_date,
        end_date,
        dataset_run_dir,
        pitcher_run_dir,
        lineup_run_dir,
        workload_run_dir,
    ) = _seed_walk_forward_window(tmp_path)

    result = validate_model_only_strikeouts_walk_forward(
        start_date=start_date,
        end_date=end_date,
        output_dir=tmp_path,
        dataset_run_dir=dataset_run_dir,
        pitcher_skill_run_dir=pitcher_run_dir,
        lineup_matchup_run_dir=lineup_run_dir,
        workload_leash_run_dir=workload_run_dir,
        first_validation_season=2023,
        now=lambda: datetime(2026, 4, 27, 18, 0, tzinfo=UTC),
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    prediction_rows = [
        json.loads(line)
        for line in result.predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert result.split_count == 2
    assert result.prediction_count == 8
    assert report["validation_design"]["random_splits_used"] is False
    assert report["scope_guardrails"]["betting_decisions_included"] is False
    assert report["scope_guardrails"]["clv_or_roi_headline_metrics_included"] is False
    assert report["walk_forward_splits"][0]["validation_season"] == 2023
    assert report["headline_metrics"]["common_line_probability_metrics"]["overall"]["mean_log_loss"] >= 0.0
    assert report["bias_and_stability"]["by_pitcher_tier"]
    assert report["bias_and_stability"]["by_handedness_matchup"]
    assert report["bias_and_stability"]["by_workload_bucket"]
    assert report["bias_and_stability"]["by_rest_layoff_bucket"]
    assert "source" in report["proposed_later_wager_approval_thresholds"]
    assert report["go_no_go_recommendation"]["recommendation"] in {
        "conditional_go_for_betting_layer_rebuild",
        "no_go_betting_layer_still_blocked",
    }
    assert prediction_rows[0]["line_probabilities"][0]["confidence_bucket"]
    assert "Model-Only Walk-Forward Validation" in result.report_markdown_path.read_text(
        encoding="utf-8"
    )
