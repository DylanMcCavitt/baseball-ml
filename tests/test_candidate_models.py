from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from mlb_props_stack.candidate_models import (
    strikeout_line_probabilities,
    train_candidate_strikeout_models,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def _seed_candidate_training_window(tmp_path: Path) -> tuple[date, date, Path, Path, Path, Path]:
    start_date = date(2026, 4, 1)
    end_date = start_date + timedelta(days=5)
    dataset_run_dir = (
        tmp_path
        / "normalized"
        / "starter_strikeout_training_dataset"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
        / "run=20260426T120000Z"
    )
    pitcher_run_dir = (
        tmp_path
        / "normalized"
        / "pitcher_skill_features"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
        / "run=20260426T120100Z"
    )
    lineup_run_dir = (
        tmp_path
        / "normalized"
        / "lineup_matchup_features"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
        / "run=20260426T120200Z"
    )
    workload_run_dir = (
        tmp_path
        / "normalized"
        / "workload_leash_features"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
        / "run=20260426T120300Z"
    )
    dataset_rows: list[dict] = []
    pitcher_rows: list[dict] = []
    lineup_rows: list[dict] = []
    workload_rows: list[dict] = []

    for day_index in range(6):
        official_date = start_date + timedelta(days=day_index)
        for pitcher_index in range(4):
            pitcher_id = 680000 + pitcher_index
            game_pk = 900000 + (day_index * 10) + pitcher_index
            key = f"starter-training:{official_date.isoformat()}:{game_pk}:{pitcher_id}"
            career_k_rate = 0.18 + (pitcher_index * 0.025)
            matchup_k = 0.20 + (((day_index + pitcher_index) % 4) * 0.02)
            expected_bf = 21.5 + pitcher_index + (0.5 * day_index)
            strikeouts = round(
                (career_k_rate * expected_bf)
                + (10.0 * (matchup_k - 0.20))
                + (0.25 * (1 if pitcher_index % 2 == 0 else -1))
            )
            strikeouts = max(1, int(strikeouts))
            dataset_rows.append(
                {
                    "training_row_id": key,
                    "official_date": official_date.isoformat(),
                    "game_pk": game_pk,
                    "pitcher_id": pitcher_id,
                    "pitcher_name": f"Pitcher {pitcher_index}",
                    "team_abbreviation": "CLE",
                    "opponent_team_abbreviation": "HOU",
                    "home_away": "home" if pitcher_index % 2 == 0 else "away",
                    "pitcher_hand": "R" if pitcher_index % 2 == 0 else "L",
                    "pitch_clock_era": "pitch_clock",
                    "league_k_environment": "modern_high_k_environment_2019_plus",
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
                    "season_k_rate_shrunk": round(career_k_rate + (0.004 * day_index), 6),
                    "last_3_starts_k_rate": round(career_k_rate + 0.01, 6),
                    "recent_15d_k_rate": round(career_k_rate + 0.005, 6),
                    "career_csw_rate": round(0.26 + (pitcher_index * 0.01), 6),
                    "career_swstr_rate": round(0.10 + (pitcher_index * 0.006), 6),
                    "career_whiff_rate": round(0.21 + (pitcher_index * 0.01), 6),
                    "prior_plate_appearance_count": 300 + (pitcher_index * 40) + day_index,
                    "average_release_speed": 92.0 + pitcher_index,
                    "average_release_extension": 6.0 + (pitcher_index * 0.1),
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
                    "expected_leash_pitch_count": round(expected_bf * 3.9, 6),
                    "recent_15d_batters_faced_mean": round(expected_bf - 1.0, 6),
                    "recent_15d_pitch_count_mean": round((expected_bf - 1.0) * 3.8, 6),
                    "last_3_starts_batters_faced_mean": round(expected_bf - 0.5, 6),
                    "season_batters_faced_mean": round(expected_bf - 0.2, 6),
                    "team_season_batters_faced_mean": round(expected_bf, 6),
                    "season_reached_22_batters_rate": 0.5 + (0.05 * pitcher_index),
                }
            )

    _write_jsonl(dataset_run_dir / "starter_game_training_dataset.jsonl", dataset_rows)
    _write_jsonl(pitcher_run_dir / "pitcher_skill_features.jsonl", pitcher_rows)
    _write_jsonl(lineup_run_dir / "lineup_matchup_features.jsonl", lineup_rows)
    _write_jsonl(workload_run_dir / "workload_leash_features.jsonl", workload_rows)
    return start_date, end_date, dataset_run_dir, pitcher_run_dir, lineup_run_dir, workload_run_dir


def test_train_candidate_strikeout_models_writes_comparable_distribution_report(tmp_path) -> None:
    (
        start_date,
        end_date,
        dataset_run_dir,
        pitcher_run_dir,
        lineup_run_dir,
        workload_run_dir,
    ) = _seed_candidate_training_window(tmp_path)

    result = train_candidate_strikeout_models(
        start_date=start_date,
        end_date=end_date,
        output_dir=tmp_path,
        dataset_run_dir=dataset_run_dir,
        pitcher_skill_run_dir=pitcher_run_dir,
        lineup_matchup_run_dir=lineup_run_dir,
        workload_leash_run_dir=workload_run_dir,
        now=lambda: datetime(2026, 4, 26, 18, 0, tzinfo=UTC),
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    selected_model = json.loads(result.selected_model_path.read_text(encoding="utf-8"))
    output_rows = [
        json.loads(line)
        for line in result.model_outputs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert result.row_count == 24
    assert result.selected_candidate in report["candidates"]
    assert report["scope_guardrails"]["betting_decisions_included"] is False
    assert report["output_contract"]["supports_arbitrary_strikeout_lines"] is True
    assert report["source_artifacts"]["pitcher_skill_matches"] == 24
    assert "poisson_glm_count_baseline" in report["candidates"]
    assert "negative_binomial_glm_count_baseline" in report["candidates"]
    assert "plate_appearance_logistic_rate" in report["candidates"]
    assert "boosted_stump_tree_ensemble" in report["candidates"]
    assert "validation_top_two_mean_blend" in report["candidates"]
    assert report["candidates"]["neural_sequence_challenger"]["status"] == "skipped"
    assert (
        report["candidates"][result.selected_candidate]["splits"]["held_out"][
            "probability_metrics"
        ]["overall"]["mean_log_loss"]
        >= 0.0
    )
    assert report["candidates"][result.selected_candidate]["feature_group_contributions"]
    assert selected_model["selected_candidate"] == result.selected_candidate
    assert output_rows[0]["point_projection"] >= 0.0
    assert output_rows[0]["feature_row_id"] == output_rows[0]["training_row_id"]
    assert output_rows[0]["lineup_snapshot_id"] is None
    assert output_rows[0]["features_as_of"] == "2026-04-01T00:00:00Z"
    assert output_rows[0]["projection_generated_at"] == "2026-04-26T18:00:00Z"
    assert output_rows[0]["model_input_refs"]["pitcher_feature_row_id"].startswith(
        "pitcher-skill:"
    )
    assert output_rows[0]["model_input_refs"]["lineup_feature_row_id"].startswith(
        "lineup-matchup:"
    )
    assert output_rows[0]["probability_distribution"]
    assert output_rows[0]["line_probability_contract"]["supports_arbitrary_lines"] is True
    assert output_rows[0]["over_under_probabilities"][0]["line"] == 2.5
    assert output_rows[0]["confidence"]["central_80_interval"][0] >= 0
    assert "Candidate Strikeout Model Comparison" in result.report_markdown_path.read_text(
        encoding="utf-8"
    )


def test_strikeout_line_probabilities_supports_arbitrary_lines() -> None:
    result = strikeout_line_probabilities([0.10, 0.20, 0.30, 0.40], 1.75)

    assert result["line"] == 1.75
    assert round(result["over_probability"], 6) == 0.70
    assert round(result["under_probability"], 6) == 0.30
