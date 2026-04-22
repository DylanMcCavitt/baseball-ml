from __future__ import annotations

import csv
import json
from datetime import UTC, date, datetime, timedelta
from io import StringIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from mlb_props_stack.modeling import (
    calibrate_starter_strikeout_ladder_probabilities,
    starter_strikeout_ladder_probabilities,
    starter_strikeout_line_probability,
    train_starter_strikeout_baseline,
)
from mlb_props_stack.tracking import TrackingConfig


STATCAST_HEADERS = [
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
    "pitch_type",
    "pitch_name",
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "plate_x",
    "plate_z",
    "zone",
    "description",
    "events",
    "stand",
    "p_throws",
    "balls",
    "strikes",
    "outs_when_up",
    "home_team",
    "away_team",
    "inning_topbot",
]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def _tracking_config(tmp_path: Path) -> TrackingConfig:
    return TrackingConfig(tracking_uri=f"file:{tmp_path / 'artifacts' / 'mlruns'}")


def _csv_text(rows: list[dict[str, object]]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=STATCAST_HEADERS)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def _seed_feature_run(
    output_dir: Path,
    *,
    official_date: date,
    feature_rows: list[dict[str, object]],
) -> None:
    run_dir = (
        output_dir
        / "normalized"
        / "statcast_search"
        / f"date={official_date.isoformat()}"
        / "run=20260421T180000Z"
    )
    pitcher_rows = []
    lineup_rows = []
    game_context_rows = []
    for row in feature_rows:
        pitcher_rows.append(
            {
                "feature_row_id": row["pitcher_feature_row_id"],
                "official_date": official_date.isoformat(),
                "game_pk": row["game_pk"],
                "pitcher_id": row["pitcher_id"],
                "pitcher_name": row["pitcher_name"],
                "team_side": row["team_side"],
                "team_abbreviation": row["team_abbreviation"],
                "opponent_team_abbreviation": row["opponent_team_abbreviation"],
                "history_start_date": (official_date - timedelta(days=30)).isoformat(),
                "history_end_date": (official_date - timedelta(days=1)).isoformat(),
                "features_as_of": f"{official_date.isoformat()}T15:00:00Z",
                "feature_status": "ok",
                "pitch_sample_size": row["pitch_sample_size"],
                "plate_appearance_sample_size": row["plate_appearance_sample_size"],
                "pitcher_hand": row["pitcher_hand"],
                "pitcher_k_rate": row["pitcher_k_rate"],
                "swinging_strike_rate": row["swinging_strike_rate"],
                "csw_rate": row["csw_rate"],
                "pitch_type_usage": row["pitch_type_usage"],
                "average_release_speed": row["average_release_speed"],
                "release_speed_delta_vs_baseline": row["release_speed_delta_vs_baseline"],
                "average_release_extension": row["average_release_extension"],
                "release_extension_delta_vs_baseline": row["release_extension_delta_vs_baseline"],
                "recent_batters_faced": row["recent_batters_faced"],
                "recent_pitch_count": row["recent_pitch_count"],
                "rest_days": row["rest_days"],
                "last_start_pitch_count": row["last_start_pitch_count"],
                "last_start_batters_faced": row["last_start_batters_faced"],
            }
        )
        lineup_rows.append(
            {
                "feature_row_id": row["lineup_feature_row_id"],
                "official_date": official_date.isoformat(),
                "game_pk": row["game_pk"],
                "pitcher_id": row["pitcher_id"],
                "pitcher_name": row["pitcher_name"],
                "team_abbreviation": row["team_abbreviation"],
                "opponent_team_abbreviation": row["opponent_team_abbreviation"],
                "opponent_team_name": row["opponent_team_name"],
                "lineup_snapshot_id": row["lineup_snapshot_id"],
                "history_start_date": (official_date - timedelta(days=30)).isoformat(),
                "history_end_date": (official_date - timedelta(days=1)).isoformat(),
                "features_as_of": f"{official_date.isoformat()}T15:30:00Z",
                "lineup_status": row["lineup_status"],
                "lineup_is_confirmed": row["lineup_is_confirmed"],
                "lineup_size": row["lineup_size"],
                "available_batter_feature_count": row["available_batter_feature_count"],
                "pitcher_hand": row["pitcher_hand"],
                "projected_lineup_k_rate": row["projected_lineup_k_rate"],
                "projected_lineup_k_rate_vs_pitcher_hand": row["projected_lineup_k_rate_vs_pitcher_hand"],
                "projected_lineup_chase_rate": row["projected_lineup_chase_rate"],
                "projected_lineup_contact_rate": row["projected_lineup_contact_rate"],
                "lineup_continuity_count": row["lineup_continuity_count"],
                "lineup_continuity_ratio": row["lineup_continuity_ratio"],
                "lineup_player_ids": row["lineup_player_ids"],
            }
        )
        game_context_rows.append(
            {
                "feature_row_id": row["game_context_feature_row_id"],
                "official_date": official_date.isoformat(),
                "game_pk": row["game_pk"],
                "pitcher_id": row["pitcher_id"],
                "pitcher_name": row["pitcher_name"],
                "team_abbreviation": row["team_abbreviation"],
                "opponent_team_abbreviation": row["opponent_team_abbreviation"],
                "home_away": row["home_away"],
                "venue_id": 5,
                "venue_name": "Test Park",
                "day_night": row["day_night"],
                "double_header": row["double_header"],
                "features_as_of": f"{official_date.isoformat()}T15:45:00Z",
                "park_factor": None,
                "park_factor_status": "missing_park_factor_source",
                "rest_days": row["rest_days"],
                "weather_status": "missing_weather_source",
                "weather_source": None,
                "weather_temperature_f": None,
                "weather_wind_mph": None,
                "weather_conditions": None,
                "expected_leash_pitch_count": row["expected_leash_pitch_count"],
                "expected_leash_batters_faced": row["expected_leash_batters_faced"],
            }
        )

    _write_jsonl(run_dir / "pitcher_daily_features.jsonl", pitcher_rows)
    _write_jsonl(run_dir / "lineup_daily_features.jsonl", lineup_rows)
    _write_jsonl(run_dir / "game_context_features.jsonl", game_context_rows)


def _build_outcome_csv(
    *,
    official_date: date,
    game_pk: int,
    pitcher_id: int,
    strikeout_count: int,
    plate_appearance_count: int,
    home_team: str,
    away_team: str,
    pitcher_hand: str,
) -> str:
    rows: list[dict[str, object]] = []
    for at_bat_number in range(1, plate_appearance_count + 1):
        rows.append(
            {
                "game_date": official_date.isoformat(),
                "game_pk": game_pk,
                "at_bat_number": at_bat_number,
                "pitch_number": 1,
                "pitcher": pitcher_id,
                "batter": 700000 + at_bat_number,
                "pitch_type": "FF",
                "pitch_name": "4-Seam Fastball",
                "release_speed": 95.0,
                "release_spin_rate": 2300,
                "release_extension": 6.2,
                "plate_x": 0.0,
                "plate_z": 2.5,
                "zone": 5,
                "description": "swinging_strike" if at_bat_number <= strikeout_count else "hit_into_play",
                "events": "strikeout" if at_bat_number <= strikeout_count else "field_out",
                "stand": "R",
                "p_throws": pitcher_hand,
                "balls": 0,
                "strikes": 2,
                "outs_when_up": (at_bat_number - 1) % 3,
                "home_team": home_team,
                "away_team": away_team,
                "inning_topbot": "Top",
            }
        )
    return _csv_text(rows)


class FakeStatcastClient:
    def __init__(self, responses: dict[tuple[str, int], str]) -> None:
        self.responses = responses

    def fetch_csv(self, url: str) -> str:
        query = parse_qs(urlparse(url).query)
        pitcher_id = int(query["pitchers_lookup[]"][0])
        inferred_date = date.fromisoformat(query["game_date_gt"][0]) + timedelta(days=1)
        return self.responses[(inferred_date.isoformat(), pitcher_id)]


def test_train_starter_strikeout_baseline_builds_artifacts_and_beats_benchmark(tmp_path):
    outcome_csv_by_pitcher_and_date: dict[tuple[str, int], str] = {}
    start_date = date(2026, 4, 16)
    end_date = start_date + timedelta(days=4)
    previous_run_dir = (
        tmp_path
        / "normalized"
        / "starter_strikeout_baseline"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
        / "run=20260420T170000Z"
    )
    previous_run_dir.mkdir(parents=True, exist_ok=True)
    (previous_run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "model": {
                    "held_out": {
                        "rmse": 3.9,
                        "mae": 3.1,
                        "spearman_rank_correlation": 0.12,
                    }
                },
                "probability_calibration": {
                    "honest_held_out": {
                        "held_out": {
                            "calibrated": {
                                "mean_brier_score": 0.245,
                                "mean_log_loss": 0.691,
                                "expected_calibration_error": 0.133,
                            }
                        }
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    all_feature_rows = []
    for date_index in range(5):
        official_date = start_date + timedelta(days=date_index)
        feature_rows = []
        for pitcher_index in range(2):
            pitcher_id = 680800 + pitcher_index
            game_pk = 824440 + (date_index * 10) + pitcher_index
            pitcher_k_rate = round(0.18 + (0.02 * pitcher_index) + (0.01 * date_index), 6)
            lineup_k_rate = round(0.21 + (0.015 * ((date_index + pitcher_index) % 3)), 6)
            lineup_contact_rate = round(0.73 - (0.015 * date_index) + (0.005 * pitcher_index), 6)
            expected_leash_batters_faced = float(23 + date_index + (2 * pitcher_index))
            naive_benchmark = pitcher_k_rate * expected_leash_batters_faced
            home_away = "home" if pitcher_index == 0 else "away"
            actual_strikeouts = int(
                round(
                    naive_benchmark
                    + (20.0 * (lineup_k_rate - 0.22))
                    - (12.0 * (lineup_contact_rate - 0.70))
                    + (0.6 if home_away == "home" else -0.3)
                )
            )
            row = {
                "game_pk": game_pk,
                "pitcher_id": pitcher_id,
                "pitcher_name": f"Pitcher {pitcher_index + 1}",
                "team_side": "home" if home_away == "home" else "away",
                "team_abbreviation": "CLE" if home_away == "home" else "HOU",
                "opponent_team_abbreviation": "HOU" if home_away == "home" else "CLE",
                "opponent_team_name": "Houston Astros" if home_away == "home" else "Cleveland Guardians",
                "pitcher_feature_row_id": f"pitcher-feature:{game_pk}:{pitcher_id}",
                "lineup_feature_row_id": f"lineup-feature:{game_pk}:{pitcher_id}",
                "game_context_feature_row_id": f"game-context:{game_pk}:{pitcher_id}",
                "lineup_snapshot_id": f"lineup:{game_pk}:{official_date.isoformat()}",
                "pitcher_hand": "R" if pitcher_index == 0 else "L",
                "pitch_sample_size": 480 + (date_index * 10) + (pitcher_index * 15),
                "plate_appearance_sample_size": 105 + (date_index * 4) + (pitcher_index * 5),
                "pitcher_k_rate": pitcher_k_rate,
                "swinging_strike_rate": round(0.11 + (0.005 * pitcher_index), 6),
                "csw_rate": round(0.27 + (0.01 * date_index), 6),
                "pitch_type_usage": {"FF": 0.58 - (0.02 * pitcher_index), "SL": 0.42 + (0.02 * pitcher_index)},
                "average_release_speed": 94.0 + pitcher_index,
                "release_speed_delta_vs_baseline": round(0.1 * date_index, 6),
                "average_release_extension": 6.1 + (0.05 * pitcher_index),
                "release_extension_delta_vs_baseline": round(0.02 * date_index, 6),
                "recent_batters_faced": 70 + (date_index * 2),
                "recent_pitch_count": 260 + (date_index * 8),
                "rest_days": 5 + pitcher_index,
                "last_start_pitch_count": 92 + (date_index * 2),
                "last_start_batters_faced": 24 + pitcher_index,
                "lineup_status": "confirmed" if date_index % 2 == 0 else "projected",
                "lineup_is_confirmed": date_index % 2 == 0,
                "lineup_size": 9,
                "available_batter_feature_count": 9,
                "projected_lineup_k_rate": lineup_k_rate,
                "projected_lineup_k_rate_vs_pitcher_hand": lineup_k_rate + 0.01,
                "projected_lineup_chase_rate": round(0.28 + (0.005 * date_index), 6),
                "projected_lineup_contact_rate": lineup_contact_rate,
                "lineup_continuity_count": 6 + pitcher_index,
                "lineup_continuity_ratio": round((6 + pitcher_index) / 9, 6),
                "lineup_player_ids": [710000 + slot for slot in range(9)],
                "home_away": home_away,
                "day_night": "night",
                "double_header": "N",
                "expected_leash_pitch_count": 95.0 + date_index,
                "expected_leash_batters_faced": expected_leash_batters_faced,
                "actual_strikeouts": actual_strikeouts,
            }
            feature_rows.append(row)
            all_feature_rows.append(row)
            outcome_csv_by_pitcher_and_date[(official_date.isoformat(), pitcher_id)] = _build_outcome_csv(
                official_date=official_date,
                game_pk=game_pk,
                pitcher_id=pitcher_id,
                strikeout_count=actual_strikeouts,
                plate_appearance_count=26 + pitcher_index,
                home_team="CLE" if home_away == "home" else "HOU",
                away_team="HOU" if home_away == "home" else "CLE",
                pitcher_hand=row["pitcher_hand"],
            )
        _seed_feature_run(tmp_path, official_date=official_date, feature_rows=feature_rows)

    result = train_starter_strikeout_baseline(
        start_date=start_date,
        end_date=end_date,
        output_dir=tmp_path,
        client=FakeStatcastClient(outcome_csv_by_pitcher_and_date),
        now=lambda: datetime(2026, 4, 21, 18, 0, tzinfo=UTC),
        tracking_config=_tracking_config(tmp_path),
    )

    evaluation = json.loads(result.evaluation_path.read_text(encoding="utf-8"))
    model_artifact = json.loads(result.model_path.read_text(encoding="utf-8"))
    probability_calibrator = json.loads(
        result.probability_calibrator_path.read_text(encoding="utf-8")
    )
    calibration_summary = json.loads(
        result.calibration_summary_path.read_text(encoding="utf-8")
    )
    evaluation_summary = json.loads(
        result.evaluation_summary_path.read_text(encoding="utf-8")
    )
    evaluation_summary_markdown = result.evaluation_summary_markdown_path.read_text(
        encoding="utf-8"
    )
    ladder_rows = [
        json.loads(line)
        for line in result.ladder_probabilities_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw_vs_calibrated_rows = [
        json.loads(line)
        for line in result.raw_vs_calibrated_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset_rows = [
        json.loads(line)
        for line in result.dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert result.row_count == 10
    assert result.outcome_count == 10
    assert result.dispersion_alpha >= 0.0
    assert result.held_out_status == "beating_benchmark"
    assert result.previous_run_id == "20260420T170000Z"
    assert result.evaluation_summary_path.exists()
    assert result.evaluation_summary_markdown_path.exists()
    assert evaluation["held_out_beats_benchmark"] == {"rmse": True, "mae": True}
    assert evaluation["model"]["held_out"]["rmse"] < evaluation["benchmark"]["held_out"]["rmse"]
    assert evaluation["model"]["held_out"]["mae"] <= evaluation["benchmark"]["held_out"]["mae"]
    assert result.held_out_model_rmse == evaluation["model"]["held_out"]["rmse"]
    assert result.held_out_benchmark_rmse == evaluation["benchmark"]["held_out"]["rmse"]
    assert result.held_out_model_mae == evaluation["model"]["held_out"]["mae"]
    assert result.held_out_benchmark_mae == evaluation["benchmark"]["held_out"]["mae"]
    assert evaluation["count_distribution"]["name"] == "negative_binomial_global_dispersion_v1"
    assert set(evaluation["count_distribution"]["held_out_beats_poisson"]) == {
        "mean_negative_log_likelihood",
        "mean_ranked_probability_score",
    }
    assert evaluation["probability_calibration"]["production_calibrator"]["name"] == (
        "isotonic_ladder_probability_calibrator_v1"
    )
    assert (
        evaluation["probability_calibration"]["honest_held_out"]["validation"][
            "calibration_training_splits"
        ]
        == ["train"]
    )
    assert (
        evaluation["probability_calibration"]["honest_held_out"]["test"][
            "calibration_training_splits"
        ]
        == ["train", "validation"]
    )
    assert evaluation["date_splits"]["train"] == ["2026-04-16", "2026-04-17", "2026-04-18"]
    assert evaluation["date_splits"]["validation"] == ["2026-04-19"]
    assert evaluation["date_splits"]["test"] == ["2026-04-20"]
    assert dataset_rows[0]["starter_strikeouts"] > 0
    assert ladder_rows[0]["count_distribution"]["dispersion_alpha"] == result.dispersion_alpha
    assert ladder_rows[0]["feature_row_id"] == dataset_rows[0]["training_row_id"]
    assert ladder_rows[0]["lineup_snapshot_id"] == dataset_rows[0]["lineup_snapshot_id"]
    assert ladder_rows[0]["features_as_of"] == dataset_rows[0]["features_as_of"]
    assert ladder_rows[0]["projection_generated_at"] == dataset_rows[0]["features_as_of"]
    assert ladder_rows[0]["ladder_probabilities"][0]["line"] == 0.5
    assert ladder_rows[0]["ladder_probabilities"][0]["over_probability"] < 1.0
    assert ladder_rows[0]["calibrated_ladder_probabilities"][0]["line"] == 0.5
    assert ladder_rows[0]["probability_calibration"]["name"] == (
        "isotonic_ladder_probability_calibrator_v1"
    )
    assert "starter_strikeouts" not in model_artifact["encoded_feature_names"]
    assert "features_as_of" not in model_artifact["encoded_feature_names"]
    assert "projected_lineup_k_rate" in model_artifact["encoded_feature_names"]
    assert model_artifact["count_distribution"]["dispersion_alpha"] == result.dispersion_alpha
    assert model_artifact["tracking"]["mlflow_run_id"] == result.mlflow_run_id
    assert model_artifact["probability_calibration"]["name"] == (
        "isotonic_ladder_probability_calibrator_v1"
    )
    assert probability_calibrator["name"] == "isotonic_ladder_probability_calibrator_v1"
    assert calibration_summary["production_calibrator"]["name"] == (
        "isotonic_ladder_probability_calibrator_v1"
    )
    assert evaluation["tracking"]["mlflow_run_id"] == result.mlflow_run_id
    assert evaluation_summary["held_out_performance"]["status"] == "beating_benchmark"
    assert evaluation_summary["mlflow_run_id"] == result.mlflow_run_id
    assert (
        evaluation_summary["mlflow_experiment_name"]
        == "mlb-props-stack-starter-strikeout-training"
    )
    assert evaluation_summary["previous_run_comparison"]["previous_run_id"] == "20260420T170000Z"
    assert (
        evaluation_summary["previous_run_comparison"]["held_out_model"]["rmse"]["status"]
        == "improved"
    )
    assert (
        evaluation_summary["previous_run_comparison"]["held_out_model"]["mae"]["status"]
        == "improved"
    )
    assert len(evaluation_summary["top_feature_importance"]) == 10
    assert "Starter Strikeout Baseline Evaluation Summary" in evaluation_summary_markdown
    assert "MLflow run ID" in evaluation_summary_markdown
    assert "Held-Out Performance" in evaluation_summary_markdown
    assert "Comparison To Previous Run" in evaluation_summary_markdown
    assert "projected_lineup_k_rate" in evaluation_summary_markdown
    assert result.reproducibility_notes_path.exists()
    assert raw_vs_calibrated_rows
    assert all(
        row["calibration_fit_through_date"] < row["official_date"]
        for row in raw_vs_calibrated_rows
        if row["calibration_fit_through_date"] is not None
    )
    assert any(
        item["feature"] == "projected_lineup_k_rate"
        for item in evaluation["feature_importance"]
    )


def test_train_starter_strikeout_baseline_skips_rows_without_same_game_outcomes(tmp_path):
    start_date = date(2026, 4, 16)
    responses: dict[tuple[str, int], str] = {}

    for date_index in range(4):
        official_date = start_date + timedelta(days=date_index)
        pitcher_id = 700100 + date_index
        game_pk = 9000 + date_index
        feature_rows = [
            {
                "game_pk": game_pk,
                "pitcher_id": pitcher_id,
                "pitcher_name": f"Pitcher {date_index}",
                "team_side": "home",
                "team_abbreviation": "CLE",
                "opponent_team_abbreviation": "HOU",
                "opponent_team_name": "Houston Astros",
                "pitcher_feature_row_id": f"pitcher-feature:{game_pk}:{pitcher_id}",
                "lineup_feature_row_id": f"lineup-feature:{game_pk}:{pitcher_id}",
                "game_context_feature_row_id": f"game-context:{game_pk}:{pitcher_id}",
                "lineup_snapshot_id": f"lineup:{game_pk}:{official_date.isoformat()}",
                "pitcher_hand": "R",
                "pitch_sample_size": 400,
                "plate_appearance_sample_size": 90,
                "pitcher_k_rate": 0.26,
                "swinging_strike_rate": 0.14,
                "csw_rate": 0.29,
                "pitch_type_usage": {"FF": 0.5},
                "average_release_speed": 95.0,
                "release_speed_delta_vs_baseline": 0.0,
                "average_release_extension": 6.1,
                "release_extension_delta_vs_baseline": 0.0,
                "recent_batters_faced": 70,
                "recent_pitch_count": 280,
                "rest_days": 5,
                "last_start_pitch_count": 92,
                "last_start_batters_faced": 25,
                "lineup_status": "projected",
                "lineup_is_confirmed": False,
                "lineup_size": 9,
                "available_batter_feature_count": 9,
                "projected_lineup_k_rate": 0.24,
                "projected_lineup_k_rate_vs_pitcher_hand": 0.25,
                "projected_lineup_chase_rate": 0.31,
                "projected_lineup_contact_rate": 0.74,
                "lineup_continuity_count": 6,
                "lineup_continuity_ratio": 0.67,
                "lineup_player_ids": [710000 + slot for slot in range(9)],
                "home_away": "home",
                "day_night": "night",
                "double_header": "N",
                "expected_leash_pitch_count": 95.0,
                "expected_leash_batters_faced": 25.0,
            }
        ]
        _seed_feature_run(tmp_path, official_date=official_date, feature_rows=feature_rows)
        if date_index == 1:
            responses[(official_date.isoformat(), pitcher_id)] = _csv_text([])
        else:
            responses[(official_date.isoformat(), pitcher_id)] = _build_outcome_csv(
                official_date=official_date,
                game_pk=game_pk,
                pitcher_id=pitcher_id,
                strikeout_count=6 + date_index,
                plate_appearance_count=24,
                home_team="CLE",
                away_team="HOU",
                pitcher_hand="R",
            )

    result = train_starter_strikeout_baseline(
        start_date=start_date,
        end_date=start_date + timedelta(days=3),
        output_dir=tmp_path,
        client=FakeStatcastClient(responses),
        now=iter(
            datetime(2026, 4, 22, 18, 0, tzinfo=UTC) + timedelta(minutes=index)
            for index in range(20)
        ).__next__,
        tracking_config=_tracking_config(tmp_path),
    )

    assert result.row_count == 3
    assert result.outcome_count == 3


def test_train_starter_strikeout_baseline_excludes_sparse_optional_lineup_features(tmp_path):
    start_date = date(2026, 4, 16)
    responses: dict[tuple[str, int], str] = {}

    for date_index in range(4):
        official_date = start_date + timedelta(days=date_index)
        feature_rows = []
        for pitcher_index in range(2):
            pitcher_id = 710100 + (date_index * 10) + pitcher_index
            game_pk = 9100 + (date_index * 10) + pitcher_index
            feature_rows.append(
                {
                    "game_pk": game_pk,
                    "pitcher_id": pitcher_id,
                    "pitcher_name": f"Sparse Pitcher {pitcher_index}",
                    "team_side": "home" if pitcher_index == 0 else "away",
                    "team_abbreviation": "CLE" if pitcher_index == 0 else "HOU",
                    "opponent_team_abbreviation": "HOU" if pitcher_index == 0 else "CLE",
                    "opponent_team_name": (
                        "Houston Astros" if pitcher_index == 0 else "Cleveland Guardians"
                    ),
                    "pitcher_feature_row_id": f"pitcher-feature:{game_pk}:{pitcher_id}",
                    "lineup_feature_row_id": f"lineup-feature:{game_pk}:{pitcher_id}",
                    "game_context_feature_row_id": f"game-context:{game_pk}:{pitcher_id}",
                    "lineup_snapshot_id": None,
                    "pitcher_hand": "R" if pitcher_index == 0 else "L",
                    "pitch_sample_size": 420 + (date_index * 10) + (pitcher_index * 8),
                    "plate_appearance_sample_size": 95 + (date_index * 4) + (pitcher_index * 3),
                    "pitcher_k_rate": round(
                        0.21 + (0.01 * date_index) + (0.015 * pitcher_index),
                        6,
                    ),
                    "swinging_strike_rate": round(
                        0.11 + (0.004 * date_index) + (0.006 * pitcher_index),
                        6,
                    ),
                    "csw_rate": round(0.27 + (0.006 * date_index), 6),
                    "pitch_type_usage": {
                        "FF": round(0.55 - (0.02 * pitcher_index), 6),
                        "SL": round(0.25 + (0.02 * pitcher_index), 6),
                    },
                    "average_release_speed": 94.0 + pitcher_index,
                    "release_speed_delta_vs_baseline": round(0.05 * date_index, 6),
                    "average_release_extension": 6.1 + (0.04 * pitcher_index),
                    "release_extension_delta_vs_baseline": round(0.01 * date_index, 6),
                    "recent_batters_faced": 66 + (date_index * 3) + pitcher_index,
                    "recent_pitch_count": 250 + (date_index * 9) + (pitcher_index * 4),
                    "rest_days": 5 + pitcher_index,
                    "last_start_pitch_count": 90 + (date_index * 2),
                    "last_start_batters_faced": 24 + pitcher_index,
                    "lineup_status": "missing",
                    "lineup_is_confirmed": False,
                    "lineup_size": 0,
                    "available_batter_feature_count": 0,
                    "projected_lineup_k_rate": None,
                    "projected_lineup_k_rate_vs_pitcher_hand": None,
                    "projected_lineup_chase_rate": None,
                    "projected_lineup_contact_rate": None,
                    "lineup_continuity_count": None,
                    "lineup_continuity_ratio": None,
                    "lineup_player_ids": [],
                    "home_away": "home" if pitcher_index == 0 else "away",
                    "day_night": "night",
                    "double_header": "N",
                    "expected_leash_pitch_count": 92.0 + date_index,
                    "expected_leash_batters_faced": 23.0 + date_index + pitcher_index,
                }
            )
            strikeout_count = 4 + date_index + pitcher_index
            responses[(official_date.isoformat(), pitcher_id)] = _build_outcome_csv(
                official_date=official_date,
                game_pk=game_pk,
                pitcher_id=pitcher_id,
                strikeout_count=strikeout_count,
                plate_appearance_count=25 + pitcher_index,
                home_team="CLE" if pitcher_index == 0 else "HOU",
                away_team="HOU" if pitcher_index == 0 else "CLE",
                pitcher_hand="R" if pitcher_index == 0 else "L",
            )
        _seed_feature_run(tmp_path, official_date=official_date, feature_rows=feature_rows)

    result = train_starter_strikeout_baseline(
        start_date=start_date,
        end_date=start_date + timedelta(days=3),
        output_dir=tmp_path,
        client=FakeStatcastClient(responses),
        now=lambda: datetime(2026, 4, 21, 18, 0, tzinfo=UTC),
        tracking_config=_tracking_config(tmp_path),
    )
    model_artifact = json.loads(result.model_path.read_text(encoding="utf-8"))

    assert "pitcher_k_rate" in model_artifact["encoded_feature_names"]
    assert "expected_leash_batters_faced" in model_artifact["encoded_feature_names"]
    assert "projected_lineup_k_rate" not in model_artifact["encoded_feature_names"]
    assert "projected_lineup_contact_rate" not in model_artifact["encoded_feature_names"]
    assert "lineup_continuity_ratio" not in model_artifact["encoded_feature_names"]


def test_line_and_ladder_probability_helpers_are_monotonic_and_complementary():
    over_probability, under_probability = starter_strikeout_line_probability(
        mean=6.2,
        line=5.5,
        dispersion_alpha=0.24,
    )

    assert round(over_probability + under_probability, 6) == 1.0
    ladder = starter_strikeout_ladder_probabilities(
        mean=6.2,
        dispersion_alpha=0.24,
    )

    assert ladder[0]["line"] == 0.5
    assert ladder[0]["over_probability"] > ladder[-1]["over_probability"]
    assert ladder[0]["under_probability"] < ladder[-1]["under_probability"]
    assert all(
        left["over_probability"] >= right["over_probability"]
        and left["under_probability"] <= right["under_probability"]
        for left, right in zip(ladder, ladder[1:])
    )
    assert next(
        item
        for item in ladder
        if item["line"] == 5.5
    ) == {
        "line": 5.5,
        "over_probability": round(over_probability, 6),
        "under_probability": round(under_probability, 6),
    }


def test_calibrated_ladder_probabilities_stay_monotonic_and_complementary():
    raw_ladder = starter_strikeout_ladder_probabilities(
        mean=5.4,
        dispersion_alpha=0.18,
    )
    calibrated_ladder = calibrate_starter_strikeout_ladder_probabilities(
        raw_ladder,
        {
            "name": "isotonic_ladder_probability_calibrator_v1",
            "source": "out_of_fold_ladder_events",
            "configured_min_sample": 10,
            "sample_count": 24,
            "fitted_from_date": "2026-04-16",
            "fitted_through_date": "2026-04-18",
            "is_identity": False,
            "reason": None,
            "sample_warning": None,
            "buckets": [
                {
                    "raw_probability_min": 0.0,
                    "raw_probability_max": 0.35,
                    "calibrated_probability": 0.08,
                    "sample_count": 8,
                    "positive_count": 1,
                },
                {
                    "raw_probability_min": 0.35,
                    "raw_probability_max": 0.65,
                    "calibrated_probability": 0.52,
                    "sample_count": 8,
                    "positive_count": 4,
                },
                {
                    "raw_probability_min": 0.65,
                    "raw_probability_max": 0.99,
                    "calibrated_probability": 0.86,
                    "sample_count": 8,
                    "positive_count": 7,
                },
            ],
        },
    )

    assert calibrated_ladder[0]["line"] == 0.5
    assert all(
        left["over_probability"] >= right["over_probability"]
        for left, right in zip(calibrated_ladder, calibrated_ladder[1:])
    )
    assert all(
        round(entry["over_probability"] + entry["under_probability"], 6) == 1.0
        for entry in calibrated_ladder
    )
