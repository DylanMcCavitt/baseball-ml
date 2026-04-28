import json
from datetime import date
from pathlib import Path

from mlb_props_stack.dashboard.lib.data import (
    DashboardSettings,
    current_slate_metrics,
    get_feature_importance,
    get_optional_feature_diagnostics,
    get_pmf,
    group_board_by_pitcher,
    list_available_board_dates,
    load_board_dataframe,
    ticker_context,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_board_can_replay_historical_backtest_rows(tmp_path: Path) -> None:
    run_dir = (
        tmp_path
        / "normalized"
        / "walk_forward_backtest"
        / "start=2026-04-21_end=2026-04-22"
        / "run=20260422T180000Z"
    )
    _write_jsonl(
        run_dir / "backtest_runs.jsonl",
        [
            {
                "backtest_run_id": "20260422T180000Z",
                "evaluated_dates": ["2026-04-21", "2026-04-22"],
            }
        ],
    )
    _write_jsonl(
        run_dir / "bet_reporting.jsonl",
        [
            {
                "official_date": "2026-04-22",
                "evaluation_status": "actionable",
                "selected_side": "over",
                "player_id": "mlb-player-1",
                "pitcher_mlb_id": 700001,
                "player_name": "Replay Arm",
                "game_pk": 824440,
                "line": 5.5,
                "selected_model_probability": 0.61,
                "selected_market_probability": 0.53,
                "market_over_probability": 0.53,
                "market_under_probability": 0.47,
                "selected_odds": 120,
                "expected_value_pct": 0.031,
                "model_run_id": "20260422T180000Z",
                "model_version": "starter-strikeout-baseline-v1",
                "decision_snapshot_captured_at": "2026-04-22T18:25:00Z",
                "commence_time": "2026-04-22T23:10:00Z",
                "settlement_status": "win",
                "clv_outcome": "beat_closing_line",
                "reason": "over clears minimum edge threshold (8.00% >= 3.00%)",
            }
        ],
    )

    available_dates = list_available_board_dates(output_root=tmp_path)
    assert available_dates == ["2026-04-22"]

    board, source = load_board_dataframe(
        tmp_path,
        target_date=date(2026, 4, 22),
        settings=DashboardSettings(),
    )

    assert source == "walk_forward_backtest"
    assert len(board) == 1
    assert board.iloc[0]["pitcher"] == "Replay Arm"
    assert bool(board.iloc[0]["cleared"]) is True
    assert board.iloc[0]["source"] == "walk_forward_backtest"
    assert "win" in board.iloc[0]["note"]
    assert "beat closing line" in board.iloc[0]["note"]

    ticker = ticker_context(board, settings=DashboardSettings())
    assert ticker["live_label"] == "HIST REPLAY"


def test_board_daily_candidates_use_shared_final_wager_gates(tmp_path: Path) -> None:
    run_dir = (
        tmp_path
        / "normalized"
        / "daily_candidates"
        / "date=2026-04-23"
        / "run=20260423T181500Z"
    )
    _write_jsonl(
        run_dir / "daily_candidates.jsonl",
        [
            {
                "daily_candidate_id": "line-high-hold|starter-strikeout-baseline-v1",
                "official_date": "2026-04-23",
                "slate_rank": 1,
                "actionable_rank": 1,
                "approved_rank": None,
                "bet_placed": False,
                "wager_approved": False,
                "wager_blocked_reason": "hold above max",
                "evaluation_status": "actionable",
                "player_id": "mlb-pitcher:700003",
                "pitcher_mlb_id": 700003,
                "player_name": "High Hold Arm",
                "game_pk": 9003,
                "line": 5.5,
                "selected_side": "over",
                "selected_odds": -200,
                "over_odds": -200,
                "under_odds": -200,
                "selected_model_probability": 0.68,
                "selected_market_probability": 0.60,
                "edge_pct": 0.08,
                "expected_value_pct": 0.02,
                "model_run_id": "20260422T180000Z",
                "model_version": "starter-strikeout-baseline-v1",
                "sportsbook_title": "DraftKings",
                "line_snapshot_id": "line-high-hold",
                "captured_at": "2026-04-23T18:10:00Z",
                "reason": "over clears minimum edge threshold (8.00% >= 4.50%)",
            }
        ],
    )

    board, source = load_board_dataframe(
        tmp_path,
        target_date=date(2026, 4, 23),
        settings=DashboardSettings(),
    )

    assert source == "daily_candidates"
    assert len(board) == 1
    assert bool(board.iloc[0]["cleared"]) is False
    assert bool(board.iloc[0]["wager_approved"]) is False
    assert board.iloc[0]["wager_blocked_reason"] == "hold above max"
    assert "hold above max" in board.iloc[0]["note"]
    assert current_slate_metrics(board)["plays_cleared"] == 0


def test_board_edge_candidates_preserve_rebuilt_approval_gates_and_distribution(
    tmp_path: Path,
) -> None:
    edge_run = (
        tmp_path
        / "normalized"
        / "edge_candidates"
        / "date=2026-04-20"
        / "run=20260420T170000Z"
    )
    distribution = [
        {"strikeouts": 5, "probability": 0.20},
        {"strikeouts": 6, "probability": 0.20},
        {"strikeouts": 7, "probability": 0.35},
        {"strikeouts": 8, "probability": 0.25},
    ]
    _write_jsonl(
        edge_run / "edge_candidates.jsonl",
        [
            {
                "candidate_id": "line-approved|candidate-v1",
                "official_date": "2026-04-20",
                "evaluation_status": "actionable",
                "approval_status": "approved",
                "approval_allowed": True,
                "approval_reason": "Approved by model-only validation evidence and market edge gates.",
                "research_readiness_status": "research_only",
                "validation_recommendation": "conditional_go_for_betting_layer_rebuild",
                "validation_threshold_status": "thresholds_observed_from_calibration",
                "validation_min_edge_pct": 0.06,
                "line_snapshot_id": "line-approved",
                "player_id": "mlb-pitcher:700001",
                "pitcher_mlb_id": 700001,
                "player_name": "Distribution Arm",
                "game_pk": 9001,
                "event_id": "event-1",
                "market": "pitcher_strikeouts",
                "line": 6.5,
                "selected_side": "over",
                "selected_odds": -110,
                "over_odds": -110,
                "under_odds": -110,
                "model_projection": 6.9,
                "model_confidence": 0.60,
                "model_confidence_bucket": "0.6_to_0.7",
                "model_over_probability": 0.60,
                "model_under_probability": 0.40,
                "selected_model_probability": 0.60,
                "market_over_probability": 0.50,
                "market_under_probability": 0.50,
                "selected_market_probability": 0.50,
                "edge_pct": 0.10,
                "expected_value_pct": 0.145,
                "stake_fraction": 0.02,
                "model_run_id": "20260420T150000Z",
                "model_version": "starter-strikeout-candidate-v1",
                "sportsbook": "draftkings",
                "sportsbook_title": "DraftKings",
                "captured_at": "2026-04-20T16:00:00Z",
                "probability_distribution": distribution,
                "feature_group_contributions": [
                    {"feature_group": "pitcher_skill", "absolute_contribution": 1.2, "share": 0.4},
                    {"feature_group": "matchup", "absolute_contribution": 0.9, "share": 0.3},
                    {"feature_group": "workload", "absolute_contribution": 0.6, "share": 0.2},
                    {"feature_group": "context", "absolute_contribution": 0.3, "share": 0.1},
                ],
                "correlation_group_key": "2026-04-20|9001|700001|6.500000",
                "correlation_group_size": 2,
                "correlation_group_rank": 1,
                "reason": "over clears minimum edge threshold (10.00% >= 3.00%)",
            },
            {
                "candidate_id": "line-duplicate|candidate-v1",
                "official_date": "2026-04-20",
                "evaluation_status": "actionable",
                "approval_status": "rejected",
                "approval_allowed": False,
                "approval_reason": "Rejected as a correlated duplicate within the same pitcher/game/line group.",
                "research_readiness_status": "research_only",
                "line_snapshot_id": "line-duplicate",
                "player_id": "mlb-pitcher:700001",
                "pitcher_mlb_id": 700001,
                "player_name": "Distribution Arm",
                "game_pk": 9001,
                "event_id": "event-1",
                "market": "pitcher_strikeouts",
                "line": 6.5,
                "selected_side": "over",
                "selected_odds": -110,
                "over_odds": -110,
                "under_odds": -110,
                "model_over_probability": 0.60,
                "model_under_probability": 0.40,
                "selected_model_probability": 0.60,
                "market_over_probability": 0.50,
                "market_under_probability": 0.50,
                "selected_market_probability": 0.50,
                "edge_pct": 0.10,
                "expected_value_pct": 0.145,
                "stake_fraction": 0.02,
                "model_run_id": "20260420T150000Z",
                "model_version": "starter-strikeout-candidate-v1",
                "sportsbook": "fanduel",
                "sportsbook_title": "FanDuel",
                "captured_at": "2026-04-20T16:00:00Z",
                "probability_distribution": distribution,
                "correlation_group_key": "2026-04-20|9001|700001|6.500000",
                "correlation_group_size": 2,
                "correlation_group_rank": 2,
                "reason": "over clears minimum edge threshold (10.00% >= 3.00%)",
            },
        ],
    )
    model_run = (
        tmp_path
        / "normalized"
        / "candidate_strikeout_models"
        / "start=2026-04-16_end=2026-04-20"
        / "run=20260420T150000Z"
    )
    _write_json(
        model_run / "selected_model.json",
        {
            "run_id": "20260420T150000Z",
            "feature_group_contributions": [
                {"feature_group": "pitcher_skill", "absolute_contribution": 1.2, "share": 0.4}
            ],
        },
    )
    _write_jsonl(
        model_run / "model_outputs.jsonl",
        [
            {
                "official_date": "2026-04-20",
                "pitcher_id": 700001,
                "point_projection": 6.9,
                "probability_distribution": distribution,
            }
        ],
    )

    board, source = load_board_dataframe(
        tmp_path,
        target_date=date(2026, 4, 20),
        settings=DashboardSettings(),
    )
    pmf_rows, pmf_source = get_pmf(
        tmp_path,
        official_date="2026-04-20",
        pitcher_mlb_id=700001,
        line=6.5,
        model_run_id="20260420T150000Z",
    )

    assert source == "edge_candidates"
    assert len(board) == 2
    assert current_slate_metrics(board)["plays_cleared"] == 1
    assert bool(board.iloc[0]["wager_approved"]) is True
    assert bool(board.iloc[1]["wager_approved"]) is False
    assert "correlated duplicate" in board.iloc[1]["note"]
    assert board.iloc[0]["research_readiness_status"] == "research_only"
    assert ticker_context(board, settings=DashboardSettings())["live_label"] == "RESEARCH ONLY"
    assert board.iloc[0]["probability_distribution"] == distribution
    assert pmf_source is not None
    assert pmf_source["model_mean"] == 6.9
    assert [row["k"] for row in pmf_rows] == [5, 6, 7, 8]


def test_board_joins_sportsbook_provenance_from_line_snapshot(tmp_path: Path) -> None:
    candidate_run = (
        tmp_path
        / "normalized"
        / "daily_candidates"
        / "date=2026-04-23"
        / "run=20260423T210236Z"
    )
    snapshot_id = (
        "prop-line:betrivers:528817a2bf72047f1124e81a7ae55de9:"
        "mlb-pitcher:594798:5_5:20260423T194826Z"
    )
    _write_jsonl(
        candidate_run / "daily_candidates.jsonl",
        [
            {
                "daily_candidate_id": f"{snapshot_id}|starter-strikeout-baseline-v1",
                "official_date": "2026-04-23",
                "evaluation_status": "actionable",
                "player_id": "mlb-pitcher:594798",
                "pitcher_mlb_id": 594798,
                "player_name": "Jacob deGrom",
                "game_pk": 822912,
                "line": 5.5,
                "selected_side": "under",
                "selected_odds": 200,
                "over_odds": -278,
                "under_odds": 200,
                "selected_model_probability": 0.72,
                "selected_market_probability": 0.50,
                "expected_value_pct": 1.27,
                "model_run_id": "20260423T201434Z",
                "model_version": "starter-strikeout-baseline-v1",
                "line_snapshot_id": snapshot_id,
                "captured_at": "2026-04-23T19:48:26Z",
                "reason": "under clears minimum edge threshold (22.00% >= 4.50%)",
            }
        ],
    )
    odds_run = (
        tmp_path
        / "normalized"
        / "the_odds_api"
        / "date=2026-04-23"
        / "run=20260423T194825Z"
    )
    _write_jsonl(
        odds_run / "prop_line_snapshots.jsonl",
        [
            {
                "line_snapshot_id": snapshot_id,
                "sportsbook_title": "BetRivers",
                "event_id": "528817a2bf72047f1124e81a7ae55de9",
                "game_pk": 822912,
                "commence_time": "2026-04-24T00:06:00Z",
                "market_last_update": "2026-04-23T19:47:32Z",
                "captured_at": "2026-04-23T19:48:26Z",
                "line": 5.5,
                "over_odds": -278,
                "under_odds": 200,
                "player_name": "Jacob deGrom",
            }
        ],
    )

    board, source = load_board_dataframe(
        tmp_path,
        target_date=date(2026, 4, 23),
        settings=DashboardSettings(),
    )

    assert source == "daily_candidates"
    assert board.iloc[0]["sportsbook"] == "BetRivers"
    assert board.iloc[0]["sportsbook_key"] == "betrivers"
    assert board.iloc[0]["source_event_id"] == "528817a2bf72047f1124e81a7ae55de9"
    assert board.iloc[0]["market_last_update"].isoformat() == "2026-04-23T19:47:32+00:00"
    assert "BetRivers" in board.iloc[0]["provenance"]
    assert "event 528817a2" in board.iloc[0]["provenance"]


def test_group_board_by_pitcher_keeps_best_row_with_hidden_summary(tmp_path: Path) -> None:
    run_dir = (
        tmp_path
        / "normalized"
        / "daily_candidates"
        / "date=2026-04-23"
        / "run=20260423T210236Z"
    )
    _write_jsonl(
        run_dir / "daily_candidates.jsonl",
        [
            {
                "daily_candidate_id": "line-a|starter-strikeout-baseline-v1",
                "official_date": "2026-04-23",
                "evaluation_status": "actionable",
                "player_id": "mlb-pitcher:663436",
                "pitcher_mlb_id": 663436,
                "player_name": "Davis Martin",
                "game_pk": 825096,
                "line": 4.5,
                "selected_side": "over",
                "selected_odds": 180,
                "over_odds": 180,
                "under_odds": -245,
                "selected_model_probability": 0.70,
                "expected_value_pct": 0.94,
                "edge_pct": 0.36,
                "model_run_id": "20260423T201434Z",
                "model_version": "starter-strikeout-baseline-v1",
                "sportsbook_title": "BetRivers",
                "line_snapshot_id": "line-a",
                "captured_at": "2026-04-23T19:48:26Z",
                "reason": "over clears minimum edge threshold (36.00% >= 4.50%)",
            },
            {
                "daily_candidate_id": "line-b|starter-strikeout-baseline-v1",
                "official_date": "2026-04-23",
                "evaluation_status": "actionable",
                "player_id": "mlb-pitcher:663436",
                "pitcher_mlb_id": 663436,
                "player_name": "Davis Martin",
                "game_pk": 825096,
                "line": 3.5,
                "selected_side": "over",
                "selected_odds": -115,
                "over_odds": -115,
                "under_odds": -115,
                "selected_model_probability": 0.69,
                "expected_value_pct": 0.58,
                "edge_pct": 0.34,
                "model_run_id": "20260423T201434Z",
                "model_version": "starter-strikeout-baseline-v1",
                "sportsbook_title": "DraftKings",
                "line_snapshot_id": "line-b",
                "captured_at": "2026-04-23T19:48:26Z",
                "reason": "over clears minimum edge threshold (34.00% >= 4.50%)",
            },
            {
                "daily_candidate_id": "line-c|starter-strikeout-baseline-v1",
                "official_date": "2026-04-23",
                "evaluation_status": "actionable",
                "player_id": "mlb-pitcher:594798",
                "pitcher_mlb_id": 594798,
                "player_name": "Jacob deGrom",
                "game_pk": 822912,
                "line": 5.5,
                "selected_side": "under",
                "selected_odds": 200,
                "over_odds": -278,
                "under_odds": 200,
                "selected_model_probability": 0.72,
                "expected_value_pct": 1.27,
                "edge_pct": 0.44,
                "model_run_id": "20260423T201434Z",
                "model_version": "starter-strikeout-baseline-v1",
                "sportsbook_title": "BetRivers",
                "line_snapshot_id": "line-c",
                "captured_at": "2026-04-23T19:48:26Z",
                "reason": "under clears minimum edge threshold (44.00% >= 4.50%)",
            },
        ],
    )

    board, _ = load_board_dataframe(
        tmp_path,
        target_date=date(2026, 4, 23),
        settings=DashboardSettings(),
    )
    grouped = group_board_by_pitcher(board)
    davis_row = grouped[grouped["pitcher"] == "Davis Martin"].iloc[0]

    assert len(grouped) == 2
    assert davis_row["line"] == 4.5
    assert davis_row["line_row_count"] == 2
    assert davis_row["hidden_line_row_count"] == 1
    assert davis_row["sportsbook_count"] == 2
    assert "BetRivers OVER 4.5" in davis_row["line_group_summary"]
    assert "DraftKings OVER 3.5" in davis_row["line_group_summary"]
    assert "grouped view hides 1 book/line rows" in davis_row["note"]


def test_pitcher_drilldown_uses_replayed_model_run(tmp_path: Path) -> None:
    older_run = (
        tmp_path
        / "normalized"
        / "starter_strikeout_baseline"
        / "start=2026-04-18_end=2026-04-22"
        / "run=20260422T180000Z"
    )
    newer_run = (
        tmp_path
        / "normalized"
        / "starter_strikeout_baseline"
        / "start=2026-04-18_end=2026-04-22"
        / "run=20260422T190000Z"
    )

    _write_jsonl(
        older_run / "ladder_probabilities.jsonl",
        [
            {
                "official_date": "2026-04-22",
                "pitcher_id": 700001,
                "model_mean": 6.0,
                "count_distribution": {"dispersion_alpha": 0.2},
            }
        ],
    )
    _write_json(
        older_run / "evaluation_summary.json",
        {
            "run_id": "20260422T180000Z",
            "top_feature_importance": [{"name": "older_feature", "importance": 0.9}],
        },
    )
    _write_jsonl(
        older_run / "training_dataset.jsonl",
        [
            {
                "official_date": "2026-04-22",
                "training_row_id": "older-row",
                "older_feature": 1.5,
            }
        ],
    )

    _write_jsonl(
        newer_run / "ladder_probabilities.jsonl",
        [
            {
                "official_date": "2026-04-22",
                "pitcher_id": 700001,
                "model_mean": 9.0,
                "count_distribution": {"dispersion_alpha": 0.6},
            }
        ],
    )
    _write_json(
        newer_run / "evaluation_summary.json",
        {
            "run_id": "20260422T190000Z",
            "top_feature_importance": [{"name": "newer_feature", "importance": 1.2}],
        },
    )
    _write_jsonl(
        newer_run / "training_dataset.jsonl",
        [
            {
                "official_date": "2026-04-22",
                "training_row_id": "newer-row",
                "newer_feature": 2.5,
            }
        ],
    )

    _, ladder_row = get_pmf(
        tmp_path,
        official_date="2026-04-22",
        pitcher_mlb_id=700001,
        line=5.5,
        model_run_id="20260422T180000Z",
    )
    assert ladder_row is not None
    assert ladder_row["model_mean"] == 6.0

    importance = get_feature_importance(tmp_path, run_id="20260422T180000Z")
    assert list(importance["name"]) == ["older_feature"]
    assert list(importance["last_value"]) == [1.5]


def test_pitcher_pmf_can_resolve_daily_inference_run_id(tmp_path: Path) -> None:
    inference_run = (
        tmp_path
        / "normalized"
        / "starter_strikeout_inference"
        / "date=2026-04-23"
        / "run=20260423T201434Z"
    )
    _write_jsonl(
        inference_run / "ladder_probabilities.jsonl",
        [
            {
                "official_date": "2026-04-23",
                "pitcher_id": 594798,
                "model_mean": 6.8,
                "count_distribution": {"dispersion_alpha": 0.25},
            }
        ],
    )
    baseline_run = (
        tmp_path
        / "normalized"
        / "starter_strikeout_baseline"
        / "start=2026-04-18_end=2026-04-23"
        / "run=20260422T205727Z"
    )
    _write_json(
        baseline_run / "evaluation_summary.json",
        {"run_id": "20260422T205727Z"},
    )

    pmf_rows, ladder_row = get_pmf(
        tmp_path,
        official_date="2026-04-23",
        pitcher_mlb_id=594798,
        line=5.5,
        model_run_id="20260423T201434Z",
    )

    assert ladder_row is not None
    assert ladder_row["model_mean"] == 6.8
    assert pmf_rows


def test_optional_feature_diagnostics_resolve_active_schema_and_missing_sources(
    tmp_path: Path,
) -> None:
    source_run = (
        tmp_path
        / "normalized"
        / "starter_strikeout_baseline"
        / "start=2026-04-20_end=2026-04-21"
        / "run=20260422T202712Z"
    )
    _write_json(
        source_run / "evaluation_summary.json",
        {
            "run_id": "20260422T202712Z",
            "date_splits": {"train": ["2026-04-20"]},
        },
    )
    _write_json(
        source_run / "baseline_model.json",
        {
            "model_version": "starter-strikeout-baseline-v1",
            "encoded_feature_names": [
                "pitch_sample_size",
                "pitcher_k_rate",
                "projected_lineup_k_rate",
            ],
        },
    )
    _write_jsonl(
        source_run / "training_dataset.jsonl",
        [
            {
                "official_date": "2026-04-20",
                "training_row_id": "train-1",
                "pitch_sample_size": 120,
                "pitcher_k_rate": 0.28,
                "projected_lineup_k_rate": 0.23,
                "park_factor_status": "missing_park_factor_source",
                "park_factor": 1.02,
                "weather_status": "missing_weather_source",
                "weather_wind_mph": 8.0,
            },
            {
                "official_date": "2026-04-20",
                "training_row_id": "train-2",
                "pitch_sample_size": 100,
                "pitcher_k_rate": 0.24,
                "projected_lineup_k_rate": 0.25,
                "park_factor_status": "missing_park_factor_source",
                "park_factor": 0.98,
                "weather_status": "missing_weather_source",
                "weather_wind_mph": 6.0,
            },
            {
                "official_date": "2026-04-21",
                "training_row_id": "held-out-1",
                "pitch_sample_size": 110,
                "pitcher_k_rate": 0.26,
                "projected_lineup_k_rate": None,
                "weather_status": "missing_weather_source",
            },
        ],
    )

    inference_run = (
        tmp_path
        / "normalized"
        / "starter_strikeout_inference"
        / "date=2026-04-23"
        / "run=20260423T210236Z"
    )
    _write_json(
        inference_run / "baseline_model.json",
        {
            "model_version": "starter-strikeout-baseline-v1",
            "source_model_run_id": "20260422T202712Z",
            "encoded_feature_names": [
                "pitch_sample_size",
                "pitcher_k_rate",
                "projected_lineup_k_rate",
            ],
        },
    )

    feature_run = (
        tmp_path
        / "normalized"
        / "statcast_search"
        / "date=2026-04-23"
        / "run=20260423T180000Z"
    )
    _write_jsonl(
        feature_run / "game_context_features.jsonl",
        [
            {
                "official_date": "2026-04-23",
                "game_pk": 1,
                "pitcher_id": 100,
                "park_factor_status": "missing_park_factor_source",
                "park_factor": 1.0,
                "weather_status": "missing_weather_source",
                "weather_temperature_f": None,
                "weather_wind_mph": 9.0,
            },
            {
                "official_date": "2026-04-23",
                "game_pk": 2,
                "pitcher_id": 101,
                "park_factor_status": "missing_park_factor_source",
                "park_factor": 1.1,
                "weather_status": "missing_weather_source",
                "weather_temperature_f": None,
                "weather_wind_mph": 12.0,
            },
        ],
    )
    _write_jsonl(
        feature_run / "lineup_daily_features.jsonl",
        [
            {
                "official_date": "2026-04-23",
                "game_pk": 1,
                "pitcher_id": 100,
                "projected_lineup_k_rate": None,
            },
            {
                "official_date": "2026-04-23",
                "game_pk": 2,
                "pitcher_id": 101,
                "projected_lineup_k_rate": None,
            },
        ],
    )
    _write_jsonl(
        feature_run / "pitcher_daily_features.jsonl",
        [
            {"official_date": "2026-04-23", "game_pk": 1, "pitcher_id": 100},
            {"official_date": "2026-04-23", "game_pk": 2, "pitcher_id": 101},
        ],
    )

    diagnostics = get_optional_feature_diagnostics(
        tmp_path,
        target_date=date(2026, 4, 23),
    )

    assert diagnostics["active_model_run_id"] == "20260423T210236Z"
    assert diagnostics["source_model_run_id"] == "20260422T202712Z"
    assert diagnostics["encoded_feature_count"] == 3
    assert diagnostics["active_optional_features"] == ["projected_lineup_k_rate"]
    assert diagnostics["source_selection_row_count"] == 2
    assert diagnostics["source_training_row_count"] == 3

    families = {row["key"]: row for row in diagnostics["family_rows"]}
    assert families["lineup_aggregate_features"]["status"] == "active"
    assert families["split_features"]["status"] == "excluded_below_coverage"
    assert families["split_features"]["source_train_coverage_label"] == "0/2 (0%)"

    assert families["weather"]["status"] == "missing_source"
    assert families["weather"]["target_coverage_label"] == "0/2 (0%)"
    assert "target-date weather source artifact missing" in families["weather"]["reason"]
    assert (
        "weather_wind_mph present; expected weather_wind_speed_mph"
        in families["weather"]["schema_notes"]
    )

    assert families["park_factors"]["status"] == "missing_source"
    assert (
        "park_factor present; expected park_k_factor"
        in families["park_factors"]["schema_notes"]
    )
    assert families["umpire"]["status"] == "missing_source"
