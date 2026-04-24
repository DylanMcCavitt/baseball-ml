import json
from datetime import date
from pathlib import Path

from mlb_props_stack.dashboard.lib.data import (
    DashboardSettings,
    current_slate_metrics,
    get_feature_importance,
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
