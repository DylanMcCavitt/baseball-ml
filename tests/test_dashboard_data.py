import json
from datetime import date
from pathlib import Path

from mlb_props_stack.dashboard.lib.data import (
    DashboardSettings,
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
