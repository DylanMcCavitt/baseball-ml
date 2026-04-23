from __future__ import annotations

from datetime import UTC, date, datetime
import json
from pathlib import Path

from mlb_props_stack.cli import main
from mlb_props_stack.dashboard.lib.data import (
    DashboardSettings,
    current_slate_metrics,
    load_board_dataframe,
)
from mlb_props_stack.wager_card import build_wager_card, render_wager_card_summary


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _seed_daily_candidates(tmp_path: Path, rows: list[dict]) -> Path:
    run_dir = (
        tmp_path
        / "normalized"
        / "daily_candidates"
        / "date=2026-04-23"
        / "run=20260423T180000Z"
    )
    _write_jsonl(run_dir / "daily_candidates.jsonl", rows)
    return run_dir


def _approved_row() -> dict:
    return {
        "daily_candidate_id": "approved-line",
        "daily_candidate_run_id": "20260423T180000Z",
        "official_date": "2026-04-23",
        "slate_rank": 1,
        "approved_rank": 1,
        "evaluation_status": "actionable",
        "wager_approved": True,
        "wager_gate_status": "approved",
        "wager_blocked_reason": "approved",
        "wager_gate_notes": [],
        "player_id": "mlb-pitcher:700001",
        "pitcher_mlb_id": 700001,
        "player_name": "Approved Arm",
        "game_pk": 9001,
        "odds_matchup_key": "2026-04-23|AAA|BBB|2026-04-23T23:10:00Z",
        "line": 5.5,
        "selected_side": "over",
        "selected_odds": -105,
        "over_odds": -105,
        "under_odds": -105,
        "model_over_probability": 0.62,
        "model_under_probability": 0.38,
        "selected_model_probability": 0.62,
        "selected_market_probability": 0.50,
        "edge_pct": 0.12,
        "expected_value_pct": 0.18,
        "stake_fraction": 0.02,
        "kelly_units": 2.0,
        "model_run_id": "",
        "model_version": "starter-strikeout-baseline-v1",
        "sportsbook": "draftkings",
        "sportsbook_title": "DraftKings",
        "line_snapshot_id": "approved-line-snapshot",
        "captured_at": "2026-04-23T18:00:00Z",
        "pitcher_status": "probable",
        "reason": "over clears minimum edge threshold (12.00% >= 4.50%)",
    }


def _blocked_row() -> dict:
    return {
        **_approved_row(),
        "daily_candidate_id": "blocked-line",
        "slate_rank": 2,
        "approved_rank": None,
        "wager_approved": False,
        "wager_gate_status": "blocked",
        "wager_blocked_reason": "hold above max",
        "wager_gate_notes": ["hold above max"],
        "player_id": "mlb-pitcher:700002",
        "pitcher_mlb_id": 700002,
        "player_name": "Blocked Arm",
        "game_pk": 9002,
        "selected_odds": -200,
        "over_odds": -200,
        "under_odds": -200,
        "sportsbook": "fanduel",
        "sportsbook_title": "FanDuel",
        "line_snapshot_id": "blocked-line-snapshot",
    }


def test_build_wager_card_excludes_blocked_rows_by_default(tmp_path: Path) -> None:
    _seed_daily_candidates(tmp_path, [_approved_row(), _blocked_row()])

    result = build_wager_card(
        target_date=date(2026, 4, 23),
        output_dir=tmp_path,
        now=lambda: datetime(2026, 4, 23, 18, 30, tzinfo=UTC),
    )

    assert result.approved_count == 1
    assert result.blocked_count == 1
    assert result.included_count == 1
    assert result.wager_card_path.exists()
    artifact_rows = _load_jsonl(result.wager_card_path)
    assert [row["pitcher"] for row in artifact_rows] == ["Approved Arm"]
    assert artifact_rows[0]["start_time"] == "2026-04-23T23:10:00Z"

    summary = render_wager_card_summary(result)
    assert "Approved Arm" in summary
    assert "Blocked Arm" not in summary
    assert "approved_wagers=1" in summary


def test_build_wager_card_can_include_blocked_diagnostics(tmp_path: Path) -> None:
    _seed_daily_candidates(tmp_path, [_approved_row(), _blocked_row()])

    result = build_wager_card(
        target_date=date(2026, 4, 23),
        output_dir=tmp_path,
        include_rejected=True,
        now=lambda: datetime(2026, 4, 23, 18, 30, tzinfo=UTC),
    )

    assert result.included_count == 2
    artifact_rows = _load_jsonl(result.wager_card_path)
    assert [row["status"] for row in artifact_rows] == ["approved", "blocked"]

    summary = render_wager_card_summary(result)
    assert "Blocked candidates" in summary
    assert "Blocked Arm" in summary
    assert "hold above max" in summary


def test_wager_card_prints_empty_card_when_no_wagers_approved(
    tmp_path: Path,
    capsys,
) -> None:
    _seed_daily_candidates(tmp_path, [_blocked_row()])

    exit_code = main(
        [
            "build-wager-card",
            "--date",
            "2026-04-23",
            "--output-dir",
            str(tmp_path),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "approved_wagers=0" in output
    assert "No approved wagers passed the final wager gates." in output


def test_wager_card_approved_count_matches_dashboard_board_count(tmp_path: Path) -> None:
    _seed_daily_candidates(tmp_path, [_approved_row(), _blocked_row()])

    result = build_wager_card(
        target_date=date(2026, 4, 23),
        output_dir=tmp_path,
        now=lambda: datetime(2026, 4, 23, 18, 30, tzinfo=UTC),
    )
    board, source = load_board_dataframe(
        tmp_path,
        target_date=date(2026, 4, 23),
        settings=DashboardSettings(),
    )

    assert source == "daily_candidates"
    assert result.approved_count == current_slate_metrics(board)["plays_cleared"]
