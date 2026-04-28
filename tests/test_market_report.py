import json
from datetime import UTC, date, datetime
from pathlib import Path

from mlb_props_stack.market_report import build_starter_strikeout_market_report
from mlb_props_stack.tracking import TrackingConfig


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


def _tracking_config(tmp_path: Path) -> TrackingConfig:
    return TrackingConfig(tracking_uri=f"file:{tmp_path / 'artifacts' / 'mlruns'}")


def _seed_ml_report_run(output_dir: Path) -> Path:
    run_dir = (
        output_dir
        / "normalized"
        / "starter_strikeout_ml_report"
        / "start=2026-04-18_end=2026-04-20"
        / "run=20260428T202955Z"
    )
    probabilities = [
        {
            "line": 5.5,
            "over_probability": 0.58,
            "under_probability": 0.42,
            "observed_over": True,
        }
    ]
    _write_jsonl(
        run_dir / "starter_strikeout_ml_predictions.jsonl",
        [
            {
                "training_row_id": "starter-training:2026-04-18:9002:700002",
                "feature_row_id": "starter-training:2026-04-18:9002:700002",
                "official_date": "2026-04-18",
                "split": "train",
                "game_pk": 9002,
                "pitcher_id": 700002,
                "pitcher_name": "Train Split Arm",
                "lineup_snapshot_id": "lineup-snapshot-2",
                "features_as_of": "2026-04-18T17:40:00Z",
                "projection_generated_at": "2026-04-18T17:40:00Z",
                "actual_strikeouts": 4,
                "point_projection": 4.8,
                "common_line_probabilities": probabilities,
            },
            {
                "training_row_id": "starter-training:2026-04-20:9001:700001",
                "feature_row_id": "starter-training:2026-04-20:9001:700001",
                "official_date": "2026-04-20",
                "split": "test",
                "game_pk": 9001,
                "pitcher_id": 700001,
                "pitcher_name": "Actionable Arm",
                "lineup_snapshot_id": "lineup-snapshot-1",
                "features_as_of": "2026-04-20T18:40:00Z",
                "projection_generated_at": "2026-04-20T18:40:00Z",
                "actual_strikeouts": 7,
                "point_projection": 6.2,
                "common_line_probabilities": probabilities,
            },
            {
                "training_row_id": "starter-training:2026-04-20:9003:700003",
                "feature_row_id": "starter-training:2026-04-20:9003:700003",
                "official_date": "2026-04-20",
                "split": "test",
                "game_pk": 9003,
                "pitcher_id": 700003,
                "pitcher_name": "Missing Timestamp Arm",
                "lineup_snapshot_id": "lineup-snapshot-3",
                "features_as_of": None,
                "projection_generated_at": None,
                "actual_strikeouts": 3,
                "point_projection": 4.0,
                "common_line_probabilities": [
                    {
                        "line": 4.5,
                        "over_probability": 0.45,
                        "under_probability": 0.55,
                        "observed_over": False,
                    }
                ],
            },
        ],
    )
    return run_dir


def _seed_odds_runs(output_dir: Path) -> None:
    _write_jsonl(
        output_dir
        / "normalized"
        / "the_odds_api"
        / "date=2026-04-20"
        / "run=20260420T180000Z"
        / "prop_line_snapshots.jsonl",
        [
            {
                "line_snapshot_id": "line-1-open",
                "official_date": "2026-04-20",
                "captured_at": "2026-04-20T18:00:00Z",
                "sportsbook": "draftkings",
                "sportsbook_title": "DraftKings",
                "event_id": "event-1",
                "game_pk": 9001,
                "odds_matchup_key": "2026-04-20|BOS|NYY|2026-04-20T20:00:00Z",
                "match_status": "matched",
                "commence_time": "2026-04-20T20:00:00Z",
                "player_id": "mlb-pitcher:700001",
                "pitcher_mlb_id": 700001,
                "player_name": "Actionable Arm",
                "market": "pitcher_strikeouts",
                "line": 5.5,
                "over_odds": -110,
                "under_odds": -110,
            }
        ],
    )
    _write_jsonl(
        output_dir
        / "normalized"
        / "the_odds_api"
        / "date=2026-04-20"
        / "run=20260420T192000Z"
        / "prop_line_snapshots.jsonl",
        [
            {
                "line_snapshot_id": "line-1-cutoff",
                "official_date": "2026-04-20",
                "captured_at": "2026-04-20T19:20:00Z",
                "sportsbook": "draftkings",
                "sportsbook_title": "DraftKings",
                "event_id": "event-1",
                "game_pk": 9001,
                "odds_matchup_key": "2026-04-20|BOS|NYY|2026-04-20T20:00:00Z",
                "match_status": "matched",
                "commence_time": "2026-04-20T20:00:00Z",
                "player_id": "mlb-pitcher:700001",
                "pitcher_mlb_id": 700001,
                "player_name": "Actionable Arm",
                "market": "pitcher_strikeouts",
                "line": 5.5,
                "over_odds": -110,
                "under_odds": -110,
            },
            {
                "line_snapshot_id": "line-missing-timestamp",
                "official_date": "2026-04-20",
                "captured_at": "2026-04-20T19:10:00Z",
                "sportsbook": "caesars",
                "sportsbook_title": "Caesars",
                "event_id": "event-missing-timestamp",
                "game_pk": 9003,
                "odds_matchup_key": "2026-04-20|SEA|TEX|2026-04-20T20:00:00Z",
                "match_status": "matched",
                "commence_time": "2026-04-20T20:00:00Z",
                "player_id": "mlb-pitcher:700003",
                "pitcher_mlb_id": 700003,
                "player_name": "Missing Timestamp Arm",
                "market": "pitcher_strikeouts",
                "line": 4.5,
                "over_odds": -110,
                "under_odds": -110,
            },
            {
                "line_snapshot_id": "line-unmatched",
                "official_date": "2026-04-20",
                "captured_at": "2026-04-20T19:05:00Z",
                "sportsbook": "betmgm",
                "sportsbook_title": "BetMGM",
                "event_id": "event-unmatched",
                "game_pk": None,
                "odds_matchup_key": "2026-04-20|BOS|NYY|2026-04-20T20:00:00Z",
                "match_status": "unmatched",
                "commence_time": "2026-04-20T20:00:00Z",
                "player_id": "odds-player:actionable-arm",
                "pitcher_mlb_id": None,
                "player_name": "Actionable Arm",
                "market": "pitcher_strikeouts",
                "line": 5.5,
                "over_odds": -110,
                "under_odds": -110,
            },
        ],
    )
    _write_jsonl(
        output_dir
        / "normalized"
        / "the_odds_api"
        / "date=2026-04-20"
        / "run=20260420T194000Z"
        / "prop_line_snapshots.jsonl",
        [
            {
                "line_snapshot_id": "line-1-close",
                "official_date": "2026-04-20",
                "captured_at": "2026-04-20T19:40:00Z",
                "sportsbook": "draftkings",
                "sportsbook_title": "DraftKings",
                "event_id": "event-1",
                "game_pk": 9001,
                "odds_matchup_key": "2026-04-20|BOS|NYY|2026-04-20T20:00:00Z",
                "match_status": "matched",
                "commence_time": "2026-04-20T20:00:00Z",
                "player_id": "mlb-pitcher:700001",
                "pitcher_mlb_id": 700001,
                "player_name": "Actionable Arm",
                "market": "pitcher_strikeouts",
                "line": 5.5,
                "over_odds": -125,
                "under_odds": 105,
            }
        ],
    )


def test_market_report_consumes_ml_predictions_and_backtest_join_path(tmp_path) -> None:
    ml_root = tmp_path / "ml"
    odds_root = tmp_path / "odds"
    output_root = tmp_path / "out"
    ml_report_run_dir = _seed_ml_report_run(ml_root)
    _seed_odds_runs(odds_root)

    result = build_starter_strikeout_market_report(
        start_date=date(2026, 4, 18),
        end_date=date(2026, 4, 20),
        output_dir=output_root,
        ml_report_run_dir=ml_report_run_dir,
        odds_input_dir=odds_root,
        cutoff_minutes_before_first_pitch=30,
        now=lambda: datetime(2026, 4, 21, 20, 0, tzinfo=UTC),
        tracking_config=_tracking_config(tmp_path),
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    backtest_rows = _load_jsonl(result.backtest_result.backtest_bets_path)
    audit_rows = _load_jsonl(result.backtest_result.join_audit_path)
    markdown = result.report_markdown_path.read_text(encoding="utf-8")

    assert result.scoreable_row_count == 1
    assert result.skipped_row_count == 2
    assert report["row_counts"]["scoreable_rows"] == 1
    assert report["join_failure_reasons"] == {
        "missing_projection_timestamp": 1,
        "unmatched_event_mapping": 1,
    }
    assert report["book_line_coverage"][0]["sportsbook"] == "betmgm"
    assert report["calibration_by_line_bucket"][0]["line_bucket"] == "5.5"
    assert report["clv"]["sample_count"] == 1
    assert report["roi"]["placed_bets"] == 1
    assert report["scope_guardrails"]["forced_joins_allowed"] is False
    assert report["adapter_summary"]["adapted_probability_rows"] == 3
    assert "Sportsbook Market Report" in markdown
    assert result.adapted_model_run_dir.joinpath("raw_vs_calibrated_probabilities.jsonl").exists()

    status_by_snapshot = {
        row["latest_observed_line_snapshot_id"]: row["evaluation_status"]
        for row in backtest_rows
    }
    assert status_by_snapshot["line-1-close"] == "actionable"
    assert status_by_snapshot["line-unmatched"] == "unmatched_event_mapping"
    assert status_by_snapshot["line-missing-timestamp"] == "missing_projection_timestamp"
    assert any(row["projection_timestamp_status"] == "ok" for row in audit_rows)
