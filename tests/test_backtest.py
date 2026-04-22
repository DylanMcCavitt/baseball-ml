import json
from datetime import UTC, date, datetime
from pathlib import Path

from mlb_props_stack.backtest import build_walk_forward_backtest
from mlb_props_stack.tracking import TrackingConfig


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(payload, sort_keys=True)}\n",
        encoding="utf-8",
    )


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


def _seed_model_run(output_dir: Path) -> Path:
    run_dir = (
        output_dir
        / "normalized"
        / "starter_strikeout_baseline"
        / "start=2026-04-18_end=2026-04-20"
        / "run=20260421T180000Z"
    )
    _write_json(
        run_dir / "baseline_model.json",
        {"model_version": "starter-strikeout-baseline-v1"},
    )
    _write_json(
        run_dir / "date_splits.json",
        {
            "train": ["2026-04-18"],
            "validation": ["2026-04-19"],
            "test": ["2026-04-20"],
        },
    )
    _write_jsonl(
        run_dir / "training_dataset.jsonl",
        [
            {
                "training_row_id": "training-row-1",
                "official_date": "2026-04-20",
                "game_pk": 9001,
                "pitcher_id": 700001,
                "lineup_snapshot_id": "lineup-snapshot-1",
                "features_as_of": "2026-04-20T18:40:00Z",
            },
            {
                "training_row_id": "training-row-2",
                "official_date": "2026-04-18",
                "game_pk": 9002,
                "pitcher_id": 700002,
                "lineup_snapshot_id": "lineup-snapshot-2",
                "features_as_of": "2026-04-18T17:40:00Z",
            },
        ],
    )
    _write_jsonl(
        run_dir / "starter_outcomes.jsonl",
        [
            {
                "outcome_id": "starter-outcome:2026-04-20:9001:700001",
                "official_date": "2026-04-20",
                "game_pk": 9001,
                "pitcher_id": 700001,
                "starter_strikeouts": 7,
            },
            {
                "outcome_id": "starter-outcome:2026-04-18:9002:700002",
                "official_date": "2026-04-18",
                "game_pk": 9002,
                "pitcher_id": 700002,
                "starter_strikeouts": 4,
            },
        ],
    )
    _write_jsonl(
        run_dir / "raw_vs_calibrated_probabilities.jsonl",
        [
            {
                "training_row_id": "training-row-1",
                "official_date": "2026-04-20",
                "split": "test",
                "game_pk": 9001,
                "pitcher_id": 700001,
                "line": 5.5,
                "model_mean": 6.2,
                "count_distribution": {
                    "name": "negative_binomial_global_dispersion_v1",
                    "dispersion_alpha": 0.21,
                },
                "raw_over_probability": 0.56,
                "raw_under_probability": 0.44,
                "calibrated_over_probability": 0.58,
                "calibrated_under_probability": 0.42,
                "model_train_from_date": "2026-04-18",
                "model_train_through_date": "2026-04-19",
                "calibration_method": "isotonic_ladder_probability_calibrator_v1",
                "calibration_training_splits": ["train", "validation"],
                "calibration_sample_count": 48,
                "calibration_fit_from_date": "2026-04-18",
                "calibration_fit_through_date": "2026-04-19",
                "calibration_is_identity": False,
            }
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
                "over_odds": -125,
                "under_odds": 105,
            }
        ],
    )
    _write_jsonl(
        output_dir
        / "normalized"
        / "the_odds_api"
        / "date=2026-04-20"
        / "run=20260420T194500Z"
        / "prop_line_snapshots.jsonl",
        [
            {
                "line_snapshot_id": "line-1-close",
                "official_date": "2026-04-20",
                "captured_at": "2026-04-20T19:45:00Z",
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
                "over_odds": -145,
                "under_odds": 120,
            },
            {
                "line_snapshot_id": "line-2-late-only",
                "official_date": "2026-04-20",
                "captured_at": "2026-04-20T19:40:00Z",
                "sportsbook": "draftkings",
                "sportsbook_title": "DraftKings",
                "event_id": "event-2",
                "game_pk": 9001,
                "odds_matchup_key": "2026-04-20|BOS|NYY|2026-04-20T20:00:00Z",
                "match_status": "matched",
                "commence_time": "2026-04-20T20:00:00Z",
                "player_id": "mlb-pitcher:700001",
                "pitcher_mlb_id": 700001,
                "player_name": "Actionable Arm",
                "market": "pitcher_strikeouts",
                "line": 6.5,
                "over_odds": 130,
                "under_odds": -150,
            },
        ],
    )
    _write_jsonl(
        output_dir
        / "normalized"
        / "the_odds_api"
        / "date=2026-04-18"
        / "run=20260418T181500Z"
        / "prop_line_snapshots.jsonl",
        [
            {
                "line_snapshot_id": "line-3-train",
                "official_date": "2026-04-18",
                "captured_at": "2026-04-18T18:15:00Z",
                "sportsbook": "draftkings",
                "sportsbook_title": "DraftKings",
                "event_id": "event-3",
                "game_pk": 9002,
                "odds_matchup_key": "2026-04-18|SEA|TEX|2026-04-18T20:00:00Z",
                "match_status": "matched",
                "commence_time": "2026-04-18T20:00:00Z",
                "player_id": "mlb-pitcher:700002",
                "pitcher_mlb_id": 700002,
                "player_name": "Train Split Arm",
                "market": "pitcher_strikeouts",
                "line": 4.5,
                "over_odds": -110,
                "under_odds": -110,
            }
        ],
    )


def test_build_walk_forward_backtest_replays_deterministically_and_preserves_traceability(
    tmp_path,
) -> None:
    _seed_model_run(tmp_path)
    _seed_odds_runs(tmp_path)

    fixed_now = lambda: datetime(2026, 4, 21, 19, 0, tzinfo=UTC)
    result = build_walk_forward_backtest(
        start_date=date(2026, 4, 20),
        end_date=date(2026, 4, 20),
        output_dir=tmp_path,
        cutoff_minutes_before_first_pitch=30,
        now=fixed_now,
        tracking_config=_tracking_config(tmp_path),
    )
    first_rows = _load_jsonl(result.backtest_bets_path)
    second_result = build_walk_forward_backtest(
        start_date=date(2026, 4, 20),
        end_date=date(2026, 4, 20),
        output_dir=tmp_path,
        cutoff_minutes_before_first_pitch=30,
        now=fixed_now,
        tracking_config=_tracking_config(tmp_path),
    )
    second_rows = _load_jsonl(second_result.backtest_bets_path)
    reporting_rows = _load_jsonl(result.bet_reporting_path)
    audit_rows = _load_jsonl(result.join_audit_path)
    summary_rows = _load_jsonl(result.backtest_runs_path)
    clv_summary_rows = _load_jsonl(result.clv_summary_path)
    roi_summary_rows = _load_jsonl(result.roi_summary_path)
    edge_bucket_summary_rows = _load_jsonl(result.edge_bucket_summary_path)

    assert first_rows == second_rows
    assert result.run_id == "20260421T190000Z"
    assert result.mlflow_run_id != second_result.mlflow_run_id
    assert result.snapshot_group_count == 2
    assert result.actionable_bet_count == 1
    assert result.skipped_count == 1
    assert result.skip_reason_counts == {"late_snapshot_after_cutoff": 1}

    assert [row["evaluation_status"] for row in first_rows] == [
        "actionable",
        "late_snapshot_after_cutoff",
    ]
    assert first_rows[0]["line_snapshot_id"] == "line-1-cutoff"
    assert first_rows[0]["closing_line_snapshot_id"] == "line-1-close"
    assert first_rows[0]["feature_row_id"] == "training-row-1"
    assert first_rows[0]["lineup_snapshot_id"] == "lineup-snapshot-1"
    assert first_rows[0]["outcome_id"] == "starter-outcome:2026-04-20:9001:700001"
    assert first_rows[0]["settlement_status"] == "win"
    assert first_rows[0]["clv_probability_delta"] > 0.0
    assert first_rows[1]["line_snapshot_id"] is None
    assert first_rows[1]["latest_observed_line_snapshot_id"] == "line-2-late-only"
    assert reporting_rows[0]["backtest_run_id"] == "20260421T190000Z"
    assert reporting_rows[0]["paper_result"] == "win"
    assert reporting_rows[0]["paper_win"] is True
    assert reporting_rows[0]["clv_outcome"] == "beat_closing_line"
    assert reporting_rows[0]["same_line_close_available"] is True
    assert reporting_rows[0]["edge_bucket"] == "2_to_5_pct"
    assert reporting_rows[0]["scatter_model_probability"] == first_rows[0][
        "selected_model_probability"
    ]
    assert reporting_rows[1]["paper_result"] == "not_placed"
    assert reporting_rows[1]["same_line_close_available"] is True
    assert reporting_rows[1]["clv_outcome"] == "no_closing_line"

    assert audit_rows[0]["audit_status"] == "ok"
    assert audit_rows[0]["training_window_before_evaluated_date"] is True
    assert audit_rows[0]["calibration_window_before_evaluated_date"] is True
    assert audit_rows[0]["selected_snapshot_before_cutoff"] is True
    assert audit_rows[1]["audit_status"] == "late_snapshot_after_cutoff"
    assert audit_rows[1]["selected_snapshot_before_cutoff"] is False

    assert summary_rows[0]["row_counts"] == {
        "actionable": 1,
        "below_threshold": 0,
        "skipped": 1,
        "snapshot_groups": 2,
    }
    assert summary_rows[0]["skip_reason_counts"] == {"late_snapshot_after_cutoff": 1}
    assert summary_rows[0]["mlflow_run_id"] == second_result.mlflow_run_id
    assert summary_rows[0]["bet_outcomes"]["placed_bets"] == 1
    assert summary_rows[0]["clv_summary"]["sample_count"] == 1
    assert summary_rows[0]["roi_summary"]["placed_bets"] == 1
    assert summary_rows[0]["reporting_artifacts"]["bet_reporting_path"].endswith(
        "bet_reporting.jsonl"
    )
    assert result.reproducibility_notes_path.exists()
    assert clv_summary_rows[0]["summary_scope"] == "date"
    assert clv_summary_rows[0]["beat_closing_line_count"] == 1
    assert clv_summary_rows[1]["summary_scope"] == "overall"
    assert clv_summary_rows[1]["sample_count"] == 1
    assert roi_summary_rows[0]["summary_scope"] == "date"
    assert roi_summary_rows[0]["roi"] == first_rows[0]["return_on_stake"]
    assert roi_summary_rows[1]["summary_scope"] == "overall"
    assert edge_bucket_summary_rows[0]["edge_bucket"] == "0_to_2_pct"
    assert edge_bucket_summary_rows[1]["edge_bucket"] == "2_to_5_pct"
    assert edge_bucket_summary_rows[1]["bet_count"] == 1
    assert edge_bucket_summary_rows[1]["beat_closing_line_count"] == 1


def test_build_walk_forward_backtest_skips_train_split_rows(tmp_path) -> None:
    _seed_model_run(tmp_path)
    _seed_odds_runs(tmp_path)

    result = build_walk_forward_backtest(
        start_date=date(2026, 4, 18),
        end_date=date(2026, 4, 18),
        output_dir=tmp_path,
        cutoff_minutes_before_first_pitch=30,
        now=lambda: datetime(2026, 4, 21, 19, 5, tzinfo=UTC),
        tracking_config=_tracking_config(tmp_path),
    )
    rows = _load_jsonl(result.backtest_bets_path)
    reporting_rows = _load_jsonl(result.bet_reporting_path)
    clv_summary_rows = _load_jsonl(result.clv_summary_path)
    roi_summary_rows = _load_jsonl(result.roi_summary_path)
    audit_rows = _load_jsonl(result.join_audit_path)

    assert result.actionable_bet_count == 0
    assert result.skipped_count == 1
    assert result.skip_reason_counts == {"train_split_projection": 1}
    assert rows[0]["evaluation_status"] == "train_split_projection"
    assert rows[0]["feature_row_id"] == "training-row-2"
    assert rows[0]["lineup_snapshot_id"] == "lineup-snapshot-2"
    assert reporting_rows[0]["paper_result"] == "not_placed"
    assert reporting_rows[0]["edge_bucket"] is None
    assert audit_rows[0]["audit_status"] == "train_split_projection"
    assert audit_rows[0]["data_split"] == "train"
    assert clv_summary_rows[0]["summary_scope"] == "overall"
    assert clv_summary_rows[0]["placed_bets"] == 0
    assert roi_summary_rows[0]["summary_scope"] == "overall"
    assert roi_summary_rows[0]["placed_bets"] == 0


def test_build_walk_forward_backtest_surfaces_precise_skip_reasons_alongside_scored_rows(
    tmp_path,
) -> None:
    _seed_model_run(tmp_path)
    _seed_odds_runs(tmp_path)
    _write_jsonl(
        tmp_path
        / "normalized"
        / "the_odds_api"
        / "date=2026-04-20"
        / "run=20260420T191000Z"
        / "prop_line_snapshots.jsonl",
        [
            {
                "line_snapshot_id": "line-4-missing-probability",
                "official_date": "2026-04-20",
                "captured_at": "2026-04-20T19:10:00Z",
                "sportsbook": "draftkings",
                "sportsbook_title": "DraftKings",
                "event_id": "event-4",
                "game_pk": 9001,
                "odds_matchup_key": "2026-04-20|BOS|NYY|2026-04-20T20:00:00Z",
                "match_status": "matched",
                "commence_time": "2026-04-20T20:00:00Z",
                "player_id": "mlb-pitcher:700001",
                "pitcher_mlb_id": 700001,
                "player_name": "Actionable Arm",
                "market": "pitcher_strikeouts",
                "line": 4.5,
                "over_odds": -135,
                "under_odds": 115,
            },
            {
                "line_snapshot_id": "line-5-below-threshold",
                "official_date": "2026-04-20",
                "captured_at": "2026-04-20T19:05:00Z",
                "sportsbook": "draftkings",
                "sportsbook_title": "DraftKings",
                "event_id": "event-5",
                "game_pk": 9001,
                "odds_matchup_key": "2026-04-20|BOS|NYY|2026-04-20T20:00:00Z",
                "match_status": "matched",
                "commence_time": "2026-04-20T20:00:00Z",
                "player_id": "mlb-pitcher:700001",
                "pitcher_mlb_id": 700001,
                "player_name": "Actionable Arm",
                "market": "pitcher_strikeouts",
                "line": 5.5,
                "over_odds": -155,
                "under_odds": 130,
            },
            {
                "line_snapshot_id": "line-6-unmatched",
                "official_date": "2026-04-20",
                "captured_at": "2026-04-20T19:00:00Z",
                "sportsbook": "betmgm",
                "sportsbook_title": "BetMGM",
                "event_id": "event-6",
                "game_pk": None,
                "odds_matchup_key": "2026-04-20|BOS|NYY|2026-04-20T20:20:00Z",
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

    result = build_walk_forward_backtest(
        start_date=date(2026, 4, 20),
        end_date=date(2026, 4, 20),
        output_dir=tmp_path,
        cutoff_minutes_before_first_pitch=30,
        now=lambda: datetime(2026, 4, 21, 19, 10, tzinfo=UTC),
        tracking_config=_tracking_config(tmp_path),
    )
    rows = _load_jsonl(result.backtest_bets_path)
    audit_rows = _load_jsonl(result.join_audit_path)
    summary_rows = _load_jsonl(result.backtest_runs_path)

    assert result.actionable_bet_count == 1
    assert result.below_threshold_count == 1
    assert result.skipped_count == 3
    assert result.skip_reason_counts == {
        "late_snapshot_after_cutoff": 1,
        "missing_line_probability": 1,
        "unmatched_event_mapping": 1,
    }

    status_by_snapshot = {
        row["latest_observed_line_snapshot_id"]: row["evaluation_status"] for row in rows
    }
    assert status_by_snapshot["line-1-close"] == "actionable"
    assert status_by_snapshot["line-5-below-threshold"] == "below_threshold"
    assert status_by_snapshot["line-2-late-only"] == "late_snapshot_after_cutoff"
    assert status_by_snapshot["line-4-missing-probability"] == "missing_line_probability"
    assert status_by_snapshot["line-6-unmatched"] == "unmatched_event_mapping"

    audit_status_by_snapshot = {
        row["latest_observed_line_snapshot_id"]: row["audit_status"] for row in audit_rows
    }
    assert audit_status_by_snapshot["line-4-missing-probability"] == "missing_line_probability"
    assert audit_status_by_snapshot["line-6-unmatched"] == "unmatched_event_mapping"

    assert summary_rows[0]["row_counts"] == {
        "actionable": 1,
        "below_threshold": 1,
        "skipped": 3,
        "snapshot_groups": 5,
    }
    assert summary_rows[0]["skip_reason_counts"] == {
        "late_snapshot_after_cutoff": 1,
        "missing_line_probability": 1,
        "unmatched_event_mapping": 1,
    }
