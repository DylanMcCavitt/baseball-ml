import json
from datetime import UTC, date, datetime
from pathlib import Path

from mlb_props_stack.backtest import build_walk_forward_backtest


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
    )
    first_rows = _load_jsonl(result.backtest_bets_path)
    second_result = build_walk_forward_backtest(
        start_date=date(2026, 4, 20),
        end_date=date(2026, 4, 20),
        output_dir=tmp_path,
        cutoff_minutes_before_first_pitch=30,
        now=fixed_now,
    )
    second_rows = _load_jsonl(second_result.backtest_bets_path)
    audit_rows = _load_jsonl(result.join_audit_path)
    summary_rows = _load_jsonl(result.backtest_runs_path)

    assert first_rows == second_rows
    assert result.run_id == "20260421T190000Z"
    assert result.snapshot_group_count == 2
    assert result.actionable_bet_count == 1
    assert result.skipped_count == 1

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
    assert summary_rows[0]["bet_outcomes"]["placed_bets"] == 1
    assert summary_rows[0]["clv_summary"]["sample_count"] == 1


def test_build_walk_forward_backtest_skips_train_split_rows(tmp_path) -> None:
    _seed_model_run(tmp_path)
    _seed_odds_runs(tmp_path)

    result = build_walk_forward_backtest(
        start_date=date(2026, 4, 18),
        end_date=date(2026, 4, 18),
        output_dir=tmp_path,
        cutoff_minutes_before_first_pitch=30,
        now=lambda: datetime(2026, 4, 21, 19, 5, tzinfo=UTC),
    )
    rows = _load_jsonl(result.backtest_bets_path)
    audit_rows = _load_jsonl(result.join_audit_path)

    assert result.actionable_bet_count == 0
    assert result.skipped_count == 1
    assert rows[0]["evaluation_status"] == "train_split_projection"
    assert rows[0]["feature_row_id"] == "training-row-2"
    assert rows[0]["lineup_snapshot_id"] == "lineup-snapshot-2"
    assert audit_rows[0]["audit_status"] == "train_split_projection"
    assert audit_rows[0]["data_split"] == "train"
