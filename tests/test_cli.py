from datetime import date

from mlb_props_stack.backtest import WalkForwardBacktestResult
from mlb_props_stack.cli import main, render_runtime_summary
from mlb_props_stack.edge import EdgeCandidateBuildResult
from mlb_props_stack.ingest import (
    MLBMetadataIngestResult,
    OddsAPIIngestResult,
    StatcastFeatureIngestResult,
)
from mlb_props_stack.modeling import StarterStrikeoutBaselineTrainingResult
from mlb_props_stack.paper_tracking import DailyCandidateWorkflowResult


def test_runtime_summary_includes_future_hooks():
    summary = render_runtime_summary()

    assert "MLB Props Stack" in summary
    assert "tracking_uri=file:./artifacts/mlruns" in summary
    assert "training_experiment_name=mlb-props-stack-starter-strikeout-training" in summary
    assert "backtest_experiment_name=mlb-props-stack-walk-forward-backtest" in summary
    assert "dashboard_module=mlb_props_stack.dashboard.app" in summary


def test_ingest_cli_renders_output_summary(monkeypatch, tmp_path, capsys):
    result = MLBMetadataIngestResult(
        target_date=date(2026, 4, 21),
        run_id="20260421T180000Z",
        schedule_raw_path=tmp_path / "schedule.json",
        feed_live_raw_paths=(tmp_path / "feed.json",),
        games_path=tmp_path / "games.jsonl",
        probable_starters_path=tmp_path / "probable_starters.jsonl",
        lineup_snapshots_path=tmp_path / "lineup_snapshots.jsonl",
        game_count=15,
        probable_starter_count=30,
        lineup_snapshot_count=30,
    )

    monkeypatch.setattr(
        "mlb_props_stack.cli.ingest_mlb_metadata_for_date",
        lambda *, target_date, output_dir: result,
    )

    main(
        [
            "ingest-mlb-metadata",
            "--date",
            "2026-04-21",
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = capsys.readouterr().out

    assert "MLB metadata ingest complete for 2026-04-21" in output
    assert "games=15" in output


def test_odds_api_ingest_cli_renders_output_summary(monkeypatch, tmp_path, capsys):
    result = OddsAPIIngestResult(
        target_date=date(2026, 4, 21),
        run_id="20260421T180000Z",
        mlb_games_path=tmp_path / "games.jsonl",
        mlb_probable_starters_path=tmp_path / "probable_starters.jsonl",
        events_raw_path=tmp_path / "events.json",
        event_odds_raw_paths=(tmp_path / "event-1.json",),
        event_mappings_path=tmp_path / "event_game_mappings.jsonl",
        prop_line_snapshots_path=tmp_path / "prop_line_snapshots.jsonl",
        candidate_event_count=15,
        matched_event_count=14,
        unmatched_event_count=1,
        skipped_unmatched_event_count=1,
        matched_events_without_props_count=3,
        prop_line_count=28,
        resolved_pitcher_prop_count=24,
        unresolved_pitcher_prop_count=4,
        skipped_prop_count=2,
    )

    monkeypatch.setattr(
        "mlb_props_stack.cli.ingest_odds_api_pitcher_lines_for_date",
        lambda *, target_date, output_dir, api_key: result,
    )

    main(
        [
            "ingest-odds-api-lines",
            "--date",
            "2026-04-21",
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = capsys.readouterr().out

    assert "Odds API pitcher strikeout ingest complete for 2026-04-21" in output
    assert "candidate_events=15" in output
    assert "skipped_unmatched_events=1" in output
    assert "prop_line_snapshots=28" in output


def test_statcast_feature_ingest_cli_renders_output_summary(monkeypatch, tmp_path, capsys):
    result = StatcastFeatureIngestResult(
        target_date=date(2026, 4, 21),
        history_start_date=date(2026, 3, 22),
        history_end_date=date(2026, 4, 20),
        run_id="20260421T180000Z",
        mlb_games_path=tmp_path / "games.jsonl",
        mlb_probable_starters_path=tmp_path / "probable_starters.jsonl",
        mlb_lineup_snapshots_path=tmp_path / "lineup_snapshots.jsonl",
        pull_manifest_path=tmp_path / "pull_manifest.jsonl",
        pitch_level_base_path=tmp_path / "pitch_level_base.jsonl",
        pitcher_daily_features_path=tmp_path / "pitcher_daily_features.jsonl",
        lineup_daily_features_path=tmp_path / "lineup_daily_features.jsonl",
        game_context_features_path=tmp_path / "game_context_features.jsonl",
        raw_pull_count=11,
        pitch_level_record_count=1234,
        pitcher_feature_count=2,
        lineup_feature_count=2,
        game_context_feature_count=2,
    )

    monkeypatch.setattr(
        "mlb_props_stack.cli.ingest_statcast_features_for_date",
        lambda *, target_date, output_dir, history_days: result,
    )

    main(
        [
            "ingest-statcast-features",
            "--date",
            "2026-04-21",
            "--output-dir",
            str(tmp_path),
            "--history-days",
            "30",
        ]
    )
    output = capsys.readouterr().out

    assert "Statcast feature build complete for 2026-04-21" in output
    assert "raw_pulls=11" in output
    assert "pitch_level_rows=1234" in output


def test_starter_strikeout_training_cli_renders_output_summary(monkeypatch, tmp_path, capsys):
    result = StarterStrikeoutBaselineTrainingResult(
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 20),
        run_id="20260421T180000Z",
        mlflow_run_id="mlflow-training-run-1",
        mlflow_experiment_name="mlb-props-stack-starter-strikeout-training",
        row_count=48,
        outcome_count=48,
        dispersion_alpha=0.183742,
        dataset_path=tmp_path / "training_dataset.jsonl",
        outcomes_path=tmp_path / "starter_outcomes.jsonl",
        date_splits_path=tmp_path / "date_splits.json",
        model_path=tmp_path / "baseline_model.json",
        evaluation_path=tmp_path / "evaluation.json",
        ladder_probabilities_path=tmp_path / "ladder_probabilities.jsonl",
        probability_calibrator_path=tmp_path / "probability_calibrator.json",
        raw_vs_calibrated_path=tmp_path / "raw_vs_calibrated_probabilities.jsonl",
        calibration_summary_path=tmp_path / "calibration_summary.json",
        evaluation_summary_path=tmp_path / "evaluation_summary.json",
        evaluation_summary_markdown_path=tmp_path / "evaluation_summary.md",
        reproducibility_notes_path=tmp_path / "reproducibility_notes.md",
        held_out_status="beating_benchmark",
        held_out_model_rmse=2.113245,
        held_out_benchmark_rmse=2.418881,
        held_out_model_mae=1.887612,
        held_out_benchmark_mae=2.002114,
        previous_run_id="20260420T170000Z",
    )

    monkeypatch.setattr(
        "mlb_props_stack.cli.train_starter_strikeout_baseline",
        lambda *, start_date, end_date, output_dir: result,
    )

    main(
        [
            "train-starter-strikeout-baseline",
            "--start-date",
            "2026-04-01",
            "--end-date",
            "2026-04-20",
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = capsys.readouterr().out

    assert "Starter strikeout baseline training complete for 2026-04-01 -> 2026-04-20" in output
    assert "mlflow_run_id=mlflow-training-run-1" in output
    assert "mlflow_experiment_name=mlb-props-stack-starter-strikeout-training" in output
    assert "training_rows=48" in output
    assert "dispersion_alpha=0.183742" in output
    assert "held_out_status=beating_benchmark" in output
    assert "held_out_model_rmse=2.113245" in output
    assert "held_out_benchmark_mae=2.002114" in output
    assert "previous_run_id=20260420T170000Z" in output
    assert "model_path=" in output
    assert "ladder_probabilities_path=" in output
    assert "probability_calibrator_path=" in output
    assert "raw_vs_calibrated_path=" in output
    assert "calibration_summary_path=" in output
    assert "evaluation_summary_path=" in output
    assert "evaluation_summary_markdown_path=" in output
    assert "reproducibility_notes_path=" in output


def test_edge_candidate_cli_renders_output_summary(monkeypatch, tmp_path, capsys):
    result = EdgeCandidateBuildResult(
        target_date=date(2026, 4, 20),
        run_id="20260421T183000Z",
        model_version="starter-strikeout-baseline-v1",
        model_run_id="20260421T180000Z",
        line_snapshots_path=tmp_path / "prop_line_snapshots.jsonl",
        model_path=tmp_path / "baseline_model.json",
        ladder_probabilities_path=tmp_path / "ladder_probabilities.jsonl",
        edge_candidates_path=tmp_path / "edge_candidates.jsonl",
        line_count=12,
        scored_line_count=10,
        actionable_count=3,
        below_threshold_count=7,
        skipped_line_count=2,
    )

    monkeypatch.setattr(
        "mlb_props_stack.cli.build_edge_candidates_for_date",
        lambda *, target_date, output_dir, model_run_dir: result,
    )

    main(
        [
            "build-edge-candidates",
            "--date",
            "2026-04-20",
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = capsys.readouterr().out

    assert "Edge candidate build complete for 2026-04-20" in output
    assert "model_version=starter-strikeout-baseline-v1" in output
    assert "actionable_candidates=3" in output
    assert "edge_candidates_path=" in output


def test_daily_candidate_cli_renders_output_summary(monkeypatch, tmp_path, capsys):
    result = DailyCandidateWorkflowResult(
        target_date=date(2026, 4, 22),
        run_id="20260422T180000Z",
        inference_run_id="20260422T180000Z",
        edge_candidate_run_id="20260422T180100Z",
        daily_candidates_path=tmp_path / "daily_candidates.jsonl",
        paper_results_path=tmp_path / "paper_results.jsonl",
        scored_candidate_count=9,
        actionable_candidate_count=3,
        settled_result_count=12,
        pending_result_count=2,
    )

    monkeypatch.setattr(
        "mlb_props_stack.cli.build_daily_candidate_workflow",
        lambda **_: result,
    )

    main(
        [
            "build-daily-candidates",
            "--date",
            "2026-04-22",
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = capsys.readouterr().out

    assert "Daily candidate workflow complete for 2026-04-22" in output
    assert "actionable_candidates=3" in output
    assert "settled_paper_results=12" in output
    assert "daily_candidates_path=" in output
    assert "paper_results_path=" in output


def test_walk_forward_backtest_cli_renders_output_summary(monkeypatch, tmp_path, capsys):
    result = WalkForwardBacktestResult(
        start_date=date(2026, 4, 19),
        end_date=date(2026, 4, 20),
        run_id="20260421T190000Z",
        mlflow_run_id="mlflow-backtest-run-1",
        mlflow_experiment_name="mlb-props-stack-walk-forward-backtest",
        model_version="starter-strikeout-baseline-v1",
        model_run_id="20260421T180000Z",
        cutoff_minutes_before_first_pitch=30,
        backtest_bets_path=tmp_path / "backtest_bets.jsonl",
        bet_reporting_path=tmp_path / "bet_reporting.jsonl",
        backtest_runs_path=tmp_path / "backtest_runs.jsonl",
        join_audit_path=tmp_path / "join_audit.jsonl",
        clv_summary_path=tmp_path / "clv_summary.jsonl",
        roi_summary_path=tmp_path / "roi_summary.jsonl",
        edge_bucket_summary_path=tmp_path / "edge_bucket_summary.jsonl",
        reproducibility_notes_path=tmp_path / "reproducibility_notes.md",
        snapshot_group_count=8,
        actionable_bet_count=3,
        below_threshold_count=2,
        skipped_count=3,
    )

    monkeypatch.setattr(
        "mlb_props_stack.cli.build_walk_forward_backtest",
        lambda **_: result,
    )

    main(
        [
            "build-walk-forward-backtest",
            "--start-date",
            "2026-04-19",
            "--end-date",
            "2026-04-20",
            "--output-dir",
            str(tmp_path),
            "--cutoff-minutes-before-first-pitch",
            "30",
        ]
    )
    output = capsys.readouterr().out

    assert "Walk-forward backtest complete for 2026-04-19 -> 2026-04-20" in output
    assert "mlflow_run_id=mlflow-backtest-run-1" in output
    assert "mlflow_experiment_name=mlb-props-stack-walk-forward-backtest" in output
    assert "snapshot_groups=8" in output
    assert "actionable_bets=3" in output
    assert "bet_reporting_path=" in output
    assert "join_audit_path=" in output
    assert "clv_summary_path=" in output
    assert "roi_summary_path=" in output
    assert "edge_bucket_summary_path=" in output
    assert "reproducibility_notes_path=" in output
