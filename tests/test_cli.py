from datetime import date

from mlb_props_stack.backtest import WalkForwardBacktestResult
from mlb_props_stack.candidate_models import CandidateStrikeoutModelTrainingResult
from mlb_props_stack.cli import main, render_runtime_summary
from mlb_props_stack.edge import EdgeCandidateBuildResult
from mlb_props_stack.ingest import (
    MLBMetadataIngestResult,
    OddsAPIIngestResult,
    StatcastFeatureIngestResult,
)
from mlb_props_stack.lineup_matchup_features import LineupMatchupFeatureBuildResult
from mlb_props_stack.modeling import StarterStrikeoutBaselineTrainingResult
from mlb_props_stack.model_comparison import StarterStrikeoutModelComparisonResult
from mlb_props_stack.paper_tracking import DailyCandidateWorkflowResult
from mlb_props_stack.pitcher_skill_features import PitcherSkillFeatureBuildResult
from mlb_props_stack.starter_dataset import StarterGameDatasetBuildResult
from mlb_props_stack.workload_leash_features import WorkloadLeashFeatureBuildResult
from tests.stage_gate_fixtures import seed_stage_gate_artifacts


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

    captured_kwargs: dict[str, object] = {}

    def _fake_ingest(**kwargs: object) -> OddsAPIIngestResult:
        captured_kwargs.update(kwargs)
        return result

    monkeypatch.setattr(
        "mlb_props_stack.cli.ingest_odds_api_pitcher_lines_for_date",
        _fake_ingest,
    )

    main(
        [
            "ingest-odds-api-lines",
            "--date",
            "2026-04-21",
            "--output-dir",
            str(tmp_path),
            "--bookmakers",
            "pinnacle,circa",
        ]
    )
    output = capsys.readouterr().out

    assert "Odds API pitcher strikeout ingest complete for 2026-04-21" in output
    assert "candidate_events=15" in output
    assert "skipped_unmatched_events=1" in output
    assert "prop_line_snapshots=28" in output
    assert captured_kwargs["bookmakers"] == ("pinnacle", "circa")


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
        feature_set="expanded",
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
        lambda *, start_date, end_date, output_dir, feature_set: result,
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
    assert "feature_set=expanded" in output
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


def test_starter_game_dataset_cli_renders_output_summary(monkeypatch, tmp_path, capsys):
    result = StarterGameDatasetBuildResult(
        start_date=date(2019, 3, 28),
        end_date=date(2025, 9, 28),
        run_id="20260425T140000Z",
        source_mode="direct_statcast_pitch_log",
        requested_date_count=2377,
        source_date_count=2370,
        row_count=29120,
        missing_target_count=12,
        excluded_start_count=12,
        season_count=7,
        dataset_path=tmp_path / "starter_game_training_dataset.jsonl",
        coverage_report_path=tmp_path / "coverage_report.json",
        coverage_report_markdown_path=tmp_path / "coverage_report.md",
        missing_targets_path=tmp_path / "missing_targets.jsonl",
        source_manifest_path=tmp_path / "source_manifest.jsonl",
        schema_drift_report_path=tmp_path / "schema_drift_report.json",
        timestamp_policy_path=tmp_path / "timestamp_policy.md",
        reproducibility_notes_path=tmp_path / "reproducibility_notes.md",
    )

    monkeypatch.setattr(
        "mlb_props_stack.cli.build_starter_strikeout_dataset",
        lambda *, start_date, end_date, output_dir, chunk_days, max_fetch_workers: result,
    )

    main(
        [
            "build-starter-strikeout-dataset",
            "--start-date",
            "2019-03-28",
            "--end-date",
            "2025-09-28",
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = capsys.readouterr().out

    assert "Starter-game strikeout dataset build complete for 2019-03-28 -> 2025-09-28" in output
    assert "source_mode=direct_statcast_pitch_log" in output
    assert "dataset_rows=29120" in output
    assert "coverage_report_path=" in output
    assert "source_manifest_path=" in output
    assert "timestamp_policy_path=" in output


def test_pitcher_skill_feature_cli_renders_output_summary(monkeypatch, tmp_path, capsys):
    result = PitcherSkillFeatureBuildResult(
        start_date=date(2019, 3, 20),
        end_date=date(2026, 4, 24),
        run_id="20260425T170000Z",
        dataset_row_count=31729,
        feature_row_count=31729,
        pitch_row_count=1200000,
        pitcher_count=812,
        feature_path=tmp_path / "pitcher_skill_features.jsonl",
        feature_report_path=tmp_path / "feature_report.json",
        feature_report_markdown_path=tmp_path / "feature_report.md",
        reproducibility_notes_path=tmp_path / "reproducibility_notes.md",
    )

    monkeypatch.setattr(
        "mlb_props_stack.cli.build_pitcher_skill_features",
        lambda *, start_date, end_date, output_dir, dataset_run_dir: result,
    )

    main(
        [
            "build-pitcher-skill-features",
            "--start-date",
            "2019-03-20",
            "--end-date",
            "2026-04-24",
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = capsys.readouterr().out

    assert "Pitcher skill feature build complete for 2019-03-20 -> 2026-04-24" in output
    assert "dataset_rows=31729" in output
    assert "feature_rows=31729" in output
    assert "feature_report_path=" in output


def test_lineup_matchup_feature_cli_renders_output_summary(monkeypatch, tmp_path, capsys):
    result = LineupMatchupFeatureBuildResult(
        start_date=date(2019, 3, 20),
        end_date=date(2026, 4, 24),
        run_id="20260425T183000Z",
        dataset_row_count=31729,
        feature_row_count=31729,
        batter_feature_row_count=281000,
        pitch_row_count=1200000,
        feature_path=tmp_path / "lineup_matchup_features.jsonl",
        batter_feature_path=tmp_path / "batter_matchup_features.jsonl",
        feature_report_path=tmp_path / "feature_report.json",
        feature_report_markdown_path=tmp_path / "feature_report.md",
        reproducibility_notes_path=tmp_path / "reproducibility_notes.md",
    )

    monkeypatch.setattr(
        "mlb_props_stack.cli.build_lineup_matchup_features",
        lambda *, start_date, end_date, output_dir, dataset_run_dir: result,
    )

    main(
        [
            "build-lineup-matchup-features",
            "--start-date",
            "2019-03-20",
            "--end-date",
            "2026-04-24",
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = capsys.readouterr().out

    assert "Lineup matchup feature build complete for 2019-03-20 -> 2026-04-24" in output
    assert "dataset_rows=31729" in output
    assert "batter_feature_rows=281000" in output
    assert "batter_feature_path=" in output


def test_workload_leash_feature_cli_renders_output_summary(monkeypatch, tmp_path, capsys):
    result = WorkloadLeashFeatureBuildResult(
        start_date=date(2019, 3, 20),
        end_date=date(2026, 4, 24),
        run_id="20260425T190000Z",
        dataset_row_count=31729,
        feature_row_count=31729,
        pitch_row_count=1200000,
        pitcher_count=812,
        feature_path=tmp_path / "workload_leash_features.jsonl",
        feature_report_path=tmp_path / "feature_report.json",
        feature_report_markdown_path=tmp_path / "feature_report.md",
        reproducibility_notes_path=tmp_path / "reproducibility_notes.md",
    )

    monkeypatch.setattr(
        "mlb_props_stack.cli.build_workload_leash_features",
        lambda *, start_date, end_date, output_dir, dataset_run_dir: result,
    )

    main(
        [
            "build-workload-leash-features",
            "--start-date",
            "2019-03-20",
            "--end-date",
            "2026-04-24",
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = capsys.readouterr().out

    assert "Workload leash feature build complete for 2019-03-20 -> 2026-04-24" in output
    assert "dataset_rows=31729" in output
    assert "pitchers=812" in output
    assert "feature_report_path=" in output


def test_model_comparison_cli_renders_output_summary(monkeypatch, tmp_path, capsys):
    result = StarterStrikeoutModelComparisonResult(
        start_date=date(2026, 4, 18),
        end_date=date(2026, 4, 23),
        run_id="20260424T180000Z",
        recommendation="keep_core_only",
        core_training_run_id="20260424T180001Z",
        expanded_training_run_id="20260424T180002Z",
        core_backtest_run_id="20260424T180003Z",
        expanded_backtest_run_id="20260424T180004Z",
        comparison_path=tmp_path / "model_comparison.json",
        comparison_markdown_path=tmp_path / "model_comparison.md",
        reproducibility_notes_path=tmp_path / "reproducibility_notes.md",
    )

    monkeypatch.setattr(
        "mlb_props_stack.cli.compare_starter_strikeout_baselines",
        lambda **_: result,
    )

    main(
        [
            "compare-starter-strikeout-baselines",
            "--start-date",
            "2026-04-18",
            "--end-date",
            "2026-04-23",
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = capsys.readouterr().out

    assert "Starter strikeout model comparison complete for 2026-04-18 -> 2026-04-23" in output
    assert "recommendation=keep_core_only" in output
    assert "core_training_run_id=20260424T180001Z" in output
    assert "expanded_backtest_run_id=20260424T180004Z" in output
    assert "comparison_path=" in output
    assert "comparison_markdown_path=" in output


def test_candidate_strikeout_models_cli_renders_output_summary(monkeypatch, tmp_path, capsys):
    result = CandidateStrikeoutModelTrainingResult(
        start_date=date(2019, 3, 20),
        end_date=date(2026, 4, 24),
        run_id="20260426T180000Z",
        selected_candidate="negative_binomial_glm_count_baseline",
        row_count=31729,
        report_path=tmp_path / "model_comparison.json",
        report_markdown_path=tmp_path / "model_comparison.md",
        selected_model_path=tmp_path / "selected_model.json",
        model_outputs_path=tmp_path / "model_outputs.jsonl",
        reproducibility_notes_path=tmp_path / "reproducibility_notes.md",
    )

    monkeypatch.setattr(
        "mlb_props_stack.cli.train_candidate_strikeout_models",
        lambda **_: result,
    )

    main(
        [
            "train-candidate-strikeout-models",
            "--start-date",
            "2019-03-20",
            "--end-date",
            "2026-04-24",
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = capsys.readouterr().out

    assert "Candidate strikeout model training complete for 2019-03-20 -> 2026-04-24" in output
    assert "selected_candidate=negative_binomial_glm_count_baseline" in output
    assert "rows=31729" in output
    assert "model_comparison_path=" in output
    assert "selected_model_path=" in output
    assert "model_outputs_path=" in output


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
    assert "approved_wagers=3" in output
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
        skip_reason_counts={"missing_line_probability": 2, "unmatched_event_mapping": 1},
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
    assert (
        'skip_reason_counts={"missing_line_probability": 2, "unmatched_event_mapping": 1}'
        in output
    )
    assert "bet_reporting_path=" in output
    assert "join_audit_path=" in output
    assert "clv_summary_path=" in output
    assert "roi_summary_path=" in output
    assert "edge_bucket_summary_path=" in output
    assert "reproducibility_notes_path=" in output


def test_stage_gate_cli_renders_report_and_optional_fail_flag(tmp_path, capsys):
    seed_stage_gate_artifacts(tmp_path, passing=False)

    exit_code = main(["evaluate-stage-gates", "--output-dir", str(tmp_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Stage-gate evaluation complete" in output
    assert "status=research_only" in output
    assert "scoreable_backtest_rows=0" in output
    assert "backtest_skip_rate=1" in output
    assert "settled_paper_bets=0" in output
    assert "paper_same_line_clv_sample=0" in output
    assert "paper_roi=n/a" in output
    assert "backtest_roi=n/a" in output
    assert "report_path=" in output

    fail_exit_code = main(
        [
            "evaluate-stage-gates",
            "--output-dir",
            str(tmp_path),
            "--fail-on-research-only",
        ]
    )

    assert fail_exit_code == 1
