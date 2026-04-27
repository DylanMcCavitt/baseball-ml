"""CLI entrypoint for the scaffold and ingestion slices."""

from __future__ import annotations

import argparse
from datetime import date
import json

from .backfill import (
    ALL_SOURCES,
    BackfillResult,
    backfill_historical,
)
from .backtest import (
    BACKTEST_CHECKLIST,
    WalkForwardBacktestResult,
    build_walk_forward_backtest,
)
from .candidate_models import (
    CandidateStrikeoutModelTrainingResult,
    train_candidate_strikeout_models,
)
from .config import StackConfig
from .data_alignment import (
    DEFAULT_COVERAGE_THRESHOLD,
    check_data_alignment,
    render_data_alignment_summary,
)
from .edge import EdgeCandidateBuildResult, build_edge_candidates_for_date
from .ingest import (
    DEFAULT_HISTORY_DAYS,
    MLBMetadataIngestResult,
    OddsAPIIngestResult,
    StatcastFeatureIngestResult,
    UmpireIngestResult,
    WeatherIngestResult,
    ingest_mlb_metadata_for_date,
    ingest_odds_api_pitcher_lines_for_date,
    ingest_statcast_features_for_date,
    ingest_umpire_for_date,
    ingest_weather_for_date,
)
from .modeling import (
    FEATURE_SET_CHOICES,
    FEATURE_SET_EXPANDED,
    StarterStrikeoutInferenceResult,
    StarterStrikeoutBaselineTrainingResult,
    train_starter_strikeout_baseline,
)
from .model_comparison import (
    compare_starter_strikeout_baselines,
    render_model_comparison_summary,
)
from .paper_tracking import (
    DailyCandidateWorkflowResult,
    build_daily_candidate_workflow,
)
from .pitcher_skill_features import (
    PitcherSkillFeatureBuildResult,
    build_pitcher_skill_features,
)
from .stage_gates import evaluate_stage_gates, render_stage_gate_summary
from .starter_dataset import (
    DEFAULT_DIRECT_CHUNK_DAYS,
    StarterGameDatasetBuildResult,
    build_starter_strikeout_dataset,
)
from .ingest.statcast_ingest import DEFAULT_MAX_FETCH_WORKERS
from .lineup_matchup_features import (
    LineupMatchupFeatureBuildResult,
    build_lineup_matchup_features,
)
from .tracking import TrackingConfig
from .wager_card import build_wager_card, render_wager_card_summary
from .workload_leash_features import (
    WorkloadLeashFeatureBuildResult,
    build_workload_leash_features,
)


def _parse_bookmaker_argument(value: str | None) -> tuple[str, ...] | None:
    """Parse ``--bookmakers pinnacle,circa`` into a filter tuple."""
    if value is None:
        return None
    keys = tuple(key.strip() for key in value.split(",") if key.strip())
    if not keys:
        raise ValueError(
            "--bookmakers must list at least one non-empty sportsbook key."
        )
    return keys


def render_runtime_summary() -> str:
    """Return a human-readable snapshot of the local runtime baseline."""
    config = StackConfig()
    tracking = TrackingConfig()
    lines = [
        "MLB Props Stack",
        f"market={config.market}",
        f"min_edge_pct={config.min_edge_pct:.2%}",
        f"kelly_fraction={config.kelly_fraction:.2f}",
        f"devig_mode={config.devig_mode}",
        f"tracking_uri={tracking.tracking_uri}",
        f"training_experiment_name={tracking.training_experiment_name}",
        f"backtest_experiment_name={tracking.backtest_experiment_name}",
        f"dashboard_module={tracking.dashboard_module}",
        "backtest_checklist:",
    ]
    lines.extend(f"- {item}" for item in BACKTEST_CHECKLIST)
    return "\n".join(lines)


def render_ingest_summary(result: MLBMetadataIngestResult) -> str:
    """Return a human-readable summary for one ingest run."""
    lines = [
        f"MLB metadata ingest complete for {result.target_date.isoformat()}",
        f"run_id={result.run_id}",
        f"games={result.game_count}",
        f"probable_starters={result.probable_starter_count}",
        f"lineup_snapshots={result.lineup_snapshot_count}",
        f"schedule_raw_path={result.schedule_raw_path}",
        f"games_path={result.games_path}",
        f"probable_starters_path={result.probable_starters_path}",
        f"lineup_snapshots_path={result.lineup_snapshots_path}",
    ]
    return "\n".join(lines)


def render_odds_api_ingest_summary(result: OddsAPIIngestResult) -> str:
    """Return a human-readable summary for one sportsbook ingest run."""
    lines = [
        f"Odds API pitcher strikeout ingest complete for {result.target_date.isoformat()}",
        f"run_id={result.run_id}",
        f"candidate_events={result.candidate_event_count}",
        f"matched_events={result.matched_event_count}",
        f"unmatched_events={result.unmatched_event_count}",
        f"skipped_unmatched_events={result.skipped_unmatched_event_count}",
        f"matched_events_without_props={result.matched_events_without_props_count}",
        f"prop_line_snapshots={result.prop_line_count}",
        f"resolved_pitcher_props={result.resolved_pitcher_prop_count}",
        f"unresolved_pitcher_props={result.unresolved_pitcher_prop_count}",
        f"skipped_prop_groups={result.skipped_prop_count}",
        f"mlb_games_path={result.mlb_games_path}",
        f"mlb_probable_starters_path={result.mlb_probable_starters_path}",
        f"events_raw_path={result.events_raw_path}",
        f"event_mappings_path={result.event_mappings_path}",
        f"prop_line_snapshots_path={result.prop_line_snapshots_path}",
    ]
    return "\n".join(lines)


def render_weather_ingest_summary(result: WeatherIngestResult) -> str:
    """Return a human-readable summary for one pregame weather build."""
    lines = [
        f"Pregame weather ingest complete for {result.target_date.isoformat()}",
        f"run_id={result.run_id}",
        f"snapshots={result.snapshot_count}",
        f"outdoor_snapshots={result.outdoor_snapshot_count}",
        f"roof_closed_snapshots={result.roof_closed_snapshot_count}",
        f"missing_venue_metadata={result.missing_venue_metadata_count}",
        f"missing_source={result.missing_source_count}",
        f"raw_snapshot_files={len(result.raw_snapshot_paths)}",
        f"mlb_games_path={result.mlb_games_path}",
        f"weather_snapshots_path={result.weather_snapshots_path}",
    ]
    return "\n".join(lines)


def render_umpire_ingest_summary(result: UmpireIngestResult) -> str:
    """Return a human-readable summary for one pregame umpire ingest build."""
    lines = [
        f"Pregame umpire ingest complete for {result.target_date.isoformat()}",
        f"run_id={result.run_id}",
        f"snapshots={result.snapshot_count}",
        f"ok_snapshots={result.ok_snapshot_count}",
        f"missing_source={result.missing_source_count}",
        f"history_start_date={result.history_start_date.isoformat()}",
        f"history_end_date={result.history_end_date.isoformat()}",
        f"raw_snapshot_files={len(result.raw_snapshot_paths)}",
        f"mlb_games_path={result.mlb_games_path}",
        f"umpire_snapshots_path={result.umpire_snapshots_path}",
    ]
    return "\n".join(lines)


def render_statcast_feature_ingest_summary(result: StatcastFeatureIngestResult) -> str:
    """Return a human-readable summary for one Statcast feature build."""
    lines = [
        f"Statcast feature build complete for {result.target_date.isoformat()}",
        f"run_id={result.run_id}",
        f"history_start_date={result.history_start_date.isoformat()}",
        f"history_end_date={result.history_end_date.isoformat()}",
        f"raw_pulls={result.raw_pull_count}",
        f"pitch_level_rows={result.pitch_level_record_count}",
        f"pitcher_daily_features={result.pitcher_feature_count}",
        f"lineup_daily_features={result.lineup_feature_count}",
        f"game_context_features={result.game_context_feature_count}",
        f"mlb_games_path={result.mlb_games_path}",
        f"mlb_probable_starters_path={result.mlb_probable_starters_path}",
        f"mlb_lineup_snapshots_path={result.mlb_lineup_snapshots_path}",
        f"pull_manifest_path={result.pull_manifest_path}",
        f"pitch_level_base_path={result.pitch_level_base_path}",
        f"pitcher_daily_features_path={result.pitcher_daily_features_path}",
        f"lineup_daily_features_path={result.lineup_daily_features_path}",
        f"game_context_features_path={result.game_context_features_path}",
    ]
    return "\n".join(lines)


def render_starter_strikeout_training_summary(
    result: StarterStrikeoutBaselineTrainingResult,
) -> str:
    """Return a human-readable summary for one baseline training run."""
    def _render_metric(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.6f}"

    lines = [
        (
            "Starter strikeout baseline training complete for "
            f"{result.start_date.isoformat()} -> {result.end_date.isoformat()}"
        ),
        f"run_id={result.run_id}",
        f"feature_set={result.feature_set}",
        f"mlflow_run_id={result.mlflow_run_id}",
        f"mlflow_experiment_name={result.mlflow_experiment_name}",
        f"training_rows={result.row_count}",
        f"starter_outcomes={result.outcome_count}",
        f"dispersion_alpha={result.dispersion_alpha:.6f}",
        f"held_out_status={result.held_out_status}",
        f"held_out_model_rmse={_render_metric(result.held_out_model_rmse)}",
        f"held_out_benchmark_rmse={_render_metric(result.held_out_benchmark_rmse)}",
        f"held_out_model_mae={_render_metric(result.held_out_model_mae)}",
        f"held_out_benchmark_mae={_render_metric(result.held_out_benchmark_mae)}",
        f"previous_run_id={result.previous_run_id or 'n/a'}",
        f"dataset_path={result.dataset_path}",
        f"outcomes_path={result.outcomes_path}",
        f"date_splits_path={result.date_splits_path}",
        f"model_path={result.model_path}",
        f"evaluation_path={result.evaluation_path}",
        f"ladder_probabilities_path={result.ladder_probabilities_path}",
        f"probability_calibrator_path={result.probability_calibrator_path}",
        f"raw_vs_calibrated_path={result.raw_vs_calibrated_path}",
        f"calibration_summary_path={result.calibration_summary_path}",
        f"evaluation_summary_path={result.evaluation_summary_path}",
        f"evaluation_summary_markdown_path={result.evaluation_summary_markdown_path}",
        f"reproducibility_notes_path={result.reproducibility_notes_path}",
    ]
    return "\n".join(lines)


def render_starter_game_dataset_summary(result: StarterGameDatasetBuildResult) -> str:
    """Return a human-readable summary for one starter-game dataset build."""
    lines = [
        (
            "Starter-game strikeout dataset build complete for "
            f"{result.start_date.isoformat()} -> {result.end_date.isoformat()}"
        ),
        f"run_id={result.run_id}",
        f"source_mode={result.source_mode}",
        f"requested_dates={result.requested_date_count}",
        f"source_dates={result.source_date_count}",
        f"dataset_rows={result.row_count}",
        f"missing_targets={result.missing_target_count}",
        f"excluded_starts={result.excluded_start_count}",
        f"seasons={result.season_count}",
        f"dataset_path={result.dataset_path}",
        f"coverage_report_path={result.coverage_report_path}",
        f"coverage_report_markdown_path={result.coverage_report_markdown_path}",
        f"missing_targets_path={result.missing_targets_path}",
        f"source_manifest_path={result.source_manifest_path}",
        f"schema_drift_report_path={result.schema_drift_report_path}",
        f"timestamp_policy_path={result.timestamp_policy_path}",
        f"reproducibility_notes_path={result.reproducibility_notes_path}",
    ]
    return "\n".join(lines)


def render_pitcher_skill_feature_summary(result: PitcherSkillFeatureBuildResult) -> str:
    """Return a human-readable summary for one pitcher skill feature build."""
    lines = [
        (
            "Pitcher skill feature build complete for "
            f"{result.start_date.isoformat()} -> {result.end_date.isoformat()}"
        ),
        f"run_id={result.run_id}",
        f"dataset_rows={result.dataset_row_count}",
        f"feature_rows={result.feature_row_count}",
        f"pitch_rows={result.pitch_row_count}",
        f"pitchers={result.pitcher_count}",
        f"feature_path={result.feature_path}",
        f"feature_report_path={result.feature_report_path}",
        f"feature_report_markdown_path={result.feature_report_markdown_path}",
        f"reproducibility_notes_path={result.reproducibility_notes_path}",
    ]
    return "\n".join(lines)


def render_lineup_matchup_feature_summary(result: LineupMatchupFeatureBuildResult) -> str:
    """Return a human-readable summary for one lineup matchup feature build."""
    lines = [
        (
            "Lineup matchup feature build complete for "
            f"{result.start_date.isoformat()} -> {result.end_date.isoformat()}"
        ),
        f"run_id={result.run_id}",
        f"dataset_rows={result.dataset_row_count}",
        f"feature_rows={result.feature_row_count}",
        f"batter_feature_rows={result.batter_feature_row_count}",
        f"pitch_rows={result.pitch_row_count}",
        f"feature_path={result.feature_path}",
        f"batter_feature_path={result.batter_feature_path}",
        f"feature_report_path={result.feature_report_path}",
        f"feature_report_markdown_path={result.feature_report_markdown_path}",
        f"reproducibility_notes_path={result.reproducibility_notes_path}",
    ]
    return "\n".join(lines)


def render_workload_leash_feature_summary(result: WorkloadLeashFeatureBuildResult) -> str:
    """Return a human-readable summary for one workload/leash feature build."""
    lines = [
        (
            "Workload leash feature build complete for "
            f"{result.start_date.isoformat()} -> {result.end_date.isoformat()}"
        ),
        f"run_id={result.run_id}",
        f"dataset_rows={result.dataset_row_count}",
        f"feature_rows={result.feature_row_count}",
        f"pitch_rows={result.pitch_row_count}",
        f"pitchers={result.pitcher_count}",
        f"feature_path={result.feature_path}",
        f"feature_report_path={result.feature_report_path}",
        f"feature_report_markdown_path={result.feature_report_markdown_path}",
        f"reproducibility_notes_path={result.reproducibility_notes_path}",
    ]
    return "\n".join(lines)


def render_starter_strikeout_inference_summary(
    result: StarterStrikeoutInferenceResult,
) -> str:
    """Return a human-readable summary for one target-date inference run."""
    lines = [
        f"Starter strikeout inference complete for {result.target_date.isoformat()}",
        f"run_id={result.run_id}",
        f"source_model_run_id={result.source_model_run_id}",
        f"projection_count={result.projection_count}",
        f"source_model_path={result.source_model_path}",
        f"feature_run_dir={result.feature_run_dir}",
        f"model_path={result.model_path}",
        f"ladder_probabilities_path={result.ladder_probabilities_path}",
    ]
    return "\n".join(lines)


def render_candidate_strikeout_model_summary(
    result: CandidateStrikeoutModelTrainingResult,
) -> str:
    """Return a human-readable summary for one candidate-family model run."""
    lines = [
        (
            "Candidate strikeout model training complete for "
            f"{result.start_date.isoformat()} -> {result.end_date.isoformat()}"
        ),
        f"run_id={result.run_id}",
        f"selected_candidate={result.selected_candidate}",
        f"rows={result.row_count}",
        f"model_comparison_path={result.report_path}",
        f"model_comparison_markdown_path={result.report_markdown_path}",
        f"selected_model_path={result.selected_model_path}",
        f"model_outputs_path={result.model_outputs_path}",
        f"reproducibility_notes_path={result.reproducibility_notes_path}",
    ]
    return "\n".join(lines)


def render_edge_candidate_summary(result: EdgeCandidateBuildResult) -> str:
    """Return a human-readable summary for one edge-candidate build."""
    lines = [
        f"Edge candidate build complete for {result.target_date.isoformat()}",
        f"run_id={result.run_id}",
        f"model_version={result.model_version}",
        f"model_run_id={result.model_run_id}",
        f"line_snapshots={result.line_count}",
        f"scored_lines={result.scored_line_count}",
        f"actionable_candidates={result.actionable_count}",
        f"below_threshold={result.below_threshold_count}",
        f"skipped_lines={result.skipped_line_count}",
        f"line_snapshots_path={result.line_snapshots_path}",
        f"model_path={result.model_path}",
        f"ladder_probabilities_path={result.ladder_probabilities_path}",
        f"edge_candidates_path={result.edge_candidates_path}",
    ]
    return "\n".join(lines)


def render_daily_candidate_workflow_summary(
    result: DailyCandidateWorkflowResult,
) -> str:
    """Return a human-readable summary for one daily candidate workflow run."""
    approved_wager_count = (
        result.approved_wager_count
        if result.approved_wager_count is not None
        else result.actionable_candidate_count
    )
    lines = [
        f"Daily candidate workflow complete for {result.target_date.isoformat()}",
        f"run_id={result.run_id}",
        f"inference_run_id={result.inference_run_id}",
        f"edge_candidate_run_id={result.edge_candidate_run_id}",
        f"scored_candidates={result.scored_candidate_count}",
        f"actionable_candidates={result.actionable_candidate_count}",
        f"approved_wagers={approved_wager_count}",
        f"settled_paper_results={result.settled_result_count}",
        f"pending_paper_results={result.pending_result_count}",
        f"daily_candidates_path={result.daily_candidates_path}",
        f"paper_results_path={result.paper_results_path}",
    ]
    return "\n".join(lines)


def render_walk_forward_backtest_summary(result: WalkForwardBacktestResult) -> str:
    """Return a human-readable summary for one walk-forward backtest run."""
    lines = [
        (
            "Walk-forward backtest complete for "
            f"{result.start_date.isoformat()} -> {result.end_date.isoformat()}"
        ),
        f"run_id={result.run_id}",
        f"mlflow_run_id={result.mlflow_run_id}",
        f"mlflow_experiment_name={result.mlflow_experiment_name}",
        f"model_version={result.model_version}",
        f"model_run_id={result.model_run_id}",
        (
            "cutoff_minutes_before_first_pitch="
            f"{result.cutoff_minutes_before_first_pitch}"
        ),
        f"snapshot_groups={result.snapshot_group_count}",
        f"actionable_bets={result.actionable_bet_count}",
        f"below_threshold={result.below_threshold_count}",
        f"skipped={result.skipped_count}",
        f"skip_reason_counts={json.dumps(result.skip_reason_counts, sort_keys=True)}",
        f"backtest_bets_path={result.backtest_bets_path}",
        f"bet_reporting_path={result.bet_reporting_path}",
        f"backtest_runs_path={result.backtest_runs_path}",
        f"join_audit_path={result.join_audit_path}",
        f"clv_summary_path={result.clv_summary_path}",
        f"roi_summary_path={result.roi_summary_path}",
        f"edge_bucket_summary_path={result.edge_bucket_summary_path}",
        f"reproducibility_notes_path={result.reproducibility_notes_path}",
    ]
    return "\n".join(lines)


def render_backfill_historical_summary(result: BackfillResult) -> str:
    """Return a human-readable summary for one backfill-historical sweep."""
    lines = [
        (
            "Backfill historical complete for "
            f"{result.start_date.isoformat()} -> {result.end_date.isoformat()}"
        ),
        f"run_id={result.run_id}",
        f"force={str(result.force).lower()}",
        f"sources={','.join(result.sources)}",
        f"history_days={result.history_days}",
        f"date_count={result.date_count}",
        f"ingested_outcomes={result.ingested_count}",
        f"skipped_outcomes={result.skipped_count}",
        f"failed_outcomes={result.failed_count}",
        f"manifest_path={result.manifest_path}",
    ]
    if result.failed_count:
        lines.append("failed_dates:")
        for date_outcome in result.dates:
            failures = [
                source_outcome
                for source_outcome in date_outcome.sources
                if source_outcome.status == "failed"
            ]
            for source_outcome in failures:
                lines.append(
                    "- "
                    f"{date_outcome.target_date.isoformat()} "
                    f"{source_outcome.source}: "
                    f"{source_outcome.error_type}: {source_outcome.error_message}"
                )
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser."""
    parser = argparse.ArgumentParser(prog="mlb-props-stack")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser(
        "ingest-mlb-metadata",
        help="Fetch schedule, probable starters, and lineup snapshots for one date.",
    )
    ingest_parser.add_argument(
        "--date",
        dest="target_date",
        required=True,
        help="Target MLB schedule date in YYYY-MM-DD format.",
    )
    ingest_parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where raw and normalized ingest artifacts will be written.",
    )

    odds_parser = subparsers.add_parser(
        "ingest-odds-api-lines",
        help="Fetch The Odds API pitcher strikeout lines and map them to MLB games.",
    )
    odds_parser.add_argument(
        "--date",
        dest="target_date",
        required=True,
        help="Target MLB schedule date in YYYY-MM-DD format.",
    )
    odds_parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where raw and normalized ingest artifacts will be written.",
    )
    odds_parser.add_argument(
        "--api-key",
        default=None,
        help="Optional Odds API key override. Defaults to ODDS_API_KEY.",
    )
    odds_parser.add_argument(
        "--bookmakers",
        default=None,
        help=(
            "Optional comma-separated list of sportsbook keys to persist "
            "(for example 'pinnacle,circa'). Defaults to every book returned "
            "for the configured regions."
        ),
    )

    weather_parser = subparsers.add_parser(
        "ingest-weather",
        help=(
            "Fetch pregame Open-Meteo weather snapshots anchored to "
            "commence_time - 60 minutes for one slate date."
        ),
    )
    weather_parser.add_argument(
        "--date",
        dest="target_date",
        required=True,
        help="Target MLB schedule date in YYYY-MM-DD format.",
    )
    weather_parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where raw and normalized ingest artifacts will be written.",
    )

    umpire_parser = subparsers.add_parser(
        "ingest-umpire",
        help=(
            "Extract the home-plate umpire per scheduled game and join "
            "30-day rolling called-strike and K/9 features for one slate date."
        ),
    )
    umpire_parser.add_argument(
        "--date",
        dest="target_date",
        required=True,
        help="Target MLB schedule date in YYYY-MM-DD format.",
    )
    umpire_parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where raw and normalized ingest artifacts will be written.",
    )

    statcast_parser = subparsers.add_parser(
        "ingest-statcast-features",
        help="Fetch targeted Statcast history and build feature tables for one date.",
    )
    statcast_parser.add_argument(
        "--date",
        dest="target_date",
        required=True,
        help="Target MLB schedule date in YYYY-MM-DD format.",
    )
    statcast_parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where raw and normalized ingest artifacts will be written.",
    )
    statcast_parser.add_argument(
        "--history-days",
        type=int,
        default=DEFAULT_HISTORY_DAYS,
        help="Number of prior official dates to include in the Statcast history window.",
    )

    backfill_parser = subparsers.add_parser(
        "backfill-historical",
        help=(
            "Iterate ingest-mlb-metadata, ingest-weather, ingest-umpire, "
            "ingest-odds-api-lines (best-effort), and ingest-statcast-features "
            "across a date window with idempotent resume."
        ),
    )
    backfill_parser.add_argument(
        "--start-date",
        required=True,
        help="Earliest official date to include in the backfill window (YYYY-MM-DD).",
    )
    backfill_parser.add_argument(
        "--end-date",
        required=True,
        help="Latest official date to include in the backfill window (YYYY-MM-DD).",
    )
    backfill_parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where raw and normalized ingest artifacts will be written.",
    )
    backfill_parser.add_argument(
        "--sources",
        default=",".join(ALL_SOURCES),
        help=(
            "Comma-separated list of sources to backfill. "
            f"Valid values: {','.join(ALL_SOURCES)}. "
            "Defaults to every source."
        ),
    )
    backfill_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-ingest dates even when the latest normalized run already "
            "contains every required artifact. Off by default so reruns "
            "are idempotent."
        ),
    )
    backfill_parser.add_argument(
        "--history-days",
        type=int,
        default=DEFAULT_HISTORY_DAYS,
        help=(
            "Number of prior official dates to include in each per-date "
            "Statcast history window."
        ),
    )
    backfill_parser.add_argument(
        "--api-key",
        default=None,
        help="Optional Odds API key override. Defaults to ODDS_API_KEY.",
    )

    training_parser = subparsers.add_parser(
        "train-starter-strikeout-baseline",
        help="Train the first date-split starter strikeout baseline model.",
    )
    training_parser.add_argument(
        "--start-date",
        required=True,
        help="Earliest official date to include in the training window.",
    )
    training_parser.add_argument(
        "--end-date",
        required=True,
        help="Latest official date to include in the training window.",
    )
    training_parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where feature inputs and training artifacts live.",
    )
    training_parser.add_argument(
        "--feature-set",
        choices=FEATURE_SET_CHOICES,
        default=FEATURE_SET_EXPANDED,
        help=(
            "Feature schema to train. 'core' uses dense pitcher/workload fields only; "
            "'expanded' also admits optional fields that pass coverage and variance gates."
        ),
    )

    starter_dataset_parser = subparsers.add_parser(
        "build-starter-strikeout-dataset",
        help=(
            "Build the standalone starter-game strikeout training dataset and "
            "coverage reports from feature runs or direct Statcast pitch logs."
        ),
    )
    starter_dataset_parser.add_argument(
        "--start-date",
        required=True,
        help="Earliest official date to include in the dataset window.",
    )
    starter_dataset_parser.add_argument(
        "--end-date",
        required=True,
        help="Latest official date to include in the dataset window.",
    )
    starter_dataset_parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where feature inputs and dataset artifacts live.",
    )
    starter_dataset_parser.add_argument(
        "--chunk-days",
        type=int,
        default=DEFAULT_DIRECT_CHUNK_DAYS,
        help=(
            "Number of calendar dates per direct Statcast pitch-log pull when "
            "normalized feature runs are absent."
        ),
    )
    starter_dataset_parser.add_argument(
        "--max-fetch-workers",
        type=int,
        default=DEFAULT_MAX_FETCH_WORKERS,
        help="Maximum concurrent direct Statcast CSV fetches.",
    )

    pitcher_skill_parser = subparsers.add_parser(
        "build-pitcher-skill-features",
        help=(
            "Build timestamp-valid pitcher skill and pitch-arsenal features "
            "from a starter-game dataset artifact."
        ),
    )
    pitcher_skill_parser.add_argument(
        "--start-date",
        required=True,
        help="Earliest official date to include in the feature build.",
    )
    pitcher_skill_parser.add_argument(
        "--end-date",
        required=True,
        help="Latest official date to include in the feature build.",
    )
    pitcher_skill_parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where starter-game dataset and feature artifacts live.",
    )
    pitcher_skill_parser.add_argument(
        "--dataset-run-dir",
        default=None,
        help=(
            "Optional exact starter-game dataset run directory. Defaults to the "
            "latest run under the requested start/end window."
        ),
    )

    lineup_matchup_parser = subparsers.add_parser(
        "build-lineup-matchup-features",
        help=(
            "Build timestamp-valid batter-by-batter and aggregate opponent-lineup "
            "matchup features from a starter-game dataset artifact."
        ),
    )
    lineup_matchup_parser.add_argument(
        "--start-date",
        required=True,
        help="Earliest official date to include in the feature build.",
    )
    lineup_matchup_parser.add_argument(
        "--end-date",
        required=True,
        help="Latest official date to include in the feature build.",
    )
    lineup_matchup_parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where starter-game dataset and feature artifacts live.",
    )
    lineup_matchup_parser.add_argument(
        "--dataset-run-dir",
        default=None,
        help=(
            "Optional exact starter-game dataset run directory. Defaults to the "
            "latest run under the requested start/end window."
        ),
    )

    workload_leash_parser = subparsers.add_parser(
        "build-workload-leash-features",
        help=(
            "Build timestamp-valid expected workload, leash, rest, and role-context "
            "features from a starter-game dataset artifact."
        ),
    )
    workload_leash_parser.add_argument(
        "--start-date",
        required=True,
        help="Earliest official date to include in the feature build.",
    )
    workload_leash_parser.add_argument(
        "--end-date",
        required=True,
        help="Latest official date to include in the feature build.",
    )
    workload_leash_parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where starter-game dataset and feature artifacts live.",
    )
    workload_leash_parser.add_argument(
        "--dataset-run-dir",
        default=None,
        help=(
            "Optional exact starter-game dataset run directory. Defaults to the "
            "latest run under the requested start/end window."
        ),
    )

    comparison_parser = subparsers.add_parser(
        "compare-starter-strikeout-baselines",
        help=(
            "Train core and expanded starter strikeout baselines, backtest both, "
            "and write a same-window comparison report."
        ),
    )
    comparison_parser.add_argument(
        "--start-date",
        required=True,
        help="Earliest official date to include in the comparison window.",
    )
    comparison_parser.add_argument(
        "--end-date",
        required=True,
        help="Latest official date to include in the comparison window.",
    )
    comparison_parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where feature, odds, model, and comparison artifacts live.",
    )
    comparison_parser.add_argument(
        "--cutoff-minutes-before-first-pitch",
        type=int,
        default=StackConfig().backtest_cutoff_minutes_before_first_pitch,
        help="Latest allowed odds snapshot timestamp before scheduled first pitch.",
    )

    candidate_models_parser = subparsers.add_parser(
        "train-candidate-strikeout-models",
        help=(
            "Train comparable candidate starter strikeout model families and "
            "write distribution-output reports without betting decisions."
        ),
    )
    candidate_models_parser.add_argument(
        "--start-date",
        required=True,
        help="Earliest official date to include in the candidate training window.",
    )
    candidate_models_parser.add_argument(
        "--end-date",
        required=True,
        help="Latest official date to include in the candidate training window.",
    )
    candidate_models_parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where starter, feature, and model artifacts live.",
    )
    candidate_models_parser.add_argument(
        "--dataset-run-dir",
        default=None,
        help=(
            "Optional exact starter-game dataset run directory. Defaults to the "
            "latest matching run, then latest available run."
        ),
    )
    candidate_models_parser.add_argument(
        "--pitcher-skill-run-dir",
        default=None,
        help="Optional exact pitcher-skill feature run directory.",
    )
    candidate_models_parser.add_argument(
        "--lineup-matchup-run-dir",
        default=None,
        help="Optional exact lineup-matchup feature run directory.",
    )
    candidate_models_parser.add_argument(
        "--workload-leash-run-dir",
        default=None,
        help="Optional exact workload/leash feature run directory.",
    )

    daily_candidates_parser = subparsers.add_parser(
        "build-daily-candidates",
        help=(
            "Score one target-date slate from the latest honest model run and "
            "refresh paper-tracking results."
        ),
    )
    daily_candidates_parser.add_argument(
        "--date",
        dest="target_date",
        default=None,
        help=(
            "Official date to evaluate in YYYY-MM-DD format. "
            "Defaults to today in the configured stack timezone."
        ),
    )
    daily_candidates_parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where normalized features, odds, and paper-tracking artifacts live.",
    )
    daily_candidates_parser.add_argument(
        "--source-model-run-dir",
        default=None,
        help=(
            "Optional explicit starter_strikeout_baseline run directory to use for "
            "pregame inference. Defaults to the latest run ending before the target date."
        ),
    )

    wager_card_parser = subparsers.add_parser(
        "build-wager-card",
        help="Print and persist the approved wager card for a saved daily candidate sheet.",
    )
    wager_card_parser.add_argument(
        "--date",
        dest="target_date",
        default=None,
        help=(
            "Official date to print in YYYY-MM-DD format. "
            "Defaults to the latest saved daily candidate date."
        ),
    )
    wager_card_parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where normalized daily candidate artifacts live.",
    )
    wager_card_parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Include blocked candidates in a separate diagnostic section and artifact rows.",
    )

    edge_parser = subparsers.add_parser(
        "build-edge-candidates",
        help="Join latest odds snapshots to saved model ladders and score edges.",
    )
    edge_parser.add_argument(
        "--date",
        dest="target_date",
        required=True,
        help="Official date to evaluate in YYYY-MM-DD format.",
    )
    edge_parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where normalized odds, model, and edge artifacts live.",
    )
    edge_parser.add_argument(
        "--model-run-dir",
        default=None,
        help=(
            "Optional explicit starter_strikeout_baseline run directory. "
            "Defaults to the latest run containing the requested date."
        ),
    )

    backtest_parser = subparsers.add_parser(
        "build-walk-forward-backtest",
        help="Select cutoff-safe odds snapshots and score a walk-forward backtest.",
    )
    backtest_parser.add_argument(
        "--start-date",
        required=True,
        help="Earliest official date to include in the backtest window.",
    )
    backtest_parser.add_argument(
        "--end-date",
        required=True,
        help="Latest official date to include in the backtest window.",
    )
    backtest_parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where normalized odds, model, and backtest artifacts live.",
    )
    backtest_parser.add_argument(
        "--model-run-dir",
        default=None,
        help=(
            "Optional explicit starter_strikeout_baseline run directory. "
            "Defaults to the latest run covering the full requested date window."
        ),
    )
    backtest_parser.add_argument(
        "--cutoff-minutes-before-first-pitch",
        type=int,
        default=StackConfig().backtest_cutoff_minutes_before_first_pitch,
        help="Latest allowed odds snapshot timestamp before scheduled first pitch.",
    )

    data_alignment_parser = subparsers.add_parser(
        "check-data-alignment",
        help=(
            "Report per-date row counts and coverage ratios for every ingest, "
            "feature, and modeling artifact across a date range."
        ),
    )
    data_alignment_parser.add_argument(
        "--start-date",
        required=True,
        help="Earliest official date to evaluate in YYYY-MM-DD format.",
    )
    data_alignment_parser.add_argument(
        "--end-date",
        required=True,
        help="Latest official date to evaluate in YYYY-MM-DD format.",
    )
    data_alignment_parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where normalized ingest, feature, and model artifacts live.",
    )
    data_alignment_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_COVERAGE_THRESHOLD,
        help=(
            "Minimum feature, outcome, and odds coverage ratio "
            "(0.0-1.0) required for a date to pass."
        ),
    )

    stage_gates_parser = subparsers.add_parser(
        "evaluate-stage-gates",
        help=(
            "Evaluate the latest training, backtest, CLV, ROI, and paper "
            "artifacts against live-use readiness gates."
        ),
    )
    stage_gates_parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where normalized readiness artifacts live.",
    )
    stage_gates_parser.add_argument(
        "--training-summary-path",
        default=None,
        help=(
            "Optional explicit evaluation_summary.json path. Defaults to the "
            "training run referenced by the latest backtest model_run_id."
        ),
    )
    stage_gates_parser.add_argument(
        "--backtest-run-dir",
        default=None,
        help=(
            "Optional explicit walk_forward_backtest run directory. Defaults "
            "to the latest run with backtest_runs.jsonl."
        ),
    )
    stage_gates_parser.add_argument(
        "--paper-results-path",
        default=None,
        help=(
            "Optional explicit paper_results.jsonl path. Defaults to the "
            "latest saved cumulative paper-results run."
        ),
    )
    stage_gates_parser.add_argument(
        "--fail-on-research-only",
        action="store_true",
        help=(
            "Exit nonzero only when the evaluated status is research_only. "
            "Without this flag the command is informational."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest-mlb-metadata":
        result = ingest_mlb_metadata_for_date(
            target_date=date.fromisoformat(args.target_date),
            output_dir=args.output_dir,
        )
        print(render_ingest_summary(result))
        return 0

    if args.command == "ingest-odds-api-lines":
        bookmakers = _parse_bookmaker_argument(args.bookmakers)
        result = ingest_odds_api_pitcher_lines_for_date(
            target_date=date.fromisoformat(args.target_date),
            output_dir=args.output_dir,
            api_key=args.api_key,
            bookmakers=bookmakers,
        )
        print(render_odds_api_ingest_summary(result))
        return 0

    if args.command == "ingest-weather":
        weather_result = ingest_weather_for_date(
            target_date=date.fromisoformat(args.target_date),
            output_dir=args.output_dir,
        )
        print(render_weather_ingest_summary(weather_result))
        return 0

    if args.command == "ingest-umpire":
        umpire_result = ingest_umpire_for_date(
            target_date=date.fromisoformat(args.target_date),
            output_dir=args.output_dir,
        )
        print(render_umpire_ingest_summary(umpire_result))
        return 0

    if args.command == "ingest-statcast-features":
        result = ingest_statcast_features_for_date(
            target_date=date.fromisoformat(args.target_date),
            output_dir=args.output_dir,
            history_days=args.history_days,
        )
        print(render_statcast_feature_ingest_summary(result))
        return 0

    if args.command == "backfill-historical":
        sources = tuple(
            source.strip()
            for source in args.sources.split(",")
            if source.strip()
        )
        backfill_result = backfill_historical(
            start_date=date.fromisoformat(args.start_date),
            end_date=date.fromisoformat(args.end_date),
            output_dir=args.output_dir,
            sources=sources,
            force=args.force,
            history_days=args.history_days,
            odds_api_key=args.api_key,
        )
        print(render_backfill_historical_summary(backfill_result))
        return 0 if backfill_result.all_succeeded else 1

    if args.command == "train-starter-strikeout-baseline":
        result = train_starter_strikeout_baseline(
            start_date=date.fromisoformat(args.start_date),
            end_date=date.fromisoformat(args.end_date),
            output_dir=args.output_dir,
            feature_set=args.feature_set,
        )
        print(render_starter_strikeout_training_summary(result))
        return 0

    if args.command == "build-starter-strikeout-dataset":
        result = build_starter_strikeout_dataset(
            start_date=date.fromisoformat(args.start_date),
            end_date=date.fromisoformat(args.end_date),
            output_dir=args.output_dir,
            chunk_days=args.chunk_days,
            max_fetch_workers=args.max_fetch_workers,
        )
        print(render_starter_game_dataset_summary(result))
        return 0

    if args.command == "build-pitcher-skill-features":
        result = build_pitcher_skill_features(
            start_date=date.fromisoformat(args.start_date),
            end_date=date.fromisoformat(args.end_date),
            output_dir=args.output_dir,
            dataset_run_dir=args.dataset_run_dir,
        )
        print(render_pitcher_skill_feature_summary(result))
        return 0

    if args.command == "build-lineup-matchup-features":
        result = build_lineup_matchup_features(
            start_date=date.fromisoformat(args.start_date),
            end_date=date.fromisoformat(args.end_date),
            output_dir=args.output_dir,
            dataset_run_dir=args.dataset_run_dir,
        )
        print(render_lineup_matchup_feature_summary(result))
        return 0

    if args.command == "build-workload-leash-features":
        result = build_workload_leash_features(
            start_date=date.fromisoformat(args.start_date),
            end_date=date.fromisoformat(args.end_date),
            output_dir=args.output_dir,
            dataset_run_dir=args.dataset_run_dir,
        )
        print(render_workload_leash_feature_summary(result))
        return 0

    if args.command == "compare-starter-strikeout-baselines":
        result = compare_starter_strikeout_baselines(
            start_date=date.fromisoformat(args.start_date),
            end_date=date.fromisoformat(args.end_date),
            output_dir=args.output_dir,
            cutoff_minutes_before_first_pitch=args.cutoff_minutes_before_first_pitch,
        )
        print(render_model_comparison_summary(result))
        return 0

    if args.command == "train-candidate-strikeout-models":
        result = train_candidate_strikeout_models(
            start_date=date.fromisoformat(args.start_date),
            end_date=date.fromisoformat(args.end_date),
            output_dir=args.output_dir,
            dataset_run_dir=args.dataset_run_dir,
            pitcher_skill_run_dir=args.pitcher_skill_run_dir,
            lineup_matchup_run_dir=args.lineup_matchup_run_dir,
            workload_leash_run_dir=args.workload_leash_run_dir,
        )
        print(render_candidate_strikeout_model_summary(result))
        return 0

    if args.command == "build-daily-candidates":
        result = build_daily_candidate_workflow(
            target_date=(
                date.fromisoformat(args.target_date)
                if args.target_date is not None
                else None
            ),
            output_dir=args.output_dir,
            source_model_run_dir=args.source_model_run_dir,
        )
        print(render_daily_candidate_workflow_summary(result))
        return 0

    if args.command == "build-wager-card":
        result = build_wager_card(
            target_date=(
                date.fromisoformat(args.target_date)
                if args.target_date is not None
                else None
            ),
            output_dir=args.output_dir,
            include_rejected=args.include_rejected,
        )
        print(render_wager_card_summary(result))
        return 0

    if args.command == "build-edge-candidates":
        result = build_edge_candidates_for_date(
            target_date=date.fromisoformat(args.target_date),
            output_dir=args.output_dir,
            model_run_dir=args.model_run_dir,
        )
        print(render_edge_candidate_summary(result))
        return 0

    if args.command == "build-walk-forward-backtest":
        result = build_walk_forward_backtest(
            start_date=date.fromisoformat(args.start_date),
            end_date=date.fromisoformat(args.end_date),
            output_dir=args.output_dir,
            model_run_dir=args.model_run_dir,
            cutoff_minutes_before_first_pitch=args.cutoff_minutes_before_first_pitch,
        )
        print(render_walk_forward_backtest_summary(result))
        return 0

    if args.command == "check-data-alignment":
        report = check_data_alignment(
            start_date=date.fromisoformat(args.start_date),
            end_date=date.fromisoformat(args.end_date),
            output_dir=args.output_dir,
            threshold=args.threshold,
        )
        print(render_data_alignment_summary(report))
        return 0 if report.passed else 1

    if args.command == "evaluate-stage-gates":
        result = evaluate_stage_gates(
            output_dir=args.output_dir,
            training_summary_path=args.training_summary_path,
            backtest_run_dir=args.backtest_run_dir,
            paper_results_path=args.paper_results_path,
        )
        print(render_stage_gate_summary(result))
        if args.fail_on_research_only and result.status == "research_only":
            return 1
        return 0

    print(render_runtime_summary())
    return 0


if __name__ == "__main__":
    main()
