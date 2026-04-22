"""CLI entrypoint for the scaffold and ingestion slices."""

from __future__ import annotations

import argparse
from datetime import date

from .backtest import BACKTEST_CHECKLIST
from .config import StackConfig
from .ingest import (
    DEFAULT_HISTORY_DAYS,
    MLBMetadataIngestResult,
    OddsAPIIngestResult,
    StatcastFeatureIngestResult,
    ingest_mlb_metadata_for_date,
    ingest_odds_api_pitcher_lines_for_date,
    ingest_statcast_features_for_date,
)
from .modeling import (
    StarterStrikeoutBaselineTrainingResult,
    train_starter_strikeout_baseline,
)
from .tracking import TrackingConfig


def render_runtime_summary() -> str:
    """Return a human-readable snapshot of the local runtime baseline."""
    config = StackConfig()
    tracking = TrackingConfig()
    lines = [
        "MLB Props Stack",
        f"market={config.market}",
        f"min_edge_pct={config.min_edge_pct:.2%}",
        f"kelly_fraction={config.kelly_fraction:.2f}",
        f"tracking_uri={tracking.tracking_uri}",
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
        f"prop_line_snapshots={result.prop_line_count}",
        f"skipped_prop_groups={result.skipped_prop_count}",
        f"mlb_games_path={result.mlb_games_path}",
        f"mlb_probable_starters_path={result.mlb_probable_starters_path}",
        f"events_raw_path={result.events_raw_path}",
        f"event_mappings_path={result.event_mappings_path}",
        f"prop_line_snapshots_path={result.prop_line_snapshots_path}",
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
    lines = [
        (
            "Starter strikeout baseline training complete for "
            f"{result.start_date.isoformat()} -> {result.end_date.isoformat()}"
        ),
        f"run_id={result.run_id}",
        f"training_rows={result.row_count}",
        f"starter_outcomes={result.outcome_count}",
        f"dispersion_alpha={result.dispersion_alpha:.6f}",
        f"dataset_path={result.dataset_path}",
        f"outcomes_path={result.outcomes_path}",
        f"date_splits_path={result.date_splits_path}",
        f"model_path={result.model_path}",
        f"evaluation_path={result.evaluation_path}",
        f"ladder_probabilities_path={result.ladder_probabilities_path}",
        f"probability_calibrator_path={result.probability_calibrator_path}",
        f"raw_vs_calibrated_path={result.raw_vs_calibrated_path}",
        f"calibration_summary_path={result.calibration_summary_path}",
    ]
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
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest-mlb-metadata":
        result = ingest_mlb_metadata_for_date(
            target_date=date.fromisoformat(args.target_date),
            output_dir=args.output_dir,
        )
        print(render_ingest_summary(result))
        return

    if args.command == "ingest-odds-api-lines":
        result = ingest_odds_api_pitcher_lines_for_date(
            target_date=date.fromisoformat(args.target_date),
            output_dir=args.output_dir,
            api_key=args.api_key,
        )
        print(render_odds_api_ingest_summary(result))
        return

    if args.command == "ingest-statcast-features":
        result = ingest_statcast_features_for_date(
            target_date=date.fromisoformat(args.target_date),
            output_dir=args.output_dir,
            history_days=args.history_days,
        )
        print(render_statcast_feature_ingest_summary(result))
        return

    if args.command == "train-starter-strikeout-baseline":
        result = train_starter_strikeout_baseline(
            start_date=date.fromisoformat(args.start_date),
            end_date=date.fromisoformat(args.end_date),
            output_dir=args.output_dir,
        )
        print(render_starter_strikeout_training_summary(result))
        return

    print(render_runtime_summary())


if __name__ == "__main__":
    main()
