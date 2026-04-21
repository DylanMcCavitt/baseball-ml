"""CLI entrypoint for the scaffold and ingestion slices."""

from __future__ import annotations

import argparse
from datetime import date

from .backtest import BACKTEST_CHECKLIST
from .config import StackConfig
from .ingest import (
    MLBMetadataIngestResult,
    OddsAPIIngestResult,
    ingest_mlb_metadata_for_date,
    ingest_odds_api_pitcher_lines_for_date,
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

    print(render_runtime_summary())


if __name__ == "__main__":
    main()
