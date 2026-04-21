"""Source ingestion helpers for the MLB props stack."""

from .mlb_stats_api import (
    GameRecord,
    LineupEntry,
    LineupSnapshot,
    MLBMetadataIngestResult,
    MLBStatsAPIClient,
    ProbableStarterRecord,
    build_odds_matchup_key,
    ingest_mlb_metadata_for_date,
)
from .odds_api import (
    OddsAPIClient,
    OddsAPIIngestResult,
    OddsEventGameMappingRecord,
    PropLineSnapshotRecord,
    ingest_odds_api_pitcher_lines_for_date,
)

__all__ = [
    "GameRecord",
    "LineupEntry",
    "LineupSnapshot",
    "MLBMetadataIngestResult",
    "MLBStatsAPIClient",
    "OddsAPIClient",
    "OddsAPIIngestResult",
    "OddsEventGameMappingRecord",
    "ProbableStarterRecord",
    "PropLineSnapshotRecord",
    "build_odds_matchup_key",
    "ingest_mlb_metadata_for_date",
    "ingest_odds_api_pitcher_lines_for_date",
]
