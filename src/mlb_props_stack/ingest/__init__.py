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
from .statcast_features import (
    DEFAULT_HISTORY_DAYS,
    GameContextFeatureRow,
    LineupDailyFeatureRow,
    PitcherDailyFeatureRow,
    StatcastFeatureIngestResult,
    StatcastPitchRecord,
    StatcastPullRecord,
    StatcastSearchClient,
    build_statcast_search_csv_url,
    ingest_statcast_features_for_date,
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
    "DEFAULT_HISTORY_DAYS",
    "GameContextFeatureRow",
    "LineupDailyFeatureRow",
    "PitcherDailyFeatureRow",
    "StatcastFeatureIngestResult",
    "StatcastPitchRecord",
    "StatcastPullRecord",
    "StatcastSearchClient",
    "build_odds_matchup_key",
    "build_statcast_search_csv_url",
    "ingest_mlb_metadata_for_date",
    "ingest_odds_api_pitcher_lines_for_date",
    "ingest_statcast_features_for_date",
]
