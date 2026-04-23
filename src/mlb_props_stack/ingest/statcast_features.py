"""Statcast feature ingest orchestrator.

Loads inputs (persisted MLB metadata, weather snapshots, umpire snapshots, park
factors), drives Statcast CSV pulls via :class:`StatcastSearchClient`, builds
the per-starter feature rows by delegating to ``pitcher_features``,
``lineup_aggregation`` and ``game_context``, and writes the normalized
artifacts to disk.

Internal helpers and dataclasses used by downstream callers, tests, or
``ingest/__init__.py`` are re-exported from this module so that moving
implementation details into focused submodules is not a breaking change.

The ``urlopen`` symbol remains imported here so tests can continue to patch
``mlb_props_stack.ingest.statcast_features.urlopen`` to stub network access.
``statcast_ingest.StatcastSearchClient.fetch_csv`` deliberately dispatches
through this orchestrator's ``urlopen`` attribute at call time so that
monkeypatch still takes effect.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, date, datetime, timedelta
import json
import os
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen  # noqa: F401  # re-exposed for test monkeypatching
from zoneinfo import ZoneInfo

from ..config import StackConfig
from .game_context import GameContextFeatureRow, _build_game_context_feature_row
from .lineup_aggregation import (
    LineupDailyFeatureRow,
    _BatterMetricBundle,
    _batter_k_rate_vs_p_throws,
    _batter_metric_bundle,
    _batting_order_weight,
    _build_lineup_daily_feature_row,
    _latest_prior_team_lineup_player_ids,
    _weighted_mean,
)
from .mlb_stats_api import (
    GameRecord,
    LineupEntry,
    LineupSnapshot,
    ProbableStarterRecord,
    format_utc_timestamp,
    parse_api_datetime,
    utc_now,
)
from .park_factors import (
    PARK_FACTOR_STATUS_MISSING_SOURCE,
    PARK_FACTOR_STATUS_OK,
    ParkKFactorRecord,
    load_park_k_factors,
    lookup_park_k_factor,
)
from .pitcher_features import (
    PitcherDailyFeatureRow,
    _build_pitcher_daily_feature_row,
    _expected_leash,
    _pitcher_hand,
    _pitcher_hand_split_rates,
)
from .statcast_ingest import (
    CALLED_STRIKE_DESCRIPTIONS,
    CONTACT_DESCRIPTIONS,
    DEFAULT_BACKOFF_MULTIPLIER,
    DEFAULT_INITIAL_BACKOFF_SECONDS,
    DEFAULT_MAX_BACKOFF_SECONDS,
    DEFAULT_MAX_FETCH_ATTEMPTS,
    DEFAULT_MAX_FETCH_WORKERS,
    STATCAST_REQUEST_HEADERS,
    STATCAST_SEARCH_CSV_ENDPOINT,
    STRIKEOUT_EVENTS,
    SWING_DESCRIPTIONS,
    WHIFF_DESCRIPTIONS,
    StatcastPitchRecord,
    StatcastPullRecord,
    StatcastSearchClient,
    _batter_rows,
    _batting_team_abbreviation,
    _coerce_optional_float,
    _coerce_optional_int,
    _count_plate_appearances,
    _fetch_csv_texts_concurrently,
    _history_cutoff,
    _is_out_of_zone,
    _is_retriable_http_error,
    _last_game_date,
    _mean,
    _opponent_team,
    _opponent_team_side,
    _optional_text,
    _pitch_record_id,
    _pitch_rows_for_player,
    _pitch_type_usage,
    _plate_appearance_key,
    _round_optional,
    _rows_grouped_by_start,
    _rows_in_recent_window,
    _safe_rate,
    _select_pregame_lineup_snapshot,
    _sorted_rows,
    build_statcast_search_csv_url,
    normalize_statcast_csv_text,
)
from .umpire import (
    UMPIRE_STATUS_MISSING_SOURCE,
    UmpireSnapshotRecord,
    load_latest_umpire_snapshots_for_date,
)
from .weather import (
    WEATHER_STATUS_MISSING_SOURCE,
    WeatherSnapshotRecord,
    load_latest_weather_snapshots_for_date,
)

DEFAULT_HISTORY_DAYS = 30


@dataclass(frozen=True)
class StatcastFeatureIngestResult:
    """Filesystem output summary for one Statcast feature build."""

    target_date: date
    history_start_date: date
    history_end_date: date
    run_id: str
    mlb_games_path: Path
    mlb_probable_starters_path: Path
    mlb_lineup_snapshots_path: Path
    pull_manifest_path: Path
    pitch_level_base_path: Path
    pitcher_daily_features_path: Path
    lineup_daily_features_path: Path
    game_context_features_path: Path
    raw_pull_count: int
    pitch_level_record_count: int
    pitcher_feature_count: int
    lineup_feature_count: int
    game_context_feature_count: int


@dataclass(frozen=True)
class _LoadedMLBMetadata:
    games_path: Path
    probable_starters_path: Path
    lineup_snapshots_path: Path
    games: tuple[GameRecord, ...]
    probable_starters: tuple[ProbableStarterRecord, ...]
    lineup_snapshots: tuple[LineupSnapshot, ...]


def _path_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, datetime):
        return format_utc_timestamp(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(_json_ready(payload), indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(_json_ready(record), sort_keys=True))
                handle.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _latest_complete_run_dir(root: Path) -> Path:
    run_dirs = sorted(
        (path for path in root.glob("run=*") if path.is_dir()),
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(
            "No normalized MLB metadata runs were found. "
            "Run `uv run python -m mlb_props_stack ingest-mlb-metadata --date ...` first."
        )
    for run_dir in run_dirs:
        games_path = run_dir / "games.jsonl"
        probable_starters_path = run_dir / "probable_starters.jsonl"
        if games_path.exists() and probable_starters_path.exists():
            return run_dir
    raise FileNotFoundError(f"Latest MLB metadata runs under {root} were incomplete.")


def _latest_pregame_valid_run_dir(root: Path) -> Path:
    run_dirs = sorted(
        (path for path in root.glob("run=*") if path.is_dir()),
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(
            "No normalized MLB metadata runs were found. "
            "Run `uv run python -m mlb_props_stack ingest-mlb-metadata --date ...` first."
        )

    for run_dir in run_dirs:
        games_path = run_dir / "games.jsonl"
        probable_starters_path = run_dir / "probable_starters.jsonl"
        if not games_path.exists() or not probable_starters_path.exists():
            continue

        games_rows = _load_jsonl_rows(games_path)
        probable_starters_rows = _load_jsonl_rows(probable_starters_path)
        if _run_is_pregame_valid(
            games_rows=games_rows,
            probable_starters_rows=probable_starters_rows,
        ):
            return run_dir

    raise FileNotFoundError(
        "No pregame-valid normalized MLB metadata runs were found. "
        "Run `uv run python -m mlb_props_stack ingest-mlb-metadata --date ...` "
        "before first pitch for the target slate."
    )


def _run_is_pregame_valid(
    *,
    games_rows: list[dict[str, Any]],
    probable_starters_rows: list[dict[str, Any]],
) -> bool:
    try:
        commence_times_by_game_pk: dict[int, datetime] = {}
        for row in games_rows:
            game_pk = int(row["game_pk"])
            commence_time = parse_api_datetime(row["commence_time"])
            captured_at = parse_api_datetime(row["captured_at"])
            if captured_at > commence_time:
                return False
            commence_times_by_game_pk[game_pk] = commence_time

        for row in probable_starters_rows:
            game_pk = int(row["game_pk"])
            commence_time = commence_times_by_game_pk.get(game_pk)
            if commence_time is None:
                return False
            captured_at = parse_api_datetime(row["captured_at"])
            if captured_at > commence_time:
                return False
    except (KeyError, TypeError, ValueError):
        return False

    return True


def _load_latest_mlb_metadata_for_date(
    *,
    target_date: date,
    output_dir: Path | str,
    reference_time: datetime,
) -> _LoadedMLBMetadata:
    normalized_root = (
        Path(output_dir)
        / "normalized"
        / "mlb_stats_api"
        / f"date={target_date.isoformat()}"
    )
    config = StackConfig()
    reference_date = reference_time.astimezone(ZoneInfo(config.timezone)).date()
    try:
        latest_run_dir = _latest_pregame_valid_run_dir(normalized_root)
    except FileNotFoundError:
        if target_date >= reference_date:
            raise
        latest_run_dir = _latest_complete_run_dir(normalized_root)
    games_path = latest_run_dir / "games.jsonl"
    probable_starters_path = latest_run_dir / "probable_starters.jsonl"
    lineup_snapshots_path = latest_run_dir / "lineup_snapshots.jsonl"
    if not games_path.exists() or not probable_starters_path.exists() or not lineup_snapshots_path.exists():
        raise FileNotFoundError(
            f"Expected MLB metadata artifacts in {latest_run_dir}, but they were incomplete."
        )

    games = tuple(
        GameRecord(
            game_pk=row["game_pk"],
            official_date=row["official_date"],
            commence_time=parse_api_datetime(row["commence_time"]),
            captured_at=parse_api_datetime(row["captured_at"]),
            status=row["status"],
            status_code=row["status_code"],
            venue_id=row["venue_id"],
            venue_name=row["venue_name"],
            home_team_id=row["home_team_id"],
            home_team_abbreviation=row["home_team_abbreviation"],
            home_team_name=row["home_team_name"],
            away_team_id=row["away_team_id"],
            away_team_abbreviation=row["away_team_abbreviation"],
            away_team_name=row["away_team_name"],
            game_number=row["game_number"],
            double_header=row["double_header"],
            day_night=row["day_night"],
            odds_matchup_key=row["odds_matchup_key"],
        )
        for row in _load_jsonl_rows(games_path)
    )
    probable_starters = tuple(
        ProbableStarterRecord(
            game_pk=row["game_pk"],
            official_date=row["official_date"],
            captured_at=parse_api_datetime(row["captured_at"]),
            team_side=row["team_side"],
            team_id=row["team_id"],
            team_abbreviation=row["team_abbreviation"],
            team_name=row["team_name"],
            pitcher_id=row["pitcher_id"],
            pitcher_name=row["pitcher_name"],
            pitcher_note=row["pitcher_note"],
            odds_matchup_key=row["odds_matchup_key"],
        )
        for row in _load_jsonl_rows(probable_starters_path)
    )
    lineup_snapshots = tuple(
        LineupSnapshot(
            lineup_snapshot_id=row["lineup_snapshot_id"],
            game_pk=row["game_pk"],
            official_date=row["official_date"],
            captured_at=parse_api_datetime(row["captured_at"]),
            team_side=row["team_side"],
            team_id=row["team_id"],
            team_abbreviation=row["team_abbreviation"],
            team_name=row["team_name"],
            game_state=row["game_state"],
            game_status_code=row["game_status_code"],
            is_confirmed=row["is_confirmed"],
            batting_order_player_ids=tuple(row["batting_order_player_ids"]),
            batter_player_ids=tuple(row["batter_player_ids"]),
            lineup_entries=tuple(
                LineupEntry(
                    lineup_position=entry["lineup_position"],
                    player_id=entry["player_id"],
                    player_name=entry["player_name"],
                    batting_order_code=entry["batting_order_code"],
                    position_abbreviation=entry["position_abbreviation"],
                )
                for entry in row["lineup_entries"]
            ),
            odds_matchup_key=row["odds_matchup_key"],
        )
        for row in _load_jsonl_rows(lineup_snapshots_path)
    )
    return _LoadedMLBMetadata(
        games_path=games_path,
        probable_starters_path=probable_starters_path,
        lineup_snapshots_path=lineup_snapshots_path,
        games=games,
        probable_starters=probable_starters,
        lineup_snapshots=lineup_snapshots,
    )


def ingest_statcast_features_for_date(
    *,
    target_date: date,
    output_dir: Path | str = "data",
    history_days: int = DEFAULT_HISTORY_DAYS,
    client: StatcastSearchClient | None = None,
    now: Callable[[], datetime] = utc_now,
    max_fetch_workers: int = DEFAULT_MAX_FETCH_WORKERS,
) -> StatcastFeatureIngestResult:
    """Fetch Statcast pulls and build normalized feature tables for one slate date."""
    if history_days < 1:
        raise ValueError("history_days must be at least 1")
    if max_fetch_workers < 1:
        raise ValueError("max_fetch_workers must be at least 1")
    if client is None:
        client = StatcastSearchClient()

    run_started_at = now().astimezone(UTC)
    history_end_date = target_date - timedelta(days=1)
    history_start_date = target_date - timedelta(days=history_days)
    mlb_metadata = _load_latest_mlb_metadata_for_date(
        target_date=target_date,
        output_dir=output_dir,
        reference_time=run_started_at,
    )
    games_by_pk = {game.game_pk: game for game in mlb_metadata.games}

    selected_lineups: dict[tuple[int, str], LineupSnapshot | None] = {}
    batter_ids: set[int] = set()
    for starter in mlb_metadata.probable_starters:
        game = games_by_pk.get(starter.game_pk)
        if game is None:
            continue
        opponent_side = _opponent_team_side(starter.team_side)
        selected_lineup = _select_pregame_lineup_snapshot(
            game=game,
            team_side=opponent_side,
            lineup_snapshots=mlb_metadata.lineup_snapshots,
        )
        selected_lineups[(starter.game_pk, opponent_side)] = selected_lineup
        if selected_lineup is not None:
            batter_ids.update(selected_lineup.batting_order_player_ids)

    pitch_records_by_id: dict[str, StatcastPitchRecord] = {}
    pull_records: list[StatcastPullRecord] = []
    output_root = Path(output_dir)
    run_id = _path_timestamp(run_started_at)

    pull_requests = [
        ("pitcher", pitcher_id)
        for pitcher_id in sorted(
            {
                starter.pitcher_id
                for starter in mlb_metadata.probable_starters
                if starter.pitcher_id is not None
            }
        )
    ]
    pull_requests.extend(("batter", batter_id) for batter_id in sorted(batter_ids))

    # Pre-compute one spec per pull serially so the `now()` test seam is
    # consumed in deterministic order even when fetches run in parallel.
    pull_specs: list[tuple[str, int, datetime, str]] = []
    for player_type, player_id in pull_requests:
        captured_at = now().astimezone(UTC)
        source_url = build_statcast_search_csv_url(
            player_type=player_type,
            player_id=player_id,
            start_date=history_start_date,
            end_date=history_end_date,
        )
        pull_specs.append((player_type, player_id, captured_at, source_url))

    csv_texts = _fetch_csv_texts_concurrently(
        client=client,
        source_urls=[spec[3] for spec in pull_specs],
        max_workers=max_fetch_workers,
    )

    for (player_type, player_id, captured_at, source_url), csv_text in zip(
        pull_specs, csv_texts, strict=True
    ):
        raw_path = (
            output_root
            / "raw"
            / "statcast_search"
            / f"date={target_date.isoformat()}"
            / f"player_type={player_type}"
            / f"player_id={player_id}"
            / f"captured_at={_path_timestamp(captured_at)}.csv"
        )
        _write_text(raw_path, csv_text)
        pull_id = f"statcast-pull:{player_type}:{player_id}:{_path_timestamp(captured_at)}"
        normalized_rows = normalize_statcast_csv_text(csv_text, pull_id=pull_id)
        pull_records.append(
            StatcastPullRecord(
                pull_id=pull_id,
                captured_at=captured_at,
                player_type=player_type,
                player_id=player_id,
                history_start_date=history_start_date,
                history_end_date=history_end_date,
                source_url=source_url,
                raw_path=raw_path,
                row_count=len(normalized_rows),
            )
        )
        for row in normalized_rows:
            pitch_records_by_id.setdefault(row.pitch_record_id, row)

    all_pitch_records = _sorted_rows(list(pitch_records_by_id.values()))
    pitcher_feature_rows: list[PitcherDailyFeatureRow] = []
    lineup_feature_rows: list[LineupDailyFeatureRow] = []
    game_context_rows: list[GameContextFeatureRow] = []
    park_k_factor_table = load_park_k_factors()
    weather_lookup = load_latest_weather_snapshots_for_date(
        output_dir=output_dir,
        target_date=target_date,
    )
    umpire_lookup = load_latest_umpire_snapshots_for_date(
        output_dir=output_dir,
        target_date=target_date,
    )

    for starter in mlb_metadata.probable_starters:
        game = games_by_pk.get(starter.game_pk)
        if game is None:
            continue
        pitcher_row = _build_pitcher_daily_feature_row(
            starter=starter,
            game=game,
            all_rows=all_pitch_records,
            history_start_date=history_start_date,
            history_end_date=history_end_date,
        )
        pitcher_feature_rows.append(pitcher_row)
        lineup_feature_rows.append(
            _build_lineup_daily_feature_row(
                starter=starter,
                game=game,
                lineup_snapshot=selected_lineups.get((starter.game_pk, _opponent_team_side(starter.team_side))),
                all_rows=all_pitch_records,
                history_start_date=history_start_date,
                history_end_date=history_end_date,
                pitcher_hand=pitcher_row.pitcher_hand,
            )
        )
        game_context_rows.append(
            _build_game_context_feature_row(
                starter=starter,
                game=game,
                lineup_snapshot=selected_lineups.get((starter.game_pk, _opponent_team_side(starter.team_side))),
                all_rows=all_pitch_records,
                park_k_factor_table=park_k_factor_table,
                weather_lookup=weather_lookup,
                umpire_lookup=umpire_lookup,
            )
        )

    normalized_root = (
        output_root
        / "normalized"
        / "statcast_search"
        / f"date={target_date.isoformat()}"
        / f"run={run_id}"
    )
    pull_manifest_path = normalized_root / "pull_manifest.jsonl"
    pitch_level_base_path = normalized_root / "pitch_level_base.jsonl"
    pitcher_daily_features_path = normalized_root / "pitcher_daily_features.jsonl"
    lineup_daily_features_path = normalized_root / "lineup_daily_features.jsonl"
    game_context_features_path = normalized_root / "game_context_features.jsonl"
    _write_jsonl(pull_manifest_path, pull_records)
    _write_jsonl(pitch_level_base_path, all_pitch_records)
    _write_jsonl(pitcher_daily_features_path, pitcher_feature_rows)
    _write_jsonl(lineup_daily_features_path, lineup_feature_rows)
    _write_jsonl(game_context_features_path, game_context_rows)

    return StatcastFeatureIngestResult(
        target_date=target_date,
        history_start_date=history_start_date,
        history_end_date=history_end_date,
        run_id=run_id,
        mlb_games_path=mlb_metadata.games_path,
        mlb_probable_starters_path=mlb_metadata.probable_starters_path,
        mlb_lineup_snapshots_path=mlb_metadata.lineup_snapshots_path,
        pull_manifest_path=pull_manifest_path,
        pitch_level_base_path=pitch_level_base_path,
        pitcher_daily_features_path=pitcher_daily_features_path,
        lineup_daily_features_path=lineup_daily_features_path,
        game_context_features_path=game_context_features_path,
        raw_pull_count=len(pull_records),
        pitch_level_record_count=len(all_pitch_records),
        pitcher_feature_count=len(pitcher_feature_rows),
        lineup_feature_count=len(lineup_feature_rows),
        game_context_feature_count=len(game_context_rows),
    )


__all__ = [
    # Public entry points.
    "DEFAULT_HISTORY_DAYS",
    "DEFAULT_MAX_FETCH_WORKERS",
    "StatcastFeatureIngestResult",
    "ingest_statcast_features_for_date",
    # Data contracts re-exported from submodules.
    "GameContextFeatureRow",
    "LineupDailyFeatureRow",
    "PitcherDailyFeatureRow",
    "StatcastPitchRecord",
    "StatcastPullRecord",
    # HTTP client + URL + normalizer re-exported from statcast_ingest.
    "StatcastSearchClient",
    "build_statcast_search_csv_url",
    "normalize_statcast_csv_text",
]
