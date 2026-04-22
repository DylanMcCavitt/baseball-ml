"""The Odds API ingest for sportsbook pitcher strikeout snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, date, datetime, timedelta
import json
import os
from pathlib import Path
from typing import Any, Callable
import unicodedata
from urllib.parse import urlencode
from urllib.request import urlopen

from ..env import load_repo_env
from .mlb_stats_api import (
    GameRecord,
    ProbableStarterRecord,
    build_odds_matchup_key,
    format_utc_timestamp,
    parse_api_datetime,
    utc_now,
)

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_API_SPORT_KEY = "baseball_mlb"
PITCHER_STRIKEOUTS_MARKET = "pitcher_strikeouts"
DEFAULT_REGIONS = "us"
MATCHUP_TIME_TOLERANCE = timedelta(minutes=10)


@dataclass(frozen=True)
class OddsEventGameMappingRecord:
    """Bridge one Odds API event id back to the MLB `gamePk` seam."""

    mapping_id: str
    official_date: str
    captured_at: datetime
    event_id: str
    sport_key: str
    commence_time: datetime
    home_team_name: str
    away_team_name: str
    home_team_abbreviation: str
    away_team_abbreviation: str
    odds_matchup_key: str
    game_pk: int | None
    match_status: str


@dataclass(frozen=True)
class PropLineSnapshotRecord:
    """Normalized two-sided pitcher strikeout line snapshot for one book."""

    line_snapshot_id: str
    official_date: str
    captured_at: datetime
    sportsbook: str
    sportsbook_title: str
    event_id: str
    game_pk: int | None
    odds_matchup_key: str
    match_status: str
    commence_time: datetime
    home_team_name: str
    away_team_name: str
    player_id: str
    pitcher_mlb_id: int | None
    player_name: str
    market: str
    line: float
    over_odds: int
    under_odds: int
    market_last_update: datetime
    bookmaker_last_update: datetime | None


@dataclass(frozen=True)
class OddsAPIIngestResult:
    """Filesystem output summary for one The Odds API ingest run."""

    target_date: date
    run_id: str
    mlb_games_path: Path
    mlb_probable_starters_path: Path
    events_raw_path: Path
    event_odds_raw_paths: tuple[Path, ...]
    event_mappings_path: Path
    prop_line_snapshots_path: Path
    candidate_event_count: int
    matched_event_count: int
    unmatched_event_count: int
    skipped_unmatched_event_count: int
    matched_events_without_props_count: int
    prop_line_count: int
    resolved_pitcher_prop_count: int
    unresolved_pitcher_prop_count: int
    skipped_prop_count: int


@dataclass(frozen=True)
class _LoadedMLBMetadata:
    games_path: Path
    probable_starters_path: Path
    games: tuple[GameRecord, ...]
    probable_starters: tuple[ProbableStarterRecord, ...]


class OddsAPIClient:
    """Small stdlib-only HTTP client for The Odds API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        load_repo_env()
        resolved_api_key = (api_key or os.environ.get("ODDS_API_KEY") or "").strip()
        if not resolved_api_key:
            raise ValueError(
                "ODDS_API_KEY must be set or passed explicitly to OddsAPIClient."
            )
        self.api_key = resolved_api_key
        self.timeout_seconds = timeout_seconds

    def fetch_json(self, url: str) -> Any:
        with urlopen(url, timeout=self.timeout_seconds) as response:
            return json.load(response)


def build_events_url(*, api_key: str) -> str:
    """Return the current-event endpoint URL for MLB."""
    query = urlencode(
        {
            "apiKey": api_key,
            "dateFormat": "iso",
        }
    )
    return f"{ODDS_API_BASE_URL}/sports/{ODDS_API_SPORT_KEY}/events?{query}"


def build_event_odds_url(
    event_id: str,
    *,
    api_key: str,
    regions: str = DEFAULT_REGIONS,
) -> str:
    """Return the event-odds endpoint URL for pitcher strikeout markets."""
    query = urlencode(
        {
            "apiKey": api_key,
            "regions": regions,
            "markets": PITCHER_STRIKEOUTS_MARKET,
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
    )
    return (
        f"{ODDS_API_BASE_URL}/sports/{ODDS_API_SPORT_KEY}/events/"
        f"{event_id}/odds?{query}"
    )


def _path_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _line_token(value: float) -> str:
    rendered = format(value, "g").replace("-", "m").replace(".", "_")
    return rendered or "0"


def _slug_token(value: str) -> str:
    return "-".join(_normalize_text_key(value).lower().split())


def _normalize_text_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    without_accents = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    cleaned = "".join(
        character if character.isalnum() else " "
        for character in without_accents
    )
    return " ".join(cleaned.upper().split())


def _match_game_for_event(
    *,
    candidate_games: list[GameRecord],
    commence_time: datetime,
) -> GameRecord | None:
    exact_match = next(
        (game for game in candidate_games if game.commence_time == commence_time),
        None,
    )
    if exact_match is not None:
        return exact_match

    tolerated_games = [
        game
        for game in candidate_games
        if abs(game.commence_time - commence_time) <= MATCHUP_TIME_TOLERANCE
    ]
    if not tolerated_games:
        return None
    return min(
        tolerated_games,
        key=lambda game: (
            abs(game.commence_time - commence_time),
            game.commence_time,
            game.game_pk,
        ),
    )


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(_json_ready(payload), indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_ready(record), sort_keys=True))
            handle.write("\n")


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _latest_run_dir(root: Path) -> Path:
    run_dirs = sorted(path for path in root.glob("run=*") if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(
            "No normalized MLB metadata runs were found. "
            "Run `uv run python -m mlb_props_stack ingest-mlb-metadata --date ...` first."
        )
    return run_dirs[-1]


def _load_latest_mlb_metadata_for_date(
    *,
    target_date: date,
    output_dir: Path | str,
) -> _LoadedMLBMetadata:
    normalized_root = (
        Path(output_dir)
        / "normalized"
        / "mlb_stats_api"
        / f"date={target_date.isoformat()}"
    )
    latest_run_dir = _latest_run_dir(normalized_root)
    games_path = latest_run_dir / "games.jsonl"
    probable_starters_path = latest_run_dir / "probable_starters.jsonl"
    if not games_path.exists() or not probable_starters_path.exists():
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
    return _LoadedMLBMetadata(
        games_path=games_path,
        probable_starters_path=probable_starters_path,
        games=games,
        probable_starters=probable_starters,
    )


def normalize_events_payload(
    payload: Any,
    *,
    target_date: date,
    captured_at: datetime,
    games: tuple[GameRecord, ...],
) -> list[OddsEventGameMappingRecord]:
    """Normalize target-date event ids and map them back to MLB games."""
    if not isinstance(payload, list):
        raise TypeError("events payload must be a list")

    team_pair_lookup: dict[tuple[str, str], list[GameRecord]] = {}
    for game in games:
        key = (
            _normalize_text_key(game.away_team_name),
            _normalize_text_key(game.home_team_name),
        )
        team_pair_lookup.setdefault(key, []).append(game)

    mappings: list[OddsEventGameMappingRecord] = []
    for event in payload:
        if not isinstance(event, dict):
            continue
        away_team_name = event.get("away_team")
        home_team_name = event.get("home_team")
        if not isinstance(away_team_name, str) or not isinstance(home_team_name, str):
            continue

        candidate_games = team_pair_lookup.get(
            (
                _normalize_text_key(away_team_name),
                _normalize_text_key(home_team_name),
            )
        )
        if not candidate_games:
            continue

        commence_time = parse_api_datetime(event["commence_time"])
        prototype_game = candidate_games[0]
        generated_odds_matchup_key = build_odds_matchup_key(
            official_date=target_date.isoformat(),
            away_team_abbreviation=prototype_game.away_team_abbreviation,
            home_team_abbreviation=prototype_game.home_team_abbreviation,
            commence_time=commence_time,
        )
        matched_game = _match_game_for_event(
            candidate_games=candidate_games,
            commence_time=commence_time,
        )
        odds_matchup_key = (
            matched_game.odds_matchup_key
            if matched_game is not None
            else generated_odds_matchup_key
        )

        mappings.append(
            OddsEventGameMappingRecord(
                mapping_id=(
                    f"odds-map:{event['id']}:{_path_timestamp(captured_at)}"
                ),
                official_date=target_date.isoformat(),
                captured_at=captured_at,
                event_id=event["id"],
                sport_key=event.get("sport_key", ODDS_API_SPORT_KEY),
                commence_time=commence_time,
                home_team_name=home_team_name,
                away_team_name=away_team_name,
                home_team_abbreviation=prototype_game.home_team_abbreviation,
                away_team_abbreviation=prototype_game.away_team_abbreviation,
                odds_matchup_key=odds_matchup_key,
                game_pk=matched_game.game_pk if matched_game is not None else None,
                match_status="matched" if matched_game is not None else "unmatched",
            )
        )

    return sorted(mappings, key=lambda record: (record.commence_time, record.event_id))


def _coerce_american_odds(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("price must be an integer American-odds value")
    rendered = int(value)
    if rendered != value:
        raise ValueError("price must be an integer American-odds value")
    if rendered == 0:
        raise ValueError("price cannot be zero")
    return rendered


def _resolve_player_identity(
    *,
    game_pk: int | None,
    player_name: str,
    probable_starter_lookup: dict[tuple[int, str], ProbableStarterRecord],
) -> tuple[str, int | None]:
    if game_pk is not None:
        probable_starter = probable_starter_lookup.get(
            (game_pk, _normalize_text_key(player_name))
        )
        if probable_starter is not None and probable_starter.pitcher_id is not None:
            return f"mlb-pitcher:{probable_starter.pitcher_id}", probable_starter.pitcher_id
    return f"odds-player:{_slug_token(player_name)}", None


def normalize_event_odds_payload(
    payload: dict[str, Any],
    *,
    captured_at: datetime,
    mapping: OddsEventGameMappingRecord,
    probable_starter_lookup: dict[tuple[int, str], ProbableStarterRecord],
) -> tuple[list[PropLineSnapshotRecord], int]:
    """Normalize one event-odds payload into paired prop-line snapshots."""
    snapshots: list[PropLineSnapshotRecord] = []
    skipped_groups = 0

    for bookmaker in payload.get("bookmakers", []):
        if not isinstance(bookmaker, dict):
            continue
        sportsbook = bookmaker.get("key")
        sportsbook_title = bookmaker.get("title")
        if not isinstance(sportsbook, str) or not isinstance(sportsbook_title, str):
            continue

        bookmaker_last_update_raw = bookmaker.get("last_update")
        bookmaker_last_update = (
            parse_api_datetime(bookmaker_last_update_raw)
            if isinstance(bookmaker_last_update_raw, str)
            else None
        )

        for market in bookmaker.get("markets", []):
            if not isinstance(market, dict):
                continue
            if market.get("key") != PITCHER_STRIKEOUTS_MARKET:
                continue

            grouped_outcomes: dict[tuple[str, float], dict[str, Any]] = {}
            for outcome in market.get("outcomes", []):
                if not isinstance(outcome, dict):
                    continue
                side = str(outcome.get("name", "")).strip().lower()
                if side not in {"over", "under"}:
                    continue

                player_name = outcome.get("description")
                point = outcome.get("point")
                price = outcome.get("price")
                if not isinstance(player_name, str) or not player_name.strip():
                    continue

                try:
                    line = float(point)
                    american_odds = _coerce_american_odds(price)
                except (TypeError, ValueError):
                    continue

                group_key = (_normalize_text_key(player_name), line)
                group = grouped_outcomes.setdefault(
                    group_key,
                    {
                        "player_name": player_name.strip(),
                        "line": line,
                    },
                )
                group[side] = american_odds

            market_last_update_raw = market.get("last_update")
            market_last_update = (
                parse_api_datetime(market_last_update_raw)
                if isinstance(market_last_update_raw, str)
                else bookmaker_last_update or captured_at
            )

            for grouped in grouped_outcomes.values():
                if "over" not in grouped or "under" not in grouped:
                    skipped_groups += 1
                    continue

                player_id, pitcher_mlb_id = _resolve_player_identity(
                    game_pk=mapping.game_pk,
                    player_name=grouped["player_name"],
                    probable_starter_lookup=probable_starter_lookup,
                )
                line = grouped["line"]
                snapshots.append(
                    PropLineSnapshotRecord(
                        line_snapshot_id=(
                            "prop-line:"
                            f"{sportsbook}:{mapping.event_id}:{player_id}:"
                            f"{_line_token(line)}:{_path_timestamp(captured_at)}"
                        ),
                        official_date=mapping.official_date,
                        captured_at=captured_at,
                        sportsbook=sportsbook,
                        sportsbook_title=sportsbook_title,
                        event_id=mapping.event_id,
                        game_pk=mapping.game_pk,
                        odds_matchup_key=mapping.odds_matchup_key,
                        match_status=mapping.match_status,
                        commence_time=mapping.commence_time,
                        home_team_name=mapping.home_team_name,
                        away_team_name=mapping.away_team_name,
                        player_id=player_id,
                        pitcher_mlb_id=pitcher_mlb_id,
                        player_name=grouped["player_name"],
                        market=PITCHER_STRIKEOUTS_MARKET,
                        line=line,
                        over_odds=grouped["over"],
                        under_odds=grouped["under"],
                        market_last_update=market_last_update,
                        bookmaker_last_update=bookmaker_last_update,
                    )
                )

    return (
        sorted(
            snapshots,
            key=lambda record: (
                record.captured_at,
                record.sportsbook,
                record.player_name,
                record.line,
            ),
        ),
        skipped_groups,
    )


def ingest_odds_api_pitcher_lines_for_date(
    *,
    target_date: date,
    output_dir: Path | str = "data",
    api_key: str | None = None,
    regions: str = DEFAULT_REGIONS,
    client: OddsAPIClient | None = None,
    now: Callable[[], datetime] = utc_now,
) -> OddsAPIIngestResult:
    """Fetch, persist, and normalize Odds API pitcher strikeout lines."""
    if client is None:
        client = OddsAPIClient(api_key=api_key)

    mlb_metadata = _load_latest_mlb_metadata_for_date(
        target_date=target_date,
        output_dir=output_dir,
    )
    probable_starter_lookup = {
        (starter.game_pk, _normalize_text_key(starter.pitcher_name)): starter
        for starter in mlb_metadata.probable_starters
        if starter.pitcher_name
    }

    output_root = Path(output_dir)
    run_started_at = now().astimezone(UTC)
    run_id = _path_timestamp(run_started_at)

    events_captured_at = now().astimezone(UTC)
    events_payload = client.fetch_json(build_events_url(api_key=client.api_key))
    events_raw_path = (
        output_root
        / "raw"
        / "the_odds_api"
        / f"date={target_date.isoformat()}"
        / "events"
        / f"captured_at={_path_timestamp(events_captured_at)}.json"
    )
    _write_json(events_raw_path, events_payload)

    event_mappings = normalize_events_payload(
        events_payload,
        target_date=target_date,
        captured_at=events_captured_at,
        games=mlb_metadata.games,
    )

    event_odds_raw_paths: list[Path] = []
    prop_line_snapshots: list[PropLineSnapshotRecord] = []
    skipped_prop_count = 0
    skipped_unmatched_event_count = 0
    matched_events_without_props_count = 0
    resolved_pitcher_prop_count = 0
    unresolved_pitcher_prop_count = 0

    for mapping in event_mappings:
        odds_captured_at = now().astimezone(UTC)
        event_odds_payload = client.fetch_json(
            build_event_odds_url(
                mapping.event_id,
                api_key=client.api_key,
                regions=regions,
            )
        )
        event_odds_raw_path = (
            output_root
            / "raw"
            / "the_odds_api"
            / f"date={target_date.isoformat()}"
            / "event_odds"
            / f"event_id={mapping.event_id}"
            / f"captured_at={_path_timestamp(odds_captured_at)}.json"
        )
        _write_json(event_odds_raw_path, event_odds_payload)
        event_odds_raw_paths.append(event_odds_raw_path)

        normalized_snapshots, skipped_groups = normalize_event_odds_payload(
            event_odds_payload,
            captured_at=odds_captured_at,
            mapping=mapping,
            probable_starter_lookup=probable_starter_lookup,
        )
        skipped_prop_count += skipped_groups
        if mapping.match_status != "matched":
            skipped_unmatched_event_count += 1
            continue
        if not normalized_snapshots:
            matched_events_without_props_count += 1
            continue
        prop_line_snapshots.extend(normalized_snapshots)
        resolved_pitcher_prop_count += sum(
            1 for snapshot in normalized_snapshots if snapshot.pitcher_mlb_id is not None
        )
        unresolved_pitcher_prop_count += sum(
            1 for snapshot in normalized_snapshots if snapshot.pitcher_mlb_id is None
        )

    normalized_root = (
        output_root
        / "normalized"
        / "the_odds_api"
        / f"date={target_date.isoformat()}"
        / f"run={run_id}"
    )
    event_mappings_path = normalized_root / "event_game_mappings.jsonl"
    prop_line_snapshots_path = normalized_root / "prop_line_snapshots.jsonl"
    _write_jsonl(event_mappings_path, event_mappings)
    _write_jsonl(prop_line_snapshots_path, prop_line_snapshots)

    matched_event_count = sum(
        1 for mapping in event_mappings if mapping.match_status == "matched"
    )
    unmatched_event_count = len(event_mappings) - matched_event_count

    return OddsAPIIngestResult(
        target_date=target_date,
        run_id=run_id,
        mlb_games_path=mlb_metadata.games_path,
        mlb_probable_starters_path=mlb_metadata.probable_starters_path,
        events_raw_path=events_raw_path,
        event_odds_raw_paths=tuple(event_odds_raw_paths),
        event_mappings_path=event_mappings_path,
        prop_line_snapshots_path=prop_line_snapshots_path,
        candidate_event_count=len(event_mappings),
        matched_event_count=matched_event_count,
        unmatched_event_count=unmatched_event_count,
        skipped_unmatched_event_count=skipped_unmatched_event_count,
        matched_events_without_props_count=matched_events_without_props_count,
        prop_line_count=len(prop_line_snapshots),
        resolved_pitcher_prop_count=resolved_pitcher_prop_count,
        unresolved_pitcher_prop_count=unresolved_pitcher_prop_count,
        skipped_prop_count=skipped_prop_count,
    )
