"""MLB Stats API ingest for schedule, probable starters, and lineup snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, date, datetime
import json
import os
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import urlopen

SCHEDULE_ENDPOINT = "https://statsapi.mlb.com/api/v1/schedule"
FEED_LIVE_ENDPOINT_TEMPLATE = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def parse_api_datetime(value: str) -> datetime:
    """Parse ISO-8601 datetimes returned by MLB Stats API."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def format_utc_timestamp(value: datetime) -> str:
    """Render datetimes consistently for keys and JSON output."""
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_odds_matchup_key(
    *,
    official_date: str,
    away_team_abbreviation: str,
    home_team_abbreviation: str,
    commence_time: datetime,
) -> str:
    """Build a deterministic cross-source game key for later odds joins."""
    return "|".join(
        [
            official_date,
            away_team_abbreviation.upper(),
            home_team_abbreviation.upper(),
            format_utc_timestamp(commence_time),
        ]
    )


def build_schedule_url(target_date: date) -> str:
    """Return the schedule endpoint URL for one target date."""
    query = urlencode(
        {
            "sportId": 1,
            "date": target_date.isoformat(),
            "hydrate": "probablePitcher(note),team",
        }
    )
    return f"{SCHEDULE_ENDPOINT}?{query}"


def build_feed_live_url(game_pk: int) -> str:
    """Return the feed/live endpoint URL for one game."""
    return FEED_LIVE_ENDPOINT_TEMPLATE.format(game_pk=game_pk)


@dataclass(frozen=True)
class GameRecord:
    """Normalized game metadata from the daily schedule feed."""

    game_pk: int
    official_date: str
    commence_time: datetime
    captured_at: datetime
    status: str
    status_code: str
    venue_id: int | None
    venue_name: str
    home_team_id: int
    home_team_abbreviation: str
    home_team_name: str
    away_team_id: int
    away_team_abbreviation: str
    away_team_name: str
    game_number: int
    double_header: str
    day_night: str
    odds_matchup_key: str


@dataclass(frozen=True)
class ProbableStarterRecord:
    """Normalized probable starter record from the schedule hydration."""

    game_pk: int
    official_date: str
    captured_at: datetime
    team_side: str
    team_id: int
    team_abbreviation: str
    team_name: str
    pitcher_id: int | None
    pitcher_name: str | None
    pitcher_note: str | None
    odds_matchup_key: str


@dataclass(frozen=True)
class LineupEntry:
    """One batter position inside a captured lineup snapshot."""

    lineup_position: int
    player_id: int
    player_name: str
    batting_order_code: str | None
    position_abbreviation: str | None


@dataclass(frozen=True)
class LineupSnapshot:
    """Captured lineup state for one team in one game at one fetch time."""

    lineup_snapshot_id: str
    game_pk: int
    official_date: str
    captured_at: datetime
    team_side: str
    team_id: int
    team_abbreviation: str
    team_name: str
    game_state: str
    game_status_code: str
    is_confirmed: bool
    batting_order_player_ids: tuple[int, ...]
    batter_player_ids: tuple[int, ...]
    lineup_entries: tuple[LineupEntry, ...]
    odds_matchup_key: str


@dataclass(frozen=True)
class MLBMetadataIngestResult:
    """Filesystem output summary for one ingest run."""

    target_date: date
    run_id: str
    schedule_raw_path: Path
    feed_live_raw_paths: tuple[Path, ...]
    games_path: Path
    probable_starters_path: Path
    lineup_snapshots_path: Path
    game_count: int
    probable_starter_count: int
    lineup_snapshot_count: int


class MLBStatsAPIClient:
    """Small stdlib-only HTTP client for MLB Stats API."""

    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    def fetch_json(self, url: str) -> dict[str, Any]:
        with urlopen(url, timeout=self.timeout_seconds) as response:
            return json.load(response)


def _path_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _lineup_snapshot_id(*, game_pk: int, team_side: str, captured_at: datetime) -> str:
    return f"lineup:{game_pk}:{team_side}:{_path_timestamp(captured_at)}"


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, indent=2, sort_keys=True)}\n", encoding="utf-8")


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


def _team_name(team_block: dict[str, Any]) -> str:
    return team_block.get("name") or "Unknown Team"


def normalize_schedule_payload(
    payload: dict[str, Any],
    *,
    captured_at: datetime,
) -> tuple[list[GameRecord], list[ProbableStarterRecord]]:
    """Normalize games and probable starters from one schedule payload."""
    games: list[GameRecord] = []
    probable_starters: list[ProbableStarterRecord] = []

    for day in payload.get("dates", []):
        for game in day.get("games", []):
            official_date = game["officialDate"]
            commence_time = parse_api_datetime(game["gameDate"])
            teams = game["teams"]
            away_team = teams["away"]["team"]
            home_team = teams["home"]["team"]
            odds_matchup_key = build_odds_matchup_key(
                official_date=official_date,
                away_team_abbreviation=away_team["abbreviation"],
                home_team_abbreviation=home_team["abbreviation"],
                commence_time=commence_time,
            )

            games.append(
                GameRecord(
                    game_pk=game["gamePk"],
                    official_date=official_date,
                    commence_time=commence_time,
                    captured_at=captured_at,
                    status=game["status"]["detailedState"],
                    status_code=game["status"]["statusCode"],
                    venue_id=(game.get("venue") or {}).get("id"),
                    venue_name=(game.get("venue") or {}).get("name", "Unknown Venue"),
                    home_team_id=home_team["id"],
                    home_team_abbreviation=home_team["abbreviation"],
                    home_team_name=_team_name(home_team),
                    away_team_id=away_team["id"],
                    away_team_abbreviation=away_team["abbreviation"],
                    away_team_name=_team_name(away_team),
                    game_number=game.get("gameNumber", 1),
                    double_header=game.get("doubleHeader", "N"),
                    day_night=game.get("dayNight", "unknown"),
                    odds_matchup_key=odds_matchup_key,
                )
            )

            for team_side in ("away", "home"):
                team_block = teams[team_side]
                probable_pitcher = team_block.get("probablePitcher") or {}
                probable_starters.append(
                    ProbableStarterRecord(
                        game_pk=game["gamePk"],
                        official_date=official_date,
                        captured_at=captured_at,
                        team_side=team_side,
                        team_id=team_block["team"]["id"],
                        team_abbreviation=team_block["team"]["abbreviation"],
                        team_name=_team_name(team_block["team"]),
                        pitcher_id=probable_pitcher.get("id"),
                        pitcher_name=probable_pitcher.get("fullName"),
                        pitcher_note=probable_pitcher.get("note"),
                        odds_matchup_key=odds_matchup_key,
                    )
                )

    return games, probable_starters


def normalize_feed_live_payload(
    payload: dict[str, Any],
    *,
    captured_at: datetime,
) -> list[LineupSnapshot]:
    """Normalize one feed/live payload into per-team lineup snapshots."""
    game_data = payload["gameData"]
    teams = game_data["teams"]
    datetime_block = game_data["datetime"]
    boxscore_teams = payload.get("liveData", {}).get("boxscore", {}).get("teams", {})

    commence_time = parse_api_datetime(datetime_block["dateTime"])
    official_date = datetime_block["officialDate"]
    odds_matchup_key = build_odds_matchup_key(
        official_date=official_date,
        away_team_abbreviation=teams["away"]["abbreviation"],
        home_team_abbreviation=teams["home"]["abbreviation"],
        commence_time=commence_time,
    )

    snapshots: list[LineupSnapshot] = []
    for team_side in ("away", "home"):
        team_boxscore = boxscore_teams.get(team_side, {})
        team_meta = teams[team_side]
        players = team_boxscore.get("players") or {}
        batting_order_player_ids = tuple(int(player_id) for player_id in team_boxscore.get("battingOrder", []))
        batter_player_ids = tuple(int(player_id) for player_id in team_boxscore.get("batters", []))
        lineup_entries: list[LineupEntry] = []

        for lineup_position, player_id in enumerate(batting_order_player_ids, start=1):
            player = players.get(f"ID{player_id}", {})
            person = player.get("person") or {}
            position = player.get("position") or {}
            lineup_entries.append(
                LineupEntry(
                    lineup_position=lineup_position,
                    player_id=player_id,
                    player_name=person.get("fullName") or f"Player {player_id}",
                    batting_order_code=player.get("battingOrder"),
                    position_abbreviation=position.get("abbreviation"),
                )
            )

        snapshots.append(
            LineupSnapshot(
                lineup_snapshot_id=_lineup_snapshot_id(
                    game_pk=payload["gamePk"],
                    team_side=team_side,
                    captured_at=captured_at,
                ),
                game_pk=payload["gamePk"],
                official_date=official_date,
                captured_at=captured_at,
                team_side=team_side,
                team_id=team_meta["id"],
                team_abbreviation=team_meta["abbreviation"],
                team_name=team_meta["name"],
                game_state=game_data["status"]["detailedState"],
                game_status_code=game_data["status"]["statusCode"],
                is_confirmed=len(batting_order_player_ids) >= 9,
                batting_order_player_ids=batting_order_player_ids,
                batter_player_ids=batter_player_ids,
                lineup_entries=tuple(lineup_entries),
                odds_matchup_key=odds_matchup_key,
            )
        )

    return snapshots


def ingest_mlb_metadata_for_date(
    *,
    target_date: date,
    output_dir: Path | str = "data",
    client: MLBStatsAPIClient | None = None,
    now: Callable[[], datetime] = utc_now,
) -> MLBMetadataIngestResult:
    """Fetch, persist, and normalize MLB metadata for one target date."""
    if client is None:
        client = MLBStatsAPIClient()

    output_root = Path(output_dir)
    run_started_at = now().astimezone(UTC)
    run_id = _path_timestamp(run_started_at)

    schedule_captured_at = now().astimezone(UTC)
    schedule_payload = client.fetch_json(build_schedule_url(target_date))
    schedule_raw_path = (
        output_root
        / "raw"
        / "mlb_stats_api"
        / f"date={target_date.isoformat()}"
        / "schedule"
        / f"captured_at={_path_timestamp(schedule_captured_at)}.json"
    )
    _write_json(schedule_raw_path, schedule_payload)

    games, probable_starters = normalize_schedule_payload(
        schedule_payload,
        captured_at=schedule_captured_at,
    )

    lineup_snapshots: list[LineupSnapshot] = []
    feed_live_raw_paths: list[Path] = []
    for game_record in games:
        feed_captured_at = now().astimezone(UTC)
        feed_payload = client.fetch_json(build_feed_live_url(game_record.game_pk))
        feed_live_raw_path = (
            output_root
            / "raw"
            / "mlb_stats_api"
            / f"date={target_date.isoformat()}"
            / "feed_live"
            / f"game_pk={game_record.game_pk}"
            / f"captured_at={_path_timestamp(feed_captured_at)}.json"
        )
        _write_json(feed_live_raw_path, feed_payload)
        feed_live_raw_paths.append(feed_live_raw_path)
        lineup_snapshots.extend(
            normalize_feed_live_payload(feed_payload, captured_at=feed_captured_at)
        )

    normalized_root = (
        output_root
        / "normalized"
        / "mlb_stats_api"
        / f"date={target_date.isoformat()}"
        / f"run={run_id}"
    )
    games_path = normalized_root / "games.jsonl"
    probable_starters_path = normalized_root / "probable_starters.jsonl"
    lineup_snapshots_path = normalized_root / "lineup_snapshots.jsonl"

    _write_jsonl(games_path, games)
    _write_jsonl(probable_starters_path, probable_starters)
    _write_jsonl(lineup_snapshots_path, lineup_snapshots)

    return MLBMetadataIngestResult(
        target_date=target_date,
        run_id=run_id,
        schedule_raw_path=schedule_raw_path,
        feed_live_raw_paths=tuple(feed_live_raw_paths),
        games_path=games_path,
        probable_starters_path=probable_starters_path,
        lineup_snapshots_path=lineup_snapshots_path,
        game_count=len(games),
        probable_starter_count=len(probable_starters),
        lineup_snapshot_count=len(lineup_snapshots),
    )
