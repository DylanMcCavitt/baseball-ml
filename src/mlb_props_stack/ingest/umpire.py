"""Pregame home-plate umpire ingest with 30-day rolling strike-zone features.

AGE-204 calls for adapter-style umpire ingest emitting one ``umpire_daily``
row per scheduled game plus rolling ``ump_called_strike_rate`` and
``ump_k_per_9_delta_vs_league`` metrics computed over the prior 30
umpiring days. The issue suggests UmpScorecards/Retrosheet, but the MLB
Stats API ``feed/live`` endpoint already returns a
``liveData.boxscore.officials`` block that includes the home-plate
umpire's id and full name. That payload is already persisted to
``data/raw/mlb_stats_api/date=.../feed_live/game_pk=.../captured_at=...``
during the MLB-metadata ingest run, which lets this module mine the
existing raw file instead of opening a new scraping dependency.

The public entry point is :func:`ingest_umpire_for_date`. It:

* loads the latest pregame-valid MLB metadata run for ``target_date``,
* for each scheduled game, picks the most recent persisted feed/live
  payload (or falls back to a fresh HTTP fetch if none exists yet),
* extracts the ``officialType == "Home Plate"`` entry as an
  :class:`UmpireAssignmentRecord`,
* writes a raw per-game artifact containing the extracted officials
  block under ``data/raw/umpire/date=.../game_pk=.../captured_at=...``,
* walks the prior 30 days of normalized umpire + Statcast pitch-level
  base artifacts to compute ``ump_called_strike_rate_30d`` and
  ``ump_k_per_9_delta_vs_league_30d`` per umpire, and
* writes one :class:`UmpireSnapshotRecord` per slate game under
  ``data/normalized/umpire/date=.../run=.../umpire_snapshots.jsonl``.

Games whose feed/live payload has no home-plate entry (e.g. the umpire
assignment has not been published yet) emit a sentinel row with
``umpire_status="missing_umpire_source"`` so downstream coverage checks
can flag the gap instead of dropping the slate row.

Each normalized row enforces ``captured_at <= commence_time`` as a
leakage guardrail, matching the weather ingest contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, date, datetime, timedelta
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError

from .mlb_stats_api import (
    GameRecord,
    MLBStatsAPIClient,
    build_feed_live_url,
    format_utc_timestamp,
    parse_api_datetime,
    utc_now,
)

UMPIRE_SOURCE_MLB_STATS_API_FEED_LIVE = "mlb_stats_api_feed_live"

UMPIRE_STATUS_OK = "ok"
UMPIRE_STATUS_MISSING_SOURCE = "missing_umpire_source"

UMPIRE_STATUSES: tuple[str, ...] = (
    UMPIRE_STATUS_OK,
    UMPIRE_STATUS_MISSING_SOURCE,
)

HOME_PLATE_OFFICIAL_TYPE = "Home Plate"

# K/9 ≈ K_rate × 38.25 (9 innings × ~4.25 plate appearances per inning).
# We use the multiplier rather than true innings because the Statcast
# pitch-level base captures plate appearances via the "final-pitch"
# marker, not innings directly, and this approximation is stable enough
# for a delta-vs-league feature.
APPROXIMATE_PA_PER_NINE_INNINGS = 38.25

DEFAULT_UMPIRE_HISTORY_DAYS = 30


@dataclass(frozen=True)
class UmpireAssignmentRecord:
    """Raw-ish home-plate umpire assignment for one scheduled game."""

    umpire_assignment_id: str
    official_date: str
    game_pk: int
    commence_time: datetime
    captured_at: datetime
    umpire_source: str | None
    umpire_status: str
    umpire_id: int | None
    umpire_name: str | None
    error_message: str | None


@dataclass(frozen=True)
class UmpireSnapshotRecord:
    """Normalized pregame umpire snapshot joined with 30-day rolling metrics."""

    umpire_snapshot_id: str
    official_date: str
    game_pk: int
    commence_time: datetime
    captured_at: datetime
    umpire_source: str | None
    umpire_status: str
    umpire_id: int | None
    umpire_name: str | None
    error_message: str | None
    history_start_date: date
    history_end_date: date
    ump_called_strike_rate_30d: float | None
    ump_k_per_9_delta_vs_league_30d: float | None


@dataclass(frozen=True)
class UmpireIngestResult:
    """Filesystem output summary for one umpire ingest build."""

    target_date: date
    run_id: str
    mlb_games_path: Path
    umpire_snapshots_path: Path
    raw_snapshot_paths: tuple[Path, ...]
    snapshot_count: int
    ok_snapshot_count: int
    missing_source_count: int
    history_start_date: date
    history_end_date: date


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        tmp_path.write_text(
            f"{json.dumps(_json_ready(payload), indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_jsonl(path: Path, records: Iterable[Any]) -> None:
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


def _latest_mlb_games_path(
    *,
    output_dir: Path,
    target_date: date,
) -> Path:
    normalized_root = (
        output_dir
        / "normalized"
        / "mlb_stats_api"
        / f"date={target_date.isoformat()}"
    )
    run_dirs = sorted(path for path in normalized_root.glob("run=*") if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(
            "No normalized MLB metadata runs were found. "
            "Run `uv run python -m mlb_props_stack ingest-mlb-metadata --date ...` first."
        )
    games_path = run_dirs[-1] / "games.jsonl"
    if not games_path.exists():
        raise FileNotFoundError(
            f"Expected MLB metadata games.jsonl in {run_dirs[-1]}, but it was missing."
        )
    return games_path


def _load_games(games_path: Path) -> tuple[GameRecord, ...]:
    rows: list[GameRecord] = []
    for line in games_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(
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
        )
    return tuple(rows)


def _latest_persisted_feed_live_path(
    *,
    output_dir: Path,
    target_date: date,
    game_pk: int,
) -> Path | None:
    """Return the most recent persisted feed/live file for one game, if any."""

    feed_live_dir = (
        output_dir
        / "raw"
        / "mlb_stats_api"
        / f"date={target_date.isoformat()}"
        / "feed_live"
        / f"game_pk={game_pk}"
    )
    if not feed_live_dir.exists():
        return None
    candidates = sorted(path for path in feed_live_dir.glob("captured_at=*.json") if path.is_file())
    if not candidates:
        return None
    return candidates[-1]


def _captured_at_from_feed_live_path(path: Path) -> datetime | None:
    """Parse ``captured_at=YYYYMMDDTHHMMSSZ`` from a feed/live artifact path."""

    stem = path.stem
    prefix = "captured_at="
    if not stem.startswith(prefix):
        return None
    try:
        return datetime.strptime(stem[len(prefix) :], "%Y%m%dT%H%M%SZ").replace(
            tzinfo=UTC
        )
    except ValueError:
        return None


def _extract_home_plate_umpire(
    officials: list[dict[str, Any]] | None,
) -> tuple[int | None, str | None]:
    """Return ``(umpire_id, umpire_name)`` for the home-plate official or ``(None, None)``.

    MLB Stats API occasionally returns officials with a different label
    casing ("home plate") or with an empty ``official`` block when the
    assignment has not been finalized yet. We normalize the type string
    and fall back to ``None`` so missing entries flow into the
    ``missing_umpire_source`` sentinel path.
    """

    if not isinstance(officials, list):
        return None, None
    for entry in officials:
        if not isinstance(entry, dict):
            continue
        official_type = entry.get("officialType")
        if not isinstance(official_type, str):
            continue
        if official_type.strip().lower() != HOME_PLATE_OFFICIAL_TYPE.lower():
            continue
        official = entry.get("official") or {}
        if not isinstance(official, dict):
            continue
        umpire_id = official.get("id")
        umpire_name = official.get("fullName")
        if isinstance(umpire_id, int) and isinstance(umpire_name, str) and umpire_name.strip():
            return umpire_id, umpire_name.strip()
        # Partial entries (e.g. id with missing name) still flow into
        # missing_umpire_source so downstream coverage can flag them.
        return None, None
    return None, None


def normalize_feed_live_officials_payload(
    payload: dict[str, Any],
) -> list[dict[str, Any]] | None:
    """Pull the officials array out of a feed/live payload, if present."""

    live_data = payload.get("liveData") or {}
    if not isinstance(live_data, dict):
        return None
    boxscore = live_data.get("boxscore") or {}
    if not isinstance(boxscore, dict):
        return None
    officials = boxscore.get("officials")
    if not isinstance(officials, list):
        return None
    return officials


def _assignment_id(*, game_pk: int, captured_at: datetime) -> str:
    return f"umpire:{game_pk}:{_path_timestamp(captured_at)}"


def _snapshot_id(*, game_pk: int, captured_at: datetime) -> str:
    return f"umpire-snapshot:{game_pk}:{_path_timestamp(captured_at)}"


def _make_assignment_from_payload(
    *,
    game: GameRecord,
    payload: dict[str, Any],
    captured_at: datetime,
) -> UmpireAssignmentRecord:
    officials = normalize_feed_live_officials_payload(payload)
    umpire_id, umpire_name = _extract_home_plate_umpire(officials)
    status = UMPIRE_STATUS_OK if umpire_id is not None else UMPIRE_STATUS_MISSING_SOURCE
    source = UMPIRE_SOURCE_MLB_STATS_API_FEED_LIVE if umpire_id is not None else None
    error_message = None
    if umpire_id is None:
        error_message = "feed_live payload had no home-plate umpire entry"
    return UmpireAssignmentRecord(
        umpire_assignment_id=_assignment_id(
            game_pk=game.game_pk, captured_at=captured_at
        ),
        official_date=game.official_date,
        game_pk=game.game_pk,
        commence_time=game.commence_time,
        captured_at=captured_at,
        umpire_source=source,
        umpire_status=status,
        umpire_id=umpire_id,
        umpire_name=umpire_name,
        error_message=error_message,
    )


def _make_missing_assignment(
    *,
    game: GameRecord,
    captured_at: datetime,
    error_message: str,
) -> UmpireAssignmentRecord:
    return UmpireAssignmentRecord(
        umpire_assignment_id=_assignment_id(
            game_pk=game.game_pk, captured_at=captured_at
        ),
        official_date=game.official_date,
        game_pk=game.game_pk,
        commence_time=game.commence_time,
        captured_at=captured_at,
        umpire_source=None,
        umpire_status=UMPIRE_STATUS_MISSING_SOURCE,
        umpire_id=None,
        umpire_name=None,
        error_message=error_message,
    )


def _load_prior_umpire_game_pks_by_umpire(
    *,
    output_dir: Path,
    target_date: date,
    history_days: int,
) -> dict[int, dict[date, set[int]]]:
    """Walk prior-date umpire runs to map ``umpire_id -> {date: {game_pks}}``.

    Returns an empty dict when no prior umpire runs have been written yet.
    """

    result: dict[int, dict[date, set[int]]] = {}
    for offset in range(1, history_days + 1):
        prior_date = target_date - timedelta(days=offset)
        normalized_root = (
            output_dir
            / "normalized"
            / "umpire"
            / f"date={prior_date.isoformat()}"
        )
        if not normalized_root.exists():
            continue
        run_dirs = sorted(path for path in normalized_root.glob("run=*") if path.is_dir())
        if not run_dirs:
            continue
        snapshots_path = run_dirs[-1] / "umpire_snapshots.jsonl"
        if not snapshots_path.exists():
            continue
        for line in snapshots_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("umpire_status") != UMPIRE_STATUS_OK:
                continue
            umpire_id = row.get("umpire_id")
            game_pk = row.get("game_pk")
            if not isinstance(umpire_id, int) or not isinstance(game_pk, int):
                continue
            result.setdefault(umpire_id, {}).setdefault(prior_date, set()).add(game_pk)
    return result


@dataclass(frozen=True)
class _DailyPitchAggregates:
    total_pitches: int
    called_strikes: int
    plate_appearances: int
    strikeouts: int
    by_game_pk: dict[int, tuple[int, int, int, int]]


def _load_prior_pitch_aggregates(
    *,
    output_dir: Path,
    target_date: date,
    history_days: int,
) -> dict[date, _DailyPitchAggregates]:
    """Walk prior-date Statcast runs to build per-date pitch/PA aggregates.

    For each prior date we aggregate:

    * league-wide pitch counts, called-strike counts, plate-appearance
      counts, and strikeout counts, and
    * the same four counts per ``game_pk`` so the rolling metric can
      filter to the games a specific umpire called.

    Missing runs silently contribute nothing; the rolling metric
    degrades to ``None`` when a prior date's Statcast ingest hasn't run.
    """

    aggregates: dict[date, _DailyPitchAggregates] = {}
    for offset in range(1, history_days + 1):
        prior_date = target_date - timedelta(days=offset)
        normalized_root = (
            output_dir
            / "normalized"
            / "statcast_search"
            / f"date={prior_date.isoformat()}"
        )
        if not normalized_root.exists():
            continue
        run_dirs = sorted(path for path in normalized_root.glob("run=*") if path.is_dir())
        if not run_dirs:
            continue
        pitch_level_path = run_dirs[-1] / "pitch_level_base.jsonl"
        if not pitch_level_path.exists():
            continue
        total_pitches = 0
        called_strikes = 0
        plate_appearances = 0
        strikeouts = 0
        by_game_pk: dict[int, list[int]] = {}
        for line in pitch_level_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            game_pk = row.get("game_pk")
            if not isinstance(game_pk, int):
                continue
            game_counts = by_game_pk.setdefault(game_pk, [0, 0, 0, 0])
            total_pitches += 1
            game_counts[0] += 1
            if bool(row.get("is_called_strike")):
                called_strikes += 1
                game_counts[1] += 1
            if bool(row.get("is_plate_appearance_final_pitch")):
                plate_appearances += 1
                game_counts[2] += 1
                if bool(row.get("is_strikeout_event")):
                    strikeouts += 1
                    game_counts[3] += 1
        aggregates[prior_date] = _DailyPitchAggregates(
            total_pitches=total_pitches,
            called_strikes=called_strikes,
            plate_appearances=plate_appearances,
            strikeouts=strikeouts,
            by_game_pk={
                key: (counts[0], counts[1], counts[2], counts[3])
                for key, counts in by_game_pk.items()
            },
        )
    return aggregates


def _compute_league_totals(
    aggregates: dict[date, _DailyPitchAggregates],
) -> tuple[int, int, int, int]:
    total_pitches = 0
    called_strikes = 0
    plate_appearances = 0
    strikeouts = 0
    for daily in aggregates.values():
        total_pitches += daily.total_pitches
        called_strikes += daily.called_strikes
        plate_appearances += daily.plate_appearances
        strikeouts += daily.strikeouts
    return total_pitches, called_strikes, plate_appearances, strikeouts


def _compute_umpire_totals(
    *,
    umpire_id: int,
    umpire_game_pks: dict[date, set[int]] | None,
    pitch_aggregates: dict[date, _DailyPitchAggregates],
) -> tuple[int, int, int, int]:
    """Sum pitch/called-strike/PA/strikeout counts across the umpire's prior games."""

    if not umpire_game_pks:
        return 0, 0, 0, 0
    total_pitches = 0
    called_strikes = 0
    plate_appearances = 0
    strikeouts = 0
    for prior_date, game_pks in umpire_game_pks.items():
        daily = pitch_aggregates.get(prior_date)
        if daily is None:
            continue
        for game_pk in game_pks:
            counts = daily.by_game_pk.get(game_pk)
            if counts is None:
                continue
            total_pitches += counts[0]
            called_strikes += counts[1]
            plate_appearances += counts[2]
            strikeouts += counts[3]
    return total_pitches, called_strikes, plate_appearances, strikeouts


def compute_rolling_umpire_metrics(
    *,
    umpire_id: int,
    umpire_game_pks: dict[date, set[int]] | None,
    pitch_aggregates: dict[date, _DailyPitchAggregates],
) -> tuple[float | None, float | None]:
    """Return ``(called_strike_rate_30d, k_per_9_delta_vs_league_30d)`` for one umpire."""

    ump_pitches, ump_called_strikes, ump_pa, ump_strikeouts = _compute_umpire_totals(
        umpire_id=umpire_id,
        umpire_game_pks=umpire_game_pks,
        pitch_aggregates=pitch_aggregates,
    )
    if ump_pitches == 0:
        return None, None
    called_strike_rate = round(ump_called_strikes / ump_pitches, 6)

    league_pitches, _, league_pa, league_strikeouts = _compute_league_totals(
        pitch_aggregates
    )
    if ump_pa == 0 or league_pa == 0 or league_pitches == 0:
        return called_strike_rate, None

    ump_k_rate = ump_strikeouts / ump_pa
    league_k_rate = league_strikeouts / league_pa
    k_per_9_delta = round(
        (ump_k_rate - league_k_rate) * APPROXIMATE_PA_PER_NINE_INNINGS, 6
    )
    return called_strike_rate, k_per_9_delta


def ingest_umpire_for_date(
    *,
    target_date: date,
    output_dir: Path | str = "data",
    client: MLBStatsAPIClient | None = None,
    now: Callable[[], datetime] = utc_now,
    history_days: int = DEFAULT_UMPIRE_HISTORY_DAYS,
) -> UmpireIngestResult:
    """Extract home-plate umpires for ``target_date`` and join rolling metrics.

    Parameters
    ----------
    target_date
        MLB official slate date to ingest.
    output_dir
        Filesystem root for raw + normalized artifacts. Defaults to ``data``.
    client
        Optional :class:`MLBStatsAPIClient` override, used only when a
        scheduled game has no persisted feed/live payload yet. Tests can
        inject a stub to avoid real HTTP traffic.
    now
        Optional UTC clock override for deterministic ``captured_at``
        stamps in tests.
    history_days
        Rolling window length (calendar days) used for
        ``ump_called_strike_rate_30d`` and
        ``ump_k_per_9_delta_vs_league_30d``. Defaults to 30 per AGE-204.

    Notes
    -----
    Each normalized row enforces ``captured_at <= commence_time``.
    Games whose feed/live payload lacks a ``Home Plate`` official (or
    whose fresh HTTP fetch fails) emit a sentinel row with
    ``umpire_status="missing_umpire_source"`` so coverage checks can
    flag the gap without dropping the slate row.
    """

    if history_days < 1:
        raise ValueError("history_days must be at least 1")

    output_root = Path(output_dir)
    run_started_at = now().astimezone(UTC)
    run_id = _path_timestamp(run_started_at)
    games_path = _latest_mlb_games_path(output_dir=output_root, target_date=target_date)
    games = _load_games(games_path)

    assignments: list[UmpireAssignmentRecord] = []
    raw_paths: list[Path] = []

    for game in games:
        persisted_path = _latest_persisted_feed_live_path(
            output_dir=output_root,
            target_date=target_date,
            game_pk=game.game_pk,
        )
        payload: dict[str, Any] | None = None
        fetch_captured_at = now().astimezone(UTC)
        captured_at: datetime | None = None
        if persisted_path is not None:
            persisted_captured_at = _captured_at_from_feed_live_path(persisted_path)
            if persisted_captured_at is None:
                assignments.append(
                    _make_missing_assignment(
                        game=game,
                        captured_at=min(fetch_captured_at, game.commence_time),
                        error_message=(
                            "persisted feed/live path did not include a parseable "
                            f"captured_at timestamp: {persisted_path}"
                        ),
                    )
                )
                continue
            if persisted_captured_at > game.commence_time:
                assignments.append(
                    _make_missing_assignment(
                        game=game,
                        captured_at=game.commence_time,
                        error_message=(
                            "latest persisted feed/live payload was captured after "
                            "commence_time and cannot be used as a pregame umpire "
                            f"source: {persisted_path}"
                        ),
                    )
                )
                continue
            try:
                payload = json.loads(persisted_path.read_text(encoding="utf-8"))
                captured_at = persisted_captured_at
            except (OSError, json.JSONDecodeError) as exc:
                assignments.append(
                    _make_missing_assignment(
                        game=game,
                        captured_at=min(fetch_captured_at, game.commence_time),
                        error_message=(
                            f"{type(exc).__name__}: failed to read persisted feed/live at "
                            f"{persisted_path}"
                        ),
                    )
                )
                continue

        if payload is None:
            if fetch_captured_at > game.commence_time:
                assignments.append(
                    _make_missing_assignment(
                        game=game,
                        captured_at=game.commence_time,
                        error_message=(
                            "feed/live fetch skipped after commence_time; no "
                            "timestamp-valid pregame umpire source was available"
                        ),
                    )
                )
                continue
            if client is None:
                client = MLBStatsAPIClient()
            source_url = build_feed_live_url(game.game_pk)
            try:
                payload = client.fetch_json(source_url)
            except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                assignments.append(
                    _make_missing_assignment(
                        game=game,
                        captured_at=min(fetch_captured_at, game.commence_time),
                        error_message=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            captured_at = fetch_captured_at

        if captured_at is None:
            captured_at = min(fetch_captured_at, game.commence_time)
        assignment = _make_assignment_from_payload(
            game=game, payload=payload, captured_at=captured_at
        )
        assignments.append(assignment)

        raw_path = (
            output_root
            / "raw"
            / "umpire"
            / f"date={target_date.isoformat()}"
            / f"game_pk={game.game_pk}"
            / f"captured_at={_path_timestamp(captured_at)}.json"
        )
        raw_payload = {
            "game_pk": game.game_pk,
            "captured_at": format_utc_timestamp(captured_at),
            "source": UMPIRE_SOURCE_MLB_STATS_API_FEED_LIVE,
            "source_feed_live_path": (
                str(persisted_path) if persisted_path is not None else None
            ),
            "officials": normalize_feed_live_officials_payload(payload) or [],
            "umpire_id": assignment.umpire_id,
            "umpire_name": assignment.umpire_name,
            "umpire_status": assignment.umpire_status,
        }
        _write_json(raw_path, raw_payload)
        raw_paths.append(raw_path)

    umpire_game_pks_by_umpire = _load_prior_umpire_game_pks_by_umpire(
        output_dir=output_root,
        target_date=target_date,
        history_days=history_days,
    )
    pitch_aggregates = _load_prior_pitch_aggregates(
        output_dir=output_root,
        target_date=target_date,
        history_days=history_days,
    )

    history_end_date = target_date - timedelta(days=1)
    history_start_date = target_date - timedelta(days=history_days)
    snapshots: list[UmpireSnapshotRecord] = []
    ok_count = 0
    missing_source_count = 0
    for assignment in assignments:
        if assignment.umpire_status == UMPIRE_STATUS_OK and assignment.umpire_id is not None:
            called_strike_rate, k_per_9_delta = compute_rolling_umpire_metrics(
                umpire_id=assignment.umpire_id,
                umpire_game_pks=umpire_game_pks_by_umpire.get(assignment.umpire_id),
                pitch_aggregates=pitch_aggregates,
            )
            ok_count += 1
        else:
            called_strike_rate = None
            k_per_9_delta = None
            missing_source_count += 1
        snapshots.append(
            UmpireSnapshotRecord(
                umpire_snapshot_id=_snapshot_id(
                    game_pk=assignment.game_pk, captured_at=assignment.captured_at
                ),
                official_date=assignment.official_date,
                game_pk=assignment.game_pk,
                commence_time=assignment.commence_time,
                captured_at=assignment.captured_at,
                umpire_source=assignment.umpire_source,
                umpire_status=assignment.umpire_status,
                umpire_id=assignment.umpire_id,
                umpire_name=assignment.umpire_name,
                error_message=assignment.error_message,
                history_start_date=history_start_date,
                history_end_date=history_end_date,
                ump_called_strike_rate_30d=called_strike_rate,
                ump_k_per_9_delta_vs_league_30d=k_per_9_delta,
            )
        )

    for snapshot in snapshots:
        if snapshot.captured_at > snapshot.commence_time:
            raise AssertionError(
                "umpire snapshot captured_at must be <= commence_time "
                f"(game_pk={snapshot.game_pk})"
            )

    normalized_root = (
        output_root
        / "normalized"
        / "umpire"
        / f"date={target_date.isoformat()}"
        / f"run={run_id}"
    )
    umpire_snapshots_path = normalized_root / "umpire_snapshots.jsonl"
    _write_jsonl(umpire_snapshots_path, snapshots)

    return UmpireIngestResult(
        target_date=target_date,
        run_id=run_id,
        mlb_games_path=games_path,
        umpire_snapshots_path=umpire_snapshots_path,
        raw_snapshot_paths=tuple(raw_paths),
        snapshot_count=len(snapshots),
        ok_snapshot_count=ok_count,
        missing_source_count=missing_source_count,
        history_start_date=history_start_date,
        history_end_date=history_end_date,
    )


def load_latest_umpire_snapshots_for_date(
    *,
    output_dir: Path | str,
    target_date: date,
) -> dict[int, UmpireSnapshotRecord]:
    """Return the most recent pregame umpire snapshots for ``target_date``.

    Keys by ``game_pk`` so the statcast feature builder can join one
    snapshot per game. Missing directories return an empty dict so
    downstream code falls back to ``missing_umpire_source`` sentinels
    without crashing on fresh slates that haven't been ingested yet.
    """

    normalized_root = (
        Path(output_dir)
        / "normalized"
        / "umpire"
        / f"date={target_date.isoformat()}"
    )
    if not normalized_root.exists():
        return {}
    run_dirs = sorted(path for path in normalized_root.glob("run=*") if path.is_dir())
    if not run_dirs:
        return {}
    snapshots_path = run_dirs[-1] / "umpire_snapshots.jsonl"
    if not snapshots_path.exists():
        return {}

    records: dict[int, UmpireSnapshotRecord] = {}
    for line in snapshots_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        records[row["game_pk"]] = UmpireSnapshotRecord(
            umpire_snapshot_id=row["umpire_snapshot_id"],
            official_date=row["official_date"],
            game_pk=row["game_pk"],
            commence_time=parse_api_datetime(row["commence_time"]),
            captured_at=parse_api_datetime(row["captured_at"]),
            umpire_source=row.get("umpire_source"),
            umpire_status=row["umpire_status"],
            umpire_id=row.get("umpire_id"),
            umpire_name=row.get("umpire_name"),
            error_message=row.get("error_message"),
            history_start_date=date.fromisoformat(row["history_start_date"]),
            history_end_date=date.fromisoformat(row["history_end_date"]),
            ump_called_strike_rate_30d=row.get("ump_called_strike_rate_30d"),
            ump_k_per_9_delta_vs_league_30d=row.get("ump_k_per_9_delta_vs_league_30d"),
        )
    return records
