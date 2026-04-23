"""Idempotent multi-date backfill orchestration for ingest sources.

This module backs the ``backfill-historical`` CLI. It walks the requested
date window once, and for each date asks each selected source whether the
latest normalized run on disk already has every required artifact. When the
answer is yes and ``force`` is not set, the source is skipped and the
existing artifacts are preserved untouched. When the answer is no the
matching ``ingest_*_for_date`` helper is invoked. Source-level exceptions
are captured per-date so a transient odds-history gap or a single
Statcast pull failure cannot abort the rest of the sweep, matching the
issue requirement that the odds backfill stay best-effort.

Every sweep also writes a manifest under
``data/normalized/backfill/run=<RUN_ID>/backfill_manifest.json`` that
records the per-date outcome, status, and any captured error so the next
run (or the next ``check-data-alignment`` invocation) can audit which
dates still need attention.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import json
import os
from pathlib import Path

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


SOURCE_MLB_METADATA = "mlb-metadata"
SOURCE_WEATHER = "weather"
SOURCE_UMPIRE = "umpire"
SOURCE_ODDS_API = "odds-api"
SOURCE_STATCAST_FEATURES = "statcast-features"

# Umpire must run after MLB metadata (depends on games.jsonl and the
# persisted feed/live payloads it writes) and before statcast-features
# (which joins umpire snapshots into game_context). Weather is
# independent of umpire so either order between them works; umpire
# follows weather here to keep the per-date sequence stable for
# resume-aware manifests.
ALL_SOURCES: tuple[str, ...] = (
    SOURCE_MLB_METADATA,
    SOURCE_WEATHER,
    SOURCE_UMPIRE,
    SOURCE_ODDS_API,
    SOURCE_STATCAST_FEATURES,
)

# Required normalized artifacts that mark a per-date run as "complete" for
# resume-skip detection. Keys are the source identifiers passed via
# ``--sources``; the values mirror what ``check-data-alignment`` reads.
REQUIRED_ARTIFACT_FILES: dict[str, tuple[str, ...]] = {
    SOURCE_MLB_METADATA: (
        "games.jsonl",
        "probable_starters.jsonl",
        "lineup_snapshots.jsonl",
    ),
    SOURCE_WEATHER: ("weather_snapshots.jsonl",),
    SOURCE_UMPIRE: ("umpire_snapshots.jsonl",),
    SOURCE_ODDS_API: (
        "event_game_mappings.jsonl",
        "prop_line_snapshots.jsonl",
    ),
    SOURCE_STATCAST_FEATURES: (
        "pull_manifest.jsonl",
        "pitch_level_base.jsonl",
        "pitcher_daily_features.jsonl",
        "lineup_daily_features.jsonl",
        "game_context_features.jsonl",
    ),
}

# Maps source identifier -> normalized data root subdirectory under
# ``data/normalized``. Keep aligned with the ingest modules' write paths.
NORMALIZED_ROOT_BY_SOURCE: dict[str, str] = {
    SOURCE_MLB_METADATA: "mlb_stats_api",
    SOURCE_WEATHER: "weather",
    SOURCE_UMPIRE: "umpire",
    SOURCE_ODDS_API: "the_odds_api",
    SOURCE_STATCAST_FEATURES: "statcast_search",
}

STATUS_INGESTED = "ingested"
STATUS_SKIPPED_RESUME = "skipped_resume"
STATUS_FAILED = "failed"


@dataclass(frozen=True)
class BackfillSourceOutcome:
    """One source's outcome for one date inside a backfill sweep."""

    source: str
    status: str  # STATUS_INGESTED | STATUS_SKIPPED_RESUME | STATUS_FAILED
    run_id: str | None
    error_type: str | None
    error_message: str | None


@dataclass(frozen=True)
class BackfillDateOutcome:
    """All source outcomes for one date in a backfill sweep."""

    target_date: date
    sources: tuple[BackfillSourceOutcome, ...]


@dataclass(frozen=True)
class BackfillResult:
    """Aggregate outcome for one ``backfill-historical`` invocation."""

    start_date: date
    end_date: date
    sources: tuple[str, ...]
    history_days: int
    force: bool
    run_id: str
    manifest_path: Path
    dates: tuple[BackfillDateOutcome, ...]

    @property
    def date_count(self) -> int:
        return len(self.dates)

    @property
    def ingested_count(self) -> int:
        return sum(
            1
            for date_outcome in self.dates
            for source_outcome in date_outcome.sources
            if source_outcome.status == STATUS_INGESTED
        )

    @property
    def skipped_count(self) -> int:
        return sum(
            1
            for date_outcome in self.dates
            for source_outcome in date_outcome.sources
            if source_outcome.status == STATUS_SKIPPED_RESUME
        )

    @property
    def failed_count(self) -> int:
        return sum(
            1
            for date_outcome in self.dates
            for source_outcome in date_outcome.sources
            if source_outcome.status == STATUS_FAILED
        )

    @property
    def all_succeeded(self) -> bool:
        return self.failed_count == 0


def iter_backfill_dates(start_date: date, end_date: date) -> list[date]:
    """Return the inclusive calendar-date range to walk during a backfill."""
    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date")
    day_count = (end_date - start_date).days + 1
    return [start_date + timedelta(days=offset) for offset in range(day_count)]


def is_source_complete(
    output_dir: Path | str,
    *,
    source: str,
    target_date: date,
) -> bool:
    """Return True when the latest normalized run for one date has every required file.

    Mirrors the same "latest run wins" rule used by
    ``data_alignment._latest_run_dir_for_date`` so a backfill resume only
    skips a date when the artifacts ``check-data-alignment`` would inspect
    are already on disk.
    """
    if source not in NORMALIZED_ROOT_BY_SOURCE:
        raise ValueError(
            f"unknown backfill source: {source!r}. "
            f"Valid sources: {list(ALL_SOURCES)}."
        )
    normalized_root_name = NORMALIZED_ROOT_BY_SOURCE[source]
    required_files = REQUIRED_ARTIFACT_FILES[source]
    date_root = (
        Path(output_dir)
        / "normalized"
        / normalized_root_name
        / f"date={target_date.isoformat()}"
    )
    if not date_root.exists():
        return False
    run_dirs = sorted(p for p in date_root.glob("run=*") if p.is_dir())
    for run_dir in reversed(run_dirs):
        if all(run_dir.joinpath(name).exists() for name in required_files):
            return True
    return False


def normalize_sources(sources: Sequence[str]) -> tuple[str, ...]:
    """Validate the requested source list and return it deduplicated in order."""
    if not sources:
        raise ValueError("at least one source must be selected")
    invalid = sorted({src for src in sources if src not in ALL_SOURCES})
    if invalid:
        raise ValueError(
            f"unknown backfill source(s): {invalid}. "
            f"Valid sources: {list(ALL_SOURCES)}."
        )
    seen: set[str] = set()
    ordered: list[str] = []
    for src in sources:
        if src not in seen:
            seen.add(src)
            ordered.append(src)
    return tuple(ordered)


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _path_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    """Stream the manifest through ``<name>.tmp`` so partial writes can't survive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _date_outcome_to_dict(outcome: BackfillDateOutcome) -> dict[str, object]:
    return {
        "target_date": outcome.target_date.isoformat(),
        "sources": [
            {
                "source": source.source,
                "status": source.status,
                "run_id": source.run_id,
                "error_type": source.error_type,
                "error_message": source.error_message,
            }
            for source in outcome.sources
        ],
    }


def backfill_historical(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str = "data",
    sources: Sequence[str] = ALL_SOURCES,
    force: bool = False,
    history_days: int = DEFAULT_HISTORY_DAYS,
    odds_api_key: str | None = None,
    mlb_metadata_runner: Callable[
        ..., MLBMetadataIngestResult
    ] = ingest_mlb_metadata_for_date,
    weather_runner: Callable[
        ..., WeatherIngestResult
    ] = ingest_weather_for_date,
    umpire_runner: Callable[
        ..., UmpireIngestResult
    ] = ingest_umpire_for_date,
    odds_api_runner: Callable[
        ..., OddsAPIIngestResult
    ] = ingest_odds_api_pitcher_lines_for_date,
    statcast_features_runner: Callable[
        ..., StatcastFeatureIngestResult
    ] = ingest_statcast_features_for_date,
    now: Callable[[], datetime] = _utc_now,
) -> BackfillResult:
    """Iterate ``ingest-*`` calls across a date window with idempotent resume.

    For each ``target_date`` in ``[start_date, end_date]`` and each source in
    ``sources`` (in the declared order), the helper:

    * skips the source when ``force`` is False and the latest normalized
      run already contains every file in ``REQUIRED_ARTIFACT_FILES``,
    * otherwise calls the matching ingest runner and records the returned
      ``run_id``,
    * captures any ``Exception`` raised by the runner so the rest of the
      sweep continues — odds-api history is sparse by design and one bad
      Statcast pull should not strand the rest of the season.

    KeyboardInterrupt and other ``BaseException`` subclasses propagate so a
    user-initiated abort still stops the run immediately.
    """
    selected_sources = normalize_sources(sources)
    requested_dates = iter_backfill_dates(start_date, end_date)

    output_root = Path(output_dir)
    run_started_at = now().astimezone(UTC)
    run_id = _path_timestamp(run_started_at)

    runners: dict[str, Callable[[date], object]] = {
        SOURCE_MLB_METADATA: lambda target_date: mlb_metadata_runner(
            target_date=target_date,
            output_dir=output_root,
        ),
        SOURCE_WEATHER: lambda target_date: weather_runner(
            target_date=target_date,
            output_dir=output_root,
        ),
        SOURCE_UMPIRE: lambda target_date: umpire_runner(
            target_date=target_date,
            output_dir=output_root,
        ),
        SOURCE_ODDS_API: lambda target_date: odds_api_runner(
            target_date=target_date,
            output_dir=output_root,
            api_key=odds_api_key,
        ),
        SOURCE_STATCAST_FEATURES: lambda target_date: statcast_features_runner(
            target_date=target_date,
            output_dir=output_root,
            history_days=history_days,
        ),
    }

    date_outcomes: list[BackfillDateOutcome] = []
    for target_date in requested_dates:
        per_source_outcomes: list[BackfillSourceOutcome] = []
        for source in selected_sources:
            if not force and is_source_complete(
                output_root, source=source, target_date=target_date
            ):
                per_source_outcomes.append(
                    BackfillSourceOutcome(
                        source=source,
                        status=STATUS_SKIPPED_RESUME,
                        run_id=None,
                        error_type=None,
                        error_message=None,
                    )
                )
                continue
            try:
                result = runners[source](target_date)
            except Exception as exc:
                per_source_outcomes.append(
                    BackfillSourceOutcome(
                        source=source,
                        status=STATUS_FAILED,
                        run_id=None,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                continue
            per_source_outcomes.append(
                BackfillSourceOutcome(
                    source=source,
                    status=STATUS_INGESTED,
                    run_id=getattr(result, "run_id", None),
                    error_type=None,
                    error_message=None,
                )
            )
        date_outcomes.append(
            BackfillDateOutcome(
                target_date=target_date,
                sources=tuple(per_source_outcomes),
            )
        )

    manifest_path = (
        output_root
        / "normalized"
        / "backfill"
        / f"run={run_id}"
        / "backfill_manifest.json"
    )

    manifest_payload = {
        "run_id": run_id,
        "run_started_at": run_started_at.isoformat(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "force": force,
        "sources": list(selected_sources),
        "history_days": history_days,
        "dates": [_date_outcome_to_dict(outcome) for outcome in date_outcomes],
    }
    _atomic_write_json(manifest_path, manifest_payload)

    return BackfillResult(
        start_date=start_date,
        end_date=end_date,
        sources=selected_sources,
        history_days=history_days,
        force=force,
        run_id=run_id,
        manifest_path=manifest_path,
        dates=tuple(date_outcomes),
    )
