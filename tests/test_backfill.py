"""Tests for the multi-date ``backfill_historical`` orchestration helper."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlb_props_stack.backfill import (
    ALL_SOURCES,
    REQUIRED_ARTIFACT_FILES,
    SOURCE_MLB_METADATA,
    SOURCE_ODDS_API,
    SOURCE_STATCAST_FEATURES,
    SOURCE_UMPIRE,
    SOURCE_WEATHER,
    BackfillResult,
    backfill_historical,
    is_source_complete,
    iter_backfill_dates,
    normalize_sources,
)
from mlb_props_stack.cli import main


def _seed_complete_run(
    output_dir: Path,
    *,
    source: str,
    target_date: date,
    run_id: str = "20260101T000000Z",
) -> Path:
    """Write empty placeholder files so ``is_source_complete`` returns True."""
    normalized_root = {
        SOURCE_MLB_METADATA: "mlb_stats_api",
        SOURCE_WEATHER: "weather",
        SOURCE_UMPIRE: "umpire",
        SOURCE_ODDS_API: "the_odds_api",
        SOURCE_STATCAST_FEATURES: "statcast_search",
    }[source]
    run_dir = (
        output_dir
        / "normalized"
        / normalized_root
        / f"date={target_date.isoformat()}"
        / f"run={run_id}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_ARTIFACT_FILES[source]:
        (run_dir / filename).write_text("", encoding="utf-8")
    return run_dir


@dataclass
class _RunnerSpy:
    """Captures the dates a fake ingest runner was invoked with."""

    name: str
    invocations: list[date] = field(default_factory=list)
    raise_for_dates: tuple[date, ...] = ()
    return_run_id: str = "fake-run-id"

    def __call__(self, **kwargs) -> SimpleNamespace:
        target_date = kwargs["target_date"]
        self.invocations.append(target_date)
        if target_date in self.raise_for_dates:
            raise RuntimeError(f"{self.name} failed for {target_date.isoformat()}")
        return SimpleNamespace(run_id=f"{self.return_run_id}-{target_date.isoformat()}")


def _fixed_now(stamp: str = "20260420T120000Z"):
    parsed = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    return lambda: parsed


def test_iter_backfill_dates_inclusive_calendar_window():
    dates = iter_backfill_dates(date(2026, 4, 18), date(2026, 4, 20))

    assert dates == [date(2026, 4, 18), date(2026, 4, 19), date(2026, 4, 20)]


def test_iter_backfill_dates_rejects_inverted_window():
    with pytest.raises(ValueError):
        iter_backfill_dates(date(2026, 4, 20), date(2026, 4, 18))


def test_normalize_sources_dedupes_and_preserves_order():
    selected = normalize_sources(
        [SOURCE_STATCAST_FEATURES, SOURCE_MLB_METADATA, SOURCE_STATCAST_FEATURES]
    )

    assert selected == (SOURCE_STATCAST_FEATURES, SOURCE_MLB_METADATA)


def test_normalize_sources_rejects_unknown_source():
    with pytest.raises(ValueError):
        normalize_sources([SOURCE_MLB_METADATA, "fangraphs"])


def test_normalize_sources_rejects_empty_list():
    with pytest.raises(ValueError):
        normalize_sources([])


def test_is_source_complete_requires_every_artifact(tmp_path):
    target = date(2026, 4, 18)
    run_dir = _seed_complete_run(tmp_path, source=SOURCE_MLB_METADATA, target_date=target)

    assert is_source_complete(tmp_path, source=SOURCE_MLB_METADATA, target_date=target)

    # Remove one required artifact and confirm the helper flips to False.
    (run_dir / "lineup_snapshots.jsonl").unlink()

    assert not is_source_complete(
        tmp_path, source=SOURCE_MLB_METADATA, target_date=target
    )


def test_is_source_complete_falls_back_to_older_run(tmp_path):
    target = date(2026, 4, 18)
    older_run = _seed_complete_run(
        tmp_path,
        source=SOURCE_MLB_METADATA,
        target_date=target,
        run_id="20260101T000000Z",
    )
    # A newer run that was started but never finished — only one of the
    # required files exists. Resume should still find the older complete run.
    newer_run = (
        tmp_path
        / "normalized"
        / "mlb_stats_api"
        / f"date={target.isoformat()}"
        / "run=20260102T000000Z"
    )
    newer_run.mkdir(parents=True, exist_ok=True)
    (newer_run / "games.jsonl").write_text("", encoding="utf-8")

    assert is_source_complete(
        tmp_path, source=SOURCE_MLB_METADATA, target_date=target
    )
    assert older_run.exists()


def test_backfill_historical_invokes_each_runner_per_date(tmp_path):
    mlb = _RunnerSpy(name="mlb")
    weather = _RunnerSpy(name="weather")
    umpire = _RunnerSpy(name="umpire")
    odds = _RunnerSpy(name="odds")
    statcast = _RunnerSpy(name="statcast")

    result = backfill_historical(
        start_date=date(2026, 4, 18),
        end_date=date(2026, 4, 19),
        output_dir=tmp_path,
        sources=ALL_SOURCES,
        mlb_metadata_runner=mlb,
        weather_runner=weather,
        umpire_runner=umpire,
        odds_api_runner=odds,
        statcast_features_runner=statcast,
        now=_fixed_now(),
    )

    assert mlb.invocations == [date(2026, 4, 18), date(2026, 4, 19)]
    assert weather.invocations == [date(2026, 4, 18), date(2026, 4, 19)]
    assert umpire.invocations == [date(2026, 4, 18), date(2026, 4, 19)]
    assert odds.invocations == [date(2026, 4, 18), date(2026, 4, 19)]
    assert statcast.invocations == [date(2026, 4, 18), date(2026, 4, 19)]
    assert result.ingested_count == 10
    assert result.skipped_count == 0
    assert result.failed_count == 0
    assert result.all_succeeded
    assert result.manifest_path.exists()


def test_backfill_historical_resume_skips_dates_with_complete_artifacts(tmp_path):
    target = date(2026, 4, 18)
    for source in ALL_SOURCES:
        _seed_complete_run(tmp_path, source=source, target_date=target)

    mlb = _RunnerSpy(name="mlb")
    weather = _RunnerSpy(name="weather")
    umpire = _RunnerSpy(name="umpire")
    odds = _RunnerSpy(name="odds")
    statcast = _RunnerSpy(name="statcast")

    result = backfill_historical(
        start_date=target,
        end_date=target,
        output_dir=tmp_path,
        sources=ALL_SOURCES,
        mlb_metadata_runner=mlb,
        weather_runner=weather,
        umpire_runner=umpire,
        odds_api_runner=odds,
        statcast_features_runner=statcast,
        now=_fixed_now(),
    )

    assert mlb.invocations == []
    assert weather.invocations == []
    assert umpire.invocations == []
    assert odds.invocations == []
    assert statcast.invocations == []
    assert result.skipped_count == 5
    assert result.ingested_count == 0
    assert result.failed_count == 0


def test_backfill_historical_force_reingests_complete_dates(tmp_path):
    target = date(2026, 4, 18)
    for source in ALL_SOURCES:
        _seed_complete_run(tmp_path, source=source, target_date=target)

    mlb = _RunnerSpy(name="mlb")
    weather = _RunnerSpy(name="weather")
    umpire = _RunnerSpy(name="umpire")
    odds = _RunnerSpy(name="odds")
    statcast = _RunnerSpy(name="statcast")

    result = backfill_historical(
        start_date=target,
        end_date=target,
        output_dir=tmp_path,
        sources=ALL_SOURCES,
        force=True,
        mlb_metadata_runner=mlb,
        weather_runner=weather,
        umpire_runner=umpire,
        odds_api_runner=odds,
        statcast_features_runner=statcast,
        now=_fixed_now(),
    )

    assert mlb.invocations == [target]
    assert weather.invocations == [target]
    assert umpire.invocations == [target]
    assert odds.invocations == [target]
    assert statcast.invocations == [target]
    assert result.ingested_count == 5
    assert result.skipped_count == 0


def test_backfill_historical_resume_picks_up_only_missing_sources(tmp_path):
    target = date(2026, 4, 18)
    _seed_complete_run(tmp_path, source=SOURCE_MLB_METADATA, target_date=target)
    # Weather, Umpire, Odds, and Statcast are missing for this date and should still be ingested.
    mlb = _RunnerSpy(name="mlb")
    weather = _RunnerSpy(name="weather")
    umpire = _RunnerSpy(name="umpire")
    odds = _RunnerSpy(name="odds")
    statcast = _RunnerSpy(name="statcast")

    result = backfill_historical(
        start_date=target,
        end_date=target,
        output_dir=tmp_path,
        sources=ALL_SOURCES,
        mlb_metadata_runner=mlb,
        weather_runner=weather,
        umpire_runner=umpire,
        odds_api_runner=odds,
        statcast_features_runner=statcast,
        now=_fixed_now(),
    )

    assert mlb.invocations == []
    assert weather.invocations == [target]
    assert umpire.invocations == [target]
    assert odds.invocations == [target]
    assert statcast.invocations == [target]
    assert result.ingested_count == 4
    assert result.skipped_count == 1


def test_backfill_historical_keeps_running_when_one_source_raises(tmp_path):
    target = date(2026, 4, 18)
    mlb = _RunnerSpy(name="mlb")
    weather = _RunnerSpy(name="weather")
    umpire = _RunnerSpy(name="umpire")
    odds = _RunnerSpy(name="odds", raise_for_dates=(target,))
    statcast = _RunnerSpy(name="statcast")

    result = backfill_historical(
        start_date=target,
        end_date=target,
        output_dir=tmp_path,
        sources=ALL_SOURCES,
        mlb_metadata_runner=mlb,
        weather_runner=weather,
        umpire_runner=umpire,
        odds_api_runner=odds,
        statcast_features_runner=statcast,
        now=_fixed_now(),
    )

    statuses = {
        source.source: source.status
        for source in result.dates[0].sources
    }
    assert statuses[SOURCE_MLB_METADATA] == "ingested"
    assert statuses[SOURCE_WEATHER] == "ingested"
    assert statuses[SOURCE_UMPIRE] == "ingested"
    assert statuses[SOURCE_ODDS_API] == "failed"
    assert statuses[SOURCE_STATCAST_FEATURES] == "ingested"
    assert mlb.invocations == [target]
    assert weather.invocations == [target]
    assert umpire.invocations == [target]
    assert odds.invocations == [target]
    # The statcast runner must run even though odds raised on the same date.
    assert statcast.invocations == [target]
    assert result.failed_count == 1
    assert not result.all_succeeded


def test_backfill_historical_only_runs_selected_sources(tmp_path):
    target = date(2026, 4, 18)
    mlb = _RunnerSpy(name="mlb")
    weather = _RunnerSpy(name="weather")
    umpire = _RunnerSpy(name="umpire")
    odds = _RunnerSpy(name="odds")
    statcast = _RunnerSpy(name="statcast")

    result = backfill_historical(
        start_date=target,
        end_date=target,
        output_dir=tmp_path,
        sources=(SOURCE_MLB_METADATA, SOURCE_STATCAST_FEATURES),
        mlb_metadata_runner=mlb,
        weather_runner=weather,
        umpire_runner=umpire,
        odds_api_runner=odds,
        statcast_features_runner=statcast,
        now=_fixed_now(),
    )

    assert weather.invocations == []
    assert umpire.invocations == []
    assert odds.invocations == []
    assert mlb.invocations == [target]
    assert statcast.invocations == [target]
    assert tuple(result.sources) == (SOURCE_MLB_METADATA, SOURCE_STATCAST_FEATURES)


def test_backfill_historical_writes_manifest_with_per_date_outcomes(tmp_path):
    target = date(2026, 4, 18)
    _seed_complete_run(tmp_path, source=SOURCE_MLB_METADATA, target_date=target)
    mlb = _RunnerSpy(name="mlb")
    weather = _RunnerSpy(name="weather")
    umpire = _RunnerSpy(name="umpire")
    odds = _RunnerSpy(name="odds", raise_for_dates=(target,))
    statcast = _RunnerSpy(name="statcast")

    result = backfill_historical(
        start_date=target,
        end_date=target,
        output_dir=tmp_path,
        sources=ALL_SOURCES,
        mlb_metadata_runner=mlb,
        weather_runner=weather,
        umpire_runner=umpire,
        odds_api_runner=odds,
        statcast_features_runner=statcast,
        history_days=15,
        now=_fixed_now("20260420T120000Z"),
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "20260420T120000Z"
    assert manifest["start_date"] == target.isoformat()
    assert manifest["end_date"] == target.isoformat()
    assert manifest["force"] is False
    assert manifest["history_days"] == 15
    assert manifest["sources"] == list(ALL_SOURCES)
    assert len(manifest["dates"]) == 1
    by_source = {entry["source"]: entry for entry in manifest["dates"][0]["sources"]}
    assert by_source[SOURCE_MLB_METADATA]["status"] == "skipped_resume"
    assert by_source[SOURCE_WEATHER]["status"] == "ingested"
    assert by_source[SOURCE_UMPIRE]["status"] == "ingested"
    assert by_source[SOURCE_ODDS_API]["status"] == "failed"
    assert by_source[SOURCE_ODDS_API]["error_type"] == "RuntimeError"
    assert by_source[SOURCE_STATCAST_FEATURES]["status"] == "ingested"
    assert (
        by_source[SOURCE_STATCAST_FEATURES]["run_id"]
        == f"fake-run-id-{target.isoformat()}"
    )


def test_backfill_historical_passes_history_days_to_statcast_runner(tmp_path):
    captured: dict[str, object] = {}

    def fake_statcast(**kwargs) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(run_id="statcast-run")

    backfill_historical(
        start_date=date(2026, 4, 18),
        end_date=date(2026, 4, 18),
        output_dir=tmp_path,
        sources=(SOURCE_STATCAST_FEATURES,),
        history_days=42,
        statcast_features_runner=fake_statcast,
        now=_fixed_now(),
    )

    assert captured["history_days"] == 42
    assert captured["target_date"] == date(2026, 4, 18)


def test_backfill_historical_passes_target_date_and_output_dir_to_umpire_runner(tmp_path):
    captured: dict[str, object] = {}

    def fake_umpire(**kwargs) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(run_id="umpire-run")

    backfill_historical(
        start_date=date(2026, 4, 18),
        end_date=date(2026, 4, 18),
        output_dir=tmp_path,
        sources=(SOURCE_UMPIRE,),
        umpire_runner=fake_umpire,
        now=_fixed_now(),
    )

    assert captured["target_date"] == date(2026, 4, 18)
    assert captured["output_dir"] == tmp_path


def test_backfill_historical_passes_api_key_to_odds_runner(tmp_path):
    captured: dict[str, object] = {}

    def fake_odds(**kwargs) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(run_id="odds-run")

    backfill_historical(
        start_date=date(2026, 4, 18),
        end_date=date(2026, 4, 18),
        output_dir=tmp_path,
        sources=(SOURCE_ODDS_API,),
        odds_api_key="abc-123",
        odds_api_runner=fake_odds,
        now=_fixed_now(),
    )

    assert captured["api_key"] == "abc-123"


def test_backfill_historical_cli_summarizes_run(monkeypatch, tmp_path, capsys):
    fake_result = BackfillResult(
        start_date=date(2026, 4, 18),
        end_date=date(2026, 4, 19),
        sources=ALL_SOURCES,
        history_days=30,
        force=False,
        run_id="20260420T120000Z",
        manifest_path=tmp_path / "backfill_manifest.json",
        dates=(),
    )

    monkeypatch.setattr(
        "mlb_props_stack.cli.backfill_historical",
        lambda **_: fake_result,
    )

    exit_code = main(
        [
            "backfill-historical",
            "--start-date",
            "2026-04-18",
            "--end-date",
            "2026-04-19",
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Backfill historical complete for 2026-04-18 -> 2026-04-19" in output
    assert "force=false" in output
    assert "history_days=30" in output
    assert "manifest_path=" in output


def test_backfill_historical_cli_returns_nonzero_when_failures_recorded(
    monkeypatch, tmp_path, capsys
):
    from mlb_props_stack.backfill import BackfillDateOutcome, BackfillSourceOutcome

    fake_result = BackfillResult(
        start_date=date(2026, 4, 18),
        end_date=date(2026, 4, 18),
        sources=ALL_SOURCES,
        history_days=30,
        force=False,
        run_id="20260420T120000Z",
        manifest_path=tmp_path / "backfill_manifest.json",
        dates=(
            BackfillDateOutcome(
                target_date=date(2026, 4, 18),
                sources=(
                    BackfillSourceOutcome(
                        source=SOURCE_ODDS_API,
                        status="failed",
                        run_id=None,
                        error_type="RuntimeError",
                        error_message="odds-history gap",
                    ),
                ),
            ),
        ),
    )

    monkeypatch.setattr(
        "mlb_props_stack.cli.backfill_historical",
        lambda **_: fake_result,
    )

    exit_code = main(
        [
            "backfill-historical",
            "--start-date",
            "2026-04-18",
            "--end-date",
            "2026-04-18",
            "--output-dir",
            str(tmp_path),
            "--sources",
            ",".join(ALL_SOURCES),
            "--force",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "failed_outcomes=1" in output
    assert "odds-history gap" in output
