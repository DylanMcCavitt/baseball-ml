"""Tests for the data-alignment coverage helper and CLI."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from mlb_props_stack.cli import main
from mlb_props_stack.data_alignment import (
    ArtifactCounts,
    DEFAULT_COVERAGE_THRESHOLD,
    DataAlignmentReport,
    build_date_coverage_rows,
    check_data_alignment,
    collect_artifact_counts_for_date,
    render_data_alignment_summary,
)


def _counts(
    target_date: date,
    *,
    games: int = 10,
    probable_starters: int = 20,
    lineup_snapshots: int = 20,
    prop_line_snapshots: int = 100,
    prop_line_pitcher_coverage: int = 18,
    pitcher_daily_features: int = 18,
    lineup_daily_features: int = 18,
    game_context_features: int = 18,
    weather_snapshots: int = 10,
    weather_ok_snapshots: int = 9,
    weather_roof_closed_snapshots: int = 1,
    training_rows: int = 18,
    calibrated_probabilities: int = 18,
    starter_outcomes: int = 18,
) -> ArtifactCounts:
    return ArtifactCounts(
        official_date=target_date,
        games=games,
        probable_starters=probable_starters,
        lineup_snapshots=lineup_snapshots,
        prop_line_snapshots=prop_line_snapshots,
        prop_line_pitcher_coverage=prop_line_pitcher_coverage,
        pitcher_daily_features=pitcher_daily_features,
        lineup_daily_features=lineup_daily_features,
        game_context_features=game_context_features,
        weather_snapshots=weather_snapshots,
        weather_ok_snapshots=weather_ok_snapshots,
        weather_roof_closed_snapshots=weather_roof_closed_snapshots,
        training_rows=training_rows,
        calibrated_probabilities=calibrated_probabilities,
        starter_outcomes=starter_outcomes,
    )


def test_build_date_coverage_rows_marks_well_covered_date_as_passing():
    counts = _counts(date(2026, 4, 18))

    rows = build_date_coverage_rows([counts])

    assert len(rows) == 1
    row = rows[0]
    assert row.below_threshold is False
    assert row.failing_artifacts == ()
    assert row.feature_coverage == pytest.approx(18 / 20)
    assert row.outcome_coverage == pytest.approx(18 / 18)
    assert row.odds_coverage == pytest.approx(18 / 20)


def test_build_date_coverage_rows_flags_missing_features_and_outcomes():
    counts = _counts(
        date(2026, 4, 19),
        pitcher_daily_features=2,
        starter_outcomes=3,
        training_rows=20,
        prop_line_pitcher_coverage=0,
        prop_line_snapshots=0,
    )

    rows = build_date_coverage_rows([counts], threshold=0.5)

    assert len(rows) == 1
    row = rows[0]
    assert row.below_threshold is True
    assert set(row.failing_artifacts) == {"feature", "outcome", "odds"}
    assert row.feature_coverage == pytest.approx(2 / 20)
    assert row.outcome_coverage == pytest.approx(3 / 20)
    assert row.odds_coverage == pytest.approx(0.0)


def test_build_date_coverage_rows_treats_missing_denominator_as_failure():
    counts = _counts(
        date(2026, 4, 20),
        probable_starters=0,
        pitcher_daily_features=0,
        training_rows=0,
        starter_outcomes=0,
        prop_line_pitcher_coverage=0,
        prop_line_snapshots=0,
    )

    rows = build_date_coverage_rows([counts])

    row = rows[0]
    assert row.feature_coverage is None
    assert row.outcome_coverage is None
    assert row.odds_coverage is None
    assert set(row.failing_artifacts) == {"feature", "outcome", "odds"}
    assert row.below_threshold is True


def test_build_date_coverage_rows_respects_custom_threshold():
    counts = _counts(
        date(2026, 4, 21),
        pitcher_daily_features=12,
        prop_line_pitcher_coverage=12,
        starter_outcomes=12,
        training_rows=20,
        probable_starters=20,
    )

    passing_at_default = build_date_coverage_rows([counts], threshold=0.5)
    assert passing_at_default[0].below_threshold is False

    failing_at_strict = build_date_coverage_rows([counts], threshold=0.8)
    assert failing_at_strict[0].below_threshold is True
    assert set(failing_at_strict[0].failing_artifacts) == {"feature", "outcome", "odds"}


def test_build_date_coverage_rows_rejects_invalid_threshold():
    counts = _counts(date(2026, 4, 22))

    with pytest.raises(ValueError):
        build_date_coverage_rows([counts], threshold=-0.1)
    with pytest.raises(ValueError):
        build_date_coverage_rows([counts], threshold=1.1)


def test_data_alignment_report_reports_failing_dates():
    rows = build_date_coverage_rows(
        [
            _counts(date(2026, 4, 18)),
            _counts(
                date(2026, 4, 19),
                pitcher_daily_features=0,
                starter_outcomes=0,
                prop_line_pitcher_coverage=0,
                prop_line_snapshots=0,
            ),
        ]
    )

    report = DataAlignmentReport(
        start_date=date(2026, 4, 18),
        end_date=date(2026, 4, 19),
        threshold=DEFAULT_COVERAGE_THRESHOLD,
        rows=tuple(rows),
    )

    assert report.passed is False
    assert report.failing_dates == (date(2026, 4, 19),)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _seed_passing_fixture(output_dir: Path, *, target_date: date) -> None:
    iso = target_date.isoformat()
    mlb_run = (
        output_dir / "normalized" / "mlb_stats_api" / f"date={iso}" / "run=20260422T180000Z"
    )
    _write_jsonl(
        mlb_run / "games.jsonl",
        [{"game_pk": 1, "official_date": iso}, {"game_pk": 2, "official_date": iso}],
    )
    _write_jsonl(
        mlb_run / "probable_starters.jsonl",
        [
            {"pitcher_id": 101, "official_date": iso},
            {"pitcher_id": 102, "official_date": iso},
        ],
    )
    _write_jsonl(
        mlb_run / "lineup_snapshots.jsonl",
        [{"lineup_snapshot_id": "a", "official_date": iso}],
    )

    odds_run = (
        output_dir / "normalized" / "the_odds_api" / f"date={iso}" / "run=20260422T180500Z"
    )
    _write_jsonl(
        odds_run / "prop_line_snapshots.jsonl",
        [
            {
                "line_snapshot_id": "s-1",
                "official_date": iso,
                "pitcher_mlb_id": 101,
            },
            {
                "line_snapshot_id": "s-2",
                "official_date": iso,
                "pitcher_mlb_id": 102,
            },
            {
                "line_snapshot_id": "s-3",
                "official_date": iso,
                "pitcher_mlb_id": None,
            },
        ],
    )

    feature_run = (
        output_dir
        / "normalized"
        / "statcast_search"
        / f"date={iso}"
        / "run=20260422T190000Z"
    )
    _write_jsonl(
        feature_run / "pitcher_daily_features.jsonl",
        [
            {"pitcher_id": 101, "official_date": iso},
            {"pitcher_id": 102, "official_date": iso},
        ],
    )
    _write_jsonl(
        feature_run / "lineup_daily_features.jsonl",
        [{"lineup_feature_row_id": "lf-1", "official_date": iso}],
    )
    _write_jsonl(
        feature_run / "game_context_features.jsonl",
        [{"game_context_feature_row_id": "gc-1", "official_date": iso}],
    )

    baseline_run = (
        output_dir
        / "normalized"
        / "starter_strikeout_baseline"
        / f"start={iso}_end={iso}"
        / "run=20260422T200000Z"
    )
    _write_jsonl(
        baseline_run / "training_dataset.jsonl",
        [
            {"training_row_id": "t-1", "official_date": iso, "pitcher_id": 101},
            {"training_row_id": "t-2", "official_date": iso, "pitcher_id": 102},
        ],
    )
    _write_jsonl(
        baseline_run / "starter_outcomes.jsonl",
        [
            {"outcome_id": "o-1", "official_date": iso, "pitcher_id": 101},
            {"outcome_id": "o-2", "official_date": iso, "pitcher_id": 102},
        ],
    )
    _write_jsonl(
        baseline_run / "raw_vs_calibrated_probabilities.jsonl",
        [
            {"training_row_id": "t-1", "official_date": iso},
            {"training_row_id": "t-2", "official_date": iso},
        ],
    )


def test_collect_artifact_counts_reads_latest_runs(tmp_path):
    target_date = date(2026, 4, 18)
    _seed_passing_fixture(tmp_path, target_date=target_date)

    counts = collect_artifact_counts_for_date(tmp_path, target_date=target_date)

    assert counts.official_date == target_date
    assert counts.games == 2
    assert counts.probable_starters == 2
    assert counts.lineup_snapshots == 1
    assert counts.prop_line_snapshots == 3
    assert counts.prop_line_pitcher_coverage == 2
    assert counts.pitcher_daily_features == 2
    assert counts.lineup_daily_features == 1
    assert counts.game_context_features == 1
    assert counts.training_rows == 2
    assert counts.starter_outcomes == 2
    assert counts.calibrated_probabilities == 2
    assert counts.weather_snapshots == 0
    assert counts.weather_ok_snapshots == 0
    assert counts.weather_roof_closed_snapshots == 0


def test_collect_artifact_counts_reads_weather_statuses(tmp_path):
    target_date = date(2026, 4, 18)
    _seed_passing_fixture(tmp_path, target_date=target_date)
    iso = target_date.isoformat()
    weather_run = (
        tmp_path / "normalized" / "weather" / f"date={iso}" / "run=20260422T180500Z"
    )
    _write_jsonl(
        weather_run / "weather_snapshots.jsonl",
        [
            {"game_pk": 1, "official_date": iso, "weather_status": "ok"},
            {"game_pk": 2, "official_date": iso, "weather_status": "roof_closed"},
            {"game_pk": 3, "official_date": iso, "weather_status": "missing_venue_metadata"},
        ],
    )

    counts = collect_artifact_counts_for_date(tmp_path, target_date=target_date)

    assert counts.weather_snapshots == 3
    assert counts.weather_ok_snapshots == 1
    assert counts.weather_roof_closed_snapshots == 1


def test_collect_artifact_counts_returns_zeros_for_missing_artifacts(tmp_path):
    counts = collect_artifact_counts_for_date(tmp_path, target_date=date(2026, 4, 19))

    assert counts.games == 0
    assert counts.probable_starters == 0
    assert counts.lineup_snapshots == 0
    assert counts.prop_line_snapshots == 0
    assert counts.prop_line_pitcher_coverage == 0
    assert counts.pitcher_daily_features == 0
    assert counts.training_rows == 0
    assert counts.calibrated_probabilities == 0
    assert counts.starter_outcomes == 0
    assert counts.weather_snapshots == 0
    assert counts.weather_ok_snapshots == 0
    assert counts.weather_roof_closed_snapshots == 0


def test_check_data_alignment_returns_failing_report_when_gaps_present(tmp_path):
    target_date = date(2026, 4, 18)
    _seed_passing_fixture(tmp_path, target_date=target_date)

    report = check_data_alignment(
        start_date=target_date,
        end_date=date(2026, 4, 19),
        output_dir=tmp_path,
    )

    assert report.start_date == date(2026, 4, 18)
    assert report.end_date == date(2026, 4, 19)
    assert len(report.rows) == 2
    assert report.rows[0].below_threshold is False
    assert report.rows[1].below_threshold is True
    assert report.failing_dates == (date(2026, 4, 19),)
    assert report.passed is False


def test_check_data_alignment_rejects_inverted_date_window(tmp_path):
    with pytest.raises(ValueError):
        check_data_alignment(
            start_date=date(2026, 4, 19),
            end_date=date(2026, 4, 18),
            output_dir=tmp_path,
        )


def test_render_data_alignment_summary_lists_failing_dates():
    rows = build_date_coverage_rows(
        [
            _counts(date(2026, 4, 18)),
            _counts(
                date(2026, 4, 19),
                pitcher_daily_features=0,
                starter_outcomes=0,
                prop_line_pitcher_coverage=0,
                prop_line_snapshots=0,
            ),
        ]
    )
    report = DataAlignmentReport(
        start_date=date(2026, 4, 18),
        end_date=date(2026, 4, 19),
        threshold=DEFAULT_COVERAGE_THRESHOLD,
        rows=tuple(rows),
    )

    summary = render_data_alignment_summary(report)

    assert "Data alignment report 2026-04-18 -> 2026-04-19" in summary
    assert "threshold=50%" in summary
    assert "2026-04-18" in summary
    assert "2026-04-19" in summary
    assert "Failing dates" in summary
    assert "failing=feature, outcome, odds" in summary
    assert "weather_ok" in summary
    assert "weather_roof" in summary
    assert "wx_cov" in summary
    assert "ok" in summary
    assert "FAIL" in summary


def test_render_data_alignment_summary_reports_passing_window():
    rows = build_date_coverage_rows([_counts(date(2026, 4, 18))])
    report = DataAlignmentReport(
        start_date=date(2026, 4, 18),
        end_date=date(2026, 4, 18),
        threshold=DEFAULT_COVERAGE_THRESHOLD,
        rows=tuple(rows),
    )

    summary = render_data_alignment_summary(report)

    assert "All dates meet the 50% coverage threshold." in summary
    assert "FAIL" not in summary


def test_check_data_alignment_cli_exits_non_zero_when_gaps_present(tmp_path, capsys):
    target_date = date(2026, 4, 18)
    _seed_passing_fixture(tmp_path, target_date=target_date)

    exit_code = main(
        [
            "check-data-alignment",
            "--start-date",
            "2026-04-18",
            "--end-date",
            "2026-04-19",
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Data alignment report 2026-04-18 -> 2026-04-19" in output
    assert "Failing dates" in output


def test_check_data_alignment_cli_exits_zero_when_covered(tmp_path, capsys):
    target_date = date(2026, 4, 18)
    _seed_passing_fixture(tmp_path, target_date=target_date)

    exit_code = main(
        [
            "check-data-alignment",
            "--start-date",
            "2026-04-18",
            "--end-date",
            "2026-04-18",
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "All dates meet" in output
