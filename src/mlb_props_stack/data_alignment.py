"""Coverage diagnostics across ingest, feature, and modeling artifacts.

This module backs the ``check-data-alignment`` CLI. It counts rows in each
per-date artifact the scoring stack depends on, derives coverage ratios, and
flags dates whose coverage falls below a configurable threshold so the root
cause of all-skipped backtest windows is obvious before anyone inspects files
by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import json
from pathlib import Path
from typing import Any


DEFAULT_COVERAGE_THRESHOLD = 0.5


@dataclass(frozen=True)
class ArtifactCounts:
    """Per-date row counts for each data-alignment artifact."""

    official_date: date
    games: int
    probable_starters: int
    lineup_snapshots: int
    prop_line_snapshots: int
    prop_line_pitcher_coverage: int
    pitcher_daily_features: int
    lineup_daily_features: int
    game_context_features: int
    weather_snapshots: int
    weather_ok_snapshots: int
    weather_roof_closed_snapshots: int
    training_rows: int
    calibrated_probabilities: int
    starter_outcomes: int


@dataclass(frozen=True)
class DateCoverageRow:
    """Coverage diagnostics for one date in the requested window."""

    counts: ArtifactCounts
    feature_coverage: float | None
    outcome_coverage: float | None
    odds_coverage: float | None
    weather_coverage: float | None
    failing_artifacts: tuple[str, ...]

    @property
    def below_threshold(self) -> bool:
        return len(self.failing_artifacts) > 0


@dataclass(frozen=True)
class DataAlignmentReport:
    """Full data-alignment report across the requested window."""

    start_date: date
    end_date: date
    threshold: float
    rows: tuple[DateCoverageRow, ...]

    @property
    def failing_dates(self) -> tuple[date, ...]:
        return tuple(row.counts.official_date for row in self.rows if row.below_threshold)

    @property
    def passed(self) -> bool:
        return len(self.failing_dates) == 0


def _requested_dates(start_date: date, end_date: date) -> list[date]:
    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date")
    day_count = (end_date - start_date).days + 1
    return [start_date + timedelta(days=offset) for offset in range(day_count)]


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _coverage_below_threshold(ratio: float | None, threshold: float) -> bool:
    if ratio is None:
        return True
    return ratio < threshold


def build_date_coverage_rows(
    counts_by_date: list[ArtifactCounts],
    *,
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> list[DateCoverageRow]:
    """Turn raw per-date row counts into coverage diagnostics.

    Kept pure of filesystem IO so the logic can be exercised from unit tests
    with synthetic fixtures.
    """
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")

    rows: list[DateCoverageRow] = []
    for counts in counts_by_date:
        feature_coverage = _ratio(counts.pitcher_daily_features, counts.probable_starters)
        outcome_coverage = _ratio(counts.starter_outcomes, counts.training_rows)
        odds_coverage = _ratio(counts.prop_line_pitcher_coverage, counts.probable_starters)
        weather_coverage = _ratio(
            counts.weather_ok_snapshots + counts.weather_roof_closed_snapshots,
            counts.games,
        )

        failing: list[str] = []
        if _coverage_below_threshold(feature_coverage, threshold):
            failing.append("feature")
        if _coverage_below_threshold(outcome_coverage, threshold):
            failing.append("outcome")
        if _coverage_below_threshold(odds_coverage, threshold):
            failing.append("odds")

        rows.append(
            DateCoverageRow(
                counts=counts,
                feature_coverage=feature_coverage,
                outcome_coverage=outcome_coverage,
                odds_coverage=odds_coverage,
                weather_coverage=weather_coverage,
                failing_artifacts=tuple(failing),
            )
        )
    return rows


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            count += 1
    return count


def _count_jsonl_rows_for_date(path: Path, *, target_date: date) -> int:
    if not path.exists():
        return 0
    target_iso = target_date.isoformat()
    count = 0
    for row in _load_jsonl_rows(path):
        if str(row.get("official_date")) == target_iso:
            count += 1
    return count


def _count_unique_field_for_date(
    path: Path,
    *,
    target_date: date,
    field_name: str,
) -> int:
    if not path.exists():
        return 0
    target_iso = target_date.isoformat()
    values: set[Any] = set()
    for row in _load_jsonl_rows(path):
        if str(row.get("official_date")) != target_iso:
            continue
        value = row.get(field_name)
        if value is None:
            continue
        values.add(value)
    return len(values)


def _latest_run_dir_for_date(
    root: Path,
    *,
    target_date: date,
    required_files: tuple[str, ...],
) -> Path | None:
    date_root = root / f"date={target_date.isoformat()}"
    if not date_root.exists():
        return None
    run_dirs = sorted(path for path in date_root.glob("run=*") if path.is_dir())
    for run_dir in reversed(run_dirs):
        if all(run_dir.joinpath(name).exists() for name in required_files):
            return run_dir
    return None


def _latest_baseline_run_for_date(
    output_root: Path,
    *,
    target_date: date,
) -> Path | None:
    model_root = output_root / "normalized" / "starter_strikeout_baseline"
    if not model_root.exists():
        return None
    run_dirs = sorted(
        path
        for path in model_root.rglob("run=*")
        if path.is_dir() and path.joinpath("training_dataset.jsonl").exists()
    )
    target_iso = target_date.isoformat()
    for run_dir in reversed(run_dirs):
        dataset_rows = _load_jsonl_rows(run_dir / "training_dataset.jsonl")
        if any(str(row.get("official_date")) == target_iso for row in dataset_rows):
            return run_dir
    return None


def collect_artifact_counts_for_date(
    output_dir: Path | str,
    *,
    target_date: date,
) -> ArtifactCounts:
    """Count per-date rows for every data-alignment artifact on disk."""
    output_root = Path(output_dir)
    normalized_root = output_root / "normalized"

    mlb_run = _latest_run_dir_for_date(
        normalized_root / "mlb_stats_api",
        target_date=target_date,
        required_files=("games.jsonl",),
    )
    games = _count_jsonl_rows(mlb_run / "games.jsonl") if mlb_run else 0
    probable_starters = (
        _count_jsonl_rows(mlb_run / "probable_starters.jsonl") if mlb_run else 0
    )
    lineup_snapshots = (
        _count_jsonl_rows(mlb_run / "lineup_snapshots.jsonl") if mlb_run else 0
    )

    odds_run = _latest_run_dir_for_date(
        normalized_root / "the_odds_api",
        target_date=target_date,
        required_files=("prop_line_snapshots.jsonl",),
    )
    if odds_run is not None:
        prop_lines_path = odds_run / "prop_line_snapshots.jsonl"
        prop_line_snapshots = _count_jsonl_rows(prop_lines_path)
        prop_line_pitcher_coverage = _count_unique_field_for_date(
            prop_lines_path,
            target_date=target_date,
            field_name="pitcher_mlb_id",
        )
    else:
        prop_line_snapshots = 0
        prop_line_pitcher_coverage = 0

    feature_run = _latest_run_dir_for_date(
        normalized_root / "statcast_search",
        target_date=target_date,
        required_files=("pitcher_daily_features.jsonl",),
    )
    pitcher_daily_features = (
        _count_jsonl_rows(feature_run / "pitcher_daily_features.jsonl")
        if feature_run
        else 0
    )
    lineup_daily_features = (
        _count_jsonl_rows(feature_run / "lineup_daily_features.jsonl")
        if feature_run
        else 0
    )
    game_context_features = (
        _count_jsonl_rows(feature_run / "game_context_features.jsonl")
        if feature_run
        else 0
    )

    weather_run = _latest_run_dir_for_date(
        normalized_root / "weather",
        target_date=target_date,
        required_files=("weather_snapshots.jsonl",),
    )
    weather_snapshots = 0
    weather_ok_snapshots = 0
    weather_roof_closed_snapshots = 0
    if weather_run is not None:
        for row in _load_jsonl_rows(weather_run / "weather_snapshots.jsonl"):
            weather_snapshots += 1
            status = row.get("weather_status")
            if status == "ok":
                weather_ok_snapshots += 1
            elif status == "roof_closed":
                weather_roof_closed_snapshots += 1

    baseline_run = _latest_baseline_run_for_date(output_root, target_date=target_date)
    if baseline_run is not None:
        training_rows = _count_jsonl_rows_for_date(
            baseline_run / "training_dataset.jsonl",
            target_date=target_date,
        )
        calibrated_probabilities = _count_jsonl_rows_for_date(
            baseline_run / "raw_vs_calibrated_probabilities.jsonl",
            target_date=target_date,
        )
        starter_outcomes = _count_jsonl_rows_for_date(
            baseline_run / "starter_outcomes.jsonl",
            target_date=target_date,
        )
    else:
        training_rows = 0
        calibrated_probabilities = 0
        starter_outcomes = 0

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


def check_data_alignment(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str = "data",
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> DataAlignmentReport:
    """Build a coverage report for every date in ``[start_date, end_date]``."""
    dates = _requested_dates(start_date, end_date)
    counts_by_date = [
        collect_artifact_counts_for_date(output_dir, target_date=target_date)
        for target_date in dates
    ]
    rows = build_date_coverage_rows(counts_by_date, threshold=threshold)
    return DataAlignmentReport(
        start_date=start_date,
        end_date=end_date,
        threshold=threshold,
        rows=tuple(rows),
    )


def _format_ratio(ratio: float | None) -> str:
    if ratio is None:
        return "n/a"
    return f"{ratio * 100:.1f}%"


def render_data_alignment_summary(report: DataAlignmentReport) -> str:
    """Render a human-readable coverage table plus a failing-dates footer."""
    header = (
        "date",
        "games",
        "probable",
        "lineups",
        "prop_lines",
        "pitchers_w_lines",
        "pitcher_feats",
        "lineup_feats",
        "context_feats",
        "weather_ok",
        "weather_roof",
        "training",
        "calibrated",
        "outcomes",
        "feat_cov",
        "out_cov",
        "odds_cov",
        "wx_cov",
        "status",
    )
    table_rows: list[tuple[str, ...]] = [header]
    for row in report.rows:
        counts = row.counts
        table_rows.append(
            (
                counts.official_date.isoformat(),
                str(counts.games),
                str(counts.probable_starters),
                str(counts.lineup_snapshots),
                str(counts.prop_line_snapshots),
                str(counts.prop_line_pitcher_coverage),
                str(counts.pitcher_daily_features),
                str(counts.lineup_daily_features),
                str(counts.game_context_features),
                str(counts.weather_ok_snapshots),
                str(counts.weather_roof_closed_snapshots),
                str(counts.training_rows),
                str(counts.calibrated_probabilities),
                str(counts.starter_outcomes),
                _format_ratio(row.feature_coverage),
                _format_ratio(row.outcome_coverage),
                _format_ratio(row.odds_coverage),
                _format_ratio(row.weather_coverage),
                "FAIL" if row.below_threshold else "ok",
            )
        )

    widths = [max(len(row[column]) for row in table_rows) for column in range(len(header))]
    formatted_rows = [
        "  ".join(value.ljust(widths[column]) for column, value in enumerate(row))
        for row in table_rows
    ]

    lines = [
        (
            "Data alignment report "
            f"{report.start_date.isoformat()} -> {report.end_date.isoformat()} "
            f"(threshold={report.threshold:.0%})"
        ),
        "",
        *formatted_rows,
    ]

    failing_rows = [row for row in report.rows if row.below_threshold]
    if failing_rows:
        lines.append("")
        lines.append(
            f"Failing dates (coverage below {report.threshold:.0%}):"
        )
        for row in failing_rows:
            reasons = ", ".join(row.failing_artifacts)
            lines.append(
                f"- {row.counts.official_date.isoformat()} "
                f"failing={reasons}"
            )
    else:
        lines.append("")
        lines.append(f"All dates meet the {report.threshold:.0%} coverage threshold.")

    return "\n".join(lines)
