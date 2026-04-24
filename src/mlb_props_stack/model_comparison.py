"""Core-vs-expanded starter strikeout baseline comparison reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import json
from pathlib import Path
from shlex import quote
from typing import Any, Callable

from .backtest import WalkForwardBacktestResult, build_walk_forward_backtest
from .ingest.mlb_stats_api import utc_now
from .ingest.statcast_features import StatcastSearchClient
from .modeling import (
    FEATURE_SET_CORE,
    FEATURE_SET_EXPANDED,
    StarterStrikeoutBaselineTrainingResult,
    train_starter_strikeout_baseline,
)
from .tracking import TrackingConfig
from .wager_approval import WagerApprovalSettings, annotate_wager_approval_rows


@dataclass(frozen=True)
class StarterStrikeoutModelComparisonResult:
    """Filesystem output summary for one model-variant comparison."""

    start_date: date
    end_date: date
    run_id: str
    recommendation: str
    core_training_run_id: str
    expanded_training_run_id: str
    core_backtest_run_id: str
    expanded_backtest_run_id: str
    comparison_path: Path
    comparison_markdown_path: Path
    reproducibility_notes_path: Path


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(_json_ready(payload), indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _path_run_id(run_dir: Path) -> str:
    return run_dir.name.split("=", 1)[-1]


def _unique_timestamp_run_id(base_time: datetime, run_root: Path) -> str:
    candidate_time = base_time.astimezone(UTC)
    while True:
        run_id = candidate_time.strftime("%Y%m%dT%H%M%SZ")
        if not run_root.joinpath(f"run={run_id}").exists():
            return run_id
        candidate_time += timedelta(seconds=1)


def _metric(payload: dict[str, Any], *path: str) -> float | None:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None
    return float(current)


def _not_worse(
    expanded: float | None,
    core: float | None,
    *,
    lower_is_better: bool,
) -> bool:
    if expanded is None and core is None:
        return True
    if expanded is None or core is None:
        return False
    return expanded <= core if lower_is_better else expanded >= core


def _improved(
    expanded: float | None,
    core: float | None,
    *,
    lower_is_better: bool,
) -> bool:
    if expanded is None or core is None:
        return False
    return expanded < core if lower_is_better else expanded > core


def _first_jsonl_row(path: Path) -> dict[str, Any]:
    rows = _load_jsonl_rows(path)
    return rows[0] if rows else {}


def _overall_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if row.get("summary_scope") == "overall":
            return row
    return {}


def _final_gate_summary(
    bet_reporting_path: Path,
    *,
    now: datetime,
) -> dict[str, Any]:
    rows = [
        row
        for row in _load_jsonl_rows(bet_reporting_path)
        if row.get("evaluation_status") in {"actionable", "below_threshold"}
    ]
    annotated_rows = annotate_wager_approval_rows(
        rows,
        settings=WagerApprovalSettings(),
        now=now,
    )
    blocked_reason_counts: dict[str, int] = {}
    for row in annotated_rows:
        if row.get("wager_approved"):
            continue
        reason = str(row.get("wager_blocked_reason") or "unknown")
        blocked_reason_counts[reason] = blocked_reason_counts.get(reason, 0) + 1
    return {
        "evaluated_rows": len(annotated_rows),
        "approved_wagers": sum(1 for row in annotated_rows if row.get("wager_approved")),
        "blocked_wagers": sum(1 for row in annotated_rows if not row.get("wager_approved")),
        "blocked_reason_counts": dict(sorted(blocked_reason_counts.items())),
    }


def _leakage_summary(join_audit_path: Path) -> dict[str, Any]:
    rows = _load_jsonl_rows(join_audit_path)
    ok_rows = [row for row in rows if row.get("audit_status") == "ok"]
    violations: list[dict[str, Any]] = []
    for row in ok_rows:
        failed_fields = [
            field_name
            for field_name in (
                "features_before_cutoff",
                "projection_before_cutoff",
                "training_window_before_evaluated_date",
                "calibration_window_before_evaluated_date",
            )
            if not bool(row.get(field_name))
        ]
        if failed_fields:
            violations.append(
                {
                    "audit_id": row.get("audit_id"),
                    "failed_fields": failed_fields,
                }
            )
    return {
        "audit_rows": len(rows),
        "ok_audit_rows": len(ok_rows),
        "timestamp_violation_count": len(violations),
        "violations": violations[:20],
    }


def _variant_payload(
    *,
    feature_set: str,
    training_result: StarterStrikeoutBaselineTrainingResult,
    backtest_result: WalkForwardBacktestResult,
    final_gate_now: datetime,
) -> dict[str, Any]:
    training_run_dir = training_result.model_path.parent
    backtest_run_dir = backtest_result.backtest_runs_path.parent
    evaluation_summary = _load_json(training_result.evaluation_summary_path)
    backtest_summary = _first_jsonl_row(backtest_result.backtest_runs_path)
    clv_summary = _overall_summary(_load_jsonl_rows(backtest_result.clv_summary_path))
    roi_summary = _overall_summary(_load_jsonl_rows(backtest_result.roi_summary_path))
    final_gates = _final_gate_summary(
        backtest_result.bet_reporting_path,
        now=final_gate_now,
    )
    leakage = _leakage_summary(backtest_result.join_audit_path)
    row_counts = backtest_summary.get("row_counts") or {}

    return {
        "feature_set": feature_set,
        "training": {
            "run_id": training_result.run_id,
            "run_dir": training_run_dir,
            "model_path": training_result.model_path,
            "evaluation_summary_path": training_result.evaluation_summary_path,
            "evaluation_path": training_result.evaluation_path,
            "row_counts": evaluation_summary.get("row_counts") or {},
            "held_out_performance": evaluation_summary.get("held_out_performance") or {},
            "held_out_probability_calibration": (
                evaluation_summary.get("held_out_probability_calibration") or {}
            ),
            "feature_schema": evaluation_summary.get("feature_schema") or {},
        },
        "backtest": {
            "run_id": backtest_result.run_id,
            "run_dir": backtest_run_dir,
            "backtest_runs_path": backtest_result.backtest_runs_path,
            "bet_reporting_path": backtest_result.bet_reporting_path,
            "join_audit_path": backtest_result.join_audit_path,
            "clv_summary_path": backtest_result.clv_summary_path,
            "roi_summary_path": backtest_result.roi_summary_path,
            "edge_bucket_summary_path": backtest_result.edge_bucket_summary_path,
            "row_counts": row_counts,
            "scoreable_rows": int(row_counts.get("actionable") or 0)
            + int(row_counts.get("below_threshold") or 0),
            "skip_reason_counts": backtest_summary.get("skip_reason_counts") or {},
            "bet_outcomes": backtest_summary.get("bet_outcomes") or {},
            "clv_summary": clv_summary,
            "roi_summary": roi_summary,
            "edge_bucket_summary": backtest_summary.get("edge_bucket_summary") or [],
        },
        "final_wager_gates": final_gates,
        "leakage_audit": leakage,
    }


def _recommendation(core: dict[str, Any], expanded: dict[str, Any]) -> dict[str, Any]:
    core_training = core["training"]
    expanded_training = expanded["training"]
    core_backtest = core["backtest"]
    expanded_backtest = expanded["backtest"]
    core_calibration = core_training["held_out_probability_calibration"].get(
        "calibrated",
        {},
    )
    expanded_calibration = expanded_training[
        "held_out_probability_calibration"
    ].get("calibrated", {})
    core_model = core_training["held_out_performance"].get("model", {})
    expanded_model = expanded_training["held_out_performance"].get("model", {})

    checks = {
        "expanded_optional_features_active": bool(
            expanded_training["feature_schema"].get("active_optional_features")
        ),
        "held_out_rmse_improved": _improved(
            _metric(expanded_model, "rmse"),
            _metric(core_model, "rmse"),
            lower_is_better=True,
        ),
        "held_out_mae_not_worse": _not_worse(
            _metric(expanded_model, "mae"),
            _metric(core_model, "mae"),
            lower_is_better=True,
        ),
        "calibrated_log_loss_not_worse": _not_worse(
            _metric(expanded_calibration, "mean_log_loss"),
            _metric(core_calibration, "mean_log_loss"),
            lower_is_better=True,
        ),
        "calibrated_ece_not_worse": _not_worse(
            _metric(expanded_calibration, "expected_calibration_error"),
            _metric(core_calibration, "expected_calibration_error"),
            lower_is_better=True,
        ),
        "scoreable_rows_not_worse": (
            int(expanded_backtest.get("scoreable_rows") or 0)
            >= int(core_backtest.get("scoreable_rows") or 0)
        ),
        "approved_wagers_not_worse": (
            int(expanded["final_wager_gates"].get("approved_wagers") or 0)
            >= int(core["final_wager_gates"].get("approved_wagers") or 0)
        ),
        "median_clv_not_worse": _not_worse(
            _metric(expanded_backtest["clv_summary"], "median_probability_delta"),
            _metric(core_backtest["clv_summary"], "median_probability_delta"),
            lower_is_better=False,
        ),
        "roi_not_worse": _not_worse(
            _metric(expanded_backtest["roi_summary"], "roi"),
            _metric(core_backtest["roi_summary"], "roi"),
            lower_is_better=False,
        ),
        "no_expanded_timestamp_violations": (
            int(expanded["leakage_audit"].get("timestamp_violation_count") or 0) == 0
        ),
    }
    promote = all(checks.values())
    return {
        "status": "promote_expanded_candidate" if promote else "keep_core_only",
        "checks": checks,
        "rationale": (
            "Expanded optional features improved held-out error while preserving "
            "calibration, backtest decision coverage, final-gate counts, and timestamp audits."
            if promote
            else (
                "Do not promote the expanded model from this comparison; one or more "
                "required decision-quality, calibration, optional-activation, or leakage checks failed."
            )
        ),
    }


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _render_comparison_markdown(report: dict[str, Any]) -> str:
    core = report["variants"][FEATURE_SET_CORE]
    expanded = report["variants"][FEATURE_SET_EXPANDED]
    recommendation = report["recommendation"]

    def _held_out_row(label: str, key: str) -> str:
        core_model = core["training"]["held_out_performance"].get("model", {})
        expanded_model = expanded["training"]["held_out_performance"].get("model", {})
        return (
            f"| {label} | {_format_metric(core_model.get(key))} | "
            f"{_format_metric(expanded_model.get(key))} |"
        )

    def _calibration_row(label: str, key: str) -> str:
        core_calibrated = core["training"]["held_out_probability_calibration"].get(
            "calibrated",
            {},
        )
        expanded_calibrated = expanded[
            "training"
        ]["held_out_probability_calibration"].get("calibrated", {})
        return (
            f"| {label} | {_format_metric(core_calibrated.get(key))} | "
            f"{_format_metric(expanded_calibrated.get(key))} |"
        )

    def _backtest_row(label: str, getter: Callable[[dict[str, Any]], Any]) -> str:
        return f"| {label} | {_format_metric(getter(core))} | {_format_metric(getter(expanded))} |"

    active_optional = expanded["training"]["feature_schema"].get(
        "active_optional_features",
        [],
    )
    excluded_optional = expanded["training"]["feature_schema"].get(
        "excluded_optional_features",
        [],
    )

    lines = [
        "# Starter Strikeout Model Variant Comparison",
        "",
        f"- Run ID: `{report['run_id']}`",
        (
            "- Date window: "
            f"`{report['date_window']['start_date']}` -> `{report['date_window']['end_date']}`"
        ),
        f"- Cutoff minutes before first pitch: `{report['cutoff_minutes_before_first_pitch']}`",
        f"- Recommendation: `{recommendation['status']}`",
        f"- Rationale: {recommendation['rationale']}",
        "",
        "## Held-Out Error",
        "",
        "| Metric | Core | Expanded |",
        "| --- | ---: | ---: |",
        _held_out_row("RMSE", "rmse"),
        _held_out_row("MAE", "mae"),
        _held_out_row("Spearman", "spearman_rank_correlation"),
        "",
        "## Held-Out Calibration",
        "",
        "| Metric | Core | Expanded |",
        "| --- | ---: | ---: |",
        _calibration_row("Mean Brier Score", "mean_brier_score"),
        _calibration_row("Mean Log Loss", "mean_log_loss"),
        _calibration_row("Expected Calibration Error", "expected_calibration_error"),
        "",
        "## Decision Metrics",
        "",
        "| Metric | Core | Expanded |",
        "| --- | ---: | ---: |",
        _backtest_row("Snapshot groups", lambda row: row["backtest"]["row_counts"].get("snapshot_groups")),
        _backtest_row("Scoreable rows", lambda row: row["backtest"].get("scoreable_rows")),
        _backtest_row("Skipped rows", lambda row: row["backtest"]["row_counts"].get("skipped")),
        _backtest_row("Placed by edge rule", lambda row: row["backtest"]["bet_outcomes"].get("placed_bets")),
        _backtest_row("Approved by final gates", lambda row: row["final_wager_gates"].get("approved_wagers")),
        _backtest_row("CLV sample", lambda row: row["backtest"]["clv_summary"].get("sample_count")),
        _backtest_row("Median CLV probability delta", lambda row: row["backtest"]["clv_summary"].get("median_probability_delta")),
        _backtest_row("ROI", lambda row: row["backtest"]["roi_summary"].get("roi")),
        "",
        "## Expanded Optional Features",
        "",
        f"- Active optional features: `{len(active_optional)}`",
    ]
    if active_optional:
        lines.extend(f"  - `{feature}`" for feature in active_optional)
    else:
        lines.append("  - none")

    lines.extend(
        [
            "",
            "| Optional Feature | Status | Coverage | Range |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for item in excluded_optional:
        lines.append(
            f"| `{item['feature']}` | `{item['status']}` | "
            f"{_format_metric(item['coverage'])} | {_format_metric(item['range'])} |"
        )

    lines.extend(
        [
            "",
            "## Artifact Paths",
            "",
            f"- Core model: `{core['training']['run_dir']}`",
            f"- Core backtest: `{core['backtest']['run_dir']}`",
            f"- Expanded model: `{expanded['training']['run_dir']}`",
            f"- Expanded backtest: `{expanded['backtest']['run_dir']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_reproducibility_notes(report: dict[str, Any], *, output_dir: Path | str) -> str:
    return "\n".join(
        [
            "# Model Comparison Reproducibility Notes",
            "",
            f"- Local comparison run ID: `{report['run_id']}`",
            (
                "- Date window: "
                f"`{report['date_window']['start_date']}` -> `{report['date_window']['end_date']}`"
            ),
            f"- Output directory: `{output_dir}`",
            f"- Core training command: `{report['commands']['core_training']}`",
            f"- Expanded training command: `{report['commands']['expanded_training']}`",
            f"- Core backtest command: `{report['commands']['core_backtest']}`",
            f"- Expanded backtest command: `{report['commands']['expanded_backtest']}`",
            (
                "- Honest rules: both variants use the same saved feature and odds "
                "window, chronological training splits, explicit model-run-dir "
                "backtests, and final wager gates applied after backtest scoring."
            ),
            "",
        ]
    )


def _training_command(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str,
    feature_set: str,
) -> str:
    return (
        "uv run python -m mlb_props_stack train-starter-strikeout-baseline "
        f"--start-date {start_date.isoformat()} "
        f"--end-date {end_date.isoformat()} "
        f"--output-dir {quote(str(output_dir))} "
        f"--feature-set {feature_set}"
    )


def _backtest_command(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str,
    model_run_dir: Path,
    cutoff_minutes_before_first_pitch: int,
) -> str:
    return (
        "uv run python -m mlb_props_stack build-walk-forward-backtest "
        f"--start-date {start_date.isoformat()} "
        f"--end-date {end_date.isoformat()} "
        f"--output-dir {quote(str(output_dir))} "
        f"--model-run-dir {quote(str(model_run_dir))} "
        "--cutoff-minutes-before-first-pitch "
        f"{cutoff_minutes_before_first_pitch}"
    )


def compare_starter_strikeout_baselines(
    *,
    start_date: date,
    end_date: date,
    output_dir: Path | str = "data",
    cutoff_minutes_before_first_pitch: int = 30,
    client: StatcastSearchClient | None = None,
    now: Callable[[], datetime] = utc_now,
    tracking_config: TrackingConfig | None = None,
) -> StarterStrikeoutModelComparisonResult:
    """Train core and expanded variants, backtest both, and write a report."""
    output_root = Path(output_dir)
    resolved_client = client or StatcastSearchClient()

    core_training = train_starter_strikeout_baseline(
        start_date=start_date,
        end_date=end_date,
        output_dir=output_root,
        client=resolved_client,
        now=now,
        tracking_config=tracking_config,
        feature_set=FEATURE_SET_CORE,
    )
    expanded_training = train_starter_strikeout_baseline(
        start_date=start_date,
        end_date=end_date,
        output_dir=output_root,
        client=resolved_client,
        now=now,
        tracking_config=tracking_config,
        feature_set=FEATURE_SET_EXPANDED,
    )
    core_backtest = build_walk_forward_backtest(
        start_date=start_date,
        end_date=end_date,
        output_dir=output_root,
        model_run_dir=core_training.model_path.parent,
        cutoff_minutes_before_first_pitch=cutoff_minutes_before_first_pitch,
        now=now,
        tracking_config=tracking_config,
    )
    expanded_backtest = build_walk_forward_backtest(
        start_date=start_date,
        end_date=end_date,
        output_dir=output_root,
        model_run_dir=expanded_training.model_path.parent,
        cutoff_minutes_before_first_pitch=cutoff_minutes_before_first_pitch,
        now=now,
        tracking_config=tracking_config,
    )

    report_now = now().astimezone(UTC)
    core_payload = _variant_payload(
        feature_set=FEATURE_SET_CORE,
        training_result=core_training,
        backtest_result=core_backtest,
        final_gate_now=report_now,
    )
    expanded_payload = _variant_payload(
        feature_set=FEATURE_SET_EXPANDED,
        training_result=expanded_training,
        backtest_result=expanded_backtest,
        final_gate_now=report_now,
    )
    recommendation = _recommendation(core_payload, expanded_payload)
    run_root = (
        output_root
        / "normalized"
        / "starter_strikeout_model_comparison"
        / f"start={start_date.isoformat()}_end={end_date.isoformat()}"
    )
    run_id = _unique_timestamp_run_id(report_now, run_root)
    normalized_root = run_root / f"run={run_id}"
    comparison_path = normalized_root / "model_comparison.json"
    comparison_markdown_path = normalized_root / "model_comparison.md"
    reproducibility_notes_path = normalized_root / "reproducibility_notes.md"

    commands = {
        "core_training": _training_command(
            start_date=start_date,
            end_date=end_date,
            output_dir=output_root,
            feature_set=FEATURE_SET_CORE,
        ),
        "expanded_training": _training_command(
            start_date=start_date,
            end_date=end_date,
            output_dir=output_root,
            feature_set=FEATURE_SET_EXPANDED,
        ),
        "core_backtest": _backtest_command(
            start_date=start_date,
            end_date=end_date,
            output_dir=output_root,
            model_run_dir=core_training.model_path.parent,
            cutoff_minutes_before_first_pitch=cutoff_minutes_before_first_pitch,
        ),
        "expanded_backtest": _backtest_command(
            start_date=start_date,
            end_date=end_date,
            output_dir=output_root,
            model_run_dir=expanded_training.model_path.parent,
            cutoff_minutes_before_first_pitch=cutoff_minutes_before_first_pitch,
        ),
    }
    report = {
        "report_version": "starter_strikeout_model_comparison_v1",
        "run_id": run_id,
        "date_window": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        "cutoff_minutes_before_first_pitch": cutoff_minutes_before_first_pitch,
        "variants": {
            FEATURE_SET_CORE: core_payload,
            FEATURE_SET_EXPANDED: expanded_payload,
        },
        "recommendation": recommendation,
        "commands": commands,
    }
    _write_json(comparison_path, report)
    _write_text(comparison_markdown_path, _render_comparison_markdown(report))
    _write_text(
        reproducibility_notes_path,
        _render_reproducibility_notes(report, output_dir=output_root),
    )

    return StarterStrikeoutModelComparisonResult(
        start_date=start_date,
        end_date=end_date,
        run_id=run_id,
        recommendation=str(recommendation["status"]),
        core_training_run_id=core_training.run_id,
        expanded_training_run_id=expanded_training.run_id,
        core_backtest_run_id=core_backtest.run_id,
        expanded_backtest_run_id=expanded_backtest.run_id,
        comparison_path=comparison_path,
        comparison_markdown_path=comparison_markdown_path,
        reproducibility_notes_path=reproducibility_notes_path,
    )


def render_model_comparison_summary(
    result: StarterStrikeoutModelComparisonResult,
) -> str:
    """Return a concise terminal summary for one comparison run."""
    return "\n".join(
        [
            (
                "Starter strikeout model comparison complete for "
                f"{result.start_date.isoformat()} -> {result.end_date.isoformat()}"
            ),
            f"run_id={result.run_id}",
            f"recommendation={result.recommendation}",
            f"core_training_run_id={result.core_training_run_id}",
            f"expanded_training_run_id={result.expanded_training_run_id}",
            f"core_backtest_run_id={result.core_backtest_run_id}",
            f"expanded_backtest_run_id={result.expanded_backtest_run_id}",
            f"comparison_path={result.comparison_path}",
            f"comparison_markdown_path={result.comparison_markdown_path}",
            f"reproducibility_notes_path={result.reproducibility_notes_path}",
        ]
    )
