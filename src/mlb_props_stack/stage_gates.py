"""Executable live-use and expansion readiness stage gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from statistics import median
from typing import Any, Callable, Literal

from .ingest.mlb_stats_api import utc_now

STATUS_RESEARCH_ONLY = "research_only"
STATUS_LIVE_DISCUSSION = "eligible_for_live_discussion"
STATUS_NEXT_MARKET = "eligible_for_next_market_expansion"

KEY_METRIC_ORDER = (
    "held_out_rows",
    "held_out_beats_benchmark_rmse",
    "held_out_beats_benchmark_mae",
    "calibrated_ece",
    "scoreable_backtest_rows",
    "backtest_skip_rate",
    "backtest_placed_bets",
    "backtest_clv_sample",
    "backtest_median_clv_probability_delta",
    "backtest_roi",
    "settled_paper_bets",
    "paper_dates",
    "paper_same_line_clv_sample",
    "paper_beat_close_rate",
    "paper_median_clv_probability_delta",
    "paper_roi",
)

CheckOp = Literal["ge", "le", "gt", "true", "live_passed"]


@dataclass(frozen=True)
class MetricCheck:
    metric: str
    op: CheckOp
    threshold: int | float | bool | None = None


@dataclass(frozen=True)
class GateSpec:
    group: str
    name: str
    checks: tuple[MetricCheck, ...]
    threshold: str


@dataclass(frozen=True)
class StageGateMetric:
    name: str
    value: int | float | bool | None
    source_path: Path | None
    definition: str


@dataclass(frozen=True)
class StageGateResult:
    group: str
    name: str
    passed: bool
    actual: str
    threshold: str
    source_paths: tuple[Path, ...]


@dataclass(frozen=True)
class StageGateEvaluationResult:
    run_id: str
    generated_at: datetime
    status: str
    output_root: Path
    training_summary_path: Path | None
    backtest_runs_path: Path | None
    clv_summary_path: Path | None
    roi_summary_path: Path | None
    paper_results_path: Path | None
    report_path: Path
    report_markdown_path: Path
    metrics: dict[str, StageGateMetric]
    live_use_gates: tuple[StageGateResult, ...]
    next_market_gates: tuple[StageGateResult, ...]
    warnings: tuple[str, ...]


LIVE_GATE_SPECS = (
    GateSpec(
        "live_use",
        "held_out_quality",
        (
            MetricCheck("held_out_rows", "ge", 100),
            MetricCheck("held_out_beats_benchmark_rmse", "true"),
            MetricCheck("held_out_beats_benchmark_mae", "true"),
            MetricCheck("calibrated_ece", "le", 0.03),
        ),
        "held_out_rows >= 100; beats RMSE and MAE; calibrated_ece <= 0.03",
    ),
    GateSpec(
        "live_use",
        "backtest_coverage",
        (
            MetricCheck("scoreable_backtest_rows", "ge", 100),
            MetricCheck("backtest_placed_bets", "ge", 75),
            MetricCheck("backtest_skip_rate", "le", 0.20),
        ),
        (
            "scoreable_backtest_rows >= 100; backtest_placed_bets >= 75; "
            "backtest_skip_rate <= 0.2"
        ),
    ),
    GateSpec(
        "live_use",
        "paper_sample",
        (
            MetricCheck("settled_paper_bets", "ge", 100),
            MetricCheck("paper_dates", "ge", 30),
        ),
        "settled_paper_bets >= 100; paper_dates >= 30",
    ),
    GateSpec(
        "live_use",
        "market_beating_evidence",
        (
            MetricCheck("paper_same_line_clv_sample", "ge", 75),
            MetricCheck("paper_beat_close_rate", "ge", 0.52),
            MetricCheck("paper_median_clv_probability_delta", "gt", 0),
            MetricCheck("backtest_clv_sample", "ge", 75),
            MetricCheck("backtest_median_clv_probability_delta", "gt", 0),
        ),
        (
            "paper_same_line_clv_sample >= 75; paper_beat_close_rate >= 0.52; "
            "paper_median_clv_probability_delta > 0; backtest_clv_sample >= 75; "
            "backtest_median_clv_probability_delta > 0"
        ),
    ),
    GateSpec(
        "live_use",
        "profit_corroboration",
        (
            MetricCheck("paper_roi", "gt", 0),
            MetricCheck("backtest_roi", "gt", 0),
        ),
        "paper_roi > 0; backtest_roi > 0",
    ),
)

NEXT_MARKET_SPECS = (
    GateSpec(
        "next_market",
        "current_market_live_gate",
        (MetricCheck("__live_use_passed__", "live_passed"),),
        "all live-use gates pass",
    ),
    GateSpec(
        "next_market",
        "held_out_depth",
        (
            MetricCheck("held_out_rows", "ge", 150),
            MetricCheck("held_out_beats_benchmark_rmse", "true"),
            MetricCheck("held_out_beats_benchmark_mae", "true"),
            MetricCheck("calibrated_ece", "le", 0.025),
        ),
        "held_out_rows >= 150; beats RMSE and MAE; calibrated_ece <= 0.025",
    ),
    GateSpec(
        "next_market",
        "backtest_depth",
        (
            MetricCheck("scoreable_backtest_rows", "ge", 250),
            MetricCheck("backtest_placed_bets", "ge", 150),
            MetricCheck("backtest_skip_rate", "le", 0.10),
        ),
        (
            "scoreable_backtest_rows >= 250; backtest_placed_bets >= 150; "
            "backtest_skip_rate <= 0.1"
        ),
    ),
    GateSpec(
        "next_market",
        "paper_depth",
        (
            MetricCheck("settled_paper_bets", "ge", 250),
            MetricCheck("paper_dates", "ge", 60),
        ),
        "settled_paper_bets >= 250; paper_dates >= 60",
    ),
    GateSpec(
        "next_market",
        "persistent_clv",
        (
            MetricCheck("paper_same_line_clv_sample", "ge", 150),
            MetricCheck("paper_beat_close_rate", "ge", 0.53),
            MetricCheck("paper_median_clv_probability_delta", "gt", 0),
            MetricCheck("backtest_clv_sample", "ge", 150),
            MetricCheck("backtest_median_clv_probability_delta", "gt", 0),
        ),
        (
            "paper_same_line_clv_sample >= 150; paper_beat_close_rate >= 0.53; "
            "paper_median_clv_probability_delta > 0; backtest_clv_sample >= 150; "
            "backtest_median_clv_probability_delta > 0"
        ),
    ),
    GateSpec(
        "next_market",
        "persistent_profitability",
        (
            MetricCheck("paper_roi", "gt", 0),
            MetricCheck("backtest_roi", "gt", 0),
        ),
        "paper_roi > 0; backtest_roi > 0",
    ),
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def _load_json(path: Path | None) -> dict[str, Any]:
    return {} if path is None or not path.exists() else json.loads(path.read_text())


def _load_jsonl_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_id_from_dir(path: Path) -> str:
    return path.name.split("=", 1)[-1]


def _latest_artifact_path(root: Path, file_name: str) -> Path | None:
    if not root.exists():
        return None
    candidates = [
        path
        for path in root.rglob(file_name)
        if path.parent.name.startswith("run=")
    ]
    return max(
        candidates,
        key=lambda path: (_run_id_from_dir(path.parent), str(path)),
        default=None,
    )


def _latest_backtest_run_dir(output_root: Path) -> Path | None:
    path = _latest_artifact_path(
        output_root / "normalized" / "walk_forward_backtest",
        "backtest_runs.jsonl",
    )
    return None if path is None else path.parent


def _find_training_summary(output_root: Path, model_run_id: str | None) -> Path | None:
    root = output_root / "normalized" / "starter_strikeout_baseline"
    if model_run_id:
        matches = [
            path
            for path in root.rglob("evaluation_summary.json")
            if path.parent.name == f"run={model_run_id}"
        ]
        if matches:
            return sorted(matches)[-1]
    return _latest_artifact_path(root, "evaluation_summary.json")


def _latest_paper_results_path(output_root: Path) -> Path | None:
    root = output_root / "normalized" / "paper_results"
    if not root.exists():
        return None
    candidates = [
        path
        for path in root.rglob("paper_results.jsonl")
        if path.parent.name.startswith("run=")
        and path.parent.parent.name.startswith("date=")
    ]
    return max(
        candidates,
        key=lambda path: (
            path.parent.parent.name.split("=", 1)[-1],
            _run_id_from_dir(path.parent),
            str(path),
        ),
        default=None,
    )


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    number = _float(value)
    return None if number is None else int(number)


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator <= 0 else round(numerator / denominator, 6)


def _overall(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return next(
        (row for row in rows if row.get("summary_scope") == "overall"),
        rows[-1] if rows else {},
    )


def _median(values: list[float]) -> float | None:
    return None if not values else round(float(median(values)), 6)


def _metric(
    metrics: dict[str, StageGateMetric],
    name: str,
    value: int | float | bool | None,
    source_path: Path | None,
    definition: str,
) -> None:
    metrics[name] = StageGateMetric(name, value, source_path, definition)


def _build_metrics(paths: dict[str, Path | None]) -> dict[str, StageGateMetric]:
    metrics: dict[str, StageGateMetric] = {}
    training = _load_json(paths["training"])
    backtest = (_load_jsonl_rows(paths["backtest"]) or [{}])[-1]
    clv = _overall(_load_jsonl_rows(paths["clv"]))
    roi = _overall(_load_jsonl_rows(paths["roi"]))
    paper_rows = _load_jsonl_rows(paths["paper"])
    settled_rows = [
        row
        for row in paper_rows
        if str(row.get("settlement_status") or row.get("paper_result") or "")
        in {"win", "loss", "push"}
    ]
    paper_clv_rows = [
        row for row in settled_rows if bool(row.get("same_line_close_available"))
    ]

    row_counts = training.get("row_counts", {}) or {}
    beats = (training.get("held_out_performance", {}) or {}).get(
        "beats_benchmark",
        {},
    ) or {}
    ece = (
        (training.get("held_out_probability_calibration", {}) or {})
        .get("calibrated", {})
        .get("expected_calibration_error")
    )
    _metric(metrics, "held_out_rows", _int(row_counts.get("held_out")), paths["training"], "evaluation_summary.json -> row_counts.held_out")
    _metric(metrics, "held_out_beats_benchmark_rmse", bool(beats.get("rmse")) if beats else None, paths["training"], "evaluation_summary.json -> held_out_performance.beats_benchmark.rmse")
    _metric(metrics, "held_out_beats_benchmark_mae", bool(beats.get("mae")) if beats else None, paths["training"], "evaluation_summary.json -> held_out_performance.beats_benchmark.mae")
    _metric(metrics, "calibrated_ece", _float(ece), paths["training"], "evaluation_summary.json -> held_out_probability_calibration.calibrated.expected_calibration_error")

    backtest_counts = backtest.get("row_counts", {}) or {}
    actionable = _int(backtest_counts.get("actionable")) or 0
    below_threshold = _int(backtest_counts.get("below_threshold")) or 0
    skipped = _int(backtest_counts.get("skipped"))
    snapshot_groups = _int(backtest_counts.get("snapshot_groups"))
    _metric(metrics, "scoreable_backtest_rows", actionable + below_threshold, paths["backtest"], "backtest_runs.jsonl -> row_counts.actionable + row_counts.below_threshold")
    _metric(metrics, "backtest_skip_rate", _ratio(float(skipped), float(snapshot_groups)) if skipped is not None and snapshot_groups is not None else None, paths["backtest"], "backtest_runs.jsonl -> row_counts.skipped / row_counts.snapshot_groups")
    _metric(metrics, "backtest_placed_bets", _int((backtest.get("bet_outcomes", {}) or {}).get("placed_bets")) or 0, paths["backtest"], "backtest_runs.jsonl -> bet_outcomes.placed_bets")
    _metric(metrics, "backtest_clv_sample", _int(clv.get("sample_count")) or 0, paths["clv"], "clv_summary.jsonl -> sample_count for summary_scope=overall")
    _metric(metrics, "backtest_median_clv_probability_delta", _float(clv.get("median_probability_delta")), paths["clv"], "clv_summary.jsonl -> median_probability_delta for summary_scope=overall")
    _metric(metrics, "backtest_roi", _float(roi.get("roi")), paths["roi"], "roi_summary.jsonl -> roi for summary_scope=overall")

    paper_stake = sum(_float(row.get("stake_fraction")) or 0.0 for row in settled_rows)
    paper_profit = sum(_float(row.get("profit_units")) or 0.0 for row in settled_rows)
    beat_close_count = sum(1 for row in paper_clv_rows if row.get("beat_closing_line") is True)
    clv_values = [
        value
        for row in paper_clv_rows
        if (value := _float(row.get("clv_probability_delta"))) is not None
    ]
    _metric(metrics, "settled_paper_bets", len(settled_rows), paths["paper"], "paper_results.jsonl -> rows where settlement_status is win, loss, or push")
    _metric(metrics, "paper_dates", len({str(row.get("official_date")) for row in settled_rows if row.get("official_date")}), paths["paper"], "paper_results.jsonl -> distinct official_date values for settled paper bets")
    _metric(metrics, "paper_same_line_clv_sample", len(paper_clv_rows), paths["paper"], "paper_results.jsonl -> settled rows where same_line_close_available=true")
    _metric(metrics, "paper_beat_close_rate", _ratio(float(beat_close_count), float(len(paper_clv_rows))), paths["paper"], "paper_results.jsonl -> beat_closing_line=true divided by paper_same_line_clv_sample")
    _metric(metrics, "paper_median_clv_probability_delta", _median(clv_values), paths["paper"], "paper_results.jsonl -> median clv_probability_delta for same-line settled bets")
    _metric(metrics, "paper_roi", _ratio(paper_profit, paper_stake), paths["paper"], "paper_results.jsonl -> sum(profit_units) / sum(stake_fraction)")
    return metrics


def _format_value(value: int | float | bool | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def _check_passed(
    check: MetricCheck,
    metrics: dict[str, StageGateMetric],
    *,
    live_use_passed: bool,
) -> bool:
    if check.op == "live_passed":
        return live_use_passed
    value = metrics[check.metric].value
    if check.op == "true":
        return value is True
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    threshold = float(check.threshold or 0)
    if check.op == "ge":
        return value >= threshold
    if check.op == "le":
        return value <= threshold
    return value > threshold


def _source_paths(
    spec: GateSpec,
    metrics: dict[str, StageGateMetric],
    live_use_gates: tuple[StageGateResult, ...],
) -> tuple[Path, ...]:
    if spec.checks[0].op == "live_passed":
        paths = [path for gate in live_use_gates for path in gate.source_paths]
    else:
        paths = [
            metrics[check.metric].source_path
            for check in spec.checks
            if metrics[check.metric].source_path is not None
        ]
    deduped: list[Path] = []
    for path in paths:
        if path is not None and path not in deduped:
            deduped.append(path)
    return tuple(deduped)


def _actual_text(
    spec: GateSpec,
    metrics: dict[str, StageGateMetric],
    *,
    live_use_passed: bool,
) -> str:
    if spec.checks[0].op == "live_passed":
        return f"live_use_gates_passed={str(live_use_passed).lower()}"
    return ", ".join(
        f"{check.metric}={_format_value(metrics[check.metric].value)}"
        for check in spec.checks
    )


def _build_gates(
    specs: tuple[GateSpec, ...],
    metrics: dict[str, StageGateMetric],
    *,
    live_use_gates: tuple[StageGateResult, ...] = (),
) -> tuple[StageGateResult, ...]:
    live_use_passed = all(gate.passed for gate in live_use_gates)
    return tuple(
        StageGateResult(
            group=spec.group,
            name=spec.name,
            passed=all(
                _check_passed(check, metrics, live_use_passed=live_use_passed)
                for check in spec.checks
            ),
            actual=_actual_text(spec, metrics, live_use_passed=live_use_passed),
            threshold=spec.threshold,
            source_paths=_source_paths(spec, metrics, live_use_gates),
        )
        for spec in specs
    )


def _status(
    live_use_gates: tuple[StageGateResult, ...],
    next_market_gates: tuple[StageGateResult, ...],
) -> str:
    if not all(gate.passed for gate in live_use_gates):
        return STATUS_RESEARCH_ONLY
    if not all(gate.passed for gate in next_market_gates):
        return STATUS_LIVE_DISCUSSION
    return STATUS_NEXT_MARKET


def _warnings(
    paths: dict[str, Path | None],
    *,
    training_run_id: str | None,
    backtest_model_run_id: str | None,
) -> tuple[str, ...]:
    warnings = [
        f"missing {file_name} artifact"
        for key, file_name in (
            ("training", "evaluation_summary.json"),
            ("backtest", "backtest_runs.jsonl"),
            ("clv", "clv_summary.jsonl"),
            ("roi", "roi_summary.jsonl"),
            ("paper", "paper_results.jsonl"),
        )
        if paths[key] is None
    ]
    if training_run_id and backtest_model_run_id and training_run_id != backtest_model_run_id:
        warnings.append(
            "training run_id does not match backtest model_run_id "
            f"({training_run_id} != {backtest_model_run_id})"
        )
    return tuple(warnings)


def _report_payload(result: StageGateEvaluationResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "generated_at": result.generated_at,
        "status": result.status,
        "output_root": result.output_root,
        "artifact_paths": {
            "training_summary_path": result.training_summary_path,
            "backtest_runs_path": result.backtest_runs_path,
            "clv_summary_path": result.clv_summary_path,
            "roi_summary_path": result.roi_summary_path,
            "paper_results_path": result.paper_results_path,
        },
        "metrics": {key: asdict(metric) for key, metric in result.metrics.items()},
        "live_use_gates": [asdict(gate) for gate in result.live_use_gates],
        "next_market_gates": [asdict(gate) for gate in result.next_market_gates],
        "warnings": list(result.warnings),
        "report_path": result.report_path,
        "report_markdown_path": result.report_markdown_path,
    }


def evaluate_stage_gates(
    *,
    output_dir: Path | str = "data",
    training_summary_path: Path | str | None = None,
    backtest_run_dir: Path | str | None = None,
    paper_results_path: Path | str | None = None,
    now: Callable[[], datetime] = utc_now,
) -> StageGateEvaluationResult:
    """Evaluate the latest coherent artifact set against readiness gates."""
    output_root = Path(output_dir)
    generated_at = now().astimezone(UTC)
    run_id = generated_at.strftime("%Y%m%dT%H%M%SZ")

    backtest_dir = Path(backtest_run_dir) if backtest_run_dir else _latest_backtest_run_dir(output_root)
    backtest_path = backtest_dir / "backtest_runs.jsonl" if backtest_dir else None
    if backtest_path is not None and not backtest_path.exists():
        backtest_path = None
    backtest_row = (_load_jsonl_rows(backtest_path) or [{}])[-1]
    backtest_model_run_id = (
        str(backtest_row["model_run_id"]) if backtest_row.get("model_run_id") else None
    )

    training_path = (
        Path(training_summary_path)
        if training_summary_path
        else _find_training_summary(output_root, backtest_model_run_id)
    )
    if training_path is not None and not training_path.exists():
        training_path = None

    paths = {
        "training": training_path,
        "backtest": backtest_path,
        "clv": backtest_dir / "clv_summary.jsonl" if backtest_dir else None,
        "roi": backtest_dir / "roi_summary.jsonl" if backtest_dir else None,
        "paper": (
            Path(paper_results_path)
            if paper_results_path
            else _latest_paper_results_path(output_root)
        ),
    }
    for key, path in tuple(paths.items()):
        if path is not None and not path.exists():
            paths[key] = None

    metrics = _build_metrics(paths)
    live_use_gates = _build_gates(LIVE_GATE_SPECS, metrics)
    next_market_gates = _build_gates(
        NEXT_MARKET_SPECS,
        metrics,
        live_use_gates=live_use_gates,
    )
    report_dir = output_root / "normalized" / "stage_gates" / f"run={run_id}"
    result = StageGateEvaluationResult(
        run_id=run_id,
        generated_at=generated_at,
        status=_status(live_use_gates, next_market_gates),
        output_root=output_root,
        training_summary_path=paths["training"],
        backtest_runs_path=paths["backtest"],
        clv_summary_path=paths["clv"],
        roi_summary_path=paths["roi"],
        paper_results_path=paths["paper"],
        report_path=report_dir / "stage_gate_report.json",
        report_markdown_path=report_dir / "stage_gate_report.md",
        metrics=metrics,
        live_use_gates=live_use_gates,
        next_market_gates=next_market_gates,
        warnings=_warnings(
            paths,
            training_run_id=(_load_json(paths["training"]).get("run_id") if paths["training"] else None),
            backtest_model_run_id=backtest_model_run_id,
        ),
    )
    _write_json(result.report_path, _report_payload(result))
    _write_text(result.report_markdown_path, render_stage_gate_summary(result))
    return result


def _format_paths(paths: tuple[Path, ...]) -> str:
    return "n/a" if not paths else "; ".join(str(path) for path in paths)


def _gate_table(gates: tuple[StageGateResult, ...]) -> list[str]:
    lines = ["gate | result | actual | threshold | source"]
    lines.append("--- | --- | --- | --- | ---")
    for gate in gates:
        lines.append(
            " | ".join(
                (
                    gate.name,
                    "pass" if gate.passed else "fail",
                    gate.actual,
                    gate.threshold,
                    _format_paths(gate.source_paths),
                )
            )
        )
    return lines


def render_stage_gate_summary(result: StageGateEvaluationResult) -> str:
    """Render a terminal-friendly readiness report."""
    lines = [
        "Stage-gate evaluation complete",
        f"run_id={result.run_id}",
        f"status={result.status}",
        f"training_summary_path={result.training_summary_path or 'n/a'}",
        f"backtest_runs_path={result.backtest_runs_path or 'n/a'}",
        f"clv_summary_path={result.clv_summary_path or 'n/a'}",
        f"roi_summary_path={result.roi_summary_path or 'n/a'}",
        f"paper_results_path={result.paper_results_path or 'n/a'}",
        f"report_path={result.report_path}",
        f"report_markdown_path={result.report_markdown_path}",
    ]
    if result.warnings:
        lines.extend(("", "Warnings"))
        lines.extend(f"- {warning}" for warning in result.warnings)

    lines.extend(("", "Key metrics"))
    lines.extend(
        f"{name}={_format_value(result.metrics[name].value)}"
        for name in KEY_METRIC_ORDER
    )
    lines.extend(("", "Live-use gates"))
    lines.extend(_gate_table(result.live_use_gates))
    lines.extend(("", "Next-market expansion gates"))
    lines.extend(_gate_table(result.next_market_gates))
    return "\n".join(lines)
