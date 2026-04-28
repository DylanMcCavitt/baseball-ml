"""Betting-layer helpers for rebuilt strikeout distribution artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import floor
import json
from pathlib import Path
from typing import Any

from .config import StackConfig


@dataclass(frozen=True)
class ValidationEvidence:
    """Validation-derived approval gates for betting-layer reporting."""

    report_path: Path | None
    report_run_id: str | None
    recommendation: str
    threshold_status: str
    approval_allowed: bool
    min_edge_pct: float
    confidence_buckets: tuple[str, ...]
    min_confidence: float | None
    excluded_line_buckets: tuple[str, ...]
    reason: str


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_datetime(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def optional_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    return parse_datetime(value)


def path_run_id(run_dir: Path) -> str:
    return run_dir.name.split("=", 1)[-1]


def line_bucket(line: float) -> str:
    if line <= 4.5:
        return "low_line_2.5_to_4.5"
    if line <= 6.5:
        return "middle_line_5.5_to_6.5"
    return "high_line_7.5_plus"


def confidence_bucket(confidence: float) -> str:
    lower = min(0.95, max(0.50, floor(confidence * 10.0) / 10.0))
    upper = min(1.00, lower + 0.10)
    return f"{lower:.1f}_to_{upper:.1f}"


def _bucket_lower(bucket: str) -> float | None:
    try:
        return float(bucket.split("_to_", 1)[0])
    except (IndexError, ValueError):
        return None


def probabilities_for_line(
    probability_distribution: list[dict[str, Any]],
    *,
    line: float,
) -> dict[str, float]:
    """Return exact over/under probabilities from a count distribution."""

    over_probability = 0.0
    under_probability = 0.0
    for row in probability_distribution:
        strikeouts = int(row["strikeouts"])
        probability = float(row["probability"])
        if strikeouts > line:
            over_probability += probability
        else:
            under_probability += probability
    total = over_probability + under_probability
    if total <= 0.0:
        raise ValueError("probability_distribution must contain positive mass")
    return {
        "over_probability": over_probability / total,
        "under_probability": under_probability / total,
    }


def find_latest_distribution_model_run_for_date(
    output_root: Path,
    *,
    target_date: str,
) -> Path | None:
    model_root = output_root / "normalized" / "candidate_strikeout_models"
    run_dirs = sorted(
        path
        for path in model_root.rglob("run=*")
        if path.is_dir() and path.joinpath("model_outputs.jsonl").exists()
    )
    for run_dir in reversed(run_dirs):
        rows = load_jsonl(run_dir / "model_outputs.jsonl")
        if any(str(row.get("official_date")) == target_date for row in rows):
            return run_dir
    return None


def find_latest_validation_report(output_root: Path) -> Path | None:
    validation_root = output_root / "normalized" / "model_only_walk_forward_validation"
    candidates = sorted(
        path
        for path in validation_root.rglob("validation_report.json")
        if path.is_file()
    )
    return candidates[-1] if candidates else None


def validation_evidence_from_report(
    report_path: Path | None,
    *,
    config: StackConfig,
) -> ValidationEvidence:
    if report_path is None:
        return ValidationEvidence(
            report_path=None,
            report_run_id=None,
            recommendation="missing_validation_report",
            threshold_status="missing_validation_report",
            approval_allowed=False,
            min_edge_pct=config.min_edge_pct,
            confidence_buckets=(),
            min_confidence=None,
            excluded_line_buckets=(),
            reason=(
                "No model-only validation report was found, so wager approval remains "
                "blocked."
            ),
        )

    report = load_json(report_path)
    recommendation_payload = report.get("go_no_go_recommendation") or {}
    recommendation = str(recommendation_payload.get("recommendation") or "")
    thresholds = report.get("proposed_later_wager_approval_thresholds") or {}
    threshold_status = str(thresholds.get("status") or "")
    confidence_rows = thresholds.get("candidate_confidence_buckets_for_later_approval")
    if not isinstance(confidence_rows, list):
        confidence_rows = []
    confidence_buckets = tuple(
        str(row.get("confidence_bucket"))
        for row in confidence_rows
        if row.get("confidence_bucket")
    )
    min_confidence_values = [
        lower
        for bucket in confidence_buckets
        for lower in [_bucket_lower(bucket)]
        if lower is not None
    ]
    excluded_rows = thresholds.get("line_buckets_to_exclude_or_discount")
    if not isinstance(excluded_rows, list):
        excluded_rows = []
    excluded_line_buckets = tuple(
        str(row.get("line_bucket"))
        for row in excluded_rows
        if row.get("line_bucket")
    )
    observed_error = thresholds.get("observed_median_confidence_bucket_error")
    min_edge_pct = config.min_edge_pct
    if isinstance(observed_error, (int, float)):
        min_edge_pct = max(min_edge_pct, float(observed_error))
    approval_allowed = (
        recommendation == "conditional_go_for_betting_layer_rebuild"
        and threshold_status == "thresholds_observed_from_calibration"
        and bool(confidence_buckets)
    )
    reason = str(recommendation_payload.get("reason") or "")
    if not approval_allowed:
        reason = (
            reason
            or "Model-only validation did not clear the betting-layer approval gate."
        )
    return ValidationEvidence(
        report_path=report_path,
        report_run_id=str(report.get("run_id") or path_run_id(report_path.parent)),
        recommendation=recommendation,
        threshold_status=threshold_status,
        approval_allowed=approval_allowed,
        min_edge_pct=min_edge_pct,
        confidence_buckets=confidence_buckets,
        min_confidence=(
            min(min_confidence_values) if min_confidence_values else None
        ),
        excluded_line_buckets=excluded_line_buckets,
        reason=reason,
    )


def approval_from_validation_evidence(
    *,
    evidence: ValidationEvidence,
    edge_pct: float,
    confidence: float,
    line: float,
) -> tuple[bool, str]:
    """Return approval status and reason from validation-derived gates."""

    if not evidence.approval_allowed:
        return False, evidence.reason
    if edge_pct < evidence.min_edge_pct:
        return (
            False,
            (
                "Edge is below the validation-derived approval threshold "
                f"({edge_pct:.2%} < {evidence.min_edge_pct:.2%})."
            ),
        )
    bucket = confidence_bucket(confidence)
    if bucket not in evidence.confidence_buckets:
        return (
            False,
            (
                "Model confidence bucket is not approved by validation evidence "
                f"({bucket})."
            ),
        )
    current_line_bucket = line_bucket(line)
    if current_line_bucket in evidence.excluded_line_buckets:
        return (
            False,
            (
                "Line bucket is excluded or requires discounting by validation "
                f"evidence ({current_line_bucket})."
            ),
        )
    return True, "Approved by model-only validation evidence and market edge gates."
