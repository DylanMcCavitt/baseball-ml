"""Approved wager card reporting from daily candidate artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
import json
from pathlib import Path
from typing import Any, Callable

from .ingest.mlb_stats_api import utc_now
from .wager_approval import WagerApprovalSettings, annotate_wager_approval_rows


@dataclass(frozen=True)
class WagerCardResult:
    """Filesystem and count summary for one approved wager card."""

    target_date: date
    run_id: str
    source_daily_candidate_run_id: str
    source_daily_candidates_path: Path
    wager_card_path: Path
    wager_card_metadata_path: Path
    total_candidate_count: int
    approved_count: int
    blocked_count: int
    included_count: int
    include_rejected: bool
    rows: list[dict[str, Any]]
    source_artifact_kind: str = "daily_candidates"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.isoformat()
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    return value


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_ready(row), sort_keys=True))
            handle.write("\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _run_id_from_dir(run_dir: Path) -> str:
    return run_dir.name.split("=", 1)[-1]


def _latest_daily_candidate_run_dir(
    output_root: Path,
    *,
    target_date: date | None,
) -> tuple[date, Path]:
    daily_root = output_root / "normalized" / "daily_candidates"
    if not daily_root.exists():
        raise FileNotFoundError(f"No daily candidate runs were found under {daily_root}.")

    date_dirs = sorted(path for path in daily_root.glob("date=*") if path.is_dir())
    if target_date is not None:
        date_dirs = [
            path
            for path in date_dirs
            if path.name.split("=", 1)[-1] == target_date.isoformat()
        ]
    if not date_dirs:
        target_label = target_date.isoformat() if target_date is not None else "latest"
        raise FileNotFoundError(f"No daily candidate run was found for {target_label}.")

    selected_date_dir = date_dirs[-1]
    run_dirs = sorted(
        path
        for path in selected_date_dir.glob("run=*")
        if path.is_dir() and path.joinpath("daily_candidates.jsonl").exists()
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"No daily_candidates.jsonl artifact was found under {selected_date_dir}."
        )
    resolved_date = date.fromisoformat(selected_date_dir.name.split("=", 1)[-1])
    return resolved_date, run_dirs[-1]


def _latest_edge_candidate_run_dir(
    output_root: Path,
    *,
    target_date: date | None,
) -> tuple[date, Path]:
    edge_root = output_root / "normalized" / "edge_candidates"
    if not edge_root.exists():
        raise FileNotFoundError(f"No edge candidate runs were found under {edge_root}.")

    date_dirs = sorted(path for path in edge_root.glob("date=*") if path.is_dir())
    if target_date is not None:
        date_dirs = [
            path
            for path in date_dirs
            if path.name.split("=", 1)[-1] == target_date.isoformat()
        ]
    if not date_dirs:
        target_label = target_date.isoformat() if target_date is not None else "latest"
        raise FileNotFoundError(f"No edge candidate run was found for {target_label}.")

    selected_date_dir = date_dirs[-1]
    run_dirs = sorted(
        path
        for path in selected_date_dir.glob("run=*")
        if path.is_dir() and path.joinpath("edge_candidates.jsonl").exists()
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"No edge_candidates.jsonl artifact was found under {selected_date_dir}."
        )
    resolved_date = date.fromisoformat(selected_date_dir.name.split("=", 1)[-1])
    return resolved_date, run_dirs[-1]


def _latest_candidate_run_dir(
    output_root: Path,
    *,
    target_date: date | None,
) -> tuple[date, Path, str, str]:
    try:
        resolved_date, run_dir = _latest_daily_candidate_run_dir(
            output_root,
            target_date=target_date,
        )
        return resolved_date, run_dir, "daily_candidates", "daily_candidates.jsonl"
    except FileNotFoundError as daily_error:
        try:
            resolved_date, run_dir = _latest_edge_candidate_run_dir(
                output_root,
                target_date=target_date,
            )
            return resolved_date, run_dir, "edge_candidates", "edge_candidates.jsonl"
        except FileNotFoundError:
            raise daily_error


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _coerce_float(row.get(key))
        if value is not None:
            return value
    return None


def _first_int(row: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _coerce_int(row.get(key))
        if value is not None:
            return value
    return None


def _start_time(row: dict[str, Any]) -> str | None:
    explicit = row.get("commence_time") or row.get("start_time")
    if explicit:
        return str(explicit)
    matchup_key = str(row.get("odds_matchup_key") or "")
    parts = matchup_key.split("|")
    if len(parts) >= 4 and parts[-1]:
        return parts[-1]
    return None


def _approval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if all("wager_approved" in row for row in rows):
        return [dict(row) for row in rows]
    if all("approval_status" in row for row in rows):
        approved_rows: list[dict[str, Any]] = []
        for row in rows:
            approved = (
                str(row.get("approval_status")) == "approved"
                and row.get("approval_allowed") is True
            )
            approved_rows.append(
                {
                    **row,
                    "wager_approved": approved,
                    "wager_gate_status": "approved" if approved else "blocked",
                    "wager_blocked_reason": (
                        "approved"
                        if approved
                        else str(row.get("approval_reason") or "rejected by rebuilt wager gates")
                    ),
                    "wager_gate_notes": (
                        []
                        if approved
                        else [str(row.get("approval_reason") or "rejected by rebuilt wager gates")]
                    ),
                    "approved_rank": (
                        row.get("correlation_group_rank") if approved else None
                    ),
                }
            )
        return approved_rows
    return annotate_wager_approval_rows(
        [dict(row) for row in rows],
        settings=WagerApprovalSettings(),
    )


def _notes_for_row(row: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    raw_notes = row.get("wager_gate_notes")
    if isinstance(raw_notes, list):
        notes.extend(str(note) for note in raw_notes if note)
    elif raw_notes:
        notes.append(str(raw_notes))
    blocked_reason = str(row.get("wager_blocked_reason") or "")
    if blocked_reason and blocked_reason != "approved" and blocked_reason not in notes:
        notes.append(blocked_reason)
    reason = str(row.get("reason") or "")
    if not notes and reason:
        notes.append(reason)
    return notes


def _card_row(row: dict[str, Any], *, source_run_id: str, card_run_id: str) -> dict[str, Any]:
    approved = bool(row.get("wager_approved"))
    rank = row.get("approved_rank") if approved else row.get("slate_rank")
    stake_units = _first_float(row, "kelly_units")
    stake_fraction = _first_float(row, "stake_fraction")
    if stake_units is None and stake_fraction is not None:
        stake_units = round(stake_fraction * WagerApprovalSettings().bankroll_units, 4)
    notes = _notes_for_row(row)
    return {
        "wager_card_run_id": card_run_id,
        "source_daily_candidate_run_id": source_run_id,
        "daily_candidate_id": str(row.get("daily_candidate_id") or row.get("candidate_id") or ""),
        "official_date": str(row.get("official_date") or ""),
        "rank": rank,
        "slate_rank": row.get("slate_rank"),
        "approved_rank": row.get("approved_rank"),
        "status": "approved" if approved else "blocked",
        "wager_approved": approved,
        "blocked_reason": None if approved else str(row.get("wager_blocked_reason") or ""),
        "pitcher": str(row.get("player_name") or ""),
        "player_id": str(row.get("player_id") or ""),
        "pitcher_mlb_id": row.get("pitcher_mlb_id"),
        "game_pk": row.get("game_pk"),
        "start_time": _start_time(row),
        "book": str(row.get("sportsbook_title") or row.get("sportsbook") or ""),
        "sportsbook": str(row.get("sportsbook") or ""),
        "side": str(row.get("selected_side") or ""),
        "line": _first_float(row, "line"),
        "odds": _first_int(row, "selected_odds"),
        "model_probability": _first_float(row, "selected_model_probability", "conf"),
        "market_probability": _first_float(row, "selected_market_probability"),
        "model_projection": _first_float(row, "model_projection", "model_mean"),
        "model_over_probability": _first_float(row, "model_over_probability"),
        "model_under_probability": _first_float(row, "model_under_probability"),
        "model_confidence": _first_float(row, "model_confidence"),
        "model_confidence_bucket": str(row.get("model_confidence_bucket") or ""),
        "edge": _first_float(row, "edge_pct", "edge"),
        "expected_value": _first_float(row, "expected_value_pct"),
        "stake_units": stake_units,
        "stake_fraction": stake_fraction,
        "market_over_probability": _first_float(row, "market_over_probability"),
        "market_under_probability": _first_float(row, "market_under_probability"),
        "no_vig_market_probability": _first_float(row, "selected_market_probability"),
        "approval_reason": str(row.get("approval_reason") or ""),
        "validation_recommendation": str(row.get("validation_recommendation") or ""),
        "validation_threshold_status": str(row.get("validation_threshold_status") or ""),
        "correlation_group_key": str(row.get("correlation_group_key") or ""),
        "correlation_group_size": row.get("correlation_group_size"),
        "correlation_group_rank": row.get("correlation_group_rank"),
        "research_readiness_status": str(row.get("research_readiness_status") or "research_only"),
        "probability_distribution": row.get("probability_distribution")
        or row.get("count_distribution")
        or [],
        "feature_group_contributions": row.get("feature_group_contributions") or [],
        "clv_context": row.get("clv_context") or row.get("clv_outcome"),
        "roi_context": row.get("roi_context") or row.get("paper_result"),
        "notes": notes,
        "note": " | ".join(notes),
        "wager_gate_details": row.get("wager_gate_details") or {},
    }


def _sort_card_row(row: dict[str, Any]) -> tuple[int, int, str]:
    status_order = 0 if row["status"] == "approved" else 1
    rank = row.get("rank")
    return (
        status_order,
        int(rank) if rank is not None else 999999,
        str(row.get("daily_candidate_id") or ""),
    )


def build_wager_card(
    *,
    target_date: date | None = None,
    output_dir: Path | str = "data",
    include_rejected: bool = False,
    now: Callable[[], datetime] = utc_now,
) -> WagerCardResult:
    """Build an approved wager card from the latest daily or rebuilt edge sheet."""
    output_root = Path(output_dir)
    (
        resolved_target_date,
        source_run_dir,
        source_artifact_kind,
        source_artifact_name,
    ) = _latest_candidate_run_dir(
        output_root,
        target_date=target_date,
    )
    source_path = source_run_dir / source_artifact_name
    source_rows = _approval_rows(_load_jsonl_rows(source_path))
    source_run_id = _run_id_from_dir(source_run_dir)
    run_id = now().astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")

    card_rows = [
        _card_row(row, source_run_id=source_run_id, card_run_id=run_id)
        for row in source_rows
        if include_rejected or bool(row.get("wager_approved"))
    ]
    card_rows.sort(key=_sort_card_row)

    approved_count = sum(1 for row in source_rows if bool(row.get("wager_approved")))
    blocked_count = len(source_rows) - approved_count
    normalized_root = (
        output_root
        / "normalized"
        / "wager_card"
        / f"date={resolved_target_date.isoformat()}"
        / f"run={run_id}"
    )
    wager_card_path = normalized_root / "wager_card.jsonl"
    wager_card_metadata_path = normalized_root / "wager_card_metadata.json"
    _write_jsonl(wager_card_path, card_rows)
    _write_json(
        wager_card_metadata_path,
        {
            "run_id": run_id,
            "official_date": resolved_target_date.isoformat(),
            "source_daily_candidate_run_id": source_run_id,
            "source_daily_candidates_path": source_path,
            "source_artifact_kind": source_artifact_kind,
            "include_rejected": include_rejected,
            "total_candidate_count": len(source_rows),
            "approved_count": approved_count,
            "blocked_count": blocked_count,
            "included_count": len(card_rows),
        },
    )

    return WagerCardResult(
        target_date=resolved_target_date,
        run_id=run_id,
        source_daily_candidate_run_id=source_run_id,
        source_daily_candidates_path=source_path,
        wager_card_path=wager_card_path,
        wager_card_metadata_path=wager_card_metadata_path,
        total_candidate_count=len(source_rows),
        approved_count=approved_count,
        blocked_count=blocked_count,
        included_count=len(card_rows),
        include_rejected=include_rejected,
        rows=card_rows,
        source_artifact_kind=source_artifact_kind,
    )


def _format_pct(value: Any) -> str:
    numeric = _coerce_float(value)
    return "n/a" if numeric is None else f"{numeric:.1%}"


def _format_float(value: Any, *, digits: int = 2, suffix: str = "") -> str:
    numeric = _coerce_float(value)
    return "n/a" if numeric is None else f"{numeric:.{digits}f}{suffix}"


def _format_odds(value: Any) -> str:
    odds = _coerce_int(value)
    return "n/a" if odds is None else f"{odds:+d}"


def _clip(value: Any, width: int) -> str:
    text = "n/a" if value is None or value == "" else str(value)
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "."


def _render_rows(rows: list[dict[str, Any]], *, include_status: bool) -> list[str]:
    base_columns: list[tuple[str, int, Callable[[dict[str, Any]], str]]] = [
        ("rk", 4, lambda row: str(row.get("rank") or "")),
        ("pitcher", 18, lambda row: str(row["pitcher"])),
        ("game", 8, lambda row: str(row.get("game_pk") or "")),
        ("start", 20, lambda row: str(row.get("start_time") or "")),
        ("book", 12, lambda row: str(row["book"])),
        ("side", 5, lambda row: str(row["side"]).upper()),
        ("line", 5, lambda row: _format_float(row["line"], digits=1)),
        ("odds", 6, lambda row: _format_odds(row["odds"])),
        ("model", 7, lambda row: _format_pct(row["model_probability"])),
        ("market", 7, lambda row: _format_pct(row["market_probability"])),
        ("edge", 7, lambda row: _format_pct(row["edge"])),
        ("ev", 7, lambda row: _format_pct(row["expected_value"])),
        (
            "stake",
            11,
            lambda row: (
                f"{_format_float(row['stake_units'], digits=2, suffix='u')}/"
                f"{_format_pct(row['stake_fraction'])}"
            ),
        ),
        ("notes", 28, lambda row: str(row.get("note") or "")),
    ]
    if include_status:
        base_columns.insert(1, ("status", 8, lambda row: str(row["status"])))

    header = "  ".join(label.ljust(width) for label, width, _ in base_columns)
    separator = "  ".join("-" * width for _, width, _ in base_columns)
    lines = [header, separator]
    for row in rows:
        lines.append(
            "  ".join(
                _clip(renderer(row), width).ljust(width)
                for _, width, renderer in base_columns
            )
        )
    return lines


def render_wager_card_summary(result: WagerCardResult) -> str:
    """Return the terminal table and artifact paths for a wager card."""
    lines = [
        f"Approved wager card for {result.target_date.isoformat()}",
        f"run_id={result.run_id}",
        f"source_artifact_kind={result.source_artifact_kind}",
        f"source_candidate_run_id={result.source_daily_candidate_run_id}",
        f"total_candidates={result.total_candidate_count}",
        f"approved_wagers={result.approved_count}",
        f"blocked_candidates={result.blocked_count}",
        f"included_rows={result.included_count}",
        f"wager_card_path={result.wager_card_path}",
        f"wager_card_metadata_path={result.wager_card_metadata_path}",
    ]

    approved_rows = [row for row in result.rows if row["status"] == "approved"]
    blocked_rows = [row for row in result.rows if row["status"] == "blocked"]
    if approved_rows:
        lines.append("")
        lines.append("Approved wagers")
        lines.extend(_render_rows(approved_rows, include_status=False))
    else:
        lines.append("")
        lines.append("No approved wagers passed the final wager gates.")

    if result.include_rejected:
        lines.append("")
        lines.append("Blocked candidates")
        if blocked_rows:
            lines.extend(_render_rows(blocked_rows, include_status=True))
        else:
            lines.append("No blocked candidates were included.")
    return "\n".join(lines)
