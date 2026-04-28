"""Compare projections to market prices and persist candidate edge rows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from .config import (
    DEVIG_MODE_CONSENSUS,
    DEVIG_MODE_PER_BOOK,
    DEVIG_MODE_TIGHTEST_BOOK,
    StackConfig,
)
from .betting import (
    approval_from_validation_evidence,
    find_latest_distribution_model_run_for_date,
    find_latest_validation_report,
    optional_datetime,
    path_run_id,
    probabilities_for_line,
    validation_evidence_from_report,
)
from .markets import EdgeDecision, PropLine, PropProjection, ProjectionInputRef
from .pricing import (
    book_hold,
    capped_fractional_kelly,
    devig_consensus_two_way,
    devig_two_way,
    expected_value,
    fair_american_odds,
)


@dataclass(frozen=True)
class EdgeCandidateBuildResult:
    """Filesystem output summary for one edge-candidate build."""

    target_date: date
    run_id: str
    model_version: str
    model_run_id: str
    line_snapshots_path: Path
    model_path: Path
    ladder_probabilities_path: Path
    edge_candidates_path: Path
    line_count: int
    scored_line_count: int
    actionable_count: int
    below_threshold_count: int
    skipped_line_count: int
    approved_count: int = 0
    rejected_count: int = 0


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
    return value


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_ready(row), sort_keys=True))
            handle.write("\n")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _path_run_id(run_dir: Path) -> str:
    return path_run_id(run_dir)


def _latest_run_dir(root: Path) -> Path:
    run_dirs = sorted(path for path in root.glob("run=*") if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No normalized runs were found under {root}.")
    return run_dirs[-1]


def _find_latest_model_run_for_date(output_root: Path, *, target_date: date) -> Path:
    distribution_run_dir = find_latest_distribution_model_run_for_date(
        output_root,
        target_date=target_date.isoformat(),
    )
    if distribution_run_dir is not None:
        return distribution_run_dir

    model_root = output_root / "normalized" / "starter_strikeout_baseline"
    run_dirs = sorted(
        path
        for path in model_root.rglob("run=*")
        if path.is_dir() and path.joinpath("ladder_probabilities.jsonl").exists()
    )
    target_date_string = target_date.isoformat()
    for run_dir in reversed(run_dirs):
        ladder_rows = _load_jsonl_rows(run_dir / "ladder_probabilities.jsonl")
        if any(row.get("official_date") == target_date_string for row in ladder_rows):
            return run_dir
    raise FileNotFoundError(
        "No starter strikeout baseline run contains ladder probabilities for "
        f"{target_date_string}."
    )


def _compute_market_probabilities(
    line: PropLine,
    *,
    peer_lines: Sequence[PropLine],
    devig_mode: str,
) -> tuple[float, float, list[str]]:
    """Return (market_over, market_under, consensus_books) for the devig mode.

    ``peer_lines`` are other book snapshots for the same (event, player,
    market, line). They are ignored in ``per_book`` mode so legacy callers
    that pass no peers still see the prior behavior.
    """
    if devig_mode == DEVIG_MODE_PER_BOOK:
        market_over, market_under = devig_two_way(line.over_odds, line.under_odds)
        return market_over, market_under, [line.sportsbook]

    books_by_key: dict[str, PropLine] = {line.sportsbook: line}
    for peer in peer_lines:
        books_by_key.setdefault(peer.sportsbook, peer)
    ordered_books = sorted(books_by_key.values(), key=lambda book: book.sportsbook)

    if devig_mode == DEVIG_MODE_TIGHTEST_BOOK:
        tightest = min(
            ordered_books,
            key=lambda book: (
                book_hold(book.over_odds, book.under_odds),
                book.sportsbook,
            ),
        )
        market_over, market_under = devig_two_way(
            tightest.over_odds, tightest.under_odds
        )
        return market_over, market_under, [tightest.sportsbook]

    if devig_mode == DEVIG_MODE_CONSENSUS:
        market_over, market_under = devig_consensus_two_way(
            [(book.over_odds, book.under_odds) for book in ordered_books]
        )
        return market_over, market_under, [book.sportsbook for book in ordered_books]

    raise ValueError(f"unknown devig_mode: {devig_mode!r}")


def analyze_projection(
    line: PropLine,
    projection: PropProjection,
    config: Optional[StackConfig] = None,
    *,
    peer_lines: Sequence[PropLine] = (),
) -> dict[str, Any]:
    """Return full pricing details for a model-vs-market comparison.

    ``peer_lines`` are other book snapshots for the same (event, player,
    market, line). They are only consulted when ``config.devig_mode`` is
    ``tightest_book`` or ``consensus`` so existing single-book callers keep
    working without passing peers.
    """
    if config is None:
        config = StackConfig()
    if line.selection_key != projection.selection_key:
        raise ValueError("line and projection must reference the same prop contract")
    if projection.input_ref.features_as_of > line.captured_at:
        raise ValueError(
            "projection.input_ref.features_as_of must be on or before line.captured_at"
        )
    if projection.generated_at > line.captured_at:
        raise ValueError("projection.generated_at must be on or before line.captured_at")

    market_over, market_under, consensus_books = _compute_market_probabilities(
        line,
        peer_lines=peer_lines,
        devig_mode=config.devig_mode,
    )
    over_edge = projection.over_probability - market_over
    under_edge = projection.under_probability - market_under

    if over_edge >= under_edge:
        side = "over"
        edge_pct = over_edge
        selected_model_probability = projection.over_probability
        selected_market_probability = market_over
        selected_odds = line.over_odds
    else:
        side = "under"
        edge_pct = under_edge
        selected_model_probability = projection.under_probability
        selected_market_probability = market_under
        selected_odds = line.under_odds

    if not 0.0 < selected_model_probability < 1.0:
        raise ValueError(
            "chosen_probability must be strictly between 0 and 1 to derive fair_odds"
        )

    expected_value_pct = expected_value(selected_model_probability, selected_odds)
    uncapped_stake_fraction = capped_fractional_kelly(
        selected_model_probability,
        selected_odds,
        fraction=config.kelly_fraction,
        max_fraction=1.0,
    )
    stake_fraction = min(uncapped_stake_fraction, config.max_bet_fraction)
    clears_min_edge = edge_pct >= config.min_edge_pct
    fair_odds = fair_american_odds(selected_model_probability)

    if clears_min_edge:
        reason = (
            f"{side} clears minimum edge threshold "
            f"({edge_pct:.2%} >= {config.min_edge_pct:.2%})"
        )
    else:
        reason = (
            f"{side} does not clear minimum edge threshold "
            f"({edge_pct:.2%} < {config.min_edge_pct:.2%})"
        )

    return {
        "side": side,
        "edge_pct": edge_pct,
        "expected_value_pct": expected_value_pct,
        "uncapped_stake_fraction": uncapped_stake_fraction,
        "stake_fraction": stake_fraction,
        "fair_odds": fair_odds,
        "market_over_probability": market_over,
        "market_under_probability": market_under,
        "market_consensus_books": list(consensus_books),
        "devig_mode": config.devig_mode,
        "selected_model_probability": selected_model_probability,
        "selected_market_probability": selected_market_probability,
        "selected_odds": selected_odds,
        "clears_min_edge": clears_min_edge,
        "reason": reason,
    }


def evaluate_projection(
    line: PropLine,
    projection: PropProjection,
    config: Optional[StackConfig] = None,
) -> Optional[EdgeDecision]:
    """Return the best actionable side if an edge clears the threshold."""
    analysis = analyze_projection(line, projection, config=config)
    if not analysis["clears_min_edge"]:
        return None
    return EdgeDecision(
        side=str(analysis["side"]),
        edge_pct=float(analysis["edge_pct"]),
        expected_value_pct=float(analysis["expected_value_pct"]),
        stake_fraction=float(analysis["stake_fraction"]),
        fair_odds=int(analysis["fair_odds"]),
        reason=str(analysis["reason"]),
    )


def _prop_line_from_snapshot_row(row: dict[str, Any]) -> PropLine:
    return PropLine(
        line_snapshot_id=str(row["line_snapshot_id"]),
        sportsbook=str(row["sportsbook"]),
        event_id=str(row["event_id"]),
        player_id=str(row["player_id"]),
        player_name=str(row["player_name"]),
        market=str(row["market"]),
        line=float(row["line"]),
        over_odds=int(row["over_odds"]),
        under_odds=int(row["under_odds"]),
        captured_at=_parse_datetime(str(row["captured_at"])),
    )


def _ladder_probability_for_line(
    ladder_rows: list[dict[str, Any]],
    *,
    line: float,
) -> dict[str, Any] | None:
    rounded_line = round(line, 6)
    for ladder_row in ladder_rows:
        if round(float(ladder_row["line"]), 6) == rounded_line:
            return ladder_row
    return None


def _model_output_lookup_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["official_date"]),
        int(row["game_pk"]),
        int(row["pitcher_id"]),
    )


def _line_model_lookup_key(line_row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(line_row["official_date"]),
        int(line_row["game_pk"]),
        int(line_row["pitcher_mlb_id"]),
    )


def _projection_timestamps_from_model_output(
    model_row: dict[str, Any],
    *,
    fallback_captured_at: datetime,
) -> tuple[datetime, datetime, str]:
    features_as_of = optional_datetime(model_row.get("features_as_of"))
    projection_generated_at = optional_datetime(model_row.get("projection_generated_at"))
    if features_as_of is None or projection_generated_at is None:
        return (
            fallback_captured_at,
            fallback_captured_at,
            "missing_projection_timestamps",
        )
    if features_as_of > fallback_captured_at or projection_generated_at > fallback_captured_at:
        return features_as_of, projection_generated_at, "invalid_after_selected_line"
    return features_as_of, projection_generated_at, "ok"


def _distribution_correlation_group_key(row: dict[str, Any]) -> str:
    if row.get("game_pk") is not None and row.get("pitcher_mlb_id") is not None:
        game_key = str(row["game_pk"])
        pitcher_key = str(row["pitcher_mlb_id"])
    else:
        game_key = str(row.get("event_id") or "")
        pitcher_key = str(row.get("player_id") or "")
    return "|".join(
        [
            str(row["official_date"]),
            game_key,
            pitcher_key,
            f"{float(row['line']):.6f}",
        ]
    )


def _annotate_distribution_correlation_groups(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("evaluation_status") in {"actionable", "below_threshold"}:
            grouped.setdefault(str(row["correlation_group_key"]), []).append(row)
    for group_rows in grouped.values():
        ranked = sorted(
            group_rows,
            key=lambda row: (
                -float(row.get("edge_pct") or 0.0),
                str(row.get("sportsbook") or ""),
                str(row.get("line_snapshot_id") or ""),
            ),
        )
        for rank, row in enumerate(ranked, start=1):
            row["correlation_group_size"] = len(ranked)
            row["correlation_group_rank"] = rank
            if (
                rank > 1
                and row.get("approval_status") == "approved"
                and row.get("approval_allowed") is True
            ):
                row["approval_status"] = "rejected"
                row["approval_allowed"] = False
                row["approval_reason"] = (
                    "Rejected as a correlated duplicate within the same "
                    "pitcher/game/line group."
                )


def _build_distribution_edge_candidates_for_date(
    *,
    target_date: date,
    output_dir: Path | str,
    model_run_dir: Path,
    config: StackConfig,
) -> EdgeCandidateBuildResult:
    output_root = Path(output_dir)
    odds_root = output_root / "normalized" / "the_odds_api" / f"date={target_date.isoformat()}"
    odds_run_dir = _latest_run_dir(odds_root)
    line_snapshots_path = odds_run_dir / "prop_line_snapshots.jsonl"
    line_rows = _load_jsonl_rows(line_snapshots_path)

    selected_model_path = model_run_dir / "selected_model.json"
    model_outputs_path = model_run_dir / "model_outputs.jsonl"
    selected_model = _load_json(selected_model_path)
    model_version = str(selected_model["model_version"])
    model_run_id = _path_run_id(model_run_dir)
    model_rows = _load_jsonl_rows(model_outputs_path)
    model_lookup = {
        _model_output_lookup_key(row): row
        for row in model_rows
        if row.get("game_pk") is not None and row.get("pitcher_id") is not None
    }
    validation_path = find_latest_validation_report(output_root)
    validation_evidence = validation_evidence_from_report(validation_path, config=config)

    peer_lookup: dict[
        tuple[str, str, str, str, float], list[PropLine]
    ] = {}
    for peer_row in line_rows:
        try:
            peer_line = _prop_line_from_snapshot_row(peer_row)
        except (KeyError, TypeError, ValueError):
            continue
        peer_lookup.setdefault(_multi_book_line_group_key(peer_row), []).append(
            peer_line
        )

    candidate_rows: list[dict[str, Any]] = []
    actionable_count = 0
    below_threshold_count = 0
    skipped_line_count = 0

    for line_row in line_rows:
        base_row = _build_candidate_base_row(
            line_row,
            model_version=model_version,
            model_run_id=model_run_id,
        )
        game_pk = line_row.get("game_pk")
        pitcher_mlb_id = line_row.get("pitcher_mlb_id")
        if game_pk is None or pitcher_mlb_id is None:
            skipped_line_count += 1
            candidate_rows.append(
                _build_skipped_candidate_row(
                    line_row,
                    model_version=model_version,
                    model_run_id=model_run_id,
                    evaluation_status="missing_join_keys",
                    reason="Line snapshot is missing mapped game or pitcher identifiers.",
                    extra_fields={
                        "approval_status": "rejected",
                        "approval_allowed": False,
                        "approval_reason": "Cannot approve without mapped game and pitcher identifiers.",
                    },
                )
            )
            continue

        model_row = model_lookup.get(_line_model_lookup_key(line_row))
        if model_row is None:
            skipped_line_count += 1
            candidate_rows.append(
                _build_skipped_candidate_row(
                    line_row,
                    model_version=model_version,
                    model_run_id=model_run_id,
                    evaluation_status="missing_projection",
                    reason="No rebuilt strikeout distribution was found for this line snapshot.",
                    extra_fields={
                        "approval_status": "rejected",
                        "approval_allowed": False,
                        "approval_reason": "Cannot approve without a model distribution.",
                    },
                )
            )
            continue

        if str(model_row.get("split")) == "train":
            skipped_line_count += 1
            candidate_rows.append(
                _build_skipped_candidate_row(
                    line_row,
                    model_version=model_version,
                    model_run_id=model_run_id,
                    evaluation_status="train_split_projection",
                    reason=(
                        "The matched rebuilt distribution belongs to the training "
                        "split and is not honest for historical edge evaluation."
                    ),
                    extra_fields={
                        "data_split": "train",
                        "approval_status": "rejected",
                        "approval_allowed": False,
                        "approval_reason": "Cannot approve a training-split projection.",
                    },
                )
            )
            continue

        try:
            distribution = list(model_row["probability_distribution"])
            line_probabilities = probabilities_for_line(
                distribution,
                line=float(line_row["line"]),
            )
            prop_line = _prop_line_from_snapshot_row(line_row)
            features_as_of, projection_generated_at, timestamp_status = (
                _projection_timestamps_from_model_output(
                    model_row,
                    fallback_captured_at=prop_line.captured_at,
                )
            )
            if timestamp_status == "invalid_after_selected_line":
                raise ValueError(
                    "The matched rebuilt projection timestamp is after the line "
                    "snapshot, so the comparison would leak future information."
                )
            projection = PropProjection(
                event_id=str(line_row["event_id"]),
                player_id=str(line_row["player_id"]),
                market=str(line_row["market"]),
                line=float(line_row["line"]),
                mean=float(model_row["point_projection"]),
                over_probability=float(line_probabilities["over_probability"]),
                under_probability=float(line_probabilities["under_probability"]),
                model_version=model_version,
                input_ref=ProjectionInputRef(
                    lineup_snapshot_id=str(model_row.get("lineup_snapshot_id") or "unknown"),
                    feature_row_id=str(
                        model_row.get("feature_row_id")
                        or model_row.get("training_row_id")
                        or "unknown"
                    ),
                    features_as_of=features_as_of,
                ),
                generated_at=projection_generated_at,
            )
            peer_lines = [
                peer
                for peer in peer_lookup.get(
                    _multi_book_line_group_key(line_row), []
                )
                if peer.line_snapshot_id != prop_line.line_snapshot_id
            ]
            analysis = analyze_projection(
                prop_line,
                projection,
                config=config,
                peer_lines=peer_lines,
            )
        except (KeyError, TypeError, ValueError) as error:
            skipped_line_count += 1
            candidate_rows.append(
                _build_skipped_candidate_row(
                    line_row,
                    model_version=model_version,
                    model_run_id=model_run_id,
                    evaluation_status="invalid_projection",
                    reason=str(error),
                    extra_fields={
                        "approval_status": "rejected",
                        "approval_allowed": False,
                        "approval_reason": str(error),
                    },
                )
            )
            continue

        if analysis["clears_min_edge"]:
            actionable_count += 1
            evaluation_status = "actionable"
        else:
            below_threshold_count += 1
            evaluation_status = "below_threshold"

        confidence = max(
            float(line_probabilities["over_probability"]),
            float(line_probabilities["under_probability"]),
        )
        approved, approval_reason = approval_from_validation_evidence(
            evidence=validation_evidence,
            edge_pct=float(analysis["edge_pct"]),
            confidence=confidence,
            line=float(line_row["line"]),
        )
        if timestamp_status == "missing_projection_timestamps":
            approved = False
            approval_reason = (
                "Projection timestamps are missing from the rebuilt model output, "
                "so the row can be priced but not approved."
            )

        candidate_row = {
            **base_row,
            "evaluation_status": evaluation_status,
            "approval_status": "approved" if approved else "rejected",
            "approval_allowed": approved,
            "approval_reason": approval_reason,
            "validation_report_path": validation_evidence.report_path,
            "validation_report_run_id": validation_evidence.report_run_id,
            "validation_recommendation": validation_evidence.recommendation,
            "validation_threshold_status": validation_evidence.threshold_status,
            "validation_min_edge_pct": round(validation_evidence.min_edge_pct, 6),
            "validation_min_confidence": validation_evidence.min_confidence,
            "validation_confidence_buckets": list(
                validation_evidence.confidence_buckets
            ),
            "data_split": str(model_row.get("split")),
            "feature_row_id": str(
                model_row.get("feature_row_id")
                or model_row.get("training_row_id")
                or "unknown"
            ),
            "lineup_snapshot_id": model_row.get("lineup_snapshot_id"),
            "features_as_of": features_as_of,
            "projection_generated_at": projection_generated_at,
            "projection_timestamp_status": timestamp_status,
            "model_projection": round(float(model_row["point_projection"]), 6),
            "model_mean": round(float(model_row["point_projection"]), 6),
            "model_confidence": round(confidence, 6),
            "count_distribution": distribution,
            "probability_distribution": distribution,
            "raw_model_over_probability": round(
                float(line_probabilities["over_probability"]),
                6,
            ),
            "raw_model_under_probability": round(
                float(line_probabilities["under_probability"]),
                6,
            ),
            "model_over_probability": round(
                float(line_probabilities["over_probability"]),
                6,
            ),
            "model_under_probability": round(
                float(line_probabilities["under_probability"]),
                6,
            ),
            "market_over_probability": round(
                float(analysis["market_over_probability"]),
                6,
            ),
            "market_under_probability": round(
                float(analysis["market_under_probability"]),
                6,
            ),
            "market_consensus_books": list(analysis["market_consensus_books"]),
            "devig_mode": str(analysis["devig_mode"]),
            "selected_side": str(analysis["side"]),
            "selected_model_probability": round(
                float(analysis["selected_model_probability"]),
                6,
            ),
            "selected_market_probability": round(
                float(analysis["selected_market_probability"]),
                6,
            ),
            "selected_odds": int(analysis["selected_odds"]),
            "edge_pct": round(float(analysis["edge_pct"]), 6),
            "expected_value_pct": round(float(analysis["expected_value_pct"]), 6),
            "uncapped_stake_fraction": round(
                float(analysis["uncapped_stake_fraction"]),
                6,
            ),
            "stake_fraction": round(float(analysis["stake_fraction"]), 6),
            "fair_odds": int(analysis["fair_odds"]),
            "clears_min_edge": bool(analysis["clears_min_edge"]),
            "reason": str(analysis["reason"]),
            "correlation_group_key": _distribution_correlation_group_key(line_row),
            "correlation_group_size": 1,
            "correlation_group_rank": 1,
        }
        candidate_rows.append(candidate_row)

    _annotate_distribution_correlation_groups(candidate_rows)
    approved_count = sum(1 for row in candidate_rows if row.get("approval_status") == "approved")
    rejected_count = sum(1 for row in candidate_rows if row.get("approval_status") == "rejected")

    candidate_rows.sort(
        key=lambda row: (
            0 if row.get("approval_status") == "approved" else 1,
            0 if row["evaluation_status"] == "actionable" else 1,
            -float(row.get("edge_pct") or 0.0),
            str(row["line_snapshot_id"]),
        )
    )

    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    normalized_root = (
        output_root
        / "normalized"
        / "edge_candidates"
        / f"date={target_date.isoformat()}"
        / f"run={run_id}"
    )
    edge_candidates_path = normalized_root / "edge_candidates.jsonl"
    _write_jsonl(edge_candidates_path, candidate_rows)

    return EdgeCandidateBuildResult(
        target_date=target_date,
        run_id=run_id,
        model_version=model_version,
        model_run_id=model_run_id,
        line_snapshots_path=line_snapshots_path,
        model_path=selected_model_path,
        ladder_probabilities_path=model_outputs_path,
        edge_candidates_path=edge_candidates_path,
        line_count=len(line_rows),
        scored_line_count=actionable_count + below_threshold_count,
        actionable_count=actionable_count,
        below_threshold_count=below_threshold_count,
        skipped_line_count=skipped_line_count,
        approved_count=approved_count,
        rejected_count=rejected_count,
    )


def _build_candidate_base_row(
    line_row: dict[str, Any],
    *,
    model_version: str,
    model_run_id: str,
) -> dict[str, Any]:
    return {
        "candidate_id": f"{line_row['line_snapshot_id']}|{model_version}",
        "official_date": str(line_row["official_date"]),
        "line_snapshot_id": str(line_row["line_snapshot_id"]),
        "model_version": model_version,
        "model_run_id": model_run_id,
        "sportsbook": str(line_row["sportsbook"]),
        "sportsbook_title": str(line_row.get("sportsbook_title") or line_row["sportsbook"]),
        "event_id": str(line_row["event_id"]),
        "game_pk": line_row.get("game_pk"),
        "odds_matchup_key": line_row.get("odds_matchup_key"),
        "match_status": line_row.get("match_status"),
        "player_id": str(line_row["player_id"]),
        "pitcher_mlb_id": line_row.get("pitcher_mlb_id"),
        "player_name": str(line_row["player_name"]),
        "market": str(line_row["market"]),
        "line": round(float(line_row["line"]), 6),
        "over_odds": int(line_row["over_odds"]),
        "under_odds": int(line_row["under_odds"]),
        "captured_at": _parse_datetime(str(line_row["captured_at"])),
    }


def _build_skipped_candidate_row(
    line_row: dict[str, Any],
    *,
    model_version: str,
    model_run_id: str,
    evaluation_status: str,
    reason: str,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = _build_candidate_base_row(
        line_row,
        model_version=model_version,
        model_run_id=model_run_id,
    )
    row.update(
        {
            "evaluation_status": evaluation_status,
            "reason": reason,
        }
    )
    if extra_fields is not None:
        row.update(extra_fields)
    return row


def _multi_book_line_group_key(
    row: dict[str, Any],
) -> tuple[str, str, str, str, float]:
    return (
        str(row["official_date"]),
        str(row["event_id"]),
        str(row["player_id"]),
        str(row["market"]),
        round(float(row["line"]), 6),
    )


def build_edge_candidates_for_date(
    *,
    target_date: date,
    output_dir: Path | str = "data",
    model_run_dir: Path | str | None = None,
    config: Optional[StackConfig] = None,
) -> EdgeCandidateBuildResult:
    """Build replayable edge-candidate rows for one official date."""
    output_root = Path(output_dir)
    odds_root = output_root / "normalized" / "the_odds_api" / f"date={target_date.isoformat()}"
    odds_run_dir = _latest_run_dir(odds_root)
    line_snapshots_path = odds_run_dir / "prop_line_snapshots.jsonl"
    line_rows = _load_jsonl_rows(line_snapshots_path)

    resolved_model_run_dir = (
        Path(model_run_dir)
        if model_run_dir is not None
        else _find_latest_model_run_for_date(output_root, target_date=target_date)
    )
    if resolved_model_run_dir.joinpath("model_outputs.jsonl").exists():
        if config is None:
            config = StackConfig()
        return _build_distribution_edge_candidates_for_date(
            target_date=target_date,
            output_dir=output_dir,
            model_run_dir=resolved_model_run_dir,
            config=config,
        )

    model_path = resolved_model_run_dir / "baseline_model.json"
    ladder_probabilities_path = resolved_model_run_dir / "ladder_probabilities.jsonl"
    model_artifact = _load_json(model_path)
    ladder_rows = _load_jsonl_rows(ladder_probabilities_path)
    model_version = str(model_artifact["model_version"])
    model_run_id = _path_run_id(resolved_model_run_dir)
    if config is None:
        config = StackConfig()

    peer_lookup: dict[
        tuple[str, str, str, str, float], list[PropLine]
    ] = {}
    for peer_row in line_rows:
        try:
            peer_line = _prop_line_from_snapshot_row(peer_row)
        except (KeyError, TypeError, ValueError):
            continue
        peer_lookup.setdefault(_multi_book_line_group_key(peer_row), []).append(
            peer_line
        )

    ladder_lookup: dict[tuple[str, int, int], dict[str, Any]] = {}
    for ladder_row in ladder_rows:
        ladder_lookup[
            (
                str(ladder_row["official_date"]),
                int(ladder_row["game_pk"]),
                int(ladder_row["pitcher_id"]),
            )
        ] = ladder_row

    candidate_rows: list[dict[str, Any]] = []
    actionable_count = 0
    below_threshold_count = 0
    skipped_line_count = 0
    scored_line_count = 0

    for line_row in line_rows:
        base_row = _build_candidate_base_row(
            line_row,
            model_version=model_version,
            model_run_id=model_run_id,
        )
        game_pk = line_row.get("game_pk")
        pitcher_mlb_id = line_row.get("pitcher_mlb_id")
        if game_pk is None or pitcher_mlb_id is None:
            skipped_line_count += 1
            candidate_rows.append(
                _build_skipped_candidate_row(
                    line_row,
                    model_version=model_version,
                    model_run_id=model_run_id,
                    evaluation_status="missing_join_keys",
                    reason="Line snapshot is missing mapped game or pitcher identifiers.",
                )
            )
            continue

        ladder_row = ladder_lookup.get(
            (
                str(line_row["official_date"]),
                int(game_pk),
                int(pitcher_mlb_id),
            )
        )
        if ladder_row is None:
            skipped_line_count += 1
            candidate_rows.append(
                _build_skipped_candidate_row(
                    line_row,
                    model_version=model_version,
                    model_run_id=model_run_id,
                    evaluation_status="missing_projection",
                    reason="No ladder probabilities were found for this line snapshot.",
                )
            )
            continue

        if str(ladder_row["split"]) == "train":
            skipped_line_count += 1
            candidate_rows.append(
                _build_skipped_candidate_row(
                    line_row,
                    model_version=model_version,
                    model_run_id=model_run_id,
                    evaluation_status="train_split_projection",
                    reason=(
                        "The matched projection only exists in the training split and "
                        "is not honest for historical edge evaluation."
                    ),
                    extra_fields={"data_split": "train"},
                )
            )
            continue

        raw_probability = _ladder_probability_for_line(
            list(ladder_row["ladder_probabilities"]),
            line=float(line_row["line"]),
        )
        calibrated_probability = _ladder_probability_for_line(
            list(ladder_row["calibrated_ladder_probabilities"]),
            line=float(line_row["line"]),
        )
        if raw_probability is None or calibrated_probability is None:
            skipped_line_count += 1
            candidate_rows.append(
                _build_skipped_candidate_row(
                    line_row,
                    model_version=model_version,
                    model_run_id=model_run_id,
                    evaluation_status="missing_line_probability",
                    reason="The selected book line is not present in the saved ladder probabilities.",
                )
            )
            continue

        lineup_snapshot_id = ladder_row.get("lineup_snapshot_id")
        if lineup_snapshot_id is None:
            skipped_line_count += 1
            candidate_rows.append(
                _build_skipped_candidate_row(
                    line_row,
                    model_version=model_version,
                    model_run_id=model_run_id,
                    evaluation_status="missing_lineup_snapshot_id",
                    reason="The matched projection does not carry a lineup snapshot reference.",
                )
            )
            continue

        try:
            prop_line = _prop_line_from_snapshot_row(line_row)
            projection = PropProjection(
                event_id=str(line_row["event_id"]),
                player_id=str(line_row["player_id"]),
                market=str(line_row["market"]),
                line=float(line_row["line"]),
                mean=float(ladder_row["model_mean"]),
                over_probability=float(calibrated_probability["over_probability"]),
                under_probability=float(calibrated_probability["under_probability"]),
                model_version=model_version,
                input_ref=ProjectionInputRef(
                    lineup_snapshot_id=str(lineup_snapshot_id),
                    feature_row_id=str(ladder_row["feature_row_id"]),
                    features_as_of=_parse_datetime(str(ladder_row["features_as_of"])),
                ),
                generated_at=_parse_datetime(str(ladder_row["projection_generated_at"])),
            )
            peer_lines = [
                peer
                for peer in peer_lookup.get(
                    _multi_book_line_group_key(line_row), []
                )
                if peer.line_snapshot_id != prop_line.line_snapshot_id
            ]
            analysis = analyze_projection(
                prop_line,
                projection,
                config=config,
                peer_lines=peer_lines,
            )
        except (TypeError, ValueError) as error:
            skipped_line_count += 1
            candidate_rows.append(
                _build_skipped_candidate_row(
                    line_row,
                    model_version=model_version,
                    model_run_id=model_run_id,
                    evaluation_status="invalid_projection",
                    reason=str(error),
                )
            )
            continue

        scored_line_count += 1
        if analysis["clears_min_edge"]:
            actionable_count += 1
            evaluation_status = "actionable"
        else:
            below_threshold_count += 1
            evaluation_status = "below_threshold"

        candidate_row = {
            **base_row,
            "evaluation_status": evaluation_status,
            "data_split": str(ladder_row["split"]),
            "feature_row_id": str(ladder_row["feature_row_id"]),
            "lineup_snapshot_id": str(lineup_snapshot_id),
            "features_as_of": _parse_datetime(str(ladder_row["features_as_of"])),
            "projection_generated_at": _parse_datetime(
                str(ladder_row["projection_generated_at"])
            ),
            "model_mean": round(float(ladder_row["model_mean"]), 6),
            "count_distribution": dict(ladder_row["count_distribution"]),
            "probability_calibration": dict(ladder_row["probability_calibration"]),
            "raw_model_over_probability": round(
                float(raw_probability["over_probability"]),
                6,
            ),
            "raw_model_under_probability": round(
                float(raw_probability["under_probability"]),
                6,
            ),
            "model_over_probability": round(
                float(calibrated_probability["over_probability"]),
                6,
            ),
            "model_under_probability": round(
                float(calibrated_probability["under_probability"]),
                6,
            ),
            "market_over_probability": round(
                float(analysis["market_over_probability"]),
                6,
            ),
            "market_under_probability": round(
                float(analysis["market_under_probability"]),
                6,
            ),
            "market_consensus_books": list(analysis["market_consensus_books"]),
            "devig_mode": str(analysis["devig_mode"]),
            "selected_side": str(analysis["side"]),
            "selected_model_probability": round(
                float(analysis["selected_model_probability"]),
                6,
            ),
            "selected_market_probability": round(
                float(analysis["selected_market_probability"]),
                6,
            ),
            "selected_odds": int(analysis["selected_odds"]),
            "edge_pct": round(float(analysis["edge_pct"]), 6),
            "expected_value_pct": round(float(analysis["expected_value_pct"]), 6),
            "uncapped_stake_fraction": round(
                float(analysis["uncapped_stake_fraction"]),
                6,
            ),
            "stake_fraction": round(float(analysis["stake_fraction"]), 6),
            "fair_odds": int(analysis["fair_odds"]),
            "clears_min_edge": bool(analysis["clears_min_edge"]),
            "reason": str(analysis["reason"]),
        }
        candidate_rows.append(candidate_row)

    candidate_rows.sort(
        key=lambda row: (
            0 if row["evaluation_status"] == "actionable" else 1,
            -float(row.get("edge_pct") or 0.0),
            str(row["line_snapshot_id"]),
        )
    )

    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    normalized_root = (
        output_root
        / "normalized"
        / "edge_candidates"
        / f"date={target_date.isoformat()}"
        / f"run={run_id}"
    )
    edge_candidates_path = normalized_root / "edge_candidates.jsonl"
    _write_jsonl(edge_candidates_path, candidate_rows)

    return EdgeCandidateBuildResult(
        target_date=target_date,
        run_id=run_id,
        model_version=model_version,
        model_run_id=model_run_id,
        line_snapshots_path=line_snapshots_path,
        model_path=model_path,
        ladder_probabilities_path=ladder_probabilities_path,
        edge_candidates_path=edge_candidates_path,
        line_count=len(line_rows),
        scored_line_count=scored_line_count,
        actionable_count=actionable_count,
        below_threshold_count=below_threshold_count,
        skipped_line_count=skipped_line_count,
    )
