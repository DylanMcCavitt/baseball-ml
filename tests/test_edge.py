import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from mlb_props_stack.edge import (
    analyze_projection,
    build_edge_candidates_for_date,
    evaluate_projection,
)
from mlb_props_stack.markets import PropLine, PropProjection, ProjectionInputRef


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, sort_keys=True)}\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def make_line(
    *,
    line_snapshot_id: str = "line-snapshot-1",
    player_id: str = "pitcher-1",
    line: float = 5.5,
    captured_at: datetime | None = None,
) -> PropLine:
    if captured_at is None:
        captured_at = datetime(2026, 4, 20, 12, 0, 0)
    return PropLine(
        line_snapshot_id=line_snapshot_id,
        sportsbook="draftkings",
        event_id="game-1",
        player_id=player_id,
        player_name="Ace Starter",
        market="pitcher_strikeouts",
        line=line,
        over_odds=-110,
        under_odds=-110,
        captured_at=captured_at,
    )


def make_projection(
    *,
    player_id: str = "pitcher-1",
    line: float = 5.5,
    over_probability: float = 0.58,
    under_probability: float = 0.42,
    feature_timestamp: datetime | None = None,
    generated_at: datetime | None = None,
) -> PropProjection:
    if feature_timestamp is None:
        feature_timestamp = datetime(2026, 4, 20, 11, 45, 0)
    if generated_at is None:
        generated_at = datetime(2026, 4, 20, 11, 50, 0)
    return PropProjection(
        event_id="game-1",
        player_id=player_id,
        market="pitcher_strikeouts",
        line=line,
        mean=6.2,
        over_probability=over_probability,
        under_probability=under_probability,
        model_version="v0",
        input_ref=ProjectionInputRef(
            lineup_snapshot_id="lineup-snapshot-1",
            feature_row_id="feature-row-1",
            features_as_of=feature_timestamp,
        ),
        generated_at=generated_at,
    )


def test_analyze_projection_returns_capped_stake_and_market_probabilities():
    line = make_line()
    projection = make_projection()

    analysis = analyze_projection(line, projection)

    assert analysis["side"] == "over"
    assert round(analysis["market_over_probability"], 6) == 0.5
    assert round(analysis["market_under_probability"], 6) == 0.5
    assert analysis["clears_min_edge"] is True
    assert analysis["stake_fraction"] == 0.02
    assert analysis["uncapped_stake_fraction"] > analysis["stake_fraction"]


def test_projection_must_clear_threshold_to_return_decision():
    line = make_line()
    projection = make_projection()

    decision = evaluate_projection(line, projection)
    assert decision is not None
    assert decision.side == "over"
    assert decision.expected_value_pct > 0


def test_projection_below_threshold_is_preserved_but_not_actionable():
    line = make_line(line=4.5)
    projection = make_projection(line=4.5, over_probability=0.52, under_probability=0.48)

    assert analyze_projection(line, projection)["clears_min_edge"] is False
    assert evaluate_projection(line, projection) is None


def test_projection_contract_must_match_line_contract():
    line = make_line()
    projection = make_projection(player_id="pitcher-2")

    with pytest.raises(
        ValueError, match="line and projection must reference the same prop contract"
    ):
        evaluate_projection(line, projection)


def test_projection_features_must_precede_market_capture():
    line = make_line(captured_at=datetime(2026, 4, 20, 12, 0, 0))
    projection = make_projection(
        feature_timestamp=datetime(2026, 4, 20, 12, 5, 0),
        generated_at=datetime(2026, 4, 20, 12, 10, 0),
    )

    with pytest.raises(
        ValueError,
        match="projection.input_ref.features_as_of must be on or before line.captured_at",
    ):
        evaluate_projection(line, projection)


def test_projection_generation_must_precede_market_capture():
    line = make_line(captured_at=datetime(2026, 4, 20, 12, 0, 0))
    projection = make_projection(generated_at=datetime(2026, 4, 20, 12, 1, 0))

    with pytest.raises(
        ValueError, match="projection.generated_at must be on or before line.captured_at"
    ):
        evaluate_projection(line, projection)


def test_decision_rejects_boundary_probabilities_without_fair_odds():
    line = make_line()
    projection = PropProjection(
        event_id="game-1",
        player_id="pitcher-1",
        market="pitcher_strikeouts",
        line=5.5,
        mean=7.0,
        over_probability=1.0,
        under_probability=0.0,
        model_version="v0",
        input_ref=ProjectionInputRef(
            lineup_snapshot_id="lineup-snapshot-1",
            feature_row_id="feature-row-1",
            features_as_of=datetime(2026, 4, 20, 11, 45, 0),
        ),
        generated_at=datetime(2026, 4, 20, 11, 50, 0),
    )

    with pytest.raises(
        ValueError,
        match="chosen_probability must be strictly between 0 and 1 to derive fair_odds",
    ):
        evaluate_projection(line, projection)


def test_build_edge_candidates_for_date_writes_actionable_below_threshold_and_skipped_rows(
    tmp_path,
) -> None:
    odds_run_dir = (
        tmp_path
        / "normalized"
        / "the_odds_api"
        / "date=2026-04-20"
        / "run=20260420T170000Z"
    )
    _write_jsonl(
        odds_run_dir / "prop_line_snapshots.jsonl",
        [
            {
                "line_snapshot_id": "line-1",
                "official_date": "2026-04-20",
                "captured_at": "2026-04-20T16:00:00Z",
                "sportsbook": "draftkings",
                "sportsbook_title": "DraftKings",
                "event_id": "event-1",
                "game_pk": 9001,
                "odds_matchup_key": "2026-04-20|BOS|NYY|2026-04-20T23:10:00Z",
                "match_status": "matched",
                "player_id": "mlb-pitcher:700001",
                "pitcher_mlb_id": 700001,
                "player_name": "Actionable Arm",
                "market": "pitcher_strikeouts",
                "line": 5.5,
                "over_odds": -110,
                "under_odds": -110,
            },
            {
                "line_snapshot_id": "line-2",
                "official_date": "2026-04-20",
                "captured_at": "2026-04-20T16:05:00Z",
                "sportsbook": "draftkings",
                "sportsbook_title": "DraftKings",
                "event_id": "event-1",
                "game_pk": 9001,
                "odds_matchup_key": "2026-04-20|BOS|NYY|2026-04-20T23:10:00Z",
                "match_status": "matched",
                "player_id": "mlb-pitcher:700001",
                "pitcher_mlb_id": 700001,
                "player_name": "Actionable Arm",
                "market": "pitcher_strikeouts",
                "line": 4.5,
                "over_odds": -110,
                "under_odds": -110,
            },
            {
                "line_snapshot_id": "line-3",
                "official_date": "2026-04-20",
                "captured_at": "2026-04-20T16:10:00Z",
                "sportsbook": "draftkings",
                "sportsbook_title": "DraftKings",
                "event_id": "event-2",
                "game_pk": 9002,
                "odds_matchup_key": "2026-04-20|SEA|TEX|2026-04-20T23:40:00Z",
                "match_status": "matched",
                "player_id": "mlb-pitcher:700002",
                "pitcher_mlb_id": 700002,
                "player_name": "Missing Projection",
                "market": "pitcher_strikeouts",
                "line": 5.5,
                "over_odds": -110,
                "under_odds": -110,
            },
            {
                "line_snapshot_id": "line-4",
                "official_date": "2026-04-20",
                "captured_at": "2026-04-20T16:15:00Z",
                "sportsbook": "draftkings",
                "sportsbook_title": "DraftKings",
                "event_id": "event-3",
                "game_pk": 9003,
                "odds_matchup_key": "2026-04-20|CHC|MIL|2026-04-20T23:50:00Z",
                "match_status": "matched",
                "player_id": "mlb-pitcher:700003",
                "pitcher_mlb_id": 700003,
                "player_name": "Missing Lineup Snapshot",
                "market": "pitcher_strikeouts",
                "line": 5.5,
                "over_odds": -110,
                "under_odds": -110,
            },
        ],
    )

    model_run_dir = (
        tmp_path
        / "normalized"
        / "starter_strikeout_baseline"
        / "start=2026-04-16_end=2026-04-20"
        / "run=20260421T180000Z"
    )
    _write_json(
        model_run_dir / "baseline_model.json",
        {"model_version": "starter-strikeout-baseline-v1"},
    )
    _write_jsonl(
        model_run_dir / "ladder_probabilities.jsonl",
        [
            {
                "training_row_id": "training-row-1",
                "official_date": "2026-04-20",
                "game_pk": 9001,
                "pitcher_id": 700001,
                "pitcher_name": "Actionable Arm",
                "split": "test",
                "feature_row_id": "training-row-1",
                "lineup_snapshot_id": "lineup-snapshot-1",
                "features_as_of": "2026-04-20T15:45:00Z",
                "projection_generated_at": "2026-04-20T15:45:00Z",
                "actual_strikeouts": 7,
                "naive_benchmark_mean": 5.4,
                "model_mean": 6.2,
                "count_distribution": {
                    "name": "negative_binomial_global_dispersion_v1",
                    "dispersion_alpha": 0.21,
                },
                "probability_calibration": {
                    "name": "isotonic_ladder_probability_calibrator_v1",
                    "sample_count": 48,
                    "is_identity": False,
                },
                "ladder_probabilities": [
                    {
                        "line": 4.5,
                        "over_probability": 0.52,
                        "under_probability": 0.48,
                    },
                    {
                        "line": 5.5,
                        "over_probability": 0.57,
                        "under_probability": 0.43,
                    },
                ],
                "calibrated_ladder_probabilities": [
                    {
                        "line": 4.5,
                        "over_probability": 0.52,
                        "under_probability": 0.48,
                    },
                    {
                        "line": 5.5,
                        "over_probability": 0.58,
                        "under_probability": 0.42,
                    },
                ],
            },
            {
                "training_row_id": "training-row-2",
                "official_date": "2026-04-20",
                "game_pk": 9003,
                "pitcher_id": 700003,
                "pitcher_name": "Missing Lineup Snapshot",
                "split": "test",
                "feature_row_id": "training-row-2",
                "lineup_snapshot_id": None,
                "features_as_of": "2026-04-20T15:40:00Z",
                "projection_generated_at": "2026-04-20T15:40:00Z",
                "actual_strikeouts": 6,
                "naive_benchmark_mean": 5.1,
                "model_mean": 5.7,
                "count_distribution": {
                    "name": "negative_binomial_global_dispersion_v1",
                    "dispersion_alpha": 0.19,
                },
                "probability_calibration": {
                    "name": "isotonic_ladder_probability_calibrator_v1",
                    "sample_count": 48,
                    "is_identity": False,
                },
                "ladder_probabilities": [
                    {
                        "line": 5.5,
                        "over_probability": 0.55,
                        "under_probability": 0.45,
                    }
                ],
                "calibrated_ladder_probabilities": [
                    {
                        "line": 5.5,
                        "over_probability": 0.56,
                        "under_probability": 0.44,
                    }
                ],
            },
        ],
    )

    result = build_edge_candidates_for_date(
        target_date=date(2026, 4, 20),
        output_dir=tmp_path,
    )

    assert result.model_version == "starter-strikeout-baseline-v1"
    assert result.model_run_id == "20260421T180000Z"
    assert result.line_count == 4
    assert result.scored_line_count == 2
    assert result.actionable_count == 1
    assert result.below_threshold_count == 1
    assert result.skipped_line_count == 2
    assert result.edge_candidates_path.exists()

    edge_rows = [
        json.loads(line)
        for line in result.edge_candidates_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert [row["evaluation_status"] for row in edge_rows] == [
        "actionable",
        "below_threshold",
        "missing_projection",
        "missing_lineup_snapshot_id",
    ]
    assert edge_rows[0]["line_snapshot_id"] == "line-1"
    assert edge_rows[0]["candidate_id"] == "line-1|starter-strikeout-baseline-v1"
    assert edge_rows[0]["lineup_snapshot_id"] == "lineup-snapshot-1"
    assert edge_rows[0]["feature_row_id"] == "training-row-1"
    assert edge_rows[0]["projection_generated_at"] == "2026-04-20T15:45:00Z"
    assert edge_rows[0]["model_over_probability"] == 0.58
    assert edge_rows[0]["market_over_probability"] == 0.5
    assert edge_rows[0]["stake_fraction"] == 0.02
    assert edge_rows[1]["edge_pct"] == 0.02
    assert edge_rows[1]["clears_min_edge"] is False
    assert edge_rows[2]["reason"] == "No ladder probabilities were found for this line snapshot."
    assert (
        edge_rows[3]["reason"]
        == "The matched projection does not carry a lineup snapshot reference."
    )
