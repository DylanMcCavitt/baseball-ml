from datetime import datetime

import pytest

from mlb_props_stack.edge import evaluate_projection
from mlb_props_stack.markets import PropLine, PropProjection, ProjectionInputRef


def make_line(*, player_id: str = "pitcher-1", captured_at: datetime | None = None) -> PropLine:
    if captured_at is None:
        captured_at = datetime(2026, 4, 20, 12, 0, 0)
    return PropLine(
        sportsbook="draftkings",
        event_id="game-1",
        player_id=player_id,
        player_name="Ace Starter",
        market="pitcher_strikeouts",
        line=5.5,
        over_odds=-110,
        under_odds=-110,
        captured_at=captured_at,
    )


def make_projection(
    *,
    player_id: str = "pitcher-1",
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
        line=5.5,
        mean=6.2,
        over_probability=0.58,
        under_probability=0.42,
        model_version="v0",
        input_ref=ProjectionInputRef(
            lineup_snapshot_id="lineup-snapshot-1",
            feature_row_id="feature-row-1",
            features_as_of=feature_timestamp,
        ),
        generated_at=generated_at,
    )


def test_projection_must_clear_threshold_to_return_decision():
    line = make_line()
    projection = make_projection()

    decision = evaluate_projection(line, projection)
    assert decision is not None
    assert decision.side == "over"
    assert decision.expected_value_pct > 0


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
