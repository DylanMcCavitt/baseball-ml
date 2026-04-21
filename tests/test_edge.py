from datetime import datetime

from mlb_props_stack.edge import evaluate_projection
from mlb_props_stack.markets import PropLine, PropProjection


def test_projection_must_clear_threshold_to_return_decision():
    captured_at = datetime(2026, 4, 20, 12, 0, 0)
    line = PropLine(
        sportsbook="draftkings",
        event_id="game-1",
        player_id="pitcher-1",
        player_name="Ace Starter",
        market="pitcher_strikeouts",
        line=5.5,
        over_odds=-110,
        under_odds=-110,
        captured_at=captured_at,
    )
    projection = PropProjection(
        event_id="game-1",
        player_id="pitcher-1",
        market="pitcher_strikeouts",
        line=5.5,
        mean=6.2,
        over_probability=0.58,
        under_probability=0.42,
        model_version="v0",
        generated_at=captured_at,
    )

    decision = evaluate_projection(line, projection)
    assert decision is not None
    assert decision.side == "over"
    assert decision.expected_value_pct > 0
