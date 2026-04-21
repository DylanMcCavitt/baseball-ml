"""Compare projections to market prices and produce candidate bets."""

from typing import Optional

from .config import StackConfig
from .markets import EdgeDecision, PropLine, PropProjection
from .pricing import (
    devig_two_way,
    expected_value,
    fair_american_odds,
    quarter_kelly,
)


def evaluate_projection(
    line: PropLine,
    projection: PropProjection,
    config: Optional[StackConfig] = None,
) -> Optional[EdgeDecision]:
    """Return the best actionable side if an edge clears the threshold."""
    if config is None:
        config = StackConfig()

    market_over, market_under = devig_two_way(line.over_odds, line.under_odds)
    over_edge = projection.over_probability - market_over
    under_edge = projection.under_probability - market_under

    if over_edge >= under_edge:
        side = "over"
        edge_pct = over_edge
        chosen_probability = projection.over_probability
        chosen_odds = line.over_odds
    else:
        side = "under"
        edge_pct = under_edge
        chosen_probability = projection.under_probability
        chosen_odds = line.under_odds

    if edge_pct < config.min_edge_pct:
        return None

    ev = expected_value(chosen_probability, chosen_odds)
    stake_fraction = min(
        quarter_kelly(chosen_probability, chosen_odds, config.kelly_fraction),
        config.max_bet_fraction,
    )
    fair_odds = fair_american_odds(chosen_probability)
    return EdgeDecision(
        side=side,
        edge_pct=edge_pct,
        expected_value_pct=ev,
        stake_fraction=stake_fraction,
        fair_odds=fair_odds,
        reason=(
            f"{side} clears minimum edge threshold "
            f"({edge_pct:.2%} >= {config.min_edge_pct:.2%})"
        ),
    )
