"""Odds conversion and sizing utilities."""

from typing import Tuple


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds."""
    if odds == 0:
        raise ValueError("odds cannot be zero")
    if odds > 0:
        return 1 + (odds / 100.0)
    return 1 + (100.0 / abs(odds))


def american_to_implied_probability(odds: int) -> float:
    """Convert American odds to the raw implied probability including vig."""
    decimal = american_to_decimal(odds)
    return 1.0 / decimal


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American odds."""
    if decimal_odds <= 1.0:
        raise ValueError("decimal odds must be greater than 1")
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1.0) * 100)
    return round(-100 / (decimal_odds - 1.0))


def devig_two_way(over_odds: int, under_odds: int) -> Tuple[float, float]:
    """Remove hold from a two-way market using proportional normalization."""
    raw_over = american_to_implied_probability(over_odds)
    raw_under = american_to_implied_probability(under_odds)
    total = raw_over + raw_under
    return raw_over / total, raw_under / total


def expected_value(probability: float, odds: int) -> float:
    """Return expected profit per 1 unit staked."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    decimal = american_to_decimal(odds)
    profit_if_win = decimal - 1.0
    return (probability * profit_if_win) - (1.0 - probability)


def fair_american_odds(probability: float) -> int:
    """Return the no-vig fair American odds for a probability."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between 0 and 1")
    return decimal_to_american(1.0 / probability)


def quarter_kelly(probability: float, odds: int, fraction: float = 0.25) -> float:
    """Return a fractional Kelly stake as a fraction of bankroll."""
    if fraction <= 0:
        raise ValueError("fraction must be positive")
    decimal = american_to_decimal(odds)
    b = decimal - 1.0
    q = 1.0 - probability
    full_kelly = ((probability * b) - q) / b
    return max(0.0, full_kelly * fraction)
