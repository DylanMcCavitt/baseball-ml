import pytest

from mlb_props_stack.pricing import (
    american_to_implied_probability,
    devig_two_way,
    expected_value,
    fair_american_odds,
    quarter_kelly,
)


def test_even_money_probability_is_half():
    assert round(american_to_implied_probability(100), 4) == 0.5


def test_devig_two_way_sums_to_one():
    over, under = devig_two_way(-110, -110)
    assert round(over + under, 8) == 1.0


def test_positive_ev_for_mispriced_plus_money():
    assert expected_value(0.50, 120) > 0.0


def test_fair_odds_for_sixty_percent_probability():
    assert fair_american_odds(0.60) == -150


def test_quarter_kelly_rejects_invalid_probability():
    with pytest.raises(ValueError, match="probability must be in \\[0, 1\\]"):
        quarter_kelly(1.1, -110)


def test_quarter_kelly_rejects_fraction_above_one():
    with pytest.raises(ValueError, match="fraction must be in \\(0, 1\\]"):
        quarter_kelly(0.55, -110, fraction=1.1)
