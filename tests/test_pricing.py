import pytest

from mlb_props_stack.pricing import (
    american_to_implied_probability,
    book_hold,
    capped_fractional_kelly,
    devig_consensus_two_way,
    devig_two_way,
    expected_value,
    fair_american_odds,
    fractional_kelly,
    quarter_kelly,
)


def test_even_money_probability_is_half():
    assert round(american_to_implied_probability(100), 4) == 0.5


def test_devig_two_way_sums_to_one():
    over, under = devig_two_way(-110, -110)
    assert round(over + under, 8) == 1.0
    assert round(over, 6) == 0.5
    assert round(under, 6) == 0.5


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


def test_fractional_kelly_returns_zero_for_negative_edge():
    assert fractional_kelly(0.40, -110, fraction=0.25) == 0.0


def test_capped_fractional_kelly_never_exceeds_configured_cap():
    assert capped_fractional_kelly(
        0.60,
        120,
        fraction=0.25,
        max_fraction=0.02,
    ) == 0.02


def test_capped_fractional_kelly_rejects_invalid_cap():
    with pytest.raises(ValueError, match="max_fraction must be in \\[0, 1\\]"):
        capped_fractional_kelly(0.55, -110, fraction=0.25, max_fraction=1.2)


def test_book_hold_matches_manual_calculation():
    # -110 / -110 is the classic 110 cents on both sides -> 100/210 each -> hold ~0.0476
    assert round(book_hold(-110, -110), 6) == round(
        american_to_implied_probability(-110) * 2 - 1.0, 6
    )
    assert book_hold(100, 100) == 0.0


def test_devig_consensus_two_way_single_pair_matches_devig_two_way():
    consensus_over, consensus_under = devig_consensus_two_way([(-110, -110)])
    per_book_over, per_book_under = devig_two_way(-110, -110)
    assert round(consensus_over, 8) == round(per_book_over, 8)
    assert round(consensus_under, 8) == round(per_book_under, 8)


def test_devig_consensus_two_way_averages_across_books():
    # Book A favors the over, Book B favors the under. Consensus should land
    # between the two per-book devigs and still sum to 1.0.
    book_a_over, book_a_under = devig_two_way(-130, 110)
    book_b_over, book_b_under = devig_two_way(100, -120)
    consensus_over, consensus_under = devig_consensus_two_way(
        [(-130, 110), (100, -120)]
    )
    expected_over = (book_a_over + book_b_over) / 2.0
    expected_under = (book_a_under + book_b_under) / 2.0
    total = expected_over + expected_under
    assert round(consensus_over + consensus_under, 8) == 1.0
    assert round(consensus_over, 6) == round(expected_over / total, 6)
    assert round(consensus_under, 6) == round(expected_under / total, 6)


def test_devig_consensus_two_way_handles_tied_books():
    # Two identical book pairs should collapse to the per-book devig exactly.
    consensus_over, consensus_under = devig_consensus_two_way(
        [(-110, -110), (-110, -110)]
    )
    per_book_over, per_book_under = devig_two_way(-110, -110)
    assert round(consensus_over, 8) == round(per_book_over, 8)
    assert round(consensus_under, 8) == round(per_book_under, 8)


def test_devig_consensus_two_way_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        devig_consensus_two_way([])
