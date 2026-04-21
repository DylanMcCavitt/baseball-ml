from datetime import datetime

import pytest

from mlb_props_stack.backtest import BacktestPolicy
from mlb_props_stack.markets import (
    EdgeDecision,
    PropLine,
    PropProjection,
    ProjectionInputRef,
)


def make_input_ref(*, features_as_of: datetime | None = None) -> ProjectionInputRef:
    if features_as_of is None:
        features_as_of = datetime(2026, 4, 20, 11, 30, 0)
    return ProjectionInputRef(
        lineup_snapshot_id="lineup-snapshot-1",
        feature_row_id="feature-row-1",
        features_as_of=features_as_of,
    )


def make_prop_line() -> PropLine:
    return PropLine(
        sportsbook="draftkings",
        event_id="game-1",
        player_id="pitcher-1",
        player_name="Ace Starter",
        market="pitcher_strikeouts",
        line=5.5,
        over_odds=-110,
        under_odds=-110,
        captured_at=datetime(2026, 4, 20, 12, 0, 0),
    )


def test_prop_line_rejects_negative_line():
    with pytest.raises(ValueError, match="line must be >= 0.0"):
        PropLine(
            sportsbook="draftkings",
            event_id="game-1",
            player_id="pitcher-1",
            player_name="Ace Starter",
            market="pitcher_strikeouts",
            line=-0.5,
            over_odds=-110,
            under_odds=-110,
            captured_at=datetime(2026, 4, 20, 12, 0, 0),
        )


def test_projection_requires_probabilities_to_sum_to_one():
    with pytest.raises(
        ValueError,
        match="over_probability and under_probability must sum to 1.0",
    ):
        PropProjection(
            event_id="game-1",
            player_id="pitcher-1",
            market="pitcher_strikeouts",
            line=5.5,
            mean=6.2,
            over_probability=0.60,
            under_probability=0.35,
            model_version="v0",
            input_ref=make_input_ref(),
            generated_at=datetime(2026, 4, 20, 11, 45, 0),
        )


def test_projection_requires_feature_inputs_to_precede_generation():
    with pytest.raises(
        ValueError, match="input_ref.features_as_of cannot be after generated_at"
    ):
        PropProjection(
            event_id="game-1",
            player_id="pitcher-1",
            market="pitcher_strikeouts",
            line=5.5,
            mean=6.2,
            over_probability=0.58,
            under_probability=0.42,
            model_version="v0",
            input_ref=make_input_ref(features_as_of=datetime(2026, 4, 20, 12, 0, 0)),
            generated_at=datetime(2026, 4, 20, 11, 59, 0),
        )


def test_line_and_projection_share_selection_key():
    line = make_prop_line()
    projection = PropProjection(
        event_id="game-1",
        player_id="pitcher-1",
        market="pitcher_strikeouts",
        line=5.5,
        mean=6.2,
        over_probability=0.58,
        under_probability=0.42,
        model_version="v0",
        input_ref=make_input_ref(),
        generated_at=datetime(2026, 4, 20, 11, 45, 0),
    )

    assert line.selection_key == projection.selection_key


def test_edge_decision_rejects_invalid_side():
    with pytest.raises(ValueError, match="side must be 'over' or 'under'"):
        EdgeDecision(
            side="pass",
            edge_pct=0.07,
            expected_value_pct=0.03,
            stake_fraction=0.01,
            fair_odds=-140,
        )


def test_backtest_policy_requires_one_reporting_output():
    with pytest.raises(ValueError, match="at least one reporting output must be enabled"):
        BacktestPolicy(
            report_clv=False,
            report_roi=False,
            report_edge_buckets=False,
            report_line_movement=False,
        )


def test_backtest_policy_defaults_keep_join_refs_and_rejections_enabled():
    policy = BacktestPolicy()

    assert policy.require_projection_input_refs is True
    assert policy.preserve_rejected_props is True
