"""Backtest policy and evaluation guardrails."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestPolicy:
    """Rules that keep historical evaluation honest."""

    use_walk_forward_only: bool = True
    require_prelock_lines: bool = True
    require_pregame_feature_timestamps: bool = True
    report_clv: bool = True
    report_roi: bool = True
    report_edge_buckets: bool = True
    report_line_movement: bool = True


BACKTEST_CHECKLIST = [
    "No feature may use information after the prop capture timestamp.",
    "Model training windows must end before the evaluated day begins.",
    "Closing-line value should be tracked separately from realized ROI.",
    "Backtests must include vig and any book-specific settlement quirks.",
    "Rejected bets should be preserved so threshold changes can be audited.",
]
