"""Backtest policy and evaluation guardrails."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestPolicy:
    """Rules that keep historical evaluation honest."""

    use_walk_forward_only: bool = True
    require_prelock_lines: bool = True
    require_pregame_feature_timestamps: bool = True
    require_projection_input_refs: bool = True
    preserve_rejected_props: bool = True
    report_clv: bool = True
    report_roi: bool = True
    report_edge_buckets: bool = True
    report_line_movement: bool = True

    def __post_init__(self) -> None:
        if not any(
            (
                self.report_clv,
                self.report_roi,
                self.report_edge_buckets,
                self.report_line_movement,
            )
        ):
            raise ValueError("at least one reporting output must be enabled")


BACKTEST_CHECKLIST = [
    "No feature may use information after the prop capture timestamp.",
    "Model training windows must end before the evaluated day begins.",
    "Each evaluated projection should point to explicit lineup and feature-row inputs.",
    "Closing-line value should be tracked separately from realized ROI.",
    "Backtests must include vig and any book-specific settlement quirks.",
    "Rejected bets should be preserved so threshold changes can be audited.",
]
