"""Runtime defaults for the MLB props stack."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StackConfig:
    """Top-level defaults for the first production slice."""

    market: str = "pitcher_strikeouts"
    min_edge_pct: float = 0.045
    kelly_fraction: float = 0.25
    max_bet_fraction: float = 0.02
    min_sample_for_calibration: int = 400
    backtest_cutoff_minutes_before_first_pitch: int = 30
    timezone: str = "America/New_York"
