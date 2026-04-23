"""Runtime defaults for the MLB props stack."""

from dataclasses import dataclass

DEVIG_MODE_PER_BOOK = "per_book"
DEVIG_MODE_TIGHTEST_BOOK = "tightest_book"
DEVIG_MODE_CONSENSUS = "consensus"

DEVIG_MODES: tuple[str, ...] = (
    DEVIG_MODE_PER_BOOK,
    DEVIG_MODE_TIGHTEST_BOOK,
    DEVIG_MODE_CONSENSUS,
)


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
    devig_mode: str = DEVIG_MODE_PER_BOOK

    def __post_init__(self) -> None:
        if self.devig_mode not in DEVIG_MODES:
            raise ValueError(
                f"devig_mode must be one of {DEVIG_MODES}, got {self.devig_mode!r}"
            )
