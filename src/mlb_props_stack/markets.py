"""Domain models for props, projections, and decisions."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class PropLine:
    """A sportsbook line for a single two-way prop market."""

    sportsbook: str
    event_id: str
    player_id: str
    player_name: str
    market: str
    line: float
    over_odds: int
    under_odds: int
    captured_at: datetime


@dataclass(frozen=True)
class PropProjection:
    """Model output for a specific prop line."""

    event_id: str
    player_id: str
    market: str
    line: float
    mean: float
    over_probability: float
    under_probability: float
    model_version: str
    generated_at: datetime


@dataclass(frozen=True)
class EdgeDecision:
    """Candidate action after comparing model and market probabilities."""

    side: str
    edge_pct: float
    expected_value_pct: float
    stake_fraction: float
    fair_odds: int
    reason: Optional[str] = None
