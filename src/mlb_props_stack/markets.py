"""Domain models for props, projections, and decisions."""

from dataclasses import dataclass
from datetime import datetime
from math import isclose, isfinite
from typing import Optional

PROBABILITY_SUM_TOLERANCE = 1e-3


def _require_text(value: str, field_name: str) -> None:
    """Reject blank identifiers in contract-bearing models."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} cannot be blank")


def _require_datetime(value: datetime, field_name: str) -> None:
    """Keep timestamp-bearing fields explicit and typed."""
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")


def _require_real_number(
    value: float,
    field_name: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> None:
    """Validate finite numeric values with optional bounds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    if not isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}")


def _require_american_odds(value: int, field_name: str) -> None:
    """Reject malformed American odds inputs."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value == 0:
        raise ValueError(f"{field_name} cannot be zero")


@dataclass(frozen=True)
class PropSelectionKey:
    """Stable identifier for one player's market at one sportsbook line."""

    event_id: str
    player_id: str
    market: str
    line: float

    def __post_init__(self) -> None:
        _require_text(self.event_id, "event_id")
        _require_text(self.player_id, "player_id")
        _require_text(self.market, "market")
        _require_real_number(self.line, "line", minimum=0.0)


@dataclass(frozen=True)
class ProjectionInputRef:
    """Minimal seam for later lineup and feature-row joins."""

    lineup_snapshot_id: str
    feature_row_id: str
    features_as_of: datetime

    def __post_init__(self) -> None:
        _require_text(self.lineup_snapshot_id, "lineup_snapshot_id")
        _require_text(self.feature_row_id, "feature_row_id")
        _require_datetime(self.features_as_of, "features_as_of")


@dataclass(frozen=True)
class PropLine:
    """A sportsbook line for a single two-way prop market."""

    line_snapshot_id: str
    sportsbook: str
    event_id: str
    player_id: str
    player_name: str
    market: str
    line: float
    over_odds: int
    under_odds: int
    captured_at: datetime

    def __post_init__(self) -> None:
        _require_text(self.line_snapshot_id, "line_snapshot_id")
        _require_text(self.sportsbook, "sportsbook")
        _require_text(self.player_name, "player_name")
        _require_american_odds(self.over_odds, "over_odds")
        _require_american_odds(self.under_odds, "under_odds")
        _require_datetime(self.captured_at, "captured_at")
        self.selection_key

    @property
    def selection_key(self) -> PropSelectionKey:
        """Return the stable join key for this line."""
        return PropSelectionKey(
            event_id=self.event_id,
            player_id=self.player_id,
            market=self.market,
            line=self.line,
        )


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
    input_ref: ProjectionInputRef
    generated_at: datetime

    def __post_init__(self) -> None:
        _require_text(self.model_version, "model_version")
        _require_real_number(self.mean, "mean", minimum=0.0)
        if not isinstance(self.input_ref, ProjectionInputRef):
            raise TypeError("input_ref must be a ProjectionInputRef")
        _require_real_number(
            self.over_probability,
            "over_probability",
            minimum=0.0,
            maximum=1.0,
        )
        _require_real_number(
            self.under_probability,
            "under_probability",
            minimum=0.0,
            maximum=1.0,
        )
        total_probability = self.over_probability + self.under_probability
        if not isclose(total_probability, 1.0, abs_tol=PROBABILITY_SUM_TOLERANCE):
            raise ValueError(
                "over_probability and under_probability must sum to 1.0 "
                f"within {PROBABILITY_SUM_TOLERANCE}"
            )
        _require_datetime(self.generated_at, "generated_at")
        self.selection_key
        if self.input_ref.features_as_of > self.generated_at:
            raise ValueError("input_ref.features_as_of cannot be after generated_at")

    @property
    def selection_key(self) -> PropSelectionKey:
        """Return the stable join key for this projection."""
        return PropSelectionKey(
            event_id=self.event_id,
            player_id=self.player_id,
            market=self.market,
            line=self.line,
        )


@dataclass(frozen=True)
class EdgeDecision:
    """Candidate action after comparing model and market probabilities."""

    side: str
    edge_pct: float
    expected_value_pct: float
    stake_fraction: float
    fair_odds: int
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.side not in {"over", "under"}:
            raise ValueError("side must be 'over' or 'under'")
        _require_real_number(self.edge_pct, "edge_pct", minimum=0.0, maximum=1.0)
        _require_real_number(self.expected_value_pct, "expected_value_pct", minimum=-1.0)
        _require_real_number(
            self.stake_fraction,
            "stake_fraction",
            minimum=0.0,
            maximum=1.0,
        )
        _require_american_odds(self.fair_odds, "fair_odds")
        if self.reason is not None:
            _require_text(self.reason, "reason")
