"""Static park-factor lookup for Statcast game-context features.

Values are anchored to FanGraphs' multi-year "Guts!" park factors for
starting-pitcher strikeouts. FanGraphs publishes them on a 100-scale;
the CSV stored under ``data/static/park_factors/park_k_factors.csv`` is
pre-converted to a ratio centered on ``1.00``. The ratio form lets the
feature layer multiply it directly against a baseline projection when a
future model wants to scale expected strikeouts.

Source: https://www.fangraphs.com/guts.aspx?type=pf&teamid=0&season=2025
Vintage: three-year rolling average, 2025 row carried into 2026 until
the season-end FanGraphs refresh lands.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PARK_FACTORS_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "static"
    / "park_factors"
    / "park_k_factors.csv"
)

PARK_FACTOR_STATUS_OK = "ok"
PARK_FACTOR_STATUS_MISSING_SOURCE = "missing_park_factor_source"


@dataclass(frozen=True)
class ParkKFactorRecord:
    """One park strikeout factor row keyed by season and MLB venue id."""

    season: int
    venue_mlb_id: int
    venue_name: str
    park_k_factor: float
    park_k_factor_vs_rhh: float
    park_k_factor_vs_lhh: float


def load_park_k_factors(
    path: Path | str = DEFAULT_PARK_FACTORS_PATH,
) -> dict[tuple[int, int], ParkKFactorRecord]:
    """Load the static park-factor CSV keyed by ``(season, venue_mlb_id)``.

    Skips rows with blank or unparseable ``season``/``venue_mlb_id`` cells so a
    partially-edited file still loads the valid rows instead of raising.
    """

    records: dict[tuple[int, int], ParkKFactorRecord] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                season = int(row["season"])
                venue_mlb_id = int(row["venue_mlb_id"])
            except (KeyError, TypeError, ValueError):
                continue
            record = ParkKFactorRecord(
                season=season,
                venue_mlb_id=venue_mlb_id,
                venue_name=row.get("venue_name", "").strip(),
                park_k_factor=float(row["park_k_factor"]),
                park_k_factor_vs_rhh=float(row["park_k_factor_vs_rhh"]),
                park_k_factor_vs_lhh=float(row["park_k_factor_vs_lhh"]),
            )
            records[(season, venue_mlb_id)] = record
    return records


def lookup_park_k_factor(
    season: int,
    venue_mlb_id: int | None,
    table: dict[tuple[int, int], ParkKFactorRecord] | None = None,
) -> ParkKFactorRecord | None:
    """Return the best-matching park-factor record or ``None``.

    Falls back to the prior season's value when the requested season is
    missing; the active FanGraphs table is only republished at season end,
    so a current-season slate should inherit last year's factor until the
    refreshed row lands.
    """

    if venue_mlb_id is None:
        return None
    records = table if table is not None else load_park_k_factors(DEFAULT_PARK_FACTORS_PATH)
    if (season, venue_mlb_id) in records:
        return records[(season, venue_mlb_id)]
    if (season - 1, venue_mlb_id) in records:
        return records[(season - 1, venue_mlb_id)]
    return None
