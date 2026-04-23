from __future__ import annotations

from pathlib import Path

from mlb_props_stack.ingest.park_factors import (
    DEFAULT_PARK_FACTORS_PATH,
    PARK_FACTOR_STATUS_MISSING_SOURCE,
    PARK_FACTOR_STATUS_OK,
    ParkKFactorRecord,
    load_park_k_factors,
    lookup_park_k_factor,
)


def _write_csv(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "season,venue_mlb_id,venue_name,park_k_factor,park_k_factor_vs_rhh,park_k_factor_vs_lhh\n"
    path.write_text(header + "".join(f"{row}\n" for row in rows), encoding="utf-8")


def test_default_park_factors_csv_exists_and_loads() -> None:
    assert DEFAULT_PARK_FACTORS_PATH.exists()
    table = load_park_k_factors(DEFAULT_PARK_FACTORS_PATH)
    assert table, "default park factor table must not be empty"

    progressive = table[(2026, 5)]
    assert isinstance(progressive, ParkKFactorRecord)
    assert progressive.venue_name == "Progressive Field"
    assert 0.5 <= progressive.park_k_factor <= 1.5
    assert 0.5 <= progressive.park_k_factor_vs_rhh <= 1.5
    assert 0.5 <= progressive.park_k_factor_vs_lhh <= 1.5


def test_default_park_factors_carry_prior_season_when_requested_season_missing() -> None:
    table = load_park_k_factors(DEFAULT_PARK_FACTORS_PATH)
    assert (2026, 5) in table
    record = lookup_park_k_factor(season=2027, venue_mlb_id=5, table=table)
    assert record is not None
    assert record.season == 2026


def test_lookup_park_k_factor_returns_none_when_venue_unknown(tmp_path) -> None:
    csv_path = tmp_path / "park_k_factors.csv"
    _write_csv(csv_path, ["2026,5,Progressive Field,1.02,1.02,1.03"])
    table = load_park_k_factors(csv_path)

    assert lookup_park_k_factor(season=2026, venue_mlb_id=999, table=table) is None


def test_lookup_park_k_factor_returns_none_when_venue_id_missing(tmp_path) -> None:
    csv_path = tmp_path / "park_k_factors.csv"
    _write_csv(csv_path, ["2026,5,Progressive Field,1.02,1.02,1.03"])
    table = load_park_k_factors(csv_path)

    assert lookup_park_k_factor(season=2026, venue_mlb_id=None, table=table) is None


def test_load_park_factors_skips_rows_with_blank_keys(tmp_path) -> None:
    csv_path = tmp_path / "park_k_factors.csv"
    _write_csv(
        csv_path,
        [
            "2026,5,Progressive Field,1.02,1.02,1.03",
            ",,,1.00,1.00,1.00",
        ],
    )
    table = load_park_k_factors(csv_path)

    assert set(table.keys()) == {(2026, 5)}


def test_park_factor_status_constants_are_stable() -> None:
    assert PARK_FACTOR_STATUS_OK == "ok"
    assert PARK_FACTOR_STATUS_MISSING_SOURCE == "missing_park_factor_source"
