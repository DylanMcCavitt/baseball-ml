# Park Factors

Static park strikeout factors keyed by MLB Stats API `venue_id`, season, and
batter handedness. Consumed by
`src/mlb_props_stack/ingest/park_factors.py` during the Statcast feature
build so `game_context_features.jsonl` carries `park_k_factor`,
`park_k_factor_vs_rhh`, and `park_k_factor_vs_lhh` for every slate row.

## Files

- `park_k_factors.csv` — one row per `(season, venue_mlb_id)`; all factors
  are ratios centered on `1.00` (values below `1.00` suppress strikeouts,
  values above `1.00` inflate them).

## Schema

| column | type | description |
| --- | --- | --- |
| `season` | int | Calendar year the factor applies to. |
| `venue_mlb_id` | int | MLB Stats API venue identifier. |
| `venue_name` | str | Human-readable venue name (denormalized for auditing). |
| `park_k_factor` | float | Overall strikeout park factor. |
| `park_k_factor_vs_rhh` | float | Park factor for right-handed batters. |
| `park_k_factor_vs_lhh` | float | Park factor for left-handed batters. |

## Source

Values are anchored to FanGraphs' multi-year "Guts!" park factors for
starting-pitcher strikeouts, converted from the FanGraphs 100-scale to a
ratio (value / 100). This checkout uses a three-year rolling-average
vintage and carries the 2025 row forward unchanged into 2026 until the
season-end FanGraphs refresh lands.

Source URL: <https://www.fangraphs.com/guts.aspx?type=pf&teamid=0&season=2025>

## Venue-id aliases

A few MLB parks appear under more than one venue id because the Stats API
renumbered them over the years and both ids still surface in different
historical payloads. Where this happens the CSV carries one row per id
with the same factor values so the join succeeds regardless of which id
the schedule response returns:

| venue | ids | note |
| --- | --- | --- |
| Yankee Stadium | `10`, `2602` | Legacy id + current id both observed in schedule responses. |
| loanDepot park | `31`, `2680` | Same park under the old Marlins Park id and the current renamed id. |

If a future refresh introduces a new alias, add the new id as its own
row(s) rather than deleting the old id — older artifacts on disk may
still carry the retired id.

## Adding a venue

1. Look up the venue's MLB Stats API id via
   `https://statsapi.mlb.com/api/v1/venues?sportId=1&season=YYYY`.
2. Append one row per season with the overall and handedness-split factors.
3. Update `tests/test_park_factors.py` if the new venue is referenced by
   fixtures elsewhere.
