# Venue Metadata

Static stadium metadata keyed by MLB Stats API `venue_id`. Consumed by
`src/mlb_props_stack/ingest/weather.py` during the pregame weather ingest
so `weather_snapshots.jsonl` can geocode each scheduled game and decide
whether a stadium is outdoor (eligible for weather fetch) or fixed-roof
(skipped with a `roof_closed` sentinel).

## Files

- `venue_metadata.csv` — one row per MLB `venue_id` covering the 30
  current regular-season parks plus legacy ids still emitted by
  historical schedule responses (e.g. Oakland Coliseum `10` for the A's
  2022-2024 seasons before Sutter Health Park `2529`).

## Schema

| column | type | description |
| --- | --- | --- |
| `venue_mlb_id` | int | MLB Stats API venue identifier. |
| `venue_name` | str | Human-readable venue name (denormalized for auditing). |
| `latitude` | float | Stadium geographic latitude in decimal degrees. |
| `longitude` | float | Stadium geographic longitude in decimal degrees. |
| `roof_type` | str | `open` (no roof), `retractable` (usually open, closeable), or `fixed` (always covered). |

## Weather source

Pregame weather is pulled from the
[Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api)
— a free, no-API-key, timestamp-valid historical source that exposes
hourly observed surface values (`temperature_2m`, `wind_speed_10m`,
`wind_direction_10m`, `relative_humidity_2m`). We anchor each snapshot
to the hour closest to `commence_time - 60 minutes` so the feature is
guaranteed pregame. Fixed-roof stadiums (currently just Tropicana Field)
skip the fetch and emit a sentinel `weather_status="roof_closed"` row;
retractable-roof stadiums are ingested as outdoor with the `roof_type`
recorded so downstream features can discount them if needed.

## Adding a venue

1. Look up the venue's MLB Stats API id via
   `https://statsapi.mlb.com/api/v1/venues?sportId=1&season=YYYY`.
2. Append one row with latitude, longitude, and roof type.
3. If the same park surfaces under multiple venue ids, add one row per
   id pointing at the same coordinates so joins succeed regardless of
   which id the schedule response returns.
