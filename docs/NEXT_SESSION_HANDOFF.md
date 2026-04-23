# Next Session Handoff

## Current State

- Repo: `nba-ml` (current product scope: `mlb-props-stack`)
- Default branch: `main`
- Last completed issue: `AGE-206` on branch
  `dylan/distracted-dewdney-da4807`
- This slice adds multi-book pitcher strikeout line ingest and a
  configurable devig mode so the stack can fuse sportsbook quotes
  instead of pricing every row against a single book. Books can now be
  narrowed from the CLI (`--bookmakers pinnacle,circa`) and the edge
  builder can devig per book, against the tightest-hold book, or across
  a consensus of all books at the same line
- Current status: existing single-book artifacts still score identically
  under the default `per_book` mode. Feeding the new fixture two books
  at the same line produces different market probabilities and edge
  rankings under `consensus`, and records which books drove each devig
  via a new `market_consensus_books` field on every scored edge row

## What Was Completed In This Slice

- `src/mlb_props_stack/config.py`
  - `StackConfig` gains a `devig_mode` field (default `per_book`)
  - new module-level constants `DEVIG_MODE_PER_BOOK`,
    `DEVIG_MODE_TIGHTEST_BOOK`, `DEVIG_MODE_CONSENSUS` and a tuple
    `DEVIG_MODES` used for validation
  - `__post_init__` rejects any string outside the supported set
- `src/mlb_props_stack/pricing.py`
  - new `book_hold(over_odds, under_odds)` helper returning the
    summed raw implied probabilities minus 1.0 (zero-hold for
    +100/+100, ~0.0476 for -110/-110)
  - new `devig_consensus_two_way(book_odds)` helper that devigs each
    `(over, under)` pair independently and returns the consensus
    no-vig probabilities, renormalized so `over + under == 1.0`
    (raises `ValueError` on empty input)
- `src/mlb_props_stack/ingest/odds_api.py`
  - `normalize_event_odds_payload` accepts an optional
    `bookmakers: frozenset[str]` filter and silently drops books
    outside the set without inflating `skipped_prop_groups`
  - `ingest_odds_api_pitcher_lines_for_date` accepts
    `bookmakers: Iterable[str] | None`, coerces it to a frozen set,
    and threads the filter through to the normalizer; passing an
    empty / all-blank iterable raises `ValueError`
- `src/mlb_props_stack/edge.py`
  - new helper `_compute_market_probabilities(line, *, peer_lines,
    devig_mode)` that returns `(market_over, market_under,
    consensus_books)` for each supported mode
    - `per_book`: devig only the primary book; peers ignored so
      existing single-book callers keep their current behavior
    - `tightest_book`: pick the book with the lowest hold across the
      primary + peer lines, devig that book, and record its key
    - `consensus`: devig each book independently, average the no-vig
      probabilities, renormalize, and record every contributing book
      sorted alphabetically
  - `analyze_projection` accepts a `peer_lines` keyword and now
    returns `market_consensus_books` and `devig_mode` on every
    analysis dict
  - `build_edge_candidates_for_date` accepts an optional
    `config: StackConfig`, builds a peer lookup keyed on
    `(official_date, event_id, player_id, market, line)` (ignoring
    sportsbook), and records the resolved `market_consensus_books` and
    `devig_mode` on every scored candidate row
- `src/mlb_props_stack/cli.py`
  - `ingest-odds-api-lines` gains a `--bookmakers pinnacle,circa`
    flag parsed via `_parse_bookmaker_argument`
  - `render_runtime_summary()` now prints `devig_mode=<value>` so the
    default stack config is visible at a glance
- `tests/test_pricing.py`
  - five new tests: `book_hold` for -110/-110 vs +100/+100,
    consensus-single-pair equivalence, consensus averaging across
    books with opposite leans, tied-book collapse, and empty-input
    rejection
- `tests/test_odds_api_ingest.py`
  - new two-book fixture `_two_book_event_odds_payload`
  - new tests: multi-book persistence (DraftKings + Pinnacle both
    landing in `prop_line_snapshots.jsonl` at one row per book) and
    the `bookmakers=("pinnacle",)` filter reducing the output to
    exactly the requested books without inflating skip counts
  - empty-iterable filter is rejected with a clear message
- `tests/test_edge.py`
  - existing happy-path test now asserts the new
    `market_consensus_books` and `devig_mode` fields land on actionable
    rows
  - new `analyze_projection` tests: per_book ignores peers,
    tightest_book picks the lower-hold book, consensus lands strictly
    between the per-book devigs and sums to 1.0
  - new `build_edge_candidates_for_date` test with two books at the
    same line verifying the two rows disagree on
    `market_over_probability` under `per_book` but agree (and name
    both books) under `consensus`, and that `edge_pct` actually shifts
    between the two modes
- `tests/test_contracts.py`
  - default `StackConfig().devig_mode == "per_book"` guard plus
    accept-all-supported-modes and reject-unknown-mode tests
- `tests/test_cli.py`
  - the odds ingest CLI test now exercises
    `--bookmakers pinnacle,circa` and asserts the parsed tuple is
    forwarded into `ingest_odds_api_pitcher_lines_for_date`

## Files Changed

- `docs/NEXT_SESSION_HANDOFF.md`
- `src/mlb_props_stack/cli.py`
- `src/mlb_props_stack/config.py`
- `src/mlb_props_stack/edge.py`
- `src/mlb_props_stack/ingest/odds_api.py`
- `src/mlb_props_stack/pricing.py`
- `tests/test_cli.py`
- `tests/test_contracts.py`
- `tests/test_edge.py`
- `tests/test_odds_api_ingest.py`
- `tests/test_pricing.py`

## Verification

Commands run successfully before merge:

```bash
uv sync --extra dev
uv run pytest
uv run python -m mlb_props_stack
```

Observed results:

- full test suite passed: `145 passed` (up from `130` on the previous
  slice; the fifteen additional tests cover the five new pricing
  helpers, three new odds-ingest scenarios, four new edge
  devig-mode scenarios, and three new `StackConfig.devig_mode`
  validation guards)
- `uv run python -m mlb_props_stack` now renders `devig_mode=per_book`
  in the runtime summary alongside the existing market + Kelly
  defaults

## Recommended Next Issue

- Wire the existing paper-tracking and backtest paths into the new
  devig modes. `paper_tracking.py` and `backtest.py` still call
  `devig_two_way` / `analyze_projection` without passing peer lines,
  which is correct under the default `per_book` mode but leaves
  `tightest_book` / `consensus` invisible to historical replays.
  Next slice should either (a) add a CLI flag to pick the mode end to
  end, or (b) route the pregame workflow through
  `_compute_market_probabilities` with peer lookup so the dashboard
  and paper results share the consensus-aware prices
- Once the backtest honors `devig_mode`, rerun the overnight 2024 +
  2025 backfill with `--bookmakers pinnacle,circa` to start building a
  sharp-consensus history, then compare ROI and CLV under `per_book`,
  `tightest_book`, and `consensus` on the same slate to confirm the
  ranking differences observed in the unit tests hold up on real data

## Constraints And Open Questions

- `devig_consensus_two_way` uses a straight arithmetic mean of the
  per-book devigged probabilities. That is the simplest consensus
  aggregator and matches the behavior the issue asked for, but a
  future iteration might want to weight books by liquidity, recency,
  or hold — leave the helper's signature open so a future `weights=`
  kwarg can slot in without breaking callers
- `tightest_book` resolves ties by picking the alphabetically-lowest
  sportsbook key. That's deterministic and matches how the ordered
  book list is constructed, but if two books end up genuinely
  equivalent we might prefer the primary line's book to avoid
  reassigning the devig on a tie — worth revisiting once real
  multi-book data lands
- `peer_lines` is matched on `(official_date, event_id, player_id,
  market, line)` without `sportsbook`, which is what enables peers to
  cross-pollinate. If two books quote slightly different lines (e.g.
  5.5 vs 6.0 for the same pitcher), they are treated as different
  contracts and will not share a consensus — intentional for the
  first cut but a candidate for a future "closest ladder rung"
  matcher
- The ingest `--bookmakers` filter is a hard drop, not a preference
  order. Books outside the filter simply vanish from
  `prop_line_snapshots.jsonl`, so retroactively widening the filter
  requires rerunning the ingest rather than relying on a cached fan
  out of every book The Odds API returned

## Known Follow-Up Nits (Non-Blocking)

Captured during AGE-206 review; none of these gate the merge but
they're worth folding into the next slice that touches this code:

- `_parse_bookmaker_argument` raises `ValueError` directly instead of
  argparse's nicer `argparse.ArgumentTypeError`, so a malformed
  `--bookmakers ""` surfaces as a Python traceback rather than a
  friendly CLI message. Trivial wrap when it next gets touched
- No test covers `tightest_book` with three or more books, so the
  alphabetical tie-break on sportsbook key is only exercised in the
  two-book path. Worth adding a 3-book fixture when we extend the
  pricing tests again
- No edge-layer test exercises `devig_mode="consensus"` with empty
  `peer_lines` (it falls through to a single-book consensus, which
  equals `devig_two_way`, already covered in `tests/test_pricing.py`
  via `test_devig_consensus_two_way_single_pair_matches_devig_two_way`).
  Adding a parallel test at the `analyze_projection` seam would keep
  the boundary assertion where the branch lives
