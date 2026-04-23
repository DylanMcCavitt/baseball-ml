import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from types import ModuleType

from mlb_props_stack.dashboard.app import render_dashboard_page
from mlb_props_stack.paper_tracking import (
    _build_daily_candidate_rows,
    _build_paper_result_rows,
    build_daily_candidate_workflow,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, sort_keys=True)}\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _seed_baseline_run(tmp_path: Path, *, end_date: str, run_id: str, intercept: float) -> Path:
    run_dir = (
        tmp_path
        / "normalized"
        / "starter_strikeout_baseline"
        / f"start=2026-04-18_end={end_date}"
        / f"run={run_id}"
    )
    _write_json(
        run_dir / "baseline_model.json",
        {
            "model_version": "starter-strikeout-baseline-v1",
            "count_distribution": {
                "name": "negative_binomial_global_dispersion_v1",
                "dispersion_alpha": 0.2,
            },
            "intercept": intercept,
            "encoded_feature_names": [],
            "coefficients": {},
            "numeric_feature_stats": {},
            "categorical_feature_levels": {
                "pitcher_feature_status": [],
                "lineup_status": [],
                "pitcher_hand": [],
                "home_away": [],
                "day_night": [],
                "double_header": [],
                "park_factor_status": [],
                "weather_status": [],
            },
            "probability_calibration": {
                "name": "isotonic_ladder_probability_calibrator_v1",
                "source": "out_of_fold_ladder_events",
                "configured_min_sample": 400,
                "sample_count": 600,
                "fitted_from_date": "2026-04-18",
                "fitted_through_date": end_date,
                "is_identity": True,
                "reason": "seeded",
                "sample_warning": None,
                "buckets": [],
            },
        },
    )
    _write_json(
        run_dir / "date_splits.json",
        {
            "train": ["2026-04-18", "2026-04-19"],
            "validation": ["2026-04-20"],
            "test": [end_date],
        },
    )
    _write_jsonl(run_dir / "ladder_probabilities.jsonl", [])
    _write_jsonl(
        run_dir / "starter_outcomes.jsonl",
        [
            {
                "outcome_id": "starter-outcome:2026-04-21:9001:700001",
                "official_date": "2026-04-21",
                "game_pk": 9001,
                "pitcher_id": 700001,
                "starter_strikeouts": 8,
            }
        ],
    )
    return run_dir


def _seed_feature_rows(tmp_path: Path) -> None:
    run_dir = (
        tmp_path
        / "normalized"
        / "statcast_search"
        / "date=2026-04-22"
        / "run=20260422T170000Z"
    )
    _write_jsonl(
        run_dir / "pitcher_daily_features.jsonl",
        [
            {
                "official_date": "2026-04-22",
                "game_pk": 9002,
                "pitcher_id": 700002,
                "pitcher_name": "Today Arm",
                "team_abbreviation": "NYY",
                "opponent_team_abbreviation": "BOS",
                "feature_row_id": "pitcher-feature-1",
                "features_as_of": "2026-04-22T16:00:00Z",
                "feature_status": "ok",
                "pitch_sample_size": 400,
                "plate_appearance_sample_size": 90,
                "pitcher_k_rate": 0.29,
                "swinging_strike_rate": 0.14,
                "csw_rate": 0.31,
                "average_release_speed": 95.0,
                "release_speed_delta_vs_baseline": 0.1,
                "average_release_extension": 6.2,
                "release_extension_delta_vs_baseline": 0.0,
                "recent_batters_faced": 75,
                "recent_pitch_count": 290,
                "rest_days": 5,
                "last_start_pitch_count": 96,
                "last_start_batters_faced": 26,
                "pitch_type_usage": {"FF": 0.5, "SL": 0.3},
            }
        ],
    )
    _write_jsonl(
        run_dir / "lineup_daily_features.jsonl",
        [
            {
                "official_date": "2026-04-22",
                "game_pk": 9002,
                "pitcher_id": 700002,
                "feature_row_id": "lineup-feature-1",
                "lineup_snapshot_id": "lineup-snapshot-1",
                "features_as_of": "2026-04-22T16:05:00Z",
                "lineup_status": "projected",
                "lineup_is_confirmed": False,
                "pitcher_hand": "R",
                "projected_lineup_k_rate": 0.24,
                "projected_lineup_k_rate_vs_pitcher_hand": 0.25,
                "projected_lineup_chase_rate": 0.31,
                "projected_lineup_contact_rate": 0.74,
                "lineup_size": 9,
                "available_batter_feature_count": 9,
                "lineup_continuity_count": 6,
                "lineup_continuity_ratio": 0.67,
            }
        ],
    )
    _write_jsonl(
        run_dir / "game_context_features.jsonl",
        [
            {
                "official_date": "2026-04-22",
                "game_pk": 9002,
                "pitcher_id": 700002,
                "feature_row_id": "game-context-feature-1",
                "features_as_of": "2026-04-22T16:10:00Z",
                "home_away": "home",
                "day_night": "night",
                "double_header": "none",
                "park_factor_status": "missing_park_factor_source",
                "weather_status": "missing_weather_source",
                "expected_leash_pitch_count": 96.0,
                "expected_leash_batters_faced": 26.0,
            }
        ],
    )


def _seed_odds_rows(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path
        / "normalized"
        / "the_odds_api"
        / "date=2026-04-22"
        / "run=20260422T182500Z"
        / "prop_line_snapshots.jsonl",
        [
            {
                "line_snapshot_id": "line-2026-04-22-1",
                "official_date": "2026-04-22",
                "captured_at": "2026-04-22T18:25:00Z",
                "sportsbook": "draftkings",
                "sportsbook_title": "DraftKings",
                "event_id": "event-2026-04-22-1",
                "game_pk": 9002,
                "odds_matchup_key": "2026-04-22|BOS|NYY|2026-04-22T23:05:00Z",
                "match_status": "matched",
                "player_id": "mlb-pitcher:700002",
                "pitcher_mlb_id": 700002,
                "player_name": "Today Arm",
                "market": "pitcher_strikeouts",
                "line": 5.5,
                "over_odds": 120,
                "under_odds": -145,
            }
        ],
    )
    _write_jsonl(
        tmp_path
        / "normalized"
        / "the_odds_api"
        / "date=2026-04-21"
        / "run=20260421T181000Z"
        / "prop_line_snapshots.jsonl",
        [
            {
                "line_snapshot_id": "line-2026-04-21-open",
                "official_date": "2026-04-21",
                "captured_at": "2026-04-21T18:10:00Z",
                "sportsbook": "draftkings",
                "sportsbook_title": "DraftKings",
                "event_id": "event-2026-04-21-1",
                "game_pk": 9001,
                "odds_matchup_key": "2026-04-21|BOS|NYY|2026-04-21T23:05:00Z",
                "match_status": "matched",
                "player_id": "mlb-pitcher:700001",
                "pitcher_mlb_id": 700001,
                "player_name": "Yesterday Arm",
                "market": "pitcher_strikeouts",
                "line": 5.5,
                "over_odds": -110,
                "under_odds": -110,
            }
        ],
    )
    _write_jsonl(
        tmp_path
        / "normalized"
        / "the_odds_api"
        / "date=2026-04-21"
        / "run=20260421T194500Z"
        / "prop_line_snapshots.jsonl",
        [
            {
                "line_snapshot_id": "line-2026-04-21-close",
                "official_date": "2026-04-21",
                "captured_at": "2026-04-21T19:45:00Z",
                "sportsbook": "draftkings",
                "sportsbook_title": "DraftKings",
                "event_id": "event-2026-04-21-1",
                "game_pk": 9001,
                "odds_matchup_key": "2026-04-21|BOS|NYY|2026-04-21T23:05:00Z",
                "match_status": "matched",
                "player_id": "mlb-pitcher:700001",
                "pitcher_mlb_id": 700001,
                "player_name": "Yesterday Arm",
                "market": "pitcher_strikeouts",
                "line": 5.5,
                "over_odds": -145,
                "under_odds": 120,
            }
        ],
    )


def _seed_prior_daily_candidates(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path
        / "normalized"
        / "daily_candidates"
        / "date=2026-04-21"
        / "run=20260421T182000Z"
        / "daily_candidates.jsonl",
        [
            {
                "daily_candidate_id": "daily-2026-04-21-1",
                "daily_candidate_run_id": "20260421T182000Z",
                "official_date": "2026-04-21",
                "slate_rank": 1,
                "actionable_rank": 1,
                "bet_placed": True,
                "model_version": "starter-strikeout-baseline-v1",
                "model_run_id": "20260421T150000Z",
                "sportsbook": "draftkings",
                "sportsbook_title": "DraftKings",
                "event_id": "event-2026-04-21-1",
                "game_pk": 9001,
                "player_id": "mlb-pitcher:700001",
                "pitcher_mlb_id": 700001,
                "player_name": "Yesterday Arm",
                "market": "pitcher_strikeouts",
                "line": 5.5,
                "selected_side": "over",
                "selected_odds": -110,
                "fair_odds": -135,
                "edge_pct": 0.052,
                "expected_value_pct": 0.061,
                "stake_fraction": 0.02,
                "line_snapshot_id": "line-2026-04-21-open",
                "captured_at": "2026-04-21T18:10:00Z",
                "selected_model_probability": 0.57,
                "selected_market_probability": 0.5,
            }
        ],
    )


def _seed_workflow_inputs(tmp_path: Path) -> None:
    _seed_baseline_run(
        tmp_path,
        end_date="2026-04-21",
        run_id="20260421T150000Z",
        intercept=7.0,
    )
    _seed_baseline_run(
        tmp_path,
        end_date="2026-04-22",
        run_id="20260422T010000Z",
        intercept=0.0,
    )
    _seed_feature_rows(tmp_path)
    _seed_odds_rows(tmp_path)
    _seed_prior_daily_candidates(tmp_path)


class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        fixed = cls(2026, 4, 22, 18, 30, 0, tzinfo=UTC)
        if tz is None:
            return fixed.replace(tzinfo=None)
        return fixed.astimezone(tz)


def test_build_daily_candidate_workflow_writes_sheet_and_refreshes_paper_results(
    monkeypatch, tmp_path
):
    _seed_workflow_inputs(tmp_path)
    monkeypatch.setattr("mlb_props_stack.edge.datetime", _FrozenDateTime)

    fixed_now = lambda: datetime(2026, 4, 22, 18, 30, 0, tzinfo=UTC)
    result = build_daily_candidate_workflow(
        target_date=date(2026, 4, 22),
        output_dir=tmp_path,
        now=fixed_now,
    )

    assert result.run_id == "20260422T183000Z"
    assert result.actionable_candidate_count == 1
    assert result.approved_wager_count == 1
    assert result.settled_result_count == 1
    assert result.pending_result_count == 1

    daily_rows = _load_jsonl(result.daily_candidates_path)
    assert len(daily_rows) == 1
    assert daily_rows[0]["official_date"] == "2026-04-22"
    assert daily_rows[0]["bet_placed"] is True

    paper_rows = _load_jsonl(result.paper_results_path)
    assert [row["official_date"] for row in paper_rows] == ["2026-04-21", "2026-04-22"]
    assert paper_rows[0]["paper_result"] == "win"
    assert paper_rows[0]["clv_outcome"] == "beat_closing_line"
    assert paper_rows[1]["paper_result"] == "pending"
    assert paper_rows[1]["settlement_status"] is None

    inference_model = _load_json(
        tmp_path
        / "normalized"
        / "starter_strikeout_inference"
        / "date=2026-04-22"
        / "run=20260422T183000Z"
        / "baseline_model.json"
    )
    assert inference_model["source_model_run_id"] == "20260421T150000Z"


def test_actionable_edge_blocked_by_final_gate_is_not_placed(tmp_path: Path):
    edge_rows = [
        {
            "candidate_id": "line-high-hold|starter-strikeout-baseline-v1",
            "official_date": "2026-04-23",
            "line_snapshot_id": "line-high-hold",
            "model_version": "starter-strikeout-baseline-v1",
            "model_run_id": "20260422T180000Z",
            "sportsbook": "draftkings",
            "sportsbook_title": "DraftKings",
            "event_id": "event-high-hold",
            "game_pk": 9003,
            "player_id": "mlb-pitcher:700003",
            "pitcher_mlb_id": 700003,
            "player_name": "High Hold Arm",
            "market": "pitcher_strikeouts",
            "line": 5.5,
            "over_odds": -200,
            "under_odds": -200,
            "captured_at": "2026-04-23T18:10:00Z",
            "evaluation_status": "actionable",
            "selected_side": "over",
            "selected_odds": -200,
            "selected_model_probability": 0.68,
            "selected_market_probability": 0.60,
            "edge_pct": 0.08,
            "expected_value_pct": 0.02,
            "stake_fraction": 0.01,
            "fair_odds": -213,
            "reason": "over clears minimum edge threshold (8.00% >= 4.50%)",
        }
    ]

    daily_rows = _build_daily_candidate_rows(
        edge_rows,
        run_id="20260423T181500Z",
        inference_run_id="20260423T181400Z",
        edge_candidate_run_id="20260423T181500Z",
        now=datetime(2026, 4, 23, 18, 15, tzinfo=UTC),
    )

    assert daily_rows[0]["evaluation_status"] == "actionable"
    assert daily_rows[0]["actionable_rank"] == 1
    assert daily_rows[0]["approved_rank"] is None
    assert daily_rows[0]["bet_placed"] is False
    assert daily_rows[0]["wager_approved"] is False
    assert daily_rows[0]["wager_blocked_reason"] == "hold above max"
    assert daily_rows[0]["wager_gate_details"]["hold"]["passed"] is False

    daily_run_dir = (
        tmp_path
        / "normalized"
        / "daily_candidates"
        / "date=2026-04-23"
        / "run=20260423T181500Z"
    )
    _write_jsonl(daily_run_dir / "daily_candidates.jsonl", daily_rows)

    paper_rows = _build_paper_result_rows(
        tmp_path,
        through_date=date(2026, 4, 23),
        run_id="20260423T181500Z",
    )
    assert paper_rows == []


def test_render_dashboard_page_uses_daily_candidates_and_paper_results(
    monkeypatch, tmp_path
):
    _seed_workflow_inputs(tmp_path)
    monkeypatch.setattr("mlb_props_stack.edge.datetime", _FrozenDateTime)
    fixed_now = lambda: datetime(2026, 4, 22, 18, 30, 0, tzinfo=UTC)
    build_daily_candidate_workflow(
        target_date=date(2026, 4, 22),
        output_dir=tmp_path,
        now=fixed_now,
    )

    class FakeBlock:
        def __init__(self, parent):
            self._parent = parent

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def markdown(self, body, **kwargs):
            self._parent.markdowns.append(body)

        def plotly_chart(self, figure, **kwargs):
            self._parent.plots.append(figure)

        def selectbox(self, label, options, index=0, **kwargs):
            self._parent.selectboxes[label] = list(options)
            return options[index]

        def toggle(self, label, value=False, **kwargs):
            self._parent.toggles[label] = value
            return value

        def text_input(self, label, value="", **kwargs):
            self._parent.text_inputs[label] = value
            return value

        def multiselect(self, label, options, default=None, **kwargs):
            self._parent.multiselects[label] = list(options)
            return list(default or [])

        def button(self, label, **kwargs):
            self._parent.buttons.append(label)
            return False

        def number_input(self, label, value=0, **kwargs):
            self._parent.number_inputs[label] = value
            return value

        def caption(self, text):
            self._parent.captions.append(text)

        def success(self, text):
            self._parent.successes.append(text)

    class FakeStreamlit(ModuleType):
        def __init__(self):
            super().__init__("streamlit")
            self.markdowns = []
            self.plots = []
            self.selectboxes = {}
            self.toggles = {}
            self.text_inputs = {}
            self.multiselects = {}
            self.buttons = []
            self.number_inputs = {}
            self.captions = []
            self.successes = []
            self.session_state = {}
            self.query_params = {}

        def set_page_config(self, **kwargs):
            self.page_config = kwargs

        def markdown(self, body, **kwargs):
            self.markdowns.append(body)

        def info(self, text):
            self.markdowns.append(text)

        def caption(self, text):
            self.captions.append(text)

        def plotly_chart(self, figure, **kwargs):
            self.plots.append(figure)

        def selectbox(self, label, options, index=0, **kwargs):
            self.selectboxes[label] = list(options)
            return options[index]

        def columns(self, count, **kwargs):
            resolved = count if isinstance(count, int) else len(count)
            return [FakeBlock(self) for _ in range(resolved)]

        def toggle(self, label, value=False, **kwargs):
            self.toggles[label] = value
            return value

        def text_input(self, label, value="", **kwargs):
            self.text_inputs[label] = value
            return value

        def multiselect(self, label, options, default=None, **kwargs):
            self.multiselects[label] = list(options)
            return list(default or [])

        def button(self, label, **kwargs):
            self.buttons.append(label)
            return False

        def number_input(self, label, value=0, **kwargs):
            self.number_inputs[label] = value
            return value

        def success(self, text):
            self.successes.append(text)

    fake_streamlit = FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)

    render_dashboard_page(output_dir=tmp_path, target_date=date(2026, 4, 22))

    assert fake_streamlit.page_config["page_title"] == "Strike Ops"
    assert "2026-04-22" in fake_streamlit.selectboxes["Slate date"]
    assert any("SLATE BOARD" in body for body in fake_streamlit.markdowns)
    assert any("Today Arm" in body for body in fake_streamlit.markdowns)
