from __future__ import annotations

import json
import runpy
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import ModuleType

from mlb_props_stack.cli import main
from mlb_props_stack.ingest import ingest_statcast_features_for_date
from mlb_props_stack.modeling import train_starter_strikeout_baseline
from tests.test_modeling import FakeStatcastClient, _build_outcome_csv, _seed_feature_run
from tests.test_statcast_feature_ingest import (
    StubStatcastClient,
    seed_postlock_mlb_metadata,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def _dashboard_app_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mlb_props_stack"
        / "dashboard"
        / "app.py"
    )


def _seed_dashboard_artifacts(output_dir: Path) -> None:
    _write_jsonl(
        output_dir
        / "normalized"
        / "daily_candidates"
        / "date=2026-04-22"
        / "run=20260422T180000Z"
        / "daily_candidates.jsonl",
        [
            {
                "daily_candidate_id": "candidate-1",
                "official_date": "2026-04-22",
                "slate_rank": 1,
                "bet_placed": True,
                "player_name": "Today Arm",
                "sportsbook_title": "DraftKings",
                "line": 5.5,
                "selected_side": "over",
                "selected_odds": 120,
                "edge_pct": 0.042,
                "expected_value_pct": 0.031,
                "stake_fraction": 0.0125,
                "captured_at": "2026-04-22T18:25:00Z",
            }
        ],
    )
    _write_jsonl(
        output_dir
        / "normalized"
        / "paper_results"
        / "date=2026-04-22"
        / "run=20260422T190000Z"
        / "paper_results.jsonl",
        [
            {
                "paper_result_id": "candidate-1|paper",
                "official_date": "2026-04-22",
                "actionable_rank": 1,
                "player_name": "Today Arm",
                "sportsbook_title": "DraftKings",
                "line": 5.5,
                "selected_side": "over",
                "edge_pct": 0.042,
                "paper_result": "win",
                "profit_units": 0.015,
                "stake_fraction": 0.0125,
                "settlement_status": "win",
                "clv_outcome": "beat_closing_line",
            }
        ],
    )


def _seed_training_fixture_runs(output_dir: Path) -> tuple[date, date, dict[tuple[str, int], str]]:
    outcome_csv_by_pitcher_and_date: dict[tuple[str, int], str] = {}
    start_date = date(2026, 4, 16)
    for date_index in range(5):
        official_date = start_date + timedelta(days=date_index)
        feature_rows: list[dict[str, object]] = []
        for pitcher_index in range(2):
            pitcher_id = 680800 + pitcher_index
            game_pk = 824440 + (date_index * 10) + pitcher_index
            pitcher_k_rate = round(
                0.18 + (0.02 * pitcher_index) + (0.01 * date_index),
                6,
            )
            lineup_k_rate = round(
                0.21 + (0.015 * ((date_index + pitcher_index) % 3)),
                6,
            )
            lineup_contact_rate = round(
                0.73 - (0.015 * date_index) + (0.005 * pitcher_index),
                6,
            )
            expected_leash_batters_faced = float(23 + date_index + (2 * pitcher_index))
            naive_benchmark = pitcher_k_rate * expected_leash_batters_faced
            home_away = "home" if pitcher_index == 0 else "away"
            actual_strikeouts = int(
                round(
                    naive_benchmark
                    + (20.0 * (lineup_k_rate - 0.22))
                    - (12.0 * (lineup_contact_rate - 0.70))
                    + (0.6 if home_away == "home" else -0.3)
                )
            )
            pitcher_hand = "R" if pitcher_index == 0 else "L"
            feature_rows.append(
                {
                    "game_pk": game_pk,
                    "pitcher_id": pitcher_id,
                    "pitcher_name": f"Pitcher {pitcher_index + 1}",
                    "team_side": "home" if home_away == "home" else "away",
                    "team_abbreviation": "CLE" if home_away == "home" else "HOU",
                    "opponent_team_abbreviation": (
                        "HOU" if home_away == "home" else "CLE"
                    ),
                    "opponent_team_name": (
                        "Houston Astros"
                        if home_away == "home"
                        else "Cleveland Guardians"
                    ),
                    "pitcher_feature_row_id": f"pitcher-feature:{game_pk}:{pitcher_id}",
                    "lineup_feature_row_id": f"lineup-feature:{game_pk}:{pitcher_id}",
                    "game_context_feature_row_id": f"game-context:{game_pk}:{pitcher_id}",
                    "lineup_snapshot_id": f"lineup:{game_pk}:{official_date.isoformat()}",
                    "pitcher_hand": pitcher_hand,
                    "pitch_sample_size": 480 + (date_index * 10) + (pitcher_index * 15),
                    "plate_appearance_sample_size": 105
                    + (date_index * 4)
                    + (pitcher_index * 5),
                    "pitcher_k_rate": pitcher_k_rate,
                    "swinging_strike_rate": round(0.11 + (0.005 * pitcher_index), 6),
                    "csw_rate": round(0.27 + (0.01 * date_index), 6),
                    "pitch_type_usage": {
                        "FF": 0.58 - (0.02 * pitcher_index),
                        "SL": 0.42 + (0.02 * pitcher_index),
                    },
                    "average_release_speed": 94.0 + pitcher_index,
                    "release_speed_delta_vs_baseline": round(0.1 * date_index, 6),
                    "average_release_extension": 6.1 + (0.05 * pitcher_index),
                    "release_extension_delta_vs_baseline": round(0.02 * date_index, 6),
                    "recent_batters_faced": 70 + (date_index * 2),
                    "recent_pitch_count": 260 + (date_index * 8),
                    "rest_days": 5 + pitcher_index,
                    "last_start_pitch_count": 92 + (date_index * 2),
                    "last_start_batters_faced": 24 + pitcher_index,
                    "lineup_status": "confirmed" if date_index % 2 == 0 else "projected",
                    "lineup_is_confirmed": date_index % 2 == 0,
                    "lineup_size": 9,
                    "available_batter_feature_count": 9,
                    "projected_lineup_k_rate": lineup_k_rate,
                    "projected_lineup_k_rate_vs_pitcher_hand": lineup_k_rate + 0.01,
                    "projected_lineup_chase_rate": round(
                        0.28 + (0.005 * date_index),
                        6,
                    ),
                    "projected_lineup_contact_rate": lineup_contact_rate,
                    "lineup_continuity_count": 6 + pitcher_index,
                    "lineup_continuity_ratio": round((6 + pitcher_index) / 9, 6),
                    "lineup_player_ids": [710000 + slot for slot in range(9)],
                    "home_away": home_away,
                    "day_night": "night",
                    "double_header": "N",
                    "expected_leash_pitch_count": 95.0 + date_index,
                    "expected_leash_batters_faced": expected_leash_batters_faced,
                }
            )
            outcome_csv_by_pitcher_and_date[(official_date.isoformat(), pitcher_id)] = (
                _build_outcome_csv(
                    official_date=official_date,
                    game_pk=game_pk,
                    pitcher_id=pitcher_id,
                    strikeout_count=actual_strikeouts,
                    plate_appearance_count=26 + pitcher_index,
                    home_team="CLE" if home_away == "home" else "HOU",
                    away_team="HOU" if home_away == "home" else "CLE",
                    pitcher_hand=pitcher_hand,
                )
            )
        _seed_feature_run(output_dir, official_date=official_date, feature_rows=feature_rows)
    return start_date, start_date + timedelta(days=4), outcome_csv_by_pitcher_and_date


class _FakeColumn:
    def __init__(self, sink: list[tuple[str, object]]) -> None:
        self._sink = sink

    def metric(self, label: str, value: object) -> None:
        self._sink.append((label, value))


class _FakeSidebar:
    def __init__(self) -> None:
        self.last_options: list[str] = []

    def selectbox(self, label: str, options: list[str], index: int) -> str:
        assert label == "Slate date"
        self.last_options = list(options)
        return options[index]


class _FakeStreamlit(ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.sidebar = _FakeSidebar()
        self.metric_calls: list[tuple[str, object]] = []
        self.dataframes: list[tuple[object, dict[str, object]]] = []

    def set_page_config(self, **kwargs: object) -> None:
        self.page_config = kwargs

    def title(self, text: str) -> None:
        self.title_text = text

    def caption(self, text: str) -> None:
        self.caption_text = text

    def info(self, text: str) -> None:
        self.info_text = text

    def columns(self, count: int) -> list[_FakeColumn]:
        return [_FakeColumn(self.metric_calls) for _ in range(count)]

    def subheader(self, text: str) -> None:
        self.last_subheader = text

    def dataframe(self, data: object, **kwargs: object) -> None:
        self.dataframes.append((data, kwargs))


def test_dashboard_file_entrypoint_smoke(monkeypatch, tmp_path) -> None:
    _seed_dashboard_artifacts(tmp_path)
    fake_streamlit = _FakeStreamlit()
    monkeypatch.setenv("MLB_PROPS_STACK_DATA_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)

    runpy.run_path(str(_dashboard_app_path()), run_name="__main__")

    assert fake_streamlit.title_text == "MLB Props Stack"
    assert fake_streamlit.sidebar.last_options == ["2026-04-22"]
    assert len(fake_streamlit.metric_calls) == 8
    assert len(fake_streamlit.dataframes) == 3


def test_statcast_feature_cli_smoke_covers_historical_metadata_backfill(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    _, probable_starters_path, _ = seed_postlock_mlb_metadata(tmp_path)
    stub_client = StubStatcastClient()
    captured: dict[str, object] = {}
    fixed_now = lambda: datetime(2026, 4, 22, 18, 0, tzinfo=UTC)

    def _run_smoke_ingest(*, target_date: date, output_dir: str, history_days: int):
        result = ingest_statcast_features_for_date(
            target_date=target_date,
            output_dir=output_dir,
            history_days=history_days,
            client=stub_client,
            now=fixed_now,
        )
        captured["result"] = result
        return result

    monkeypatch.setattr(
        "mlb_props_stack.cli.ingest_statcast_features_for_date",
        _run_smoke_ingest,
    )

    main(
        [
            "ingest-statcast-features",
            "--date",
            "2026-04-21",
            "--output-dir",
            str(tmp_path),
            "--history-days",
            "30",
        ]
    )
    output = capsys.readouterr().out
    result = captured["result"]

    assert result.mlb_probable_starters_path == probable_starters_path
    assert result.pitcher_daily_features_path.exists()
    assert "Statcast feature build complete for 2026-04-21" in output
    assert f"mlb_probable_starters_path={probable_starters_path}" in output


def test_training_cli_smoke_writes_seeded_baseline_artifacts(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    start_date, end_date, responses = _seed_training_fixture_runs(tmp_path)
    stub_client = FakeStatcastClient(responses)
    captured: dict[str, object] = {}
    fixed_now = lambda: datetime(2026, 4, 21, 18, 0, tzinfo=UTC)

    def _run_smoke_training(*, start_date: date, end_date: date, output_dir: str):
        result = train_starter_strikeout_baseline(
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            client=stub_client,
            now=fixed_now,
        )
        captured["result"] = result
        return result

    monkeypatch.setattr(
        "mlb_props_stack.cli.train_starter_strikeout_baseline",
        _run_smoke_training,
    )

    main(
        [
            "train-starter-strikeout-baseline",
            "--start-date",
            start_date.isoformat(),
            "--end-date",
            end_date.isoformat(),
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = capsys.readouterr().out
    result = captured["result"]
    evaluation = json.loads(result.evaluation_path.read_text(encoding="utf-8"))

    assert result.row_count == 10
    assert result.model_path.exists()
    assert result.evaluation_path.exists()
    assert evaluation["model_version"] == "starter-strikeout-baseline-v1"
    assert "Starter strikeout baseline training complete" in output
    assert f"evaluation_path={result.evaluation_path}" in output
