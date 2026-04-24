from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from mlb_props_stack.model_comparison import compare_starter_strikeout_baselines
from mlb_props_stack.tracking import TrackingConfig
from tests.test_modeling import (
    FakeStatcastClient,
    _build_outcome_csv,
    _seed_feature_run,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def _tracking_config(tmp_path: Path) -> TrackingConfig:
    return TrackingConfig(tracking_uri=f"file:{tmp_path / 'artifacts' / 'mlruns'}")


def _seed_comparison_feature_window(tmp_path: Path) -> tuple[date, date, dict[tuple[str, int], str]]:
    start_date = date(2026, 4, 16)
    end_date = start_date + timedelta(days=4)
    responses: dict[tuple[str, int], str] = {}

    for date_index in range(5):
        official_date = start_date + timedelta(days=date_index)
        feature_rows: list[dict[str, object]] = []
        for pitcher_index in range(2):
            pitcher_id = 680800 + pitcher_index
            game_pk = 824440 + (date_index * 10) + pitcher_index
            pitcher_k_rate = round(0.18 + (0.02 * pitcher_index) + (0.01 * date_index), 6)
            lineup_k_rate = round(0.21 + (0.015 * ((date_index + pitcher_index) % 3)), 6)
            lineup_contact_rate = round(0.73 - (0.015 * date_index) + (0.005 * pitcher_index), 6)
            expected_leash_batters_faced = float(23 + date_index + (2 * pitcher_index))
            naive_benchmark = pitcher_k_rate * expected_leash_batters_faced
            actual_strikeouts = int(
                round(
                    naive_benchmark
                    + (20.0 * (lineup_k_rate - 0.22))
                    - (12.0 * (lineup_contact_rate - 0.70))
                )
            )
            home_away = "home" if pitcher_index == 0 else "away"
            pitcher_hand = "R" if pitcher_index == 0 else "L"
            feature_rows.append(
                {
                    "game_pk": game_pk,
                    "pitcher_id": pitcher_id,
                    "pitcher_name": f"Pitcher {pitcher_index + 1}",
                    "team_side": home_away,
                    "team_abbreviation": "CLE" if home_away == "home" else "HOU",
                    "opponent_team_abbreviation": "HOU" if home_away == "home" else "CLE",
                    "opponent_team_name": "Houston Astros" if home_away == "home" else "Cleveland Guardians",
                    "pitcher_feature_row_id": f"pitcher-feature:{game_pk}:{pitcher_id}",
                    "lineup_feature_row_id": f"lineup-feature:{game_pk}:{pitcher_id}",
                    "game_context_feature_row_id": f"game-context:{game_pk}:{pitcher_id}",
                    "lineup_snapshot_id": f"lineup:{game_pk}:{official_date.isoformat()}",
                    "pitcher_hand": pitcher_hand,
                    "pitch_sample_size": 480 + (date_index * 10) + (pitcher_index * 15),
                    "plate_appearance_sample_size": 105 + (date_index * 4) + (pitcher_index * 5),
                    "pitcher_k_rate": pitcher_k_rate,
                    "pitcher_k_rate_vs_rhh": round(pitcher_k_rate + 0.015, 6),
                    "pitcher_k_rate_vs_lhh": round(pitcher_k_rate - 0.01, 6),
                    "swinging_strike_rate": round(0.11 + (0.005 * pitcher_index), 6),
                    "pitcher_whiff_rate_vs_rhh": round(0.24 + (0.01 * date_index), 6),
                    "pitcher_whiff_rate_vs_lhh": round(0.22 + (0.008 * date_index), 6),
                    "csw_rate": round(0.27 + (0.01 * date_index), 6),
                    "pitch_type_usage": {"FF": 0.58 - (0.02 * pitcher_index), "SL": 0.42 + (0.02 * pitcher_index)},
                    "average_release_speed": 94.0 + pitcher_index,
                    "release_speed_delta_vs_baseline": round(0.1 * date_index, 6),
                    "average_release_extension": 6.1 + (0.05 * pitcher_index),
                    "release_extension_delta_vs_baseline": round(0.02 * date_index, 6),
                    "recent_batters_faced": 70 + (date_index * 2),
                    "recent_pitch_count": 260 + (date_index * 8),
                    "rest_days": 5 + pitcher_index,
                    "last_start_pitch_count": 92 + (date_index * 2),
                    "last_start_batters_faced": 24 + pitcher_index,
                    "lineup_status": "confirmed",
                    "lineup_is_confirmed": True,
                    "lineup_size": 9,
                    "available_batter_feature_count": 9,
                    "projected_lineup_k_rate": lineup_k_rate,
                    "projected_lineup_k_rate_vs_pitcher_hand": lineup_k_rate + 0.01,
                    "lineup_k_rate_vs_rhp": lineup_k_rate + 0.012,
                    "lineup_k_rate_vs_lhp": lineup_k_rate - 0.008,
                    "projected_lineup_chase_rate": round(0.28 + (0.005 * date_index), 6),
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
            responses[(official_date.isoformat(), pitcher_id)] = _build_outcome_csv(
                official_date=official_date,
                game_pk=game_pk,
                pitcher_id=pitcher_id,
                strikeout_count=actual_strikeouts,
                plate_appearance_count=26 + pitcher_index,
                home_team="CLE" if home_away == "home" else "HOU",
                away_team="HOU" if home_away == "home" else "CLE",
                pitcher_hand=pitcher_hand,
            )
        _seed_feature_run(tmp_path, official_date=official_date, feature_rows=feature_rows)
    return start_date, end_date, responses


def _seed_comparison_odds(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path
        / "normalized"
        / "the_odds_api"
        / "date=2026-04-20"
        / "run=20260420T190000Z"
        / "prop_line_snapshots.jsonl",
        [
            {
                "line_snapshot_id": "line-open",
                "official_date": "2026-04-20",
                "captured_at": "2026-04-20T19:00:00Z",
                "sportsbook": "draftkings",
                "sportsbook_title": "DraftKings",
                "event_id": "event-1",
                "game_pk": 824480,
                "odds_matchup_key": "2026-04-20|HOU|CLE|2026-04-20T20:00:00Z",
                "match_status": "matched",
                "commence_time": "2026-04-20T20:00:00Z",
                "player_id": "mlb-pitcher:680800",
                "pitcher_mlb_id": 680800,
                "player_name": "Pitcher 1",
                "market": "pitcher_strikeouts",
                "line": 5.5,
                "over_odds": 120,
                "under_odds": -145,
            },
            {
                "line_snapshot_id": "line-close",
                "official_date": "2026-04-20",
                "captured_at": "2026-04-20T19:45:00Z",
                "sportsbook": "draftkings",
                "sportsbook_title": "DraftKings",
                "event_id": "event-1",
                "game_pk": 824480,
                "odds_matchup_key": "2026-04-20|HOU|CLE|2026-04-20T20:00:00Z",
                "match_status": "matched",
                "commence_time": "2026-04-20T20:00:00Z",
                "player_id": "mlb-pitcher:680800",
                "pitcher_mlb_id": 680800,
                "player_name": "Pitcher 1",
                "market": "pitcher_strikeouts",
                "line": 5.5,
                "over_odds": 105,
                "under_odds": -125,
            },
        ],
    )


def test_compare_starter_strikeout_baselines_writes_core_vs_expanded_report(tmp_path) -> None:
    start_date, end_date, responses = _seed_comparison_feature_window(tmp_path)
    _seed_comparison_odds(tmp_path)

    result = compare_starter_strikeout_baselines(
        start_date=start_date,
        end_date=end_date,
        output_dir=tmp_path,
        client=FakeStatcastClient(responses),
        now=lambda: datetime(2026, 4, 21, 18, 0, tzinfo=UTC),
        tracking_config=_tracking_config(tmp_path),
    )

    report = json.loads(result.comparison_path.read_text(encoding="utf-8"))
    markdown = result.comparison_markdown_path.read_text(encoding="utf-8")
    core = report["variants"]["core"]
    expanded = report["variants"]["expanded"]

    assert result.recommendation in {"keep_core_only", "promote_expanded_candidate"}
    assert result.core_training_run_id != result.expanded_training_run_id
    assert result.core_backtest_run_id != result.expanded_backtest_run_id
    assert report["date_window"] == {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }
    assert core["training"]["feature_schema"]["feature_set"] == "core"
    assert expanded["training"]["feature_schema"]["feature_set"] == "expanded"
    assert "projected_lineup_k_rate" not in core["training"]["feature_schema"]["active_optional_features"]
    assert "projected_lineup_k_rate" in expanded["training"]["feature_schema"]["active_optional_features"]
    assert "scoreable_rows" in core["backtest"]
    assert "approved_wagers" in expanded["final_wager_gates"]
    assert expanded["leakage_audit"]["timestamp_violation_count"] == 0
    assert "Starter Strikeout Model Variant Comparison" in markdown
    assert "Expanded Optional Features" in markdown
