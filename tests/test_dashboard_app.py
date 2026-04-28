from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from mlb_props_stack.dashboard.app import _query_params_dict, _render_nav_controls
from mlb_props_stack.dashboard.screens.board import _board_table_html


class _FakeNavColumn:
    def __init__(self, parent: "_FakeNavStreamlit") -> None:
        self._parent = parent

    def button(self, label: str, **kwargs: object) -> bool:
        self._parent.buttons.append((label, dict(kwargs)))
        return label == self._parent.clicked_label


class _FakeNavStreamlit:
    def __init__(self, *, clicked_label: str) -> None:
        self.clicked_label = clicked_label
        self.buttons: list[tuple[str, dict[str, object]]] = []
        self.query_params = {"screen": "board", "board_date": "2026-04-23"}
        self.rerun_called = False

    def columns(self, count: int, **kwargs: object) -> list[_FakeNavColumn]:
        return [_FakeNavColumn(self) for _ in range(count)]

    def rerun(self) -> None:
        self.rerun_called = True


def test_nav_controls_update_query_params_for_regular_screens() -> None:
    fake_streamlit = _FakeNavStreamlit(clicked_label="BACKTEST \u23183")

    _render_nav_controls(
        streamlit_module=fake_streamlit,
        active_screen="board",
        board_date=date(2026, 4, 23),
        selected_pitcher_id="mlb-pitcher:594798",
    )

    assert fake_streamlit.query_params == {
        "screen": "backtest",
        "board_date": "2026-04-23",
    }
    assert fake_streamlit.rerun_called is True
    assert [label for label, _ in fake_streamlit.buttons] == [
        "BOARD \u23181",
        "PITCHER \u23182",
        "BACKTEST \u23183",
        "MLFLOW \u23184",
        "FEATURES \u23185",
        "CONFIG \u23186",
    ]


def test_nav_controls_keep_pitcher_context_for_pitcher_screen() -> None:
    fake_streamlit = _FakeNavStreamlit(clicked_label="PITCHER \u23182")

    _render_nav_controls(
        streamlit_module=fake_streamlit,
        active_screen="board",
        board_date=date(2026, 4, 23),
        selected_pitcher_id="mlb-pitcher:594798",
    )

    assert fake_streamlit.query_params == {
        "screen": "pitcher",
        "board_date": "2026-04-23",
        "pitcher_id": "mlb-pitcher:594798",
    }
    assert fake_streamlit.rerun_called is True


def test_query_params_dict_handles_list_values_from_browser_runtime() -> None:
    class FakeStreamlit:
        query_params = {
            "screen": ["pitcher"],
            "board_date": ["2026-04-20"],
            "pitcher_id": ["mlb-pitcher:700001"],
        }

    assert _query_params_dict(FakeStreamlit()) == {
        "screen": "pitcher",
        "board_date": "2026-04-20",
        "pitcher_id": "mlb-pitcher:700001",
    }


def test_board_table_renders_sportsbook_provenance_and_group_summary() -> None:
    html = _board_table_html(
        pd.DataFrame(
            [
                {
                    "cleared": False,
                    "pitcher_id": "mlb-pitcher:594798",
                    "pitcher": "Jacob deGrom",
                    "team": "TEX",
                    "opp": "OAK",
                    "hand": "R",
                    "sportsbook": "BetRivers",
                    "source_event_id": "528817a2bf72047f1124e81a7ae55de9",
                    "line_snapshot_id": "prop-line:betrivers:event:pitcher:5_5:run",
                    "market_last_update": datetime(2026, 4, 23, 19, 47, 32, tzinfo=UTC),
                    "line": 5.5,
                    "side": "under",
                    "p_model": 0.72,
                    "p_market": 0.50,
                    "american": 200,
                    "edge": 0.22,
                    "kelly_units": 0.0,
                    "conf": 0.72,
                    "line_row_count": 3,
                    "hidden_line_row_count": 2,
                    "sportsbook_count": 2,
                    "line_group_summary": "BetRivers UNDER 5.5, DraftKings UNDER 6.5",
                    "note": "hold above max",
                }
            ]
        ),
        selected_pitcher_id=None,
    )

    assert "SOURCE" in html
    assert "GROUP" in html
    assert "BetRivers" in html
    assert "event 528817a2" in html
    assert "+2 hidden" in html
    assert "DraftKings UNDER 6.5" in html
