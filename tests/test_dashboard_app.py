from __future__ import annotations

from datetime import date

from mlb_props_stack.dashboard.app import _render_nav_controls


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
