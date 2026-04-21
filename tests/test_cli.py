from mlb_props_stack.cli import render_runtime_summary


def test_runtime_summary_includes_future_hooks():
    summary = render_runtime_summary()

    assert "MLB Props Stack" in summary
    assert "tracking_uri=file:./artifacts/mlruns" in summary
    assert "dashboard_module=mlb_props_stack.dashboard.app" in summary
