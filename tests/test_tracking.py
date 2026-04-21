from mlb_props_stack.dashboard.app import build_dashboard_banner
from mlb_props_stack.tracking import TrackingConfig


def test_tracking_config_uses_reserved_mlflow_location():
    config = TrackingConfig()

    assert config.tracking_uri == "file:./artifacts/mlruns"
    assert config.dashboard_module == "mlb_props_stack.dashboard.app"


def test_dashboard_banner_marks_placeholder_module():
    assert "future Streamlit app" in build_dashboard_banner()
