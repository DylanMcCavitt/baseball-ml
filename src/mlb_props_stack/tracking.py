"""Reserved experiment-tracking config for later MLflow integration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackingConfig:
    """Keep future experiment logging config in one stable place."""

    tracking_uri: str = "file:./artifacts/mlruns"
    experiment_name: str = "mlb-props-stack"
    dashboard_module: str = "mlb_props_stack.dashboard.app"
