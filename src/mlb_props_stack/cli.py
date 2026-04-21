"""Tiny CLI entrypoint for the scaffold."""

from .backtest import BACKTEST_CHECKLIST
from .config import StackConfig
from .tracking import TrackingConfig


def render_runtime_summary() -> str:
    """Return a human-readable snapshot of the local runtime baseline."""
    config = StackConfig()
    tracking = TrackingConfig()
    lines = [
        "MLB Props Stack",
        f"market={config.market}",
        f"min_edge_pct={config.min_edge_pct:.2%}",
        f"kelly_fraction={config.kelly_fraction:.2f}",
        f"tracking_uri={tracking.tracking_uri}",
        f"dashboard_module={tracking.dashboard_module}",
        "backtest_checklist:",
    ]
    lines.extend(f"- {item}" for item in BACKTEST_CHECKLIST)
    return "\n".join(lines)


def main() -> None:
    print(render_runtime_summary())


if __name__ == "__main__":
    main()
