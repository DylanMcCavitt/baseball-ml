"""MLflow experiment-tracking helpers for training and backtests."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
import json
from pathlib import Path
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class TrackingConfig:
    """Keep experiment logging config in one stable place."""

    tracking_uri: str = "file:./artifacts/mlruns"
    experiment_name: str = "mlb-props-stack"
    training_experiment_name: str = "mlb-props-stack-starter-strikeout-training"
    backtest_experiment_name: str = "mlb-props-stack-walk-forward-backtest"
    dashboard_module: str = "mlb_props_stack.dashboard.app"


@dataclass(frozen=True)
class TrackingRun:
    """Metadata for one active MLflow run."""

    tracking_uri: str
    experiment_name: str
    run_id: str


def _load_mlflow() -> Any:
    try:
        return import_module("mlflow")
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "MLflow is required for experiment tracking. Run `uv sync --extra dev` "
            "to install the repo dependencies."
        ) from exc


def _string_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _ensure_local_tracking_store(tracking_uri: str) -> None:
    if not tracking_uri.startswith("file:"):
        return
    location = tracking_uri.removeprefix("file:")
    if not location:
        return
    Path(location).expanduser().mkdir(parents=True, exist_ok=True)


@contextmanager
def start_experiment_run(
    *,
    experiment_name: str,
    run_name: str,
    tags: Mapping[str, Any] | None = None,
    config: TrackingConfig | None = None,
) -> Iterator[TrackingRun]:
    """Start one MLflow run and expose the assigned run id immediately."""
    tracking = config or TrackingConfig()
    _ensure_local_tracking_store(tracking.tracking_uri)
    mlflow = _load_mlflow()
    mlflow.set_tracking_uri(tracking.tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as active_run:
        if tags:
            mlflow.set_tags(
                {
                    key: _string_value(value)
                    for key, value in tags.items()
                    if value is not None
                }
            )
        yield TrackingRun(
            tracking_uri=tracking.tracking_uri,
            experiment_name=experiment_name,
            run_id=str(active_run.info.run_id),
        )


def log_run_params(params: Mapping[str, Any]) -> None:
    """Log a flat parameter mapping to the active MLflow run."""
    mlflow = _load_mlflow()
    payload = {
        key: _string_value(value)
        for key, value in params.items()
        if value is not None
    }
    if payload:
        mlflow.log_params(payload)


def log_run_metrics(metrics: Mapping[str, float | int | None]) -> None:
    """Log numeric metrics to the active MLflow run."""
    mlflow = _load_mlflow()
    payload = {
        key: float(value)
        for key, value in metrics.items()
        if value is not None
    }
    if payload:
        mlflow.log_metrics(payload)


def log_run_artifact(path: Path | str, *, artifact_path: str | None = None) -> None:
    """Log one file or one directory to the active MLflow run."""
    mlflow = _load_mlflow()
    artifact = Path(path)
    if artifact.is_dir():
        mlflow.log_artifacts(str(artifact), artifact_path=artifact_path)
        return
    mlflow.log_artifact(str(artifact), artifact_path=artifact_path)
