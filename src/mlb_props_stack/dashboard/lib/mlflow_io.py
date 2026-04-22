"""MLflow access helpers for the dashboard registry screen."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib import import_module
from typing import Any


def _load_mlflow() -> Any:
    try:
        return import_module("mlflow")
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "MLflow is required for the dashboard registry screen. Run `uv sync --extra dev`."
        ) from exc


def search_runs(*, tracking_uri: str, experiment_names: list[str]) -> list[dict[str, Any]]:
    """Return flattened MLflow runs across the requested experiments."""
    if not experiment_names:
        return []
    mlflow = _load_mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    try:
        runs = mlflow.search_runs(
            experiment_names=experiment_names,
            output_format="list",
        )
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for run in runs:
        start_time = getattr(run.info, "start_time", None)
        started_at = (
            datetime.fromtimestamp(start_time / 1000, tz=UTC)
            if start_time is not None
            else None
        )
        rows.append(
            {
                "run_id": str(run.info.run_id),
                "run_name": (
                    str(run.data.tags.get("mlflow.runName"))
                    if run.data.tags.get("mlflow.runName")
                    else str(run.info.run_id)
                ),
                "experiment_id": str(run.info.experiment_id),
                "status": str(run.info.status),
                "lifecycle_stage": str(run.info.lifecycle_stage),
                "started_at": started_at,
                "metrics": {str(key): value for key, value in run.data.metrics.items()},
                "params": {str(key): value for key, value in run.data.params.items()},
                "tags": {str(key): value for key, value in run.data.tags.items()},
            }
        )
    rows.sort(
        key=lambda row: (
            row["started_at"] or datetime.min.replace(tzinfo=UTC),
            row["run_id"],
        ),
        reverse=True,
    )
    return rows


def registered_versions_by_run_id(*, tracking_uri: str) -> dict[str, dict[str, str]]:
    """Return the latest registered-model metadata keyed by run id."""
    mlflow = _load_mlflow()
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    mapping: dict[str, dict[str, str]] = {}
    try:
        models = client.search_registered_models()
    except Exception:
        return mapping
    for model in models:
        for version in getattr(model, "latest_versions", []) or []:
            run_id = getattr(version, "run_id", None)
            if not run_id:
                continue
            mapping[str(run_id)] = {
                "model_name": str(getattr(version, "name", "")),
                "version": str(getattr(version, "version", "")),
                "stage": str(getattr(version, "current_stage", "None")),
            }
    return mapping


def transition_stage(
    *,
    tracking_uri: str,
    model_name: str,
    version: str,
    stage: str,
) -> None:
    """Transition one registered model version stage."""
    mlflow = _load_mlflow()
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=stage == "Production",
    )
