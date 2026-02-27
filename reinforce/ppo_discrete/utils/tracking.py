from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .experiment import to_jsonable


class MetricTracker:
    """Metrics logger: MLflow (primary) + JSONL (offline fallback).

    Outputs:
    - logs/metrics.jsonl  (always written)
    - logs/events.jsonl   (always written)
    - MLflow run          (if mlflow_tracking_uri is set)
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        run_name: str,
        mlflow_tracking_uri: str = "",
        mlflow_experiment: str = "ppo_discrete",
        mlflow_run_name: str = "",
        config: dict[str, Any] | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.run_name = run_name
        self.metrics_path = self.logs_dir / "metrics.jsonl"
        self.events_path = self.logs_dir / "events.jsonl"
        self._mlflow = None
        self._mlflow_run = None
        safe_config_obj = to_jsonable(config) if config is not None else None
        safe_config: dict[str, Any] | None = safe_config_obj if isinstance(safe_config_obj, dict) else None

        if mlflow_tracking_uri:
            try:
                import mlflow

                mlflow.set_tracking_uri(mlflow_tracking_uri)
                mlflow.set_experiment(mlflow_experiment)
                self._mlflow = mlflow
                self._mlflow_run = mlflow.start_run(run_name=(mlflow_run_name or run_name))
                if safe_config is not None:
                    flat = _flatten_dict(safe_config)
                    for k, v in flat.items():
                        mlflow.log_param(k[:250], str(v)[:500])
            except Exception as e:
                self.log_event("mlflow_init_failed", {"error": str(e), "uri": mlflow_tracking_uri})

        self.log_event(
            "run_start",
            {
                "run_name": run_name,
                "pid": os.getpid(),
                "time": time.time(),
            },
        )
        if safe_config is not None:
            cfg_path = self.run_dir / "config" / "run_config.json"
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(json.dumps(safe_config, indent=2, ensure_ascii=True), encoding="utf-8")

    def log_metrics(self, step: int, metrics: dict[str, float | int]) -> None:
        row = {"step": int(step), "time": time.time(), **{k: _safe_numeric(v) for k, v in metrics.items()}}
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

        if self._mlflow is not None:
            clean = {k: vv for k, v in metrics.items() if (vv := _safe_numeric(v)) is not None}
            if clean:
                try:
                    self._mlflow.log_metrics(clean, step=step)
                except Exception:
                    pass

    def log_event(self, name: str, payload: dict[str, Any] | None = None) -> None:
        row = {"event": name, "time": time.time(), "payload": to_jsonable(payload or {})}
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    def close(self) -> None:
        self.log_event("run_end", {})
        if self._mlflow is not None and self._mlflow_run is not None:
            try:
                self._mlflow.end_run()
            except Exception:
                pass


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


def _safe_numeric(v: Any) -> float | int | None:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return v
    try:
        return float(v)
    except Exception:
        return None
