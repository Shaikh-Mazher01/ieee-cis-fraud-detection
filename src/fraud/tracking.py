from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class Tracker:
    def __init__(self, cfg: dict):
        self.enabled = bool(cfg["mlflow"]["enabled"])
        self.mlflow = None
        if not self.enabled:
            return
        try:
            import mlflow

            mlflow.set_tracking_uri(cfg["paths"]["mlflow_uri"])
            mlflow.set_experiment(f"{cfg['mlflow']['experiment']}-{cfg['profile']}")
            self.mlflow = mlflow
        except Exception as err:  # never let tracking break training
            log.warning("MLflow disabled: %s", err)
            self.enabled = False

    def log_run(self, name: str, params: dict, metrics: dict, tags: dict | None = None) -> None:
        if not self.enabled:
            return
        with self.mlflow.start_run(run_name=name):
            self.mlflow.log_params({k: v for k, v in params.items() if v is not None})
            self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            if tags:
                self.mlflow.set_tags(tags)
