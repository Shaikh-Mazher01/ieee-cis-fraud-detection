from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(profile: str = "real") -> dict:
    """profile = 'real' (Kaggle files) or 'synthetic' (generated fake data, same schema)."""
    if profile not in ("real", "synthetic"):
        raise ValueError("profile must be 'real' or 'synthetic'")
    with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["profile"] = profile

    raw_dir = ROOT / ("data/synthetic" if profile == "synthetic" else cfg["data"]["raw_dir"])
    out = ROOT / "outputs" / profile
    cfg["paths"] = {
        "raw_dir": raw_dir,
        "interim_dir": ROOT / "data" / "interim" / profile,
        "out_dir": out,
        "models_dir": out / "models",
        "figures_dir": out / "figures",
        "tables_dir": out / "tables",
        "reports_dir": out / "reports",
        "sql_dir": ROOT / "sql",
    }
    for key, p in cfg["paths"].items():
        if key not in ("raw_dir", "sql_dir"):
            p.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["merged"] = cfg["paths"]["interim_dir"] / "merged.parquet"
    cfg["paths"]["bundle"] = cfg["paths"]["models_dir"] / "bundle.joblib"
    cfg["paths"]["mlflow_uri"] = f"sqlite:///{(out / 'mlflow.db').as_posix()}"
    return cfg
