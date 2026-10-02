from __future__ import annotations

import logging

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import load_merged  # noqa: E402

log = logging.getLogger(__name__)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with a separate bucket for missing values."""
    e_nan, a_nan = np.isnan(expected).mean(), np.isnan(actual).mean()
    e, a = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    if len(e) == 0 or len(a) == 0:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.array([-np.inf, np.inf]) if len(edges) < 2 else np.array([-np.inf, edges[0], np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e) * (1 - e_nan)
    pa = np.histogram(a, edges)[0] / len(a) * (1 - a_nan)
    pe, pa = np.append(pe, e_nan), np.append(pa, a_nan)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def drift_report(cfg: dict) -> pd.DataFrame:
    id_col = cfg["data"]["id_col"]
    split = pd.read_parquet(cfg["paths"]["tables_dir"] / "split_ids.parquet")
    bundle = joblib.load(cfg["paths"]["bundle"])
    fb = bundle["builder"]
    n = cfg["drift"]["sample_size"]
    rng = np.random.default_rng(cfg["seed"])

    def sample(name):
        ids = split.loc[split.split == name, id_col].to_numpy()
        ids = rng.choice(ids, min(n, len(ids)), replace=False)
        return fb.transform(load_merged(cfg, filters=[(id_col, "in", ids.tolist())]))

    ref, new = sample("train"), sample("test")
    rows = []
    for c in fb.feature_names_:
        a, b = ref[c].to_numpy(dtype="float64"), new[c].to_numpy(dtype="float64")
        ok_a, ok_b = a[~np.isnan(a)], b[~np.isnan(b)]
        ks = stats.ks_2samp(ok_a, ok_b).statistic if len(ok_a) > 30 and len(ok_b) > 30 else np.nan
        rows.append({"feature": c, "psi": psi(a, b), "ks_stat": ks,
                     "missing_train": np.isnan(a).mean(), "missing_test": np.isnan(b).mean()})
    out = pd.DataFrame(rows).sort_values("psi", ascending=False)
    out.to_csv(cfg["paths"]["tables_dir"] / "drift_train_vs_test.csv", index=False)

    top = out.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(top.feature, top.psi, color=np.where(top.psi > cfg["drift"]["psi_alert"], "#d62728", "#f0a030"))
    ax.axvline(cfg["drift"]["psi_alert"], color="black", ls="--", lw=1)
    ax.set(xlabel="PSI (train period vs test period)", title="Which inputs changed the most?")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg["paths"]["figures_dir"] / "drift_psi.png", dpi=140)
    plt.close(fig)

    alerts = int((out.psi > cfg["drift"]["psi_alert"]).sum())
    log.info("drift: %d of %d features above PSI %.2f; worst = %s", alerts, len(out),
             cfg["drift"]["psi_alert"], out.iloc[0]["feature"])
    print("\nTOP 10 DRIFTING FEATURES\n", out.head(10).round(3).to_string(index=False), "\n")
    return out
