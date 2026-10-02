from __future__ import annotations

import logging

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import load_merged  # noqa: E402

log = logging.getLogger(__name__)
BLUE, RED, GREY = "#4c72b0", "#d62728", "#8c8c8c"


def _save(fig, cfg, name):
    fig.tight_layout()
    fig.savefig(cfg["paths"]["figures_dir"] / f"eda_{name}.png", dpi=140)
    plt.close(fig)


def run_eda(cfg: dict) -> None:
    target = cfg["data"]["target"]
    df = load_merged(cfg)
    n, rate = len(df), df[target].mean()
    notes = [f"rows={n:,} columns={df.shape[1]} fraud_rate={rate:.3%}"]

    # 1. class imbalance
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["legit", "fraud"], [(1 - rate) * n, rate * n], color=[BLUE, RED])
    ax.set_yscale("log")
    ax.set(ylabel="Transactions (log scale)", title=f"Only {rate:.1%} of transactions are fraud: accuracy will lie")
    _save(fig, cfg, "01_class_balance")

    # 2. product
    g = df.groupby("ProductCD", observed=True)[target].agg(["mean", "size"]).sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(g.index.astype(str), g["mean"] * 100, color=[RED if v == g["mean"].max() else BLUE for v in g["mean"]])
    ax.axhline(rate * 100, color=GREY, ls="--")
    ax.set(ylabel="Fraud rate (%)", title=f"Product {g.index[0]} is {g['mean'].iloc[0] / rate:.1f}x riskier than average")
    _save(fig, cfg, "02_product")
    notes.append(f"riskiest product {g.index[0]} {g['mean'].iloc[0]:.2%}")

    # 3. hour of day
    hour = ((df["TransactionDT"] // 3600) % 24).astype(int)
    h = df.groupby(hour)[target].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h.index, h.values, marker="o", color=BLUE)
    ax.axhline(rate * 100, color=GREY, ls="--")
    ax.set(xlabel="Hour of day", ylabel="Fraud rate (%)",
           title=f"Fraud peaks around hour {int(h.idxmax())} ({h.max():.1f}% vs {rate * 100:.1f}% average)")
    ax.grid(alpha=0.3)
    _save(fig, cfg, "03_hour")

    # 4. amount by class (log scale because amounts are heavy tailed)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(-0.5, 4, 60)
    for val, col, lab in ((0, BLUE, "legit"), (1, RED, "fraud")):
        ax.hist(df.loc[df[target] == val, "TransactionAmt"], bins=bins, alpha=0.55, color=col, label=lab, density=True)
    ax.set_xscale("log")
    med0, med1 = df.groupby(target)["TransactionAmt"].median().tolist()
    ax.set(xlabel="Transaction amount (log scale)", ylabel="Share", title=f"Median amount: legit ${med0:.0f}, fraud ${med1:.0f}")
    ax.legend()
    _save(fig, cfg, "04_amount")

    # 5. identity coverage
    gi = df.groupby("has_identity")[target].mean() * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["no identity data", "has identity data"], gi.values, color=[BLUE, RED])
    ax.set(ylabel="Fraud rate (%)", title=f"Having identity data: fraud {gi.get(1, np.nan) / gi.get(0, np.nan):.1f}x higher")
    _save(fig, cfg, "05_identity")

    # 6. fraud rate over time
    day = (df["TransactionDT"] // 86400).astype(int)
    d = df.groupby(day)[target].agg(["mean", "size"])
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(d.index, d["size"], color="#cfd8e6", label="transactions")
    ax1.set(xlabel="Day", ylabel="Daily transactions")
    ax2 = ax1.twinx()
    ax2.plot(d.index, d["mean"].rolling(7, min_periods=1).mean() * 100, color=RED, label="fraud rate, 7-day avg")
    ax2.set_ylabel("Fraud rate (%)")
    ax1.set_title("Fraud rate moves over time: we must split by time, not randomly")
    _save(fig, cfg, "06_time")

    # 7. missingness
    miss = df.isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(miss.values * 100, bins=20, color=BLUE)
    ax.set(xlabel="% of values missing in a column", ylabel="Number of columns",
           title=f"{int((miss > 0.5).sum())} of {len(miss)} columns are more than half empty")
    _save(fig, cfg, "07_missing")
    miss.head(40).rename("missing_share").to_csv(cfg["paths"]["tables_dir"] / "eda_missing_top40.csv")

    # 8. which numeric columns relate to fraud? (on a sample to save time)
    samp = df.sample(min(200_000, n), random_state=cfg["seed"])
    num = samp.select_dtypes("number").drop(columns=[target, cfg["data"]["id_col"]], errors="ignore")
    with np.errstate(invalid="ignore", divide="ignore"):  # constant columns have no correlation
        corr = num.corrwith(samp[target]).dropna()
    top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(top.index, top.values, color=[RED if v > 0 else BLUE for v in top.values])
    ax.set(xlabel="Correlation with isFraud", title=f"{top.index[-1]} has the strongest linear link to fraud")
    _save(fig, cfg, "08_correlation")
    corr.rename("corr_with_fraud").sort_values(key=abs, ascending=False).head(50).to_csv(
        cfg["paths"]["tables_dir"] / "eda_correlation_top50.csv")

    (cfg["paths"]["reports_dir"] / "eda_notes.txt").write_text("\n".join(notes))
    log.info("EDA done: 8 figures in %s", cfg["paths"]["figures_dir"])
