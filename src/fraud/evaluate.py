from __future__ import annotations

import json
import logging

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (average_precision_score, precision_recall_curve, roc_auc_score,  # noqa: E402
                             roc_curve)

from .data import load_merged  # noqa: E402

log = logging.getLogger(__name__)
GROUP_COLS = ["ProductCD", "card4", "card6", "DeviceType", "has_identity"]


# --------------------------------------------------------------------------- business cost
def cost_table(y, p, amount, review_cost, loss_mult=1.0, thresholds=None) -> pd.DataFrame:
    """For every threshold: how many we flag, what we catch, and the total money cost.

    cost = (money lost on frauds we MISSED) + (review cost x every transaction we flagged)
    """
    y = np.asarray(y).astype(bool)
    p = np.asarray(p)
    amount = np.asarray(amount, dtype="float64")
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)
    rows = []
    for t in thresholds:
        flag = p >= t
        tp, fp = int((flag & y).sum()), int((flag & ~y).sum())
        fn = int((~flag & y).sum())
        missed_loss = float(amount[~flag & y].sum()) * loss_mult
        cost = missed_loss + review_cost * int(flag.sum())
        rows.append({
            "threshold": float(t), "flagged": int(flag.sum()), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "missed_loss": missed_loss, "review_spend": review_cost * int(flag.sum()), "cost": cost,
        })
    return pd.DataFrame(rows)


def best_threshold(table: pd.DataFrame) -> float:
    return float(table.loc[table["cost"].idxmin(), "threshold"])


def baselines(y, amount, review_cost, loss_mult=1.0) -> dict:
    y = np.asarray(y).astype(bool)
    amount = np.asarray(amount, dtype="float64")
    return {
        "approve_everything": float(amount[y].sum()) * loss_mult,   # no model: eat every fraud
        "review_everything": review_cost * len(y),                  # perfect safety, impossible workload
    }


def capture_at_top(y, p, fractions=(0.01, 0.05, 0.10)) -> dict:
    """If the review team can only look at the riskiest X% of traffic, how much fraud do they see?"""
    y = np.asarray(y)
    order = np.argsort(-np.asarray(p))
    out = {}
    for f in fractions:
        k = max(int(len(y) * f), 1)
        out[f"top_{int(f * 100)}pct"] = float(y[order[:k]].sum() / max(y.sum(), 1))
    return out


# --------------------------------------------------------------------------- plots
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_curves(y, probs: dict, out):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, p in probs.items():
        pr, rc, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(rc, pr, label=f"{name} (AP {average_precision_score(y, p):.3f})")
        ax[1].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    ax[0].axhline(np.mean(y), color="grey", ls="--", lw=1, label="random")
    ax[0].set(xlabel="Recall (fraud caught)", ylabel="Precision (flags that were real)",
              title="Precision-recall: the honest view for rare fraud")
    ax[1].plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
    ax[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (looks rosy when fraud is rare)")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    _save(fig, out)


def plot_cost(table, thr, out, title):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(table.threshold, table.cost / 1e3, color="#1f77b4", label="total cost")
    ax.plot(table.threshold, table.missed_loss / 1e3, color="#d62728", ls="--", label="missed fraud $")
    ax.plot(table.threshold, table.review_spend / 1e3, color="#2ca02c", ls="--", label="review $")
    ax.axvline(thr, color="black", lw=1)
    ax.annotate(f"chosen threshold {thr:.2f}", (thr, table.cost.min() / 1e3), xytext=(10, 25),
                textcoords="offset points", arrowprops=dict(arrowstyle="->"))
    ax.set(xlabel="Threshold", ylabel="Cost (USD thousand)", title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out)


def plot_confusion(tp, fp, fn, tn, out, thr):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    m = np.array([[tn, fp], [fn, tp]])
    ax.imshow(np.log10(m + 1), cmap="Blues")
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=12, color="black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["approve", "flag"], yticklabels=["legit", "fraud"],
           xlabel="Model decision", ylabel="Truth", title=f"Test confusion matrix @ {thr:.2f}")
    _save(fig, out)


# --------------------------------------------------------------------------- group + time views
def group_metrics(meta: pd.DataFrame, p, thr, min_pos=20) -> pd.DataFrame:
    d = meta.copy()
    d["p"] = p
    d["flag"] = d["p"] >= thr
    rows = []
    for col in GROUP_COLS:
        if col not in d:
            continue
        for val, g in d.groupby(d[col].astype(object).fillna("missing"), observed=True):
            pos = int(g.y.sum())
            rows.append({
                "group": col, "value": val, "n": len(g), "fraud_rate": g.y.mean(), "flag_rate": g.flag.mean(),
                "recall": (g.flag & (g.y == 1)).sum() / pos if pos else np.nan,
                "precision": (g.flag & (g.y == 1)).sum() / max(g.flag.sum(), 1),
                "pr_auc": average_precision_score(g.y, g.p) if pos >= min_pos else np.nan,
            })
    return pd.DataFrame(rows)


def plot_groups(gm: pd.DataFrame, out):
    g = gm[(gm.n >= 500)].copy()
    g["label"] = g.group + " = " + g.value.astype(str)
    g = g.sort_values("recall")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(g))))
    ax.barh(g.label, g.recall, color="#4c72b0")
    ax.set(xlabel="Recall at the chosen threshold", title="Fraud caught by segment: where is the model weaker?")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, out)


def performance_over_time(meta, p, day_col="day", bucket=14):
    d = pd.DataFrame({"day": meta[day_col].to_numpy(), "y": meta["y"].to_numpy(), "p": p})
    d["bucket"] = (d["day"] // bucket) * bucket
    rows = []
    for b, g in d.groupby("bucket"):
        if g.y.sum() >= 10:
            rows.append({"day_from": int(b), "n": len(g), "fraud_rate": g.y.mean(), "pr_auc": average_precision_score(g.y, g.p)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- SHAP
def shap_report(cfg, bundle, ids) -> pd.DataFrame | None:
    model = bundle["model"]
    if not hasattr(model, "get_booster"):
        log.info("SHAP skipped: best model is not tree based")
        return None
    import shap

    n = min(cfg["shap"]["sample_size"], len(ids))
    pick = np.random.default_rng(cfg["seed"]).choice(ids, n, replace=False)
    sample = load_merged(cfg, filters=[(cfg["data"]["id_col"], "in", pick.tolist())])
    X = bundle["builder"].transform(sample)
    
    # Fix for XGBoost 2.x / SHAP base_score compatibility string formatting issue
    if hasattr(model, "base_score") and not isinstance(model.base_score, (int, float)):
        try:
            val = model.base_score
            if isinstance(val, (list, tuple, np.ndarray)):
                val = val[0]
            model.base_score = float(str(val).strip("[]"))
        except Exception:
            pass

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    imp = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(0)}) \
        .sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(cfg["paths"]["tables_dir"] / "shap_importance.csv", index=False)
    plt.figure()
    shap.summary_plot(sv, X, max_display=20, show=False)
    plt.title("What pushes a transaction toward 'fraud' (SHAP, test sample)")
    plt.tight_layout()
    plt.savefig(cfg["paths"]["figures_dir"] / "shap_beeswarm.png", dpi=140, bbox_inches="tight")
    plt.close()
    return imp


# --------------------------------------------------------------------------- main step
def evaluate(cfg: dict) -> dict:
    tables, figs = cfg["paths"]["tables_dir"], cfg["paths"]["figures_dir"]
    id_col, target = cfg["data"]["id_col"], cfg["data"]["target"]
    rc, lm = cfg["business"]["review_cost"], cfg["business"]["fraud_loss_multiplier"]

    bundle = joblib.load(cfg["paths"]["bundle"])
    best = bundle["model_name"]
    preds = pd.read_parquet(tables / "predictions.parquet")
    meta_cols = [id_col, "TransactionDT", "TransactionAmt", *GROUP_COLS]
    import pyarrow.parquet as pq
    present = set(pq.read_schema(cfg["paths"]["merged"]).names)
    meta = load_merged(cfg, columns=[c for c in meta_cols if c in present])
    preds = preds.merge(meta, on=id_col, how="left")
    preds["day"] = (preds["TransactionDT"] // 86400).astype(int)
    preds = preds.rename(columns={target: "y"})
    va, te = preds[preds.split == "valid"].reset_index(drop=True), preds[preds.split == "test"].reset_index(drop=True)
    model_names = [c[5:] for c in preds.columns if c.startswith("prob_")]

    # ---- threshold per model from validation cost; test numbers computed once
    rows, thresholds = [], {}
    for name in model_names:
        ct = cost_table(va.y, va[f"prob_{name}"], va.TransactionAmt, rc, lm)
        thr = best_threshold(ct)
        thresholds[name] = thr
        pt = te[f"prob_{name}"]
        tt = cost_table(te.y, pt, te.TransactionAmt, rc, lm, thresholds=[thr]).iloc[0]
        base = baselines(te.y, te.TransactionAmt, rc, lm)
        rows.append({
            "model": name, "threshold": thr,
            "test_roc_auc": roc_auc_score(te.y, pt), "test_pr_auc": average_precision_score(te.y, pt),
            "precision": tt.precision, "recall": tt.recall,
            "f1": 2 * tt.precision * tt.recall / max(tt.precision + tt.recall, 1e-9),
            "flag_rate": tt.flagged / len(te), "cost": tt.cost,
            "saving_vs_no_model": base["approve_everything"] - tt.cost,
        })
    comp = pd.DataFrame(rows).sort_values("test_pr_auc", ascending=False)
    comp.to_csv(tables / "model_comparison_test.csv", index=False)
    print("\nTEST RESULTS (threshold from validation cost)\n", comp.round(4).to_string(index=False), "\n")

    # ---- accuracy lies: dummy model
    dummy_acc = 1 - te.y.mean()

    # ---- the chosen model in detail
    thr = thresholds[best]
    bundle["threshold"] = thr
    joblib.dump(bundle, cfg["paths"]["bundle"], compress=3)
    pbest = te[f"prob_{best}"].to_numpy()
    ct_valid = cost_table(va.y, va[f"prob_{best}"], va.TransactionAmt, rc, lm)
    ct_valid.to_csv(tables / "threshold_cost_valid.csv", index=False)
    plot_cost(ct_valid, thr, figs / "threshold_cost.png", "Pick the threshold with money, not 0.5 (validation)")
    r = cost_table(te.y, pbest, te.TransactionAmt, rc, lm, thresholds=[thr]).iloc[0]
    tn = int(len(te) - r.tp - r.fp - r.fn)
    plot_confusion(int(r.tp), int(r.fp), int(r.fn), tn, figs / "confusion_matrix.png", thr)
    plot_curves(te.y, {n: te[f"prob_{n}"] for n in model_names}, figs / "pr_roc_curves.png")
    base = baselines(te.y, te.TransactionAmt, rc, lm)
    cap = capture_at_top(te.y, pbest)

    gm = group_metrics(te.rename(columns={"y": "y"}), pbest, thr)
    gm.to_csv(tables / "group_metrics_test.csv", index=False)
    plot_groups(gm, figs / "group_recall.png")

    pot = performance_over_time(pd.concat([va, te]), np.concatenate([va[f"prob_{best}"], pbest]))
    pot.to_csv(tables / "performance_over_time.csv", index=False)
    if len(pot) > 1:
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.plot(pot.day_from, pot.pr_auc, marker="o")
        ax.set(xlabel="Day (start of 14-day window)", ylabel="PR-AUC",
               title="Does the model hold up as time passes?")
        ax.grid(alpha=0.3)
        _save(fig, figs / "performance_over_time.png")

    imp = shap_report(cfg, bundle, te[id_col].to_numpy())

    metrics = {
        "profile": cfg["profile"], "best_model": best, "threshold": thr,
        "business": cfg["business"], "test_rows": int(len(te)), "test_fraud_rate": float(te.y.mean()),
        "test_frauds": int(te.y.sum()), "test_fraud_dollars": float(te.TransactionAmt[te.y == 1].sum()),
        "dummy_accuracy": float(dummy_acc), "roc_auc": float(roc_auc_score(te.y, pbest)),
        "pr_auc": float(average_precision_score(te.y, pbest)),
        "precision": float(r.precision), "recall": float(r.recall), "flagged": int(r.flagged),
        "flag_rate": float(r.flagged / len(te)), "tp": int(r.tp), "fp": int(r.fp), "fn": int(r.fn), "tn": tn,
        "cost_model": float(r.cost), "cost_approve_all": base["approve_everything"],
        "cost_review_all": base["review_everything"],
        "saving": float(base["approve_everything"] - r.cost),
        "missed_loss": float(r.missed_loss), "review_spend": float(r.review_spend),
        "capture": cap,
        "top_features": imp.head(10).feature.tolist() if imp is not None else [],
        "weakest_segments": gm[gm.n >= 500].nsmallest(3, "recall")[["group", "value", "recall"]].to_dict("records"),
    }
    (cfg["paths"]["reports_dir"] / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    log.info("saved metrics.json | chosen threshold %.2f | est. saving $%.0f on the test period", thr, metrics["saving"])
    return metrics
