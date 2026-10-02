from __future__ import annotations

import logging
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_merged
from .features import FeatureBuilder
from .splits import time_split
from .tracking import Tracker

log = logging.getLogger(__name__)


def _xgb(cfg, params, spw, fast):
    base = dict(cfg["model"]["xgb"])
    if fast:
        base.update(cfg["model"]["fast"])
    base.update(params)
    return xgb.XGBClassifier(
        **base,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        early_stopping_rounds=cfg["model"]["early_stopping_rounds"],
        n_jobs=-1,
        random_state=cfg["seed"],
        verbosity=0,
    )


def _scores(y, p):
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p)}


def train(cfg: dict, fast: bool = False, n_trials: int | None = None) -> pd.DataFrame:
    seed = cfg["seed"]
    target, id_col = cfg["data"]["target"], cfg["data"]["id_col"]
    tracker = Tracker(cfg)

    log.info("loading merged data")
    df = load_merged(cfg)
    idx = time_split(df, cfg)
    ids = df[id_col].to_numpy()
    y_all = df[target].to_numpy()
    t_col = df[cfg["data"]["time_col"]].to_numpy()
    for name, ix in idx.items():
        days = t_col[ix] / 86400
        log.info("%-5s rows=%-8d fraud=%.2f%%  days %.0f -> %.0f", name, len(ix), 100 * y_all[ix].mean(),
                 days.min(), days.max())

    # ---- features: learn on TRAIN only
    t0 = time.time()
    fb = FeatureBuilder(cfg)
    X = {"train": fb.fit_transform(df.iloc[idx["train"]])}
    X["valid"] = fb.transform(df.iloc[idx["valid"]])
    X["test"] = fb.transform(df.iloc[idx["test"]])
    del df
    y = {k: y_all[ix] for k, ix in idx.items()}
    log.info("features: %d columns kept, %d dropped, built in %.0fs",
             len(fb.feature_names_), len(fb.dropped_cols_), time.time() - t0)

    split_ids = pd.concat([pd.DataFrame({id_col: ids[ix], "split": k, target: y[k]}) for k, ix in idx.items()])
    split_ids.to_parquet(cfg["paths"]["tables_dir"] / "split_ids.parquet", index=False)

    n_pos, n_neg = int(y["train"].sum()), int((y["train"] == 0).sum())
    spw = n_neg / max(n_pos, 1)
    log.info("train positives=%d negatives=%d -> scale_pos_weight=%.1f", n_pos, n_neg, spw)

    models: dict[str, object] = {}
    probs: dict[str, dict[str, np.ndarray]] = {}
    rows = []

    def register(name, model, params, seconds, extra=None):
        probs[name] = {s: model.predict_proba(X[s])[:, 1] for s in ("valid", "test")}
        m = _scores(y["valid"], probs[name]["valid"])
        m["fit_seconds"] = seconds
        if extra:
            m.update(extra)
        models[name] = model
        rows.append({"model": name, **m})
        tracker.log_run(name, params, m, tags={"profile": cfg["profile"], "split": "time"})
        log.info("%-14s valid ROC-AUC %.4f | PR-AUC %.4f | %.0fs", name, m["roc_auc"], m["pr_auc"], seconds)

    # 1. dummy: always predicts 'not fraud'
    base_rate = y["train"].mean()
    rows.append({"model": "dummy_majority", "roc_auc": 0.5, "pr_auc": float(y["valid"].mean()), "fit_seconds": 0.0})
    tracker.log_run("dummy_majority", {}, {"roc_auc": 0.5, "pr_auc": float(y["valid"].mean())})

    # 2. logistic regression baseline (scaled + class weighted), on a subsample for speed
    t0 = time.time()
    lr_rows = min(cfg["model"]["logreg"]["max_train_rows"], len(X["train"]))
    sel = np.random.default_rng(seed).choice(len(X["train"]), lr_rows, replace=False)
    lr = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=cfg["model"]["logreg"]["C"], class_weight="balanced", max_iter=200,
                                   random_state=seed)),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr.fit(X["train"].iloc[sel], y["train"][sel])
        register("logreg", lr, {"C": cfg["model"]["logreg"]["C"], "rows": lr_rows}, time.time() - t0)

    # 3 and 4. XGBoost without and with imbalance handling
    for name, w in (("xgb_plain", 1.0), ("xgb_spw", spw)):
        t0 = time.time()
        clf = _xgb(cfg, {}, w, fast)
        clf.fit(X["train"], y["train"], eval_set=[(X["valid"], y["valid"])], verbose=False)
        params = {**clf.get_params(), "n_features": X["train"].shape[1]}
        keep = {k: params[k] for k in ("max_depth", "learning_rate", "subsample", "colsample_bytree",
                                       "scale_pos_weight", "n_features")}
        register(name, clf, keep, time.time() - t0, {"best_iteration": int(clf.best_iteration)})

    # 5. random search (Module 8, section 6). Uses the validation period, never the test period.
    trials = cfg["model"]["tuning"]["n_trials"] if n_trials is None else n_trials
    if fast and n_trials is None:
        trials = 0
    best_trial = None
    if trials > 0:
        space = cfg["model"]["tuning"]["space"]
        for i, params in enumerate(ParameterSampler(space, n_iter=trials, random_state=seed)):
            t0 = time.time()
            clf = _xgb(cfg, params, spw, fast)
            clf.fit(X["train"], y["train"], eval_set=[(X["valid"], y["valid"])], verbose=False)
            p = clf.predict_proba(X["valid"])[:, 1]
            m = _scores(y["valid"], p)
            tracker.log_run(f"tune_trial_{i}", {**params, "scale_pos_weight": spw},
                            {**m, "best_iteration": int(clf.best_iteration)}, tags={"stage": "tuning"})
            log.info("trial %d/%d %s -> PR-AUC %.4f (%.0fs)", i + 1, trials, params, m["pr_auc"], time.time() - t0)
            if best_trial is None or m["pr_auc"] > best_trial[0]:
                best_trial = (m["pr_auc"], params, clf, time.time() - t0)
            else:
                del clf
        _, params, clf, secs = best_trial
        register("xgb_tuned", clf, {**params, "scale_pos_weight": spw}, secs, {"best_iteration": int(clf.best_iteration)})

    board = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    board.to_csv(cfg["paths"]["tables_dir"] / "leaderboard_valid.csv", index=False)
    print("\nVALIDATION LEADERBOARD (sorted by PR-AUC)\n", board.round(4).to_string(index=False), "\n")

    winner = board[board.model != "dummy_majority"].iloc[0]["model"]
    log.info("winner on validation: %s", winner)

    pred_frames = []
    for s in ("valid", "test"):
        f = pd.DataFrame({id_col: ids[idx[s]], "split": s, target: y[s]})
        for name in probs:
            f[f"prob_{name}"] = probs[name][s].astype("float32")
        pred_frames.append(f)
    pd.concat(pred_frames).to_parquet(cfg["paths"]["tables_dir"] / "predictions.parquet", index=False)

    bundle = {
        "builder": fb,
        "model": models[winner],
        "model_name": winner,
        "feature_names": fb.feature_names_,
        "threshold": 0.5,  # replaced by the cost-based one in the 'evaluate' step
        "scale_pos_weight": spw,
        "train_fraud_rate": float(base_rate),
        "profile": cfg["profile"],
    }
    joblib.dump(bundle, cfg["paths"]["bundle"], compress=3)
    log.info("saved %s", cfg["paths"]["bundle"])
    return board
