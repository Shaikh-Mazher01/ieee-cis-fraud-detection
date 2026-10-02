import numpy as np
import pandas as pd
import pytest

from fraud.evaluate import best_threshold, capture_at_top, cost_table
from fraud.features import FeatureBuilder
from fraud.splits import time_split
from fraud.sql_analysis import split_queries


def test_synthetic_matches_real_schema(merged):
    assert merged.shape[1] == 435            # 394 transaction + 40 identity (id merged) + has_identity
    assert 0.02 < merged["isFraud"].mean() < 0.05
    assert merged["TransactionID"].is_unique


def test_time_split_has_no_overlap_in_time(merged, cfg):
    idx = time_split(merged, cfg)
    t = merged["TransactionDT"].to_numpy()
    assert t[idx["train"]].max() <= t[idx["valid"]].min()
    assert t[idx["valid"]].max() <= t[idx["test"]].min()
    assert sum(len(v) for v in idx.values()) == len(merged)


def test_feature_builder_learns_from_train_only(merged, cfg):
    idx = time_split(merged, cfg)
    train, test = merged.iloc[idx["train"]], merged.iloc[idx["test"]]
    fb = FeatureBuilder(cfg)
    Xtr = fb.fit_transform(train)
    Xte = fb.transform(test)
    assert list(Xtr.columns) == list(Xte.columns)                  # same columns offline and online
    assert Xtr.dtypes.eq("float32").all()
    # a card never seen in training must get frequency 0, not a value learned from the test period
    unseen = test[~test["card1"].isin(train["card1"])]
    if len(unseen):
        assert (Xte.loc[unseen.index, "freq_card1"] == 0).all()
    assert not np.isinf(Xte.to_numpy()).any()


def test_feature_builder_handles_a_sparse_api_row(merged, cfg):
    fb = FeatureBuilder(cfg).fit(merged.iloc[:4000])
    row = pd.DataFrame([{"TransactionAmt": 25.0, "ProductCD": "W"}])  # nearly everything missing
    X = fb.transform(row)
    assert X.shape == (1, len(fb.feature_names_))


def test_cost_table_math():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    amt = np.array([100, 50, 10, 10, 10])
    t = cost_table(y, p, amt, review_cost=5, thresholds=[0.5]).iloc[0]
    assert (t.tp, t.fp, t.fn) == (1, 1, 1)
    assert t.cost == 50 + 5 * 2                                      # missed $50 + two reviews
    assert best_threshold(cost_table(y, p, amt, 5)) <= 0.9


def test_capture_at_top():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.linspace(1, 0, 10)
    assert capture_at_top(y, p, fractions=(0.1,))["top_10pct"] == 1.0


def test_sql_file_parses_into_named_queries(cfg):
    q = split_queries((cfg["paths"]["sql_dir"] / "fraud_analysis.sql").read_text())
    assert len(q) >= 8 and "01_overview" in q and all("SELECT" in v for v in q.values())
