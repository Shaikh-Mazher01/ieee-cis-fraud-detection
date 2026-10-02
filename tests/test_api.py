import pytest
import xgboost as xgb
from fastapi.testclient import TestClient

from fraud.api import app
from fraud.features import FeatureBuilder
from fraud.splits import time_split


@pytest.fixture(scope="module")
def client(merged, cfg):
    idx = time_split(merged, cfg)
    train = merged.iloc[idx["train"]]
    fb = FeatureBuilder(cfg)
    X = fb.fit_transform(train)
    model = xgb.XGBClassifier(n_estimators=15, max_depth=3, learning_rate=0.3, tree_method="hist", verbosity=0)
    model.fit(X, train["isFraud"])
    app.state.bundle = {"builder": fb, "model": model, "model_name": "tiny", "threshold": 0.3,
                        "feature_names": fb.feature_names_, "profile": "synthetic"}
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_probability_and_decision(client):
    r = client.post("/predict", json={"TransactionAmt": 120.0, "ProductCD": "C", "P_emaildomain": "anonymous.com",
                                      "TransactionDT": 9_000_000})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["decision"] in {"APPROVE", "REVIEW"}
    assert isinstance(body["reasons"], list)


def test_predict_batch(client):
    txs = [{"TransactionAmt": 10 + i} for i in range(5)]
    r = client.post("/predict-batch", json={"transactions": txs})
    assert r.status_code == 200 and len(r.json()) == 5


def test_bad_amount_is_rejected(client):
    assert client.post("/predict", json={"TransactionAmt": -5}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
