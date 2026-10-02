from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "outputs" / "real" / "models" / "bundle.joblib"

app = FastAPI(
    title="Fraud scoring API",
    version="1.0.0",
    description="Send a transaction, get a fraud probability, a decision and the top reasons.",
)


class Transaction(BaseModel):
    """Only TransactionAmt is required. Any other IEEE-CIS column (card1, addr1, C1, D1, V12, id_31 ...)
    may be added; missing ones are treated as unknown, exactly like in the training data."""

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"example": {
            "TransactionDT": 9000000, "TransactionAmt": 117.5, "ProductCD": "C", "card1": 9500, "card4": "visa",
            "card6": "credit", "addr1": 315, "P_emaildomain": "anonymous.com", "C1": 6, "D1": 0,
            "DeviceType": "mobile", "id_31": "chrome 63.0 for android"}},
    )
    TransactionAmt: float = Field(gt=0, description="Amount in USD")
    TransactionDT: float | None = Field(default=None, ge=0, description="Seconds since a reference date")


class Reason(BaseModel):
    feature: str
    value: float | None
    impact: float


class Score(BaseModel):
    fraud_probability: float
    decision: str
    threshold: float
    reasons: list[Reason]


class Batch(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


def get_bundle() -> dict:
    if getattr(app.state, "bundle", None) is None:
        path = Path(os.environ.get("FRAUD_BUNDLE", DEFAULT_BUNDLE))
        if not path.exists():
            raise HTTPException(503, f"Model file not found at {path}. Train first: python -m fraud all")
        app.state.bundle = joblib.load(path)
    return app.state.bundle


def _reasons(bundle: dict, X: pd.DataFrame, top: int = 3) -> list[list[Reason]]:
    model = bundle["model"]
    if not hasattr(model, "get_booster"):
        return [[] for _ in range(len(X))]
    import xgboost as xgb

    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    out = []
    for i in range(len(X)):
        order = contrib[i].argsort()[::-1][:top]
        out.append([
            Reason(feature=X.columns[j], value=None if pd.isna(X.iloc[i, j]) else float(X.iloc[i, j]),
                   impact=round(float(contrib[i, j]), 4))
            for j in order if contrib[i, j] > 0
        ])
    return out


def score_frame(bundle: dict, raw: pd.DataFrame) -> list[Score]:
    X = bundle["builder"].transform(raw)
    p = bundle["model"].predict_proba(X)[:, 1]
    thr = float(bundle["threshold"])
    reasons = _reasons(bundle, X)
    return [Score(fraud_probability=round(float(pi), 6), decision="REVIEW" if pi >= thr else "APPROVE",
                  threshold=thr, reasons=r) for pi, r in zip(p, reasons)]


@app.get("/health")
def health():
    try:
        b = get_bundle()
    except HTTPException as e:
        return {"status": "no_model", "detail": e.detail}
    return {"status": "ok", "model": b["model_name"], "threshold": b["threshold"]}


@app.get("/model-info")
def model_info():
    b = get_bundle()
    return {"model": b["model_name"], "threshold": b["threshold"], "n_features": len(b["feature_names"]),
            "trained_on": b.get("profile", "unknown"),
            "warning": "Trained on SYNTHETIC data, demo only." if b.get("profile") == "synthetic" else None}


@app.post("/predict", response_model=Score)
def predict(tx: Transaction):
    return score_frame(get_bundle(), pd.DataFrame([tx.model_dump()]))[0]


@app.post("/predict-batch", response_model=list[Score])
def predict_batch(batch: Batch):
    return score_frame(get_bundle(), pd.DataFrame([t.model_dump() for t in batch.transactions]))
