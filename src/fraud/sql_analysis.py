from __future__ import annotations

import logging
import re
import sqlite3

import pandas as pd

from .data import load_merged

log = logging.getLogger(__name__)
SQL_COLS = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD", "card4", "card6",
            "P_emaildomain", "DeviceType", "has_identity"]


def build_db(cfg: dict) -> sqlite3.Connection:
    """SQLite keeps one file on disk; 590k rows x 12 columns is tiny for it."""
    import pyarrow.parquet as pq

    present = set(pq.read_schema(cfg["paths"]["merged"]).names)
    df = load_merged(cfg, columns=[c for c in SQL_COLS if c in present])
    for c in df.select_dtypes("category").columns:
        df[c] = df[c].astype(object)
    df["day"] = (df["TransactionDT"] // 86400).astype(int)
    df["hour"] = ((df["TransactionDT"] // 3600) % 24).astype(int)
    db_path = cfg["paths"]["interim_dir"] / "fraud.db"
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    df.to_sql("txn", con, index=False, chunksize=100_000)
    # indexes (Module 3, section 7): speed up the GROUP BY / WHERE columns we use most
    for col in ("ProductCD", "day", "P_emaildomain"):
        con.execute(f"CREATE INDEX idx_txn_{col} ON txn({col})")
    con.commit()
    log.info("SQLite table txn: %d rows -> %s", len(df), db_path)
    return con


def split_queries(sql_text: str) -> dict[str, str]:
    blocks = re.split(r"^-- name:\s*(\S+)\s*$", sql_text, flags=re.M)
    return {blocks[i]: blocks[i + 1].strip() for i in range(1, len(blocks), 2)}


def run_sql(cfg: dict) -> dict[str, pd.DataFrame]:
    con = build_db(cfg)
    queries = split_queries((cfg["paths"]["sql_dir"] / "fraud_analysis.sql").read_text(encoding="utf-8"))
    out_dir = cfg["paths"]["tables_dir"] / "sql"
    out_dir.mkdir(exist_ok=True)
    results = {}
    for name, q in queries.items():
        res = pd.read_sql_query(q, con)
        res.to_csv(out_dir / f"{name}.csv", index=False)
        results[name] = res
        print(f"\n--- {name} ---\n{res.head(12).to_string(index=False)}")
    con.close()
    return results
