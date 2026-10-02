from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from .schema import CAT_COLS, INT_COLS, normalise_name

log = logging.getLogger(__name__)


def _dtype_map(path, nrows=None) -> dict:
    """Read only the header, then decide a compact dtype for each column."""
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {}
    for orig in header:
        name = normalise_name(orig)
        if name in INT_COLS:
            dtypes[orig] = INT_COLS[name]
        elif name in CAT_COLS:
            dtypes[orig] = "category"
        else:
            dtypes[orig] = "float32"
    return dtypes


def read_csv_compact(path, nrows: int | None = None) -> pd.DataFrame:
    dtypes = _dtype_map(path)
    try:
        df = pd.read_csv(path, dtype=dtypes, nrows=nrows)
    except (ValueError, TypeError) as err:  # a column was not the type we expected
        log.warning("Fast dtype read failed (%s). Falling back to type inference.", err)
        df = pd.read_csv(path, nrows=nrows)
        for c in df.select_dtypes("float64").columns:
            df[c] = df[c].astype("float32")
        for c in df.select_dtypes("object").columns:
            df[c] = df[c].astype("category")
    df.columns = [normalise_name(c) for c in df.columns]
    return df


def build_merged(cfg: dict, nrows: int | None = None) -> pd.DataFrame:
    p = cfg["paths"]
    d = cfg["data"]
    t_path = p["raw_dir"] / d["transaction_file"]
    i_path = p["raw_dir"] / d["identity_file"]
    for f in (t_path, i_path):
        if not f.exists():
            raise FileNotFoundError(
                f"{f} not found.\n"
                "Real data: run scripts/download_data.sh (see README, step 2).\n"
                "No Kaggle account yet? Use fake data: python -m fraud synth --rows 100000, "
                "then add --profile synthetic to every command."
            )
    t0 = time.time()
    trans = read_csv_compact(t_path, nrows=nrows)
    ident = read_csv_compact(i_path)
    log.info("transactions %s | identity %s | read in %.0fs", trans.shape, ident.shape, time.time() - t0)

    id_col = d["id_col"]
    ident = ident.drop_duplicates(id_col)
    # LEFT join: only ~24% of transactions have an identity row. The rest must stay.
    df = trans.merge(ident, on=id_col, how="left")
    flag = df[id_col].isin(ident[id_col]).astype("int8").rename("has_identity")
    df = pd.concat([df, flag], axis=1)
    df = df.sort_values(d["time_col"]).reset_index(drop=True)
    return df


def prepare(cfg: dict, nrows: int | None = None) -> pd.DataFrame:
    df = build_merged(cfg, nrows)
    target = cfg["data"]["target"]
    mem = df.memory_usage(deep=True).sum() / 1e9
    log.info("merged %s | %.2f GB in memory | fraud rate %.3f%%", df.shape, mem, 100 * df[target].mean())
    # sanity checks (Module 6: never trust data blindly)
    assert df[cfg["data"]["id_col"]].is_unique, "TransactionID must be unique"
    assert set(df[target].unique()) <= {0, 1}, "target must be 0/1"
    df.to_parquet(cfg["paths"]["merged"], index=False)
    log.info("saved %s", cfg["paths"]["merged"])
    return df


def load_merged(cfg: dict, columns=None, filters=None) -> pd.DataFrame:
    path = cfg["paths"]["merged"]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run the 'prepare' step first.")
    return pd.read_parquet(path, columns=columns, filters=filters)
