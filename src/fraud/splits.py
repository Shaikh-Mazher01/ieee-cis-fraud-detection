from __future__ import annotations

import numpy as np
import pandas as pd


def time_split(df: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    """Return positional index arrays for train / valid / test, oldest to newest."""
    t = df[cfg["data"]["time_col"]].to_numpy()
    order = np.argsort(t, kind="stable")
    n = len(order)
    n_train = int(n * cfg["split"]["train_frac"])
    n_valid = int(n * cfg["split"]["valid_frac"])
    return {
        "train": order[:n_train],
        "valid": order[n_train:n_train + n_valid],
        "test": order[n_train + n_valid:],
    }
