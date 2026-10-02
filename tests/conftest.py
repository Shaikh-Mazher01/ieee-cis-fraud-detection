import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.config import load_config  # noqa: E402
from fraud.synth import generate  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config("synthetic")


@pytest.fixture(scope="session")
def merged():
    """A small merged frame shaped exactly like the real one."""
    trans, ident = generate(6000, seed=7)
    df = trans.merge(ident, on="TransactionID", how="left")
    df["has_identity"] = df["TransactionID"].isin(ident["TransactionID"]).astype("int8")
    return df.sort_values("TransactionDT").reset_index(drop=True)
