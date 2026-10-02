from __future__ import annotations

# Columns that hold text categories. Everything else is numeric.
CAT_COLS = (
    ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"]
    + [f"M{i}" for i in range(1, 10)]
    + ["id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29", "id_30", "id_31",
       "id_33", "id_34", "id_35", "id_36", "id_37", "id_38", "DeviceType", "DeviceInfo"]
)

INT_COLS = {"TransactionID": "int32", "isFraud": "int8", "TransactionDT": "int32"}


def normalise_name(col: str) -> str:
    """The test identity file uses id-01, the train file id_01. Make both id_01."""
    return col.replace("-", "_")
