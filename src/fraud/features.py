from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .schema import CAT_COLS

GROUP_PATTERNS = {"C": r"^C\d+$", "D": r"^D\d+$", "V": r"^V\d+$", "id": r"^id_\d+$", "M": r"^M\d+$"}


def _obj(s: pd.Series) -> pd.Series:
    """Object dtype with real NaN for missing, whatever the input dtype was."""
    s = s.astype("object")
    return s.where(s.notna(), np.nan)


def _provider(s: pd.Series) -> pd.Series:
    low = _obj(s).str.lower()
    conds = [
        low.str.contains("gmail", na=False),
        low.str.contains("yahoo|ymail|rocketmail", na=False),
        low.str.contains("hotmail|outlook|live\\.|msn", na=False),
        low.str.contains("icloud|me\\.com|mac\\.com", na=False),
        low.str.contains("anonymous", na=False),
        low.str.contains("proton", na=False),
    ]
    labels = ["google", "yahoo", "microsoft", "apple", "anonymous", "proton"]
    out = pd.Series(np.select(conds, labels, default="other"), index=s.index, dtype=object)
    return out.where(low.notna(), np.nan)


def _key(*cols: pd.Series) -> pd.Series:
    parts = [_obj(c).fillna("na").astype(str) for c in cols]
    out = parts[0]
    for p in parts[1:]:
        out = out + "_" + p
    return out.astype(object)


class FeatureBuilder:
    def __init__(self, cfg: dict):
        f = cfg["features"]
        self.target = cfg["data"]["target"]
        self.id_col = cfg["data"]["id_col"]
        self.time_col = cfg["data"]["time_col"]
        self.drop_missing_above = f["drop_missing_above"]
        self.freq_cols = list(f["freq_encode"])
        self.agg_keys = list(f["amount_agg_by"])
        self.fitted = False

    # ------------------------------------------------------------------ raw input
    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make the frame have exactly the raw columns seen in training, with clean dtypes.
        Missing columns become NaN, so an API caller may send only a few fields."""
        cols = {}
        for c in self.raw_columns_:
            if c in df.columns:
                s = df[c]
            else:
                s = pd.Series(np.nan, index=df.index, dtype="float64")
            if c in self.raw_cat_cols_:
                cols[c] = _obj(s)
            else:
                cols[c] = pd.to_numeric(s, errors="coerce").astype("float32")
        return pd.DataFrame(cols, index=df.index)

    # ------------------------------------------------------------------ stateless features
    def _derive(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._ensure_columns(df)
        new: dict[str, pd.Series | np.ndarray] = {}

        dt = X[self.time_col].astype("float64")
        day = np.floor(dt / 86400.0)
        new["dt_hour"] = (dt // 3600) % 24
        new["dt_dow"] = day % 7

        amt = X["TransactionAmt"].astype("float64")
        cents = amt - np.floor(amt)
        new["amt_log"] = np.log1p(amt)
        new["amt_cents"] = cents
        new["amt_is_round"] = (cents == 0).astype(float)

        # e-mail: provider group and top-level domain (gmail.com / googlemail.com -> 'google')
        new["P_provider"] = _provider(X["P_emaildomain"])
        new["R_provider"] = _provider(X["R_emaildomain"])
        new["P_tld"] = _obj(X["P_emaildomain"]).str.extract(r"\.([A-Za-z]+)$")[0].str.lower()
        both = X["P_emaildomain"].notna() & X["R_emaildomain"].notna()
        new["email_match"] = np.where(both, (X["P_emaildomain"] == X["R_emaildomain"]).astype(float), np.nan)

        # device / browser: clean the very messy text fields
        new["os_family"] = _obj(X["id_30"]).str.extract(r"^([A-Za-z]+)")[0].str.lower()
        new["browser_family"] = (
            _obj(X["id_31"]).str.lower().str.extract(r"^([a-z][a-z ]*?)\s*(?:\d|$)")[0].str.strip()
        )
        new["device_family"] = _obj(X["DeviceInfo"]).str.extract(r"^([A-Za-z]+)")[0].str.lower()
        res = _obj(X["id_33"]).str.extract(r"^(\d+)x(\d+)$")
        new["screen_w"] = pd.to_numeric(res[0], errors="coerce")
        new["screen_h"] = pd.to_numeric(res[1], errors="coerce")

        # "account start day" trick: today minus 'days since first activity' ~ the same for one customer
        new["D1_start"] = day - X["D1"] if "D1" in X else np.nan
        new["D15_start"] = day - X["D15"] if "D15" in X else np.nan

        # pseudo customer ids (a card number is not a person, so we approximate)
        uid = _key(X["card1"], X["card2"], X["card3"], X["card5"], X["addr1"])
        new["uid"] = uid
        new["uid2"] = _key(X["card1"], X["addr1"], pd.Series(new["D1_start"], index=X.index).round())

        # how empty is this row? (missingness is a signal in fraud)
        for name, pat in GROUP_PATTERNS.items():
            grp = [c for c in X.columns if re.match(pat, c)]
            if grp and name != "M":
                new[f"n_missing_{name}"] = X[grp].isna().sum(axis=1)
        new["n_missing_total"] = X.isna().sum(axis=1)

        added = pd.DataFrame({k: (v if isinstance(v, pd.Series) else pd.Series(v, index=X.index))
                              for k, v in new.items()}, index=X.index)
        for c in added.columns:
            if added[c].dtype.kind in "fiub":
                added[c] = added[c].astype("float32")
        return pd.concat([X, added], axis=1)

    # ------------------------------------------------------------------ learned state
    def _fit_state(self, D: pd.DataFrame) -> None:
        self.freq_maps_ = {}
        for c in self.freq_cols:
            if c in D:
                self.freq_maps_[c] = _obj(D[c]).value_counts(normalize=True)
        amt = D["TransactionAmt"].astype("float64")
        self.agg_maps_ = {}
        for k in self.agg_keys:
            if k in D:
                g = amt.groupby(_obj(D[k])).agg(["mean", "std"])
                self.agg_maps_[k] = g
        obj_cols = [c for c in D.columns if D[c].dtype == object and c not in ("uid", "uid2")]
        self.cat_maps_ = {c: _obj(D[c]).value_counts().index.tolist() for c in obj_cols}
        skip = set(obj_cols) | {"uid", "uid2", self.time_col}
        self.numeric_cols_ = [c for c in D.columns if c not in skip]

    def _apply_state(self, D: pd.DataFrame) -> pd.DataFrame:
        new = {}
        for c, freq in self.freq_maps_.items():
            new[f"freq_{c}"] = _obj(D[c]).map(freq).fillna(0).astype("float32")
        amt = D["TransactionAmt"].astype("float64")
        for k, stats in self.agg_maps_.items():
            key = _obj(D[k])
            mean = key.map(stats["mean"]).astype("float64")
            std = key.map(stats["std"]).astype("float64")
            new[f"amt_mean_{k}"] = mean.astype("float32")
            new[f"amt_std_{k}"] = std.astype("float32")
            new[f"amt_ratio_{k}"] = (amt / mean.replace(0, np.nan)).astype("float32")
        for c, cats in self.cat_maps_.items():
            codes = pd.Categorical(D[c], categories=cats).codes
            new[f"{c}"] = np.where(codes < 0, np.nan, codes).astype("float32")
        num = D[self.numeric_cols_]
        extra = pd.DataFrame(new, index=D.index)
        out = pd.concat([num, extra], axis=1)
        return out.replace([np.inf, -np.inf], np.nan)  # a ratio with a zero denominator is 'unknown'

    # ------------------------------------------------------------------ public API
    def _derive_chunked(self, df: pd.DataFrame, chunk_rows: int = 100_000) -> pd.DataFrame:
        parts = [self._derive(df.iloc[i:i + chunk_rows]) for i in range(0, max(len(df), 1), chunk_rows)]
        return pd.concat(parts) if len(parts) > 1 else parts[0]

    def _fit_all(self, df_train: pd.DataFrame) -> pd.DataFrame:
        drop = {self.target, self.id_col}
        self.raw_columns_ = [c for c in df_train.columns if c not in drop]
        self.raw_cat_cols_ = [c for c in self.raw_columns_ if c in CAT_COLS]
        D = self._derive_chunked(df_train)
        self._fit_state(D)
        X = self._apply_state(D)
        del D
        miss = X.isna().mean()
        constant = (X.max() == X.min())
        too_empty = miss > self.drop_missing_above
        self.dropped_cols_ = sorted(X.columns[too_empty | constant].tolist())
        self.feature_names_ = [c for c in X.columns if c not in set(self.dropped_cols_)]
        self.fitted = True
        return X.reindex(columns=self.feature_names_).astype("float32")

    def fit(self, df_train: pd.DataFrame) -> "FeatureBuilder":
        self._fit_all(df_train)
        return self

    def transform(self, df: pd.DataFrame, chunk_rows: int = 100_000) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("call fit() on the training rows first")
        parts = []
        for start in range(0, max(len(df), 1), chunk_rows):
            chunk = df.iloc[start:start + chunk_rows]
            X = self._apply_state(self._derive(chunk))
            parts.append(X.reindex(columns=self.feature_names_).astype("float32"))
        return pd.concat(parts) if len(parts) > 1 else parts[0]

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        return self._fit_all(df_train)
