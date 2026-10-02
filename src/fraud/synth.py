from __future__ import annotations

import numpy as np
import pandas as pd

OS_VALUES = ["Windows 10", "Windows 7", "iOS 11.1.0", "iOS 11.2.1", "Android 7.0", "Android 8.0.0",
             "Mac OS X 10_12_6", "Mac OS X 10_13_2", "Linux"]
BROWSERS = ["chrome 63.0", "chrome 64.0", "chrome 63.0 for android", "mobile safari generic",
            "safari generic", "firefox 57.0", "edge 15.0", "ie 11.0 for desktop", "samsung browser 6.2",
            "opera 49.0", "chrome 65.0 for ios"]
RESOLUTIONS = ["1920x1080", "1366x768", "2220x1080", "1334x750", "1280x800", "2560x1600", "1440x900"]
DEVICES = ["Windows", "iOS Device", "MacOS", "SAMSUNG SM-G892A Build/NRD90M", "HUAWEI VNS-L21 Build/HUAWEIVNS-L21",
           "Moto G (5) Build/NPPS25.137-93-14", "LG-H900 Build/NRD90C", "rv:57.0", "Trident/7.0"]
EMAILS = ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com", "outlook.com", "icloud.com",
          "comcast.net", "msn.com", "protonmail.com", "mail.com", "live.com", "att.net", "yahoo.co.uk",
          "gmail", "me.com", "sbcglobal.net"]
EMAIL_P = np.array([38, 17, 7, 4, 4, 3, 3, 3, 2, 0.5, 1, 1.5, 1, 0.5, 1, 1, 1.5], dtype=float)
EMAIL_P /= EMAIL_P.sum()
EMAIL_EFFECT = {"anonymous.com": 1.6, "protonmail.com": 1.4, "mail.com": 1.0, "hotmail.com": 0.25,
                "gmail.com": 0.1, "yahoo.com": -0.2, "icloud.com": -0.3, "comcast.net": -0.4}


def _pick(rng, values, p, n):
    return np.asarray(values, dtype=object)[rng.choice(len(values), size=n, p=np.asarray(p) / np.sum(p))]


def _mask(rng, n, rate):
    return rng.random(n) < rate


def generate(n: int = 100_000, seed: int = 42, fraud_rate: float = 0.035, identity_share: float = 0.24):
    rng = np.random.default_rng(seed)

    # ---------------- time: 183 days, busier in the evening
    hour_w = np.array([2, 1.5, 1, 1, 1, 1.5, 2.5, 4, 5, 6, 6.5, 7, 7, 7, 7, 7, 7.5, 8, 8.5, 8.5, 8, 6, 4, 3], float)
    hours = rng.choice(24, size=n, p=hour_w / hour_w.sum())
    day = rng.integers(1, 183, size=n)
    dt = 86400 + day * 86400 + hours * 3600 + rng.integers(0, 3600, size=n)
    order = np.argsort(dt, kind="stable")
    dt, day, hours = dt[order], day[order], hours[order]
    t_norm = (dt - dt.min()) / (dt.max() - dt.min())

    # ---------------- cards (repeat customers, Zipf-like activity)
    n_cards = max(300, n // 12)
    card_ids = rng.choice(np.arange(1000, 18397), size=n_cards, replace=n_cards > 17000)
    card2 = np.where(_mask(rng, n_cards, 0.015), np.nan, rng.integers(100, 600, n_cards)).astype("float32")
    card3 = _pick(rng, [150, 185, 144, 117], [88, 9, 2, 1], n_cards).astype("float32")
    card4 = _pick(rng, ["visa", "mastercard", "american express", "discover"], [65, 32, 1.5, 1.5], n_cards)
    card5 = _pick(rng, [226, 224, 166, 102, 117, 137, 195], [40, 20, 20, 5, 5, 5, 5], n_cards).astype("float32")
    card6 = _pick(rng, ["debit", "credit", "debit or credit"], [75, 24, 1], n_cards)
    addr1 = np.where(_mask(rng, n_cards, 0.11), np.nan, rng.integers(100, 540, n_cards)).astype("float32")
    first_day = rng.integers(-400, 150, n_cards)
    risk_card = rng.normal(0, 0.6, n_cards)
    comp_start = rng.integers(10, 170, n_cards)
    compromised = rng.random(n_cards) < 0.04
    w = 1.0 / np.arange(1, n_cards + 1) ** 0.8
    w = w[rng.permutation(n_cards)]
    ci = rng.choice(n_cards, size=n, p=w / w.sum())

    # ---------------- transaction fields
    product = _pick(rng, ["W", "C", "R", "H", "S"], [74, 11, 6, 8, 2], n)
    amt = np.exp(rng.normal(4.1, 1.05, n)) * (1 + 0.15 * t_norm)
    amt = np.clip(amt, 0.3, 31000)
    is_round = _mask(rng, n, 0.30) & (product == "W")
    amt = np.where(is_round, np.round(amt), np.round(amt, 2))
    p_email = _pick(rng, EMAILS, EMAIL_P, n)
    p_email = np.where(_mask(rng, n, 0.16), None, p_email).astype(object)
    r_email = np.where(_mask(rng, n, 0.76), None, _pick(rng, EMAILS, EMAIL_P, n)).astype(object)
    has_id = _mask(rng, n, identity_share) | (product == "C") & _mask(rng, n, 0.15)

    age = np.maximum(day - first_day[ci], 0).astype("float32")
    activity = np.log1p(np.bincount(ci, minlength=n_cards))[ci]
    in_burst = compromised[ci] & (day >= comp_start[ci]) & (day <= comp_start[ci] + 10)

    # latent factors that drive the V-columns and also shift fraud risk
    Z = rng.normal(size=(n, 12)).astype("float32")

    # ---------------- fraud probability (logit scale)
    prod_eff = pd.Series(product).map({"C": 1.0, "R": -0.3, "H": -0.4, "S": -0.3, "W": -0.1}).to_numpy()
    mail_eff = pd.Series(p_email).map(EMAIL_EFFECT).fillna(0.0).to_numpy() * (1 - 0.45 * t_norm)  # drifts
    logit = (
        prod_eff + mail_eff
        + 0.25 * (np.log(amt) - 4.1) + 0.3 * is_round
        + 0.5 * (hours < 7)
        + 0.9 * has_id
        + 0.35 * (card6[ci] == "credit") - 0.2 * (card4[ci] == "discover")
        + 0.8 * (age < 3) + 0.25 * np.clip(activity - 2, -2, 3)
        + 0.9 * Z[:, 0] - 0.6 * Z[:, 1] + 0.5 * Z[:, 2] * (product == "C")
        + risk_card[ci] + 2.6 * in_burst
        + rng.normal(0, 0.35, n)
    )
    lo, hi = -15.0, 5.0
    for _ in range(60):  # bisection: find the intercept that gives the wanted fraud rate
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(logit + mid)))).mean() > fraud_rate:
            hi = mid
        else:
            lo = mid
    prob = 1 / (1 + np.exp(-(logit + (lo + hi) / 2)))
    fraud = (rng.random(n) < prob).astype("int8")

    # ---------------- count (C) and delta-days (D) columns
    base_cnt = rng.poisson(1 + 1.5 * activity)
    cols: dict[str, np.ndarray] = {}
    cols["TransactionID"] = np.arange(2987000, 2987000 + n, dtype="int32")
    cols["isFraud"] = fraud
    cols["TransactionDT"] = dt.astype("int32")
    cols["TransactionAmt"] = amt.astype("float32")
    cols["ProductCD"] = product
    cols["card1"] = card_ids[ci].astype("float32")
    cols["card2"] = card2[ci]
    cols["card3"] = card3[ci]
    cols["card4"] = card4[ci]
    cols["card5"] = card5[ci]
    cols["card6"] = card6[ci]
    cols["addr1"] = addr1[ci]
    cols["addr2"] = np.where(np.isnan(addr1[ci]), np.nan, 87.0).astype("float32")
    cols["dist1"] = np.where(_mask(rng, n, 0.60), np.nan, np.abs(rng.gamma(0.8, 60, n))).astype("float32")
    cols["dist2"] = np.where(_mask(rng, n, 0.93), np.nan, np.abs(rng.gamma(0.8, 200, n))).astype("float32")
    cols["P_emaildomain"] = p_email
    cols["R_emaildomain"] = r_email
    for k in range(1, 15):
        c = base_cnt + rng.poisson(0.3 * k % 3) + (in_burst * rng.poisson(3, n) if k in (1, 2, 6, 11, 13) else 0)
        cols[f"C{k}"] = c.astype("float32")
    miss_d = {1: 0.002, 2: 0.47, 3: 0.52, 4: 0.28, 5: 0.52, 6: 0.76, 7: 0.93, 8: 0.87, 9: 0.87,
              10: 0.13, 11: 0.47, 12: 0.89, 13: 0.88, 14: 0.88, 15: 0.15}
    for k in range(1, 16):
        d = age * rng.uniform(0.4, 1.1, n) if k != 3 else rng.integers(0, 60, n).astype(float)
        cols[f"D{k}"] = np.where(_mask(rng, n, miss_d[k]), np.nan, np.floor(d)).astype("float32")
    for k, rate in zip(range(1, 10), [0.46, 0.46, 0.46, 0.48, 0.60, 0.28, 0.86, 0.86, 0.86]):
        tf = np.where(rng.random(n) < (0.85 - 0.15 * (fraud if k in (3, 4, 6) else 0)), "T", "F")
        cols[f"M{k}"] = np.where(_mask(rng, n, rate), None, tf).astype(object)

    # ---------------- V columns: 339 engineered features in groups sharing latent factors + NaN blocks
    sizes = []
    left = 339
    while left > 0:
        s = int(min(left, rng.integers(9, 28)))
        sizes.append(s)
        left -= s
    group_miss = [0.0, 0.0, 0.13, 0.28, 0.28, 0.46, 0.50, 0.0, 0.13, 0.76, 0.86, 0.0, 0.28, 0.46, 0.5, 0.86, 0.0, 0.13]
    v_idx = 1
    for gi, s in enumerate(sizes):
        f1, f2 = rng.integers(0, 12, 2)
        miss_rows = _mask(rng, n, group_miss[gi % len(group_miss)])
        load = rng.uniform(0.3, 1.4, size=(s, 2)).astype("float32")
        base = np.stack([Z[:, f1], Z[:, f2]], 1) @ load.T + rng.normal(0, 0.7, (n, s)).astype("float32")
        base = np.maximum(base + 0.5, 0)
        if gi % 2 == 0:
            base = np.round(base * 2)
        base[miss_rows] = np.nan
        for j in range(s):
            cols[f"V{v_idx}"] = base[:, j].astype("float32")
            v_idx += 1

    trans = pd.DataFrame(cols)

    # ---------------- identity table (only rows that have identity info, like the real file)
    m = int(has_id.sum())
    r = np.random.default_rng(seed + 1)
    idc: dict[str, np.ndarray] = {"TransactionID": cols["TransactionID"][has_id]}
    f_id = fraud[has_id].astype(bool)

    def num(rate, gen):
        return np.where(r.random(m) < rate, np.nan, gen).astype("float32")

    idc["id_01"] = num(0.0, -r.choice([0, 5, 10, 15, 20, 25, 30, 40, 55, 75, 100], m, p=[.05, .6, .1, .06, .05, .03, .03, .03, .02, .02, .01]))
    idc["id_02"] = num(0.02, np.exp(r.normal(11.5, 1.3, m)).round())
    idc["id_03"] = num(0.54, r.choice([0, 1, 2, -5], m, p=[.9, .05, .03, .02]))
    idc["id_04"] = num(0.54, r.choice([0, -5, -1], m, p=[.9, .08, .02]))
    idc["id_05"] = num(0.06, r.choice([0, 1, 2, 5, 10, 52], m, p=[.7, .1, .08, .06, .04, .02]))
    idc["id_06"] = num(0.06, -r.choice([0, 6, 10, 100], m, p=[.7, .1, .1, .1]))
    for k in (7, 8):
        idc[f"id_0{k}"] = num(0.99, r.normal(10, 15, m))
    for k in (9, 10):
        idc[f"id_{k:02d}"] = num(0.50, r.choice([0, 1, 2, 3, 5], m, p=[.4, .3, .15, .1, .05]))
    idc["id_11"] = num(0.02, r.choice([100.0, 90.0, 95.0, 98.0], m, p=[.9, .04, .03, .03]))
    idc["id_12"] = _pick(r, ["Found", "NotFound"], [.5, .5], m)
    idc["id_13"] = num(0.12, r.choice([49, 52, 33, 63, 27], m))
    idc["id_14"] = num(0.60, r.choice([-300, -360, -480, 0, 60, 120], m))
    idc["id_15"] = _pick(r, ["Found", "New", "Unknown"], [.45, .45, .1], m)
    idc["id_16"] = np.where(_mask(r, m, 0.12), None, _pick(r, ["Found", "NotFound"], [.6, .4], m)).astype(object)
    idc["id_17"] = num(0.01, r.choice([166, 225, 153, 100], m, p=[.7, .1, .1, .1]))
    idc["id_18"] = num(0.70, r.choice([15, 20, 25, 13], m))
    idc["id_19"] = num(0.01, r.integers(100, 700, m))
    idc["id_20"] = num(0.01, r.integers(100, 700, m))
    for k in range(21, 27):
        idc[f"id_{k}"] = num(0.99, r.integers(100, 600, m))
    idc["id_23"] = np.where(_mask(r, m, 0.99), None, "IP_PROXY:TRANSPARENT").astype(object)
    idc["id_24"] = num(0.99, 11.0)
    idc["id_25"] = num(0.99, r.integers(100, 600, m))
    idc["id_26"] = num(0.99, r.integers(100, 600, m))
    idc["id_27"] = np.where(_mask(r, m, 0.99), None, "Found").astype(object)
    idc["id_28"] = np.where(_mask(r, m, 0.01), None, _pick(r, ["New", "Found"], [.5, .5], m)).astype(object)
    idc["id_29"] = np.where(_mask(r, m, 0.01), None, _pick(r, ["Found", "NotFound"], [.5, .5], m)).astype(object)
    os_ = _pick(r, OS_VALUES, [25, 10, 12, 6, 12, 6, 8, 4, 2], m)
    idc["id_30"] = np.where(_mask(r, m, 0.46), None, os_).astype(object)
    br = _pick(r, BROWSERS, [28, 9, 12, 12, 4, 6, 4, 6, 3, 1, 1], m)
    br = np.where(f_id & (r.random(m) < 0.2), "chrome 63.0 for android", br)  # a fraud-heavy browser
    idc["id_31"] = br.astype(object)
    idc["id_32"] = num(0.46, r.choice([24, 32, 16], m, p=[.7, .25, .05]))
    idc["id_33"] = np.where(_mask(r, m, 0.50), None, _pick(r, RESOLUTIONS, [30, 20, 10, 10, 10, 10, 10], m)).astype(object)
    idc["id_34"] = np.where(_mask(r, m, 0.46), None, "match_status:2").astype(object)
    for k, p_t in zip((35, 36, 37, 38), (.6, .1, .7, .5)):
        idc[f"id_{k}"] = np.where(_mask(r, m, 0.01), None, np.where(r.random(m) < p_t, "T", "F")).astype(object)
    mobile = np.isin(os_, ["iOS 11.1.0", "iOS 11.2.1", "Android 7.0", "Android 8.0.0"])
    idc["DeviceType"] = np.where(_mask(r, m, 0.01), None, np.where(mobile, "mobile", "desktop")).astype(object)
    idc["DeviceInfo"] = np.where(_mask(r, m, 0.20), None, _pick(r, DEVICES, [25, 12, 10, 12, 8, 6, 6, 8, 13], m)).astype(object)
    ident = pd.DataFrame(idc)
    return trans, ident


def write_csvs(cfg: dict, rows: int, seed: int | None = None) -> None:
    out = cfg["paths"]["raw_dir"]
    out.mkdir(parents=True, exist_ok=True)
    trans, ident = generate(rows, seed or cfg["seed"])
    trans.to_csv(out / cfg["data"]["transaction_file"], index=False)
    ident.to_csv(out / cfg["data"]["identity_file"], index=False)
    print(f"wrote {len(trans):,} transactions x {trans.shape[1]} cols and {len(ident):,} identity rows to {out}")
    print(f"fraud rate {trans['isFraud'].mean():.2%} | identity coverage {len(ident) / len(trans):.1%}")
