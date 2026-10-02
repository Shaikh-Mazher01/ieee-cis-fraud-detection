from __future__ import annotations

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def sample_size(cfg: dict) -> dict:
    a = cfg["ab_test"]
    p1 = a["baseline_loss_rate"]
    p2 = p1 * (1 - a["relative_reduction"])
    effect = proportion_effectsize(p1, p2)
    n_arm = NormalIndPower().solve_power(effect_size=effect, alpha=a["alpha"], power=a["power"], alternative="two-sided")
    n_arm = int(round(n_arm))
    days = 2 * n_arm / a["daily_transactions"]
    return {
        "baseline_rate": p1, "target_rate": p2, "per_arm": n_arm, "total": 2 * n_arm,
        "days": round(days, 1), "days_rounded_to_full_weeks": int(-(-days // 7) * 7),
        "note": "Run whole weeks so every weekday appears equally in both arms.",
    }


if __name__ == "__main__":
    from .config import load_config
    print(sample_size(load_config()))
