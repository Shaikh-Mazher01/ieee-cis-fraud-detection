from __future__ import annotations

import json

import pandas as pd

from .ab_plan import sample_size


def _usd(x: float) -> str:
    return f"${x:,.0f}"


def write_summary(cfg: dict) -> str:
    reports = cfg["paths"]["reports_dir"]
    m = json.loads((reports / "metrics.json").read_text())
    ab = sample_size(cfg)
    cap = m["capture"]
    weak = ", ".join(f"{w['group']}={w['value']} (recall {w['recall']:.0%})" for w in m["weakest_segments"])
    synthetic_note = ("\n> **WARNING: these numbers come from SYNTHETIC data. They only prove the pipeline runs. "
                      "Do not present them as results.**\n") if cfg["profile"] == "synthetic" else ""

    text = f"""# Fraud detection: executive summary
{synthetic_note}
**Problem.** About {m['test_fraud_rate']:.1%} of card transactions are fraud. Missed fraud is lost money and
reviewing every transaction is impossible. We need to decide, in real time, which transactions a human should look at.

**Approach.** A gradient-boosted tree model ({m['best_model']}) trained on past transactions and tested on the
*most recent* period it had never seen, the way it would be used in real life. The decision threshold
({m['threshold']:.2f}) was chosen to minimise money lost, assuming {_usd(m['business']['review_cost'])} per manual review.

**Result (unseen recent period, {m['test_rows']:,} transactions, {m['test_frauds']:,} frauds).**
- The model catches **{m['recall']:.0%}** of fraud and **{m['precision']:.0%}** of its flags are real fraud
  (PR-AUC {m['pr_auc']:.2f}; a model that guesses "never fraud" is {m['dummy_accuracy']:.1%} accurate but catches nothing).
- Reviewing only the riskiest 5% of traffic finds **{cap['top_5pct']:.0%}** of all fraud.
- It flags {m['flag_rate']:.1%} of transactions for review.

**Impact.** Cost over the test period: no model {_usd(m['cost_approve_all'])} -> with model {_usd(m['cost_model'])},
a saving of **{_usd(m['saving'])}** ({m['saving'] / max(m['cost_approve_all'], 1):.0%}). Reviewing everything would cost {_usd(m['cost_review_all'])}.

**Risks.**
- Fraudsters adapt; performance should be watched weekly (see `drift_train_vs_test.csv`).
- Weakest segments: {weak}.
- Savings depend on the review-cost assumption; re-run with the real figure from operations.

**Next step.** Run a live A/B test: model vs current rules, {ab['per_arm']:,} transactions per arm
(about {ab['days_rounded_to_full_weeks']} days at {cfg['ab_test']['daily_transactions']:,} transactions a day) to confirm the
fraud-loss rate falls from {ab['baseline_rate']:.1%} to {ab['target_rate']:.2%} without declining more good customers.

**What drives the score (SHAP, top 10):** {', '.join(m['top_features']) or 'n/a'}.
These are patterns the model uses, not proven causes of fraud.
"""
    out = reports / "executive_summary.md"
    out.write_text(text, encoding="utf-8")
    return text
