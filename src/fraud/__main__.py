from __future__ import annotations

import argparse
import logging
import sys
import time

from .config import load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fraud", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["synth", "prepare", "sql", "eda", "train", "evaluate", "drift", "report",
                                     "abplan", "all", "serve"])
    ap.add_argument("--profile", default="real", choices=["real", "synthetic"],
                    help="'synthetic' = fake data with the same schema (for a 2-minute smoke test)")
    ap.add_argument("--fast", action="store_true", help="smaller/faster XGBoost, no tuning (laptop check)")
    ap.add_argument("--nrows", type=int, default=None, help="read only the first N transactions (debug)")
    ap.add_argument("--rows", type=int, default=100_000, help="rows to generate for the 'synth' step")
    ap.add_argument("--trials", type=int, default=None, help="random-search trials (0 = skip tuning)")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    cfg = load_config(a.profile)

    def run(step: str):
        t0 = time.time()
        logging.info(">>> %s (%s)", step, a.profile)
        if step == "synth":
            from .synth import write_csvs
            write_csvs(cfg, a.rows)
        elif step == "prepare":
            from .data import prepare
            prepare(cfg, a.nrows)
        elif step == "sql":
            from .sql_analysis import run_sql
            run_sql(cfg)
        elif step == "eda":
            from .eda import run_eda
            run_eda(cfg)
        elif step == "train":
            from .train import train
            train(cfg, fast=a.fast, n_trials=a.trials)
        elif step == "evaluate":
            from .evaluate import evaluate
            evaluate(cfg)
        elif step == "drift":
            from .drift import drift_report
            drift_report(cfg)
        elif step == "report":
            from .report import write_summary
            write_summary(cfg)
            logging.info("wrote %s", cfg["paths"]["reports_dir"] / "executive_summary.md")
        elif step == "abplan":
            from .ab_plan import sample_size
            print(sample_size(cfg))
        logging.info("<<< %s done in %.0fs", step, time.time() - t0)

    if a.step == "all":
        for s in ("prepare", "sql", "eda", "train", "evaluate", "drift", "report"):
            run(s)
        print(f"\nAll done. Open: outputs/{a.profile}/reports/executive_summary.md and outputs/{a.profile}/figures/")
    elif a.step == "serve":
        import os

        import uvicorn
        os.environ.setdefault("FRAUD_BUNDLE", str(cfg["paths"]["bundle"]))
        uvicorn.run("fraud.api:app", host="127.0.0.1", port=a.port)
    else:
        run(a.step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
