#!/usr/bin/env python
"""Aggregate a scaling study: figures, power-law exponents and simulation-savings factors.

    python scripts/plot_scaling.py --results results/scaling
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from unified_sbi.analysis import aggregate, load_rows, power_law_table, savings_table
from unified_sbi.plotting import plot_excess, plot_scaling


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results/scaling"))
    p.add_argument("--baseline", default="raw")
    p.add_argument("--fit-min-budget", type=int, default=0, help="Smallest N_total used in power-law fits.")
    args = p.parse_args(argv)

    fig_dir = args.results / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = load_rows(args.results)
    agg = aggregate(df)
    oracle_path = args.results / "oracle.json"
    oracle = json.loads(oracle_path.read_text()) if oracle_path.exists() else None
    cfg_path = args.results / "config.json"
    n_disc = json.loads(cfg_path.read_text()).get("n_discovery") if cfg_path.exists() else None

    plot_scaling(agg, oracle, n_disc).savefig(fig_dir / "scaling.png", dpi=150)
    plot_excess(agg, args.fit_min_budget).savefig(fig_dir / "excess_power_law.png", dpi=150)

    agg.to_csv(args.results / "summary_by_budget.csv", index=False)
    laws = power_law_table(agg, args.fit_min_budget)
    laws.to_csv(args.results / "power_laws.csv", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    view = agg[["arm", "n_total", "n_seeds", "val_log_prob_mean", "test_log_prob_mean",
                "crps_mean", "crps_sem", "coverage90_mean"]]
    print("\nPer-budget means over seeds\n", view.round(4).to_string(index=False))
    if oracle:
        print(f"\nExact posterior: test log p = {oracle['test_log_prob']:.4f}, CRPS = {oracle['crps']:.4f}")
    print("\nPower laws for excess over the exact posterior (y = A N^-alpha)\n", laws.round(3).to_string(index=False))

    if args.baseline in set(agg.arm):
        sav = savings_table(agg, args.baseline)
        sav.to_csv(args.results / "savings.csv", index=False)
        cols = ["arm", "n_total"] + [c for c in sav.columns if c.endswith("savings (text)")]
        print(f"\nSimulation savings vs '{args.baseline}' = baseline budget needed / N_total "
              f"(>= : baseline never matches inside the sweep; <= : worse than the baseline's "
              f"smallest budget)\n", sav[cols].rename(columns=lambda c: c.replace(" (text)", ""))
              .to_string(index=False))
    print(f"\nFigures in {fig_dir}")


if __name__ == "__main__":
    main()
