#!/usr/bin/env python
"""Figures and tables for a lowrank_scaling.py results folder.

    python scripts/plot_lowrank.py --results results/lowrank
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from unified_sbi.lowrank.plots import aggregate, load, plot_curves, plot_screen, savings


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results/lowrank"))
    args = p.parse_args(argv)
    fig_dir = args.results / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df, screen = load(args.results)
    agg = aggregate(df)
    floor = df.oracle_crps.iloc[0]
    agg.to_csv(args.results / "summary_by_budget.csv", index=False)
    plot_curves(agg, floor).savefig(fig_dir / "learning_curves.png", dpi=150)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(agg[["arm", "n_total", "n_seeds", "posterior_kl_mean", "posterior_kl_sem", "crps_mean",
               "sw_active_mean", "val_log_prob_mean", "k_mean", "m_mean"]].round(3).to_string(index=False))
    print(f"\nexact-posterior CRPS floor: {floor:.4f}")
    if screen is not None:
        screen.to_csv(args.results / "screen_all.csv", index=False)
        plot_screen(screen).savefig(fig_dir / "screen.png", dpi=150)
        picks = screen[screen.selected.astype(bool)].groupby("n_total")[["k", "m"]].agg(lambda s: list(s))
        print("\nscreen selections per budget (one per seed)\n", picks.to_string())
    for base in ("raw", "mean"):
        if base in set(agg.arm):
            print(f"\nsimulation savings vs '{base}' (baseline budget to match / N; >= never matched, "
                  f"<= worse than its smallest budget)\n", savings(agg, base).to_string(index=False))
    print(f"\nfigures in {fig_dir}")


if __name__ == "__main__":
    main()
