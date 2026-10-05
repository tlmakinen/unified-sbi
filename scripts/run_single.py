#!/usr/bin/env python
"""One budget, every arm, full diagnostics (the unified_sbi.ipynb workflow as a script).

    python scripts/run_single.py --n-total 2000 --out results/single

Writes the metrics table, the one-step training curve, coverage curves and an exact
corner plot for one held-out observation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from unified_sbi.exact import chain_posterior
from unified_sbi.plotting import corner_exact, plot_coverage
from unified_sbi.study import StudyConfig, run_cell
from unified_sbi.utils import atomic_write_json, resolve_device


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results/single"))
    p.add_argument("--simulator", default="rosenbrock8_hidden2d")
    p.add_argument("--n-total", type=int, default=2000)
    p.add_argument("--n-discovery", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", nargs="+", default=["raw", "theta_t", "eta_t"])
    p.add_argument("--n-test", type=int, default=256)
    p.add_argument("--obs-index", type=int, default=0, help="Test observation for the corner plot.")
    p.add_argument("--learn-scales", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    torch.set_num_threads(4)
    cfg = StudyConfig(simulator=args.simulator, budgets=(args.n_total,), seeds=(args.seed,),
                      arms=tuple(args.arms), n_discovery=args.n_discovery, n_test=args.n_test,
                      maps=dict(learn_scales=args.learn_scales), device=resolve_device(args.device))
    args.out.mkdir(parents=True, exist_ok=True)
    rows, art = run_cell(cfg, args.seed, args.n_total, out_dir=None, return_artifacts=True)
    atomic_write_json(args.out / "metrics.json", rows)
    table = pd.DataFrame(rows).set_index("arm")[
        ["n_total", "n_discovery", "val_log_prob", "test_log_prob", "oracle_test_log_prob",
         "crps", "oracle_crps", "coverage68", "coverage90", "npe_epochs"]]
    print(table.round(4).to_string())

    if art["maps"] is not None:
        hist = np.array(art["maps"].history)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(hist[:, 0], hist[:, 1], label="train batch", alpha=0.6)
        ax.plot(hist[:, 0], hist[:, 2], label="validation")
        ax.set(xlabel="one-step update", ylabel="one-step loss"); ax.legend()
        fig.tight_layout(); fig.savefig(args.out / "one_step_training.png", dpi=150)

    plot_coverage(art["draws"], art["theta_test"]).savefig(args.out / "coverage.png", dpi=150)

    k = args.obs_index
    post = chain_posterior(art["simulator"], art["x_test"][k], n_grid=250)
    draws = {arm: d[k] for arm, d in art["draws"].items()}
    corner_exact(post, draws, art["theta_test"][k],
                 title=f"test observation {k}, N_total = {args.n_total}").savefig(
        args.out / f"corner_obs{k}.png", dpi=130)
    print(f"figures in {args.out}")


if __name__ == "__main__":
    main()
