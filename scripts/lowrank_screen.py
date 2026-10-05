#!/usr/bin/env python
"""Run only the (k, m) screen on a fixed discovery budget, across seeds (no NPE).

    python scripts/lowrank_screen.py --n-discovery 400 1200 4000 --seeds 0 1 2 --m-mode all

Writes screen.csv (one row per seed, n_discovery, k, m) and figures/screen.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch

from unified_sbi.lowrank.maps import StructuredMapConfig
from unified_sbi.lowrank.plots import plot_screen
from unified_sbi.lowrank.screen import candidate_grid, run_screen
from unified_sbi.lowrank.simulator import HiddenRosenbrock
from unified_sbi.utils import resolve_device, seed_everything, to_tensor


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results/lowrank_screen"))
    p.add_argument("--n-discovery", type=int, nargs="+", default=[400, 1200, 4000])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--m-mode", default="standard", choices=["standard", "all", "identity"])
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    device = resolve_device(args.device)
    torch.set_num_threads(4)
    sim = HiddenRosenbrock()
    cfg = StructuredMapConfig(max_steps=args.max_steps)
    grid = candidate_grid(args.k_values, args.m_mode)
    tables = []
    for seed in args.seeds:
        seed_everything(seed)
        theta, x = sim.simulate(max(args.n_discovery), seed=1000 * seed + 11)
        is_val = np.random.default_rng(1000 * seed + 12).random(len(theta)) < args.val_fraction
        for n in args.n_discovery:
            idx = np.arange(n)
            tr, va = idx[~is_val[:n]], idx[is_val[:n]]
            th, xx = to_tensor(theta[:n], device), to_tensor(x[:n], device)
            print(f"seed {seed}, n_discovery {n}: {len(tr)} fit / {len(va)} validation")
            res = run_screen(th[tr], xx[tr], th[va], xx[va], grid, cfg, seed=seed, true_rows=sim.Q[:2])
            print(f"  -> selected k={res.selected[0]}, m={res.selected[1]}")
            tables.append(res.table.assign(seed=seed, n_total=n, n_discovery=n))
    screen = pd.concat(tables, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    screen.to_csv(args.out / "screen.csv", index=False)
    (args.out / "figures").mkdir(exist_ok=True)
    plot_screen(screen).savefig(args.out / "figures" / "screen.png", dpi=150)
    print(screen[screen.selected].groupby("n_discovery")[["k", "m"]].agg(list).to_string())


if __name__ == "__main__":
    main()
