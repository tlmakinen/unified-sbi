#!/usr/bin/env python
"""Plot theta|t vs eta|t on the active (theta_1, theta_2) plane. Writes only into <results>/active_plane/.

    python scripts/plot_active_plane.py --results results/preview_seeds012_cpu
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL = {"theta_t": "#2a78d6", "eta_t": "#e34948"}
LAB = {"theta_t": r"NPE $\theta\mid t$", "eta_t": r"NPE $\eta\mid t$"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

ap = argparse.ArgumentParser(); ap.add_argument("--results", type=Path, required=True)
a = ap.parse_args()
d = a.results / "active_plane"
rows = [{k: v for k, v in r.items() if k != "per_obs"} for f in sorted(d.glob("seed*_n*.json"))
        for r in json.loads(f.read_text())]
df = pd.DataFrame(rows)
df.to_csv(d / "active_plane.csv", index=False)

metrics = [("excess2d_kde", "active-plane excess log density\n(KDE-matched, nats) ↓"),
           ("energy2d", "active-plane energy distance\nto exact posterior ↓"),
           ("crps_theta1", r"CRPS $\theta_1$ ↓"), ("crps_theta2", r"CRPS $\theta_2$ ↓")]
fig, axes = plt.subplots(1, 5, figsize=(22, 4.3))
for ax, (m, lab) in zip(axes, metrics):
    for arm in ("theta_t", "eta_t"):
        g = df[df.arm == arm].groupby("n_total")[m].agg(["mean", "std", "count"])
        ax.errorbar(g.index, g["mean"], g["std"] / np.sqrt(g["count"]), color=COL[arm], marker="o",
                    lw=2, capsize=3, label=LAB[arm])
    ax.set_xscale("log"); ax.set_ylabel(lab, color=INK); ax.set_xlabel(r"total simulations $N$", color=INK)
    ax.grid(color=GRID, lw=0.8)
# paired difference (eta - theta) per seed, for the 2D excess and the 8D test log q
p = df.pivot_table(index=["seed", "n_total"], columns="arm", values=["excess2d_kde", "test_log_prob_8d"])
diff = pd.DataFrame({"2D excess: η|t − θ|t": p["excess2d_kde"]["eta_t"] - p["excess2d_kde"]["theta_t"],
                     "8D −log q: η|t − θ|t": -(p["test_log_prob_8d"]["eta_t"] - p["test_log_prob_8d"]["theta_t"])})
ax = axes[-1]
for (c, col) in zip(diff.columns, ("#4a3aa7", "#eb6834")):
    g = diff[c].groupby("n_total").agg(["mean", "std", "count"])
    ax.errorbar(g.index, g["mean"], g["std"] / np.sqrt(g["count"]), color=col, marker="s", lw=2, capsize=3, label=c)
ax.axhline(0, color=MUTED, ls=":", lw=1)
ax.set_xscale("log"); ax.set_ylabel("paired gap, nats (>0: η|t worse)", color=INK)
ax.set_xlabel(r"total simulations $N$", color=INK); ax.grid(color=GRID, lw=0.8); ax.legend(frameon=False, fontsize=8)
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle(r"Active plane $(\theta_1,\theta_2)$ of rosenbrock8_hidden2d: mean ± SE over seeds "
             f"{[int(s) for s in sorted(df.seed.unique())]}", color=INK)
fig.tight_layout(); fig.savefig(d / "active_plane.png", dpi=150); plt.close(fig)
s = diff.groupby("n_total").agg(["mean", "std"]).round(3)
print(df.groupby(["arm", "n_total"])[[m for m, _ in metrics]].mean().round(3).unstack(0).to_string())
print("\npaired gaps (eta - theta), mean/std over seeds:\n", s.to_string())
print("wrote", d / "active_plane.png")
