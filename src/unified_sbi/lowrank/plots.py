"""Aggregation and figures for the rank-deficient study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..analysis import equivalent_budget

STYLE = {
    "raw":        dict(color="#6b6b6b", marker="o", label="θ | raw x"),
    "mean":       dict(color="#9467bd", marker="v", label="θ | mean x"),
    "summary":    dict(color="#1f77b4", marker="s", label="θ | t"),
    "eta":        dict(color="#d62728", marker="D", label="η(D) | t"),
    "structured": dict(color="#2ca02c", marker="^", label="η(k) | t × prior(D−k)"),
    "oracle":     dict(color="#8c564b", marker="*", label="oracle: u(2) | s(3)"),
    "gaussian":   dict(color="#e377c2", marker="P", label="one-step Gaussian"),
    "prior":      dict(color="#bcbd22", marker="x", label="prior only"),
}
METRICS = [("posterior_kl", "posterior KL  (log p − log q at θ*)", False),
           ("crps", "test CRPS (native θ)", False),
           ("sw_active", "sliced Wasserstein, true active plane", False),
           ("sw_full", "sliced Wasserstein, native θ", False),
           ("val_log_prob", "end-of-training validation log q(θ|x)", True)]


def load(results_dir):
    rows = []
    for p in sorted(Path(results_dir, "rows").glob("*.json")):
        rows += json.loads(p.read_text())
    screens = [pd.read_csv(p) for p in sorted(Path(results_dir, "screens").glob("*.csv"))]
    return pd.DataFrame(rows), (pd.concat(screens, ignore_index=True) if screens else None)


def aggregate(df):
    cols = [m for m, _, _ in METRICS] + ["coverage90", "active_rank_error", "null_cov_error", "k", "m"]
    g = df.groupby(["arm", "n_total"])[cols]
    return (g.mean().add_suffix("_mean").join(g.sem().add_suffix("_sem"))
            .join(g.size().rename("n_seeds")).reset_index())


def _ordered(agg):
    order = {a: i for i, a in enumerate(STYLE)}
    return sorted(agg.groupby("arm"), key=lambda kv: order.get(kv[0], 99))


def plot_curves(agg, floor_crps=None):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, (col, label, _) in zip(axes.flat, METRICS):
        for arm, g in _ordered(agg):
            if arm == "prior" and col == "val_log_prob":
                continue
            g = g.sort_values("n_total")
            st = STYLE.get(arm, dict(marker="o", label=arm))
            ax.errorbar(g.n_total, g[f"{col}_mean"], yerr=g[f"{col}_sem"], capsize=3, lw=1.5, ms=5,
                        color=st.get("color"), marker=st["marker"], label=st["label"])
        if col == "posterior_kl":
            ax.axhline(0, color="k", ls="--", lw=0.8)
        if col == "crps" and floor_crps is not None:
            ax.axhline(floor_crps, color="k", ls="--", lw=0.8, label="exact posterior")
        ax.set(xscale="log", xlabel="total simulations N", ylabel=label)
        ax.grid(alpha=0.3, which="both")
    ax = axes.flat[5]
    for arm, g in _ordered(agg[agg.k_mean.notna()]):
        if arm != "summary":
            continue
        g = g.sort_values("n_total")
        ax.errorbar(g.n_total, g.k_mean, yerr=g.k_sem, marker="o", capsize=3, label="selected k")
        ax.errorbar(g.n_total, g.m_mean, yerr=g.m_sem, marker="s", capsize=3, label="selected m")
    ax.axhline(2, color="C0", ls=":", lw=1, label="true k = 2")
    ax.axhline(3, color="C1", ls=":", lw=1, label="sufficient m = 3")
    ax.set(xscale="log", xlabel="total simulations N", ylabel="screen choice (mean over seeds)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_screen(screen, n_total=None):
    """Held-out one-step NLL relative to the best candidate (paired within seed), per k."""
    s = screen if n_total is None else screen[screen.n_total == n_total]
    s = s.copy()
    s["rel_nll"] = s.val_nll - s.groupby(["seed", "n_total"]).val_nll.transform("min")
    budgets = sorted(s.n_total.unique())
    fig, axes = plt.subplots(1, len(budgets), figsize=(3.6 * len(budgets), 3.4), squeeze=False, sharey=True)
    for ax, n in zip(axes[0], budgets):
        sn = s[s.n_total == n]
        for k, g in sn.groupby("k"):
            agg = g.groupby("m").rel_nll.agg(["mean", "sem"]).reset_index()
            ax.errorbar(agg.m, agg["mean"], yerr=agg["sem"].fillna(0), marker="o", capsize=3, label=f"k={k}")
        sel = sn[sn.selected.astype(bool)]
        ax.scatter(sel.m, sel.rel_nll, s=80, facecolors="none", edgecolors="k", zorder=5, label="selected")
        ax.set(title=f"N = {n} (n_disc = {int(sn.n_discovery.iloc[0])})", xlabel="m")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("held-out one-step NLL − best  (nats)")
    axes[0, -1].legend(fontsize=7)
    fig.tight_layout()
    return fig


def savings(agg, baseline):
    base = agg[agg.arm == baseline].sort_values("n_total")
    rows = []
    for _, r in agg[~agg.arm.isin([baseline, "prior"])].iterrows():
        row = dict(arm=r.arm, n_total=int(r.n_total))
        for col in ("posterior_kl", "crps"):
            n_eq, bound = equivalent_budget(base.n_total.values, base[f"{col}_mean"].values,
                                            r[f"{col}_mean"], higher_is_better=False)
            row[col] = f"{'' if bound == '=' else bound}{n_eq / r.n_total:.2f}"
        rows.append(row)
    return pd.DataFrame(rows)
