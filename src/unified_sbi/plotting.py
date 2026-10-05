"""Figures: scaling curves, coverage curves and an exact-reference corner plot."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .analysis import power_law_fit

ARM_STYLE = {
    "raw":     dict(color="#6b6b6b", marker="o", label="NPE on raw x (uncompressed)"),
    "xbar":    dict(color="#9467bd", marker="v", label="NPE on replicate means"),
    "theta_t": dict(color="#1f77b4", marker="s", label=r"NPE $\theta\mid t$ (ours)"),
    "eta_t":   dict(color="#d62728", marker="D", label=r"NPE $\eta\mid t$ (ours)"),
}


def _style(arm):
    return ARM_STYLE.get(arm, dict(marker="o", label=arm))


def _ordered_groups(agg):
    """groupby('arm') in the fixed ARM_STYLE order, so colours and legends are stable."""
    order = {a: i for i, a in enumerate(ARM_STYLE)}
    return sorted(agg.groupby("arm"), key=lambda kv: order.get(kv[0], len(order)))


def plot_scaling(agg, oracle: dict | None = None, n_discovery: int | None = None):
    """Validation log q, test log q and test CRPS against total simulation budget."""
    panels = [("val_log_prob", r"end-of-training validation $\log q(\theta\mid x)$", None),
              ("test_log_prob", r"test $\log q(\theta^\ast\mid x)$", "oracle_log_prob"),
              ("crps", "test CRPS (mean over coordinates)", "oracle_crps")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (col, ylabel, oracle_key) in zip(axes, panels):
        for arm, g in _ordered_groups(agg):
            g = g.sort_values("n_total")
            st = _style(arm)
            ax.errorbar(g.n_total, g[f"{col}_mean"], yerr=g[f"{col}_sem"], capsize=3, lw=1.6,
                        ms=5, color=st.get("color"), marker=st["marker"], label=st["label"])
        if oracle and oracle_key:
            value = oracle["test_log_prob"] if oracle_key == "oracle_log_prob" else oracle["crps"]
            ax.axhline(value, color="k", ls="--", lw=1, label="exact posterior")
        if n_discovery:
            ax.axvline(n_discovery, color="0.7", ls=":", lw=1)
        ax.set(xscale="log", xlabel="total simulations $N_{\\rm total}$", ylabel=ylabel)
        ax.grid(alpha=0.3, which="both")
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_excess(agg, min_budget: int = 0):
    """Log-log excess over the exact posterior, with power-law fits N^-alpha."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, (col, ylabel) in zip(axes, [("excess_crps", "CRPS − exact CRPS"),
                                         ("excess_nll", r"exact $\log p$ − test $\log q$  (nats)")]):
        for arm, g in _ordered_groups(agg):
            g = g.sort_values("n_total")
            st = _style(arm)
            y = g[f"{col}_mean"].values
            ax.errorbar(g.n_total, y, yerr=g[f"{col}_sem"], capsize=3, lw=0, elinewidth=1.2, ms=5,
                        color=st.get("color"), marker=st["marker"])
            fit = g[g.n_total >= min_budget]
            a, alpha = power_law_fit(fit.n_total.values.astype(float), fit[f"{col}_mean"].values)
            if np.isfinite(alpha):
                nn = np.geomspace(fit.n_total.min(), fit.n_total.max(), 50)
                ax.plot(nn, a * nn ** (-alpha), color=st.get("color"), lw=1.4,
                        label=f"{st['label']}: α = {alpha:.2f}")
        ax.set(xscale="log", yscale="log", xlabel="total simulations $N_{\\rm total}$", ylabel=ylabel)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_coverage(samples_by_name: dict, truth: np.ndarray, labels=None):
    levels = np.linspace(0.05, 0.95, 19)
    dim = truth.shape[1]
    labels = labels or [fr"$\theta_{j+1}$" for j in range(dim)]
    fig, axes = plt.subplots(1, dim, figsize=(2.2 * dim, 2.6), sharex=True, sharey=True)
    for name, draws in samples_by_name.items():
        st = _style(name)
        curve = []
        for level in levels:
            lo, hi = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2], axis=1)
            curve.append(((truth >= lo) & (truth <= hi)).mean(0))
        curve = np.array(curve)
        for j, ax in enumerate(axes):
            ax.plot(levels, curve[:, j], color=st.get("color"), label=st["label"])
    band = 1.96 * np.sqrt(levels * (1 - levels) / len(truth))
    for j, ax in enumerate(axes):
        ax.fill_between(levels, levels - band, levels + band, color="k", alpha=0.08)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(title=labels[j], xlim=(0, 1), ylim=(0, 1), xlabel="nominal")
    axes[0].set_ylabel("empirical coverage")
    axes[-1].legend(fontsize=6)
    fig.tight_layout()
    return fig


def hpd_thresholds(density, mass, probabilities=(0.68, 0.95)):
    order = np.argsort(density.ravel())[::-1]
    cumulative = np.cumsum(mass.ravel()[order])
    cumulative /= cumulative[-1]
    return np.sort(density.ravel()[order[np.searchsorted(cumulative, probabilities)]])


def corner_exact(post, draws_by_name: dict, truth: np.ndarray, title: str = "", max_points: int = 1000):
    """Exact quadrature marginals (grey HPD regions) with learned posterior draws overlaid."""
    d, grid, w = post.sim.dim, post.grid, post.weights
    dens1, mass2 = post.density_1d, post.mass_2d()
    limits = []
    for j in range(d):
        cdf = np.cumsum(post.mass_1d[j])
        lo, hi = np.interp([0.001, 0.999], cdf / cdf[-1], grid)
        pooled = np.concatenate([s[:, j] for s in draws_by_name.values()])
        slo, shi = np.quantile(pooled, [0.001, 0.999])
        lo, hi = min(lo, slo), max(hi, shi)
        pad = 0.1 * max(hi - lo, 1e-3)
        limits.append((max(-post.sim.a_box, lo - pad), min(post.sim.a_box, hi + pad)))
    fig, axes = plt.subplots(d, d, figsize=(2.1 * d, 2.1 * d))
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.fill_between(grid, dens1[j], color="0.82")
                ax.plot(grid, dens1[j], color="0.25", lw=1.4)
                bins = np.linspace(*limits[j], 45)
                for name, s in draws_by_name.items():
                    counts, edges = np.histogram(s[:, j], bins=bins)
                    ax.stairs(counts / (len(s) * np.diff(edges)), edges, color=_style(name).get("color"), lw=1.2)
                ax.axvline(truth[j], color="k", ls=":", lw=1)
            else:
                mass = mass2[i, j]
                density = mass / (w[:, None] * w[None, :])
                th = hpd_thresholds(density, mass)
                ax.contourf(grid, grid, density.T, levels=[*th, density.max() * (1 + 1e-8)],
                            colors=["0.90", "0.70"])
                ax.contour(grid, grid, density.T, levels=th, colors="0.35", linewidths=0.7)
                for name, s in draws_by_name.items():
                    ax.scatter(s[:max_points, j], s[:max_points, i], s=3, alpha=0.15,
                               color=_style(name).get("color"), rasterized=True)
                ax.set_ylim(limits[i])
            ax.set_xlim(limits[j])
            ax.tick_params(labelsize=6)
            if i == d - 1:
                ax.set_xlabel(fr"$\theta_{j+1}$")
            if j == 0 and i > 0:
                ax.set_ylabel(fr"$\theta_{i+1}$")
    handles = [Patch(facecolor="0.70", label="exact 68%"), Patch(facecolor="0.90", label="exact 95%"),
               *[Line2D([], [], color=_style(n).get("color"), label=_style(n)["label"]) for n in draws_by_name]]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, 0.95))
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig
