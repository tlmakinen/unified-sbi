#!/usr/bin/env python3
"""Figures for the shear-calibration nuisance study (scripts/nuisance_m_study.py).

    python scripts/plot_nuisance.py --results results/nuisance_m_gauss_cpu --budget 4000 --obs 138

1. scaling: excess KL (3D joint) against N, and the posterior std of m and sigma8 against the exact ones;
2. one test map: marginals of sigma8 and m and the (sigma8, m) joint, exact vs each arm.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from unified_sbi.shared.lensing import LensingConfig
from unified_sbi.shared.nuisance import ARMS, NuisanceData, NuisPipeline
from unified_sbi.shared.study import StudyConfig

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"hybrid_maf": "#2a78d6", "hybrid_shared_K2": "#008300", "hybrid_rect_K2": "#e34948", "hybrid_rect_sq": "#eb6834"}
LS = {"hybrid_maf": "--", "hybrid_shared_K2": "-", "hybrid_rect_K2": ":", "hybrid_rect_sq": "-."}
LAB = {"hybrid_maf": "MAF (NPE)", "hybrid_shared_K2": "normalised shared, prior base (K = 2)",
       "hybrid_rect_K2": "rectangular, vol J = √det(JJᵀ) (K = 2 < d)", "hybrid_rect_sq": "square-completed rectangular (K = d = 3)"}


def style(ax):
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def load_data(res, cache):
    c = json.loads((Path(res) / "config.json").read_text())
    m_sigma = c.pop("m_sigma"); c.pop("run_budgets", None); c.pop("run_arms", None)
    cfg = StudyConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in c.items()})
    cfg.device = "cpu"
    return cfg, NuisanceData(cfg, cache, m_sigma=m_sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--cache", default="cache/shared")
    ap.add_argument("--budget", type=int, default=4000); ap.add_argument("--obs", type=int, nargs="+", default=[138, 46])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    res = Path(a.results); figs = res / "figures"; figs.mkdir(exist_ok=True)
    cfg, data = load_data(res, a.cache)
    rows = []
    for p in res.glob("seed*/N*/*/metrics.json"):
        m = json.loads(p.read_text())
        rows.append(dict(arm=m["arm"], seed=m["seed"], budget=m["budget"], excess=m["excess_nll"], se=m["excess_nll_se"],
                         jcov=m["joint_cov_mae"], std_om=m["post_std"][0], std_s8=m["post_std"][1], std_m=m["post_std"][2]))
    df = pd.DataFrame(rows).sort_values(["arm", "budget"])
    df.to_csv(res / "metrics.csv", index=False)
    o = data.oracle()
    ex_std = o["std"].mean(0)
    print(df.round(4).to_string(index=False)); print("exact mean posterior std (Om, s8, m):", ex_std.round(4))

    # ---- 1. scaling
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for arm in ARMS:
        d = df[df.arm == arm]
        if d.empty:
            continue
        kw = dict(color=COL[arm], ls=LS[arm], marker="o", lw=2, ms=6, mec="white", label=LAB[arm])
        axes[0].plot(d.budget, d.excess, **kw)
        axes[1].plot(d.budget, d.std_s8, **kw)
        axes[2].plot(d.budget, d.std_m, **kw)
    axes[1].axhline(ex_std[1], color=INK, lw=1.2, label="exact"); axes[2].axhline(ex_std[2], color=INK, lw=1.2, label="exact")
    axes[2].axhline(data.m_sigma * np.sqrt(3), color=MUTED, lw=1, ls=":", label="flat on ±3σ_m (std 0.087)")
    axes[0].set_yscale("log"); axes[0].set_ylabel("excess NLL = E KL(p || q) [nats], 3D", color=INK)
    axes[1].set_ylabel(r"mean posterior std of $\sigma_8$", color=INK)
    axes[2].set_ylabel(r"mean posterior std of $m$", color=INK)
    for ax in axes:
        ax.set_xscale("log"); b = sorted(df.budget.unique()); ax.set_xticks(b, [str(x) for x in b])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.set_xlabel("simulations N", color=INK); style(ax)
    axes[2].legend(frameon=False, fontsize=7.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle(f"Shear-calibration nuisance m ~ N(0, {data.m_sigma:g}²), θ = ($\\Omega_m$, $\\sigma_8$, m), Gaussian field, seed {a.seed}", color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95)); fig.savefig(figs / "nuisance_scaling.png", dpi=160); plt.close(fig)

    # ---- 2. one map: marginals
    th_t, x_t = data.test()
    n = (72, 72, 40)
    logpost, P, cv, axes_g = data.grid_logpost(n)          # points ordered (m, Omega_m, sigma8)
    shape = (n[2], n[0], n[1])
    stats = data.sim.sufficient_stats(x_t[a.obs])
    models = {}
    for arm in ARMS:
        p = res / f"seed{a.seed}" / f"N{a.budget}" / arm / "model.pt"
        if p.exists():
            m = NuisPipeline(arm, cfg, data); m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
            models[arm] = m.eval()
    T = lambda z: torch.as_tensor(np.asarray(z), dtype=torch.float32)
    fig, axs = plt.subplots(len(a.obs), 3, figsize=(14, 4.0 * len(a.obs)), squeeze=False)
    for r, j in enumerate(a.obs):
        lp_ex = logpost(stats[r])
        dens = {"exact": lp_ex}
        with torch.no_grad():
            for arm, m in models.items():
                t = m.t(T(x_t[j:j + 1]))
                if arm == "hybrid_maf":
                    dens[arm] = torch.cat([m.head.log_prob(T(P[i:i + 20000]), t.expand(len(P[i:i + 20000]), -1))
                                           for i in range(0, len(P), 20000)]).double().numpy()
                else:
                    E = torch.cat([m.head.eta(T(P[i:i + 20000])) for i in range(0, len(P), 20000)]).double()
                    X = torch.cat([m.head.extra(T(P[i:i + 20000])) for i in range(0, len(P), 20000)]).double()
                    dens[arm] = (-0.5 * (m.head.tt(t).double()[0] - E).square().sum(-1) + X).numpy()
        for name, lp in dens.items():
            p = np.exp(lp - lp.max()); p = (p / p.sum()).reshape(shape)
            col, ls, lw = (INK, "-", 2.2) if name == "exact" else (COL[name], LS[name], 1.7)
            ps8 = p.sum((0, 1)); pm = p.sum((1, 2))
            axs[r, 0].plot(axes_g[1], ps8 / ps8.max(), color=col, ls=ls, lw=lw, label=name if name == "exact" else LAB[name])
            axs[r, 1].plot(axes_g[2], pm / pm.max(), color=col, ls=ls, lw=lw)
            j2 = p.sum(1)                                                     # (m, sigma8)
            srt = np.sort(j2.ravel())[::-1]; lv = srt[np.searchsorted(np.cumsum(srt), 0.95)]
            axs[r, 2].contour(axes_g[1], axes_g[2], j2, levels=[lv], colors=[col], linestyles=[ls], linewidths=lw)
        pr = np.exp(-0.5 * (axes_g[2] / data.m_sigma) ** 2)
        axs[r, 1].plot(axes_g[2], pr / pr.max(), color=MUTED, lw=1, ls=(0, (1, 2)), label="prior on m")
        for c, (xl, v) in enumerate([(r"$\sigma_8$", th_t[j, 1]), (r"$m$", th_t[j, 2])]):
            axs[r, c].axvline(v, color=MUTED, lw=0.8); axs[r, c].set_xlabel(xl, color=INK)
            axs[r, c].set_ylabel("marginal (peak = 1)", color=INK)
        axs[r, 2].plot(th_t[j, 1], th_t[j, 2], "+", color=INK, ms=11, mew=2)
        axs[r, 2].set_xlabel(r"$\sigma_8$", color=INK); axs[r, 2].set_ylabel(r"$m$", color=INK)
        axs[r, 0].set_title(f"test map {j}: θ* = ({th_t[j, 0]:.2f}, {th_t[j, 1]:.2f}, {th_t[j, 2]:+.3f})", fontsize=9, color=MUTED)
        axs[r, 2].set_title("95% region of the (σ₈, m) marginal", fontsize=9, color=MUTED)
        for ax in axs[r]:
            style(ax)
    axs[0, 0].legend(frameon=False, fontsize=7.5); axs[0, 1].legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"Marginal posteriors at N = {a.budget}: does each loss recover the prior on m and the right σ₈ width?", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(figs / f"nuisance_marginals_N{a.budget}.png", dpi=150); plt.close(fig)
    print("wrote", figs)


if __name__ == "__main__":
    main()
