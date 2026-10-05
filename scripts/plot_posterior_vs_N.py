#!/usr/bin/env python3
"""One held-out map, every simulation budget: how each method's posterior sharpens (or degrades) with N.

    python scripts/plot_posterior_vs_N.py --results results/shared_sigma8_gauss_seed0_cpu --obs 46 \
        --arms hybrid_maf hybrid_affine_shared_K2 hybrid_shared_K2 hybrid_rect_K2 cnn_maf cnn_shared_K2

One panel per arm. Grey fills: exact 68/95% regions (Gaussian field). Blue lines: the arm's 95% region at each
budget, light to dark with increasing N (single-hue ramp; line width also grows with N). Legend entries give the
per-map KL(p || q) on the window grid, so the numbers belong to this map only (the scaling plots average 256).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_posterior_grid import GRID, INK, MUTED, exact_alpha, hpd_levels, load_cfg, load_model, log_density  # noqa: E402
from unified_sbi.shared.lensing import NAMES  # noqa: E402
from unified_sbi.shared.study import Data  # noqa: E402

RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281", "#0d366b"]   # sequential blue, light -> dark
EXACT_FILL = ["#d9d8d4", "#b5b4af"]                                         # neutral greys: 95%, 68%


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--cache", default="cache/shared")
    ap.add_argument("--arms", nargs="+", required=True); ap.add_argument("--obs", type=int, default=46)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n-grid", type=int, default=160)
    ap.add_argument("--budgets", type=int, nargs="+", default=[250, 500, 1000, 2000, 4000])
    ap.add_argument("--out")
    a = ap.parse_args()
    res = Path(a.results); figs = res / "figures"; figs.mkdir(exist_ok=True, parents=True)
    cfg = load_cfg(res); cfg.device = "cpu"
    data = Data(cfg, a.cache)
    gaussian = data.lcfg.field == "gaussian"
    lo, hi = data.sim.lo, data.sim.hi
    names = NAMES[data.lcfg.params]
    th_t, x_t = data.test()
    j, t0 = a.obs, th_t[a.obs]
    x = x_t[j:j + 1]

    # window: exact 99.9% region (Gaussian) or the union of the largest-budget arms' regions, widened
    nc = 100
    g1, g2 = [lo[i] + (hi[i] - lo[i]) * (np.arange(nc) + 0.5) / nc for i in range(2)]
    Gc = np.stack(np.meshgrid(g1, g2, indexing="ij"), -1).reshape(-1, 2)
    if gaussian:
        stats = data.sim.sufficient_stats(x)[0]
        dens = [data.sim.loglik_from_stats(stats, Gc)]
    else:
        dens = [log_density(m, Gc, x) for m in (load_model(res, cfg, data, a.seed, max(a.budgets), arm) for arm in a.arms) if m is not None]
    w0, w1 = hi.copy(), lo.copy()
    for lp in dens:
        p, _ = hpd_levels(lp)
        srt = np.sort(p)[::-1]; sel = Gc[p >= srt[np.searchsorted(np.cumsum(srt), 0.999)]]
        w0, w1 = np.minimum(w0, sel.min(0)), np.maximum(w1, sel.max(0))
    pad = 0.7 * (w1 - w0)
    w0, w1 = np.maximum(lo, w0 - pad), np.minimum(hi, w1 + pad)
    ng = a.n_grid
    f1, f2 = [w0[i] + (w1[i] - w0[i]) * (np.arange(ng) + 0.5) / ng for i in range(2)]
    G = np.stack(np.meshgrid(f1, f2, indexing="ij"), -1).reshape(-1, 2)
    lp_ex = data.sim.loglik_from_stats(stats, G) if gaussian else None
    p_ex = None
    if gaussian:
        p_ex, lv_ex = hpd_levels(lp_ex)
    al = exact_alpha(data.sim.exact_fisher(t0[None])[0], t0)

    na = len(a.arms)
    ncol = 4 if na > 6 else min(na, 3)
    nrow = int(np.ceil(na / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 4.1 * nrow + 0.4), squeeze=False)
    for k, arm in enumerate(a.arms):
        ax = axes.flat[k]
        if gaussian:
            ax.contourf(f1, f2, p_ex.reshape(ng, ng).T, levels=[lv_ex[0], lv_ex[1], p_ex.max() * 1.01],
                        colors=EXACT_FILL)
        handles = []
        for b, n in enumerate(a.budgets):
            m = load_model(res, cfg, data, a.seed, n, arm)
            if m is None:
                continue
            lq = log_density(m, G, x)
            q, lvq = hpd_levels(lq)
            col, lw = RAMP[min(b, len(RAMP) - 1)], 1.0 + 0.45 * b
            ax.contour(f1, f2, q.reshape(ng, ng).T, levels=[lvq[0]], colors=[col], linewidths=lw)
            lab = f"N = {n}"
            if gaussian:
                kl = float(np.sum(p_ex * (np.log(np.clip(p_ex, 1e-300, None)) - np.log(np.clip(q, 1e-300, None)))))
                lab += f"   KL {kl:.2f}"
            handles.append(plt.Line2D([], [], color=col, lw=lw, label=lab))
        om = np.linspace(w0[0], w1[0], 200)
        ax.plot(om, t0[1] * (om / t0[0]) ** (-al), color=MUTED, lw=0.8, ls=":")
        ax.plot(*t0, marker="+", color=INK, ms=12, mew=2, zorder=5)
        ax.set_xlim(w0[0], w1[0]); ax.set_ylim(w0[1], w1[1])
        ax.set_title(arm, fontsize=10, color=INK)
        ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper right")
        ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel(r"$\Omega_m$", color=INK)
        ax.set_ylabel(r"$\sigma_8$" if names[1] == "sigma8" else r"$S_8$", color=INK)
    for ax in axes.flat[na:]:
        ax.set_visible(False)
    what = "grey: exact 68/95%; " if gaussian else ""
    fig.suptitle(f"Test map {j} (θ* = {t0[0]:.2f}, {t0[1]:.2f}), {data.lcfg.field} field, seed {a.seed}: "
                 f"95% regions by simulation budget ({what}blue: light → dark = more simulations)", color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(a.out) if a.out else figs / f"posterior_vs_N_obs{j}_seed{a.seed}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
