#!/usr/bin/env python3
"""Posterior contours of every arm against the exact posterior: rows = budgets, columns = test maps.

    python scripts/plot_posterior_grid.py --results results/shared_sigma8_gauss_seed0_cpu \
        --groups "hybrid_maf,hybrid_affine_shared_K2,hybrid_shared_K2" \
                 "hybrid_rect_K2,hybrid_shared,hybrid_manifold_shared"

Gaussian field: the exact posterior is drawn. Other fields: no exact contour; the window is the union of
the arms' 99.9% regions at the largest budget, and the dotted curve uses the Gaussian-field Fisher matrix. Each column uses one window for every
budget: the exact 99.9% HPD bounding box, widened by 60% on each side and clipped to the prior box.
HPD levels are computed from the density normalised on that window. A dotted grey curve marks the
exact local degeneracy through theta*: p2 * Omega_m^alpha = const, with alpha = -C12/C11 of the inverse
exact Fisher matrix at theta* in log coordinates (for params="sigma8" this is the S8-like combination).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from unified_sbi.shared.lensing import NAMES
from unified_sbi.shared.study import Data, Pipeline, StudyConfig

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a"]          # first three categorical slots (all-pairs safe)
DASH = ["--", (0, (5, 1.5, 1, 1.5)), (0, (1.2, 1.2))]


def load_cfg(res):
    d = json.loads((Path(res) / "config.json").read_text())
    return StudyConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})


def load_model(res, cfg, data, seed, budget, arm):
    p = Path(res) / f"seed{seed}" / f"N{budget}" / arm / "model.pt"
    if not p.exists():
        return None
    m = Pipeline(arm, cfg, data.lcfg.n, data.sim.sigma_n)
    m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
    return m.eval()


def log_density(model, G, x):
    """Unnormalised log q(theta | x) at grid points G (M, 2) for one map x (1, 4, n, n)."""
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
    with torch.no_grad():
        t = model.t(T(x))
        if model.spec["head"] == "maf":
            return model.head.log_prob(T(G), t.expand(len(G), -1)).double().numpy()
        return model.head.log_f(T(G)[None], t)[0].double().numpy()


def hpd_levels(lp, masses=(0.68, 0.95)):
    p = np.exp(lp - lp.max()); p /= p.sum()
    s = np.sort(p.ravel())[::-1]; c = np.cumsum(s)
    return p, sorted(s[min(np.searchsorted(c, m), len(s) - 1)] for m in masses)


def exact_alpha(fisher, theta):
    """alpha minimising the Fisher-forecast variance of  log p2 + alpha log Omega_m  (= -C12 / C11 in log coords)."""
    D = np.diag(theta)
    C = np.linalg.inv(D @ fisher @ D)
    return -C[0, 1] / C[0, 0]


def reference_windows(a, res, cfg, data, obs, x_t, lo, hi, nc=100):
    """No exact posterior: window = union of the 99.9% regions of every plotted arm at the largest budget."""
    arms = sorted({arm for g in a.groups for arm in g.split(",")})
    g1, g2 = [lo[i] + (hi[i] - lo[i]) * (np.arange(nc) + 0.5) / nc for i in range(2)]
    Gc = np.stack(np.meshgrid(g1, g2, indexing="ij"), -1).reshape(-1, 2)
    models = [m for m in (load_model(res, cfg, data, a.seed, max(a.budgets), arm) for arm in arms) if m is not None]
    wins, Gs = [], []
    for j in range(len(obs)):
        w0, w1 = hi.copy(), lo.copy()
        for m in models:
            p, _ = hpd_levels(log_density(m, Gc, x_t[obs[j]:obs[j] + 1]))
            srt = np.sort(p)[::-1]; thr = srt[np.searchsorted(np.cumsum(srt), 0.999)]
            sel = Gc[p >= thr]
            w0, w1 = np.minimum(w0, sel.min(0)), np.maximum(w1, sel.max(0))
        pad = 0.3 * (w1 - w0)
        w0, w1 = np.maximum(lo, w0 - pad), np.minimum(hi, w1 + pad)
        f1, f2 = [w0[i] + (w1[i] - w0[i]) * (np.arange(a.n_grid) + 0.5) / a.n_grid for i in range(2)]
        wins.append(np.stack([w0, w1])); Gs.append(np.stack(np.meshgrid(f1, f2, indexing="ij"), -1).reshape(-1, 2))
    return np.array(wins), np.array(Gs), None


def choose_obs(th, lo, hi, k=4):
    mid = (th[:, 1] > lo[1] + 0.25 * (hi[1] - lo[1])) & (th[:, 1] < hi[1] - 0.25 * (hi[1] - lo[1]))
    idx = np.where(mid)[0]
    order = idx[np.argsort(th[idx, 0])]
    return [int(order[int(q * (len(order) - 1))]) for q in np.linspace(0.12, 0.88, k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--cache", default="cache/shared")
    ap.add_argument("--groups", nargs="+", required=True, help="comma-separated arms per figure (<= 3 each)")
    ap.add_argument("--budgets", type=int, nargs="+", default=[250, 1000, 4000])
    ap.add_argument("--obs", type=int, nargs="+"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-grid", type=int, default=150)
    a = ap.parse_args()
    res = Path(a.results); figs = res / "figures"; figs.mkdir(exist_ok=True, parents=True)
    cfg = load_cfg(res); cfg.device = "cpu"
    data = Data(cfg, a.cache)
    gaussian = data.lcfg.field == "gaussian"   # exact posterior available only for the Gaussian field
    lo, hi = data.sim.lo, data.sim.hi
    names = NAMES[data.lcfg.params]
    th_t, x_t = data.test()
    obs = a.obs or choose_obs(th_t, lo, hi)
    F = data.sim.exact_fisher(th_t[obs])        # Gaussian-field Fisher (a reference for non-Gaussian fields)

    # windows from the exact posterior, then exact log density on the fine window grid
    cache = figs / f"posterior_grid_exact_obs{'-'.join(map(str, obs))}.npz"
    if not gaussian:
        wins, Gs, exact = reference_windows(a, res, cfg, data, obs, x_t, lo, hi)
    elif cache.exists():
        z = np.load(cache); wins, Gs, exact = z["wins"], z["Gs"], z["exact"]
    else:
        stats = data.sim.sufficient_stats(x_t[obs])
        nc = 100
        g1, g2 = [lo[i] + (hi[i] - lo[i]) * (np.arange(nc) + 0.5) / nc for i in range(2)]
        Gc = np.stack(np.meshgrid(g1, g2, indexing="ij"), -1).reshape(-1, 2)
        wins, Gs, exact = [], [], []
        for j in range(len(obs)):
            p, _ = hpd_levels(data.sim.loglik_from_stats(stats[j], Gc))
            s = np.sort(p)[::-1]; thr = s[np.searchsorted(np.cumsum(s), 0.999)]
            sel = Gc[p >= thr]
            w0, w1 = sel.min(0) - 0.5 * (hi - lo) / nc, sel.max(0) + 0.5 * (hi - lo) / nc
            pad = 0.6 * (w1 - w0)
            w0, w1 = np.maximum(lo, w0 - pad), np.minimum(hi, w1 + pad)
            f1, f2 = [w0[i] + (w1[i] - w0[i]) * (np.arange(a.n_grid) + 0.5) / a.n_grid for i in range(2)]   # cell centres
            G = np.stack(np.meshgrid(f1, f2, indexing="ij"), -1).reshape(-1, 2)
            wins.append(np.stack([w0, w1])); Gs.append(G)
            exact.append(data.sim.loglik_from_stats(stats[j], G))
            print(f"exact obs {obs[j]} done", flush=True)
        wins, Gs, exact = np.array(wins), np.array(Gs), np.array(exact)
        np.savez(cache, wins=wins, Gs=Gs, exact=exact, obs=np.array(obs))

    for gi, group in enumerate(a.groups):
        arms = group.split(",")
        nb, no = len(a.budgets), len(obs)
        fig, axes = plt.subplots(nb, no, figsize=(3.5 * no, 3.2 * nb + 0.6), squeeze=False)
        for r, n in enumerate(a.budgets):
            models = {arm: load_model(res, cfg, data, a.seed, n, arm) for arm in arms}
            for c, j in enumerate(range(no)):
                ax = axes[r, c]
                G = Gs[j]; ng = a.n_grid
                f1, f2 = G.reshape(ng, ng, 2)[:, 0, 0], G.reshape(ng, ng, 2)[0, :, 1]
                curves = [("exact", exact[j], INK, "-", 2.2)] if gaussian else []
                for k, arm in enumerate(arms):
                    if models[arm] is not None:
                        curves.append((arm, log_density(models[arm], G, x_t[obs[j]:obs[j] + 1]), SLOTS[k], DASH[k], 1.6))
                for name, lp, col, ls, lw in curves:
                    p, lv = hpd_levels(lp)
                    ax.contour(f1, f2, p.reshape(ng, ng).T, levels=lv, colors=[col], linewidths=lw, linestyles=[ls])
                # exact local degeneracy through theta*
                t0 = th_t[obs[j]]; al = exact_alpha(F[j], t0)
                om = np.linspace(wins[j, 0, 0], wins[j, 1, 0], 200)
                ax.plot(om, t0[1] * (om / t0[0]) ** (-al), color=MUTED, lw=0.9, ls=":")
                ax.plot(*t0, marker="+", color=INK, ms=11, mew=2)
                ax.set_xlim(wins[j, 0, 0], wins[j, 1, 0]); ax.set_ylim(wins[j, 0, 1], wins[j, 1, 1])
                ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                ax.tick_params(colors=MUTED, labelsize=8)
                if r == nb - 1:
                    ax.set_xlabel(r"$\Omega_m$", color=INK)
                if c == 0:
                    ax.set_ylabel((r"$\sigma_8$" if names[1] == "sigma8" else r"$S_8$") + f"\nN = {n}", color=INK)
                if r == 0:
                    ax.set_title(f"test map {obs[j]}   (α{'*' if gaussian else '_G'} = {al:.2f})", fontsize=9, color=MUTED)
        handles = [plt.Line2D([], [], color=INK, lw=2.2, label="exact")] if gaussian else []
        handles += [plt.Line2D([], [], color=SLOTS[k], lw=1.8, ls=DASH[k], label=arm) for k, arm in enumerate(arms)]
        handles += [plt.Line2D([], [], color=MUTED, lw=0.9, ls=":", label=("exact" if gaussian else "Gaussian-field Fisher") + " local degeneracy through θ*")]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
        fig.suptitle(f"68/95% posterior regions, {data.lcfg.field} field in ({names[0]}, {names[1]}), seed {a.seed}", color=INK)
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        out = figs / f"posterior_grid_{gi}.png"
        fig.savefig(out, dpi=150); plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
