#!/usr/bin/env python3
"""Inspect trained shared-space models: posteriors vs exact, Fisher field, two-point emulator.

    python scripts/inspect_shared.py --results results/shared_gauss --budget 1000
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from unified_sbi.shared.lensing import HI, LO
from unified_sbi.shared.study import CatalogueData, Data, Pipeline, StudyConfig

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"exact": INK, "pk_shared": "#2a78d6", "hybrid_shared": "#1baf7a",
       "hybrid_split_shared": "#4a3aa7", "cnn_shared": "#eb6834", "hybrid_rect": "#e34948",
       "hybrid_manifold_shared": "#008300"}


def load_model(res, cfg, data, seed, budget, arm):
    p = Path(res) / f"seed{seed}" / f"N{budget}" / arm / "model.pt"
    if not p.exists():
        return None
    m = Pipeline(arm, cfg, data.lcfg.n, data.sim.sigma_n)
    m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
    return m.eval()


def hpd_levels(logp, masses=(0.68, 0.95)):
    p = np.exp(logp - logp.max()); p /= p.sum()
    s = np.sort(p.ravel())[::-1]; c = np.cumsum(s)
    return sorted(s[np.searchsorted(c, m)] for m in masses), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--cache", default="cache/shared")
    ap.add_argument("--budget", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obs", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--catalogue", help="path to the catalogue if it moved since the run")
    a = ap.parse_args()
    res = Path(a.results); figs = res / "figures"; figs.mkdir(exist_ok=True, parents=True)
    cfg = StudyConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                         for k, v in json.loads((res / "config.json").read_text()).items()})
    cfg.device = "cpu"
    dj = res / "data.json"
    if dj.exists():                      # a --catalogue run
        info = json.loads(dj.read_text())
        data = CatalogueData(cfg, a.catalogue or info["catalogue"], info["noise_amp"])
    else:
        data = Data(cfg, a.cache)
    th_t, x_t = data.test()
    gaussian = isinstance(data, Data) and data.lcfg.field == "gaussian"
    arms = [k for k in COL if k != "exact"]
    models = {arm: load_model(res, cfg, data, a.seed, a.budget, arm) for arm in arms}
    models = {k: v for k, v in models.items() if v is not None}
    T = lambda z: torch.as_tensor(np.asarray(z), dtype=torch.float32)
    report = {}

    # ---- 1. posterior contours for a few test observations
    fig, axes = plt.subplots(1, len(a.obs), figsize=(4.2 * len(a.obs), 4.2))
    stats = data.sim.sufficient_stats(x_t[a.obs]) if gaussian else None
    with torch.no_grad():
        ts = {arm: m.t(T(x_t[a.obs])) for arm, m in models.items()}
    for j, (ax, i) in enumerate(zip(np.atleast_1d(axes), a.obs)):
        c0 = th_t[i]
        half = np.array([0.25, 0.2])
        lo = np.maximum(LO, c0 - half); hi = np.minimum(HI, c0 + half)
        g1, g2 = np.linspace(lo[0], hi[0], 160), np.linspace(lo[1], hi[1], 160)
        G = np.stack(np.meshgrid(g1, g2, indexing="ij"), -1).reshape(-1, 2)
        curves = {}
        if gaussian:
            curves["exact"] = data.sim.loglik_from_stats(stats[j], G).reshape(160, 160)
        for arm, m in models.items():
            with torch.no_grad():
                curves[arm] = m.head.log_f(T(G)[None], ts[arm][j:j + 1])[0].numpy().reshape(160, 160)
        for name, lp in curves.items():
            lv, _ = hpd_levels(lp)
            ax.contour(g1, g2, np.exp(lp - lp.max()).T / np.exp(lp - lp.max()).sum(), levels=lv,
                       colors=[COL[name]], linewidths=2.2 if name == "exact" else 1.5,
                       linestyles="-" if name == "exact" else "--")
            ax.plot([], [], color=COL[name], lw=2, ls="-" if name == "exact" else "--", label=name)
        ax.plot(*c0, marker="+", color=INK, ms=12, mew=2)
        ax.set_xlabel(r"$\Omega_m$"); ax.set_ylabel(r"$S_8$")
        ax.grid(color=GRID, lw=0.8)
    np.atleast_1d(axes)[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"68/95% regions, N = {a.budget} simulations (local window)", color=INK)
    fig.tight_layout(); fig.savefig(figs / f"posteriors_N{a.budget}.png", dpi=150); plt.close(fig)

    # ---- 2a. learned J^T J vs exact Fisher at the test parameters (all shared/rect arms)
    if gaussian:
        Fx_t = data.sim.exact_fisher(th_t)
        for arm, m in models.items():
            Fl = m.head.fisher(T(th_t)).double().numpy()
            r = np.linalg.slogdet(Fl)[1] - np.linalg.slogdet(Fx_t)[1]
            report[f"fisher_logdet_ratio_test/{arm}"] = dict(median=float(np.median(r)),
                                                              q25=float(np.percentile(r, 25)), q75=float(np.percentile(r, 75)))
    # residual scale ||t - eta(theta*)||: ~sqrt(K) if t ~ N(eta, I)
    for arm, m in models.items():
        with torch.no_grad():
            r = (m.t(T(x_t)) - m.head.eta(T(th_t))).norm(dim=-1).numpy()
        report[f"residual_norm/{arm}"] = dict(median=float(np.median(r)), expected_if_gaussian=float(np.sqrt(m.head.K)))

    # ---- 2. learned Fisher field J^T J vs exact Fisher
    if gaussian:
        g = np.stack(np.meshgrid(np.linspace(0.2, 0.65, 6), np.linspace(0.45, 1.4, 6), indexing="ij"), -1).reshape(-1, 2)
        Fx = data.sim.exact_fisher(g)
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.2), squeeze=False)
        for ax, (arm, m) in zip(axes[0], models.items()):
            Fl = m.head.fisher(T(g)).double().numpy()
            ratio = np.linalg.slogdet(Fl)[1] - np.linalg.slogdet(Fx)[1]
            report[f"fisher_logdet_ratio_grid/{arm}"] = dict(median=float(np.median(ratio)),
                                                              iqr=[float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))])
            for k, (F, col, lw) in enumerate([(Fx, INK, 2.0), (Fl, COL[arm], 1.5)]):
                for p0, Fi in zip(g, F):
                    C = np.linalg.inv(Fi); w, V = np.linalg.eigh(C)
                    t = np.linspace(0, 2 * np.pi, 60)
                    e = (V * np.sqrt(np.clip(w, 0, None))) @ np.stack([np.cos(t), np.sin(t)])
                    ax.plot(p0[0] + 0.5 * e[0], p0[1] + 0.5 * e[1], color=col, lw=lw,
                            ls="-" if k == 0 else "--")
            ax.set_title(f"{arm}: median log det ratio {np.median(ratio):+.2f}", fontsize=9)
            ax.set_xlabel(r"$\Omega_m$"); ax.set_ylabel(r"$S_8$"); ax.grid(color=GRID, lw=0.8)
        fig.suptitle("Fisher ellipses (half 1σ): exact (solid) vs learned $J^\\top J$ (dashed)", color=INK)
        fig.tight_layout(); fig.savefig(figs / f"fisher_N{a.budget}.png", dpi=150); plt.close(fig)

    # ---- 3. the two-point emulator eta_P(theta) of the split hybrid
    m = models.get("hybrid_split_shared")
    if m is not None:
        with torch.no_grad():
            t = m.t(T(x_t)).numpy(); eta = m.head.eta(T(th_t)).numpy()
        kp = m.summary.k_pk
        r = t - eta
        tot = t.var(0); res_var = r.var(0)
        report["emulator"] = dict(
            explained_variance_pk_block=[float(1 - res_var[k] / tot[k]) for k in range(kp)],
            explained_variance_cnn_block=[float(1 - res_var[k] / tot[k]) for k in range(kp, t.shape[1])],
            residual_cov=np.cov(r.T).round(3).tolist(),
            note="t ~ N(eta(theta), I) predicts residual covariance = identity")
        fig, axes = plt.subplots(1, t.shape[1], figsize=(3.6 * t.shape[1], 3.4))
        for k, ax in enumerate(axes):
            ax.scatter(eta[:, k], t[:, k], s=8, color=COL["hybrid_split_shared"] if k < kp else COL["cnn_shared"], alpha=0.6)
            lim = [min(eta[:, k].min(), t[:, k].min()), max(eta[:, k].max(), t[:, k].max())]
            ax.plot(lim, lim, color=MUTED, lw=1, ls=":")
            ax.set_xlabel(rf"$\eta_{k + 1}(\theta^\ast)$"); ax.set_ylabel(rf"$t_{k + 1}(x)$")
            ax.set_title(("W·Pk block" if k < kp else "CNN block") + f"  R²={1 - res_var[k] / tot[k]:.2f}", fontsize=9)
            ax.grid(color=GRID, lw=0.8)
        fig.suptitle("Shared space of hybrid_split_shared: parameter embedding vs data summary (test set)", color=INK)
        fig.tight_layout(); fig.savefig(figs / f"emulator_N{a.budget}.png", dpi=150); plt.close(fig)

        # information decomposition from one model: q_P (Pk block only) vs q_{P+N}
        with torch.no_grad():
            tt = m.t(T(x_t)); full, _ = m.head.evaluate(T(th_t), tt)
            head = m.head
            orig_K = head.K

            def lf_pk(p, t_):
                z = head.eta(p)[..., :kp]
                return -0.5 * (z - t_[:, None, :kp]).square().sum(-1)
            from unified_sbi.shared.quadrature import integrate
            c = integrate(lambda p: lf_pk(p, tt), head.lo, head.hi, len(tt), **head.quad_eval)
            lq_pk = lf_pk(T(th_t)[:, None], tt)[:, 0] - c.logZ
        report["information_decomposition_nats"] = dict(
            E_log_q_full=float(full["log_q"].mean()), E_log_q_pk_block=float(lq_pk.mean()),
            gain_beyond_pk=float((full["log_q"] - lq_pk).mean()))
    (figs / f"inspect_N{a.budget}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
