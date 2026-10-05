#!/usr/bin/env python3
"""Does the learned eta(theta) recover the S8 combination?  (sigma8-coordinate runs, Gaussian field)

    python scripts/eta_s8_analysis.py --results results/shared_sigma8_gauss_seed0_cpu

For theta = (Omega_m, p2) the S8-like combination is  log p2 + alpha log Omega_m  with

    alpha(theta) = -C12 / C11,   C = (D F D)^-1,  D = diag(theta)       (log coordinates)

i.e. the alpha that minimises the variance of the combination under a Gaussian posterior with
precision F. Truth: F = exact Fisher matrix. Learned: F = J^T J of the arm's eta (the posterior
precision at the mode when t lies on eta(Theta), always for K = d). Reported per arm and budget:

* alpha_local: median and IQR over the 256 test parameters, and median |alpha_learned - alpha_exact|;
* alpha_pooled: -sum C12 / sum C11 over the test parameters (one global power law);
  its exact counterpart is alpha_exact_pooled; alpha_coord's is alpha_exact_grid_info_weighted, the grid
  average of the exact alpha weighted by the largest eigenvalue of the log-coordinate Fisher matrix;
* alpha_coord: a coordinate-level readout. s(theta) = e1 . eta(theta), with e1 the leading
  eigenvector of E_prior[J J^T], is regressed on a degree-5 polynomial of
  c_alpha = log p2 + alpha log Omega_m over a 48 x 48 interior grid; alpha_coord maximises R^2
  (a one-family symbolic regression). R^2 at alpha_coord, at the exact pooled alpha and at the
  conventional 0.5 are reported, plus R^2 of the second coordinate e2 . eta on log Omega_m.
* whiteness (K = d arms only): eigenvalues of Cov[eta(theta) | x] under the exact posterior for
  the maps cached by plot_posterior_grid.py (unit if eta is a flattening coordinate).
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

from unified_sbi.shared.lensing import NAMES
from unified_sbi.shared.study import Data, Pipeline, StudyConfig

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"hybrid_shared_K2": "#1baf7a", "hybrid_affine_shared_K2": "#eb6834", "hybrid_rect_K2": "#e34948",
       "hybrid_shared": "#2a78d6", "hybrid_manifold_shared": "#4a3aa7", "cnn_shared_K2": "#eda100", "cnn_shared": "#e87ba4"}
ETA_ARMS = list(COL)


def load_cfg(res):
    d = json.loads((Path(res) / "config.json").read_text())
    return StudyConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})


def alpha_from_F(F, theta):
    D = theta[:, :, None] * np.eye(2)[None]
    C = np.linalg.inv(D @ F @ D)
    return -C[:, 0, 1] / C[:, 0, 0], C


def eta_jac(head, theta):
    th = torch.as_tensor(theta, dtype=torch.float32)
    with torch.no_grad():
        e = head.eta(th).double().numpy()
    J = torch.func.vmap(torch.func.jacrev(lambda p: head.eta(p[None])[0]))(th).detach().double().numpy()
    return e, J                                           # (M, K), (M, K, 2)


def poly_r2(y, x, deg=5):
    x = (x - x.mean()) / x.std()
    X = np.vander(x, deg + 1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return 1 - r.var() / y.var()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--cache", default="cache/shared")
    ap.add_argument("--seed", type=int, default=0, help="seed shown in the level-set and scatter panels")
    ap.add_argument("--show-budget", type=int, default=4000)
    a = ap.parse_args()
    res = Path(a.results); figs = res / "figures"; figs.mkdir(exist_ok=True)
    cfg = load_cfg(res); cfg.device = "cpu"
    data = Data(cfg, a.cache)
    lo, hi = data.sim.lo, data.sim.hi
    if NAMES[data.lcfg.params][1] != "sigma8":
        print("note: params is not sigma8; alpha then refers to log S8 + alpha log Omega_m")
    th_t, _ = data.test()
    gaussian = data.lcfg.field == "gaussian"
    ref = "exact" if gaussian else "Gaussian-field Fisher (reference)"
    oracle = data.oracle()
    if oracle is not None:
        F_test = oracle["fisher"]
    else:          # non-Gaussian field: the Gaussian-field Fisher matrix (same two-point function) as a reference
        fpath = figs / "gaussian_fisher_test.npz"
        if fpath.exists():
            F_test = np.load(fpath)["F"]
        else:
            F_test = data.sim.exact_fisher(th_t); np.savez(fpath, F=F_test)
    a_ex, C_ex = alpha_from_F(F_test, th_t)
    a_ex_pool = -C_ex[:, 0, 1].sum() / C_ex[:, 0, 0].sum()

    ng = 48
    u = (np.arange(ng) + 0.5) / ng
    G = np.stack(np.meshgrid(*[lo[i] + (hi[i] - lo[i]) * u for i in range(2)], indexing="ij"), -1).reshape(-1, 2)
    gpath = figs / "exact_fisher_grid.npz"
    if gpath.exists():
        F_grid = np.load(gpath)["F"]
    else:
        F_grid = data.sim.exact_fisher(G); np.savez(gpath, F=F_grid, G=G)
    a_ex_grid, _ = alpha_from_F(F_grid, G)
    # information-weighted grid average: weight = largest eigenvalue of the log-coordinate Fisher matrix,
    # the counterpart of alpha_coord (the R^2 fit is dominated by where the leading coordinate varies fastest)
    Dg = G[:, :, None] * np.eye(2)[None]
    w_info = np.linalg.eigvalsh(Dg @ F_grid @ Dg)[:, -1]
    a_ex_info = float((a_ex_grid * w_info).sum() / w_info.sum())
    exact_cache = sorted(figs.glob("posterior_grid_exact_obs*.npz")) if gaussian else []

    rows, keep = [], {}
    cells = sorted({(int(p.parts[-4][4:]), int(p.parts[-3][1:]), p.parts[-2]) for p in res.glob("seed*/N*/*/model.pt")})
    budgets = sorted({n for _, n, _ in cells})
    for seed, n, arm in cells:
            if arm not in ETA_ARMS:
                continue
            p = res / f"seed{seed}" / f"N{n}" / arm / "model.pt"
            if not p.exists():
                continue
            m = Pipeline(arm, cfg, data.lcfg.n, data.sim.sigma_n)
            m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True)); m.eval()
            with torch.no_grad():
                F_l = m.head.fisher(torch.as_tensor(th_t, dtype=torch.float32)).double().numpy()
            a_l, C_l = alpha_from_F(F_l, th_t)
            e, J = eta_jac(m.head, G)
            M = np.einsum("mka,mla->kl", J, J) / len(G)
            w, V = np.linalg.eigh(M)
            s, s2 = e @ V[:, -1], e @ V[:, -2]
            lom, lp2 = np.log(G[:, 0]), np.log(G[:, 1])
            alphas = np.linspace(0.0, 1.5, 301)
            r2 = np.array([poly_r2(s, lp2 + al * lom) for al in alphas])
            row = dict(arm=arm, seed=seed, budget=n, K=m.head.K,
                       alpha_local_median=float(np.median(a_l)),
                       alpha_local_iqr=float(np.subtract(*np.percentile(a_l, [75, 25]))),
                       abs_err_alpha_median=float(np.median(np.abs(a_l - a_ex))),
                       alpha_pooled=float(-C_l[:, 0, 1].sum() / C_l[:, 0, 0].sum()),
                       alpha_coord=float(alphas[r2.argmax()]), r2_coord=float(r2.max()),
                       r2_at_exact_pooled=float(poly_r2(s, lp2 + a_ex_pool * lom)),
                       r2_at_half=float(poly_r2(s, lp2 + 0.5 * lom)),
                       r2_at_exact_info=float(poly_r2(s, lp2 + a_ex_info * lom)),
                       r2_second_vs_logOm=float(poly_r2(s2, lom)),
                       metric_eig_ratio=float(w[-1] / max(w[-2], 1e-12)))
            if m.head.K == 2 and exact_cache:
                z = np.load(exact_cache[0])
                eig = []
                for Gj, lpj in zip(z["Gs"], z["exact"]):
                    pj = np.exp(lpj - lpj.max()); pj /= pj.sum()
                    with torch.no_grad():
                        ej = m.head.eta(torch.as_tensor(Gj, dtype=torch.float32)).double().numpy()
                    mu = pj @ ej; Cj = (pj[:, None] * (ej - mu)).T @ (ej - mu)
                    eig.append(np.linalg.eigvalsh(Cj))
                row["whiteness_eig_median"] = [float(v) for v in np.nanmedian(np.array(eig), 0)]
            rows.append(row)
            keep[(arm, n, seed)] = dict(a_l=a_l, s=s, s2=s2, r2=r2, alphas=alphas)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(res / "eta_s8.csv", index=False)
    num = [c for c in df.columns if c not in ("arm", "seed", "budget", "K", "whiteness_eig_median")]
    agg = df.groupby(["arm", "budget"])[num].agg(["mean", "sem", "count"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.reset_index().to_csv(res / "eta_s8_by_budget.csv", index=False)
    summary = dict(reference=ref, alpha_exact_test_median=float(np.median(a_ex)),
                   alpha_exact_test_iqr=[float(np.percentile(a_ex, 25)), float(np.percentile(a_ex, 75))],
                   alpha_exact_pooled=float(a_ex_pool),
                   alpha_exact_grid_range=[float(a_ex_grid.min()), float(a_ex_grid.max())],
                   alpha_exact_grid_info_weighted=a_ex_info)
    (res / "eta_s8_exact.json").write_text(json.dumps(summary, indent=1))
    print(summary)

    # ---- figure
    nb = a.show_budget
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    g1, g2 = [lo[i] + (hi[i] - lo[i]) * u for i in range(2)]
    # exact local degeneracy direction field (tangent (Om, -alpha p2) in linear coordinates)
    cg = 12
    uc = (np.arange(cg) + 0.5) / cg
    Gc = np.stack(np.meshgrid(*[lo[i] + (hi[i] - lo[i]) * uc for i in range(2)], indexing="ij"), -1).reshape(-1, 2)
    idx = [np.argmin(((G - q) ** 2).sum(1)) for q in Gc]
    for ax, arm in zip(axes[:2], ["hybrid_shared_K2", "hybrid_affine_shared_K2"]):
        k = keep.get((arm, nb, a.seed))
        if k is None:
            ax.set_visible(False); continue
        ax.contour(g1, g2, k["s"].reshape(ng, ng).T, levels=14, colors=[COL[arm]], linewidths=1.3, linestyles="solid")
        for q, i in zip(Gc, idx):
            d = np.array([q[0], -a_ex_grid[i] * q[1]])
            d = d / np.hypot(d[0] / (hi[0] - lo[0]), d[1] / (hi[1] - lo[1])) * 0.03
            ax.plot([q[0] - d[0], q[0] + d[0]], [q[1] - d[1], q[1] + d[1]], color=INK, lw=1.4, solid_capstyle="round")
        ax.set_title(f"{arm}, N = {nb}: level sets of the leading η coordinate\n(black ticks: {'exact' if gaussian else 'Gaussian-Fisher'} local degeneracy direction)",
                     fontsize=9, color=INK)
        ax.set_xlabel(r"$\Omega_m$", color=INK); ax.set_ylabel(r"$\sigma_8$", color=INK)
    ax = axes[2]
    for arm in ["hybrid_affine_shared_K2", "hybrid_shared_K2"]:
        k = keep.get((arm, nb, a.seed))
        if k is not None:
            ax.scatter(a_ex, k["a_l"], s=10, color=COL[arm], alpha=0.6, label=arm, edgecolors="none")
    lim = [min(a_ex.min(), 0.3), max(a_ex.max(), 1.1)]
    ax.plot(lim, lim, color=MUTED, lw=1, ls=":")
    ax.set_xlabel(("exact" if gaussian else "Gaussian-Fisher") + r" local $\alpha(\theta^\ast)$", color=INK); ax.set_ylabel(r"learned local $\alpha(\theta^\ast)$ from $J^\top J$", color=INK)
    ax.set_title(f"S8 exponent at each test parameter, N = {nb}, seed {a.seed}", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[3]
    for arm in ETA_ARMS:
        d = df[df.arm == arm].groupby("budget").abs_err_alpha_median.agg(["mean", "sem", "count"]).reset_index()
        if len(d):
            ax.errorbar(d.budget, d["mean"], d["sem"].fillna(0), marker="o", color=COL[arm], lw=2, capsize=3,
                        label=arm + (f" ({int(d['count'].max())} seeds)" if d["count"].max() > 1 else ""), mec="white")
    ax.set_xscale("log"); ax.set_xticks(budgets, [str(b) for b in budgets])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("simulations N", color=INK)
    ax.set_ylabel(r"median $|\alpha_{\rm learned}-\alpha_{\rm %s}|$" % ("exact" if gaussian else "G"), color=INK)
    ax.set_title("error in the local S8 exponent (mean ± SE over seeds)", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle(f"Recovering the S8 combination from η(θ), {data.lcfg.field} field   ({ref}: median α = {np.median(a_ex):.2f}, pooled α = {a_ex_pool:.2f})", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figs / "eta_s8.png", dpi=150); plt.close(fig)
    print("wrote", figs / "eta_s8.png")


if __name__ == "__main__":
    main()
