#!/usr/bin/env python
"""Compare theta|t and eta|t on the active (banana) plane (theta_1, theta_2) only.

The full-rank study grades the 8D posterior. Here every arm is re-trained exactly as in
``run_cell`` (same seeds, cached discovery maps) and its posterior draws are projected onto
(theta_1, theta_2), which is where the curved Rosenbrock link lives. Per test observation:

  exact_logp2d   exact 2D marginal log density at the truth (chain quadrature, mass_2d)
  kde_logq       log of a Gaussian KDE of the arm's 2D draws at the truth
  kde_logp       the same KDE estimator applied to exact 2D marginal draws (same n, same rule),
                 so  kde_logp - kde_logq  is a KDE-bias-matched excess on the active plane
  energy         2D energy distance between the arm's draws and exact draws (0 iff equal)

    python scripts/active_plane.py --results results/preview_seeds012_cpu --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde

from unified_sbi.exact import chain_posterior
from unified_sbi.study import StudyConfig, run_cell

ACTIVE = (0, 1)


def exact_plane(post, rng, n):
    """Exact 2D marginal of (theta_0, theta_1): log-density interpolator and n draws."""
    g, w = post.grid, post.weights
    m = post.mass_2d()[1, 0]                              # rows theta_0, columns theta_1
    dens = np.maximum(m / (w[:, None] * w[None, :]), 1e-300)
    interp = RegularGridInterpolator((g, g), np.log(dens), bounds_error=False, fill_value=None)
    a = post.sim.a_box
    edges = np.concatenate([[-a], 0.5 * (g[1:] + g[:-1]), [a]])
    p = (m / m.sum()).ravel()
    idx = rng.choice(p.size, size=n, p=p)
    i, j = np.unravel_index(idx, m.shape)
    x0 = rng.uniform(edges[i], edges[i + 1]); x1 = rng.uniform(edges[j], edges[j + 1])
    return interp, np.stack([x0, x1], -1)


def energy_distance(a, b):
    return 2 * cdist(a, b).mean() - cdist(a, a).mean() - cdist(b, b).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--budgets", type=int, nargs="+")
    ap.add_argument("--arms", nargs="+", default=["theta_t", "eta_t"])
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg_d = json.loads((a.results / "config.json").read_text())
    out = a.results / "active_plane"; out.mkdir(exist_ok=True)
    budgets = a.budgets or cfg_d["budgets"]
    for seed in a.seeds:
        for n in budgets:
            path = out / f"seed{seed}_n{n}.json"
            if path.exists():
                print("skip", path); continue
            cfg = StudyConfig(**{**cfg_d, "arms": list(a.arms), "seeds": [seed], "device": "cpu"})
            rows, art = run_cell(cfg, seed, n, a.results, verbose=True, return_artifacts=True)
            sim, th, xt = art["simulator"], art["theta_test"], art["x_test"]
            rng = np.random.default_rng(1234)
            exact = []
            for i in range(len(th)):
                interp, draws = exact_plane(chain_posterior(sim, xt[i], cfg.exact_grid), rng, cfg.n_post)
                exact.append((float(interp(th[i, list(ACTIVE)][None])[0]), draws))
            res = []
            for row in rows:
                arm = row["arm"]
                d2 = art["draws"][arm][:, :, list(ACTIVE)]
                kq, kp, en = [], [], []
                for i in range(len(th)):
                    t2 = th[i, list(ACTIVE)][:, None]
                    kq.append(float(np.log(max(gaussian_kde(d2[i].T)(t2)[0], 1e-300))))
                    kp.append(float(np.log(max(gaussian_kde(exact[i][1].T)(t2)[0], 1e-300))))
                    en.append(energy_distance(d2[i], exact[i][1]))
                lp_exact = np.array([e[0] for e in exact])
                kq, kp, en = map(np.array, (kq, kp, en))
                res.append(dict(seed=seed, n_total=n, arm=arm,
                                test_log_prob_8d=row["test_log_prob"],
                                exact_logp2d=float(lp_exact.mean()),
                                kde_logq2d=float(kq.mean()), kde_logp2d=float(kp.mean()),
                                excess2d_kde=float((kp - kq).mean()),
                                excess2d_kde_se=float((kp - kq).std(ddof=1) / np.sqrt(len(kq))),
                                energy2d=float(en.mean()), energy2d_se=float(en.std(ddof=1) / np.sqrt(len(en))),
                                crps_theta1=row["crps_per_dim"][0], crps_theta2=row["crps_per_dim"][1],
                                per_obs=dict(kde_logq=kq.tolist(), kde_logp=kp.tolist(), energy=en.tolist())))
                print(f"[seed {seed} | N={n}] {arm:8s} 2D excess(KDE) {res[-1]['excess2d_kde']:.3f}"
                      f" ± {res[-1]['excess2d_kde_se']:.3f} | energy {res[-1]['energy2d']:.4f}", flush=True)
            path.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
