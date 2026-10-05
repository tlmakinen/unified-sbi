"""Posterior diagnostics against the exact reference (ported from the rank-2 notebook).

* posterior KL: mean over test pairs of log p(theta* | x) - log q(theta* | x), an unbiased
  Monte Carlo estimate of E_x KL(p || q). It can be negative at finite test size; never clipped.
* CRPS per native theta coordinate against theta*, and the same for the reference draws.
* sliced Wasserstein distance to reference draws, in native theta and in the true active plane.
* nuisance checks: learned posteriors in the true inactive coordinates u_3..u_D should stay N(0, 1).
* random-reference ranks (TARP-style) in the true active plane.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

from ..metrics import central_coverage, crps_ensemble

LEVELS = np.linspace(0.1, 0.95, 10)


def sliced_wasserstein(p: np.ndarray, q: np.ndarray, directions: np.ndarray) -> float:
    pp, qq = p @ directions.T, q @ directions.T
    return float(np.mean([wasserstein_distance(pp[:, j], qq[:, j]) for j in range(len(directions))]))


def random_reference_ranks(samples, truth, anchors, rng):
    values = []
    for draws, th, aa in zip(samples, truth, anchors):
        ds = np.sum((draws[:, None, :] - aa[None]) ** 2, -1)
        dt = np.sum((th[None] - aa) ** 2, -1)
        values.append(((ds < dt).sum(0) + rng.random(len(aa))) / (len(draws) + 1))
    return np.array(values)


def evaluate(samples: np.ndarray, log_q: np.ndarray, ref, n_shape: int = 64, n_projections: int = 32,
             n_anchors: int = 4, seed: int = 9010) -> dict:
    """samples (B, S, D) in theta; log_q (B,) at theta*; ref: ExactReference."""
    sim = ref.sim
    rng = np.random.default_rng(seed)
    delta = ref.true_log_prob - log_q
    crps = crps_ensemble(samples, ref.theta)
    u = samples @ sim.Q.T
    u_true = sim.to_active(ref.theta)
    u_ref = ref.samples @ sim.Q.T
    nuis = u[:, :, 2:]
    null_mean = float(np.mean(np.sqrt(np.mean(nuis.mean(1) ** 2, axis=-1))))
    null_cov = float(np.mean([np.linalg.norm(np.cov(d, rowvar=False) - np.eye(sim.dim - 2), "fro")
                              / np.sqrt(sim.dim - 2) for d in nuis]))
    dirs = rng.normal(size=(n_projections, sim.dim))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs_a = rng.normal(size=(n_projections, 2))
    dirs_a /= np.linalg.norm(dirs_a, axis=1, keepdims=True)
    n_shape = min(n_shape, len(samples))
    sw_full = np.mean([sliced_wasserstein(samples[i], ref.samples[i], dirs) for i in range(n_shape)])
    sw_active = np.mean([sliced_wasserstein(u[i, :, :2], u_ref[i, :, :2], dirs_a) for i in range(n_shape)])
    anchors = np.stack([g["mean"] for g in ref.grids])[:, None, :] + rng.normal(size=(len(samples), n_anchors, 2)) * 0.7
    ranks = random_reference_ranks(u[:, :, :2], u_true[:, :2], anchors, rng)
    rank_curve = np.array([(ranks <= a).mean() for a in LEVELS])
    return dict(
        posterior_kl=float(delta.mean()),
        posterior_kl_se=float(delta.std(ddof=1) / np.sqrt(len(delta))),
        test_log_prob=float(log_q.mean()),
        crps=float(crps.mean()), crps_per_dim=crps.mean(0).tolist(),
        sw_full=float(sw_full), sw_active=float(sw_active),
        null_mean_rms=null_mean, null_cov_error=null_cov,
        coverage90=float(central_coverage(samples, ref.theta, 0.9).mean()),
        active_rank_error=float(np.abs(rank_curve - LEVELS).mean()),
    ), delta


def reference_crps(ref) -> dict:
    """CRPS of the reference draws themselves: the floor for every arm (Monte Carlo estimate)."""
    crps = crps_ensemble(ref.samples, ref.theta)
    return dict(crps=float(crps.mean()), crps_per_dim=crps.mean(0).tolist())
