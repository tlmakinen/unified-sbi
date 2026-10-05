"""Exact posterior for HiddenRosenbrock: converged 2D quadrature over the active plane.

log p(theta | x) = log N(theta; 0, I) + l((Q theta)_{1:2}; s) - log Z(s),
Z(s) = integral over R^2 of N(a; 0, I_2) exp[l(a; s)] da.

Quadrature resolution and domain are refined until log Z and the first two moments change by
less than the tolerances (as in the source notebook). Reference draws take active coordinates
from the normalised quadrature masses (a finite-grid approximation) and exact N(0, 1) draws
for the six nuisance coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.special import logsumexp

from .simulator import HiddenRosenbrock


def reference_grid(sim: HiddenRosenbrock, s: np.ndarray, n: int, bound: float) -> dict:
    g, w = leggauss(n)
    g, w = bound * g, bound * w
    a = np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1)
    logd = -0.5 * np.sum(a * a, axis=-1) - np.log(2 * np.pi) + sim.active_loglike(a, s)
    logmass = logd + np.log(w)[:, None] + np.log(w)[None, :]
    log_z = logsumexp(logmass)
    mass = np.exp(logmass - log_z)
    mean = np.einsum("ij,ijk->k", mass, a)
    second = np.einsum("ij,ijk,ijl->kl", mass, a, a)
    return dict(grid=g, weights=w, points=a, mass=mass, density=np.exp(logd - log_z),
                log_z=float(log_z), mean=mean, cov=second - np.outer(mean, mean), n=n, bound=bound)


def _difference(a, b):
    return (abs(a["log_z"] - b["log_z"]),
            max(np.max(np.abs(a["mean"] - b["mean"])), np.max(np.abs(a["cov"] - b["cov"]))))


def converged_reference(sim: HiddenRosenbrock, s: np.ndarray, n: int = 100, bound: float = 6.0,
                        log_tol: float = 1e-4, moment_tol: float = 2e-3, max_attempts: int = 5) -> dict:
    coarse = reference_grid(sim, s, n, bound)
    for _ in range(max_attempts):
        n = int(np.ceil(1.5 * n))
        fine = reference_grid(sim, s, n, bound)
        res = _difference(coarse, fine)
        expanded = reference_grid(sim, s, int(np.ceil(1.25 * n)), 1.25 * bound)
        dom = _difference(fine, expanded)
        if not np.isfinite([*res, *dom]).all():
            raise RuntimeError("Non-finite reference quadrature")
        if max(res[0], dom[0]) < log_tol and max(res[1], dom[1]) < moment_tol:
            return expanded
        if dom[0] >= log_tol or dom[1] >= moment_tol:
            coarse, n, bound = expanded, expanded["n"], expanded["bound"]
        else:
            coarse = fine
    raise RuntimeError("Reference quadrature did not converge; raise n or bound.")


def analytic_normaliser_check(sim: HiddenRosenbrock, s: np.ndarray, log_z: float) -> float:
    """Independent check: the exponent is quadratic in u2 at fixed u1, so integrate u2 analytically."""
    sd1, sd2, sl = sim.core_sd ** 2
    r = sim.n_rep
    precision = 1 + r / sd2 + r / sl

    def integrand(a):
        linear = r * s[1] / sd2 + r * (a * a + s[2]) / sl
        constant = a * a + r * (a - s[0]) ** 2 / sd1 + r * s[1] ** 2 / sd2 + r * (a * a + s[2]) ** 2 / sl
        return np.exp(-0.5 * constant + 0.5 * linear ** 2 / precision
                      - 0.5 * np.log(2 * np.pi * precision) - log_z)

    ratio, _ = quad(integrand, -np.inf, np.inf, epsabs=1e-9, epsrel=1e-9, limit=250)
    return abs(np.log(ratio))


@dataclass
class ExactReference:
    """Reference posteriors for a whole test set."""
    sim: HiddenRosenbrock
    theta: np.ndarray            # (B, D) test parameters
    x: np.ndarray                # (B, n_rep, data_dim)
    grids: list                  # converged quadrature dicts
    true_log_prob: np.ndarray    # (B,) exact log p(theta* | x)
    samples: np.ndarray          # (B, S, D) reference draws in theta

    @classmethod
    def build(cls, sim: HiddenRosenbrock, theta: np.ndarray, x: np.ndarray, n_samples: int,
              seed: int = 9002, **quad_kw) -> "ExactReference":
        s_all = sim.sufficient(x)
        grids = [converged_reference(sim, s, **quad_kw) for s in s_all]
        rng = np.random.default_rng(seed)
        samples = []
        for g in grids:
            idx = rng.choice(g["mass"].size, size=n_samples, p=g["mass"].ravel())
            u = rng.normal(size=(n_samples, sim.dim))
            u[:, :2] = g["points"].reshape(-1, 2)[idx]
            samples.append(sim.from_active(u))
        u_true = sim.to_active(theta.astype(float))
        logp = np.array([-0.5 * np.sum(th.astype(float) ** 2) - sim.dim / 2 * np.log(2 * np.pi)
                         + sim.active_loglike(u[:2], s) - g["log_z"]
                         for th, u, s, g in zip(theta, u_true, s_all, grids)])
        return cls(sim, theta, x, grids, logp, np.stack(samples).astype("float32"))
