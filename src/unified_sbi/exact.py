"""Exact posteriors for RosenbrockChain by Gauss-Legendre quadrature along the chain.

With a uniform box prior the posterior factorises as a chain,

    p(theta | x) ∝ prod_j unary_j(theta_j) * prod_j link_j(theta_j, theta_{j+1}),

and the replicate means are sufficient. Forward/backward passes give the normaliser,
all 1D marginals and, via conditional transition matrices, all 2D marginals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp

from .metrics import crps_from_cdf
from .simulators import RosenbrockChain


def gl_grid(n: int, half_width: float):
    z, w = leggauss(n)
    return half_width * z, half_width * w


def log_likelihood_exponent(sim: RosenbrockChain, theta: np.ndarray, x_bar: np.ndarray) -> np.ndarray:
    """-1/2 n_rep sum_k ((mu_k(theta) - xbar_k) / sigma_k)^2; theta-independent constants dropped."""
    return -0.5 * sim.n_rep * (((sim.mu(theta) - x_bar) / sim.noise) ** 2).sum(-1)


@dataclass
class ChainPosterior:
    sim: RosenbrockChain
    grid: np.ndarray
    weights: np.ndarray
    x_bar: np.ndarray
    log_z: float
    mass_1d: list
    forward: list
    backward: list
    log_unary: np.ndarray
    log_links: list

    # ------------------------------------------------------------ densities
    def log_prob(self, theta: np.ndarray) -> np.ndarray:
        """Exact normalised log p(theta | x); -inf outside the prior box."""
        theta = np.asarray(theta, dtype=float)
        logp = log_likelihood_exponent(self.sim, theta, self.x_bar) - self.log_z
        inside = (np.abs(theta) <= self.sim.a_box).all(-1)
        return np.where(inside, logp, -np.inf)

    @property
    def density_1d(self):
        return [m / self.weights for m in self.mass_1d]

    def cdf_1d(self, j: int, z: np.ndarray) -> np.ndarray:
        """Marginal CDF of theta_j, interpolated between quadrature nodes."""
        m = self.mass_1d[j]
        mid = np.cumsum(m) - 0.5 * m
        nodes = np.concatenate([[-self.sim.a_box], self.grid, [self.sim.a_box]])
        values = np.concatenate([[0.0], mid, [1.0]])
        return np.interp(z, nodes, values)

    def crps(self, theta_true: np.ndarray, n_fine: int = 4001) -> np.ndarray:
        z = np.linspace(-self.sim.a_box, self.sim.a_box, n_fine)
        return np.array([crps_from_cdf(z, self.cdf_1d(j, z), theta_true[j])
                         for j in range(self.sim.dim)])

    def mean_std(self):
        mean = np.array([(m * self.grid).sum() for m in self.mass_1d])
        var = np.array([(m * self.grid ** 2).sum() for m in self.mass_1d]) - mean ** 2
        return mean, np.sqrt(np.maximum(var, 0))

    # ------------------------------------------------------------ 2D marginals
    def mass_2d(self) -> dict:
        """{(i, j): (N, N) joint mass, rows theta_j, columns theta_i} for all i > j."""
        d, logw = self.sim.dim, np.log(self.weights)
        trans = [np.exp(self.log_links[j]
                        + (self.log_unary[j + 1] + logw + self.backward[j + 1])[None, :]
                        - self.backward[j][:, None]) for j in range(d - 1)]
        out = {}
        for j in range(d - 1):
            bridge = trans[j]
            for i in range(j + 1, d):
                if i > j + 1:
                    bridge = bridge @ trans[i - 1]
                out[i, j] = self.mass_1d[j][:, None] * bridge
        return out


def chain_posterior(sim: RosenbrockChain, x_obs: np.ndarray, n_grid: int = 200) -> ChainPosterior:
    """x_obs: one observation, shape (n_rep, 2D-1)."""
    d, n_rep = sim.dim, sim.n_rep
    sigma, curve = sim.noise, sim.curvature
    x_bar = np.asarray(x_obs, dtype=float).mean(0)
    grid, weights = gl_grid(n_grid, sim.a_box)
    logw = np.log(weights)
    log_unary = -0.5 * n_rep * ((grid[None, :] - x_bar[:d, None]) / sigma[:d, None]) ** 2
    # Row a is theta_j = grid[a]; column c is theta_{j+1} = grid[c].
    log_links = [-0.5 * n_rep * ((grid[None, :] - curve[j] * grid[:, None] ** 2 - x_bar[d + j])
                                 / sigma[d + j]) ** 2 for j in range(d - 1)]
    forward = [log_unary[0] + logw]
    for j in range(d - 1):
        forward.append(log_unary[j + 1] + logw
                       + logsumexp(forward[j][:, None] + log_links[j], axis=0))
    backward = [None] * d
    backward[-1] = np.zeros(n_grid)
    for j in range(d - 2, -1, -1):
        backward[j] = logsumexp(log_links[j] + (log_unary[j + 1] + logw + backward[j + 1])[None, :],
                                axis=1)
    log_z = float(logsumexp(forward[-1]))
    mass_1d = [np.exp(forward[j] + backward[j] - log_z) for j in range(d)]
    return ChainPosterior(sim, grid, weights, x_bar, log_z, mass_1d, forward, backward,
                          log_unary, log_links)


def exact_test_metrics(sim: RosenbrockChain, x_test: np.ndarray, theta_test: np.ndarray,
                       n_grid: int = 200) -> dict:
    """Oracle log p(theta* | x) and per-coordinate CRPS for every test pair."""
    log_prob, crps = [], []
    for x, th in zip(x_test, theta_test):
        post = chain_posterior(sim, x, n_grid)
        log_prob.append(post.log_prob(th[None].astype(float))[0])
        crps.append(post.crps(th.astype(float)))
    return dict(log_prob=np.array(log_prob), crps=np.array(crps))
