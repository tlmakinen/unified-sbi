"""A hidden 2D Rosenbrock likelihood inside an 8D parameter space (rank-deficient benchmark).

Ported from ``unified_sbi_rosenbrock_2d_in_8d.ipynb``. theta ~ N(0, I_D), u = Q theta for a fixed
hidden orthogonal Q. Each of ``n_rep`` replicates contains

    y_r = (u_1 + e1, u_2 + e2, u_2 - u_1^2 + e3, n_4, ..., n_{data_dim}),

with variances (var_direct, var_direct, var_link) on the three core channels and pure
nuisance noise (sd = nuisance_sd) elsewhere. A second hidden rotation mixes channels, x_r = O y_r.

Only k = 2 parameter directions affect the data, but the exact sufficient statistic,
s(x) = (O^T xbar)_{1:3}, is 3D. The posterior is p(u_1, u_2 | s) x prod_{j>2} N(u_j; 0, 1).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def orthogonal_matrix(dim: int, seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(dim, dim)))
    return q @ np.diag(np.sign(np.diag(r)))


@dataclass
class HiddenRosenbrock:
    dim: int = 8
    n_rep: int = 4
    data_dim: int = 8
    var_direct: float = 4.0
    var_link: float = 0.25
    nuisance_sd: float = 3.0
    q_seed: int = 1043
    o_seed: int = 2043

    true_active_dim = 2      # k*: parameter directions the likelihood depends on
    true_sufficient_dim = 3  # m*: size of the exact sufficient statistic

    def __post_init__(self):
        if self.dim <= 2 or self.data_dim < 3:
            raise ValueError("Need dim > 2 and data_dim >= 3.")
        self.Q = orthogonal_matrix(self.dim, self.q_seed)
        self.O = orthogonal_matrix(self.data_dim, self.o_seed)

    @property
    def core_sd(self) -> np.ndarray:
        return np.sqrt([self.var_direct, self.var_direct, self.var_link])

    @property
    def raw_dim(self) -> int:
        return self.n_rep * self.data_dim

    @staticmethod
    def core_mean(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        return np.stack([a[..., 0], a[..., 1], a[..., 1] - a[..., 0] ** 2], axis=-1)

    def to_active(self, theta: np.ndarray) -> np.ndarray:
        """u = Q theta (all D rotated coordinates; the first two are active)."""
        return np.asarray(theta) @ self.Q.T

    def from_active(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(u) @ self.Q

    def simulate(self, n: int, seed: int):
        """theta (n, D) and x (n, n_rep, data_dim), float32."""
        rng = np.random.default_rng(seed)
        theta = rng.normal(size=(n, self.dim))
        u = self.to_active(theta)
        y = rng.normal(size=(n, self.n_rep, self.data_dim)) * self.nuisance_sd
        y[:, :, :3] = self.core_mean(u[:, :2])[:, None, :] + rng.normal(size=(n, self.n_rep, 3)) * self.core_sd
        x = y @ self.O.T
        return theta.astype("float32"), x.astype("float32")

    def sufficient(self, x: np.ndarray) -> np.ndarray:
        """Exact 3D sufficient statistic (privileged: uses the hidden O)."""
        return (np.asarray(x, dtype="float64").mean(-2) @ self.O)[..., :3]

    def active_loglike(self, a: np.ndarray, s: np.ndarray) -> np.ndarray:
        return -0.5 * self.n_rep * np.sum(((self.core_mean(a) - s) / self.core_sd) ** 2, axis=-1)

    def to_dict(self) -> dict:
        return asdict(self)
