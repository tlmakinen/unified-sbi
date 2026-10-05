"""Posterior-quality metrics: CRPS (ensemble, Gaussian, from a CDF) and interval coverage."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

_trapezoid = getattr(np, "trapezoid", None) or np.trapz   # numpy < 2 compatibility


def crps_ensemble(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Fair (unbiased) ensemble CRPS per coordinate.

    samples: (B, S, D) posterior draws; truth: (B, D). Returns (B, D).
    CRPS = E|X - y| - 1/2 E|X - X'|, with E|X - X'| estimated without the i = j terms:
    sum_{i,j} |x_i - x_j| = 2 sum_k (2k - S - 1) x_(k) for sorted draws, k = 1..S.
    """
    samples = np.asarray(samples, dtype=float)
    truth = np.asarray(truth, dtype=float)
    s = samples.shape[1]
    term1 = np.abs(samples - truth[:, None, :]).mean(axis=1)
    ordered = np.sort(samples, axis=1)
    k = np.arange(1, s + 1)[None, :, None]
    spread = (2 * k - s - 1) * ordered
    term2 = spread.sum(axis=1) / (s * (s - 1))   # = 1/2 * E|X - X'|
    return term1 - term2


def crps_gaussian(mu, sigma, y) -> np.ndarray:
    z = (np.asarray(y) - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_from_cdf(z_grid: np.ndarray, cdf: np.ndarray, y: float) -> float:
    """CRPS = integral of (F(z) - 1[z >= y])^2 dz on a grid spanning the support."""
    heaviside = (z_grid >= y).astype(float)
    return float(_trapezoid((cdf - heaviside) ** 2, z_grid))


def central_coverage(samples: np.ndarray, truth: np.ndarray, level: float) -> np.ndarray:
    """Fraction of truths inside the central `level` interval, per coordinate. Returns (D,)."""
    lo, hi = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2], axis=1)
    return ((truth >= lo) & (truth <= hi)).mean(axis=0)
