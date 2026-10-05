"""The (k, m) screen: fit one-step maps over a grid and select by held-out one-step NLL.

Each candidate defines a normalised density q_{k,m}(theta | x) on the full D-dimensional
parameter space, so held-out NLLs are comparable across k and m. Selection uses the
one-standard-error rule: among candidates whose mean held-out NLL is within one paired
standard error of the best, pick the smallest m, then the smallest k.

Two readouts accompany the NLL:

* information spectrum: rotation-invariant Gaussian-channel information per latent direction
  (see ``information_spectrum``). A diagnostic only: selection uses the held-out NLL.
* subspace recovery (only when the true active plane is known): principal angles between the
  learned active span and the true plane, and the fraction of the true plane captured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from .maps import (LatentGaussian, StructuredMapConfig, StructuredMaps, fit_structured_maps,
                   max_extra)


def candidate_grid(k_values, m_mode: str = "standard", m_values=None) -> list[tuple[int, int]]:
    """(k, m) pairs.

    m_mode "standard": m in {k (identity covariance), 2k (diagonal), k + k(k+1)/2 (full)}.
    m_mode "all": every m from k to k + k(k+1)/2.  m_mode "identity": m = k only.
    ``m_values`` (explicit list) overrides m_mode; values outside [k, k + k(k+1)/2] are skipped.
    """
    out = []
    for k in k_values:
        lo, hi = k, k + max_extra(k)
        if m_values is not None:
            ms = [m for m in m_values if lo <= m <= hi]
        elif m_mode == "all":
            ms = list(range(lo, hi + 1))
        elif m_mode == "identity":
            ms = [k]
        elif m_mode == "standard":
            ms = sorted({k, 2 * k, hi})
        else:
            raise ValueError(m_mode)
        out += [(k, m) for m in ms]
    return out


@torch.no_grad()
def information_spectrum(maps: StructuredMaps, x: torch.Tensor):
    """Rotation-invariant Gaussian-channel information in the k active latent coordinates.

    With C = Cov_x[mu(x)] and S = E_x[Sigma(x)], the spectrum is 1/2 log(1 + lambda_j) for the
    eigenvalues lambda_j of S^{-1/2} C S^{-1/2}, sorted in decreasing order. The total adds the
    heteroscedastic term: 1/2 log det(C + S) - 1/2 E_x log det Sigma(x). Returns (spectrum, total).
    Uninformative directions have values near 0 whatever the rotation inside the active block.
    """
    lat = LatentGaussian(maps.summarise(x), maps.k)
    mu = lat.mu.double()
    sigma = (lat.L @ lat.L.transpose(-1, -2)).double()
    C = torch.cov(mu.T).reshape(maps.k, maps.k)
    S = sigma.mean(0)
    evals, evecs = torch.linalg.eigh(S)
    S_inv_half = evecs @ torch.diag(evals.clamp_min(1e-12).rsqrt()) @ evecs.T
    lam = torch.linalg.eigvalsh(S_inv_half @ C @ S_inv_half).clamp_min(0).flip(0)
    spectrum = 0.5 * torch.log1p(lam)
    total = 0.5 * torch.logdet(C + S) - 0.5 * torch.logdet(sigma).mean()
    return spectrum.cpu().numpy(), float(total)


@torch.no_grad()
def subspace_recovery(maps: StructuredMaps, true_rows: np.ndarray) -> dict:
    """true_rows: (k_true, D) orthonormal rows spanning the true active plane."""
    W = maps.phi.rotation().cpu().numpy()[: maps.k]
    overlap = W @ true_rows.T
    cos = np.clip(np.linalg.svd(overlap, compute_uv=False), 0, 1)
    captured = float(np.sum(overlap ** 2) / len(true_rows))
    return dict(principal_angles_deg=np.degrees(np.arccos(cos)).tolist(), plane_captured=captured)


@dataclass
class ScreenResult:
    table: pd.DataFrame
    selected: tuple[int, int]
    maps: dict = field(repr=False)          # (k, m) -> StructuredMaps

    def selected_maps(self) -> StructuredMaps:
        return self.maps[self.selected]


def select_one_se(table: pd.DataFrame, per_sample: dict) -> tuple[int, int]:
    best = table.loc[table.val_nll.idxmin()]
    best_key = (int(best.k), int(best.m))
    ok = []
    for key, nll in per_sample.items():
        diff = nll - per_sample[best_key]            # paired over the same validation sims
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
        if diff.mean() <= se:
            ok.append(key)
    return min(ok, key=lambda km: (km[1], km[0]))


def run_screen(theta_fit, x_fit, theta_val, x_val, grid, cfg: StructuredMapConfig, seed: int = 0,
               true_rows: np.ndarray | None = None, verbose: bool = True) -> ScreenResult:
    rows, fitted, per_sample = [], {}, {}
    for k, m in grid:
        maps = fit_structured_maps(theta_fit, x_fit, theta_val, x_val, k, m, cfg, seed=seed)
        fitted[k, m] = maps
        per_sample[k, m] = maps.val_nll_per_sample
        spectrum, total = information_spectrum(maps, x_val)
        row = dict(k=k, m=m, val_nll=maps.best_val_nll,
                   val_nll_se=float(maps.val_nll_per_sample.std(ddof=1) / np.sqrt(len(maps.val_nll_per_sample))),
                   info_total=total, info_spectrum=np.round(spectrum, 4).tolist(),
                   best_step=maps.best_step, seconds=maps.seconds)
        if true_rows is not None:
            row.update(subspace_recovery(maps, true_rows))
        rows.append(row)
        if verbose:
            extra = f" | plane captured {row['plane_captured']:.3f}" if true_rows is not None else ""
            print(f"  screen k={k} m={m}: val NLL {row['val_nll']:.3f} ± {row['val_nll_se']:.3f} | "
                  f"info {row['info_total']:.2f} nats{extra} | {maps.seconds:.0f}s")
    table = pd.DataFrame(rows)
    selected = select_one_se(table, per_sample)
    table["selected"] = [(r.k, r.m) == selected for r in table.itertuples()]
    return ScreenResult(table, selected, fitted)
