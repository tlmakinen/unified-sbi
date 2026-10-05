"""Structured one-step maps with an active dimension k and a summary dimension m.

Parameter map (from the rank-2 notebook, generalised to any k):

    (a, v) = W theta,  W in SO(D) (matrix exponential of a skew matrix),
    eta_phi(theta) = (f_phi(a), v),   f_phi: R^k -> R^k an affine-coupling flow.

Summary map: t = eta_psi(x) in R^m, m >= k, split as t = (mu, c):

    mu in R^k     the latent mean of f_phi(a);
    c  in R^(m-k) entries of a lower-triangular Cholesky factor L(x) of the latent covariance,
                  filled as log-diagonal entries first, then off-diagonals (row-major).
                  Unfilled diagonal entries are 1 and unfilled off-diagonals are 0.

One-step family and loss (a proper full-D density on theta for every (k, m)):

    q(theta | x) = N(f_phi(a); mu, L L^T) N(v; 0, I_{D-k}) |det D f_phi(a)|,
    L = E[ 1/2 ||L^-1 (f_phi(a) - mu)||^2 + sum log L_ii + 1/2 ||v||^2 - log|det D f_phi| ] + const.

With m = k this is the notebook's loss. Since every (k, m) gives a normalised density on
theta, held-out values of L are directly comparable, which is what the screen uses.
Every summary output is trained by this loss; none are left for the NPE to discover.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from ..maps import ParameterMap, SummaryMap

LOG2PI = float(np.log(2 * np.pi))


def max_extra(k: int) -> int:
    return k * (k + 1) // 2


class StructuredParameterMap(nn.Module):
    """theta -> (f(a), v) with (a, v) = W theta. Exact inverse; log|det| = log|det Df(a)|."""

    def __init__(self, dim: int, k: int, n_layers: int = 4, hidden: int = 32):
        super().__init__()
        if not 1 <= k <= dim:
            raise ValueError("Need 1 <= k <= dim.")
        self.dim, self.k = dim, k
        self.rotation_raw = nn.Parameter(torch.zeros(dim, dim))
        self.flow = ParameterMap(k, n_layers, hidden)   # coupling flow on the k active coordinates

    def rotation(self) -> torch.Tensor:
        s = self.rotation_raw - self.rotation_raw.T
        return torch.matrix_exp(s)

    def forward(self, theta):
        r = theta @ self.rotation().T
        a, ld = self.flow(r[..., :self.k])
        return torch.cat([a, r[..., self.k:]], -1), ld

    def inverse(self, eta):
        a, ld = self.flow.inverse(eta[..., :self.k])
        return torch.cat([a, eta[..., self.k:]], -1) @ self.rotation(), ld

    def active(self, theta):
        """f(a) and log|det Df| only."""
        r = theta @ self.rotation().T
        return self.flow(r[..., :self.k])


def cholesky_from_extras(extras: torch.Tensor, k: int) -> torch.Tensor:
    """Lower-triangular L (..., k, k) from m - k free outputs (log-diagonals first)."""
    batch = extras.shape[:-1]
    n = extras.shape[-1]
    L = torch.zeros(*batch, k, k, device=extras.device, dtype=extras.dtype)
    n_diag = min(n, k)
    log_diag = torch.zeros(*batch, k, device=extras.device, dtype=extras.dtype)
    if n_diag:
        log_diag[..., :n_diag] = 3.0 * torch.tanh(extras[..., :n_diag] / 3.0)
    L = L + torch.diag_embed(log_diag.exp())
    if n > k:
        rows, cols = torch.tril_indices(k, k, offset=-1, device=extras.device)
        m_off = n - k
        L[..., rows[:m_off], cols[:m_off]] = extras[..., k:]
    return L, log_diag


class LatentGaussian:
    """Helpers for N(f(a); mu, L L^T) given t = (mu, extras)."""

    def __init__(self, t: torch.Tensor, k: int):
        self.mu = t[..., :k]
        self.L, self.log_diag = cholesky_from_extras(t[..., k:], k)
        self.k = k

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        r = (z - self.mu).unsqueeze(-1)
        w = torch.linalg.solve_triangular(self.L, r, upper=False).squeeze(-1)
        return -0.5 * w.square().sum(-1) - self.log_diag.sum(-1) - 0.5 * self.k * LOG2PI

    def sample(self, n: int) -> torch.Tensor:
        """(n, ..., k) draws for a single or batched t."""
        eps = torch.randn((n, *self.mu.shape), device=self.mu.device)
        return self.mu + (self.L @ eps.unsqueeze(-1)).squeeze(-1)

    def variance(self) -> torch.Tensor:
        return (self.L @ self.L.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1)


def one_step_log_prob(phi: StructuredParameterMap, psi: SummaryMap, theta, x):
    """log q(theta | x) of the one-step family, per sample."""
    eta, ld = phi(theta)
    k = phi.k
    lat = LatentGaussian(psi(x), k)
    v = eta[..., k:]
    log_v = -0.5 * v.square().sum(-1) - 0.5 * (phi.dim - k) * LOG2PI
    return lat.log_prob(eta[..., :k]) + log_v + ld


@dataclass
class StructuredMapConfig:
    max_steps: int = 4000
    batch_size: int = 256
    lr: float = 1e-3
    clip_norm: float = 10.0
    eval_every: int = 50
    patience_steps: int = 1000
    n_layers: int = 4
    hidden: int = 32
    summary_hidden: int = 64

    def to_dict(self):
        return asdict(self)


@dataclass
class StructuredMaps:
    phi: StructuredParameterMap
    psi: SummaryMap
    k: int
    m: int
    history: list
    best_val_nll: float
    val_nll_per_sample: np.ndarray
    best_step: int
    seconds: float

    @torch.no_grad()
    def summarise(self, x):
        return self.psi(x)

    def state(self):
        return dict(phi=self.phi.state_dict(), psi=self.psi.state_dict(), k=self.k, m=self.m,
                    history=self.history, best_val_nll=self.best_val_nll,
                    val_nll_per_sample=self.val_nll_per_sample, best_step=self.best_step,
                    seconds=self.seconds)


def fit_structured_maps(theta_fit, x_fit, theta_val, x_val, k: int, m: int,
                        cfg: StructuredMapConfig, seed: int = 0) -> StructuredMaps:
    if not k <= m <= k + max_extra(k):
        raise ValueError(f"m must lie in [k, k + k(k+1)/2] = [{k}, {k + max_extra(k)}].")
    torch.manual_seed(seed)
    device = theta_fit.device
    dim = theta_fit.shape[-1]
    phi = StructuredParameterMap(dim, k, cfg.n_layers, cfg.hidden).to(device)
    psi = SummaryMap(x_fit, m, cfg.summary_hidden).to(device)
    params = list(phi.parameters()) + list(psi.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)
    n_fit = len(theta_fit)
    best, best_step, best_state, history = np.inf, 0, None, []
    start = time.time()
    for step in range(cfg.max_steps):
        idx = torch.randint(n_fit, (min(cfg.batch_size, n_fit),), device=device)
        opt.zero_grad()
        loss = -one_step_log_prob(phi, psi, theta_fit[idx], x_fit[idx]).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite one-step loss; reduce the learning rate.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.clip_norm)
        opt.step()
        if step % cfg.eval_every == 0 or step == cfg.max_steps - 1:
            with torch.no_grad():
                val = -one_step_log_prob(phi, psi, theta_val, x_val).mean().item()
            history.append((step, loss.item(), val))
            if val < best:
                best, best_step = val, step
                best_state = copy.deepcopy((phi.state_dict(), psi.state_dict()))
            elif step - best_step >= cfg.patience_steps:
                break
    phi.load_state_dict(best_state[0])
    psi.load_state_dict(best_state[1])
    phi.eval().requires_grad_(False)
    psi.eval().requires_grad_(False)
    with torch.no_grad():
        per_sample = (-one_step_log_prob(phi, psi, theta_val, x_val)).cpu().numpy()
    return StructuredMaps(phi, psi, k, m, history, best, per_sample, best_step, time.time() - start)


def load_structured_maps(state: dict, dim: int, x_like, cfg: StructuredMapConfig) -> StructuredMaps:
    phi = StructuredParameterMap(dim, state["k"], cfg.n_layers, cfg.hidden).to(x_like.device)
    psi = SummaryMap(x_like, state["m"], cfg.summary_hidden).to(x_like.device)
    phi.load_state_dict(state["phi"])
    psi.load_state_dict(state["psi"])
    phi.eval().requires_grad_(False)
    psi.eval().requires_grad_(False)
    return StructuredMaps(phi, psi, state["k"], state["m"], state["history"], state["best_val_nll"],
                          np.asarray(state["val_nll_per_sample"]), state["best_step"], state["seconds"])


def check_structured_maps(maps: StructuredMaps, theta_probe, atol: float = 5e-4) -> None:
    eta, ld = maps.phi(theta_probe)
    back, ild = maps.phi.inverse(eta)
    assert torch.allclose(back, theta_probe, atol=atol, rtol=atol), "inverse round trip failed"
    assert torch.allclose(ld + ild, torch.zeros_like(ld), atol=atol), "log-det inverse mismatch"
    jac = torch.autograd.functional.jacobian(lambda u: maps.phi(u)[0], theta_probe[0])
    assert torch.allclose(torch.linalg.slogdet(jac)[1], ld[0], atol=atol), "log-det vs autograd"
