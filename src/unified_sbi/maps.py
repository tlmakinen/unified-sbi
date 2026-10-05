"""The one-step "discovery" step: an invertible parameter map eta_phi and a summary map eta_psi.

Both are fitted jointly by minimising the one-step loss

    L = E[ 1/2 || (eta_phi(theta) - eta_psi(x)) / s(x) ||^2 + sum log s(x) - log |det D eta_phi(theta)| ],

i.e. the negative log of q(theta | x) = N(eta_phi(theta); eta_psi(x), diag s^2) |det D eta_phi|.
With ``learn_scales=False`` (default) s = 1, which is the loss in unified_sbi.ipynb.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


# ------------------------------------------------------------------------- networks
class Coupling(nn.Module):
    """Affine coupling layer; ``mask`` marks the coordinates that condition the others."""

    def __init__(self, mask, hidden: int = 32):
        super().__init__()
        dim = len(mask)
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2 * dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, inverse: bool = False):
        fixed = z * self.mask
        s, b = self.net(fixed).chunk(2, dim=-1)
        s = 1.5 * torch.tanh(s) * (1 - self.mask)
        b = b * (1 - self.mask)
        moved = (z - b) * torch.exp(-s) if inverse else z * torch.exp(s) + b
        return fixed + (1 - self.mask) * moved, (-s if inverse else s).sum(-1)


class ParameterMap(nn.Module):
    """Invertible eta_phi: R^D -> R^D with an exact inverse and log-Jacobian."""

    def __init__(self, dim: int, n_layers: int = 4, hidden: int = 32):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList(
            [Coupling([(j + i) % 2 for j in range(dim)], hidden) for i in range(n_layers)])
        self.log_scale = nn.Parameter(torch.zeros(dim))

    def forward(self, theta):
        """Return eta and log |det d eta / d theta|."""
        z, ld = theta, theta.new_zeros(theta.shape[:-1])
        for layer in self.layers:
            z, delta = layer(z)
            ld = ld + delta
        return z * self.log_scale.exp(), ld + self.log_scale.sum()

    def inverse(self, eta):
        """Return theta and log |det d theta / d eta|."""
        z = eta * (-self.log_scale).exp()
        ld = eta.new_zeros(eta.shape[:-1]) - self.log_scale.sum()
        for layer in reversed(self.layers):
            z, delta = layer(z, inverse=True)
            ld = ld + delta
        return z, ld


class SummaryMap(nn.Module):
    """eta_psi: mean-pools standardised replicates, then an MLP with a linear skip.

    Input x has shape (..., n_rep, x_dim). An optional second head gives log-scales s(x).
    """

    def __init__(self, fit_x: torch.Tensor, out_dim: int, hidden: int = 32,
                 learn_scales: bool = False):
        super().__init__()
        x_dim = fit_x.shape[-1]
        self.learn_scales = learn_scales
        self.register_buffer("mean", fit_x.mean((0, 1)))
        self.register_buffer("std", fit_x.std((0, 1)).clamp_min(1e-6))
        self.net = nn.Sequential(nn.Linear(x_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))
        self.skip = nn.Linear(x_dim, out_dim)
        self.scale_net = nn.Sequential(nn.Linear(x_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))
        nn.init.zeros_(self.scale_net[-1].weight)
        nn.init.zeros_(self.scale_net[-1].bias)

    def pooled(self, x):
        return ((x - self.mean) / self.std).mean(-2)

    def forward(self, x):
        h = self.pooled(x)
        return self.net(h) + self.skip(h)

    def log_scale(self, x):
        log_s = 3.0 * torch.tanh(self.scale_net(self.pooled(x)) / 3.0)
        return log_s if self.learn_scales else torch.zeros_like(log_s)


def one_step_loss(eta_phi: ParameterMap, eta_psi: SummaryMap, theta, x):
    eta, logdet = eta_phi(theta)
    t, log_s = eta_psi(x), eta_psi.log_scale(x)
    return ((0.5 * ((eta - t) / log_s.exp()).square() + log_s).sum(-1) - logdet).mean()


# ------------------------------------------------------------------------- fitting
@dataclass
class MapConfig:
    max_steps: int = 4000
    batch_size: int = 256
    lr: float = 1e-3
    clip_norm: float = 10.0
    eval_every: int = 50
    patience_steps: int = 1000   # stop when validation loss has not improved for this long
    n_layers: int = 4
    hidden: int = 32
    learn_scales: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class Maps:
    eta_phi: ParameterMap
    eta_psi: SummaryMap
    history: list
    best_val_loss: float
    best_step: int
    seconds: float

    @torch.no_grad()
    def summarise(self, x: torch.Tensor, batch: int = 8192) -> torch.Tensor:
        return torch.cat([self.eta_psi(x[i:i + batch]) for i in range(0, len(x), batch)])

    @torch.no_grad()
    def to_eta(self, theta: torch.Tensor):
        return self.eta_phi(theta)

    @torch.no_grad()
    def to_theta(self, eta: torch.Tensor):
        return self.eta_phi.inverse(eta)

    def state(self) -> dict:
        return dict(eta_phi=self.eta_phi.state_dict(), eta_psi=self.eta_psi.state_dict(),
                    history=self.history, best_val_loss=self.best_val_loss,
                    best_step=self.best_step, seconds=self.seconds)


def fit_maps(theta_fit, x_fit, theta_val, x_val, cfg: MapConfig, seed: int = 0) -> Maps:
    """Jointly fit eta_phi and eta_psi on (theta_fit, x_fit), early-stopping on the validation pair."""
    torch.manual_seed(seed)
    device = theta_fit.device
    dim = theta_fit.shape[-1]
    eta_phi = ParameterMap(dim, cfg.n_layers, cfg.hidden).to(device)
    eta_psi = SummaryMap(x_fit, dim, cfg.hidden, cfg.learn_scales).to(device)
    params = list(eta_phi.parameters()) + list(eta_psi.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)
    n_fit = len(theta_fit)
    best, best_step, best_state, history = np.inf, 0, None, []
    start = time.time()
    for step in range(cfg.max_steps):
        idx = torch.randint(n_fit, (min(cfg.batch_size, n_fit),), device=device)
        opt.zero_grad()
        loss = one_step_loss(eta_phi, eta_psi, theta_fit[idx], x_fit[idx])
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite one-step loss; reduce the learning rate.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.clip_norm)
        opt.step()
        if step % cfg.eval_every == 0 or step == cfg.max_steps - 1:
            with torch.no_grad():
                val = one_step_loss(eta_phi, eta_psi, theta_val, x_val).item()
            history.append((step, loss.item(), val))
            if val < best:
                best, best_step = val, step
                best_state = copy.deepcopy((eta_phi.state_dict(), eta_psi.state_dict()))
            elif step - best_step >= cfg.patience_steps:
                break
    eta_phi.load_state_dict(best_state[0])
    eta_psi.load_state_dict(best_state[1])
    eta_phi.eval().requires_grad_(False)
    eta_psi.eval().requires_grad_(False)
    return Maps(eta_phi, eta_psi, history, best, best_step, time.time() - start)


def load_maps(state: dict, dim: int, x_like: torch.Tensor, cfg: MapConfig) -> Maps:
    device = x_like.device
    eta_phi = ParameterMap(dim, cfg.n_layers, cfg.hidden).to(device)
    eta_psi = SummaryMap(x_like, dim, cfg.hidden, cfg.learn_scales).to(device)
    eta_phi.load_state_dict(state["eta_phi"])
    eta_psi.load_state_dict(state["eta_psi"])
    eta_phi.eval().requires_grad_(False)
    eta_psi.eval().requires_grad_(False)
    return Maps(eta_phi, eta_psi, state["history"], state["best_val_loss"],
                state["best_step"], state["seconds"])


def check_maps(maps: Maps, theta_probe: torch.Tensor, atol: float = 2e-4) -> None:
    """Assert the inverse round-trips and the analytic log-Jacobian matches autograd."""
    eta, ld = maps.eta_phi(theta_probe)
    back, ild = maps.eta_phi.inverse(eta)
    assert torch.allclose(back, theta_probe, atol=atol, rtol=atol), "inverse round trip failed"
    assert torch.allclose(ld + ild, torch.zeros_like(ld), atol=atol), "log-det inverse mismatch"
    jac = torch.autograd.functional.jacobian(lambda u: maps.eta_phi(u)[0], theta_probe[0])
    assert torch.allclose(torch.linalg.slogdet(jac)[1], ld[0], atol=atol), "log-det vs autograd"
