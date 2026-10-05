"""Inference arms for the rank-deficient benchmark (all densities reported in native theta).

    raw         theta(D)     | flattened x            NPE, all N simulations
    mean        theta(D)     | replicate means         NPE, all N simulations
    summary     theta(D)     | t (m)                   NPE on the learned summary
    eta         eta(D)       | t (m)                   NPE in structured coordinates, exact inverse + Jacobian
    structured  f(a)(k)      | t (m)  x  N(v; 0, I)    k-dim NPE; the D-k learned inactive coordinates keep the prior
    oracle      u_active(2)  | s(x) (3)  x  N(0, I)    privileged: true plane and exact sufficient statistic
    gaussian    one-step family q_{k,m}(theta | x)      no NPE; uses only the discovery simulations
    prior       N(0, I_D)                               ignores x

The prior is N(0, I) with full support, so no box truncation or leakage correction is needed.
"""

from __future__ import annotations

import numpy as np
import torch

from ..npe import NPEConfig, train_npe
from .maps import LatentGaussian, StructuredMaps, one_step_log_prob

ALL_ARMS = ("raw", "mean", "summary", "eta", "structured", "oracle", "gaussian", "prior")
USES_MAPS = {"raw": False, "mean": False, "summary": True, "eta": True, "structured": True,
             "oracle": False, "gaussian": True, "prior": False}
NEEDS_NPE = {a: a not in ("gaussian", "prior") for a in ALL_ARMS}
LOG2PI = float(np.log(2 * np.pi))


def std_normal_log_prob(z: torch.Tensor) -> torch.Tensor:
    return -0.5 * z.square().sum(-1) - 0.5 * z.shape[-1] * LOG2PI


class Arm:
    """One arm: builds NPE targets/contexts, and maps samples and densities back to theta."""

    def __init__(self, name: str, sim, maps: StructuredMaps | None, device: str):
        self.name, self.sim, self.maps, self.device = name, sim, maps, device
        self.Q = torch.as_tensor(sim.Q, dtype=torch.float32, device=device)
        self.net = None
        self.fit_info = {}

    # ------------------------------------------------------------ coordinates
    def context(self, x: torch.Tensor) -> torch.Tensor:
        n = self.name
        if n == "raw":
            return x.reshape(len(x), -1)
        if n == "mean":
            return x.mean(-2)
        if n == "oracle":
            s = self.sim.sufficient(x.cpu().numpy())
            return torch.as_tensor(s, dtype=torch.float32, device=self.device)
        return self.maps.summarise(x)

    @torch.no_grad()
    def target(self, theta: torch.Tensor):
        """Estimator target and the log-density term that completes log q(theta | x):
        log q(theta|x) = log q_net(target | context) + extra(theta)."""
        n = self.name
        if n in ("raw", "mean", "summary"):
            return theta, theta.new_zeros(theta.shape[:-1])
        if n == "eta":
            return self.maps.phi(theta)
        if n == "structured":
            eta, ld = self.maps.phi(theta)
            k = self.maps.k
            return eta[..., :k], ld + std_normal_log_prob(eta[..., k:])
        if n == "oracle":
            u = theta @ self.Q.T
            return u[..., :2], std_normal_log_prob(u[..., 2:])
        raise KeyError(n)

    @torch.no_grad()
    def to_theta(self, draws: torch.Tensor) -> torch.Tensor:
        """Map estimator draws (..., dim_target) to theta, filling prior coordinates."""
        n = self.name
        if n in ("raw", "mean", "summary"):
            return draws
        if n == "eta":
            return self.maps.phi.inverse(draws)[0]
        if n == "structured":
            k = self.maps.k
            v = torch.randn((*draws.shape[:-1], self.sim.dim - k), device=self.device)
            return self.maps.phi.inverse(torch.cat([draws, v], -1))[0]
        if n == "oracle":
            rest = torch.randn((*draws.shape[:-1], self.sim.dim - 2), device=self.device)
            return torch.cat([draws, rest], -1) @ self.Q
        raise KeyError(n)

    # ------------------------------------------------------------ training
    def fit(self, theta_tr, x_tr, theta_va, x_va, cfg: NPEConfig, seed: int):
        if not NEEDS_NPE[self.name]:
            if self.name == "gaussian":
                with torch.no_grad():
                    self.fit_info = dict(best_val_log_prob=float(
                        one_step_log_prob(self.maps.phi, self.maps.psi, theta_va, x_va).mean()))
            else:
                self.fit_info = dict(best_val_log_prob=float(std_normal_log_prob(theta_va).mean()))
            return
        tgt_tr, _ = self.target(theta_tr)
        tgt_va, extra_va = self.target(theta_va)
        ctx_tr, ctx_va = self.context(x_tr), self.context(x_va)
        fit = train_npe(tgt_tr, ctx_tr, tgt_va, ctx_va, cfg, seed=seed)
        self.net = fit.net
        self.fit_info = dict(best_val_log_prob=fit.best_val_log_prob + extra_va.mean().item(),
                             epochs=fit.epochs_trained, best_epoch=fit.best_epoch, seconds=fit.seconds,
                             n_train=len(theta_tr), n_val=len(theta_va))

    # ------------------------------------------------------------ evaluation
    @torch.no_grad()
    def sample(self, x: torch.Tensor, n: int, chunk: int = 64) -> np.ndarray:
        """(B, n, D) posterior draws in theta."""
        out = []
        for i in range(0, len(x), chunk):
            xb = x[i:i + chunk]
            b = len(xb)
            if self.name == "prior":
                th = torch.randn((b, n, self.sim.dim), device=self.device)
            elif self.name == "gaussian":
                k = self.maps.k
                lat = LatentGaussian(self.maps.summarise(xb), k)
                a = lat.sample(n).transpose(0, 1)                                   # (b, n, k)
                v = torch.randn((b, n, self.sim.dim - k), device=self.device)
                th = self.maps.phi.inverse(torch.cat([a, v], -1))[0]
            else:
                draws = self.net.sample((n,), self.context(xb)).transpose(0, 1)     # (b, n, dim)
                th = self.to_theta(draws)
            out.append(th.cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def log_prob(self, theta: torch.Tensor, x: torch.Tensor) -> np.ndarray:
        """log q(theta_i | x_i), one theta per observation."""
        if self.name == "prior":
            return std_normal_log_prob(theta).cpu().numpy()
        if self.name == "gaussian":
            return one_step_log_prob(self.maps.phi, self.maps.psi, theta, x).cpu().numpy()
        tgt, extra = self.target(theta)
        return (self.net.log_prob(tgt[None], self.context(x))[0] + extra).cpu().numpy()
