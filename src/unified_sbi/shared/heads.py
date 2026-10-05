"""Shared-space posteriors: parameters and data mapped into one K-dimensional space.

    log f(theta, t) = c(eta(theta), t) + log b(theta)
    q(theta | x)    = f(theta, t(x)) / Z(t(x)),     Z(t) = int_box f(theta, t) dtheta

critic c:   "gauss"  -1/2 ||eta - t||^2            (the user's form; t ~ N(eta(theta), I_K))
            "expfam" eta . t - A(theta)            (any exponential family with K statistics)
base b:     "prior"     uniform on the box (the actual prior)
            "jeffreys"  vol(J) = sqrt(det J^T J)    (the user's log-volume term)

``normalised=False`` with ``critic="gauss", base="jeffreys"`` reproduces the agent's
rectangular one-step loss  1/2||eta - t||^2 - log vol(J)  (plus K/2 log 2 pi), which is
-log q - log Z(t) and is unbounded below when K > d (see docs/shared_space.md).

Information loss (``info``, Gaussian critic, prior base, normalised heads only):
    "quad"  -log q with Z(t) by adaptive quadrature (the default; ``hybrid_shared``)
    "nce"   InfoNCE with in-batch negatives: -c_ii + log (1/B) sum_j exp c_ij   (Z-free)
    "hyv"   conditional Hyvarinen score matching in logit coordinates            (Z-free)
GMIHead (not a posterior head): Gaussian mutual information 1/2 log det Sigma_res - 1/2 log det Cov(eta),
with a separate stop-gradient emulator tower eta_emu; evaluated through a downstream density estimator.
Optional stop-gradient emulator term (``beta > 0``):  beta * 1/2 ||Sigma^{-1/2}(sg[t] - eta(theta))||^2,
which regresses eta onto E[t | theta] without letting t move towards eta.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .nets import ParamEmbed, mlp
from .quadrature import integrate

LOGIT_SCALE = math.pi / math.sqrt(3)     # ParamEmbed's logit coordinates are y = logit(u) / LOGIT_SCALE


class EmulatorTerm(nn.Module):
    """1/2 ||Sigma^{-1/2}(sg[t] - eta)||^2 with Sigma the running residual second moment.

    The gradient reaches eta only (t is detached). Sigma is updated (no grad) in training mode.
    """

    def __init__(self, K, decay=0.99, shrink=1e-3):
        super().__init__()
        self.decay, self.shrink = decay, shrink
        self.register_buffer("S", torch.eye(K))

    def forward(self, eta, t):
        r = t.detach() - eta                                         # the stop-gradient
        if self.training:
            with torch.no_grad():
                self.S.mul_(self.decay).add_((1 - self.decay) * (r.T @ r) / len(r))
        L = torch.linalg.cholesky(self.S + self.shrink * torch.eye(len(self.S), device=r.device, dtype=r.dtype))
        z = torch.linalg.solve_triangular(L, r.T, upper=False)      # Sigma^{-1/2} r
        return 0.5 * (z ** 2).sum(0)                                 # per sample


def infonce_loss(eta, t):
    """Per-sample InfoNCE with in-batch negatives for the Gaussian critic.

    eta (B, K) = eta(theta_j) for the batch's prior-drawn theta; t (B, K) = t(x_i).
    -c_ii + log (1/B) sum_j exp c_ij  with c_ij = -1/2 ||eta_j - t_i||^2.
    """
    c = -0.5 * (t[:, None, :] - eta[None, :, :]).square().sum(-1)    # (B_t, B_theta)
    return -c.diagonal() + torch.logsumexp(c, -1) - math.log(c.shape[-1])


def hyvarinen_loss(f, y, t, prior_score=None):
    """Per-sample conditional Hyvarinen loss 1/2||s||^2 + div_y s for log q = -1/2||f(y) - t||^2 + log pi(y).

    s = J^T (t - f(y)) + prior_score(y). The divergence is exact (d autograd passes); the prior
    score's own divergence is constant in the network parameters and is dropped. Returns the
    per-sample loss and f(y) (for reuse by the emulator term).
    """
    with torch.enable_grad():
        y = y.detach().requires_grad_(True)
        z = f(y)
        ell = -0.5 * (z - t).square().sum(-1)
        s = torch.autograd.grad(ell.sum(), y, create_graph=True)[0]          # (B, d)
        div = sum(torch.autograd.grad(s[:, a].sum(), y, create_graph=True)[0][:, a] for a in range(y.shape[-1]))
        if prior_score is not None:
            s = s + prior_score(y)
    return 0.5 * s.square().sum(-1) + div, z


def gmi_loss(eta, t, eps=1e-4):
    """1/2 log det Sigma_res - 1/2 log det Cov(eta), batch level (a scalar).

    Sigma_res = mean of r r^T with r = t - eta (gradients reach both towers). At the optimum this is
    sum_k 1/2 log(1 - rho_k^2) = -I_Gauss(eta; t), rho_k the canonical correlations.
    """
    r = t - eta
    S_res = r.T @ r / len(r)
    e = eta - eta.mean(0)
    S_eta = e.T @ e / (len(e) - 1)
    I = torch.eye(eta.shape[1], device=eta.device, dtype=eta.dtype)
    jit = lambda S: S + eps * (S.trace() / len(S)) * I              # relative jitter (loss is scale-invariant)
    return 0.5 * (torch.logdet(jit(S_res)) - torch.logdet(jit(S_eta)))


def logit_prior_score(y):
    """d/dy log density of y = logit(u)/LOGIT_SCALE for u ~ U(0, 1), per coordinate."""
    return LOGIT_SCALE * (1 - 2 * torch.sigmoid(LOGIT_SCALE * y))


class SharedHead(nn.Module):
    def __init__(self, K, lo, hi, critic="gauss", base="prior", normalised=True,
                 coords="box", width=64, nested_k=0, manifold_weight=0.0, info="quad",
                 emulator=False, act=nn.GELU):
        super().__init__()
        if info not in ("quad", "nce", "hyv"):
            raise ValueError(info)
        if info != "quad" and not (normalised and critic == "gauss" and base == "prior"):
            raise ValueError("nce/hyv need the normalised Gaussian critic with the prior base")
        if info == "hyv" and coords != "logit":
            raise ValueError("hyv works in logit coordinates (the box boundary term does not vanish)")
        if emulator and not (critic == "gauss" and base == "prior" and normalised):
            raise ValueError("the emulator term is defined for the normalised Gaussian critic")
        self.info = info
        self.beta = 0.0                     # emulator weight; set per step by the training loop
        self.emu = EmulatorTerm(K) if emulator else None
        # >0: add  w * 1/2 dist(t, eta(Theta))^2  -- a bounded stand-in for the -log Z(t) term of the
        # rectangular loss (equal to it, up to a constant, for a flat embedding); keeps t on the manifold
        self.manifold_weight = manifold_weight
        self.nested_k = nested_k   # >0: also train q_P from the first nested_k coordinates
        if critic not in ("gauss", "expfam") or base not in ("prior", "jeffreys"):
            raise ValueError((critic, base))
        self.K, self.critic, self.base, self.normalised = K, critic, base, normalised
        self.eta = ParamEmbed(K, lo, hi, width=width, coords=coords, act=act)
        if critic == "expfam":
            self.A = mlp(2, 1, width)
        self.register_buffer("lo", torch.as_tensor(lo, dtype=torch.float32))
        self.register_buffer("hi", torch.as_tensor(hi, dtype=torch.float32))
        self.quad_train = dict(n0=32, n=16, levels=2, jitter=True)
        self.quad_eval = dict(n0=64, n=32, levels=3, jitter=False)

    # -- unnormalised log integrand ------------------------------------------------------
    def _embed(self, theta):
        if self.base == "jeffreys":
            z, lv = self.eta.log_volume(theta)
            return z, lv
        return self.eta(theta), torch.zeros(theta.shape[:-1], device=theta.device)

    def log_f(self, theta, t, k=None):
        """theta (B, M, 2) or (1, M, 2); t (B, K) -> (B, M). ``k``: use the first k coordinates."""
        z, lb = self._embed(theta)
        if k is not None:
            z, t = z[..., :k], t[..., :k]
        if self.critic == "gauss":
            c = -0.5 * (z - t[:, None, :]).square().sum(-1) - 0.5 * z.shape[-1] * math.log(2 * math.pi)
        else:
            c = (z * t[:, None, :]).sum(-1) - self.A(self.eta.to_coords(theta)[0]).squeeze(-1)
        return c + lb

    def cells(self, t, train=False, k=None):
        q = self.quad_train if train else self.quad_eval
        return integrate(lambda p: self.log_f(p, t, k), self.lo, self.hi, t.shape[0], **q)

    # -- objectives / evaluation -----------------------------------------------------------
    def loss(self, theta, t):
        """Training objective per sample. In eval mode every normalised head returns the
        quadrature-normalised NLL (plus the manifold term if set), so validation is comparable."""
        if self.training and self.info != "quad":
            if self.info == "nce":
                z = self.eta(theta)
                L = infonce_loss(z, t)
            else:
                y = self.eta.to_coords(theta)[0]
                L, z = hyvarinen_loss(self.eta.f, y, t, logit_prior_score)
            return L + self._emulator(z, t)
        if self.training and self.emu is not None and self.info == "quad":
            z = self.eta(theta)
            lf = -0.5 * (z - t).square().sum(-1) - 0.5 * self.K * math.log(2 * math.pi)
            return self._quad_nll(lf, t) + self._emulator(z, t)
        lf = self.log_f(theta[:, None, :], t)[:, 0]
        return self._quad_nll(lf, t)

    def _emulator(self, z, t):
        if self.emu is None:
            return torch.zeros_like(z[:, 0])
        e = self.emu(z, t)
        return self.beta * e if self.beta > 0 else 0.0 * e

    def _quad_nll(self, lf, t):
        if not self.normalised:
            return -lf
        cells = self.cells(t, train=self.training)
        nll = -(lf - cells.logZ)
        if self.manifold_weight and self.critic == "gauss" and self.base == "prior":
            # max over all quadrature points of -1/2||eta - t||^2  ->  -1/2 dist^2 (grid approximation)
            c = cells.logf + 0.5 * self.K * math.log(2 * math.pi)
            nll = nll + self.manifold_weight * (-c.max(-1).values)
        if self.nested_k and self.training:
            k = self.nested_k
            lfk = self.log_f(theta[:, None, :], t, k)[:, 0]
            nll = nll - (lfk - self.cells(t, train=True, k=k).logZ)
        return nll

    @torch.no_grad()
    def evaluate(self, theta, t):
        """Normalised log q(theta|x) on the box, HPD level, posterior mean/std, log Z."""
        c = self.cells(t)
        lf = self.log_f(theta[:, None, :], t)[:, 0]
        mean, std = c.moments()
        cdf = torch.stack([c.marginal_cdf_at(theta[:, a], a) for a in range(2)], -1)
        return dict(log_q=lf - c.logZ, hpd=c.hpd_level(lf), mean=mean, std=std, logZ=c.logZ,
                    cdf=cdf), c

    @torch.no_grad()
    def fisher(self, theta):
        """J^T J of eta at theta (the Fisher matrix of t ~ N(eta(theta), I) in theta)."""
        y, _ = self.eta.to_coords(theta)
        cols = []
        for a in range(2):
            e = torch.zeros_like(y); e[..., a] = 1
            cols.append(torch.func.jvp(self.eta.f, (y,), (e,))[1])
        J = torch.stack(cols, -1)
        if self.eta.coords == "box":
            J = J * (2 / (self.hi - self.lo))
        else:                               # logit coordinates: dy/dtheta = 1 / (s (hi-lo) u (1-u))
            u = (theta - self.lo) / (self.hi - self.lo)
            J = J / ((math.pi / math.sqrt(3)) * (self.hi - self.lo) * u * (1 - u))[..., None, :]
        return J.transpose(-1, -2) @ J


class GMIHead(nn.Module):
    """Gaussian-MI embedding objective with a separate emulator tower.

    ``eta`` is trained by the GMI loss together with t. ``eta_emu`` (same architecture) is trained only
    by the stop-gradient emulator term, with weight ``beta`` (scheduled by the training loop), so it
    neither affects t nor the GMI tower. The GMI optimum for eta is inflated relative to E[t | theta]
    (gauge 1/rho_k^2 per canonical direction); eta_emu is the calibrated emulator.
    """

    info = "gmi"

    def __init__(self, K, lo, hi, width=64, act=nn.SiLU):
        super().__init__()
        self.K = K
        self.eta = ParamEmbed(K, lo, hi, width=width, coords="box", act=act)
        self.eta_emu = ParamEmbed(K, lo, hi, width=width, coords="box", act=act)
        self.emu = EmulatorTerm(K)
        self.beta = 0.0
        self.register_buffer("lo", torch.as_tensor(lo, dtype=torch.float32))
        self.register_buffer("hi", torch.as_tensor(hi, dtype=torch.float32))

    def loss(self, theta, t):
        """Per-sample view of the batch-level loss (the scalar repeated), so ``.mean()`` is the loss."""
        L = gmi_loss(self.eta(theta), t)
        if self.training:
            e = self.emu(self.eta_emu(theta), t)
            L = L + (self.beta * e if self.beta > 0 else 0.0 * e)
        return L.expand(len(theta))
