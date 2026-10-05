"""Inference 'arms': what each NPE conditions on and in which coordinates it models theta.

    raw      q(theta | x)      all n_rep x (2D-1) entries, flattened (uncompressed baseline)
    xbar     q(theta | xbar)   replicate means (known sufficient statistic; optional reference)
    theta_t  q(theta | t)      t = eta_psi(x), the learned summary
    eta_t    q(eta | t)        eta = eta_phi(theta), mapped back with the exact inverse and Jacobian

Every arm returns samples and log-densities in physical theta, truncated to the prior box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .maps import Maps

ARMS = ("raw", "xbar", "theta_t", "eta_t")
USES_MAPS = {"raw": False, "xbar": False, "theta_t": True, "eta_t": True}


def context_for(arm: str, x: torch.Tensor, maps: Maps | None) -> torch.Tensor:
    if arm == "raw":
        return x.reshape(len(x), -1)
    if arm == "xbar":
        return x.mean(-2)
    if arm in ("theta_t", "eta_t"):
        return maps.summarise(x)
    raise KeyError(arm)


@torch.no_grad()
def target_for(arm: str, theta: torch.Tensor, maps: Maps | None):
    """Target in the estimator's coordinates and log |det d target / d theta|."""
    if arm == "eta_t":
        return maps.to_eta(theta)
    return theta, theta.new_zeros(theta.shape[:-1])


@torch.no_grad()
def target_to_theta(arm: str, target: torch.Tensor, maps: Maps | None) -> torch.Tensor:
    if arm == "eta_t":
        return maps.to_theta(target)[0]
    return target


def inside_box(theta: torch.Tensor, a_box: float) -> torch.Tensor:
    return (theta.abs() <= a_box).all(-1)


@dataclass
class PosteriorDraws:
    samples: np.ndarray      # (B, S, D) draws in theta, inside the prior box
    acceptance: np.ndarray   # (B,) fraction of flow draws inside the prior box
    n_topped_up: int = 0     # observations whose S draws had to be completed by resampling


@torch.no_grad()
def sample_posteriors(net, arm: str, contexts: torch.Tensor, n: int, maps: Maps | None,
                      a_box: float, chunk: int = 64, max_draws_factor: float = 10.0,
                      max_batch_draws: int = 100_000, seed: int = 0) -> PosteriorDraws:
    """Rejection-sample n in-box theta draws per context (the truncation sbi applies).

    Each observation gets at most ``max_draws_factor * n`` flow draws, so acceptance rates
    down to 1 / max_draws_factor are handled exactly. Below that, the accepted draws are
    resampled with replacement up to n (or, if none were accepted, the draws are clipped
    to the box), and the observation is counted in ``n_topped_up``.
    """
    rng = np.random.default_rng(seed)
    cap = int(max_draws_factor * n)
    all_samples, all_acc, topped = [], [], 0
    for start in range(0, len(contexts), chunk):
        ctx = contexts[start:start + chunk]
        b = len(ctx)
        kept = [[] for _ in range(b)]
        have = np.zeros(b, dtype=int)
        drawn, accepted = np.zeros(b), np.zeros(b)
        last = None
        while True:
            active = (have < n) & (drawn < cap)
            if not active.any():
                break
            acc = np.where(drawn > 0, (accepted + 1) / (drawn + 1), 1.0)
            want = np.ceil(1.2 * (n - have) / acc) + 16
            m = int(min(max_batch_draws // b, want[active].max(), (cap - drawn[active]).max()))
            m = max(m, 16)
            draws = net.sample((m,), ctx)                                  # (m, b, dim_target)
            theta = target_to_theta(arm, draws.reshape(-1, draws.shape[-1]), maps).reshape(draws.shape)
            ok = inside_box(theta, a_box)
            theta, ok = theta.cpu().numpy(), ok.cpu().numpy()
            last = theta
            for i in np.flatnonzero(active):
                drawn[i] += m
                good = theta[ok[:, i], i]
                accepted[i] += len(good)
                take = good[: n - have[i]]
                kept[i].append(take)
                have[i] += len(take)
        for i in np.flatnonzero(have < n):
            pool = np.concatenate(kept[i]) if have[i] > 0 else None
            if pool is not None and len(pool) >= 2:
                kept[i].append(pool[rng.integers(len(pool), size=n - have[i])])
            else:
                kept[i].append(np.clip(last[: n - have[i], i], -a_box, a_box))
            topped += 1
        all_samples.append(np.stack([np.concatenate(k)[:n] for k in kept]))
        all_acc.append(np.maximum(accepted, 1.0) / np.maximum(drawn, 1.0))
    return PosteriorDraws(np.concatenate(all_samples), np.concatenate(all_acc), topped)


@torch.no_grad()
def log_prob_theta(net, arm: str, theta: torch.Tensor, contexts: torch.Tensor, maps: Maps | None,
                   acceptance: np.ndarray | None = None, batch: int = 4096) -> np.ndarray:
    """log q(theta | x) in theta coordinates, one theta per context.

    With ``acceptance`` the density is renormalised to the prior box (log q - log P_q(box)).
    Without it, this is the flow density that sbi uses for its validation loss.
    """
    out = []
    for i in range(0, len(theta), batch):
        target, logdet = target_for(arm, theta[i:i + batch], maps)
        lp = net.log_prob(target[None], contexts[i:i + batch])[0] + logdet
        out.append(lp.cpu().numpy())
    lp = np.concatenate(out)
    if acceptance is not None:
        lp = lp - np.log(np.clip(acceptance, 1e-12, 1.0))
    return lp
