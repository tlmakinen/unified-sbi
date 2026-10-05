"""Shear-calibration nuisance on the (Omega_m, sigma8) lensing toy: theta = (Omega_m, sigma8, m), d = 3.

The observed convergence is (1 + m) kappa, so C_ell -> (1 + m)^2 C_ell and m is exactly degenerate with the
amplitude. Prior: Omega_m and sigma8 uniform on the sigma8 box of ``lensing.param_box``, and
m ~ N(0, m_sigma^2) truncated to +-3 m_sigma. The Gaussian-field likelihood stays exact; the oracle
normalises it on a 3D grid (an eigen-decomposition per (Omega_m, sigma8) point makes the m axis cheap).

Arms (all use the hybrid summary MLP([Pk, CNN])):

* ``hybrid_maf``        MAF over theta (standard NPE), context dimension cfg.emb_dim.
* ``hybrid_shared_K2``  normalised gauss head, prior as base measure, Z on a jittered 3D grid. Reference.
* ``hybrid_rect_K2``    rectangular loss with K < d:  1/2 ||eta - t||^2 - log vol J,  vol J = sqrt det(J J^T)
                        (the product of the K singular values of J = d eta / d theta). No prior term.
* ``hybrid_rect_sq``    square-completed rectangular loss: eta: R^3 -> R^3, t padded with one zero, so the
                        third eta coordinate sees no data:  1/2 ||eta_a - t||^2 + 1/2 eta_b^2 - log |det J|.

Rectangular arms use logit coordinates (as ``hybrid_rect``) and are normalised on the grid only for evaluation.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .nets import CrossSpectrum, DataSummary, LogitBox, Standardise, mlp
from .study import Data, StudyConfig, evaluate_maf, metrics_from, train

ARMS = {
    "hybrid_maf": dict(head="maf"),
    "hybrid_shared_K2": dict(head="shared", K=2),
    "hybrid_rect_K2": dict(head="rect", K=2),
    "hybrid_rect_sq": dict(head="rect", K=3, pad=1),
}


# =============================================================================== data

class NuisanceData:
    """Signal pools of the sigma8 study times (1 + m); fixed noisy test set; exact 3D-grid oracle."""

    def __init__(self, cfg: StudyConfig, cache, m_sigma=0.05, oracle_grid=(96, 96, 48)):
        if cfg.lensing.get("params", "S8") != "sigma8" or cfg.lensing.get("field", "gaussian") != "gaussian":
            raise ValueError("NuisanceData needs the Gaussian field in sigma8 coordinates")
        self.cfg, self.cache = cfg, Path(cache)
        self.base = Data(cfg, cache)
        self.sim, self.lcfg = self.base.sim, self.base.lcfg
        self.m_sigma = float(m_sigma)
        self.lo = np.r_[self.sim.lo, -3 * m_sigma]
        self.hi = np.r_[self.sim.hi, 3 * m_sigma]
        self.tag = self.base.tag + f"_m{m_sigma:g}"
        self.oracle_grid = tuple(oracle_grid)

    # ---- prior
    def log_prior(self, theta):
        """Unnormalised log prior (Gaussian in m; Omega_m, sigma8 flat on the box). numpy or torch."""
        return -0.5 * (theta[..., 2] / self.m_sigma) ** 2

    def sample_m(self, n, rng):
        out = np.empty(0)
        while len(out) < n:
            z = rng.standard_normal(2 * n + 16)
            out = np.r_[out, z[np.abs(z) < 3]]
        return self.m_sigma * out[:n]

    # ---- simulations
    def pool(self, seed):
        th2, x, val = self.base.pool(seed)
        m = self.sample_m(len(th2), np.random.default_rng(3000 + seed))
        return np.c_[th2, m], (x * (1 + m)[:, None, None, None]).astype(np.float32), val

    def test(self):
        path = self.cache / f"test_{self.tag}_n{self.cfg.test_n}.npz"
        if path.exists():
            d = np.load(path)
            return d["theta"], d["x"]
        rng = np.random.default_rng(77)
        th2 = self.sim.prior_sample(self.cfg.test_n, rng)
        m = self.sample_m(self.cfg.test_n, rng)
        sig = (self.sim.simulate(th2, rng) * (1 + m)[:, None, None, None]).astype(np.float32)
        x = self.sim.add_noise(sig, np.random.default_rng(78))
        theta = np.c_[th2, m]
        np.savez(path, theta=theta, x=x)
        return theta, x

    # ---- exact posterior
    def grid(self, n):
        axes = [self.lo[i] + (self.hi[i] - self.lo[i]) * (np.arange(n[i]) + 0.5) / n[i] for i in range(3)]
        cv = float(np.prod((self.hi - self.lo) / np.asarray(n)))
        return axes, cv

    def loglik(self, stats, theta):
        """Exact log p(x | theta) up to a constant; stats (A, nb, nb) of one map, theta (M, 3)."""
        theta = np.atleast_2d(theta)
        S = self.sim.signal_sigma(theta[:, :2])
        a = (1 + theta[:, 2]) ** 2
        Sig = a[:, None, None, None] * S + self.sim.sigma_n ** 2 * np.eye(self.lcfg.n_bins)
        inv = np.linalg.inv(Sig)
        logdet = np.linalg.slogdet(Sig)[1]
        quad = np.einsum("maij,aji->ma", inv, stats)
        cnt = self.sim.geo["counts"].astype(np.float64)
        return -0.5 * (quad + cnt * logdet).sum(-1)

    def grid_logpost(self, n):
        """Callable stats -> exact log posterior (unnormalised) on the 3D grid ``n``, and the grid points.

        Points are ordered (m, Omega_m, sigma8) with m slowest: reshape a (M,) array to (n[2], n[0], n[1]).
        An eigen-decomposition of the signal covariance per (Omega_m, sigma8) point makes the m axis cheap.
        """
        axes, cv = self.grid(n)
        G2 = np.stack(np.meshgrid(axes[0], axes[1], indexing="ij"), -1).reshape(-1, 2)
        s, U = np.linalg.eigh(self.sim.signal_sigma(G2))                 # (g, A, nb), (g, A, nb, nb)
        cnt = self.sim.geo["counts"].astype(np.float64)
        sn2 = self.sim.sigma_n ** 2
        am = (1 + axes[2]) ** 2
        invden = np.empty((len(am), len(G2), s.shape[1] * s.shape[2]))
        logdet = np.empty((len(am), len(G2)))
        for i, a in enumerate(am):
            den = a * s + sn2
            invden[i] = (1 / den).reshape(len(G2), -1)
            logdet[i] = (cnt[None, :, None] * np.log(den)).sum((1, 2))
        lp_m = -0.5 * (axes[2] / self.m_sigma) ** 2
        P = np.concatenate([np.repeat(G2[None], len(am), 0),
                            np.broadcast_to(axes[2][:, None, None], (len(am), len(G2), 1))], -1).reshape(-1, 3)

        def logpost(stats):
            w = np.einsum("gaik,aij,gajk->gak", U, stats, U).reshape(len(G2), -1)
            quad = np.einsum("mgk,gk->mg", invden, w)
            return (-0.5 * (quad + logdet) + lp_m[:, None]).ravel()
        return logpost, P, cv, axes

    def oracle(self):
        path = self.cache / f"oracle_{self.tag}_n{self.cfg.test_n}_g{'x'.join(map(str, self.oracle_grid))}.npz"
        if path.exists():
            return dict(np.load(path))
        theta, x = self.test()
        stats = self.sim.sufficient_stats(x)
        logpost, P, cv, _ = self.grid_logpost(self.oracle_grid)
        out = {k: [] for k in ("log_q", "hpd", "mean", "std", "cdf", "logZ")}
        t0 = time.time()
        for j in range(len(theta)):
            lt = float(self.loglik(stats[j], theta[j:j + 1])[0] + self.log_prior(theta[j]))
            r = grid_summary(logpost(stats[j]), lt, P, cv, theta[j])
            for k in out:
                out[k].append(r[k])
            if j % 64 == 0:
                print(f"nuisance oracle {j}/{len(theta)}  {time.time() - t0:.0f}s", flush=True)
        res = {k: np.array(v) for k, v in out.items()}
        np.savez(path, **res)
        return res


def grid_summary(lf, lt, P, cv, theta_true):
    """Normalised log q at the truth, HPD level, moments and marginal CDFs from a log density on grid cells."""
    lf = np.asarray(lf, np.float64)
    mx = lf.max()
    lse = mx + np.log(np.exp(lf - mx).sum())
    p = np.exp(lf - lse)
    mean = p @ P
    std = np.sqrt(np.clip(p @ (P - mean) ** 2, 0, None))
    cdf = np.array([p[P[:, k] < theta_true[k]].sum() for k in range(P.shape[1])])
    return dict(log_q=lt - (lse + np.log(cv)), hpd=float(p[lf > lt].sum()), mean=mean, std=std, cdf=cdf,
                logZ=float(lse + np.log(cv)))


# =============================================================================== heads

class EtaD(nn.Module):
    """eta: R^d -> R^K, linear skip plus MLP on box or logit coordinates; Jacobian and log vol in theta."""

    def __init__(self, K, lo, hi, width=64, coords="box"):
        super().__init__()
        d = len(lo)
        self.register_buffer("lo", torch.as_tensor(lo, dtype=torch.float32))
        self.register_buffer("hi", torch.as_tensor(hi, dtype=torch.float32))
        self.K, self.d, self.coords = K, d, coords
        self.lin = nn.Linear(d, K)
        self.net = mlp(d, K, width)
        nn.init.normal_(self.net[-1].weight, std=1e-2)
        nn.init.zeros_(self.net[-1].bias)

    def to_coords(self, theta):
        u = (theta - self.lo) / (self.hi - self.lo)
        if self.coords == "box":
            return 2 * u - 1, (2 / (self.hi - self.lo)).expand_as(theta)
        s = math.pi / math.sqrt(3)
        u = u.clamp(1e-6, 1 - 1e-6)
        return (torch.log(u) - torch.log1p(-u)) / s, 1 / (s * (self.hi - self.lo) * u * (1 - u))

    def f(self, y):
        return self.lin(y) + self.net(y)

    def forward(self, theta):
        return self.f(self.to_coords(theta)[0])

    def jacobian(self, theta):
        """J = d eta / d theta, (..., K, d), by forward-mode JVPs (differentiable for training)."""
        y, dyd = self.to_coords(theta)
        cols = []
        for a in range(self.d):
            e = torch.zeros_like(y); e[..., a] = 1
            cols.append(torch.func.jvp(self.f, (y,), (e,))[1] * dyd[..., a:a + 1])
        return torch.stack(cols, -1)

    def log_vol(self, theta):
        """log vol J = 1/2 log det(J^T J) if K >= d, 1/2 log det(J J^T) if K < d (sum of log singular values)."""
        J = self.jacobian(theta).double()
        G = J @ J.transpose(-1, -2) if self.K < self.d else J.transpose(-1, -2) @ J
        return (0.5 * torch.logdet(G)).to(theta.dtype)


class GridSharedHead(nn.Module):
    """q(theta | x) = exp(-1/2 ||eta(theta) - t||^2) pi(theta) / Z(t), Z on a (jittered) grid over the box."""

    kind = "shared"

    def __init__(self, K, lo, hi, log_prior, n_train=(40, 40, 16)):
        super().__init__()
        self.eta = EtaD(K, lo, hi, coords="box")
        self.log_prior = log_prior
        lo_, hi_ = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
        n = np.asarray(n_train)
        axes = [lo_[i] + (hi_[i] - lo_[i]) * (np.arange(n[i]) + 0.5) / n[i] for i in range(len(n))]
        C = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(n))
        self.register_buffer("C", torch.as_tensor(C, dtype=torch.float32))
        self.register_buffer("w", torch.as_tensor((hi_ - lo_) / n, dtype=torch.float32))
        self.logcv = float(np.log(np.prod((hi_ - lo_) / n)))

    def extra(self, theta):
        return self.log_prior(theta)

    def tt(self, t):
        return t

    def log_f(self, theta, t):
        e = self.eta(theta)
        return -0.5 * (t[:, None, :] - e[None]).square().sum(-1) + self.log_prior(theta)[None]

    def loss(self, theta, t):
        e = self.eta(theta)
        lf = -0.5 * (t - e).square().sum(-1) + self.log_prior(theta)
        P = self.C + (torch.rand_like(self.C) - 0.5) * self.w if self.training else self.C
        logZ = torch.logsumexp(self.log_f(P, t), -1) + self.logcv
        return -(lf - logZ)


class RectHead(nn.Module):
    """Unnormalised rectangular loss 1/2 ||eta(theta) - t~||^2 - log vol J (t~ = t padded with ``pad`` zeros)."""

    kind = "rect"

    def __init__(self, K, lo, hi, pad=0):
        super().__init__()
        self.eta = EtaD(K, lo, hi, coords="logit")
        self.pad = pad

    def tt(self, t):
        return torch.cat([t, t.new_zeros(*t.shape[:-1], self.pad)], -1) if self.pad else t

    def extra(self, theta):
        return self.eta.log_vol(theta)

    def loss(self, theta, t):
        e = self.eta(theta)
        return 0.5 * (self.tt(t) - e).square().sum(-1) - self.eta.log_vol(theta)


class MAFHeadD(nn.Module):
    """zuko MAF over a d-dimensional box (logit coordinates), conditioned on the summary."""

    kind = "maf"

    def __init__(self, context, lo, hi, transforms=5, hidden=50):
        super().__init__()
        import zuko
        self.flow = zuko.flows.MAF(features=len(lo), context=context, transforms=transforms,
                                   hidden_features=[hidden, hidden])
        self.box = LogitBox(lo, hi)

    def log_prob(self, theta, c):
        y, ld = self.box.forward(theta)
        return self.flow(c).log_prob(y) + ld

    @torch.no_grad()
    def sample_and_log_prob(self, c, n):
        y, lp = self.flow(c).rsample_and_log_prob((n,))
        th = self.box.inverse(y)
        _, ld = self.box.forward(th)
        return th.transpose(0, 1), (lp + ld).transpose(0, 1)


class NuisPipeline(nn.Module):
    def __init__(self, arm, cfg: StudyConfig, data: NuisanceData):
        super().__init__()
        spec = ARMS[arm]
        self.arm, self.spec = arm, spec
        n = data.lcfg.n
        self.register_buffer("sigma_n", torch.as_tensor(np.asarray(data.sim.sigma_n, np.float32)), persistent=False)
        self.pk = CrossSpectrum(n, 4, cfg.pk_bins)
        self.pk_std = Standardise(self.pk.dim)
        self.im_std = Standardise(4, (1, 4, 1, 1))
        if spec["head"] == "maf":
            out = cfg.emb_dim
        else:
            out = spec["K"] - spec.get("pad", 0)
        self.summary = DataSummary("hybrid", out, n, self.pk.dim, cfg.cnn_width, cfg.cnn_extra, cfg.k_pk)
        if spec["head"] == "maf":
            self.head = MAFHeadD(out, data.lo, data.hi, cfg.maf_transforms, cfg.maf_hidden)
        elif spec["head"] == "shared":
            self.head = GridSharedHead(spec["K"], data.lo, data.hi, data.log_prior)
        else:
            self.head = RectHead(spec["K"], data.lo, data.hi, pad=spec.get("pad", 0))

    def fit_preprocessing(self, x_noisy):
        self.pk_std.fit(self.pk(x_noisy))
        self.im_std.fit(x_noisy, dims=(0, 2, 3))

    def t(self, x):
        return self.summary(self.im_std(x), self.pk_std(self.pk(x)))

    def loss(self, theta, x):
        t = self.t(x)
        if self.spec["head"] == "maf":
            return -self.head.log_prob(theta, t)
        return self.head.loss(theta, t)


@torch.no_grad()
def grid_eval(head, t_test, th_t, data: NuisanceData, n=(80, 80, 32), chunk=16384):
    """Normalise a shared or rect head on a 3D grid; per-map log q at the truth, HPD, moments, CDFs."""
    axes, cv = data.grid(n)
    P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    Pt = torch.as_tensor(P, dtype=torch.float32)
    E = torch.cat([head.eta(Pt[i:i + chunk]) for i in range(0, len(Pt), chunk)]).double()
    X = torch.cat([head.extra(Pt[i:i + chunk]) for i in range(0, len(Pt), chunk)]).double()
    tht = torch.as_tensor(th_t, dtype=torch.float32)
    Et, Xt = head.eta(tht).double(), head.extra(tht).double()
    tt = head.tt(t_test).double()
    out = {k: [] for k in ("log_q", "hpd", "mean", "std", "cdf", "logZ")}
    for j in range(len(th_t)):
        lf = (-0.5 * (tt[j] - E).square().sum(-1) + X).numpy()
        lt = float(-0.5 * (tt[j] - Et[j]).square().sum() + Xt[j])
        r = grid_summary(lf, lt, P, cv, th_t[j])
        for k in out:
            out[k].append(r[k])
    return {k: np.array(v) for k, v in out.items()}


# =============================================================================== one cell

def run_cell(cfg: StudyConfig, data: NuisanceData, seed, budget, arm, out, log=print):
    torch.set_num_threads(cfg.threads)
    torch.set_flush_denormal(True)
    dest = Path(out) / f"seed{seed}" / f"N{budget}" / arm
    if (dest / "metrics.json").exists():
        return json.loads((dest / "metrics.json").read_text())
    dest.mkdir(parents=True, exist_ok=True)
    theta_all, x_all, val_all = data.pool(seed)
    th, xs, va = theta_all[:budget], x_all[:budget], val_all[:budget]
    th_t, x_t = data.test()
    sigma_n = data.sim.sigma_n
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
    vrng = np.random.default_rng(5000 + seed)
    xv = xs[va] + sigma_n * vrng.standard_normal(xs[va].shape).astype(np.float32)
    torch.manual_seed(10 * seed + 1)
    model = NuisPipeline(arm, cfg, data)
    tr = ~va
    model.fit_preprocessing(T(xs[tr] + sigma_n * vrng.standard_normal(xs[tr].shape).astype(np.float32)))
    started = time.time()
    info = train(model, list(model.parameters()), T(th[tr]), T(xs[tr]), T(th[va]), T(xv), cfg, seed, arm, log=log)
    model.eval()
    with torch.no_grad():
        t_test = torch.cat([model.t(T(x_t[i:i + 128])) for i in range(0, len(x_t), 128)])
    if model.spec["head"] == "maf":
        res = evaluate_maf(model.head, t_test, T(th_t), cfg.posterior_samples, 123)
        res = {k: v.detach().cpu().numpy() for k, v in res.items()}
    else:
        res = grid_eval(model.head, t_test, th_t, data)
    oracle = data.oracle()
    m = dict(arm=arm, seed=seed, budget=budget, n_params=int(sum(p.numel() for p in model.parameters())),
             m_sigma=data.m_sigma)
    m.update(metrics_from(res, th_t, oracle))
    m.update(epochs=info["epochs"], best_val=info["best_val"], seconds=time.time() - started)
    torch.save(model.state_dict(), dest / "model.pt")
    np.savez(dest / "test.npz", **{k: np.asarray(v) for k, v in res.items()})
    (dest / "history.json").write_text(json.dumps(info["history"]))
    (dest / "metrics.json").write_text(json.dumps(m, indent=1))
    log(f"[nuisance seed {seed} N {budget} {arm}] excess {m['excess_nll']:.3f}±{m['excess_nll_se']:.3f} "
        f"jcov {m['joint_cov_mae']:.3f} post_std {np.round(m['post_std'], 4).tolist()}  {m['seconds']:.0f}s")
    return m
