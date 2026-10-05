"""Scaling study for shared parameter/data representations on the tomographic lensing toy.

One *cell* = (seed, budget N, arm). Every arm sees the same N signal simulations (nested
across budgets within a seed), the same per-simulation train/validation labels, freshly
redrawn shape noise every epoch (noise is not a simulation), random dihedral symmetries, and
the same fixed noisy test set. One-stage arms use all N simulations for their single model.
The two-stage arm reproduces the agent's pipeline (representation on the first N/2, MAF on
the second N/2).
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .heads import GMIHead, SharedHead, gmi_loss
from .lensing import HI, LO, LensingConfig, LensingSimulator, param_box
from .nets import CrossSpectrum, DataSummary, MAFHead, Standardise
from .quadrature import integrate

ARMS = {
    "pk_maf":              dict(features="pk", head="maf"),
    "cnn_maf":             dict(features="cnn", head="maf"),
    "hybrid_maf":          dict(features="hybrid", head="maf"),
    "pk_shared":           dict(features="pk", head="shared"),
    "pklin_shared":        dict(features="pk_linear", head="shared"),
    "cnn_shared":          dict(features="cnn", head="shared"),
    "hybrid_shared":       dict(features="hybrid", head="shared"),
    "hybrid_split_shared": dict(features="hybrid_split", head="shared"),
    "hybrid_nested_shared": dict(features="hybrid_split", head="shared", nested=True),
    "hybrid_manifold_shared": dict(features="hybrid", head="shared", manifold=1.0),
    "hybrid_rect":         dict(features="hybrid", head="rect"),
    "hybrid_rect_2stage":  dict(features="hybrid", head="rect", two_stage=True),
    # K = d arms (sigma8 study): per-arm K overrides cfg.K; "affine": eta restricted to an affine map
    "hybrid_shared_K2":        dict(features="hybrid", head="shared", K=2),
    "hybrid_affine_shared_K2": dict(features="hybrid", head="shared", K=2, affine=True),
    "hybrid_rect_K2":          dict(features="hybrid", head="rect", K=2),
    "cnn_shared_K2":           dict(features="cnn", head="shared", K=2),
}

# Arms of the embedding-objective study (docs/shared_space.md section 10). All use a SiLU eta.
#   hybrid_{quad|nce|hyv}_b{beta}[_B{batch}]   info loss + beta * stop-gradient emulator on the same eta
#   hybrid_gmi[_B{batch}]                        Gaussian-MI objective + separate emulator tower (batch 256)
# Aliases: hybrid_hyv = hybrid_hyv_b0 (the bounded counterpart of hybrid_rect), hybrid_nce = hybrid_nce_b0.
_PARAM_ARM = re.compile(r"^hybrid_(quad|nce|hyv)_b([0-9.]+)(?:_B(\d+))?$")
_GMI_ARM = re.compile(r"^hybrid_gmi(?:_B(\d+))?$")
_ALIASES = {"hybrid_hyv": "hybrid_hyv_b0", "hybrid_nce": "hybrid_nce_b0"}


def arm_spec(arm):
    if arm in ARMS:
        return ARMS[arm]
    arm = _ALIASES.get(arm, arm)
    m = _GMI_ARM.match(arm)
    if m:
        return dict(features="hybrid", head="gmi", info="gmi", beta=1.0, batch=int(m.group(1) or 256))
    m = _PARAM_ARM.match(arm)
    if not m:
        raise KeyError(f"unknown arm {arm!r}")
    info, beta, bs = m.group(1), float(m.group(2)), m.group(3)
    spec = dict(features="hybrid", head="shared", info=info, beta=beta, act="silu")
    if bs:
        spec["batch"] = int(bs)
    return spec


def is_arm(arm):
    try:
        arm_spec(arm)
        return True
    except KeyError:
        return False


@dataclass
class StudyConfig:
    lensing: dict = field(default_factory=lambda: LensingConfig().to_dict())
    budgets: tuple = (250, 500, 1000, 2000, 4000)
    seeds: tuple = (0,)
    arms: tuple = tuple(a for a in ARMS if a not in ("hybrid_nested_shared", "pklin_shared", "hybrid_manifold_shared"))
    test_n: int = 256
    val_frac: float = 0.15
    K: int = 4                 # shared-space dimension
    k_pk: int = 2              # power-spectrum block of hybrid_split
    emb_dim: int = 8           # MAF context dimension
    cnn_width: int = 8
    cnn_extra: int = 8
    pk_bins: int = 6
    critic: str = "gauss"
    base: str = "prior"
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20
    maf_transforms: int = 5
    maf_hidden: int = 50
    posterior_samples: int = 1000
    augment: bool = True
    emu_start: float = 0.2    # emulator weight is 0 for this fraction of max_epochs * steps ...
    emu_ramp: float = 0.1     # ... then ramps linearly to beta over this fraction
    emu_select_after_ramp: bool = True   # early stopping / model selection only once beta is at full value
    eval_frozen: bool = True  # evaluation B: MAF on the frozen, standardised t(x) of every eta arm
    logz_every: int = 5       # E log Z(t) trace on 64 test maps every this many epochs (eta heads; 0: off)
    threads: int = 2
    device: str = "cpu"        # "cpu", "cuda" or "auto" (operational; not part of the config check)

    def lensing_cfg(self):
        d = dict(self.lensing)
        for k in ("z_sources", "lognormal_shift"):
            d[k] = tuple(d[k])
        return LensingConfig(**d)

    def to_dict(self):
        return asdict(self)


# =============================================================================== data

class Data:
    """Signal pools per seed, fixed noisy test set, exact oracle (Gaussian field only)."""

    def __init__(self, cfg: StudyConfig, cache: Path):
        self.cfg, self.cache = cfg, Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lcfg = cfg.lensing_cfg()
        self.sim = LensingSimulator(self.lcfg)
        self.tag = f"n{self.lcfg.n}_{self.lcfg.field}_noise{self.lcfg.noise_amp:g}"
        if self.lcfg.params != "S8":
            self.tag += f"_{self.lcfg.params}"

    def pool(self, seed):
        n = max(self.cfg.budgets)
        path = self.cache / f"pool_{self.tag}_seed{seed}_N{n}.npz"
        if path.exists():
            d = np.load(path)
            return d["theta"], d["x"], d["val"]
        rng = np.random.default_rng(1000 + seed)
        theta = self.sim.prior_sample(n, rng)
        x = self.sim.simulate(theta, rng)
        val = rng.random(n) < self.cfg.val_frac
        np.savez(path, theta=theta, x=x, val=val)
        return theta, x, val

    def test(self):
        path = self.cache / f"test_{self.tag}_n{self.cfg.test_n}.npz"
        if path.exists():
            d = np.load(path)
            return d["theta"], d["x"]
        rng = np.random.default_rng(77)
        theta = self.sim.prior_sample(self.cfg.test_n, rng)
        x = self.sim.add_noise(self.sim.simulate(theta, rng), np.random.default_rng(78))
        np.savez(path, theta=theta, x=x)
        return theta, x

    def oracle(self):
        """Exact log p(theta*|x), HPD level, moments and marginal CDFs for the test set."""
        if self.lcfg.field != "gaussian":
            return None
        path = self.cache / f"oracle_{self.tag}_n{self.cfg.test_n}.npz"
        if path.exists():
            return dict(np.load(path))
        theta, x = self.test()
        stats = self.sim.sufficient_stats(x)
        quad = dict(n0=64, n=32, levels=3)
        lo, hi = torch.tensor(self.sim.lo), torch.tensor(self.sim.hi)
        # level-0 grid is shared: precompute its covariance inverses once
        g0 = ((torch.arange(quad["n0"], dtype=torch.float64) + 0.5) / quad["n0"])
        g0 = torch.stack(torch.meshgrid(g0, g0, indexing="ij"), -1).reshape(-1, 2)
        g0 = (lo + (hi - lo) * g0).numpy()
        inv0, logdet0 = self._inv_logdet(g0)
        cnt = self.sim.geo["counts"].astype(np.float64)
        out = {k: [] for k in ("log_q", "hpd", "mean", "std", "cdf", "logZ")}
        t0 = time.time()
        for i in range(len(theta)):
            S = stats[i]

            def logf(p):
                P = p.reshape(-1, 2).numpy()
                if P.shape[0] == g0.shape[0] and np.allclose(P, g0):
                    ll = -0.5 * (np.einsum("maij,aji->ma", inv0, S) + cnt * logdet0).sum(-1)
                else:
                    ll = self.sim.loglik_from_stats(S, P)
                return torch.from_numpy(ll).reshape(p.shape[:-1])

            c = integrate(logf, lo, hi, 1, **quad)
            lt = logf(torch.tensor(theta[i:i + 1, None, :]))[:, 0]
            m, s = c.moments()
            out["log_q"].append((lt - c.logZ).item())
            out["hpd"].append(c.hpd_level(lt).item())
            out["mean"].append(m[0].numpy()); out["std"].append(s[0].numpy())
            out["cdf"].append(np.array([c.marginal_cdf_at(torch.tensor(theta[i:i + 1, a]), a).item() for a in range(2)]))
            out["logZ"].append(c.logZ.item())
            if i % 32 == 0:
                print(f"oracle {i}/{len(theta)}  {time.time() - t0:.0f}s", flush=True)
        res = {k: np.array(v) for k, v in out.items()}
        res["fisher"] = self.sim.exact_fisher(theta)
        np.savez(path, **res)
        return res

    def _inv_logdet(self, theta):
        eye = np.eye(self.lcfg.n_bins)
        Sig = self.sim.signal_sigma(theta) + self.sim.sigma_n ** 2 * eye
        sign, logdet = np.linalg.slogdet(Sig)
        return np.linalg.inv(Sig), logdet


class CatalogueData:
    """Drop-in for a fixed simulation catalogue such as the hybrid-statistics demo archive
    ``prior_S8_L_250_N_128_Nz_512.npz`` (keys prior_sims (N,4,n,n) and prior_theta (N,2)).

    Shape noise is ``noise_amp * sqrt(noisevars)`` per tomographic bin, redrawn per epoch,
    as in ``hybrid_scaling.py``. A fixed 20% of the catalogue is the test set; the rest is
    permuted per seed, and budgets take nested prefixes. No exact oracle exists.
    """
    NOISEVARS = (0.00045021, 0.00087473, 0.00134725, 0.00183411)

    def __init__(self, cfg: StudyConfig, path, noise_amp=0.125):
        d = np.load(path, allow_pickle=False)
        x, th = d["prior_sims"].astype(np.float32), d["prior_theta"].astype(np.float64)
        if x.ndim != 4 or th.shape != (len(x), 2):
            raise ValueError(f"unexpected catalogue shapes {x.shape}, {th.shape}")
        if np.any(th <= LO) or np.any(th >= HI):
            raise ValueError("theta outside the prior box")
        self.cfg = cfg
        self.lcfg = replace_n(cfg.lensing_cfg(), x.shape[-1])
        self.sim = type("S", (), {})()
        self.sim.sigma_n = (noise_amp * np.sqrt(np.array(self.NOISEVARS))[:, None, None]).astype(np.float32)
        # rows with identical theta (e.g. shared phases) stay on one side of the test split
        _, groups = np.unique(th, axis=0, return_inverse=True)
        groups = groups.ravel()
        gperm = np.random.default_rng(20260930).permutation(groups.max() + 1)
        order = np.argsort(np.argsort(gperm)[groups], kind="stable")
        n_test = min(cfg.test_n, len(x) // 5)
        test_groups = set(groups[order[:n_test]])
        is_test = np.isin(groups, list(test_groups))
        self._test = np.flatnonzero(is_test)[:max(n_test, 1)]
        self._pool = np.flatnonzero(~is_test)
        self.n_duplicate_theta = int(len(x) - groups.max() - 1)
        self.path, self.noise_amp = str(path), noise_amp
        self.x, self.theta = x, th
        if max(cfg.budgets) > len(self._pool):
            raise ValueError(f"largest budget exceeds the {len(self._pool)} non-test maps")

    def pool(self, seed):
        idx = np.random.default_rng(1000 + seed).permutation(self._pool)
        val = np.random.default_rng(2000 + seed).random(len(idx)) < self.cfg.val_frac
        return self.theta[idx], self.x[idx], val

    def test(self):
        rng = np.random.default_rng(78)
        x = self.x[self._test]
        return self.theta[self._test], x + self.sim.sigma_n * rng.standard_normal(x.shape).astype(np.float32)

    def oracle(self):
        return None


def replace_n(lcfg, n):
    from dataclasses import replace
    return replace(lcfg, n=int(n))


# =============================================================================== training

DIHEDRAL = [(k, f) for k in range(4) for f in (False, True)]


def augment(x, gen):
    k, f = DIHEDRAL[int(torch.randint(8, (1,), generator=gen))]
    x = torch.rot90(x, k, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if f else x


class Pipeline(torch.nn.Module):
    """Preprocessing + data summary + head for one arm."""

    def __init__(self, arm, cfg: StudyConfig, n, sigma_n):
        super().__init__()
        spec = arm_spec(arm)
        self.arm, self.spec = arm, spec
        self.register_buffer("sigma_n", torch.as_tensor(np.asarray(sigma_n, np.float32)), persistent=False)
        self.pk = CrossSpectrum(n, 4, cfg.pk_bins)
        self.pk_std = Standardise(self.pk.dim)
        self.im_std = Standardise(4, (1, 4, 1, 1))
        K = spec.get("K", cfg.K)
        LO, HI = param_box(cfg.lensing.get("params", "S8"))   # prior box of this study
        self.box = (LO, HI)
        out = cfg.emb_dim if spec["head"] == "maf" else K
        self.summary = DataSummary(spec["features"], out, n, self.pk.dim, cfg.cnn_width,
                                   cfg.cnn_extra, cfg.k_pk)
        if spec["head"] == "maf":
            self.head = MAFHead(out, LO, HI, cfg.maf_transforms, cfg.maf_hidden)
        elif spec["head"] == "shared":
            info = spec.get("info", "quad")
            self.head = SharedHead(K, LO, HI, critic=cfg.critic, base=cfg.base,
                                   nested_k=cfg.k_pk if spec.get("nested") else 0,
                                   manifold_weight=spec.get("manifold", 0.0), info=info,
                                   emulator="beta" in spec,
                                   coords="logit" if info == "hyv" else "box",
                                   act=torch.nn.SiLU if spec.get("act") == "silu" else torch.nn.GELU)
        elif spec["head"] == "gmi":
            self.head = GMIHead(K, LO, HI)
        else:   # the agent's rectangular one-step loss: unnormalised, Jeffreys base, logit coords
            self.head = SharedHead(K, LO, HI, critic="gauss", base="jeffreys",
                                   normalised=False, coords="logit")
        if spec.get("affine"):          # eta = A y + b only: zero and freeze the MLP part
            for prm in self.head.eta.net.parameters():
                prm.data.zero_()
                prm.requires_grad_(False)

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
    def val_loss(self, theta, x, chunk=256):
        """Mean held-out loss: quadrature NLL for shared heads, MAF NLL, or GMI on the whole set."""
        if self.spec["head"] == "gmi":
            t = torch.cat([self.t(x[i:i + chunk]) for i in range(0, len(x), chunk)])
            return gmi_loss(self.head.eta(theta), t).item()
        return torch.cat([self.loss(theta[i:i + chunk], x[i:i + chunk]) for i in range(0, len(x), chunk)]).mean().item()

    def has_eta(self):
        return self.spec["head"] in ("shared", "rect", "gmi")

    def emulator_eta(self):
        """The eta whose residual t - eta is the calibrated emulator (eta_emu for gmi)."""
        return self.head.eta_emu if self.spec["head"] == "gmi" else self.head.eta


def _batches(n, bs, gen):
    perm = torch.randperm(n, generator=gen)
    return [perm[i:i + bs] for i in range(0, n, bs)]


def train(model, params, theta, x, theta_v, x_v, cfg: StudyConfig, seed, label="",
          loss_fn=None, log=print, batch_size=None, beta=0.0, val_fn=None, monitor=None):
    """Minibatch AdamW with early stopping on a fixed-noise validation set.

    ``x`` are noise-free signals; noise is redrawn every time a map is used. Each batch holds
    one (augmented) copy of each of its simulations, with prior-drawn theta, so in-batch
    negatives are valid. ``beta > 0``: emulator weight schedule on ``model.head.beta``.
    ``val_fn(theta_v, x_v)`` -> scalar overrides the chunked mean of ``loss_fn``; ``monitor()`` -> float
    is recorded every ``cfg.logz_every`` epochs (in eval mode) as a fourth history column.
    """
    loss_fn = loss_fn or model.loss
    bs = batch_size or cfg.batch_size
    steps_per_epoch = math.ceil(len(theta) / bs)
    total = cfg.max_epochs * steps_per_epoch
    s0, s1 = cfg.emu_start * total, (cfg.emu_start + cfg.emu_ramp) * total
    select_from = math.ceil(s1 / steps_per_epoch) if (beta > 0 and cfg.emu_select_after_ramp) else 0
    step = 0
    gen = torch.Generator().manual_seed(seed)
    noise_gen = gen if x.device.type == "cpu" else torch.Generator(device=x.device).manual_seed(seed + 7)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    best, best_state, wait, hist, best_epoch = math.inf, None, 0, [], -1
    t0 = time.time()
    for epoch in range(cfg.max_epochs):
        model.train()
        tot = 0.0
        for idx in _batches(len(theta), bs, gen):
            if beta > 0:
                model.head.beta = beta * min(max((step - s0) / max(s1 - s0, 1), 0.0), 1.0)
            step += 1
            xb = x[idx] + model.sigma_n * torch.randn(x[idx].shape, generator=noise_gen, device=x.device)
            if cfg.augment:
                xb = augment(xb, gen)
            loss = loss_fn(theta[idx], xb).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{label}: nonfinite loss at epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            if val_fn is not None:
                val = val_fn(theta_v, x_v)
            else:
                val = torch.cat([loss_fn(theta_v[i:i + 256], x_v[i:i + 256]) for i in range(0, len(theta_v), 256)]).mean().item()
            mon = monitor() if (monitor is not None and cfg.logz_every and epoch % cfg.logz_every == 0) else None
        hist.append((epoch, tot / len(theta), val) + ((mon,) if mon is not None else ()))
        if epoch % 25 == 0:
            log(f"    {label} epoch {epoch}: train {tot / len(theta):.3f} val {val:.3f}  {time.time() - t0:.0f}s")
        if epoch < select_from:
            continue
        if val < best - 1e-4:
            best, wait = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            wait += 1
        if wait >= cfg.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    log(f"  {label}: {epoch + 1} epochs, best val {best:.3f}, {time.time() - t0:.0f}s")
    return dict(best_val=best, epochs=epoch + 1, best_epoch=best_epoch, seconds=time.time() - t0,
                history=hist, steps=step)


# =============================================================================== evaluation

@torch.no_grad()
def evaluate_maf(head, t, theta, n_samples, seed):
    torch.manual_seed(seed)
    lq = head.log_prob(theta, t)
    hpd, mean, std, cdf = [], [], [], []
    for i in range(0, len(t), 64):
        s, lp = head.sample_and_log_prob(t[i:i + 64], n_samples)
        hpd.append((lp > lq[i:i + 64, None]).float().mean(1))
        mean.append(s.mean(1)); std.append(s.std(1))
        cdf.append((s < theta[i:i + 64, None]).float().mean(1))
    return dict(log_q=lq, hpd=torch.cat(hpd), mean=torch.cat(mean), std=torch.cat(std),
                cdf=torch.cat(cdf), logZ=torch.zeros(len(t)))


def metrics_from(res, theta, oracle=None):
    levels = np.linspace(0.05, 0.95, 19)
    lq = np.asarray(res["log_q"], np.float64)
    hpd = np.asarray(res["hpd"]); cdf = np.asarray(res["cdf"])
    joint = np.array([(hpd <= a).mean() for a in levels])
    marg = np.stack([((cdf >= (1 - a) / 2) & (cdf <= (1 + a) / 2)).mean(0) for a in levels])
    m = dict(nll=float(-lq.mean()), nll_se=float(lq.std(ddof=1) / np.sqrt(len(lq))),
             joint_cov_mae=float(np.abs(joint - levels).mean()),
             marg_cov_mae=float(np.abs(marg - levels[:, None]).mean()),
             post_std=list(np.asarray(res["std"]).mean(0).astype(float)),
             rmse=list(np.sqrt(((np.asarray(res["mean"]) - theta) ** 2).mean(0)).astype(float)),
             mean_logZ=float(np.mean(res["logZ"])))
    if oracle is not None:
        d = np.asarray(oracle["log_q"]) - lq
        m.update(excess_nll=float(d.mean()), excess_nll_se=float(d.std(ddof=1) / np.sqrt(len(d))))
    return m


def emulator_diagnostics(eta_mod, theta, t, eta, oracle=None, grid=5, eta_info=None):
    """Fisher and emulator-calibration diagnostics on the test set.

    ``eta_mod`` is the emulator tower (eta, or eta_emu for gmi) and ``eta`` = eta_mod(theta*).

    * J^T J and J^T Sigma^-1 J log-det ratios against the exact Fisher (median, IQR), with
      Sigma the second moment of the test residuals t - eta(theta*);
    * eigenvalues of Sigma (the critic assumes I);
    * residual bias E[t - eta | theta] on a grid x grid binning of the prior box, summarised by
      chi2 = sum_b n_b m_b^T Sigma^-1 m_b over bins (approximately chi2 with K * bins dof if unbiased);
    * median residual norm (sqrt(K) if t ~ N(eta, I));
    * canonical correlations of t with eta and the harmonics R^2 (see ``harmonics``);
    * ``eta_info``: the information tower's eta(theta*) if different (gmi): -L_GMI on the test set.
    """
    r = (t - eta).astype(np.float64)
    K = r.shape[1]
    Sig = r.T @ r / len(r)
    Sinv = np.linalg.inv(Sig)
    with torch.no_grad():
        th = torch.as_tensor(theta, dtype=torch.float32, device=eta_mod.lo.device)
        J = _jacobian(eta_mod, th).double().cpu().numpy()                      # (n, K, 2)
    F1 = np.einsum("nka,nkb->nab", J, J)
    F2 = np.einsum("nka,kl,nlb->nab", J, Sinv, J)
    out = dict(sigma_eig=sorted(np.linalg.eigvalsh(Sig).tolist()),
               residual_norm_median=float(np.median(np.linalg.norm(r, axis=1))))
    lo_, hi_ = eta_mod.lo.cpu().numpy(), eta_mod.hi.cpu().numpy()
    u = (theta - lo_) / (hi_ - lo_)
    b = np.clip((u * grid).astype(int), 0, grid - 1)
    b = b[:, 0] * grid + b[:, 1]
    chi2, nb, bias_norm = 0.0, 0, []
    for k in np.unique(b):
        sel = b == k
        mk = r[sel].mean(0)
        chi2 += sel.sum() * mk @ Sinv @ mk
        nb += 1
        bias_norm.append(np.linalg.norm(mk))
    out.update(bias_chi2=float(chi2), bias_dof=int(nb * K), bias_norm_rms=float(np.sqrt(np.mean(np.square(bias_norm)))))
    rho, T = cca(t, eta_info if eta_info is not None else eta)
    out.update(canon_corr=rho.tolist(), harmonics_r2=harmonics(T))
    if eta_info is not None:
        with torch.no_grad():
            out["gmi_info"] = float(-gmi_loss(torch.as_tensor(eta_info, dtype=torch.float64),
                                              torch.as_tensor(t, dtype=torch.float64)))
    arrays = dict(fisher=F1.astype(np.float32), fisher_sigma=F2.astype(np.float32), sigma_hat=Sig)
    if oracle is not None:
        ld = np.linalg.slogdet(oracle["fisher"])[1]
        for name, F in (("fisher_logdet_ratio", F1), ("fisher_sigma_logdet_ratio", F2)):
            q = np.linalg.slogdet(F)[1] - ld
            out[name] = float(np.median(q))
            out[name + "_iqr"] = [float(np.percentile(q, 25)), float(np.percentile(q, 75))]
    return out, arrays


def cca(t, e, ridge=1e-9):
    """Canonical correlations rho (descending) of t with e and t's canonical variates (n, K), unit variance."""
    t = t - t.mean(0); e = e - e.mean(0)
    n = len(t)
    Ct, Ce, Cte = t.T @ t / n, e.T @ e / n, t.T @ e / n

    def isqrt(C):
        w, V = np.linalg.eigh(C + ridge * np.trace(C) * np.eye(len(C)))
        w = np.clip(w, 1e-12 * max(w.max(), 1e-300), None)      # rank-deficient t (e.g. t on a 2D surface)
        return V @ np.diag(w ** -0.5) @ V.T
    Wt, We = isqrt(Ct), isqrt(Ce)
    U, S, _ = np.linalg.svd(Wt @ Cte @ We)
    return np.clip(S, 0, 1), t @ (Wt @ U)


def harmonics(T, degree=3):
    """R^2 of each canonical variate k >= 2 regressed on degree-3 polynomials of the variates before it.

    High R^2: slot k re-encodes earlier directions (e.g. S8 and S8^2) instead of adding a new one.
    In-sample, so with p features and n points chance R^2 is about p / n.
    """
    from itertools import combinations_with_replacement
    out = []
    for k in range(1, T.shape[1]):
        prev = T[:, :k]
        cols = [np.ones(len(T))]
        for d in range(1, degree + 1):
            for c in combinations_with_replacement(range(k), d):
                cols.append(np.prod(prev[:, list(c)], axis=1))
        X = np.stack(cols, 1)
        coef, *_ = np.linalg.lstsq(X, T[:, k], rcond=None)
        res = T[:, k] - X @ coef
        out.append(float(1 - res.var() / T[:, k].var()))
    return out


def _jacobian(eta_mod, theta):
    """J = d eta / d theta, (n, K, 2), through box or logit coordinates."""
    lo, hi = eta_mod.lo, eta_mod.hi
    y, _ = eta_mod.to_coords(theta)
    cols = []
    for a in range(2):
        e = torch.zeros_like(y); e[..., a] = 1
        cols.append(torch.func.jvp(eta_mod.f, (y,), (e,))[1])
    J = torch.stack(cols, -1)
    if eta_mod.coords == "box":
        return J * (2 / (hi - lo))
    u = (theta - lo) / (hi - lo)
    return J / ((math.pi / math.sqrt(3)) * (hi - lo) * u * (1 - u))[..., None, :]


def frozen_summary_eval(model, cfg, th, xs, tr, va, x_v, sigma_n, th_t, x_t, seed, T, log=print, fit_noise_seed=0):
    """Evaluation B: a MAF (the hybrid_maf head) on the frozen, standardised t(x) of a trained model.

    Same simulations and train/validation split as the embedding; noise redrawn every epoch;
    early stopping on validation NLL. Returns test-set posterior results, the training info, and
    the final train-minus-validation NLL gap (train NLL on one fixed noise draw of the training maps).
    """
    from .nets import Standardise
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    rng = np.random.default_rng(9000 + seed + fit_noise_seed)
    noisy = lambda a: T(a + sigma_n * rng.standard_normal(a.shape).astype(np.float32))
    x_tr_fixed = noisy(xs[tr])
    with torch.no_grad():
        t_tr = torch.cat([model.t(x_tr_fixed[i:i + 256]) for i in range(0, len(x_tr_fixed), 256)])
    std = Standardise(t_tr.shape[1]).to(t_tr.device).fit(t_tr)
    torch.manual_seed(10 * seed + 3)
    maf = MAFHead(t_tr.shape[1], *model.box, cfg.maf_transforms, cfg.maf_hidden).to(t_tr.device)
    model.frozen_maf = maf                   # registered so train()'s best-state checkpoint includes it

    def maf_loss(theta, x):
        with torch.no_grad():
            t = std(model.t(x))
        return -maf.log_prob(theta, t)
    info = train(model, list(maf.parameters()), T(th[tr]), T(xs[tr]), T(th[va]), x_v, cfg, seed + 1,
                 f"{model.arm}/frozenMAF", loss_fn=maf_loss, log=log)
    maf.eval()
    with torch.no_grad():
        tr_nll = torch.cat([maf_loss(T(th[tr])[i:i + 256], x_tr_fixed[i:i + 256]) for i in range(0, len(x_tr_fixed), 256)]).mean().item()
        t_test = torch.cat([std(model.t(T(x_t[i:i + 128]))) for i in range(0, len(x_t), 128)])
    res = evaluate_maf(maf, t_test, T(th_t), cfg.posterior_samples, 123)
    del model.frozen_maf
    return res, info, tr_nll - info["best_val"], maf


# =============================================================================== one cell

def run_cell(cfg: StudyConfig, data: Data, seed: int, budget: int, arm: str, out: Path, log=print):
    torch.set_num_threads(cfg.threads)
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device)
    torch.set_flush_denormal(True)      # softmax over wide log-ranges otherwise produces slow denormals
    dest = Path(out) / f"seed{seed}" / f"N{budget}" / arm
    if (dest / "metrics.json").exists():
        return json.loads((dest / "metrics.json").read_text())
    dest.mkdir(parents=True, exist_ok=True)
    theta_all, x_all, val_all = data.pool(seed)
    th, xs, va = theta_all[:budget], x_all[:budget], val_all[:budget]
    th_t, x_t = data.test()
    sigma_n = data.sim.sigma_n
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)
    vrng = np.random.default_rng(5000 + seed)
    xv_noisy_all = xs[va] + sigma_n * vrng.standard_normal(xs[va].shape).astype(np.float32)

    torch.manual_seed(10 * seed + 1)
    model = Pipeline(arm, cfg, data.lcfg.n, sigma_n).to(dev)
    spec = arm_spec(arm)
    started = time.time()
    if spec.get("two_stage"):
        half = budget // 2
        idx1, idx2 = np.arange(budget) < half, np.arange(budget) >= half
        tr1, va1 = idx1 & ~va, idx1 & va
        tr2, va2 = idx2 & ~va, idx2 & va
        model.fit_preprocessing(T(xs[tr1] + sigma_n * vrng.standard_normal(xs[tr1].shape).astype(np.float32)))
        x_v1 = T(xs[va1] + sigma_n * vrng.standard_normal(xs[va1].shape).astype(np.float32))
        info1 = train(model, list(model.summary.parameters()) + list(model.head.parameters()),
                      T(th[tr1]), T(xs[tr1]), T(th[va1]), x_v1, cfg, seed, f"{arm}/rep", log=log)
        for p in model.summary.parameters():
            p.requires_grad_(False)
        maf = MAFHead(spec.get("K", cfg.K), *model.box, cfg.maf_transforms, cfg.maf_hidden).to(dev)
        model.maf = maf
        x_v2 = T(xs[va2] + sigma_n * vrng.standard_normal(xs[va2].shape).astype(np.float32))

        def maf_loss(theta, x):
            with torch.no_grad():
                t = model.t(x)
            return -maf.log_prob(theta, t)
        info2 = train(model, list(maf.parameters()), T(th[tr2]), T(xs[tr2]), T(th[va2]), x_v2,
                      cfg, seed + 1, f"{arm}/maf", loss_fn=maf_loss, log=log)
        info = dict(rep=info1, maf=info2)
        with torch.no_grad():
            t_test = torch.cat([model.t(T(x_t[i:i + 128])) for i in range(0, len(x_t), 128)])
        res = evaluate_maf(maf, t_test, T(th_t), cfg.posterior_samples, 123)
        n_params = sum(p.numel() for n_, p in model.named_parameters() if not n_.startswith("head."))
    else:
        tr = ~va
        model.fit_preprocessing(T(xs[tr] + sigma_n * vrng.standard_normal(xs[tr].shape).astype(np.float32)))
        monitor = None
        if spec["head"] in ("shared", "rect") and cfg.logz_every:
            x_mon = T(x_t[:64])

            def monitor():                       # E log Z(t) on 64 test maps (coiling alarm for rect)
                return model.head.cells(model.t(x_mon)).logZ.mean().item()
        info = train(model, list(model.parameters()), T(th[tr]), T(xs[tr]), T(th[va]),
                     T(xv_noisy_all), cfg, seed, arm, log=log, batch_size=spec.get("batch"),
                     beta=spec.get("beta", 0.0), val_fn=model.val_loss, monitor=monitor)
        with torch.no_grad():
            t_test = torch.cat([model.t(T(x_t[i:i + 128])) for i in range(0, len(x_t), 128)])
        if spec["head"] == "maf":
            res = evaluate_maf(model.head, t_test, T(th_t), cfg.posterior_samples, 123)
        elif spec["head"] == "gmi":
            res = None                           # no posterior head of its own: evaluation B only
        else:
            res, _ = model.head.evaluate(T(th_t), t_test)
        n_params = sum(p.numel() for p in model.parameters())
    oracle = data.oracle()
    m = dict(arm=arm, seed=seed, budget=budget, n_params=int(n_params))
    if res is not None:
        res = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in res.items()}
        m.update(metrics_from(res, th_t, oracle))
    else:
        res = {}
    hist = info if "rep" in info else dict(main=info)
    m["epochs"] = {k: v["epochs"] for k, v in hist.items()}
    m["best_val"] = {k: v["best_val"] for k, v in hist.items()}
    if model.has_eta():
        m.update(eta_diagnostics(model, th_t, t_test, oracle, T)[0])
        m["best_epoch"] = info.get("best_epoch") if "rep" not in info else None
        torch.save(model.state_dict(), dest / "model.pt")
        if cfg.eval_frozen and not spec.get("two_stage"):
            resB, infoB, gap, _ = frozen_summary_eval(model, cfg, th, xs, tr, va, T(xv_noisy_all), sigma_n,
                                                      th_t, x_t, seed, T, log=log)
            resB = {k: v.detach().cpu().numpy() for k, v in resB.items()}
            m.update({f"B_{k}": v for k, v in metrics_from(resB, th_t, oracle).items()})
            m.update(B_gap=gap, B_epochs=infoB["epochs"], B_seconds=infoB["seconds"])
            hist["frozen_maf"] = infoB
            np.savez(dest / "test_B.npz", **{k: np.asarray(v) for k, v in resB.items()})
    if not model.has_eta():              # MAF arms too, so their posteriors can be plotted later
        torch.save(model.state_dict(), dest / "model.pt")
    m["seconds"] = time.time() - started
    (dest / "history.json").write_text(json.dumps({k: v["history"] for k, v in hist.items()}))
    np.savez(dest / "test.npz", **{k: np.asarray(v) for k, v in res.items()})
    (dest / "metrics.json").write_text(json.dumps(m, indent=1))
    msg = f"[seed {seed} N {budget} {arm}]"
    if "nll" in m:
        msg += f" A: nll {m['nll']:.3f}" + (f" excess {m['excess_nll']:.3f}±{m['excess_nll_se']:.3f}" if "excess_nll" in m else "")
        msg += f" jcov {m['joint_cov_mae']:.3f}"
    if "B_nll" in m:
        msg += f" | B: nll {m['B_nll']:.3f}" + (f" excess {m['B_excess_nll']:.3f}±{m['B_excess_nll_se']:.3f}" if "B_excess_nll" in m else "")
        msg += f" gap {m['B_gap']:+.3f}"
    if "fisher_sigma_logdet_ratio" in m:
        msg += f" | Fisher JJ {m['fisher_logdet_ratio']:+.2f} JSJ {m['fisher_sigma_logdet_ratio']:+.2f}"
    log(msg + f"  {m['seconds']:.0f}s")
    return m


def eta_diagnostics(model, th_t, t_test, oracle, T):
    """emulator_diagnostics for a Pipeline with an eta tower (eta_emu for gmi)."""
    with torch.no_grad():
        eta_emu = model.emulator_eta()(T(th_t)).cpu().numpy()
        eta_info = model.head.eta(T(th_t)).cpu().numpy() if model.spec["head"] == "gmi" else None
    t = t_test.cpu().numpy() if torch.is_tensor(t_test) else t_test
    return emulator_diagnostics(model.emulator_eta(), th_t, t, eta_emu, oracle, eta_info=eta_info)


def posthoc_cell(cfg: StudyConfig, data, seed, budget, arm, dest, frozen=True, log=print):
    """Diagnostics and evaluation B for a saved eta-arm model (e.g. runs made before they existed)."""
    torch.set_num_threads(cfg.threads)
    torch.set_flush_denormal(True)
    dest = Path(dest)
    theta_all, x_all, val_all = data.pool(seed)
    th, xs, va = theta_all[:budget], x_all[:budget], val_all[:budget]
    th_t, x_t = data.test()
    sigma_n = data.sim.sigma_n
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
    vrng = np.random.default_rng(5000 + seed)
    xv_noisy_all = xs[va] + sigma_n * vrng.standard_normal(xs[va].shape).astype(np.float32)
    model = Pipeline(arm, cfg, data.lcfg.n, sigma_n)
    model.load_state_dict(torch.load(dest / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        t_test = torch.cat([model.t(T(x_t[i:i + 128])) for i in range(0, len(x_t), 128)])
    oracle = data.oracle()
    m, _ = eta_diagnostics(model, th_t, t_test, oracle, T)
    if frozen:
        resB, infoB, gap, _ = frozen_summary_eval(model, cfg, th, xs, ~va, va, T(xv_noisy_all), sigma_n,
                                                  th_t, x_t, seed, T, log=log)
        resB = {k: v.detach().cpu().numpy() for k, v in resB.items()}
        m.update({f"B_{k}": v for k, v in metrics_from(resB, th_t, oracle).items()})
        m.update(B_gap=gap, B_epochs=infoB["epochs"], B_seconds=infoB["seconds"])
        np.savez(dest / "test_B.npz", **{k: np.asarray(v) for k, v in resB.items()})
    (dest / "posthoc.json").write_text(json.dumps(m, indent=1))
    return m


def run_study(cfg: StudyConfig, out, cache, cells=None, log=print, catalogue=None, catalogue_noise=0.125):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cpath = out / "config.json"
    if cpath.exists():
        old = json.loads(cpath.read_text())
        if isinstance(old.get("lensing"), dict):
            old["lensing"].setdefault("params", "S8")     # configs written before the sigma8 option
        diff = {k: (old.get(k), v) for k, v in cfg.to_dict().items()
                if k not in ("budgets", "seeds", "arms", "threads", "device") and json.dumps(old.get(k)) != json.dumps(v)}
        if diff:
            raise ValueError(f"config differs from {cpath}: {diff}; use a new --out")
    cpath.write_text(json.dumps(cfg.to_dict(), indent=1))
    data = CatalogueData(cfg, catalogue, catalogue_noise) if catalogue else Data(cfg, cache)
    if catalogue:
        info = dict(catalogue=str(catalogue), noise_amp=catalogue_noise, n_test=len(data._test),
                    n_pool=len(data._pool), n_duplicate_theta=data.n_duplicate_theta, map_side=data.lcfg.n)
        (out / "data.json").write_text(json.dumps(info, indent=1))
        log(f"catalogue: {info}")
    data.test()
    oracle = data.oracle()
    if oracle is not None:
        (out / "oracle.json").write_text(json.dumps(metrics_from(oracle, data.test()[0]), indent=1))
    cells = cells or [(s, n, a) for s in cfg.seeds for n in cfg.budgets for a in cfg.arms]
    rows = []
    for s, n, a in cells:
        try:
            rows.append(run_cell(cfg, data, s, n, a, out, log=log))
        except Exception as exc:          # record and continue
            log(f"FAILED seed {s} N {n} {a}: {type(exc).__name__}: {exc}")
    return rows
