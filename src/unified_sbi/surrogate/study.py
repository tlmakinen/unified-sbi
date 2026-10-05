"""Few-run conditional surrogate study on the Hasegawa-Wakatani testbed.

Implements the ablation of docs/plasma_surrogate_conditioning.md, section 8:

    arm             conditioning                                         section
    A0_film         FiLM/adaLN on raw standardised theta (baseline)
    A1_g            learned g(theta), weight decay on the conditioning path
    A2_iso          A1 + R_iso against the surrogate's own Fisher A_self  3.2-3.3
    A3_iso_wn       A2 + Fisher-whitened theta-noise                     3.5
    A4_iso_jac      A2 + Jacobian supervision from branch restarts, with
                    the physical Fisher A_phys as the R_iso target        4.3
    A5_iso_in       A2 + isotropic theta-noise of matched magnitude       control for 3.5
    A6_full         A3 + Jacobian supervision + A_phys target             full method

The first decisive comparison is A0 / A1 / A2 / A4 (section 8); A3 vs A5 isolates the
whitening, and A6 is the full method.

One *cell* = (seed, budget N_theta, arm). Within a seed, budgets are nested prefixes of one
permutation of the simulated pool; every arm of a cell sees the same runs, the same
initialisation seed and the same number of optimiser steps. Validation runs (for picking the
best checkpoint) and test runs are disjoint from the pool and fixed across seeds and budgets:
all splits are *by run*.

Metrics (all on held-out test runs, in the relative frame y = s_{t+1} / rms(s_t)):
    one-step NRMSE per field, and persistence for reference;
    rollout NRMSE at several horizons and the mean radial flux Gamma_n of rollouts against
    the run's true mean flux (|log ratio| and Spearman rank correlation across test runs);
    sensitivity error against branch-restart Jacobians: relative Jacobian error (all and per
    parameter), cosine per parameter, Fisher log-det ratio and affine-invariant distance.

The design experiment (section 7, prediction P5) trains the design arm on N/2 random runs,
then adds N/2 runs chosen at random, by Latin hypercube, by maximin in standardised theta, by
maximin on the chord distance in the learned g-space, or by maximin on graph-geodesic distance
along the learned conditioning manifold; it simulates them and retrains on all N.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import fisher as F
from .hw import (HWConfig, branch_jacobian, flux, latin_hypercube, sample_prior, simulate,
                 standardise)
from .nets import Model, n_params, normalise

ARMS = {
    "A0_film":    dict(identity=True,  R=False, noise=None,       jac=False),
    "A1_g":       dict(identity=False, R=False, noise=None,       jac=False),
    "A2_iso":     dict(identity=False, R=True,  noise=None,       jac=False),
    "A3_iso_wn":  dict(identity=False, R=True,  noise="whitened", jac=False),
    "A4_iso_jac": dict(identity=False, R=True,  noise=None,       jac=True),
    "A5_iso_in":  dict(identity=False, R=True,  noise="iso",      jac=False),
    "A6_full":    dict(identity=False, R=True,  noise="whitened", jac=True),
}
DECISIVE = ("A0_film", "A1_g", "A2_iso", "A4_iso_jac")
DESIGNS = ("random", "lhs", "maximin_theta", "maximin_g", "maximin_geo")


@dataclass
class StudyConfig:
    hw: dict = field(default_factory=lambda: HWConfig().to_dict())
    # data (all splits by run)
    pool_size: int = 256
    n_val: int = 16
    n_test: int = 48
    data_seed: int = 0
    n_checkpoints: int = 4          # branch-restart states per run (pool and test)
    branch_eps: float = 0.02        # standardised units, simulator and surrogate alike
    budgets: tuple = (25, 50, 100, 200)
    seeds: tuple = (0, 1)
    arms: tuple = tuple(ARMS)
    # model
    K: int = 8
    g_hidden: int = 128
    width: int = 192
    depth: int = 6
    heads: int = 6
    patch: int = 4
    # optimisation
    steps: int = 8000
    batch: int = 32
    lr: float = 3e-4
    lr_warmup: int = 300
    wd: float = 1e-4
    wd_cond: float = 1e-2           # weight decay on g and every conditioning projection
    grad_clip: float = 1.0
    eval_every: int = 500
    n_val_pairs: int = 512
    # R_iso (section 3.3)
    r_start: float = 0.3            # fraction of steps with beta = 0 (warm-up, 4.2)
    r_ramp: float = 0.1             # fraction of steps to ramp beta up
    beta: float = 0.1
    lam: tuple = (0.3, 0.03)        # Fisher floor: held at lam[0], then linear to lam[1]
    lam_hold: float = 0.5           # fraction of the R phase before lam decays (Fisher estimate stable)
    eps_g: float = 1e-3             # G_eps = J_g^T J_g + eps_g I (section 3.3)
    eps_s: float = 1e-12            # s_eps = max(s_bar, eps_s) (section 3.2)
    fisher_every: int = 25
    fisher_runs: int = 16           # runs refreshed per update
    fisher_states: int = 2          # states per run per update
    fisher_decay: float = 0.7
    # Fisher-whitened smoothing (3.5): sigma_c = kappa * median NN spacing in g-space
    kappa_noise: tuple = (0.3, 0.1)
    # Jacobian supervision (4.3)
    gamma_jac: float = 0.1
    jac_batch: int = 4
    # evaluation
    rollout_starts: int = 3
    rollout_h: int = 40
    flux_from: int = 10             # rollout steps >= this enter the flux average
    horizons: tuple = (1, 2, 5, 10, 20, 40)
    # design experiment (section 7)
    design_budgets: tuple = (50,)
    design_arm: str = "A3_iso_wn"
    designs: tuple = DESIGNS
    design_neighbours: int = 10     # k of the kNN graph for geodesic maximin
    design_candidates: int = 2048
    # hardware
    device: str = "auto"
    sim_device: str = "auto"
    threads: int = 0

    def hcfg(self):
        return HWConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in self.hw.items()})

    def to_dict(self):
        return asdict(self)

    def data_key(self):
        h = {k: v for k, v in self.hw.items() if k != "batch"}
        keys = dict(hw=h, pool=self.pool_size, val=self.n_val, test=self.n_test,
                    seed=self.data_seed, ck=self.n_checkpoints, eps=self.branch_eps)
        return hashlib.sha1(json.dumps(keys, sort_keys=True, default=list).encode()).hexdigest()[:10]


def pick_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


# ---------------------------------------------------------------------------------------------
# data


class HWData:
    """Simulated runs (pool + validation + test), cached to disk.

    Run index layout: [0, pool) pool, [pool, pool + val) validation, then test, then any
    runs added by the design experiment."""

    def __init__(self, cfg: StudyConfig, cache, log=print):
        self.cfg, self.h, self.log = cfg, cfg.hcfg(), log
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        path = self.cache / f"hw_{cfg.data_key()}.pt"
        if path.exists():
            d = torch.load(path, weights_only=False)
            log(f"loaded {path}")
        else:
            d = self._generate()
            torch.save(d, path)
            log(f"saved {path}")
        self.theta = d["theta"]                   # (R, d) natural, float64
        self.states = d["states"]                 # (R, T, 2, n, n) float32
        self.flux = d["flux"]                     # (R, T)
        self.finite = d["finite"]
        self.ck = d["ck"]                         # (R, n_ck) checkpoint time indices
        self.J = d["J"]                           # (R, n_ck, d, 2, n, n) relative frame (nan if absent)
        self.theta_std = standardise(self.theta, self.h).float()
        P, V, T = cfg.pool_size, cfg.n_val, cfg.n_test
        fin = self.finite.numpy()
        self.pool = [i for i in range(P) if fin[i]]
        self.val = [i for i in range(P, P + V) if fin[i]]
        self.test = [i for i in range(P + V, P + V + T) if fin[i]]
        bad = int((~self.finite).sum())
        if bad:
            log(f"warning: {bad} runs were not finite and are excluded")

    def _simulate(self, theta, seeds, with_branches):
        h, dev = self.h, pick_device(self.cfg.sim_device)
        t0 = time.time()
        r = simulate(theta, seeds, h, device=dev, log=self.log)
        n_ck = self.cfg.n_checkpoints
        T = h.n_save
        ck = torch.from_numpy(np.linspace(0, T - 2, n_ck).round().astype(np.int64))[None].repeat(len(theta), 1)
        J = torch.full((len(theta), n_ck, h.d, 2, h.n, h.n), float("nan"))
        if with_branches.any():
            idx = np.where(with_branches)[0]
            S = r["states"][torch.as_tensor(idx)][:, ck[0]].reshape(-1, 2, h.n, h.n)
            th = np.repeat(np.asarray(theta)[idx], n_ck, 0)
            Jb = torch.empty(len(S), h.d, 2, h.n, h.n)
            for i in range(0, len(S), 128):                               # bounded memory
                Jb[i:i + 128], _ = branch_jacobian(S[i:i + 128], th[i:i + 128], h,
                                                   eps=self.cfg.branch_eps, device=dev)
            rms = S.pow(2).mean((-2, -1)).sqrt()
            Jb = Jb / rms[:, None, :, None, None]
            J[torch.as_tensor(idx)] = Jb.reshape(len(idx), n_ck, *Jb.shape[1:])
            self.log(f"branch restarts for {len(idx)} runs x {n_ck} checkpoints done")
        self.log(f"simulated {len(theta)} runs in {time.time() - t0:.0f} s")
        return dict(theta=torch.as_tensor(np.asarray(theta)), states=r["states"], flux=r["flux"],
                    finite=r["finite"], ck=ck, J=J)

    def _generate(self):
        c, h = self.cfg, self.h
        rng = np.random.default_rng(c.data_seed)
        n = c.pool_size + c.n_val + c.n_test
        theta = sample_prior(n, h, rng)
        seeds = c.data_seed * 1_000_000 + np.arange(n)
        branches = np.ones(n, bool)
        branches[c.pool_size:c.pool_size + c.n_val] = False             # validation needs none
        self.log(f"simulating {n} runs ({h.n}^2, {h.n_save} snapshots, spin-up {h.t_spin})")
        return self._simulate(theta, seeds, branches)

    def add_runs(self, theta, tag, with_branches=False):
        """Simulate extra runs (design experiment), cache them, append, return their indices."""
        path = self.cache / f"hw_{self.cfg.data_key()}_extra_{tag}.pt"
        if path.exists():
            d = torch.load(path, weights_only=False)
        else:
            off = int(hashlib.sha1(tag.encode()).hexdigest()[:6], 16)
            seeds = 7_000_000_000 + off * 10_000 + np.arange(len(theta))
            d = self._simulate(theta, seeds, np.full(len(theta), with_branches))
            torch.save(d, path)
        start = len(self.theta)
        self.theta = torch.cat([self.theta, d["theta"]])
        self.states = torch.cat([self.states, d["states"]])
        self.flux = torch.cat([self.flux, d["flux"]])
        self.finite = torch.cat([self.finite, d["finite"]])
        self.ck = torch.cat([self.ck, d["ck"]])
        self.J = torch.cat([self.J, d["J"]])
        self.theta_std = standardise(self.theta, self.h).float()
        return [start + i for i in range(len(d["theta"])) if bool(d["finite"][i])]

    def pool_order(self, seed):
        rng = np.random.default_rng(1000 + seed)
        return [self.pool[i] for i in rng.permutation(len(self.pool))]

    def pairs(self, runs, t):
        """Normalised (x, logr, y) for runs[k] at time t[k] -> t[k] + 1."""
        s0 = self.states[runs, t]
        s1 = self.states[runs, t + 1]
        x, logr = normalise(s0)
        y = s1 / logr.exp()[..., None, None]
        return x, logr, y


# ---------------------------------------------------------------------------------------------
# training


def build_model(arm, cfg: StudyConfig, data: HWData):
    a = ARMS[arm]
    return Model(data.h.d, data.h.n, K=cfg.K, identity=a["identity"], g_hidden=cfg.g_hidden,
                 width=cfg.width, depth=cfg.depth, heads=cfg.heads, patch=cfg.patch)


def _lr_factor(step, cfg):
    if step < cfg.lr_warmup:
        return (step + 1) / cfg.lr_warmup
    p = (step - cfg.lr_warmup) / max(1, cfg.steps - cfg.lr_warmup)
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))


def _val_pairs(cfg, data, seed=0):
    rng = np.random.default_rng(5000 + seed)
    T = data.states.shape[1]
    runs = rng.choice(data.val, cfg.n_val_pairs)
    t = rng.integers(0, T - 1, cfg.n_val_pairs)
    return torch.as_tensor(runs), torch.as_tensor(t)


@torch.no_grad()
def one_step_mse(model, data, runs, t, dev, bs=256):
    tot, n = 0.0, 0
    for i in range(0, len(runs), bs):
        r, tt = runs[i:i + bs], t[i:i + bs]
        x, logr, y = (v.to(dev) for v in data.pairs(r, tt))
        pred = model(x, data.theta_std[r].to(dev), logr)
        tot += (pred - y).pow(2).mean((1, 2, 3)).sum().item()
        n += len(r)
    return tot / n


def _exact_float32():
    """Finite-difference Jacobians (eps = 0.02) need full fp32: TF32 convolutions/matmuls
    (cuDNN's default on Ampere) add ~1e-3 relative rounding noise, i.e. tens of per cent of J."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def train(arm, cfg: StudyConfig, data: HWData, train_runs, seed, log=print):
    a = ARMS[arm]
    dev = pick_device(cfg.device)
    _exact_float32()
    torch.manual_seed(10_000 * seed + len(train_runs))
    model = build_model(arm, cfg, data).to(dev)
    cond = {id(p) for p in model.conditioning_parameters()}
    main = [p for p in model.parameters() if id(p) not in cond]
    opt = torch.optim.AdamW([dict(params=main, weight_decay=cfg.wd),
                             dict(params=model.conditioning_parameters(), weight_decay=cfg.wd_cond)],
                            lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_factor(s, cfg))
    gen = torch.Generator().manual_seed(seed)
    ngen = torch.Generator(device=dev).manual_seed(seed + 1)
    runs_t = torch.as_tensor(train_runs)
    N, d = len(train_runs), data.h.d
    T = data.states.shape[1]
    th_train = data.theta_std[runs_t].to(dev)
    vr, vt = _val_pairs(cfg, data)

    bank = F.FisherBank(N, d, cfg.fisher_decay, dev)
    if a["jac"]:                                    # physical Fisher A_phys from branch restarts (4.3)
        bank.set(F.fisher_from_jacobian(data.J[runs_t].flatten(0, 1)).reshape(N, -1, d, d).mean(1))
        ck_states = data.states[runs_t[:, None], data.ck[runs_t]]          # (N, n_ck, 2, n, n)
        ck_J = data.J[runs_t]
    r_on = int(cfg.r_start * cfg.steps) if a["R"] else cfg.steps + 1
    ramp = max(1, int(cfg.r_ramp * cfg.steps))
    G_cache, sigma_c = None, 0.0

    def refresh_bank(local):
        """Surrogate Fisher (section 4.1) for the given local run indices."""
        loc = torch.as_tensor(local).repeat_interleave(cfg.fisher_states)
        t = torch.randint(0, T - 1, (len(loc),), generator=gen)
        x, logr, _ = data.pairs(runs_t[loc], t)
        with torch.no_grad():
            J = F.surrogate_jacobian(model, x.to(dev), th_train[loc.to(dev)], logr.to(dev),
                                     cfg.branch_eps)
        bank.update(loc.to(dev), F.fisher_from_jacobian(J))

    hist, best, t_start = [], (float("inf"), None, -1), time.time()
    for step in range(cfg.steps):
        model.train()
        phase = min(1.0, max(0.0, (step - r_on) / max(1, cfg.steps - r_on)))
        lp = min(1.0, max(0.0, (phase - cfg.lam_hold) / max(1e-9, 1 - cfg.lam_hold)))
        lam = cfg.lam[0] + (cfg.lam[1] - cfg.lam[0]) * lp
        kap = cfg.kappa_noise[0] + (cfg.kappa_noise[1] - cfg.kappa_noise[0]) * phase
        beta = cfg.beta * min(1.0, (step - r_on) / ramp) if step >= r_on else 0.0
        if a["R"] and step >= r_on and (step == r_on or step % cfg.fisher_every == 0):
            if not a["jac"]:
                if step == r_on:
                    for i in range(0, N, 64):
                        refresh_bank(list(range(i, min(N, i + 64))))
                else:
                    refresh_bank(torch.randperm(N, generator=gen)[:cfg.fisher_runs].tolist())
            if step == r_on:
                # gauge fix before R_iso switches on: rescale c -> a c (predictions unchanged)
                # with a^2 = mean_r tr(G_r^{-1} M_r) / d, the optimal scalar for R_iso
                with torch.no_grad():
                    G0 = F.metric_of_g(model.g, th_train)
                    M0 = bank.target(lam, cfg.eps_s)
                    L0 = torch.linalg.cholesky(G0 + cfg.eps_g * torch.eye(d, device=dev))
                    a2 = torch.cholesky_solve(M0, L0).diagonal(dim1=-2, dim2=-1).sum(-1).mean() / d
                model.rescale_gauge(float(a2.sqrt()))
                log(f"    [{arm}] R_iso on at step {step}: gauge rescale a = {float(a2.sqrt()):.3g}")
            with torch.no_grad():
                G_cache = F.metric_of_g(model.g, th_train, cfg.eps_g)
                sigma_c = kap * F.nn_spacing(model.g(th_train)).item() if N > 1 else 0.0

        loc = torch.randint(0, N, (cfg.batch,), generator=gen)
        t = torch.randint(0, T - 1, (cfg.batch,), generator=gen)
        x, logr, y = (v.to(dev) for v in data.pairs(runs_t[loc], t))
        th = th_train[loc.to(dev)]
        if a["noise"] and step >= r_on and G_cache is not None:
            Gb = G_cache[loc.to(dev)]
            noise = F.whitened_noise if a["noise"] == "whitened" else F.isotropic_noise_matched
            th = th + noise(Gb, sigma_c, ngen)
        pred = model(x, th, logr)
        l_pred = (pred - y).pow(2).mean()
        loss = l_pred
        l_r = l_j = torch.zeros((), device=dev)
        if beta > 0:
            G = F.metric_of_g(model.g, th_train, cfg.eps_g)
            l_r = F.r_iso(G, bank.target(lam, cfg.eps_s))
            loss = loss + beta * l_r
        if a["jac"]:
            k = torch.randint(0, N * cfg.n_checkpoints, (cfg.jac_batch,), generator=gen)
            rl, cl = k // cfg.n_checkpoints, k % cfg.n_checkpoints
            xs, lr_ = normalise(ck_states[rl, cl].to(dev))
            Jt = ck_J[rl, cl].to(dev)
            Jh = F.surrogate_jacobian(model, xs, th_train[rl.to(dev)], lr_, cfg.branch_eps)
            l_j = (Jh - Jt).pow(2).mean() / Jt.pow(2).mean().clamp_min(1e-12)
            loss = loss + cfg.gamma_jac * l_j
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); sched.step()
        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            model.eval()
            v = one_step_mse(model, data, vr, vt, dev)
            hist.append(dict(step=step + 1, train=l_pred.item(), val=v, r_iso=l_r.item(),
                             jac=l_j.item(), beta=beta, sigma_c=sigma_c))
            log(f"    [{arm} N={N} s={seed}] step {step + 1:5d}  train {l_pred.item():.4f}  "
                f"val {v:.4f}  R {l_r.item():.3f}  jac {l_j.item():.3f}  "
                f"({time.time() - t_start:.0f} s)")
            if v < best[0]:
                best = (v, copy.deepcopy({k: w.detach().cpu() for k, w in model.state_dict().items()}),
                        step + 1)
    model.load_state_dict(best[1])
    model.eval()
    info = dict(best_val=best[0], best_step=best[2], train_time=time.time() - t_start, history=hist,
                r_iso_final=hist[-1]["r_iso"] if hist else float("nan"))
    return model, info


# ---------------------------------------------------------------------------------------------
# evaluation


def _spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])); rb = np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def evaluate(model, cfg: StudyConfig, data: HWData, train_runs, runs=None, bs=128):
    dev = pick_device(cfg.device)
    _exact_float32()
    model.eval()
    runs = data.test if runs is None else runs
    T = data.states.shape[1]
    d = data.h.d
    k0_idx = 4
    out = {}

    # 1. one-step errors on every transition of every test run
    os_err, pers = [], []
    for r in runs:
        rr = torch.full((T - 1,), r)
        tt = torch.arange(T - 1)
        e = torch.zeros(2); p = torch.zeros(2)
        for i in range(0, T - 1, bs):
            x, logr, y = (v.to(dev) for v in data.pairs(rr[i:i + bs], tt[i:i + bs]))
            pred = model(x, data.theta_std[rr[i:i + bs]].to(dev), logr)
            e += (pred - y).pow(2).mean((-2, -1)).sum(0).cpu()
            p += (x - y).pow(2).mean((-2, -1)).sum(0).cpu()
        os_err.append((e / (T - 1)).sqrt().numpy()); pers.append((p / (T - 1)).sqrt().numpy())
    os_err, pers = np.array(os_err), np.array(pers)
    out["onestep_nrmse_n"], out["onestep_nrmse_phi"] = map(float, np.median(os_err, 0))
    out["onestep_nrmse"] = float(np.median(os_err.mean(1)))
    out["persistence_nrmse"] = float(np.median(pers.mean(1)))

    # 2. rollouts
    H = min(cfg.rollout_h, T - 1)
    starts = np.linspace(0, T - 1 - H, cfg.rollout_starts).round().astype(int)
    hz = [h for h in cfg.horizons if h <= H]
    roll = np.full((len(runs), len(starts), len(hz)), np.nan)
    fl_pred = np.full((len(runs), len(starts)), np.nan)
    blow = 0
    for i, r in enumerate(runs):
        th = data.theta_std[[r] * len(starts)].to(dev)
        k0 = data.theta[r, k0_idx].item()
        s = data.states[r, starts].to(dev)                          # (S, 2, n, n)
        fl = []
        for k in range(1, H + 1):
            x, logr = normalise(s)
            s = model(x, th, logr) * logr.exp()[..., None, None]
            if k >= cfg.flux_from:
                fl.append(flux(s, k0).cpu())
            if k in hz:
                truth = data.states[r, starts + k].to(dev)
                err = ((s - truth).pow(2).mean((-3, -2, -1)) / truth.pow(2).mean((-3, -2, -1))).sqrt()
                roll[i, :, hz.index(k)] = err.cpu().numpy()
        bad = ~torch.isfinite(s).flatten(1).all(1).cpu()
        blow += int(bad.sum())
        if fl:
            f = torch.stack(fl).mean(0).numpy()
            f[bad.numpy()] = np.nan
            fl_pred[i] = f
    true_flux = data.flux[runs].mean(1).numpy()
    with np.errstate(all="ignore"):
        pred_flux = np.nanmean(fl_pred, 1)
    for j, h in enumerate(hz):
        out[f"rollout_nrmse_h{h}"] = float(np.nanmedian(roll[:, :, j]))
    out["rollout_blowups"] = blow
    # flux spans ~3 decades over the prior: relative error (a sign error counts as >= 1) and
    # the rank correlation across test runs (does the rollout order the runs' transport?)
    out["flux_rel_err"] = float(np.nanmedian(np.abs(pred_flux / true_flux - 1)))
    out["flux_spearman"] = _spearman(pred_flux, true_flux)

    # 3. sensitivity against branch-restart Jacobians
    have = [r for r in runs if torch.isfinite(data.J[r]).all()]
    if have:
        rj = torch.as_tensor(have)
        S = data.states[rj[:, None], data.ck[rj]].flatten(0, 1)           # (R*ck, 2, n, n)
        Jt = data.J[rj].flatten(0, 1)                                       # (R*ck, d, 2, n, n)
        th = data.theta_std[rj].repeat_interleave(cfg.n_checkpoints, 0)
        Jh = []
        for i in range(0, len(S), 16):
            x, logr = normalise(S[i:i + 16].to(dev))
            Jh.append(F.surrogate_jacobian(model, x, th[i:i + 16].to(dev), logr, cfg.branch_eps).cpu())
        Jh = torch.cat(Jh)
        num = (Jh - Jt).flatten(1).norm(dim=1); den = Jt.flatten(1).norm(dim=1)
        out["jac_rel_err"] = float((num / den).median())
        pp = (Jh - Jt).flatten(2).norm(dim=2) / Jt.flatten(2).norm(dim=2)        # (S, d)
        cos = torch.nn.functional.cosine_similarity(Jh.flatten(2), Jt.flatten(2), dim=2)
        out["jac_rel_err_per_param"] = pp.median(0).values.tolist()
        out["jac_cos_per_param"] = cos.mean(0).tolist()
        At = F.fisher_from_jacobian(Jt).reshape(len(have), -1, d, d).mean(1).double()
        Ah = F.fisher_from_jacobian(Jh).reshape(len(have), -1, d, d).mean(1).double()
        # regularised log-det ratio: same scale s (from the true Fisher) and same floor in both
        s_t = At.diagonal(dim1=-2, dim2=-1).sum(-1).mean() / d
        eye = torch.eye(d, dtype=At.dtype)
        Mh, Mt = Ah / s_t + cfg.lam[1] * eye, At / s_t + cfg.lam[1] * eye
        ld = torch.linalg.slogdet(Mh)[1] - torch.linalg.slogdet(Mt)[1]
        out["fisher_logdet_ratio"] = float(ld.median())
        tn = lambda A: A / A.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]
        out["fisher_shape_dist"] = float(F.affine_invariant_distance(tn(At), tn(Ah)).median())
        out["fisher_true_condition"] = float(torch.linalg.cond(At).median())

    # 4. global folding of g (section 10): chord |g(a) - g(b)| over the local Fisher length
    #    sqrt(dtheta^T G(mid) dtheta) for random prior pairs; values << 1 at large separation
    #    mean distant theta map close together in c.
    if not model.g.identity:
        gen = torch.Generator().manual_seed(123)
        ta = torch.rand(2048, d, generator=gen) * 2 * math.sqrt(3) - math.sqrt(3)
        tb = torch.rand(2048, d, generator=gen) * 2 * math.sqrt(3) - math.sqrt(3)
        with torch.enable_grad():
            Gm = F.metric_of_g(model.g, ((ta + tb) / 2).to(dev)).detach().cpu()
        dt = tb - ta
        loc = torch.einsum("bi,bij,bj->b", dt, Gm, dt).clamp_min(1e-12).sqrt()
        chord = (model.g(ta.to(dev)) - model.g(tb.to(dev))).norm(dim=1).cpu()
        far = dt.norm(dim=1) > dt.norm(dim=1).median()
        out["g_fold_ratio_p01"] = float(torch.quantile(chord[far] / loc[far], 0.01))

    # 5. per-run arrays: error vs distance to the nearest training run (standardised theta)
    tr = data.theta_std[train_runs]
    dist = torch.cdist(data.theta_std[runs], tr).min(1).values.numpy()
    arrays = dict(onestep=os_err.mean(1), dist=dist, flux_pred=pred_flux, flux_true=true_flux,
                  theta=data.theta[runs].numpy())
    return out, arrays


# ---------------------------------------------------------------------------------------------
# cells, studies and the design experiment


def _save(path: Path, metrics, arrays, model=None):
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "arrays.npz", **arrays)
    if model is not None:
        torch.save(model.state_dict(), path / "model.pt")
    (path / "metrics.json").write_text(json.dumps(metrics, indent=1))


def run_cell(cfg: StudyConfig, data: HWData, seed, budget, arm, out: Path, log=print, save_model=True):
    path = out / f"seed{seed}" / f"N{budget}" / arm
    if (path / "metrics.json").exists():
        log(f"  skip {path} (done)")
        return json.loads((path / "metrics.json").read_text())
    runs = data.pool_order(seed)[:budget]
    if len(runs) < budget:
        log(f"  only {len(runs)} finite pool runs for budget {budget}")
    log(f"  cell seed={seed} N={budget} arm={arm}")
    model, info = train(arm, cfg, data, runs, seed, log)
    m, arrays = evaluate(model, cfg, data, runs)
    m.update(info, arm=arm, budget=budget, seed=seed, n_params=n_params(model),
             n_params_cond=sum(p.numel() for p in model.conditioning_parameters()))
    _save(path, m, arrays, model if save_model else None)
    log(f"  -> one-step {m['onestep_nrmse']:.4f}  rollout@10 {m.get('rollout_nrmse_h10', float('nan')):.3f}  "
        f"flux rel {m['flux_rel_err']:.3f}  jac {m.get('jac_rel_err', float('nan')):.3f}")
    return m


def _check_config(cfg: StudyConfig, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    cpath = out / "config.json"
    free = {"arms", "seeds", "budgets", "design_budgets", "device", "sim_device", "threads"}
    mine = {k: v for k, v in json.loads(json.dumps(cfg.to_dict(), default=list)).items() if k not in free}
    if cpath.exists():
        old = {k: v for k, v in json.loads(cpath.read_text()).items() if k not in free}
        if old != mine:
            diff = sorted(k for k in set(old) | set(mine) if old.get(k) != mine.get(k))
            raise RuntimeError(f"{out} holds results for a different configuration ({diff}); "
                               "use a new output folder")
    cpath.write_text(json.dumps(json.loads(json.dumps(cfg.to_dict(), default=list)), indent=1))


def run_study(cfg: StudyConfig, out, cache, cells=None, log=print):
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    out = Path(out)
    _check_config(cfg, out)
    data = HWData(cfg, cache, log)
    cells = cells or [(s, n, a) for s in cfg.seeds for n in cfg.budgets for a in cfg.arms]
    for s, n, a in cells:
        run_cell(cfg, data, s, n, a, out, log)
    return data


def run_design(cfg: StudyConfig, out, cache, data=None, log=print):
    """Section 7 / prediction P5: random vs maximin-theta vs maximin-g additions."""
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    out = Path(out)
    _check_config(cfg, out)
    data = data or HWData(cfg, cache, log)
    arm, dev = cfg.design_arm, pick_device(cfg.device)
    for seed in cfg.seeds:
        for N in cfg.design_budgets:
            base_dir = out / "design" / f"seed{seed}" / f"N{N}"
            half = N // 2
            order = data.pool_order(seed)
            base = order[:half]
            todo = [g for g in cfg.designs if not (base_dir / g / "metrics.json").exists()]
            if not todo:
                log(f"  skip {base_dir} (done)")
                continue
            log(f"  design seed={seed} N={N}: stage model on {half} runs")
            stage, _ = train(arm, cfg, data, base, seed, log)
            rng = np.random.default_rng(9000 + seed)
            cand = sample_prior(cfg.design_candidates, data.h, rng)
            cand_std = standardise(torch.as_tensor(cand), data.h).float()
            with torch.no_grad():
                cg = stage.g(cand_std.to(dev)).cpu()
                bg = stage.g(data.theta_std[base].to(dev)).cpu()
            k = N - half
            picks = dict(
                lhs=lambda: latin_hypercube(k, data.h, np.random.default_rng(9100 + seed)),
                maximin_theta=lambda: cand[F.maximin(cand_std, data.theta_std[base], k)],
                maximin_g=lambda: cand[F.maximin(cg, bg, k)],
                maximin_geo=lambda: cand[F.maximin_geodesic(cg, bg, k, cfg.design_neighbours)])
            for g in todo:
                if g == "random":
                    added = order[half:N]
                else:
                    added = data.add_runs(picks[g](), f"s{seed}_N{N}_{g}",
                                          with_branches=ARMS[arm]["jac"])
                runs = base + added
                log(f"  design {g}: training {arm} on {len(runs)} runs")
                model, info = train(arm, cfg, data, runs, seed, log)
                m, arrays = evaluate(model, cfg, data, runs)
                m.update(info, arm=arm, budget=N, seed=seed, design=g,
                         added_theta=data.theta[added].tolist())
                _save(base_dir / g, m, arrays)
                log(f"  -> {g}: one-step {m['onestep_nrmse']:.4f}  jac {m.get('jac_rel_err', float('nan')):.3f}")
    return data


def load_results(out):
    rows = []
    for p in Path(out).glob("seed*/N*/*/metrics.json"):
        m = json.loads(p.read_text())
        rows.append({k: v for k, v in m.items() if not isinstance(v, (list, dict))})
    return rows
