#!/usr/bin/env python3
"""Evaluation B with 2-fold cross-fitting (leakage calibration for the frozen-summary protocol).

    python scripts/shared_crossfit.py --out results/toy_emu --cache cache/shared --budget 1000 --arms hybrid_quad_b0 hybrid_gmi

Two embeddings are trained on complementary halves of the N simulations (each with its own
train/validation labels from the study split). Embedding 2 is aligned to embedding 1 by an affine map
fitted by least squares over all N maps (one fixed noise draw). A MAF is then trained on out-of-fold,
aligned summaries: maps of half 1 are summarised by embedding 2 and vice versa, with noise redrawn
every epoch. Test maps alternate between the two embeddings. Writes <out>/seed*/N*/<arm>/crossfit.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from unified_sbi.shared.lensing import HI, LO
from unified_sbi.shared.nets import MAFHead, Standardise
from unified_sbi.shared.study import Data, Pipeline, StudyConfig, arm_spec, evaluate_maf, metrics_from, train

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--cache", default="cache/shared")
ap.add_argument("--budget", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--arms", nargs="+", required=True)
a = ap.parse_args()
out = Path(a.out)
raw = json.loads((out / "config.json").read_text())
cfg = StudyConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in StudyConfig.__dataclass_fields__})
cfg.device = "cpu"
torch.set_num_threads(cfg.threads)
torch.set_flush_denormal(True)
data = Data(cfg, a.cache)
seed, N = a.seed, a.budget
theta_all, x_all, val_all = data.pool(seed)
th, xs, va = theta_all[:N], x_all[:N], val_all[:N]
th_t, x_t = data.test()
sig = data.sim.sigma_n
oracle = data.oracle()
T = lambda z: torch.as_tensor(np.asarray(z), dtype=torch.float32)
fold = (np.arange(N) % 2).astype(int)          # nested-budget pools are random, so parity is a random split
rng = np.random.default_rng(7000 + seed)
noisy = lambda z: z + sig * rng.standard_normal(z.shape).astype(np.float32)

for arm in a.arms:
    spec = arm_spec(arm)
    t0 = time.time()
    models = []
    for f in (0, 1):
        sel = fold == f
        tr, vv = sel & ~va, sel & va
        torch.manual_seed(10 * seed + 1 + 100 * f)
        m = Pipeline(arm, cfg, data.lcfg.n, sig)
        m.fit_preprocessing(T(noisy(xs[tr])))
        train(m, list(m.parameters()), T(th[tr]), T(xs[tr]), T(th[vv]), T(noisy(xs[vv])), cfg, seed + 100 * f,
              f"{arm}/fold{f}", batch_size=spec.get("batch"), beta=spec.get("beta", 0.0), val_fn=m.val_loss)
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()
        models.append(m)
    # affine alignment of embedding 2 to embedding 1 over all N maps
    with torch.no_grad():
        xa = T(noisy(xs))
        t1 = torch.cat([models[0].t(xa[i:i + 256]) for i in range(0, N, 256)]).double()
        t2 = torch.cat([models[1].t(xa[i:i + 256]) for i in range(0, N, 256)]).double()
        X = torch.cat([t2, torch.ones(N, 1, dtype=torch.float64)], 1)
        W = torch.linalg.lstsq(X, t1).solution.float()
        align_r2 = float(1 - ((X @ W.double() - t1) ** 2).sum() / ((t1 - t1.mean(0)) ** 2).sum())

    def summ(x, which):                         # which: (B,) long, embedding used per map
        t = torch.empty(len(x), cfg.K)
        for f in (0, 1):
            s = which == f
            if s.any():
                tf = models[f].t(x[s])
                t[s] = tf if f == 0 else torch.cat([tf, torch.ones(len(tf), 1)], 1) @ W
        return t
    oof = torch.as_tensor(1 - fold)             # out of fold: half 0 -> embedding 1 and vice versa
    trm = ~va
    with torch.no_grad():
        t_tr = summ(T(noisy(xs[trm])), oof[trm])
    std = Standardise(cfg.K).fit(t_tr)
    torch.manual_seed(10 * seed + 3)
    holder = torch.nn.Module()
    holder.maf = MAFHead(cfg.K, LO, HI, cfg.maf_transforms, cfg.maf_hidden)
    holder.sigma_n = torch.as_tensor(sig)
    maf = holder.maf

    def loss(theta_f, x):                       # theta column 2 carries the embedding id
        with torch.no_grad():
            t = std(summ(x, theta_f[:, 2].long()))
        return -maf.log_prob(theta_f[:, :2], t)
    thf = torch.cat([T(th), oof[:, None].float()], 1)
    info = train(holder, list(maf.parameters()), thf[trm], T(xs[trm]), thf[va], T(noisy(xs[va])), cfg, seed + 1,
                 f"{arm}/crossfit-MAF", loss_fn=loss)
    with torch.no_grad():
        which_t = torch.as_tensor(np.arange(len(x_t)) % 2)
        t_test = std(summ(T(x_t), which_t))
    res = evaluate_maf(maf, t_test, T(th_t), cfg.posterior_samples, 123)
    res = {k: v.numpy() for k, v in res.items()}
    mt = metrics_from(res, th_t, oracle)
    mt.update(arm=arm, budget=N, seed=seed, align_r2=align_r2, maf_epochs=info["epochs"], seconds=time.time() - t0)
    dest = out / f"seed{seed}" / f"N{N}" / arm
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "crossfit.json").write_text(json.dumps(mt, indent=1))
    np.savez(dest / "test_crossfit.npz", **res)
    print(f"[crossfit seed {seed} N {N} {arm}] nll {mt['nll']:.3f}"
          + (f" excess {mt['excess_nll']:.3f}±{mt['excess_nll_se']:.3f}" if "excess_nll" in mt else "")
          + f" align R2 {align_r2:.3f}  {mt['seconds']:.0f}s", flush=True)
