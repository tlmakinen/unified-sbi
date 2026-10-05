#!/usr/bin/env python3
"""Post-hoc emulator refits vs the in-training stop-gradient emulator (Gaussian toy).

    python scripts/shared_emulator_refit.py --results results/toy_emu --cache cache/shared \
        --cells hybrid_quad_b0.3:4000 hybrid_rect:4000 hybrid_quad_b0:4000 hybrid_quad_b0.3:1000

Refits a SiLU ParamEmbed to the frozen t on (a) the training split, (b) the validation split and
(c) 2000 fresh prior simulations (4 noise draws per map for a, b). Compares each with the model's own
eta_emu on the local check of shared_local_fisher.py: local Fisher ratio, Jacobian error and bias chi2/dof.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from unified_sbi.shared.lensing import HI, LO
from unified_sbi.shared.nets import ParamEmbed
from unified_sbi.shared.study import Data, Pipeline, StudyConfig, _jacobian
ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True); ap.add_argument("--cache", default="cache/shared")
ap.add_argument("--cells", nargs="+", required=True, help="arm:N")
ap.add_argument("--threads", type=int, default=2)
args = ap.parse_args()
torch.set_num_threads(args.threads)
res = Path(args.results)
raw = json.loads((res / 'config.json').read_text())
cfg = StudyConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in StudyConfig.__dataclass_fields__})
data = Data(cfg, args.cache); sim = data.sim
cells = [(c.split(':')[0], int(c.split(':')[1])) for c in args.cells]
theta_all, x_all, val_all = data.pool(0)
rng = np.random.default_rng(4242)
th_f = sim.prior_sample(2000, rng); x_f = sim.add_noise(sim.simulate(th_f, rng), rng).astype(np.float32)
g = (np.arange(5) + 0.5) / 5
grid = LO + (HI - LO) * np.stack(np.meshgrid(g, g, indexing='ij'), -1).reshape(-1, 2)
step = 0.01 * (HI - LO); R = 128
X = []
for gi, th in enumerate(grid):
    sets = [th] + [th + s * step * np.eye(2)[k] for k in range(2) for s in (+1, -1)]
    xs = []
    for p in sets:
        r = np.random.default_rng(10_000 + gi)
        xs.append(sim.add_noise(sim.simulate(np.repeat(p[None], R, 0), r), r).astype(np.float32))
    X.append(xs)
F_ld = np.linalg.slogdet(sim.exact_fisher(grid))[1]
T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)

def tt(m, x):
    with torch.no_grad():
        return torch.cat([m.t(T(x[i:i + 256])) for i in range(0, len(x), 256)])

def fit_emu(th, t, steps=3000, seed=0):
    torch.manual_seed(seed)
    e = ParamEmbed(t.shape[1], LO, HI, act=torch.nn.SiLU)
    opt = torch.optim.Adam(e.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    th, t = T(th), t.float()
    for _ in range(steps):
        L = (e(th) - t).square().sum(-1).mean()
        opt.zero_grad(); L.backward(); opt.step(); sched.step()
    return e.eval()

for arm, N in cells:
    m = Pipeline(arm, cfg, data.lcfg.n, sim.sigma_n)
    m.load_state_dict(torch.load(res / f'seed0/N{N}/{arm}/model.pt', map_location='cpu', weights_only=True)); m.eval()
    th, xs, va = theta_all[:N], x_all[:N], val_all[:N]
    tr = ~va
    r2 = np.random.default_rng(1)
    rep = 4
    t_tr = torch.cat([tt(m, xs[tr] + sig) for sig in [sim.sigma_n * r2.standard_normal(xs[tr].shape).astype(np.float32) for _ in range(rep)]])
    t_va = torch.cat([tt(m, xs[va] + sig) for sig in [sim.sigma_n * r2.standard_normal(xs[va].shape).astype(np.float32) for _ in range(rep)]])
    emus = {'in-training eta_emu': m.emulator_eta(),
            f'refit: train split ({tr.sum()} sims)': fit_emu(np.tile(th[tr], (rep, 1)), t_tr),
            f'refit: validation split ({va.sum()} sims)': fit_emu(np.tile(th[va], (rep, 1)), t_va),
            'refit: 2000 fresh sims': fit_emu(th_f, tt(m, x_f))}
    # local truth
    mu, S, Jt = [], [], []
    for gi, xs5 in enumerate(X):
        ts = [tt(m, x).double().numpy() for x in xs5]
        mu.append(ts[0].mean(0)); S.append(np.cov(ts[0].T))
        Jt.append(np.stack([(ts[1 + 2 * k] - ts[2 + 2 * k]).mean(0) / (2 * step[k]) for k in range(2)], -1))
    mu, S, Jt = map(np.array, (mu, S, Jt))
    K = mu.shape[1]; Si = (R - K - 2) / (R - 1) * np.linalg.inv(S); W = np.linalg.cholesky(Si)
    print(f'\n== {arm} N={N}: best-linear ratio {np.median(np.linalg.slogdet(np.einsum("gka,gkl,glb->gab", Jt, Si, Jt))[1] - F_ld):+.2f}')
    for name, e in emus.items():
        with torch.no_grad():
            gt = T(grid); eta = e(gt).double().numpy(); Je = _jacobian(e, gt).double().numpy()
        ratio = np.median(np.linalg.slogdet(np.einsum('gka,gkl,glb->gab', Je, Si, Je))[1] - F_ld)
        jerr = np.median(np.linalg.norm(np.einsum('gkl,gka->gla', W, Je - Jt), axis=(1, 2)) / np.linalg.norm(np.einsum('gkl,gka->gla', W, Jt), axis=(1, 2)))
        b = mu - eta; chi = R * np.einsum('gi,gij,gj->', b, Si, b) / (25 * K)
        print(f'  {name:36s} local Fisher ratio {ratio:+.2f}   Jacobian error {jerr:.2f}   bias chi2/dof {chi:.0f}', flush=True)
