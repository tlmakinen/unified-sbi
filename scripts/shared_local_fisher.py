#!/usr/bin/env python3
"""Local (heteroscedastic) emulator check on the Gaussian toy, from fresh simulations.

    python scripts/shared_local_fisher.py --results results/toy_emu [results/toy_ref ...] --cache cache/shared

The test-set diagnostics use one global Sigma-hat from residuals t - eta_emu(theta*) (bias included).
Here, at each centre of a 5 x 5 grid over the prior, ``--reps`` fresh noisy maps with common random
numbers at theta and theta +- h e_a give, per arm and N:

* m(theta) = E[t | theta], its Jacobian J_t by central differences (common random numbers), and the
  local covariance Sigma(theta) (Hartlap-corrected inverse);
* lin_fisher_ratio: log det(J_t^T Sigma^-1 J_t) - log det F_exact, the information of the best linear
  function of t. It is <= 0 by the Cramer-Rao bound (up to Monte-Carlo noise): the check of the method;
* emu_local_ratio: the same with the emulator Jacobian J_emu in place of J_t. > lin_fisher_ratio means
  the emulator is steeper than E[t|theta] (miscalibrated); the gap is the emulator's Fisher error;
* emu_jac_err: median relative Frobenius error ||J_emu - J_t|| / ||J_t|| (in Sigma^-1/2 units);
* bias chi^2/dof: sum_grid reps (m - eta_emu)^T Sigma^-1 (m - eta_emu), dof = 25 K;
* heteroscedasticity: median |log det Sigma(theta) - log det Sigma_pooled|.
Writes local_fisher.json per cell and a markdown table.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from unified_sbi.shared.lensing import HI, LO
from unified_sbi.shared.study import Data, Pipeline, StudyConfig, _jacobian

ap = argparse.ArgumentParser()
ap.add_argument("--results", nargs="+", required=True)
ap.add_argument("--cache", default="cache/shared")
ap.add_argument("--reps", type=int, default=128)
ap.add_argument("--grid", type=int, default=5)
ap.add_argument("--h", type=float, default=0.01, help="finite-difference step, fraction of the prior width")
ap.add_argument("--arms", nargs="+")
ap.add_argument("--out")
a = ap.parse_args()
torch.set_num_threads(1)

models = []
for res in a.results:
    res = Path(res)
    raw = json.loads((res / "config.json").read_text())
    cfg = StudyConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in StudyConfig.__dataclass_fields__})
    data = Data(cfg, a.cache)
    for mp in sorted(res.glob("seed*/N*/*/model.pt")):
        arm = mp.parent.name
        if a.arms and arm not in a.arms:
            continue
        m = Pipeline(arm, cfg, data.lcfg.n, data.sim.sigma_n)
        m.load_state_dict(torch.load(mp, map_location="cpu", weights_only=True))
        models.append((arm, int(mp.parent.parent.name[1:]), mp.parent, m.eval()))
sim = data.sim
g = (np.arange(a.grid) + 0.5) / a.grid
grid = LO + (HI - LO) * np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
step = a.h * (HI - LO)
F_exact = sim.exact_fisher(grid)
ld_exact = np.linalg.slogdet(F_exact)[1]
R = a.reps
stats = {i: dict(mean=[], S=[], Jt=[], eta=[], Je=[]) for i in range(len(models))}
for gi, th in enumerate(grid):
    sets = [th] + [th + s * step * np.eye(2)[k] for k in range(2) for s in (+1, -1)]
    xs = []
    for p in sets:                                   # common random numbers across the five sets
        rng = np.random.default_rng(10_000 + gi)
        xs.append(sim.add_noise(sim.simulate(np.repeat(p[None], R, 0), rng), rng).astype(np.float32))
    for i, (arm, N, d, m) in enumerate(models):
        with torch.no_grad():
            ts = [torch.cat([m.t(torch.as_tensor(x[j:j + 128])) for j in range(0, R, 128)]).double().numpy() for x in xs]
            em = m.emulator_eta()
            tt = torch.as_tensor(th[None], dtype=torch.float32)
            eta = em(tt).double().numpy()[0]
            Je = _jacobian(em, tt).double().numpy()[0]
        t0 = ts[0]
        mu = t0.mean(0)
        S = (t0 - mu).T @ (t0 - mu) / (R - 1)
        Jt = np.stack([(ts[1 + 2 * k] - ts[2 + 2 * k]).mean(0) / (2 * step[k]) for k in range(2)], -1)   # (K, 2)
        for key, v in (("mean", mu), ("S", S), ("Jt", Jt), ("eta", eta), ("Je", Je)):
            stats[i][key].append(v)
    print(f"grid point {gi + 1}/{len(grid)}", flush=True)

rows = []
for i, (arm, N, d, m) in enumerate(models):
    st = {k: np.array(v) for k, v in stats[i].items()}
    K = st["mean"].shape[1]
    hart = (R - K - 2) / (R - 1)
    Si = hart * np.linalg.inv(st["S"])
    lin = np.linalg.slogdet(np.einsum("gka,gkl,glb->gab", st["Jt"], Si, st["Jt"]))[1] - ld_exact
    emu = np.linalg.slogdet(np.einsum("gka,gkl,glb->gab", st["Je"], Si, st["Je"]))[1] - ld_exact
    W = np.linalg.cholesky(Si)                       # Sigma^-1 = W W^T
    jerr = np.linalg.norm(np.einsum("gkl,gka->gla", W, st["Je"] - st["Jt"]), axis=(1, 2)) / \
        np.linalg.norm(np.einsum("gkl,gka->gla", W, st["Jt"]), axis=(1, 2))
    b = st["mean"] - st["eta"]
    chi2 = float(R * np.einsum("gi,gij,gj->", b, Si, b))
    S_pool = st["S"].mean(0)
    spread = float(np.median(np.abs(np.linalg.slogdet(st["S"])[1] - np.linalg.slogdet(S_pool)[1])))
    r = dict(arm=arm, budget=N, lin_fisher_ratio=float(np.median(lin)),
             lin_fisher_iqr=[float(np.percentile(lin, 25)), float(np.percentile(lin, 75))],
             emu_local_ratio=float(np.median(emu)), emu_local_iqr=[float(np.percentile(emu, 25)), float(np.percentile(emu, 75))],
             emu_jac_err=float(np.median(jerr)), local_bias_chi2=chi2, local_bias_dof=int(len(grid) * K),
             sigma_logdet_spread=spread)
    (d / "local_fisher.json").write_text(json.dumps(r, indent=1))
    rows.append(r)
df = pd.DataFrame(rows)
df["chi2_s"] = [f"{c / dof:.0f}" for c, dof in zip(df.local_bias_chi2, df.local_bias_dof)]
out = []
for col, f, title in (("lin_fisher_ratio", "{:+.2f}", "Best-linear-summary Fisher log-det ratio, J_tᵀΣ(θ)⁻¹J_t vs exact (median over the 5×5 grid; must be ≤ 0)"),
                      ("emu_local_ratio", "{:+.2f}", "Emulator Fisher with local Σ, J_emuᵀΣ(θ)⁻¹J_emu vs exact (median)"),
                      ("emu_jac_err", "{:.2f}", "Emulator Jacobian error ‖Σ^{-1/2}(J_emu − J_t)‖/‖Σ^{-1/2}J_t‖ (median)"),
                      ("chi2_s", "{}", "Local emulator bias χ² per dof (128 reps per grid point; 1 if unbiased)"),
                      ("sigma_logdet_spread", "{:.2f}", "Heteroscedasticity: median |log det Σ(θ) − log det Σ_pooled|")):
    p = df.pivot_table(index="arm", columns="budget", values=col, aggfunc="first")
    out.append(f"\n### {title}\n\n" + p.map(lambda v: "" if (not isinstance(v, str) and pd.isna(v)) else f.format(v)).to_markdown())
text = "\n".join(out)
print(text)
if a.out:
    Path(a.out).write_text(text + "\n")
