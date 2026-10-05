#!/usr/bin/env python3
"""Regenerate notebooks/surrogate_colab.ipynb from the package source (run after any edit).

    python scripts/build_surrogate_colab.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ["src/unified_sbi/surrogate/__init__.py", "src/unified_sbi/surrogate/hw.py",
       "src/unified_sbi/surrogate/nets.py", "src/unified_sbi/surrogate/fisher.py",
       "src/unified_sbi/surrogate/study.py", "scripts/plot_surrogate.py", "tests/test_surrogate.py"]

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [md(r"""# Few-run conditional surrogates: Fisher-isometric conditioning on Hasegawa–Wakatani (Colab)

This notebook runs the Tier-1 testbed of `docs/plasma_surrogate_conditioning.md` (§8.1). It is self-contained: the cells below write the package, so no repository checkout is needed.

**Testbed.** A batched pseudo-spectral solver for the modified Hasegawa–Wakatani model on a doubly periodic $64^2$ grid, with $d=5$ control parameters
$\theta=(\log_{10}\alpha,\ \kappa,\ \log_{10}\hat\nu,\ \log_{10}\hat D,\ k_0)$: adiabaticity, gradient drive, grid-relative hyperviscosity and hyperdiffusion, and box wavenumber.
- One surrogate step is $\Delta t=1$ (50 solver steps), taken after a spin-up of $t=250$, with 120 snapshots per run.
- **Branch restarts** from stored states at $\theta\pm\varepsilon e_i$ give the *true* one-step sensitivity $\partial s_{t+1}/\partial\theta$, which is used for Jacobian supervision and for the sensitivity metrics.
- Viscosity and diffusion are sloppy directions (one-step sensitivity 10–100× smaller); α, κ and $k_0$ are stiff.

**Surrogate.** A patch transformer (circular-conv stem + patchify → adaLN-Zero blocks → unpatchify + conv refinement). The conditioning $c=g(\theta)$ enters the tokens, every block and the decoder. States are divided by their own per-field RMS, the target is $s_{t+1}/\mathrm{rms}(s_t)$, and $\log\mathrm{rms}$ is passed as extra conditioning. The output metric is therefore a fixed relative-error metric ($W=I$ in that frame).

| arm | conditioning |
|---|---|
| `A0_film` | FiLM/adaLN on raw standardised θ (baseline) |
| `A1_g` | learned $g$, weight decay on the conditioning path |
| `A2_iso` | A1 + $\mathcal R_{\rm iso}$ against the surrogate's own Fisher $A_{\rm self}$ |
| `A3_iso_wn` | A2 + Fisher-whitened θ-noise |
| `A4_iso_jac` | A2 + Jacobian supervision from branch restarts, with the physical Fisher $A_{\rm phys}$ as target |
| `A5_iso_in` | A2 + isotropic θ-noise of matched magnitude (control for A3) |
| `A6_full` | A3 + Jacobian supervision + $A_{\rm phys}$ target |

The **decisive** first comparison is A0 / A1 / A2 / A4. A3 vs A5 isolates the whitening, and A6 is the full method.

All splits are by run: the validation runs (checkpoint selection) and the 48 test runs are disjoint from the training pool and fixed across seeds and budgets. Budgets are nested prefixes of one permutation of the pool per seed.

**Use a GPU runtime** (Runtime → Change runtime type). An A100 or L4 is best; a T4 works but is slower."""),
         code(r"""import os, sys, torch
from pathlib import Path
WORK = Path('/content/unified_sbi_surrogate')
for sub in ('src/unified_sbi/surrogate', 'scripts', 'tests'):
    (WORK / sub).mkdir(parents=True, exist_ok=True)
os.chdir(WORK); sys.path.insert(0, str(WORK / 'src'))
print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"""),
         md("## 1. Package source\nThe next cells write the package exactly as it is in the repository (`src/unified_sbi/surrogate/`). Expand them to read the code."),
         code('%%writefile src/unified_sbi/__init__.py\n"""Standalone copy for Colab: only the surrogate subpackage."""\n')]
for f in SRC:
    cells.append(code(f"%%writefile {f}\n" + (ROOT / f).read_text()))
cells += [
    code(r"""# Checks (~1 min): energy conservation of the bracket, restart = trajectory, branch FD consistency,
# R_iso minimum/invariance, whitened-noise covariance, FD Jacobians vs autograd, exact gauge rescaling,
# chord and geodesic maximin. The end-to-end smoke test is marked slow and skipped here.
!cd {WORK} && PYTHONPATH=src python -m pytest -q -m "not slow" -p no:cacheprovider tests/test_surrogate.py"""),
    md(r"""## 2. Storage and a look at the simulator
Results **and the simulation cache** go to Drive, so the simulations survive a disconnected runtime. Every finished cell is skipped on rerun.

The cell below integrates 4 runs at prior corners. It times the solver, shows that the energy saturates during spin-up, and plots the final $n$ and $\phi$ fields."""),
    code(r"""from google.colab import drive
drive.mount('/content/drive')
BASE = Path('/content/drive/MyDrive/Colab Notebooks/unified-sbi/surrogate_hw')
CACHE = BASE / 'cache'; CACHE.mkdir(parents=True, exist_ok=True)

import time, math, numpy as np, matplotlib.pyplot as plt
from unified_sbi.surrogate.hw import HWConfig, simulate
h = HWConfig(n_save=2, batch=4)
lo, hi = np.array(h.lo), np.array(h.hi)
th = np.array([[hi[0], lo[1], hi[2], hi[3], hi[4]],      # adiabatic, weak drive, strong damping
               [lo[0], hi[1], lo[2], lo[3], lo[4]],      # hydrodynamic, strong drive
               (lo + hi) / 2,
               [lo[0], lo[1], hi[2], lo[3], hi[4]]])
t0 = time.time()
r = simulate(th, [0, 1, 2, 3], h, device='cuda' if torch.cuda.is_available() else 'cpu', record_spinup_every=250)
print(f'{time.time() - t0:.1f} s for 4 runs x {int((h.t_spin + h.n_save * h.save_dt) / h.dt)} steps; finite: {r["finite"].tolist()}')
fig, ax = plt.subplots(1, 9, figsize=(18, 2.3), constrained_layout=True)
tt = np.arange(1, r['spinup_energy'].shape[1] + 1) * 250 * h.dt
for i in range(4):
    ax[0].semilogy(tt, r['spinup_energy'][i], lw=2)
    ax[1 + 2 * i].imshow(r['states'][i, -1, 0], cmap='RdBu_r'); ax[1 + 2 * i].set_title(f'run {i}: n', fontsize=8)
    ax[2 + 2 * i].imshow(r['states'][i, -1, 1], cmap='RdBu_r'); ax[2 + 2 * i].set_title(f'run {i}: phi', fontsize=8)
ax[0].set_title('spin-up energy', fontsize=8); ax[0].set_xlabel('t', fontsize=8)
for a in ax[1:]: a.axis('off')
plt.show()"""),
    md(r"""## 3. Configure and run
- **`smoke`**: a few minutes. It runs the whole pipeline (simulation, training, evaluation, plots) on a small pool and prints step timings. Run it first.
- **`decisive`**: A0/A1/A2/A4 at $N_\theta\in\{25,50,100,200\}$ with 2 seeds.
- **`full`**: all seven arms with 3 seeds.

**Cost.**
- Simulating the pool (256 + 16 + 48 runs, ≈18.5k solver steps each), plus 4 checkpoints × 11 branch restarts per pool and test run, should take minutes on an A100. The data are cached on Drive.
- Training is 8000 steps per cell. The Fisher refreshes add roughly 10–20% to the R arms, and the Jacobian arms (A4, A6) cost about 2× per step.
- These estimates are untested on Colab: check the `smoke` timings first.

**Settings.** Changing any training or data setting needs a new results folder; the runner refuses to mix configurations. `arms`, `seeds`, `budgets` and `design_budgets` can be changed freely, so you can add arms to a finished folder later."""),
    code(r"""from unified_sbi.surrogate.study import ARMS, DECISIVE, StudyConfig, run_study, run_design
from unified_sbi.surrogate.hw import HWConfig

MODE = 'smoke'              # 'smoke' | 'decisive' | 'full'
hw = HWConfig(batch=128).to_dict()
if MODE == 'smoke':
    hw.update(n_save=60)
    cfg = StudyConfig(hw=hw, pool_size=32, n_val=8, n_test=16, budgets=(8, 24), seeds=(0,),
                      arms=DECISIVE, steps=600, eval_every=200, width=128, depth=4, heads=4,
                      design_budgets=(16,), design_candidates=512)
elif MODE == 'decisive':
    cfg = StudyConfig(hw=hw, arms=DECISIVE, seeds=(0, 1))
else:
    cfg = StudyConfig(hw=hw, arms=tuple(ARMS), seeds=(0, 1, 2))
OUT = BASE / MODE
print(OUT)
data = run_study(cfg, OUT, CACHE)       # order: seed -> budget -> arm; finished cells are skipped"""),
    md("## 4. Figures and tables\nThe paired table gives seed-level differences against A0 (positive = better than the FiLM baseline), with standard errors over seeds."),
    code(r"""from IPython.display import Image, display
import pandas as pd
!cd {WORK} && PYTHONPATH=src python scripts/plot_surrogate.py --results "{OUT}"
for f in ['scaling.png', 'sensitivity_by_param.png']:
    if (OUT / 'figures' / f).exists(): display(Image(filename=str(OUT / 'figures' / f)))
p = OUT / 'tables' / 'paired_vs_A0.csv'
if p.exists():
    t = pd.read_csv(p)
    cols = ['arm', 'budget'] + [c for c in t.columns if c.endswith('_gain') or c.endswith('_se')]
    display(t[cols].round(4))"""),
    md(r"""## 5. Design experiment (P5, optional)
For each seed and each $N$ in `design_budgets`, this trains the design arm (`A3_iso_wn` by default) on $N/2$ random runs. It then adds $N/2$ runs chosen in one of five ways and retrains on all $N$:
- `random`, the next pool runs;
- `lhs`, a Latin hypercube;
- `maximin_theta`, maximin in standardised θ;
- `maximin_g`, maximin on the chord distance in the learned $g$-space;
- `maximin_geo`, maximin on the graph-geodesic distance along the learned manifold.

New runs are simulated and cached."""),
    code(r"""RUN_DESIGN = False
if RUN_DESIGN:
    import dataclasses
    dcfg = dataclasses.replace(cfg, design_budgets=(50, 100) if MODE != 'smoke' else (16,))
    run_design(dcfg, OUT, CACHE, data=data)
    !cd {WORK} && PYTHONPATH=src python scripts/plot_surrogate.py --results "{OUT}"
    display(Image(filename=str(OUT / 'figures' / 'design.png')))
    display(pd.read_csv(OUT / 'tables' / 'design.csv').round(4))"""),
    md("## 6. Export\nThis zips the metrics, tables, figures and per-run arrays. Model weights and the simulation cache are left out."),
    code(r"""import shutil, tempfile
from google.colab import files
with tempfile.TemporaryDirectory() as tmp:
    for p in OUT.rglob('*'):
        if p.suffix in {'.json', '.csv', '.png', '.npz'}:
            q = Path(tmp) / p.relative_to(OUT); q.parent.mkdir(parents=True, exist_ok=True); shutil.copy(p, q)
    archive = shutil.make_archive(f'/content/surrogate_{MODE}', 'zip', tmp)
files.download(archive)"""),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.update(accelerator="GPU", colab=dict(provenance=[], gpuType="A100"),
                   kernelspec=dict(name="python3", display_name="Python 3"))
out = ROOT / "notebooks" / "surrogate_colab.ipynb"
nbf.write(nb, out)
print("wrote", out)
