# unified-sbi

One-step Gaussian representations of parameters and data for simulation-based inference.

A **discovery** step jointly learns an invertible parameter map $\eta_\phi(\theta)$ and a summary network $t=\eta_\psi(x)$ by minimising

$$
L=\mathbb E\Big[\tfrac12\|\eta_\phi(\theta)-\eta_\psi(x)\|^2-\log|\det D\eta_\phi(\theta)|\Big],
$$

the negative log of $q(\theta\mid x)=\mathcal N(\eta_\phi(\theta);\eta_\psi(x),I)\,|\det D\eta_\phi|$. Downstream NPE (a MAF) then models $\theta\mid t$ or $\eta\mid t$.
The question this repository answers: **does this save simulations** compared with NPE on the uncompressed data?

The code consolidates the torch implementation from `unified_sbi.ipynb`.

## Install

```bash
cd ~/repositories/unified-sbi
pip install -e ".[dev]"      # torch, sbi==0.27.0, numpy, scipy, pandas, matplotlib, pytest
pytest -q                    # about 20 s
```

`ltu-ili` is no longer needed. The MAF is sbi's `posterior_nn(model="maf")`, which is what `ili.utils.load_nde_sbi` builds, but it is trained by an explicit loop in `npe.py` so that every method gets exactly the same train/validation split.

## Layout

```
src/unified_sbi/
  simulators.py   RosenbrockChain + presets (default: 8D chain with a hidden 2D Rosenbrock)
  maps.py         Coupling, ParameterMap (eta_phi), SummaryMap (eta_psi), one_step_loss, fit_maps
  npe.py          sbi MAF + training loop with early stopping on our validation split
  methods.py      the NPE "arms": raw, xbar, theta_t, eta_t; box-truncated sampling and log q in theta
  metrics.py      fair ensemble CRPS, Gaussian CRPS, CRPS from a CDF, interval coverage
  exact.py        exact posteriors by Gauss-Legendre chain quadrature (normaliser, 1D/2D marginals, CRPS)
  study.py        StudyConfig and run_cell: one (seed, budget) cell of the scaling study
  analysis.py     aggregation, power-law fits, "baseline budget needed to match" savings factors
  plotting.py     scaling curves, excess power laws, coverage, exact corner plot
scripts/
  scaling_study.py   the sweep (all cells, or one per SLURM array task)
  plot_scaling.py    figures + summary tables
  run_single.py      one budget with full diagnostics (the notebook workflow)
slurm/scaling_array.sbatch
notebooks/scaling_colab.ipynb
tests/
```

## The default problem

`rosenbrock8_hidden2d`: $\theta\sim\mathcal U([-3,3]^8)$ and 8 replicates of

$$\mu(\theta)=\big(\theta_1,\dots,\theta_8,\ \theta_2-\theta_1^2,\ \theta_3-\theta_2,\ \dots,\ \theta_8-\theta_7\big),$$

with variances 4 on the direct coordinates and 0.25 on the links. That is a linear Gaussian chain with one Rosenbrock banana hidden in $(\theta_1,\theta_2)$.
The raw data have $8\times15=120$ entries. Other presets: `rosenbrock8_full` (every link curved) and `rosenbrock4_full` (the original notebook).
Any `RosenbrockChain` field can be overridden through `simulator_overrides` in a JSON config.

## Scaling study

```bash
python scripts/scaling_study.py --out results/scaling --quick   # ~2 min smoke test
python scripts/scaling_study.py --out results/scaling           # everything, sequentially
python scripts/plot_scaling.py  --results results/scaling
```

**SLURM:** `python scripts/scaling_study.py --list-tasks` prints the grid (default 3 seeds × 6 budgets = 18 tasks). Then run `sbatch slurm/scaling_array.sbatch` after editing its environment lines.
**Colab:** open `notebooks/scaling_colab.ipynb`, which writes results to Drive. Finished cells are skipped on rerun in both cases.

### Budget accounting

Every method sees the same $N_{\rm total}$ simulations and the same split:

| | NPE on raw $x$ (baseline) | ours ($\theta\mid t$, $\eta\mid t$) |
|---|---|---|
| simulations used | $N_{\rm total}$ | $N_{\rm total}$ |
| discovery (map learning) | none | first $n_{\rm disc}=\min(500,N_{\rm total})$ |
| NPE training set | all training members of the $N_{\rm total}$ | the same, **reusing** the discovery simulations |
| NPE validation set | all validation members | the same |

- Each seed draws one pool of `max(budgets)` simulations. A budget uses the pool's first $N_{\rm total}$ entries, so budgets are nested.
- Each simulation is marked validation with probability 0.15 (fixed per seed), so it keeps that role across budgets and across both stages. Discovery early-stops on its own validation members, which are never used to fit the maps or the NPE.
- At $N_{\rm total}=250 < 500$, all 250 simulations go to discovery.
- For budgets $\ge500$, discovery is identical within a seed, so the fitted maps are cached in `results/.../cache/`.

### What is recorded (`results/.../rows/seed{s}_n{N}.json`, one row per arm)

| column | meaning |
|---|---|
| `val_log_prob` | end-of-training (best-epoch) mean $\log q(\theta\mid x)$ on the NPE validation split, in $\theta$ coordinates. For $\eta\mid t$ it includes the Jacobian. Like sbi's validation loss, it is **not** renormalised to the prior box. |
| `test_log_prob` | mean $\log q(\theta^\ast\mid x)$ on a fixed independent test set (default 500 pairs, not counted in any budget), renormalised to the prior box using the rejection-sampling acceptance rate |
| `crps` | fair ensemble CRPS per coordinate against $\theta^\ast$, averaged over coordinates and test pairs (500 box-truncated posterior draws each). `crps_per_dim` keeps the per-coordinate values. |
| `coverage68/90` | central-interval coverage, averaged over coordinates |
| `oracle_*` | the same test metrics for the exact posterior (`exact.py`): the floor each curve should approach |
| `acceptance`, `n_obs_clipped` | mean in-box acceptance of the flow's draws; number of test observations with acceptance below 10%, whose 500 draws were completed by resampling the accepted ones. Treat CRPS as approximate when this is non-zero. |

`plot_scaling.py` writes:

- `figures/scaling.png`: the three metrics against $N_{\rm total}$, with the exact posterior as a dashed line.
- `figures/excess_power_law.png`: CRPS and NLL in excess of the exact posterior, on log–log axes with fits $A\,N^{-\alpha}$.
- `summary_by_budget.csv`, `power_laws.csv`.
- `savings.csv`: for each method and budget, the budget the raw baseline needs to match it (interpolated in $\log N$), and that ratio. `>=` means the baseline never catches up inside the sweep; `<=` means the method is worse than the baseline's smallest budget.

### Caveat on reusing discovery simulations

The summary network was fitted on the discovery simulations, so $t$ can be slightly over-informative on exactly those points, and the NPE can learn a conditional that is too sharp. The NPE validation split was never used to fit the maps and the test set is independent, so this would show up as worse validation and test metrics rather than being hidden. To measure it, rerun with `--no-reuse` into a separate `--out`: the map-based arms then train NPE only on the non-discovery simulations, at the same $N_{\rm total}$. They are skipped at budgets where none remain.

## Programmatic use

```python
from unified_sbi.study import StudyConfig, run_cell
cfg = StudyConfig(arms=("raw", "eta_t"), n_test=200, device="cuda")
rows, art = run_cell(cfg, seed=0, n_total=2000, return_artifacts=True)
```

`scripts/run_single.py --n-total 2000` does the same and adds the one-step training curve, coverage curves and an exact corner plot.

## Options worth knowing

- `--arms raw xbar theta_t eta_t`: `xbar` (NPE on the replicate means, which are sufficient here) isolates how much of the gain comes from pooling replicates alone.
- `--learn-scales`: per-observation latent scales $s(x)$, i.e. $q=\mathcal N(\eta_\phi(\theta);t,\operatorname{diag}s^2)\ldots$ (from the nested-model notebook).
- `--n-discovery`, `--val-fraction`, `--n-test`, `--n-post`, `--map-steps`, `--npe-max-epochs`; or pass a full JSON `StudyConfig` with `--config`.

## Rank-deficient benchmark with a (k, m) screen (`unified_sbi.lowrank`)

A separate study, ported from `unified_sbi_rosenbrock_2d_in_8d.ipynb`. The full-rank scaling study above does not use the screen.

**The problem.** $\theta\sim\mathcal N(0,I_8)$ and a hidden rotation $u=Q\theta$. Only $(u_1,u_2)$ enter the data, through a 2D Rosenbrock: $(u_1,\ u_2,\ u_2-u_1^2)$ plus noise. Five further data channels are pure noise, and a second rotation mixes all 8 channels. So the likelihood has rank $k^\ast=2$, while the exact sufficient statistic is $m^\ast=3$ dimensional. The exact posterior comes from converged 2D quadrature, with $\mathcal N(0,1)$ in the six inactive directions.

**The structured one-step family.** $(a,v)=W\theta$ with $W\in SO(8)$, and $\eta=(f_\phi(a),v)$ where $f_\phi$ is a flow on the $k$ active coordinates. The summary $t=\eta_\psi(x)\in\mathbb R^m$ holds a latent mean $\mu$ ($k$ entries) plus $m-k$ entries of a data-dependent Cholesky factor. That gives

$$q_{k,m}(\theta\mid x)=\mathcal N\big(f_\phi(a);\mu,LL^\top\big)\,\mathcal N(v;0,I)\,|\det Df_\phi|,$$

with $k\le m\le k+k(k+1)/2$. With $m=k$ this is the source notebook's loss. Every summary output is trained by this loss.

**The screen** (`lowrank/screen.py`) fits every $(k,m)$ in a grid on the discovery simulations. Each candidate is a normalised density on all 8 parameters, so the held-out one-step NLLs are directly comparable. The rule: among candidates within one *paired* standard error of the best, take the smallest $m$, then the smallest $k$. Each fit also reports:

- a rotation-invariant information spectrum;
- principal angles to the true plane.

**The arms:** raw, mean, summary ($\theta\mid t$), eta ($\eta\mid t$), structured ($f(a)\mid t$ with the prior on $v$), oracle (true plane and exact $s(x)$), the one-step Gaussian, and the prior. The source notebook's budget accounting is the default: discovery is $0.4N$ and disjoint from the NPE training set. Change it with `--discovery fixed --discovery-value 500` and `--reuse`. Metrics:

- posterior KL (mean of $\log p-\log q$ at $\theta^\ast$), with paired standard errors;
- CRPS against the reference-draw floor;
- sliced Wasserstein distance, in native $\theta$ and in the true active plane;
- end-of-training validation $\log q$;
- coverage, random-reference ranks, and checks on the nuisance directions.

```bash
python scripts/lowrank_screen.py  --n-discovery 400 1200 4000 --seeds 0 1 2    # screen only, no NPE
python scripts/lowrank_scaling.py --out results/lowrank --quick                 # ~2 min smoke test
python scripts/lowrank_scaling.py --out results/lowrank                         # or: sbatch slurm/lowrank_array.sbatch
python scripts/plot_lowrank.py    --results results/lowrank
```

Colab: `notebooks/lowrank_colab.ipynb`.

**Reading CRPS here.** Every native-$\theta_j$ marginal mixes the two active directions with the six prior-width ones, so native-$\theta$ CRPS changes little between methods. Posterior KL and the active-plane Wasserstein distance are the discriminating metrics.

## Shared parameter–data spaces (`unified_sbi.shared`)

This study tests the one-step loss with a **rectangular** parameter embedding, $\eta:\mathbb R^d\to\mathbb R^K$ with $K>d$, on a weak-lensing setting. The theory note `docs/shared_space.md` makes three points:

- The unnormalised loss $\tfrac12\|\eta-t\|^2-\log\mathrm{vol}\,J$ is unbounded below when $K>d$.
- Normalising over θ fixes this. For $d\le3$ the normaliser comes from exact quadrature; for larger $d$ the method becomes InfoNCE/CLIP.
- $K$ is the number of exponential-family statistics, and $J^\top J$ is a learned Fisher field.

Results so far are in §8 of `docs/shared_space.md`:

- **Lensing toy:** an exact-posterior Gaussian field, plus a lognormal variant.
- **Real 128² catalogue:** the normalised shared head beats MAF NPE on the same features in every seed and at every budget, saving 3–8× in simulations at N ≤ 1000.
- **Embedding objectives (§12):** InfoNCE, Hyvärinen and Gaussian-MI alternatives to the quadrature loss, a stop-gradient emulator term, and a frozen-summary (evaluation B) protocol. Toy, seed 0 only.

```
src/unified_sbi/shared/
  lensing.py     tomographic GRF / lognormal convergence maps for (Omega_m, S8); exact likelihood and Fisher (Gaussian field)
  quadrature.py  adaptive 2D midpoint quadrature (log Z, HPD, moments, sampling)
  nets.py        signed cross-spectra, CNN, parameter embedding eta, zuko MAF head
  heads.py       SharedHead: normalised gauss/expfam critic, prior/Jeffreys base, nested and on-manifold options;
                 the unnormalised rectangular loss; info = quad | nce | hyv, stop-gradient EmulatorTerm;
                 GMIHead (Gaussian-MI objective with a separate emulator tower)
  study.py       arms, data pools, training, evaluation, catalogue drop-in (CatalogueData), GPU via device="auto"
scripts/shared_scaling.py   scaling sweep (Gaussian field + exact oracle, --field lognormal, or --catalogue <npz>)
scripts/shared_kscreen.py   K / critic screen
scripts/plot_shared.py      scaling figure, tables, savings factors (--kscreen for the K screen)
scripts/inspect_shared.py   posteriors vs exact, Fisher ellipses, shared-space scatter, information split
notebooks/shared_colab.ipynb  self-contained Colab runner for the real catalogues (downloads by Drive file ID)
scripts/shared_posthoc.py   Fisher / emulator diagnostics and evaluation B for saved models
scripts/shared_crossfit.py  2-fold cross-fitted evaluation B (leakage calibration)
scripts/shared_local_fisher.py  local Sigma(theta), dE[t|theta]/dtheta and emulator Fisher from fresh simulations (toy)
scripts/shared_emulator_refit.py  post-hoc emulator refits vs the in-training emulator (toy)
scripts/shared_emu_report.py  evaluation A/B tables, paired differences, Fisher ratios, Sigma-hat, harmonics
scripts/shared_timing.py    ms per training step and parameter counts
tests/test_shared.py        quadrature, exact likelihood vs dense covariance, log-volume, helix unboundedness,
                            Hyvarinen helix/scale, stop-gradient, InfoNCE limit, GMI optimum and helix invariance
```

```bash
pip install -e ".[dev,shared]"
python scripts/shared_scaling.py --out results/shared_gauss                       # Gaussian field, exact oracle
python scripts/shared_scaling.py --out results/shared_lognormal --field lognormal
python scripts/shared_scaling.py --out results/shared_cat --catalogue /path/prior_S8_L_250_N_128_Nz_512.npz --device auto
python scripts/plot_shared.py    --results results/shared_gauss
python scripts/plot_shared.py    --kscreen results/shared_kscreen --ref results/shared_gauss
# embedding-objective study (arms: hybrid_{quad|nce|hyv}_b{beta}[_B{batch}], hybrid_hyv, hybrid_nce, hybrid_gmi)
python scripts/shared_scaling.py --out results/toy_emu --arms hybrid_quad_b0 hybrid_quad_b0.3 hybrid_nce hybrid_hyv hybrid_gmi hybrid_rect
python scripts/shared_emu_report.py --results results/toy_emu --out results/toy_emu/report.md
```

## Few-run conditional surrogates (`unified_sbi.surrogate`)

A testbed for **Fisher-isometric conditioning**: learning $s_{t+1}=F(s_t,\theta)$ from very few distinct θ, each with a long trajectory. The method, arms and predictions are in `docs/plasma_surrogate_conditioning.md`; §11 there covers the implementation.

- **Simulator:** a batched modified Hasegawa–Wakatani solver ($64^2$) with $d=5$ control parameters. Branch restarts give the true one-step Jacobians.
- **Arms:**
  - A0: FiLM on raw θ;
  - A1: learned $g$;
  - A2: + $\mathcal R_{\rm iso}$ against the surrogate's own Fisher;
  - A3: + Fisher-whitened θ-noise;
  - A4: A2 + Jacobian supervision with the physical Fisher as target;
  - A5: isotropic-noise control;
  - A6: full method.
- **Design experiment:** random, LHS, or maximin in θ, in $g$ (chord) or along the learned manifold (geodesic).

```bash
python scripts/surrogate_study.py --out /tmp/surr_quick --quick          # software check
python scripts/surrogate_study.py --out results/surrogate --arms A0_film A1_g A2_iso A4_iso_jac --seeds 0 1
python scripts/surrogate_study.py --out results/surrogate --design       # + design experiment (P5)
python scripts/plot_surrogate.py  --results results/surrogate
```

Colab: `notebooks/surrogate_colab.ipynb`. It is self-contained and caches simulations on Drive. Regenerate it with `python scripts/build_surrogate_colab.py` after editing the package.
